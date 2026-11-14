"""A faithful pure-PyTorch twin of JLrumberger/MambaViT (MNIST config).

  reference/.venv-rf/bin/python reference/mamba_vit_twin.py --epochs 2

The reference repo needs `mamba-ssm` (Mamba-1) and `causal-conv1d`, neither
installed on this lane, so this file reimplements the SAME model in plain
torch: the ViT shell (patch embed, cls token, pos embed, mean/cls pool,
head) and Mamba-1 blocks with mamba_ssm's shapes and math:

  in_proj(d_model -> 2*d_inner) | depthwise causal conv1d(d_conv) + silu
  x_proj(d_inner -> dt_rank + 2*d_state) | dt_proj(dt_rank -> d_inner)
  dt = softplus(dt_proj(x) + dt_bias); A = -exp(A_log)
  h_t = exp(dt*A) h_{t-1} + dt*B x_t ;  y_t = <C, h_t> + D x_t
  y = y * silu(z) | out_proj(d_inner -> d_model)

Block order is the reference's: residual = x + residual; h = mixer(LN(res));
the final residual is summed once and normed.  `--check` prints the
parameter count and the shapes; `--epochs N` trains on MNIST and prints the
validation accuracy the reference reports (~90% in a couple of epochs).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Mamba1(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = expand * d_model
        self.d_conv = d_conv
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_bias = nn.Parameter(torch.empty(self.d_inner))
        # mamba_ssm's init: dt_bias log-uniform in [dt_min, dt_max], A_log log(1..d_state)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = torch.clamp(dt, min=1e-4)
        with torch.no_grad():
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            self.A_log.copy_(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
            self.conv1d.bias.zero_()
            self.conv1d.weight.data.normal_(0, 0.02 * math.sqrt(self.d_inner / d_conv))
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, u):
        b, l, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.act(self.conv1d(F.pad(x, (self.d_conv - 1, 0)))[..., :l]).transpose(1, 2)
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt) + self.dt_bias)
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(dt[..., None] * A)                 # (b,l,inner,state)
        dB = dt[..., None] * B[..., None, :]              # (b,l,inner,state)
        h = torch.zeros(b, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(l):
            h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
            ys.append((h * C[:, t, None, :]).sum(-1) + self.D * x[:, t])
        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.mixer = Mamba1(dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x, residual=None):
        residual = (x + residual) if residual is not None else x
        return self.mixer(self.norm(residual)), residual


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MambaViT(nn.Module):
    def __init__(self, *, image_size=28, patch_size=4, num_classes=10, channels=1,
                 n_layer=8, dim=32, pool="mean", d_state=16, d_conv=4, expand=2,
                 eps=1e-5, emb_dropout=0.0):
        super().__init__()
        h, w = pair(image_size)
        p1, p2 = pair(patch_size)
        assert h % p1 == 0 and w % p2 == 0
        self.num_patches = (h // p1) * (w // p2)
        patch_dim = channels * p1 * p2
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim, eps=eps),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim, eps=eps))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([Block(dim, d_state, d_conv, expand, eps)
                                     for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(dim, eps=eps)
        self.pool = pool
        self.mlp_head = nn.Linear(dim, num_classes)
        self.p1, self.p2 = p1, p2
        # a deterministic init for the port to load
        with torch.no_grad():
            self.pos_embedding.normal_(0, 1.0)
            self.cls_token.normal_(0, 1.0)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.weight.fill_(1.0)
                    m.bias.zero_()

    def patches(self, img):
        b, c, h, w = img.shape
        x = img.reshape(b, c, h // self.p1, self.p1, w // self.p2, self.p2)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, self.num_patches, -1)
        return self.to_patch_embedding(x)

    def forward(self, img):
        x = self.patches(img)
        b, n, _ = x.shape
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x = x + self.pos_embedding[:, : n + 1]
        x = self.dropout(x)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        residual = (x + residual) if residual is not None else x
        x = self.norm_f(residual)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        return self.mlp_head(x)


def param_count(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=os.path.join("/tmp/opencode", "mnist"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    m = MambaViT().to(DEV)
    print(f"params: {param_count(m)}  device: {DEV}")
    if args.epochs <= 0:
        return
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor()])
    tr = torchvision.datasets.MNIST(args.data, train=True, download=True, transform=tf)
    va = torchvision.datasets.MNIST(args.data, train=False, download=True, transform=tf)
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=2)
    vl = torch.utils.data.DataLoader(va, batch_size=512, num_workers=2)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
    for ep in range(1, args.epochs + 1):
        m.train()
        tot = hit = 0
        for i, (img, y) in enumerate(tl):
            img, y = img.to(DEV), y.to(DEV)
            out = m(img)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            hit += (out.argmax(-1) == y).sum().item()
            tot += y.numel()
            if i % 200 == 0:
                print(f"ep {ep} step {i} loss {float(loss):.4f} acc {hit/tot:.4f}")
        m.eval()
        vh = vt = 0
        with torch.no_grad():
            for img, y in vl:
                img, y = img.to(DEV), y.to(DEV)
                out = m(img)
                vh += (out.argmax(-1) == y).sum().item()
                vt += y.numel()
        print(f"epoch {ep}: train acc {hit/tot:.4f}  val acc {vh/vt:.4f}")


if __name__ == "__main__":
    main()
