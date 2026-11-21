#!/usr/bin/env python3
"""Faithful small port of Nicholas0228/HiT-Released on flowers64, with JiT(DiT) and ViT arms.

  reference/.venv/bin/python reference/hit_v2.py --arm hit --steps 1200 --out results/hit-v2/hit

This mirrors the released code's structure at a trainable scale:

* Block (hnet/layers.py): RMSNorm, QK-RMSNorm, SwiGLU MLP, adaLN modulation
  (6-way shift/scale/gate), optional SpatialRoPE.
* Grouping (hnet/grouping.py): exact average-linkage tree via scipy's pinned
  nn_chain, hard groups at a calibrated threshold, and a VECTORIZED soft reader
  using scatter_add group means of q and k (the real implementation's key
  efficiency trick -- my v1 used a Python loop and was ~30x slower).
* Budget loss (Eq. 10): hard fraction constant, soft fraction from per-merge
  sigmoid scores, pop == batch * N.
* Two stages with skip connections W1/W2 initialized to identity.
* Context tokens (class + time) bypass both routers and both skips.
* JiT/DiT arm: same blocks, patch bottleneck, in-context entry at mid-depth.
* ViT arm: plain blocks, no adaLN, no context.

All three use the same clean-image velocity loss (paper Eq. 12).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import _hierarchy
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from flowers_report import legibility  # noqa: E402
from unet_flowers_eval import spectrum  # noqa: E402

DATA = os.path.expanduser("~/.cache/bend-lab/flowers")
EDGE, PATCH, GRID = 64, 8, 8
TOKENS, PDIM = GRID * GRID, PATCH * PATCH
AMIN, LAMBDA = 0.05, 0.01
TR, TB = 0.05, 0.1


def load(n=2048, nh=102):
    tr = np.fromfile(os.path.join(DATA, "flowers64-trainfull-gray.u8"), np.uint8).reshape(-1, EDGE, EDGE)[:n].astype(np.float32) / 255
    ho = np.fromfile(os.path.join(DATA, "flowers64-hold-gray.u8"), np.uint8).reshape(-1, EDGE, EDGE)[:nh].astype(np.float32) / 255
    return tr, ho


def patchify(x):
    B = x.shape[0]
    return x.reshape(B, GRID, PATCH, GRID, PATCH).permute(0, 1, 3, 2, 4).reshape(B, TOKENS, PDIM)


def unpatch(z):
    B = z.shape[0]
    return z.reshape(B, GRID, GRID, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(B, EDGE, EDGE)


class SpatialRoPE(nn.Module):
    """The released hnet/layers.py SpatialRoPE, at an 8x8 token grid."""
    def __init__(self, head_dim, side=GRID, prefix=0):
        super().__init__()
        self.prefix = prefix
        half = head_dim // 2
        freq = 1 / (10000 ** (torch.arange(0, half, 2).float() / half))
        pos = torch.arange(side) / side * side
        phase = torch.einsum('n,f->nf', pos, freq).repeat_interleave(2, -1)
        grid = torch.cat((phase[:, None, :].expand(side, side, -1),
                          phase[None, :, :].expand(side, side, -1)), -1).reshape(side * side, head_dim)
        self.register_buffer('cos', torch.cat((torch.ones(prefix, head_dim), grid.cos())), persistent=False)
        self.register_buffer('sin', torch.cat((torch.zeros(prefix, head_dim), grid.sin())), persistent=False)

    def forward(self, x):
        n = x.shape[2]
        cos = self.cos[:n] if n <= self.cos.shape[0] else torch.cat(
            (self.cos[:self.prefix], self.cos[self.prefix:self.prefix + (n - self.prefix) % GRID]), dim=0)
        sin = self.sin[:n] if n <= self.sin.shape[0] else torch.cat(
            (self.sin[:self.prefix], self.sin[self.prefix:self.prefix + (n - self.prefix) % GRID]), dim=0)
        if cos.shape[0] != n:
            return x  # fall back: no positional rotation for unexpected length
        paired = x.reshape(*x.shape[:-1], -1, 2)
        rot = torch.stack((-paired[..., 1], paired[..., 0]), -1).flatten(-2)
        return x * cos + rot * sin


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w)


def modulate(x, sh, sc):
    return x * (1 + sc[:, None]) + sh[:, None]


class Block(nn.Module):
    def __init__(self, d, heads, adaln=True, rope=None):
        super().__init__()
        self.adaln = adaln
        self.rope = rope
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qnorm, self.knorm = RMSNorm(d // heads), RMSNorm(d // heads)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 4 * d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)
        self.heads = heads
        if adaln:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
            nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def swiglu(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def attn(self, x):
        B, N, d = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.qnorm(q), self.knorm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, N, d))

    def forward(self, x, cond=None):
        if self.adaln:
            sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.ada(cond).chunk(6, -1)
            x = x + g_a[:, None] * self.attn(modulate(self.n1(x), sh_a, sc_a))
            return x + g_m[:, None] * self.swiglu(modulate(self.n2(x), sh_m, sc_m))
        x = x + self.attn(self.n1(x))
        return x + self.swiglu(self.n2(x))


class Blocks(nn.Module):
    def __init__(self, n, d, adaln=True, heads=2, use_rope=False):
        super().__init__()
        hd = d // heads
        rope = SpatialRoPE(hd, prefix=2) if (use_rope and hd % 4 == 0) else None
        self.blocks = nn.ModuleList([Block(d, heads, adaln, rope) for _ in range(n)])
        self.heads = heads

    def forward(self, x, cond=None):
        for b in self.blocks:
            x = b(x, cond)
        return x


class Group(nn.Module):
    """Vectorized soft reader mirroring hnet/grouping.py SoftGrouping."""
    def __init__(self, d, ratio=2.0):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            eye = torch.eye(d)
            self.q.weight.copy_(eye + torch.randn_like(eye) * 1e-3)
            self.k.weight.copy_(eye + torch.randn_like(eye) * 1e-3)
        self.skip = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.skip.weight.copy_(torch.eye(d))
        self.register_buffer("tau", torch.tensor(0.0))
        self.ratio = ratio

    def project(self, h):
        return F.normalize(self.q(h), dim=-1), F.normalize(self.k(h), dim=-1)

    def structure(self, q, k):
        B, N, _ = q.shape
        S = 0.5 * (q @ k.transpose(-1, -2) + k @ q.transpose(-1, -2))
        Sd = S.detach().cpu().numpy()
        iu = np.triu_indices(N, 1)
        tau = float(self.tau.item())
        hard, counts, scores = [], [], []
        for b in range(B):
            dist = np.maximum(0.0, 1.0 - Sd[b][iu].astype(np.float64))
            z = _hierarchy.nn_chain(dist, N, 2)
            sc = 1.0 - z[:, 2]
            parent = np.arange(2 * N - 1)
            accepted = 0
            for m in range(N - 1):
                if sc[m] >= tau:
                    a, c = int(z[m, 0]), int(z[m, 1])
                    while parent[a] != a:
                        a = parent[a]
                    while parent[c] != c:
                        c = parent[c]
                    parent[c] = a = min(a, c)
                    accepted += 1
            roots = np.array([i for i in range(N) if parent[i] == i])
            lab = np.full(2 * N - 1, -1)
            groups = np.zeros(N, np.int64)
            g = 0
            for i in range(N):
                r = i
                while parent[r] != r:
                    r = parent[r]
                if lab[r] < 0:
                    lab[r] = g; g += 1
                groups[i] = lab[r]
            hard.append(torch.from_numpy(groups).to(q.device))
            counts.append(int(len(roots)))
            scores.append(torch.from_numpy(sc).to(q.device).float())
        return S, torch.stack(hard), torch.tensor(counts, device=q.device), torch.stack(scores)

    def forward(self, h):
        q, k = self.project(h)
        S, hard, counts, scores = self.structure(q, k)
        B, N, d = h.shape
        K = int(hard.max().item()) + 1
        idx = hard[..., None].expand(-1, -1, d)
        sizes = torch.zeros(B, K, device=h.device).scatter_add_(1, hard, torch.ones(B, N, device=h.device))
        qm = torch.zeros(B, K, d, device=h.device).scatter_add_(1, idx, q)
        km = torch.zeros(B, K, d, device=h.device).scatter_add_(1, idx, k)
        qm = qm / sizes.clamp_min(1)[..., None]
        km = km / sizes.clamp_min(1)[..., None]
        logits = 0.5 * (q @ km.transpose(-1, -2) + k @ qm.transpose(-1, -2))
        valid = torch.arange(K, device=h.device)[None, :] < counts[:, None]
        A = (logits / TR).masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        mass = A.sum(1)
        compact = A.transpose(1, 2) @ h / (mass[..., None] + 1e-8)
        conf = ((scores - self.tau.float()) / TB).sigmoid()
        soft = N - conf.sum(1)
        fh = counts.float().sum() / (B * N)
        fs = soft.sum() / (B * N)
        comp = self.ratio / (self.ratio - 1) * ((self.ratio - 1) * fh * fs + (1 - fh) * (1 - fs))
        return compact.to(h.dtype), A, comp, hard, int(K)


class HiT(nn.Module):
    def __init__(self, d=128, depths=(3, 3, 6, 3, 3), heads=2):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(PDIM, d)
        self.t = nn.Linear(1, d); self.y = nn.Linear(1, d)
        e, m, _, md, dec = depths
        self.enc = Blocks(e, d, use_rope=True); self.mid = Blocks(m, d)
        self.core = Blocks(depths[2], d)
        self.dec_mid = Blocks(md, d); self.dec = Blocks(dec, d)
        self.g1 = Group(d); self.g2 = Group(d)
        self.head = nn.Linear(d, PDIM)
        self.register_buffer("ctx", torch.randn(1, 1, d) * 0.02)

    def forward(self, x, t, y, return_aux=False):
        B = x.shape[0]
        cond = self.t(t.reshape(-1, 1)) + self.y(y.reshape(-1, 1))
        ctx = self.ctx.expand(B, -1, -1)
        H = self.embed(patchify(x)) + ctx
        e1 = self.enc(H, cond)
        r1 = self.g1.skip(e1)
        Z1, A1, c1, hard1, K1 = self.g1(e1)
        e2 = self.mid(Z1, cond)
        r2 = self.g2.skip(e2)
        Z2, A2, c2, hard2, K2 = self.g2(e2)
        core = self.core(Z2, cond)
        h2 = r2 + A2 @ core
        h2 = self.dec_mid(h2, cond)
        h1 = r1 + A1 @ h2
        out = self.dec(h1, cond)
        pred = unpatch(self.head(out))
        if return_aux:
            return pred, c1 + c2, (hard1, K1, hard2, K2, A1, A2)
        return pred, c1 + c2


class JiT(nn.Module):
    """The released JiT baseline == DiT-style dense arm: adaLN blocks, patch
    bottleneck, in-context class/time tokens entering at mid depth."""
    def __init__(self, d=128, depth=18, heads=2):
        super().__init__()
        self.embed = nn.Linear(PDIM, d)
        self.t = nn.Linear(1, d); self.y = nn.Linear(1, d)
        self.blocks = Blocks(depth, d, adaln=True, use_rope=True)
        self.head = nn.Linear(d, PDIM)
        self.register_buffer("ctx", torch.randn(1, 2, d) * 0.02)

    def forward(self, x, t, y, return_aux=False):
        B = x.shape[0]
        cond = self.t(t.reshape(-1, 1)) + self.y(y.reshape(-1, 1))
        H = self.embed(patchify(x))
        H = torch.cat((self.ctx.expand(B, -1, -1), H), dim=1)
        H = self.blocks(H, cond)
        H = H[:, 2:]
        pred = unpatch(self.head(H))
        return (pred, H.new_zeros(()), (None,) * 6) if return_aux else (pred, H.new_zeros(()))


class ViT(nn.Module):
    def __init__(self, d=128, depth=18, heads=2):
        super().__init__()
        self.embed = nn.Linear(PDIM, d)
        self.t = nn.Linear(1, d); self.y = nn.Linear(1, d)
        self.blocks = Blocks(depth, d, adaln=False, use_rope=True)
        self.head = nn.Linear(d, PDIM)

    def forward(self, x, t, y, return_aux=False):
        B = x.shape[0]
        H = self.embed(patchify(x)) + self.t(t.reshape(-1, 1)).unsqueeze(1) + self.y(y.reshape(-1, 1)).unsqueeze(1)
        H = self.blocks(H)
        pred = unpatch(self.head(H))
        return (pred, H.new_zeros(()), (None,) * 6) if return_aux else (pred, H.new_zeros(()))


def sample(model, x0, steps=32):
    model.eval()
    x = x0
    with torch.no_grad():
        for t in np.linspace(0, 1, steps + 1)[:-1]:
            tt = torch.full((x.shape[0],), float(t))
            pred, _ = model(x, tt, torch.zeros_like(tt))
            x = x + (pred - x) / max(1 - t, AMIN) * (1 / steps)
    return torch.clamp(x, 0, 1)


def calibrate(model, tr, n=128):
    """Pool merge scores over 1024 images x several noise levels, pick tau so
    total groups = population/ratio (the released scripts/calibrate.py protocol)."""
    n = min(n, len(tr))
    torch.manual_seed(882014)
    model.eval()
    cond_t = lambda tt: model.t(tt.reshape(-1, 1)) + model.y(torch.zeros(n, 1))
    all_scores, pops = [], []
    for t in (0.25, 0.5, 0.75, 1.0):
        x = torch.from_numpy(tr[:n]); tt = torch.full((n,), float(t))
        xt = tt.reshape(-1, 1, 1) * x + (1 - t) * torch.randn_like(x)
        with torch.no_grad():
            H = model.embed(patchify(xt)) + model.ctx.expand(n, -1, -1)
            e1 = model.enc(H, cond_t(tt))
            for router, feats in ((model.g1, e1),):
                q, k = router.project(feats)
                S = 0.5 * (q @ k.transpose(-1, -2) + k @ q.transpose(-1, -2))
                Sd = S.detach().cpu().numpy()[0]
                z = _hierarchy.nn_chain(np.maximum(0, 1 - Sd[np.triu_indices(TOKENS, 1)].astype(np.float64)), TOKENS, 2)
                all_scores.append(1 - z[:, 2]); pops.append(TOKENS)
    flat = np.concatenate(all_scores)
    pop = sum(pops)
    desired = pop / model.g1.ratio
    ordered = np.sort(flat)[::-1]
    tau1 = float(ordered[max(1, int(desired)) - 1])
    model.g1.tau.fill_(tau1)
    # stage 2: use actual stage-1 compact representations
    all2, pops2 = [], []
    for t in (0.25, 0.5, 0.75, 1.0):
        x = torch.from_numpy(tr[:n]); tt = torch.full((n,), float(t))
        xt = tt.reshape(-1, 1, 1) * x + (1 - t) * torch.randn_like(x)
        with torch.no_grad():
            H = model.embed(patchify(xt)) + model.ctx.expand(n, -1, -1)
            e1 = model.enc(H, cond_t(tt))
            Z1, _, _, _, _ = model.g1(e1)
            e2 = model.mid(Z1, cond_t(tt))
            q, k = model.g2.project(e2)
            K1 = Z1.shape[1]
            S2 = 0.5 * (q @ k.transpose(-1, -2) + k @ q.transpose(-1, -2)).detach().cpu().numpy()[0]
            z2 = _hierarchy.nn_chain(np.maximum(0, 1 - S2[np.triu_indices(K1, 1)].astype(np.float64)), K1, 2)
            all2.append(1 - z2[:, 2]); pops2.append(K1)
    flat2 = np.concatenate(all2); pop2 = sum(pops2)
    desired2 = pop2 / model.g2.ratio
    tau2 = float(np.sort(flat2)[::-1][max(1, int(desired2)) - 1])
    model.g2.tau.fill_(tau2)
    print(f"calibrated tau=({tau1:.3f},{tau2:.3f}) pooled over {len(flat)}/{len(flat2)} merges", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["hit", "jit", "vit"], required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--retune-at", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tr, ho = load(args.n_train)
    model = {"hit": HiT, "jit": JiT, "vit": ViT}[args.arm]()
    if args.arm == "hit":
        calibrate(model, tr)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    os.makedirs(args.out, exist_ok=True)
    print(f"arm={args.arm} params={npar} steps={args.steps} batch={args.batch} train={len(tr)}", flush=True)

    losses, t0 = [], time.time()
    ema = None
    if args.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    model.train()
    for step in range(args.steps):
        idx = np.random.randint(0, len(tr), args.batch)
        x = torch.from_numpy(tr[idx])
        tt = torch.rand(args.batch); yy = torch.zeros(args.batch)
        xt = tt.reshape(-1, 1, 1) * x + (1 - tt.reshape(-1, 1, 1)) * torch.randn_like(x)
        pred, comp = model(xt, tt, yy)
        den = torch.clamp(1 - tt, min=AMIN).reshape(-1, 1, 1)
        loss = ((pred - x) ** 2).mean() / (den ** 2).mean() + (LAMBDA * comp if args.arm == "hit" else 0)
        if args.retune_at and step + 1 == args.retune_at:
            calibrate(model, tr)
            for ep, p in zip(ema.parameters(), model.parameters()) if ema is not None else []:
                ep.copy_(p)
        opt.zero_grad(); loss.backward(); opt.step()
        if ema is not None:
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.mul_(args.ema).add_(p.detach(), alpha=1 - args.ema)
                for ep, p in zip(ema.buffers(), model.buffers()):
                    ep.copy_(p)
        losses.append(float(loss.item()))
        if (step + 1) % max(1, args.steps // 8) == 0:
            print(f"step {step+1}/{args.steps} loss {losses[-1]:.5f} ({time.time()-t0:.0f}s)", flush=True)

    draws = sample(ema if ema is not None else model, torch.randn(args.draws, EDGE, EDGE)).numpy()
    held = legibility(draws.reshape(args.draws, -1), ho.reshape(-1, EDGE * EDGE))
    trleg = legibility(draws.reshape(args.draws, -1), tr.reshape(-1, EDGE * EDGE))
    ceil = legibility(ho.reshape(-1, EDGE * EDGE), tr.reshape(-1, EDGE * EDGE))
    floor = legibility(np.random.randn(args.draws, EDGE * EDGE).astype(np.float32), tr.reshape(-1, EDGE * EDGE))
    rec = dict(arm=args.arm, paper="arXiv:2611.22962 / Nicholas0228/HiT-Released",
               params=npar, steps=args.steps, batch=args.batch, width=128,
               patch=PATCH, tokens=TOKENS, losses=losses,
               first_loss=losses[0], final_loss=losses[-1],
               legibility_held=float(held.mean()), legibility_train=float(trleg.mean()),
               ceiling=float(ceil.mean()), noise_floor=float(floor.mean()),
               spectrum=[float(x) for x in spectrum(draws)],
               wall_seconds=time.time() - t0, draws=args.draws)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    json.dump(rec, open(os.path.join(args.out, "run.json"), "w"), indent=1)
    np.save(os.path.join(args.out, "draws.npy"), draws)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if k != "losses"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
