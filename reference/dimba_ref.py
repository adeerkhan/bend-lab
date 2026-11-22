#!/usr/bin/env python3
"""Small Dimba-style Transformer-Mamba hybrid on flowers64.

  reference/.venv/bin/python reference/dimba_ref.py --steps 1200 --out results/dimba/hybrid

Learned from ref/Dimba (feizc/Dimba, arXiv:2406.01159):
  - DiT blocks with adaLN-Zero (our Block from hit_v4, FullAttention analogue)
  - Mamba SSM blocks interleaved after a gap: gap = depth // mamba_depth
    (the released Dimba.forward loops depth blocks and runs mamba_blocks[i//gap]
     when i % gap == 1)
  - patch embedding + 2D sincos pos, final patch-wise head

No VAE, no T5 captions, no cross-attention: flowers64 is grayscale and has no
text. This is a small structural analogue on our data, comparable to the ViT /
JiT(DiT) / HiT arms at the same step budget. The Mamba block is our validated
CPU Mamba-3 SISO recurrence (discrete A, B, C, D, dt), not Dimba's fused CUDA
selective scan.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from flowers_report import legibility  # noqa: E402
from unet_flowers_eval import spectrum  # noqa: E402
from hit_v4 import (load, patchify, unpatch, sincos_pos, RMSNorm, Block, Blocks,
                    AMIN, DATA, EDGE, GRID, PATCH, TOKENS, PDIM)  # noqa: E402

D = 128
DEPTH = 18
MAMBA_RATIO = 1.0


class MambaBlock(nn.Module):
    """A small CPU Mamba-3 SISO mixer: selective scan over the token sequence.

    State per channel: 16 cells; dt, A, B, C, D learned; output = scan(x)*D + x*Dskip.
    Uses our repo's validated discrete recurrence, no CUDA kernels.
    """
    def __init__(self, d=128, state=16):
        super().__init__()
        self.d, self.n = d, state
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.dt = nn.Linear(d, self.n, bias=True)
        self.A = nn.Parameter(torch.randn(state, d) * 0.02 - 2.0)   # negative
        self.B = nn.Linear(d, state, bias=False)
        self.C = nn.Linear(d, state, bias=False)
        self.D = nn.Parameter(torch.randn(d) * 0.02)
        self.norm = RMSNorm(d)

    def forward(self, x):
        # x: (B,N,d); scan over N in order (causal, as Mamba is)
        B, N, d = x.shape
        h = self.norm(x)
        u, b_in, c_in = self.in_proj(h).chunk(3, -1)
        dt = F.softplus(self.dt(h))                  # (B,N,state)
        Bm = self.B(b_in); Cm = self.C(c_in)        # (B,N,state)
        state = torch.zeros(B, self.n, d, device=x.device)
        outs = []
        for t in range(N):
            # state: (B,state,d); dt: (B,d); A: (state,d); Bm: (B,state)
            a = torch.exp(dt[:, t, :, None] * self.A[None])      # (B,state,d)
            bb = dt[:, t, :, None] * Bm[:, t, :, None] * u[:, t, None, :]
            state = a * state + bb
            y = (state * Cm[:, t, :, None]).sum(1)                 # (B,d)
            outs.append(y + self.D * u[:, t])
        return torch.stack(outs, 1)


class Hybrid(nn.Module):
    def __init__(self, d=D, depth=DEPTH, mamba_ratio=MAMBA_RATIO, ctx=8):
        super().__init__()
        self.ctx_n = ctx + 1
        self.embed = nn.Linear(PDIM, d)
        self.register_buffer('pos', torch.from_numpy(sincos_pos(d)))
        self.t = nn.Linear(1, d); self.y = nn.Linear(1, d)
        self.register_buffer('ctx_pos', torch.randn(1, ctx, d) * 0.02)
        self.register_buffer('time_pos', torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, 2, adaln=True, rope=None) for _ in range(depth)])
        self.mamba_depth = int(mamba_ratio * depth)
        self.gap = max(1, int(depth // max(1, self.mamba_depth)))
        self.mambas = nn.ModuleList([MambaBlock(d) for _ in range(self.mamba_depth)])
        self.mamba_norms = nn.ModuleList([RMSNorm(d) for _ in range(self.mamba_depth)])
        self.head = nn.Linear(d, PDIM)

    def forward(self, x, t, y, return_aux=False):
        B = x.shape[0]
        tc = self.t(t.reshape(-1, 1)); yc = self.y(y.reshape(-1, 1))
        cond = tc + yc
        ctx = torch.cat((tc[:, None] + self.time_pos, yc[:, None] + self.ctx_pos), dim=1)
        H = torch.cat((ctx, self.embed(patchify(x)) + self.pos.unsqueeze(0)), dim=1)
        residual = None
        mi = 0
        for i in range(len(self.blocks)):
            H = self.blocks[i](H, cond)
            if self.mamba_depth > 0 and (self.gap == 1 or i % self.gap == 1):
                if mi < len(self.mambas):
                    img, c = H[:, self.ctx_n:], H[:, :self.ctx_n]
                    res = self.mambas[mi](img if residual is None else img + residual)
                    residual = res
                    H = torch.cat((c, img + res), dim=1)
                    mi += 1
        pred = unpatch(self.head(H[:, self.ctx_n:]))
        return (pred, H.new_zeros(()), {}) if return_aux else (pred, H.new_zeros(()))


def sample(model, x0, steps=32):
    model.eval(); x = x0
    with torch.no_grad():
        for t in np.linspace(0, 1, steps + 1)[:-1]:
            tt = torch.full((x.shape[0],), float(t))
            pred, _ = model(x, tt, torch.zeros_like(tt))
            x = x + (pred - x) / max(1 - t, AMIN) * (1 / steps)
    return torch.clamp(x, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--generated_samples", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-mamba", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    tr, ho = load(args.n_train)
    model = Hybrid(mamba_ratio=0.0 if args.no_mamba else MAMBA_RATIO)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    os.makedirs(args.out, exist_ok=True)
    print(f"arm=Dimba-hybrid params={npar} depth={DEPTH} mamba_blocks={model.mamba_depth} "
          f"gap={model.gap} steps={args.steps}", flush=True)
    losses, t0 = [], time.time()
    model.train()
    for step in range(args.steps):
        idx = np.random.randint(0, len(tr), args.batch)
        x = torch.from_numpy(tr[idx])
        tt = torch.rand(args.batch); yy = torch.zeros(args.batch)
        xt = tt.reshape(-1,1,1)*x + (1-tt.reshape(-1,1,1))*torch.randn_like(x)
        pred, _ = model(xt, tt, yy)
        den = torch.clamp(1-tt, min=AMIN).reshape(-1,1,1)
        loss = ((pred-x)**2).mean()/(den**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (step+1) % max(1,args.steps//8) == 0:
            print(f"step {step+1}/{args.steps} loss {losses[-1]:.5f} ({time.time()-t0:.0f}s)", flush=True)
    generated_samples = sample(model, torch.randn(args.generated_samples, EDGE, EDGE)).numpy()
    held = legibility(generated_samples.reshape(args.generated_samples,-1), ho.reshape(-1,EDGE*EDGE))
    trleg = legibility(generated_samples.reshape(args.generated_samples,-1), tr.reshape(-1,EDGE*EDGE))
    ceil = legibility(ho.reshape(-1,EDGE*EDGE), tr.reshape(-1,EDGE*EDGE))
    floor = legibility(np.random.randn(args.generated_samples,EDGE*EDGE).astype(np.float32), tr.reshape(-1,EDGE*EDGE))
    rec = dict(arm=("dimba-no-mamba" if args.no_mamba else "dimba-hybrid"), paper="arXiv:2406.01159 / feizc/Dimba", params=npar,
               steps=args.steps, batch=args.batch, depth=DEPTH, mamba_blocks=model.mamba_depth,
               gap=model.gap, losses=losses, first_loss=losses[0], final_loss=losses[-1],
               legibility_test=float(held.mean()), legibility_training=float(trleg.mean()),
               positive_control=float(ceil.mean()), negative_control=float(floor.mean()),
               spectrum=[float(v) for v in spectrum(generated_samples)],
               wall_seconds=time.time()-t0, generated_samples=args.generated_samples)
    torch.save(model.state_dict(), os.path.join(args.out,"model.pt"))
    json.dump(rec, open(os.path.join(args.out,"run.json"),"w"), indent=1)
    np.save(os.path.join(args.out,"generated_samples.npy"), generated_samples)
    print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in rec.items() if k!="losses"},indent=1), flush=True)


if __name__ == "__main__":
    main()
