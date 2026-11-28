#!/usr/bin/env python3
"""DiT denoiser model (embed -> conditioned block -> head) forward against PyTorch.

  python3 reference/dit_denoiser_check.py [binary]

`examples/diffusion/dit/main.bend`: X=(cond:Mat<B,CD>, hfeat:Mat<N,FE>),
Embed(N,FE,C) -> conditioned DiT block -> Head(N,C,PX). b=2, t=3, N=6,
FE=8, C=8, CD=3, PX=4, two kv heads. Forward is the gate.
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle import close, parse  # noqa: E402

B, TK, N, FE, PX, C, CD, KV, HD, SC, EPS = 2, 3, 6, 8, 4, 8, 3, 2, 4, 0.5, 1e-5


def vals(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def vals2(n, start=0):
    return np.array([((i * i * 5 + i * 3) % 17 - 8) / 4 for i in range(start, start + n)], dtype=np.float32)


def tt(a, r, c):
    return torch.tensor(a.reshape(r, c))


def lin(i, o, s):
    return torch.tensor(vals(o * i, s).reshape(o, i)), torch.tensor(vals(o, s + 40).reshape(1, o))


def tile(x):
    return x.repeat_interleave(TK, dim=0)


def main():
    binary = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(HERE), ".scratch", "dit_denoiser")
    got = parse(subprocess.run([binary], check=True, capture_output=True, text=True).stdout)

    cond = tt(vals2(B * CD, 23), B, CD)
    hfeat = tt(vals(N * FE, 33), N, FE)
    we = torch.tensor(vals(C * FE, 3).reshape(C, FE))  # embed (bias-free)
    wh = torch.tensor(vals(PX * C, 97).reshape(PX, C))  # head (bias-free)

    # block params, seeds from wide_block.bend block_p
    wg, bg = lin(CD, C, 7); wm, bm = lin(CD, C, 9); ws, bs = lin(CD, C, 11)
    lg = torch.tensor(vals(C, 61).reshape(1, C)); lb = torch.tensor(vals(C, 67).reshape(1, C))
    wq, bq = lin(C, C, 3); wk, bk = lin(C, C, 7); wv, bv = lin(C, C, 11); wo, bo = lin(C, C, 15)

    h = hfeat @ we.T
    c = F.silu(cond)
    g, m, sh = F.linear(c, wg, bg), F.linear(c, wm, bm), F.linear(c, ws, bs)
    mod = F.layer_norm(h, (C,), lg.view(C), lb.view(C), EPS) * tile(m) + tile(sh)
    q, k, v = (mod @ w.T + b for w, b in ((wq, bq), (wk, bk), (wv, bv)))
    q, k, v = (z.reshape(B, TK, KV, HD).transpose(1, 2) for z in (q, k, v))
    a = (F.softmax((q @ k.transpose(-1, -2)) * SC, dim=-1) @ v).transpose(1, 2).reshape(N, C)
    y = (h + tile(g) * (a @ wo.T + bo)) @ wh.T

    failures, worst = [], 0.0
    worst = max(worst, close("model.y", got["model.y"], y, failures))
    for f in failures:
        print("FAIL", f)
    if failures:
        raise SystemExit(1)
    print(f"dit-denoiser: forward matches PyTorch ({worst:.3f} of tolerance); PASS")


if __name__ == "__main__":
    main()
