#!/usr/bin/env python3
"""Isolation gate: LayerNorm -> FullAttention -> residual, NO adaLN gate or modulation.

  python3 reference/dit_lnattn_check.py [binary]

`examples/diffusion/dit/ln_attn_probe.bend`: y = h + AttnF(LN(h)), same shapes
and dy as the full DiT block. This decides whether the block's 5.45x gradient
gap comes from depth (LN stacked on attention) or from the gate interaction.
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

B, T, HD, EPS = 2, 3, 4, 1e-5
N = B * T


def vals(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def tt(a, r, c):
    return torch.tensor(a.reshape(r, c)).requires_grad_(True)


def lin(i, o, s):
    return tt(vals(o * i, s), o, i), tt(vals(o, s + 40), 1, o)


def main():
    binary = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(HERE), ".scratch", "dit_lnattn")
    got = parse(subprocess.run([binary], check=True, capture_output=True, text=True).stdout)

    h = tt(vals(24, 33), 6, 4)
    lg, lb = tt(vals(4, 61), 1, 4), tt(vals(4, 67), 1, 4)
    wq, bq = lin(4, 4, 3); wk, bk = lin(4, 4, 7)
    wv, bv = lin(4, 4, 11); wo, bo = lin(4, 4, 15)

    norm = F.layer_norm(h, (4,), lg.view(4), lb.view(4), EPS)
    q, k, v = (norm @ w.T + b for w, b in ((wq, bq), (wk, bk), (wv, bv)))
    q, k, v = (z.reshape(B, T, HD) for z in (q, k, v))
    attn = (F.softmax((q @ k.transpose(-1, -2)) * 0.5, dim=-1) @ v).reshape(N, HD)
    y = h + (attn @ wo.T + bo)
    y.backward(tt(vals(24, 97), 6, 4).detach())

    checks = [
        ("lnattn.y", y), ("lnattn.dh", h.grad),
        ("lnattn.ln.dg", lg.grad), ("lnattn.ln.db", lb.grad),
        ("lnattn.q.dw", wq.grad), ("lnattn.q.db", bq.grad),
        ("lnattn.k.dw", wk.grad), ("lnattn.k.db", bk.grad),
        ("lnattn.v.dw", wv.grad), ("lnattn.v.db", bv.grad),
        ("lnattn.o.dw", wo.grad), ("lnattn.o.db", bo.grad),
    ]
    failures, worst = [], 0.0
    for name, ref in checks:
        if name not in got:
            failures.append(f"{name}: not printed")
            continue
        worst = max(worst, close(name, got[name], ref, failures))
    for f in failures:
        print("FAIL", f)
    if failures:
        raise SystemExit(1)
    print(f"dit-lnattn: {len(checks)} tensors, worst error {worst:.3f} of the tolerance; PASS")


if __name__ == "__main__":
    main()
