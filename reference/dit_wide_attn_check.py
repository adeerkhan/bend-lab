#!/usr/bin/env python3
"""Wider DiT attention (b=2, t=4, w=8, kv=2 heads of hd=4) against PyTorch.

  python3 reference/dit_wide_attn_check.py [binary]

`examples/diffusion/dit/wide_attn.bend` generalises the attention branch to
more tokens and heads via top-level Nat constants. Same oracle and seeds.
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "bendtorch", "reference"))
from oracle import close, parse  # noqa: E402

B, T, W, KV, HD, SC = 2, 4, 8, 2, 4, 0.5
N = B * T


def vals(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def tt(a, r, c):
    return torch.tensor(a.reshape(r, c)).requires_grad_(True)


def lin(i, o, s):
    return tt(vals(o * i, s), o, i), tt(vals(o, s + 40), 1, o)


def main():
    binary = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(HERE), ".scratch", "dit_wide")
    got = parse(subprocess.run([binary], check=True, capture_output=True, text=True).stdout)

    x = tt(vals(N * W, 33), N, W)
    wq, bq = lin(W, W, 3); wk, bk = lin(W, W, 7)
    wv, bv = lin(W, W, 11); wo, bo = lin(W, W, 15)
    q, k, v = (x @ w.T + b for w, b in ((wq, bq), (wk, bk), (wv, bv)))
    q, k, v = (z.reshape(B, T, KV, HD).transpose(1, 2) for z in (q, k, v))
    attn = (F.softmax((q @ k.transpose(-1, -2)) * SC, dim=-1) @ v)
    attn = attn.transpose(1, 2).reshape(N, W)
    y = attn @ wo.T + bo
    y.backward(tt(vals(N * W, 97), N, W).detach())

    checks = [
        ("wide.y", y), ("wide.dh", x.grad),
        ("wide.q.dw", wq.grad), ("wide.q.db", bq.grad),
        ("wide.k.dw", wk.grad), ("wide.k.db", bk.grad),
        ("wide.v.dw", wv.grad), ("wide.v.db", bv.grad),
        ("wide.o.dw", wo.grad), ("wide.o.db", bo.grad),
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
    print(f"dit-wide-attn: {len(checks)} tensors, worst error {worst:.3f} of the tolerance; PASS")


if __name__ == "__main__":
    main()
