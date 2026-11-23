#!/usr/bin/env python3
"""DiT attention probe — QKV + FullAttention + output projection, forward and backward against PyTorch.

  python3 reference/dit_attn_check.py [binary]

The Bend side is `examples/diffusion/dit/attn_probe.bend`: b=2 images of t=3
tokens, width 4, one kv head (kv=1, g=1, hd=4, kvw=4), scale 0.5. It prints
y, dh, and the four Linears' weight/bias gradients as raw F32 bits. This
builds the same tensors in float64 and compares with oracle.py's tolerance.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "bendtorch", "reference"))
from oracle import close, parse  # noqa: E402

B, T, W, HD = 2, 3, 4, 4
N = B * T


def vals(left, i):
    return [((i * 7) % 13 - 6) / 4 for i in range(i, i + left)]


def mat(r, c, s):
    return torch.tensor(vals(r * c, s), dtype=torch.float32).reshape(r, c)


def main():
    binary = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(HERE), ".scratch", "dit_attn")
    out = __import__("subprocess").run([binary], check=True, capture_output=True, text=True)
    got = parse(out.stdout)

    x = mat(N, W, 33).requires_grad_(True)  # Bend's mat(6,4,33)
    wq, bq = mat(W, W, 3).requires_grad_(True), mat(1, W, 43).requires_grad_(True)
    wk, bk = mat(W, W, 7).requires_grad_(True), mat(1, W, 47).requires_grad_(True)
    wv, bv = mat(W, W, 11).requires_grad_(True), mat(1, W, 51).requires_grad_(True)
    wo, bo = mat(W, W, 15).requires_grad_(True), mat(1, W, 55).requires_grad_(True)

    q, k, v = (x @ w.T + b for w, b in ((wq, bq), (wk, bk), (wv, bv)))
    q, k, v = (z.reshape(B, T, HD) for z in (q, k, v))
    scores = (q @ k.transpose(-1, -2)) * 0.5
    attn = F.softmax(scores, dim=-1) @ v
    y = attn.reshape(N, HD) @ wo.T + bo

    dy = mat(N, W, 97)  # Bend's fixed dY
    grads = torch.autograd.grad(y, [x, wq, bq, wk, bk, wv, bv, wo, bo], dy,
                                retain_graph=False, allow_unused=True)
    names = ["x", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"]
    refs = dict(zip(names, grads))

    checks = [
        ("attn.y", y), ("attn.dh", grads[0]),
        ("attn.q.dw", grads[1]), ("attn.q.db", grads[2]),
        ("attn.k.dw", grads[3]), ("attn.k.db", grads[4]),
        ("attn.v.dw", grads[5]), ("attn.v.db", grads[6]),
        ("attn.o.dw", grads[7]), ("attn.o.db", grads[8]),
    ]
    failures, worst = [], 0.0
    for name, ref in checks:
        if name not in got:
            failures.append(f"{name}: not printed")
            continue
        # Both sides compute in f32; Bend prints f32 bits.
        worst = max(worst, close(name, got[name], ref, failures))
    if failures:
        for f in failures:
            print("FAIL", f)
        raise SystemExit(1)
    print(f"dit-attn: {len(checks)} tensors, worst error {worst:.3f} of the tolerance; PASS")


if __name__ == "__main__":
    main()
