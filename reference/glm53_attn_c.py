"""PyTorch-autograd goldens for the integrated Stage B/C attention cores.

  reference/.venv-rf/bin/python reference/glm53_attn_c.py \
      --golden reference/glm53_attn_c_golden.npz

The inputs are the same 64 x 8 pre-RoPE q, k, v and dout files
`glm53_linear_bwd.py` writes.  Both cores apply RoPE first (interleaved
even/odd, one head), then either the ELU+1 linear core or the gathered causal
sparse core with window 16 and stride 16.  The loss is sum(y * dout).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "examples/language/glm53_flash/data")
N, HD = 64, 8
SCALE = HD ** -0.5


def load(directory=DATA):
    return {name: torch.from_numpy(np.fromfile(os.path.join(directory, f"lin64_{name}.bin"),
                                               dtype="<f4").reshape(N, HD)).requires_grad_(True)
            for name in ("q", "k", "v", "dy")}


def rope(x):
    positions = torch.arange(N, dtype=torch.float32)[:, None]
    freqs = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))[None, :]
    angles = positions * freqs
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[:, 0::2], x[:, 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def linear(q, k, v):
    q, k = rope(q), rope(k)
    qe, ke = F.elu(q) + 1.0, F.elu(k) + 1.0
    K = torch.zeros(HD)
    KV = torch.zeros(HD, HD)
    ys = []
    for t in range(N):
        K = K + ke[t]
        KV = KV + torch.outer(ke[t], v[t])
        num = KV.T @ qe[t]
        ys.append(num / (qe[t] @ K).clamp_min(1e-6))
    return torch.stack(ys)


def sparse(q, k, v):
    q, k = rope(q), rope(k)
    ys = []
    for t in range(N):
        idx = [s for s in range(t + 1) if s % 16 == 0 or t - s < 16]
        scores = torch.stack([q[t] @ k[s] for s in idx]) * SCALE
        p = torch.softmax(scores, dim=0)
        ys.append(sum(p[j] * v[s] for j, s in enumerate(idx)))
    return torch.stack(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=os.path.abspath, default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    w = load()
    out = {}
    for name, fn in (("lin", linear), ("sp", sparse)):
        q = w["q"].detach().clone().requires_grad_(True)
        k = w["k"].detach().clone().requires_grad_(True)
        v = w["v"].detach().clone().requires_grad_(True)
        y = fn(q, k, v)
        (y * w["dy"]).sum().backward()
        out[f"{name}_y"] = y.detach().numpy().reshape(-1).astype(np.float32)
        out[f"{name}_dq"] = q.grad.numpy().reshape(-1).astype(np.float32)
        out[f"{name}_dk"] = k.grad.numpy().reshape(-1).astype(np.float32)
        out[f"{name}_dv"] = v.grad.numpy().reshape(-1).astype(np.float32)
        print(f"{name}: y first {y[0].detach().numpy()}")
    if args.golden:
        np.savez(args.golden, **out)
        print(f"wrote {args.golden}")


if __name__ == "__main__":
    main()
