"""The PyTorch-autograd golden for the Stage B linear-attention backward at 64.

  reference/.venv-rf/bin/python reference/glm53_linear_bwd.py \
      --golden reference/glm53_linear_bwd_golden.npz

One head, 64 positions, head width 8.  The inputs q, k, v and dout are the
same deterministic files the Bend probe reads.  The forward is the reference
formula written out with autograd: q~ = ELU(q)+1, k~ = ELU(k)+1, causal prefix
K and KV, y = KV^T q~ / clamp(q~·K, 1e-6), and the loss is sum(y * dout).
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


def inputs(seed=13, n=N):
    rng = np.random.default_rng(seed)
    return {name: (rng.standard_normal((n, HD)) * 0.7).astype(np.float32)
            for name in ("q", "k", "v", "dy")}


def write_inputs(directory=DATA, n=N, seed=13):
    os.makedirs(directory, exist_ok=True)
    w = inputs(seed, n)
    for name, value in w.items():
        value.astype("<f4").tofile(os.path.join(directory, f"lin{n}_{name}.bin"))
    return w


def forward(w):
    q, k, v = w["q"], w["k"], w["v"]
    qe, ke = F.elu(q) + 1.0, F.elu(k) + 1.0
    K = torch.zeros(HD)
    KV = torch.zeros(HD, HD)
    ys = []
    for t in range(N):
        K = K + ke[t]
        KV = KV + torch.outer(ke[t], v[t])
        den = qe[t] @ K
        num = KV.T @ qe[t]
        ys.append(num / den.clamp_min(1e-6))
    return torch.stack(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=os.path.abspath, default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    w = write_inputs()
    q, k, v, dy = (torch.from_numpy(w[name]).requires_grad_(True) for name in ("q", "k", "v", "dy"))
    y = forward({"q": q, "k": k, "v": v})
    (y * dy).sum().backward()
    print(f"y: {y.numel()} values, dq/dk/dv: {q.grad.numel()} each")
    print(f"y[0] {y[0].detach().numpy()}")
    if args.golden:
        np.savez(args.golden, y=y.detach().numpy().reshape(-1).astype(np.float32),
                 dq=q.grad.numpy().reshape(-1).astype(np.float32),
                 dk=k.grad.numpy().reshape(-1).astype(np.float32),
                 dv=v.grad.numpy().reshape(-1).astype(np.float32))
        print(f"wrote {args.golden}")


if __name__ == "__main__":
    main()
