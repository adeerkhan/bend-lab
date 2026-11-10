"""The PyTorch-autograd golden for the four-head attention backward at N=8.

  reference/.venv-rf/bin/python reference/glm53_attn_bwd.py \
      --golden reference/glm53_attn_bwd_golden.npz

The modules are the vendored `LinearAttention` and `SparseAttention` (dim 32,
four heads, window 16, stride 16).  qkv and out are the same weights the Bend
probe loads; the loss is sum(y * dout).  The gradients are dx, dqkv (the qkv
projection's weight) and dout (the out projection's weight).  A second pass
with dout_bad gives the can-fail gradient: it must match the bad-dout golden
and must NOT match the correct one.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/glm53_flash"))
import glm53_attn_bwd128_data as D  # noqa: E402
from glm53_flash.model import LinearAttention, SparseAttention, ModelConfig  # noqa: E402

N, DIM, H = 128, 32, 4


def module(name):
    cfg = ModelConfig(dim=DIM, heads=H, expert_hidden=128, experts=8, streams=4,
                      sparse_window=16, sparse_stride=16, max_sequence_length=192)
    return LinearAttention(cfg) if name == "lin" else SparseAttention(cfg)


def grads(name, w, dout):
    torch.set_num_threads(1)
    x = torch.from_numpy(w["x"]).clone().requires_grad_(True)
    m = module(name)
    m.qkv.weight = torch.nn.Parameter(torch.from_numpy(w["qkv"]).clone())
    m.out.weight = torch.nn.Parameter(torch.from_numpy(w["out"]).clone())
    y = m(x)
    (y * torch.from_numpy(dout)).sum().backward()
    return {
        "y": y.detach().numpy().reshape(-1).astype(np.float32),
        "dx": x.grad.numpy().reshape(-1).astype(np.float32),
        "dqkv": m.qkv.weight.grad.numpy().reshape(-1).astype(np.float32),
        "dout": m.out.weight.grad.numpy().reshape(-1).astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=os.path.abspath, default=None)
    args = ap.parse_args()
    w = D.write()
    golden = {}
    for name in ("lin", "sp"):
        good = grads(name, w, w["dout"])
        bad = grads(name, w, w["dout_bad"])
        for key, value in good.items():
            golden[f"{name}_{key}"] = value
            print(f"{name}/{key}: {value.size} values, max |value| {float(np.abs(value).max()):.6e}")
        if name == "lin":
            for key in ("dx", "dqkv", "dout"):
                golden[f"lin_bad_{key}"] = bad[key]
    if args.golden:
        np.savez(args.golden, **golden)
        print(f"wrote {args.golden}")


if __name__ == "__main__":
    main()
