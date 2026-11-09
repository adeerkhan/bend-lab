"""Forward outputs of the vendored reference attention modules at N = 8.

  reference/.venv-rf/bin/python reference/glm53_attn_ref.py

Writes x, qkv, out, y_linear and y_sparse as F32 files for
examples/language/glm53_flash/data/ref8.  The modules are the vendored
`LinearAttention` and `SparseAttention` with dim 32, four heads, window 16,
stride 16, and the same qkv/out weights the Bend probe loads.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference/glm53_flash"))
from glm53_flash.model import LinearAttention, SparseAttention, ModelConfig  # noqa: E402

N, D, H = 8, 32, 4
OUT = os.path.join(ROOT, "examples/language/glm53_flash/data/ref8")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(71)
    x = (rng.standard_normal((1, N, D)) * 0.5).astype(np.float32)
    qkv = (rng.standard_normal((3 * D, D)) * 0.05).astype(np.float32)
    out = (rng.standard_normal((D, D)) * 0.05).astype(np.float32)
    for name, value in (("x", x), ("qkv", qkv), ("out", out)):
        value.astype("<f4").tofile(os.path.join(OUT, name + ".bin"))
    cfg = ModelConfig(dim=D, heads=H, expert_hidden=128, experts=8, streams=4,
                      sparse_window=16, sparse_stride=16, max_sequence_length=192)
    torch.set_num_threads(1)
    xt = torch.from_numpy(x)
    for name, module in (("y_linear", LinearAttention(cfg)), ("y_sparse", SparseAttention(cfg))):
        module.qkv.weight = torch.nn.Parameter(torch.from_numpy(qkv))
        module.out.weight = torch.nn.Parameter(torch.from_numpy(out))
        with torch.no_grad():
            y = module(xt)
        y.numpy().astype("<f4").tofile(os.path.join(OUT, name + ".bin"))
        print(name, y.shape, float(y.abs().max()))


if __name__ == "__main__":
    main()
