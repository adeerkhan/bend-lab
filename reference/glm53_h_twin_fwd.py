"""Forward twin of the four-block hybrid at N=8.

  reference/.venv/bin/python reference/glm53_h_twin_fwd.py

Loads the same per-tensor weights as the Bend model, runs the same four
blocks (linear, linear, linear, sparse) with the NumPy reference attention,
and writes the ids and expected logits for the probe.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import glm53_h_data as D  # noqa: E402
import test_glm53_mh as R  # noqa: E402

N, DIM, HID = 8, 32, 128


def rms(x, g):
    return g * (x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6))


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    w = D.weights()
    ids = np.fromfile(os.path.join(ROOT, "examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin"),
                      dtype=np.uint8)[:N]
    os.makedirs(os.path.join(ROOT, "examples/language/glm53_flash/data/ref8"), exist_ok=True)
    ids.astype(np.uint8).tofile(os.path.join(ROOT, "examples/language/glm53_flash/data/ref8/ids.bin"))
    x = w["emb"][ids].astype(np.float64)
    for i in range(4):
        h = rms(x, w[f"b{i}/ln1"].astype(np.float64))
        if i == 3:
            a, _ = R.sparse(h, w[f"b{i}/qkv"].astype(np.float64), w[f"b{i}/out"].astype(np.float64))
        else:
            a, _ = R.linear(h, w[f"b{i}/qkv"].astype(np.float64), w[f"b{i}/out"].astype(np.float64))
        x = x + a.reshape(N, DIM)
        h = rms(x, w[f"b{i}/ln2"].astype(np.float64))
        gate = h @ w[f"b{i}/gate"].astype(np.float64).T
        up = h @ w[f"b{i}/up"].astype(np.float64).T
        x = x + (silu(gate) * up) @ w[f"b{i}/down"].astype(np.float64).T
    x = rms(x, w["lnf"].astype(np.float64))
    logits = x @ w["emb"].astype(np.float64).T
    logits.astype("<f4").tofile(os.path.join(ROOT, "examples/language/glm53_flash/data/ref8/logits.bin"))
    print("ids", ids.tolist())
    print("logits", logits.shape, float(np.abs(logits).max()))


if __name__ == "__main__":
    main()
