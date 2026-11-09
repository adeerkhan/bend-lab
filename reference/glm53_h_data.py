"""Weights for the four-block hybrid at 128 positions.

  reference/.venv/bin/python reference/glm53_h_data.py --prepare
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "examples/language/glm53_flash/data")
H = os.path.join(DATA, "h")
V, D, FF, LAYERS = 260, 32, 128, 4


def shapes():
    out = [("emb", (V, D))]
    for i in range(LAYERS):
        out += [(f"b{i}/ln1", (1, D)), (f"b{i}/qkv", (3 * D, D)), (f"b{i}/out", (D, D)),
                (f"b{i}/ln2", (1, D)), (f"b{i}/gate", (FF, D)), (f"b{i}/up", (FF, D)),
                (f"b{i}/down", (D, FF))]
    return out + [("lnf", (1, D))]


def weights(seed=41):
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in shapes():
        out[name] = (np.ones(shape, dtype=np.float32) if name.endswith(("ln1", "ln2", "lnf"))
                     else (rng.standard_normal(shape) * 0.02).astype(np.float32))
    return out


def write_weights(directory=H):
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    w = weights()
    for name, _ in shapes():
        path = os.path.join(directory, name + ".bin")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(w[name].astype("<f4").tobytes())
    return w


def flatten(w):
    return np.concatenate([np.asarray(w[name], dtype=np.float32).reshape(-1) for name, _ in shapes()])


def write_flat_weights(flat, directory):
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    at = 0
    for name, shape in shapes():
        n = int(np.prod(shape))
        value = np.asarray(flat[at:at + n], dtype=np.float32).reshape(shape)
        path = os.path.join(directory, name + ".bin")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(value.astype("<f4").tobytes())
        at += n
    return at


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    if not args.prepare:
        ap.error("--prepare is required")
    write_weights()
    print(f"weights: {sum(int(np.prod(s)) for _, s in shapes())} values in {H}")


if __name__ == "__main__":
    main()
