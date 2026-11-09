"""Weights and the single 64-byte batch for the integrated Stage C model.

  reference/.venv/bin/python reference/glm53_c_data.py --prepare

The model is one batch, 64 positions, one head of width 8, two blocks: linear
attention then sparse attention, each with a SwiGLU hidden 32.  The parameter
order here is the order of `T.Params.save` for `model_c.bend`.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glm53_dense_data as D  # noqa: E402

DATA = os.path.join(ROOT, "examples/language/glm53_flash/data/c")
V, DIM, H, T = 260, 8, 32, 64


def shapes():
    out = [("emb", (V, DIM))]
    for name in ("b0", "b1"):
        out += [
            (f"{name}/ln1", (1, DIM)), (f"{name}/qkv", (3 * DIM, DIM)), (f"{name}/out", (DIM, DIM)),
            (f"{name}/ln2", (1, DIM)), (f"{name}/gate", (H, DIM)), (f"{name}/up", (H, DIM)),
            (f"{name}/down", (DIM, H)),
        ]
    return out + [("lnf", (1, DIM))]


def weights(seed=17):
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in shapes():
        out[name] = (np.ones(shape, dtype=np.float32) if name.endswith(("ln1", "ln2", "lnf"))
                     else (rng.standard_normal(shape) * 0.02).astype(np.float32))
    return out


def write_weights(directory):
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    w = weights()
    for name, _ in shapes():
        path = os.path.join(directory, name + ".bin")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(w[name].astype("<f4").tobytes())
    return w


def batch():
    stream = D.corpus_stream()
    x = np.asarray(stream[:T], dtype=np.uint8)
    y = np.asarray(stream[1:T + 1], dtype=np.uint8)
    return x, y


def write_batch(directory):
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    x, y = batch()
    open(os.path.join(directory, "c_batch.bin"), "wb").write(x.tobytes())
    open(os.path.join(directory, "c_labels.bin"), "wb").write(y.tobytes())
    return x, y


def flatten(w):
    return np.concatenate([np.asarray(w[name], dtype=np.float32).reshape(-1) for name, _ in shapes()])


def prepare(directory=DATA):
    write_weights(directory)
    write_batch(directory)
    return directory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    if not args.prepare:
        ap.error("--prepare is required")
    prepare()
    w = weights()
    x, y = batch()
    print(f"weights: {sum(int(np.prod(s)) for _, s in shapes())} values")
    print(f"batch: {x.size} inputs, {y.size} labels, masked {int((y == D.IGNORE).sum())}")


if __name__ == "__main__":
    main()
