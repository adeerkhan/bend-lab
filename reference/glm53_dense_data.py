"""Weights, corpus and batches for the dense Stage A baseline.

  reference/.venv/bin/python reference/glm53_dense_data.py --prepare

The tensors are the same NumPy-generated set the Stage A gate uses
(`reference/test_glm53_layers.py`): width 32, four blocks, four heads of 8,
SwiGLU 128, vocabulary 260, tied embedding/head, with the q/k rows stored in
BendTorch's half-split RoPE order.  The training batch is the first 129 tokens
of the reference's synthetic Python corpus, encoded with its byte tokenizer:
two fixed rows of 64 inputs and two rows of 64 next-token labels.  The
evaluation prompts are the frozen dev/confirm tasks of the vendored reference,
encoded with `bos=True` and padded to the 192 positions the generation model
uses.

Nothing here imports PyTorch; `reference/glm53_dense_twin.py` imports this
module and adds the autograd twin.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V, DIM, LAYERS, H, HD, FF, T, B = 260, 32, 4, 4, 8, 128, 64, 2
N = B * T
P = [0, 4, 1, 5, 2, 6, 3, 7]
PI = [P.index(i) for i in range(len(P))]
GEN_POSITIONS = 192
IGNORE = 260
DATA = os.path.join(ROOT, "examples/language/glm53_flash/data")
VENDOR = os.path.join(ROOT, "reference/glm53_flash")


def shapes():
    out = [("emb", (V, DIM))]
    for i in range(LAYERS):
        out += [(f"b{i}/ln1", (DIM,)), (f"b{i}/wq", (DIM, DIM)), (f"b{i}/wk", (DIM, DIM)),
                (f"b{i}/wv", (DIM, DIM)), (f"b{i}/wo", (DIM, DIM)), (f"b{i}/ln2", (DIM,)),
                (f"b{i}/wg", (FF, DIM)), (f"b{i}/wu", (FF, DIM)), (f"b{i}/wd", (DIM, FF))]
    return out + [("lnf", (DIM,))]


def weights():
    """The exact initial set of the Stage A gate, in the file layout."""
    rng = np.random.default_rng(7)
    out = {}
    for name, shape in shapes():
        x = np.ones(shape, dtype=np.float32) if name.endswith(("ln1", "ln2", "lnf")) else \
            (rng.standard_normal(shape) * 0.02).astype(np.float32)
        y = x.copy()
        if name.endswith(("wq", "wk")):
            for h in range(H):
                for j in range(HD):
                    y[h * HD + j] = x[h * HD + PI[j]]
        out[name] = y
    return out


def permute_rows(x):
    """File layout from reference layout (the q/k RoPE bridge)."""
    y = x.copy()
    if x.shape == (DIM, DIM):
        for h in range(H):
            for j in range(HD):
                y[h * HD + j] = x[h * HD + PI[j]]
    return y


def flatten(values, *, store_qk=False):
    """The order of `T.Params.save`: emb, each block's nine tensors, lnf."""
    parts = []
    for name, _ in shapes():
        x = permute_rows(values[name]) if store_qk and name.endswith(("wq", "wk")) else values[name]
        parts.append(np.asarray(x, dtype=np.float32).reshape(-1))
    return np.concatenate(parts)


def unflatten(flat):
    out = {}
    at = 0
    for name, shape in shapes():
        n = int(np.prod(shape))
        out[name] = np.asarray(flat[at:at + n], dtype=np.float32).reshape(shape).copy()
        at += n
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


def write_flat_weights(flat, directory):
    """Split a `T.Params.save` file into the per-tensor files the loader reads."""
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    w = unflatten(flat)
    for name, _ in shapes():
        path = os.path.join(directory, name + ".bin")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(w[name].astype("<f4").tobytes())
    return w


def _tokenizer():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from glm53_flash.tokenizer import ByteTokenizer
    return ByteTokenizer()


def corpus_stream(examples=512):
    """The reference's synthetic corpus, encoded, concatenated."""
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from glm53_flash.tasks import pretraining_text
    tokenizer = _tokenizer()
    stream = []
    for index in range(examples):
        stream += tokenizer.encode(pretraining_text(index), bos=True, eos=True)
    return stream


def train_batch():
    """Two fixed 64-byte rows of inputs and the next-byte labels."""
    stream = corpus_stream()
    x = np.asarray(stream[:N], dtype=np.uint8)
    y = np.asarray(stream[1:N + 1], dtype=np.uint8)
    if x.max() >= 256 or y.max() >= 256:
        raise SystemExit("the first corpus window does not fit a byte token id")
    return x, y


def write_train_batch(directory):
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    x, y = train_batch()
    open(os.path.join(directory, "train_batch.bin"), "wb").write(x.tobytes())
    open(os.path.join(directory, "train_labels.bin"), "wb").write(y.tobytes())
    return x, y


def eval_prompts(split, per_family=8):
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from glm53_flash.tasks import frozen_tasks
    tokenizer = _tokenizer()
    tasks = frozen_tasks(split, per_family=per_family)
    rows, lengths = [], []
    for task in tasks:
        ids = tokenizer.encode(task.prompt, bos=True)
        if len(ids) + 48 > GEN_POSITIONS:
            raise SystemExit(f"{task.task_id}: prompt plus 48 tokens exceeds {GEN_POSITIONS}")
        rows.append(ids + [0] * (GEN_POSITIONS - len(ids)))
        lengths.append(len(ids))
    return tasks, np.asarray(rows, dtype=np.uint8), np.asarray(lengths, dtype=np.uint8)


def write_eval_prompts(directory, split, per_family=8):
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    tasks, rows, lengths = eval_prompts(split, per_family)
    open(os.path.join(directory, f"eval-{split}.bin"), "wb").write(rows.tobytes())
    open(os.path.join(directory, f"eval-{split}-len.bin"), "wb").write(lengths.tobytes())
    return tasks, rows, lengths


def prepare(directory=DATA, per_family=8):
    write_weights(directory)
    write_train_batch(directory)
    for split in ("dev", "confirm"):
        write_eval_prompts(directory, split, per_family)
    return directory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--directory", default=DATA)
    args = ap.parse_args()
    if not args.prepare:
        ap.error("--prepare is required")
    prepare(args.directory)
    x, y = train_batch()
    print(f"weights: {sum(int(np.prod(s)) for _, s in shapes())} values")
    print(f"train inputs: {x.size} bytes, labels: {y.size} bytes, masked labels: {int((y == IGNORE).sum())}")
    for split in ("dev", "confirm"):
        _, rows, lengths = eval_prompts(split)
        print(f"{split}: {len(rows)} prompts, {int(lengths.min())}-{int(lengths.max())} tokens, {rows.shape[1]} positions")


if __name__ == "__main__":
    main()
