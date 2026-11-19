"""The tiny-LM data: a deterministic subset of the byte corpus, with a manifest.

  reference/.venv/bin/python reference/mamba3_tiny_data.py

Reads `examples/language/glm53_flash/data/corpus128-b32/` (4000 train
examples, 8 held-out, 128 byte positions each, uint32 tokens) and writes the
tiny-LM's own files to `.scratch/mamba3_tiny/`:

  train_tokens.bin  200 examples x 128 positions = 25600 uint32 tokens
  heldout_tokens.bin  32 examples x 128 positions = 4096 uint32 tokens
  manifest.json      source, hashes, shapes, vocab (max token + 1)

The subset is the first 200 train examples and the whole held-out split: a
real text stream, one named split, no shuffling that would leak the split.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "examples/language/glm53_flash/data/corpus128-b32")
OUT = os.path.join(ROOT, ".scratch/mamba3_tiny")
POSITIONS = 128
TRAIN_EXAMPLES = 200


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    train = np.fromfile(os.path.join(SRC, "corpus128-train_batch.bin"), dtype=np.uint8)
    held = np.fromfile(os.path.join(SRC, "corpus128-val_batch.bin"), dtype=np.uint8)[: 32 * POSITIONS]
    train = train[: TRAIN_EXAMPLES * POSITIONS]
    train.astype(np.uint8).tofile(os.path.join(OUT, "train_tokens.bin"))
    held.astype(np.uint8).tofile(os.path.join(OUT, "heldout_tokens.bin"))
    # the corpus is a byte stream (uint8 ids, vocab at most 256); keep the
    # observed maximum so the model's head is exact for this subset
    vocab = int(max(train.max(), held.max())) + 1
    manifest = {
        "source": os.path.relpath(SRC, ROOT),
        "source_sha256": {
            "corpus128-train_batch.bin": sha256(os.path.join(SRC, "corpus128-train_batch.bin")),
            "corpus128-val_batch.bin": sha256(os.path.join(SRC, "corpus128-val_batch.bin")),
        },
        "train_tokens": {"file": "train_tokens.bin", "examples": TRAIN_EXAMPLES,
                         "positions": POSITIONS, "count": int(train.size), "dtype": "uint8",
                         "sha256": sha256(os.path.join(OUT, "train_tokens.bin"))},
        "heldout_tokens": {"file": "heldout_tokens.bin",
                           "examples": int(held.size // POSITIONS),
                           "positions": POSITIONS, "count": int(held.size),
                           "sha256": sha256(os.path.join(OUT, "heldout_tokens.bin"))},
        "vocab": vocab,
        "split": "train = the first 200 corpus examples; heldout = the corpus's own 32-example val split",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps({k: manifest[k] for k in ("vocab", "split")}, indent=2))
    print("train", train.size, "tokens  heldout", held.size, "tokens")


if __name__ == "__main__":
    main()
