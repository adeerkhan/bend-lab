"""The deterministic batch for the full-model gradient gate at N=8.

  reference/.venv/bin/python reference/glm53_h_bwd_data.py --write

The ids and labels are the first eight bytes of the corpus batch and its
next-token labels (`glm53_dense_data.write_windows` wrote them).  labels_bad
moves one label by +1, so the loss gradient changes at step 1 and the
comparison must see it.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "examples/language/glm53_flash/data/corpus128-b32")
OUT = os.path.join(ROOT, "examples/language/glm53_flash/data/ref8")
N = 8


def batch():
    ids = np.fromfile(os.path.join(CORPUS, "corpus128-train_batch.bin"), dtype=np.uint8)[:N]
    labels = np.fromfile(os.path.join(CORPUS, "corpus128-train_labels.bin"), dtype=np.uint8)[:N]
    bad = labels.copy()
    bad[0] = (int(bad[0]) + 1) % 256
    return ids.astype(np.uint8), labels.astype(np.uint8), bad.astype(np.uint8)


def write(directory=OUT):
    os.makedirs(directory, exist_ok=True)
    ids, labels, bad = batch()
    ids.tofile(os.path.join(directory, "model_ids.bin"))
    labels.tofile(os.path.join(directory, "model_labels.bin"))
    bad.tofile(os.path.join(directory, "model_labels_bad.bin"))
    return ids, labels, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if not args.write:
        ap.error("--write is required")
    ids, labels, bad = write()
    print(f"ids {ids.tolist()}")
    print(f"labels {labels.tolist()}")
    print(f"bad {bad.tolist()}")


if __name__ == "__main__":
    main()
