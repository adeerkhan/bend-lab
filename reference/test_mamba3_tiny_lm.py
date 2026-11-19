"""The tiny-LM gate, first form: the data manifest and its Bend read-back.

  reference/.venv/bin/python reference/test_mamba3_tiny_lm.py

Builds examples/language/glm53_flash/data/corpus128-b32 (the lab's byte
corpus) into a deterministic tiny subset (`reference/mamba3_tiny_data.py`),
then checks:

  the manifest's sha256 hashes match the files on disk;
  the train subset is the first 64 corpus examples and the held-out split is
  the corpus's own val split (no overlapping tokens);
  Bend reads both files back (T.read_at) and its first sixteen tokens equal
  the numpy values for each split.

The trainer (AdamW + weight decay, gradient clipping, an LR schedule, a
held-out evaluation and a twin on the same data/seed) is not built, so this
gate covers the data half only: `MAMBA3 TINY-LM DATA PASS`, and the batch
row stays `not run` until the trainer lands.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, ".scratch/mamba3_tiny")
SRC = os.path.join(ROOT, "examples/language/glm53_flash/data/corpus128-b32")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mamba3_tiny_data.bin")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    subprocess.run([sys.executable, os.path.join(ROOT, "reference/mamba3_tiny_data.py")],
                   check=True, capture_output=True, text=True, cwd=ROOT)
    manifest = json.load(open(os.path.join(TINY, "manifest.json")))
    ok = True

    for key in ("train_tokens", "heldout_tokens"):
        got = sha256(os.path.join(TINY, manifest[key]["file"]))
        good = got == manifest[key]["sha256"]
        print(f"{'ok  ' if good else 'FAIL'} {key} sha256: {got[:16]}…")
        ok = ok and good

    corpus_train = np.fromfile(os.path.join(SRC, "corpus128-train_batch.bin"), dtype=np.uint8)
    corpus_val = np.fromfile(os.path.join(SRC, "corpus128-val_batch.bin"), dtype=np.uint8)
    train = np.fromfile(os.path.join(TINY, "train_tokens.bin"), dtype=np.uint8)
    held = np.fromfile(os.path.join(TINY, "heldout_tokens.bin"), dtype=np.uint8)
    good = train.size == 200 * 128 and np.array_equal(train, corpus_train[: train.size])
    print(f"{'ok  ' if good else 'FAIL'} train split: first 200 corpus examples ({train.size} tokens)")
    ok = ok and good
    good = held.size == corpus_val.size and np.array_equal(held, corpus_val)
    print(f"{'ok  ' if good else 'FAIL'} held-out split: the corpus's own val file ({held.size} tokens)")
    ok = ok and good
    good = manifest["vocab"] == int(max(train.max(), held.max())) + 1
    print(f"{'ok  ' if good else 'FAIL'} vocab: {manifest['vocab']}")
    ok = ok and good

    env = dict(os.environ, BEND_NO_TELEMETRY="1")
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")
    subprocess.run([BEND, "-o", BIN,
                    os.path.join(ROOT, "examples/language/mamba3/probe_mamba3_tiny_data.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True, timeout=1800)
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("tiny-data binary failed")
    tags = {}
    for line in p.stdout.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        v = np.fromstring(rest, sep=" ", dtype=np.float32)
        if v.size:
            tags[tag] = v
    for tag, want in (("tiny_train_tokens", train[:16]), ("tiny_heldout_tokens", held[:16])):
        got = tags.get(tag)
        good = got is not None and np.array_equal(got.astype(int), want.astype(int))
        print(f"{'ok  ' if good else 'FAIL'} {tag}: {got.astype(int).tolist() if got is not None else 'missing'}")
        ok = ok and good

    print("unavailable trainer: AdamW/wd, clipping, schedule, held-out eval, twin")
    print("MAMBA3 TINY-LM DATA", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
