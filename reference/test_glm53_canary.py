"""The bit-identical step canary for the hybrid at 128 positions.

  reference/.venv/bin/python reference/test_glm53_canary.py

Runs `train_h128 --steps 1 --no-val` twice from the same weights and requires
the same loss and the same final-weight sha256.  A third run on a corpus copy
with one input token changed must differ.  This is the gate that catches a
row-order or scale bug a tolerance would hide.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
BIN = f"/tmp/glm53-canary-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")
DATA = os.path.join(EX, "data/corpus128-b32")


def binary():
    deps = [os.path.join(EX, name) for name in
            ("train_h128.bend", "model_h.bend", "weights_h.bend", "loss_masked.bend",
             "attn_mh.bend", "lin_dense.bend", "sp_dense.bend", "lin_bwd.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(EX, "train_h128.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"train_h128.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def run(data, out):
    os.makedirs(out, exist_ok=True)
    proc = subprocess.run([binary(), "--steps", "1", "--no-val", "--data", data, "--out", out],
                          capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"train_h128 did not run:\n{proc.stdout}{proc.stderr}")
    m = re.search(r"^final_loss: ([0-9.eE+-]+)$", proc.stdout, re.M)
    if m is None:
        raise SystemExit(f"no final_loss:\n{proc.stdout}")
    weights = os.path.join(out, "final_weights.bin")
    return float(m.group(1)), hashlib.sha256(open(weights, "rb").read()).hexdigest()


def main():
    tmp = tempfile.mkdtemp(prefix="glm53-canary-", dir="/tmp/opencode")
    try:
        bad_data = os.path.join(tmp, "bad-data")
        shutil.copytree(DATA, bad_data)
        path = os.path.join(bad_data, "corpus128-train_batch.bin")
        blob = bytearray(open(path, "rb").read())
        blob[0] = (blob[0] + 1) % 256
        open(path, "wb").write(blob)
        a = run(DATA, os.path.join(tmp, "a"))
        b = run(DATA, os.path.join(tmp, "b"))
        c = run(bad_data, os.path.join(tmp, "c"))
        ok = []
        same_loss = a[0] == b[0]
        print(f"{'ok  ' if same_loss else 'FAIL'} two runs, same loss: {a[0]} vs {b[0]}")
        ok.append(same_loss)
        same_hash = a[1] == b[1]
        print(f"{'ok  ' if same_hash else 'FAIL'} two runs, same weight sha256: {a[1]}")
        ok.append(same_hash)
        diff_loss = a[0] != c[0]
        print(f"{'ok  ' if diff_loss else 'FAIL'} changed token, different loss: {a[0]} -> {c[0]}")
        ok.append(diff_loss)
        diff_hash = a[1] != c[1]
        print(f"{'ok  ' if diff_hash else 'FAIL'} changed token, different weight sha256")
        ok.append(diff_hash)
        print(f"{sum(ok)}/{len(ok)} gates passed")
        return 0 if all(ok) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
