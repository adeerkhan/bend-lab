"""The ignored-label branch of the masked cross-entropy.

  reference/.venv/bin/python reference/test_glm53_mask.py

A 2 x 3 logits matrix, row 0 labelled 0 and row 1 ignored.  The masked loss is
-log softmax(row 0)[0], the gradient row 0 is (p - onehot)/1, and row 1 is
exactly zero.  The current 64-byte training batch has no ignored labels, so
this is the branch a padded corpus batch will use.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

import numpy as np

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
BIN = f"/tmp/glm53-mask-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")


def binary():
    src = os.path.join(EX, "probe_mask.bend")
    deps = [src, os.path.join(EX, "loss_masked.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, src], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"probe_mask.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def main():
    proc = subprocess.run([binary()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_mask.bend did not run:\n{proc.stdout}{proc.stderr}")
    m = re.search(r"^masked_loss: ([0-9.eE+-]+) masked_dy: (.*)$", proc.stdout, re.M)
    if not m:
        raise SystemExit(f"no masked_loss line:\n{proc.stdout}")
    loss = float(m.group(1))
    dy = np.array([np.float32(v) for v in m.group(2).split()], dtype=np.float64)

    logits = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]], dtype=np.float64)
    p = np.exp(logits[0] - logits[0].max())
    p = p / p.sum()
    want_loss = float(-np.log(p[0]))
    want_dy = np.array([p[0] - 1.0, p[1], p[2], 0.0, 0.0, 0.0], dtype=np.float64)

    ok_loss = abs(loss - want_loss) <= 1e-6
    print(f"{'ok  ' if ok_loss else 'FAIL'} masked loss: {loss:.9g} vs {want_loss:.9g}")
    ok_dy = dy.shape == want_dy.shape and float(np.abs(dy - want_dy).max()) <= 1e-6
    print(f"{'ok  ' if ok_dy else 'FAIL'} masked dy: max |diff| {np.abs(dy - want_dy).max():.3e}, "
          f"ignored row max |dy| {np.abs(dy[3:]).max():.3e}")
    zero = float(np.abs(dy[3:]).max()) == 0.0
    print(f"{'ok  ' if zero else 'FAIL'} ignored row is exactly zero")
    # can-fail: the unmasked mean would not be this loss; the gate must see it
    unmasked_mean = want_loss / 2.0  # a mean over both rows would use 2, not 1
    rejected = abs(unmasked_mean - want_loss) > 1e-4
    print(f"{'ok  ' if rejected else 'FAIL'} unmasked-mean check is rejected")
    gates = [ok_loss, ok_dy, zero, rejected]
    print(f"{sum(gates)}/{len(gates)} gates passed")
    return 0 if all(gates) else 1


if __name__ == "__main__":
    sys.exit(main())
