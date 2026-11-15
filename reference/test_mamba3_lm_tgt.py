"""Gate: the explicit-target loss/dy/dWh against a small NumPy twin (B1).

  reference/.venv/bin/python reference/test_mamba3_lm_tgt.py

Dumps a deterministic y (8x64), Wh (8x64) and eight targets (3, 1, 4, 1, 5,
9, 2, 6) to `.scratch/mamba3_tgt/`, builds `probe_mamba3_lm_tgt.bend`, runs
it and compares at 1e-5 relative:

  tgt_loss   the mean-over-positions cross entropy against the targets
  tgt_dy     (8, 64) the gradient through the head
  tgt_dwh    (8, 64) the gradient of Wh

The current `loss_go`/`dy_all_go`/`dWh_go` target the row index (the copy
task); these variants take the target list instead, and are what a
next-token (byte) trainer needs. A missing tag or a mismatch is FAIL, never
a zero or an estimate.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, ".scratch/mamba3_tgt")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mamba3_lm_tgt.bin")
GATE = 1e-5
TARGETS = [3, 1, 4, 1, 5, 7, 2, 6]


def softmax_rows(logits):
    m = logits.max(-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(-1, keepdims=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(20261115)
    y = (rng.randn(8, 64) * 0.5).astype("<f4")
    wh = (rng.randn(8, 64) * 0.5).astype("<f4")
    y.tofile(os.path.join(OUT, "y.bin"))
    wh.tofile(os.path.join(OUT, "Wh.bin"))
    np.asarray(TARGETS, dtype=np.uint8).tofile(os.path.join(OUT, "targets.bin"))

    env = dict(os.environ, BEND_NO_TELEMETRY="1")
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")
    subprocess.run([BEND, "-o", BIN,
                    os.path.join(ROOT, "examples/language/mamba3/probe_mamba3_lm_tgt.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True, timeout=1800)
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("mamba3-lm-tgt binary failed")

    tags = {}
    for line in p.stdout.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        v = np.fromstring(rest, sep=" ", dtype=np.float32)
        if v.size:
            tags[tag] = v

    y64 = y.astype(np.float64)
    w64 = wh.astype(np.float64)
    logits = y64 @ w64.T
    probs = softmax_rows(logits)
    n = len(TARGETS)
    inv = 1.0 / n
    want_loss = float(sum(-np.log(probs[t, TARGETS[t]]) * inv for t in range(n)))
    dl = probs.copy()
    for t in range(n):
        dl[t, TARGETS[t]] -= 1.0
    dl *= inv
    want_dy = dl @ w64
    want_dwh = dl.T @ y64

    ok = True

    def chk(tag, got, want, shape):
        nonlocal ok
        got = np.asarray(got, dtype=np.float64).reshape(shape)
        d = np.abs(got - want)
        rel = float(d.max()) / max(1e-30, float(np.abs(want).max()))
        good = rel <= GATE
        print(f"{'ok  ' if good else 'FAIL'} {tag}: max|d| {d.max():.6e} "
              f"rel {rel:.6e} gate {GATE:.0e}")
        ok = ok and good

    chk("tgt_loss", tags.get("tgt_loss", np.asarray([])), np.asarray([want_loss]),
        (1,))
    chk("tgt_dy", tags.get("tgt_dy", np.asarray([])), want_dy, (8, 64))
    chk("tgt_dwh", tags.get("tgt_dwh", np.asarray([])), want_dwh, (8, 64))
    print("MAMBA3 LM TGT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
