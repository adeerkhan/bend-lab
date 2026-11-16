"""Gate B6: determinism — two builds, identical seqlen-8 checksums.

  reference/.venv/bin/python reference/test_mamba3_det.py

Builds examples/language/mamba3/mamba3_siso.bend twice (two separate
output paths), runs both binaries, and compares the printed forward
tags exactly.  If they differ, FAIL prints both runs' sums of squares
and the mismatch is recorded in the ledger (.local/issues-hit.md).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
SCRATCH = os.path.join(ROOT, ".scratch")
SRC = os.path.join(ROOT, "examples/language/mamba3/mamba3_siso.bend")
TAGS = ["out_v", "y_inner", "y"]


def parse(out):
    tags = {}
    for line in out.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        try:
            vals = np.fromstring(rest, sep=" ", dtype=np.float32)
        except Exception:
            continue
        if vals.size:
            tags[tag] = vals
    return tags


def build(bin_path, env):
    t0 = time.time()
    p = subprocess.run([BEND, "-o", bin_path, SRC], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return p.returncode, time.time() - t0, (p.stdout or "") + (p.stderr or "")


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    env = dict(os.environ)
    env["BEND_NO_TELEMETRY"] = "1"
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")

    ok = True
    runs = []
    for label in ("build A", "build B"):
        bin_path = os.path.join(SCRATCH, "mamba3_det_" + label[-1].lower() + ".bin")
        code, dt, out = build(bin_path, env)
        print(f"{label}: build exit {code} in {dt:.1f}s")
        if code != 0:
            print(out)
            print("MAMBA3 DET FAIL")
            sys.exit(1)
        r = subprocess.run([bin_path], cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=3600)
        runs.append(parse(r.stdout))

    for tag in TAGS:
        a, b = runs[0].get(tag), runs[1].get(tag)
        if a is None or b is None:
            print(f"FAIL {tag}: not printed by both builds")
            ok = False
            continue
        same = np.array_equal(a, b)
        sumsq = float((a.astype(np.float64) ** 2).sum())
        sumsq_b = float((b.astype(np.float64) ** 2).sum())
        print(f"{'ok  ' if same else 'FAIL'} {tag:7s} identical={same} "
              f"sumsq A={sumsq:.9f} B={sumsq_b:.9f}")
        ok = ok and same

    if not ok:
        print("MAMBA3 DET FAIL: the two builds differ; see the ledger")
        sys.exit(1)
    print("MAMBA3 DET PASS")


if __name__ == "__main__":
    main()
