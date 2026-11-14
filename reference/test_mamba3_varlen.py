"""Gate: the packed varlen input resets the state at the segment boundary.

  reference/.venv/bin/python reference/test_mamba3_varlen.py

Builds examples/language/mamba3/probe_mamba3_varlen.bend, runs it once and
checks that a packed [u ; u] input with the boundary after the first eight
rows reproduces the twin-gated 8-row golden in BOTH segments:

  varlen_y_a      == golden["y"]        (1e-5 relative)
  varlen_y_b      == golden["y"]
  varlen_angle_a  == golden["angle"]    (1e-5 relative)
  varlen_angle_b  == golden["angle"]
  varlen_states_a == golden["states"]   (1e-5 relative)
  varlen_states_b == golden["states"]

`make mamba3` gates the plain forward against the same golden; this gate
proves the boundary reset (angle state, SSM array and the trapezoid's
shifted term) does not leak between segments. A missing tag or a mismatch
is FAIL, never a zero or an estimate.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "reference/mamba3_golden.npz")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mamba3_varlen.bin")
GATE = 1e-5
PAIRS = [("varlen_y_a", "y"), ("varlen_y_b", "y"),
         ("varlen_angle_a", "angle"), ("varlen_angle_b", "angle"),
         ("varlen_states_a", "states"), ("varlen_states_b", "states")]


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


def main():
    os.makedirs(os.path.join(ROOT, ".scratch"), exist_ok=True)
    env = dict(os.environ)
    env["BEND_NO_TELEMETRY"] = "1"
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")

    t0 = time.monotonic()
    subprocess.run([BEND, "-o", BIN,
                    os.path.join(ROOT, "examples/language/mamba3/probe_mamba3_varlen.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True,
                   timeout=1800)
    build = time.monotonic() - t0
    t0 = time.monotonic()
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    seconds = time.monotonic() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("varlen binary failed")
    tags = parse(p.stdout)

    golden = np.load(GOLDEN)
    ok = True
    for tag, key in PAIRS:
        got = tags.get(tag)
        want = golden[key].astype(np.float64).reshape(-1)
        if got is None:
            print(f"FAIL {tag}: not printed")
            ok = False
            continue
        got = got.astype(np.float64)
        if got.size != want.size:
            print(f"FAIL {tag}: size {got.size} != {want.size}")
            ok = False
            continue
        d = np.abs(got - want)
        rel = float(d.max()) / max(1e-30, float(np.abs(want).max()))
        good = rel <= GATE
        print(f"{'ok  ' if good else 'FAIL'} {tag}: max|d| {d.max():.6e} "
              f"rel {rel:.6e} gate {GATE:.0e}")
        ok = ok and good

    print(f"     build {build:.1f} s, run {seconds:.2f} s wall")
    print("MAMBA3 VARLEN", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
