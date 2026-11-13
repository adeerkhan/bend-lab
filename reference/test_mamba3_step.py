"""Gate: the Mamba-3 SISO decode step produces the full-prefix tokens.

  reference/.venv/bin/python reference/test_mamba3_step.py

Builds examples/language/mamba3/step_mamba3.bend (the decode step: one row
processed at a time, state carried across tokens, and the trapezoid's
next-row DT/trap applied as an additive rank-one correction of the previous
row), runs it once, and checks:

  step_tokens == reference/mamba3_golden.npz["gen_tokens"]  (exactly, 16 tokens)

The full-prefix generator is the same computation recomputed from scratch
every step; `make mamba3` gates that probe against the same golden. This
gate exists because the step was previously reported PASS while being an
identical copy of the probe, so the step must be a real decode path and
must match the golden exactly. Missing output or a mismatch is FAIL, never
a zero or an estimate.
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
BIN = os.path.join(ROOT, ".scratch/mamba3_step.bin")


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
                    os.path.join(ROOT, "examples/language/mamba3/step_mamba3.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True,
                   timeout=1800)
    build = time.monotonic() - t0
    t0 = time.monotonic()
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    seconds = time.monotonic() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("step binary failed")
    tags = parse(p.stdout)
    tok = tags.get("step_tokens")
    if tok is None:
        raise SystemExit("missing step_tokens in output")
    got = tok.astype(int).reshape(-1)

    golden = np.load(GOLDEN)
    want = golden["gen_tokens"].astype(int).reshape(-1)
    note = ""
    if got.size != want.size:
        ok = False
    else:
        ok = np.array_equal(got, want)
        if not ok:
            i = int(np.argmax(got != want))
            note = f"first differing position {i}: bend {got[i]} golden {want[i]}"

    print(f"{'ok  ' if ok else 'FAIL'} step_tokens: {got.tolist()}")
    print(f"     golden gen_tokens: {want.tolist()}")
    if note:
        print(f"     {note}")
    print(f"     build {build:.1f} s, run {seconds:.2f} s wall")
    print("MAMBA3 STEP", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
