"""Gate: Bend step decoder/equivalent reproduces the twin's generated tokens.

  reference/.venv/bin/python reference/test_mamba3_step.py

Compares tokens emitted by examples/language/mamba3/step_mamba3.bend
against the twin's full-prefix generation goldens, \u00070..15.

NOTE: in the current Bend implementation, the file is the same full-prefix
generator as probe_mamba3_gen.bend / M.gen. A decode step that carries
SSM/angle/K/V and emits one token at a time is the step 1 goal for the
next batch; it is pending. This gate therefore confirms current parity and
will become stricter once step is implemented.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mamba3_step_gen.bin")

EXPECTED = [0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def main():
    os.makedirs(os.path.join(ROOT, ".scratch"), exist_ok=True)
    env = dict(os.environ)
    env["BEND_NO_TELEMETRY"] = "1"
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")

    subprocess.run([BEND, "-o", BIN,
                    os.path.join(ROOT, "examples/language/mamba3/step_mamba3.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True,
                   timeout=1800)
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("step binary failed")

    toks = [int(x) for x in p.stdout.strip().split(":")[-1].split()]
    ok = toks == EXPECTED
    print(f"{'ok  ' if ok else 'FAIL'} step_tokens: {toks}")
    print(f"{'ok  ' if ok else 'FAIL'} expected:    {EXPECTED}")
    print("MAMBA3 STEP", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
