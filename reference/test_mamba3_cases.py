"""Gate A3: the kernel's explanatory cases — T = 1, T = 2, T = 64.

  reference/.venv/bin/python reference/test_mamba3_cases.py

Checks (same tolerances as the A1 lane table):

  T = 1    out_v[0] == (D + qk[0]) * x[0]        (skip; SSM term is 0)
  T = 2    state_1 == a_0 * 0 + x_0 outer (scale_0 * k_0)  (state enters)
           out_v[1] == a_1 * q_1 . state_1 + (D + qk_1) * x_1
  T = 64   one full chunk (chunk_size = 64): DA_CS == cumsum(ADT * log2 e)
           over the block (in-chunk cumsum)
  kernel   out_v vs the twin at the A1 bound   (measured <= 4.1e-07)
  module   out vs the twin y at the A1 bound   (measured <= 6.5e-03)

Bounds are measured on 2026-11-16 on this lane; a device/version change
is a banner, never a delta.  Everything here is a subprocess of
`reference/.venv-rf/bin/python` (CUDA torch); the default gate runner
(`reference/.venv`) has no CUDA torch.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))
TOOL = os.path.join(ROOT, "reference", "mamba3_cases.py")

CLOSED_GATE = 1e-05  # measured 0.0 (the closed forms are the twin's own math)
IEEE_GATE = 1e-05    # A1 bound; measured <= 4.1e-07
DACS_GATE = 1e-05    # measured 8.5e-08; F32 cumsum association order
MODULE_GATE = 5e-02  # A1 tripwire; measured <= 6.5e-03

NUM = re.compile(r"max rel ([0-9.e+-]+)")


def run(label, args, env_extra=None):
    env = dict(os.environ)
    env.pop("TRITON_F32_DEFAULT", None)
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    p = subprocess.run([PYRF, TOOL] + args, cwd=ROOT, env=env,
                       capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    lines = [l for l in out.splitlines() if "max rel" in l]
    print(f"[{label}] exit {p.returncode} in {time.time() - t0:.1f}s", flush=True)
    if p.returncode != 0:
        print("\n".join(out.strip().splitlines()[-12:]), flush=True)
        print("MAMBA3 CASES FAIL")
        sys.exit(1)
    return lines


def main():
    ok = True

    def check(name, value, bound):
        nonlocal ok
        good = value <= bound
        print(("ok   " if good else "FAIL ") +
              f"{name:38s} max rel {value:.3e} (bound {bound:.0e})")
        ok = ok and good

    for l in run("twin", ["twin"]):
        m = NUM.search(l)
        check(l.split("max rel")[0].strip(), float(m.group(1)), CLOSED_GATE)
    for l in run("kernel ieee", ["kernel"], {"TRITON_F32_DEFAULT": "ieee"}):
        m = NUM.search(l)
        bound = DACS_GATE if "DA_CS" in l else IEEE_GATE
        check(l.split("max rel")[0].strip(), float(m.group(1)), bound)
    for l in run("module bf16", ["module"]):
        m = NUM.search(l)
        check(l.split("max rel")[0].strip(), float(m.group(1)), MODULE_GATE)

    if not ok:
        print("MAMBA3 CASES FAIL")
        sys.exit(1)
    print("MAMBA3 CASES PASS")


if __name__ == "__main__":
    main()
