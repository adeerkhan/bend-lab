"""The 2D rectified flow's gates: `reference/probe_flow2d.bend` against `reference/twin_flow2d.py`.

  reference/.venv/bin/python reference/test_flow2d.py

The example was the one measured only end to end (its README said so: "the frozen probe and its NumPy twin are
still owed"), so a wrong vector field or a reordered token row had nothing to fail against. This closes it with six
elementwise gates on the probe's own `Layer.dump` (neither side's initialisation is trusted):

  input    the 128 rows of (x_t, y_t, t) at t = 0.5, rebuilt from the fixed cloud
  forward  one step at t = 0.5 — the example's `fwd_v` returns x + dt*v, not the velocity
  step     one Euler step at t = 0
  euler 8 / 16 / 32   the states after 8, 16 and 32 steps, each in one call (the level is i/32, so a restart would
           sample t = 0 again)

The probe is built once and cached by mtime in /tmp. Exit code 0 = every gate passed. Nothing here writes.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from test_diffusion_digits import BEND, ENV

import twin_flow2d as twin

PROBE_BIN = "/tmp/mlbend-flow2d-frozen"
PROBE_SRC = ("reference/probe_flow2d.bend", "examples/diffusion/flow_matching/main.bend")


def probe_binary():
    """The probe's binary, rebuilt only when its sources are newer."""
    srcs = [os.path.join(ROOT, p) for p in PROBE_SRC]
    if os.path.exists(PROBE_BIN) and all(os.path.getmtime(PROBE_BIN) >= os.path.getmtime(p) for p in srcs):
        return PROBE_BIN
    build = subprocess.run([BEND, "-o", PROBE_BIN, os.path.join(ROOT, "reference/probe_flow2d.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the frozen probe did not build:\n" + build.stdout + build.stderr)
    return PROBE_BIN


def main():
    out = subprocess.run([probe_binary()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        print(f"FAIL the probe did not run:\n{out.stdout}{out.stderr}")
        return 1
    rows = twin.compare(out.stdout)
    ok = True
    for name, good, detail in rows:
        ok = ok and good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {detail}")
    print("PASS: the 2D flow's gates" if ok else "FAIL: the 2D flow's gates")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
