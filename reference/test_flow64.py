"""The 64x64 flow's gates: `reference/probe_flow64.bend` against its NumPy twin.

  reference/.venv/bin/python reference/test_flow64.py

The flow was the one task whose gate was a measured expectation rather than a
second implementation: `flowers-flow` checks the legibility a 6,000-step run
reaches, which says the run reproduced a number, not that the forward and the
sampler are the same computation. This file closes that gap with five gates,
each one elementwise against `reference/twin_flow64.py` on the same weights
(the probe's own `Layer.dump`, so neither side's initialisation is trusted):

  params   413,440, the structure's count, on both sides
  input    the first two tokens' features: the pixel window, then position and
           then time, so a reordered token builder fails here before the model
  forward  one forward at t = 0.5 on the fixed noise
  step     one Euler step at t = 0 (`step_vs`, dt = 1/32)
  euler    the states after 8, 16 and 32 steps — the 8- and 16-step runs are
           prefixes of the 32-step one, because `step_vs` divides the level by
           the preset's EULER (32), not by the number of steps it is given

The probe is built once and cached by mtime in /tmp; the build is the example's
own model, so it is the compile the preset already pays for. Exit code 0 = every
gate passed. Nothing here writes.
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from test_diffusion_digits import BEND, ENV, F

import twin_flow64 as twin

PROBE_BIN = "/tmp/mlbend-flow64-frozen"
PROBE_SRC = ("reference/probe_flow64.bend", "examples/diffusion/flowers_flow/main.bend")

# Measured first, then fixed from the printout: `input` 1.19e-07, `forward`
# 2.68e-07, one step 3.58e-07, the Euler endpoints 3.58e-07 / 4.77e-07 /
# 4.77e-07. That is float32 summation order over 128 terms and `F32.show`'s
# round trip, not a different computation, so each gate sits ~20x above its
# measurement rather than at a round decimal.
TOL_FWD = 1e-5
TOL_STEP = 1e-5
TOL_EULER = 1e-5


def probe_binary():
    """The probe's binary, rebuilt only when its sources are newer."""
    srcs = [os.path.join(ROOT, p) for p in PROBE_SRC]
    if os.path.exists(PROBE_BIN) and all(os.path.getmtime(PROBE_BIN) >= os.path.getmtime(p)
                                         for p in srcs):
        return PROBE_BIN
    build = subprocess.run([BEND, "-o", PROBE_BIN, os.path.join(ROOT, "reference/probe_flow64.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the frozen probe did not build:\n"
                         + build.stdout[-2000:] + build.stderr[-2000:])
    return PROBE_BIN


def run_probe():
    """The probe's lines, the weight dump reassembled in part order."""
    binary = probe_binary()
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the frozen probe did not run:\n"
                         + out.stdout[-2000:] + out.stderr[-2000:])
    lines, parts = {}, {}
    for line in out.stdout.strip().split("\n"):
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        if tag.startswith("w") and tag[1:].isdigit():
            parts[int(tag[1:])] = np.array([float(v) for v in rest.split()], dtype=F)
        else:
            lines[tag] = np.array([float(v) for v in rest.split()], dtype=F)
    if sorted(parts) != list(range(len(parts))):
        raise SystemExit(f"the weight dump has parts {sorted(parts)}")
    lines["weights"] = np.concatenate([parts[i] for i in range(len(parts))])
    return lines


def maxabs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main():
    ok = True
    bend = run_probe()
    x0 = twin.noise()
    tokens = twin.tokens_of(F(0.5), x0)

    p = twin.from_flat(bend["weights"])
    params = int(bend["params"][0])
    ok &= check("params", params == twin.count(),
                f"probe {params}  twin {twin.count()}")

    d = maxabs(bend["inp"], tokens[:2].reshape(-1))
    ok &= check("input", d <= TOL_FWD,
                f"first two tokens, maxabs {d:.3g}  tol {TOL_FWD:g}")

    d = maxabs(bend["fwd"], twin.forward(p, tokens).reshape(-1))
    ok &= check("forward", d <= TOL_FWD,
                f"t = 0.5, maxabs {d:.3g}  tol {TOL_FWD:g}")

    d = maxabs(bend["step"], twin.euler(p, x0, 1))
    ok &= check("euler step", d <= TOL_STEP,
                f"one step at t = 0, maxabs {d:.3g}  tol {TOL_STEP:g}")

    ends = {}
    for k in (8, 16, 32):
        ends[k] = twin.euler(p, x0, k)
        d = maxabs(bend[f"euler{k}"], ends[k])
        ok &= check(f"euler {k}", d <= TOL_EULER,
                    f"maxabs {d:.3g}  tol {TOL_EULER:g}")

    # the 8- and 16-step runs are prefixes: their distance to the 32-step state
    # has to shrink, which is what makes the level schedule testable at all
    d8 = maxabs(ends[8], ends[32])
    d16 = maxabs(ends[16], ends[32])
    ok &= check("euler prefixes", d8 > d16 > 0.0,
                f"dist to 32: 8 steps {d8:.3g}  16 steps {d16:.3g}")

    print(f"\n{'PASS' if ok else 'FAIL'}: the 64x64 flow's gates")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
