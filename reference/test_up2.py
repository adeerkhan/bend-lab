"""The UNet's upsample layer against NumPy, and its adjoint against itself.

  reference/.venv/bin/python reference/test_up2.py

`examples/diffusion/unet/main.bend` holds `Up2`, the parameterless 2x
nearest-neighbour upsample a UNet's decoder needs, because BendTorch has Conv2d
but no transposed convolution and no channel concatenation. Two checks:

  forward   a 2x2 image of 2 channels (cells 1..8) upsampled to 4x4 must equal
            NumPy's repeat along both axes, which is what nearest-neighbour is
  adjoint   `<Up2(x), y> == <x, down(y)>` for the layer's own backward: down is
            the sum of the four cells that came from one input cell, so on
            x = up(x) both sides must read 816. The probe prints PASS when they
            agree exactly.

The layer is small and exact, so this is a tight gate rather than a tolerance:
any drift means the upsample and its backward are no longer adjoint, which would
make the gradient of every decoder block wrong.

Exit code 0 = every check passed. Nothing here writes.
"""
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

PROGRAM = os.path.join(ROOT, "examples/diffusion/unet/main.bend")
BIN = "/tmp/mlbend-up2"


def build():
    if os.path.exists(BIN) and os.path.getmtime(BIN) >= os.path.getmtime(PROGRAM):
        return BIN
    out = subprocess.run([BEND, "-o", BIN, PROGRAM], capture_output=True, text=True,
                         env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the up2 probe did not build:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    return BIN


def run():
    out = subprocess.run([build()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the up2 probe did not run:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    return out.stdout


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def expected_up():
    """cells 1..8 as a 2x2 image of 2 channels, nearest-neighbour to 4x4."""
    x = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    up = np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)
    return up.reshape(-1)


def main():
    text = run()
    ok = True

    up_line = re.search(r"^up: (.*)$", text, re.M)
    got = np.array([float(v) for v in up_line.group(1).split()], dtype=np.float32) if up_line else np.zeros(0)
    want = expected_up()
    ok &= check("forward", got.shape == want.shape and np.array_equal(got, want),
                f"{got.size} cells, equal to NumPy's repeat {np.array_equal(got, want)}"
                + (f" (first [{' '.join(str(v) for v in got[:6])}])" if got.size else " (no `up:` line)"))

    adjoint = re.search(r"^adjoint: (PASS|FAIL)$", text, re.M)
    ok &= check("adjoint", bool(adjoint) and adjoint.group(1) == "PASS",
                "the layer's own numbers: " + (adjoint.group(0) if adjoint else "no adjoint line"))

    # the two sides of the identity, read out of the same run
    left = re.search(r"^left:\s+([0-9.eE+-]+)$", text, re.M)
    right = re.search(r"^right:\s+([0-9.eE+-]+)$", text, re.M)
    same = bool(left and right) and float(left.group(1)) == float(right.group(1))
    ok &= check("identity", same,
                f"<up(x), up(x)> = <x, down(up(x))>: "
                f"{left.group(1) if left else '?'} = {right.group(1) if right else '?'}")

    # a gate that cannot fail is not a gate: one cell moved must be rejected
    broken = want.copy()
    broken[1] += np.float32(1.0)
    ok &= check("can-fail", not (got.shape == broken.shape and np.array_equal(got, broken)),
                "a wrong upsample (one cell off) is not equal to the probe's output")

    print(f"\n{'PASS' if ok else 'FAIL'}: the UNet's upsample layer")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
