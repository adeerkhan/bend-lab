"""The trained UNet's checkpoint: a stopped run resumes bit for bit.

  reference/.venv/bin/python reference/test_unet_flowers_checkpoint.py

Three runs of `examples/diffusion/unet/train.bend`, on the same data:

  A: 40 steps, --probe          the whole run, weights printed at the end
  B: 20 steps, --save FILE      stopped halfway, state written by Train.save
  C: --resume FILE --start 20 --steps 20 --probe

A and C must end on the same weights and the same final loss, bit for bit: that
is BendTorch's `Train.save`/`Train.restore` contract, and it is what makes a
long run resumable. The can-fail is a 39-step run, whose weights must differ.

Exit 0 = the resumed run is the same run. Nothing here writes outside /tmp.
"""
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import test_unet_flowers as twin

BIN = "/tmp/mlbend-unet-flowers"
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
CKPT = "/tmp/mlbend-unet-flowers.ckpt"


def run(*args):
    out = subprocess.run([BIN, *args], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit(f"the run {' '.join(args)} failed:\n{out.stdout[-1500:]}{out.stderr[-1500:]}")
    return out.stdout


def weights(text):
    m = re.search(r"^w: (.*)$", text, re.M)
    if not m:
        raise SystemExit("no `w:` line: the run needs --probe")
    return np.array([float(v) for v in m.group(1).split()], dtype=np.float32)


def final_loss(text):
    return re.search(r"^final_loss: ([0-9.eE+-]+)$", text, re.M).group(1)


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main():
    twin.build()
    if os.path.exists(CKPT):
        os.remove(CKPT)

    a = run("--steps", "40", "--probe")
    run("--steps", "20", "--save", CKPT, "--no-sample")
    c = run("--resume", CKPT, "--start", "20", "--steps", "20", "--probe")
    d = run("--steps", "39", "--probe")

    wa, wc, wd = weights(a), weights(c), weights(d)
    ok = True
    ok &= check("checkpoint file", os.path.exists(CKPT), f"{os.path.getsize(CKPT)} bytes")
    same = wa.size == wc.size and np.array_equal(wa, wc)
    ok &= check("weights", same, f"40 steps == 20 + 20: {np.array_equal(wa, wc)}, "
                                 f"{wa.size} values")
    la, lc = final_loss(a), final_loss(c)
    ok &= check("final loss", la == lc, f"40 steps {la}  resumed {lc}")
    ok &= check("can-fail", not np.array_equal(wa, wd), "a 39-step run differs")
    print(f"\n{'PASS' if ok else 'FAIL'}: the checkpoint resumes bit for bit")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
