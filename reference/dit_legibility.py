#!/usr/bin/env python3
"""Legibility of the full-width Bendlang DiT's DDIM samples.

  python3 reference/dit_legibility.py <checkpoint>

Runs examples/diffusion/dit/full_sample.bend on a checkpoint, parses the
`samples:` line (4096 pixels in image order), and scores it with the repo's
standard mean best-|corr| metric (reference/test_diffusion_digits.legibility).
Reference points on this data: noise ~0.33, dense MLP ~0.41, Bend-ViT ~0.795.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from test_diffusion_digits import legibility, load  # noqa: E402

BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
GUARD = os.path.join(ROOT, "scripts", "bend-guard.sh")
SAMPLE_BIN = os.path.join(ROOT, ".scratch", "dit_sample")


def main():
    ckpt = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "/tmp/opencode/dit3000.ckpt"
    if not os.path.exists(ckpt):
        raise SystemExit(f"dit-legibility: checkpoint missing: {ckpt}")
    env = dict(os.environ, BEND_NO_TELEMETRY="1",
               LD_LIBRARY_PATH=(os.path.expanduser("~/.local/lib/llvm22") + ":" +
                                os.environ.get("LD_LIBRARY_PATH", "")),
               BEND_LIB=os.path.join(ROOT, ".bend", "env"),
               BEND_MEM=os.environ.get("BEND_MEM", "2500M"))
    build = subprocess.run([GUARD, BEND, os.path.join(ROOT, "examples", "diffusion", "dit", "full_sample.bend"),
                            "-o", SAMPLE_BIN], env=env, capture_output=True, text=True, timeout=900)
    if build.returncode != 0:
        raise SystemExit(f"dit-legibility: sampler build failed\n{build.stdout[-500:]}{build.stderr[-500:]}")
    run = subprocess.run([SAMPLE_BIN, ckpt], capture_output=True, text=True, timeout=900)
    line = next((l for l in run.stdout.splitlines() if l.startswith("samples: ")), None)
    if line is None:
        raise SystemExit(f"dit-legibility: no samples line\n{run.stdout[-500:]}{run.stderr[-500:]}")
    pixels = np.array([float(x) for x in line[len("samples: "):].split()], dtype=np.float64)
    if pixels.size != 4096:
        raise SystemExit(f"dit-legibility: {pixels.size} pixels, expected 4096")
    score, lo, hi = legibility(pixels, load())
    print(f"dit-legibility: {ckpt}: mean {score:.4f} (min {lo:.4f}, max {hi:.4f}); "
          f"n={pixels.size}, in_unit={int(pixels.min() >= 0 and pixels.max() <= 1)}")


if __name__ == "__main__":
    main()
