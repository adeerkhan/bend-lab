#!/usr/bin/env python3
"""Judge a trained UNet run on the flowers splits, with the repo's own statistics.

  reference/.venv/bin/python reference/unet_flowers_eval.py <run stdout> [--edge 64] [--train PATH] [--hold PATH] [--tag NAME]

Reads `train.bend`'s stdout (`params:`, `loss:` lines, `final_loss:` and one
`sample: <n floats>` line per drawn image) and prints the numbers the objective
is judged on, using `reference/flowers_report.py`'s `legibility` - the same
function the `flowers-flow` task gates with, so the gate and this report cannot
disagree:

  legibility        the draws' best |corr| with any training flower
  legibility_held   the draws' best |corr| with any held-out flower
  ceiling           what a REAL held-out flower scores against the training set

The reference set must be the one the model trained on: the 3,874-image subset is
inside the 7,067-image training set, so scoring against the smaller set inflates
both `legibility` and the ceiling. The default is the full export, and the two
line counts are printed so a mismatch is visible. A trained legibility above the
ceiling means memorisation; the held-out number is then the only honest one.

plus the white-noise test the loss cannot see: the draws' radial power spectrum
against the data's (a flat spectrum is noise, whatever the loss says).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(ROOT, "reference", "flowers_report.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise SystemExit("cannot find bend-lab: no reference/flowers_report.py above this file")
    ROOT = parent
sys.path.insert(0, os.path.join(ROOT, "reference"))
from flowers_report import legibility  # noqa: E402  (one implementation)

FLOWERS = os.path.expanduser("~/.cache/bend-lab/flowers")
OLD = os.path.join(ROOT, "results", "diffusion-unet", "trained", "run.json")
EDGE = 64


def read_gray(path):
    x = np.fromfile(path, dtype=np.uint8).astype(np.float32) / 255.0
    return x.reshape(-1, EDGE * EDGE)


def parse(text):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", text, re.M)]
    final = re.search(r"^final_loss: ([0-9.eE+-]+)$", text, re.M)
    draws = []
    for line in re.findall(r"^sample: (.*)$", text, re.M):
        vals = np.array([float(v) for v in line.split()], dtype=np.float32)
        draws.extend(vals.reshape(-1, EDGE, EDGE))
    params = re.search(r"^params: (\d+)$", text, re.M)
    return (int(params.group(1)) if params else None), losses, \
        (float(final.group(1)) if final else None), np.array(draws, dtype=np.float32)


BANDS = {64: ((1, 5), (5, 11), (11, 21), (21, 33)),
         32: ((1, 3), (3, 6), (6, 11), (11, 17)),
         16: ((1, 2), (2, 3), (3, 5), (5, 9))}


def spectrum(images, bands=None):
    """mean power per Fourier coefficient, normalised: 1.0 everywhere is white."""
    n = EDGE
    bands = bands or BANDS[n]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    out = []
    for img in images:
        f = np.abs(np.fft.fft2(img - img.mean())) ** 2
        tot = f.mean()
        out.append([f[(r >= lo) & (r < hi)].mean() / tot for lo, hi in bands])
    return np.mean(out, axis=0)


def main():
    global EDGE
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--edge", type=int, default=64)
    ap.add_argument("--train")
    ap.add_argument("--hold")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    EDGE = a.edge
    train_path = a.train or os.path.join(FLOWERS, f"flowers{EDGE}-trainfull-gray.u8")
    hold_path = a.hold or os.path.join(FLOWERS, f"flowers{EDGE}-hold-gray.u8")

    text = open(a.log).read()
    params, losses, final, draws = parse(text)
    if draws.size == 0:
        raise SystemExit("no `sample:` lines: did the run reach the sampler?")

    train = read_gray(train_path)
    hold = read_gray(hold_path)
    flat = draws.reshape(len(draws), EDGE * EDGE)

    leg = legibility(flat, train)
    leg_held = legibility(flat, hold)
    ceiling = legibility(hold, train).mean()
    rng = np.random.default_rng(0)
    floor = legibility(rng.standard_normal(flat.shape).astype(np.float32), train).mean()

    print(f"== {a.tag}  ({EDGE}x{EDGE})")
    print(f"params            {params}")
    print(f"steps logged      {len(losses)}   first {losses[0] if losses else float('nan'):.4f}"
          f"   last {losses[-1] if losses else float('nan'):.4f}"
          f"   min {min(losses) if losses else float('nan'):.4f}")
    print(f"final_loss        {final if final is not None else 'unavailable'}")
    print(f"draws             {len(draws)}")
    print(f"sample mean/std   {draws.mean():+.4f} / {draws.std():.4f}"
          f"    data {train.mean():+.4f} / {train.std():.4f}")
    print(f"legibility        {leg.mean():.3f}   (min {leg.min():.3f} max {leg.max():.3f})")
    print(f"legibility_held   {leg_held.mean():.3f}")
    print(f"ceiling           {ceiling:.3f}   (a real held-out flower vs the training set)")
    print(f"reference set     {len(train)} training x {len(hold)} held out"
          f"   ({os.path.basename(train_path)})")
    print(f"floor             {floor:.3f}   (noise)")
    print(f"of the ceiling    {100.0 * leg.mean() / ceiling:.1f}% trained, "
          f"{100.0 * leg_held.mean() / ceiling:.1f}% held out")
    # the same, with the noise floor taken out: the only cross-resolution comparison
    span = ceiling - floor
    print(f"of the span       {100.0 * (leg.mean() - floor) / span:.1f}% trained, "
          f"{100.0 * (leg_held.mean() - floor) / span:.1f}% held out"
          f"   (0 = noise, 100 = a real held-out flower)")
    if leg.mean() > ceiling:
        print(f"note             trained legibility is ABOVE the ceiling ({leg.mean():.3f} > {ceiling:.3f}): "
              f"the draws memorise the training set, read the held-out number")

    sd, dd = spectrum(draws), spectrum(train.reshape(-1, EDGE, EDGE))
    print("spectrum (power per coefficient, radius 1-4 / 5-10 / 11-20 / 21-32):")
    print(f"  draws {' '.join(f'{v:.3f}' for v in sd)}")
    print(f"  data  {' '.join(f'{v:.3f}' for v in dd)}   (flat 1.0 = white noise)")

    if EDGE == 64 and os.path.exists(OLD):
        old = json.load(open(OLD))
        od = np.array(old["draws_u8"], dtype=np.float32).reshape(-1, EDGE * EDGE) / 255.0
        old_leg = legibility(od, train).mean()
        old_sd = spectrum(od.reshape(-1, EDGE, EDGE))
        print(f"previous record   legibility {old_leg:.3f}, spectrum "
              f"{' '.join(f'{v:.3f}' for v in old_sd)}")


if __name__ == "__main__":
    main()
