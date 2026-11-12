"""The UNet's held-out gate: legibility and spectrum, with can-fail arms.

  reference/.venv/bin/python reference/test_unet_flowers_metric.py

Reads the recorded run `results/diffusion-unet/trained/run.json` (64 draws on
the 64x64 flowers split), the full 7,067-image training set and the 102-image
held-out set, and applies the two measurements the training loss cannot see:

  legibility_held   best |corr| with a held-out flower, above the noise floor
  span              the fraction of the floor-to-ceiling span it reaches
  spectrum          the high-frequency band must not be white (1.0)

The recorded draws pass. White noise and the data mean are the can-fail arms:
each must fail at least one check. This is a gate on the record, not a fresh
3,600-step retrain; run `train.bend` and `unet_flowers_eval.py` when a new
record is needed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from unet_flowers_eval import FLOWERS, legibility, read_gray, spectrum  # noqa: E402

RUN = os.path.join(ROOT, "results", "diffusion-unet", "trained", "run.json")
TRAIN = os.path.join(FLOWERS, "flowers64-trainfull-gray.u8")
HOLD = os.path.join(FLOWERS, "flowers64-hold-gray.u8")
EDGE = 64

MIN_HELD = 0.40
MIN_SPAN = 0.60
MAX_ABOVE_CEILING = 0.02
MAX_HIGH_BAND = 0.85
MAX_LOW_BAND = 0.90


def metrics(draws, train, hold):
    flat = draws.reshape(len(draws), EDGE * EDGE)
    held = legibility(flat, hold).mean()
    ceiling = legibility(hold, train).mean()
    rng = np.random.default_rng(0)
    floor = legibility(rng.standard_normal(flat.shape).astype(np.float32), train).mean()
    span = (held - floor) / (ceiling - floor) if ceiling > floor else 0.0
    bands = spectrum(draws)
    return {
        "held": float(held), "ceiling": float(ceiling), "floor": float(floor),
        "span": float(span), "bands": [float(v) for v in bands],
    }


def passes(m):
    checks = [
        ("held above the floor", m["held"] >= MIN_HELD),
        ("held span", m["span"] >= MIN_SPAN),
        ("held not above the ceiling", m["held"] <= m["ceiling"] + MAX_ABOVE_CEILING),
        ("high band not white", m["bands"][3] <= MAX_HIGH_BAND),
        ("low band not white", m["bands"][0] <= MAX_LOW_BAND),
    ]
    return checks, all(ok for _, ok in checks)


def main():
    if not os.path.exists(RUN):
        raise SystemExit(f"missing {RUN}")
    for path in (TRAIN, HOLD):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: run the flowers export first")

    run = json.load(open(RUN))
    raw = run.get("draws_u8") or run.get("draws")
    draws = np.array(raw, dtype=np.float32).reshape(-1, EDGE, EDGE) / 255.0
    train = read_gray(TRAIN)
    hold = read_gray(HOLD)

    m = metrics(draws, train, hold)
    checks, ok = passes(m)
    print(f"record            {os.path.relpath(RUN, ROOT)}")
    print(f"draws             {len(draws)}")
    print(f"legibility_held   {m['held']:.3f}   (floor {m['floor']:.3f}, ceiling {m['ceiling']:.3f})")
    print(f"span              {100 * m['span']:.1f}% of the floor-to-ceiling span")
    print(f"spectrum          {' '.join(f'{v:.3f}' for v in m['bands'])}   (flat 1.0 = white)")
    for name, passed in checks:
        print(f"{'ok  ' if passed else 'FAIL'} {name}")

    # The can-fail arms: a gate that passes white noise or the mean is not a gate.
    rng = np.random.default_rng(7)
    white = rng.random(draws.shape, dtype=np.float32)
    mean = np.full_like(draws, float(train.mean()))
    for name, arm in (("white noise", white), ("data mean", mean)):
        m2 = metrics(arm, train, hold)
        _, arm_ok = passes(m2)
        print(f"{'ok  ' if not arm_ok else 'FAIL'} can-fail {name}: "
              f"held {m2['held']:.3f}, span {100 * m2['span']:.1f}%, "
              f"bands {' '.join(f'{v:.3f}' for v in m2['bands'])}")
        ok = ok and not arm_ok

    print("UNET METRIC PASS" if ok else "UNET METRIC FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
