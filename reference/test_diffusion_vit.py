"""The ViT denoiser's gates (milestone M2).

  reference/.venv/bin/python reference/test_diffusion_vit.py

Four gates, each one a measurement rather than a claim:

  first loss   the twin's step-0 loss against the number the Bend run
               printed at step 0 (`results/diffusion-vit/run.json`) — the
               gate that catches a staging or weight-order error on either
               side (the moons bug, NOTES 2026-10-27, was exactly this)
  forward      Bend's frozen `eps` against the twin's, same weights
  step+ddim    one ancestral step fed Bend's own eps, then the 16-step
               DDIM cloud — the sampler's arithmetic and the token order
  eta = 0      the sampler's noise must not move the cloud at all: with
               eta = 0 the sigma coefficient is zero (`sigma_of`)
  legibility   the trained clouds' mean nearest-digit |correlation| must be
               at least 0.75 (torch's ViT reaches 0.774 causal, 0.809
               bidirectional; the MLP 0.408; random images 0.326)

The frozen probe is built once and cached by mtime in /tmp; the full build
is about six minutes on this machine, the gates themselves seconds.

Exit code 0 = every gate passed. Nothing here writes.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from test_diffusion_digits import F, load, noise_of, rev_step_eps

import twin_vit as twin
import vit_gate

RESULTS = os.path.join(ROOT, "results", "diffusion-vit", "run.json")
# measured: Bend 1.0091865 against the twin's 1.0091865, a delta of 6.3e-09 once
# the twin mirrored the example's level reversal (ISSUE-040); it was 1.27e-4
# before that, on the same window. Set from the measurement, not by feel.
TOL_FIRST = 1e-5
GATE_LEGIBILITY = 0.75
_DATA = load()
DATA_MEAN, DATA_STD = float(_DATA.mean()), float(_DATA.std())


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main():
    ok = True

    # --- the recorded run, and the twin's first loss -------------------------
    if not os.path.exists(RESULTS):
        print(f"FAIL first loss: {os.path.relpath(RESULTS, ROOT)} is missing — "
              f"run the example and `reference/vit_results.py` first")
        ok = False
        run = None
    else:
        run = json.load(open(RESULTS))
        # the run's step 0 uses the per-image levels (ISSUE-016), so the twin's
        # mirror of *that* window is the comparison, not the old single-t one
        twin_first = twin.loss_at(0)
        ok &= check("first loss", abs(twin_first - run["first_loss"]) <= TOL_FIRST,
                    f"twin {twin_first:.7f}  bend {run['first_loss']:.7f}  "
                    f"delta {abs(twin_first - run['first_loss']):.2e}  tol {TOL_FIRST:g}")

    # --- the frozen forward and the sampler ---------------------------------
    bend = vit_gate.run_probe()
    p = twin.bend_init()
    twin_eps = twin.frozen_eps()
    start = twin.patch(noise_of(2000000, 4096).reshape(64, 64)).reshape(-1).astype(F)
    twin_step = rev_step_eps(start, 64, bend["eps"])         # shared eps
    twin_ddim = twin.ddim_cloud(p)

    d = vit_gate.maxabs(bend["eps"], twin_eps)
    ok &= check("forward", d <= vit_gate.TOL_EPS,
                f"eps maxabs {d:.3g}  tol {vit_gate.TOL_EPS:g}")
    d = vit_gate.maxabs(bend["step"], twin_step)
    ok &= check("ancestral step", d <= vit_gate.TOL_STEP,
                f"maxabs {d:.3g}  tol {vit_gate.TOL_STEP:g}")
    d = vit_gate.maxabs(bend["ddim"], twin_ddim)
    ok &= check("ddim eta=0 cloud", d <= vit_gate.TOL_DDIM,
                f"maxabs {d:.3g}  tol {vit_gate.TOL_DDIM:g}")

    # --- eta = 0: the sampler's own noise cannot matter ----------------------
    z = noise_of(7000000, 4096).reshape(64, 64).astype(F)
    d = vit_gate.maxabs(twin.ddim_cloud(p, z=z), twin_ddim)
    ok &= check("eta=0 invariance", d == 0.0,
                f"a random sampler draw moves the cloud by {d:.3g}")

    # --- legibility, the trained model's number -----------------------------
    if run is None:
        print("FAIL legibility: no recorded run")
        print("\nFAIL: the ViT's gates")
        return 1
    m = float(run["legibility"]["mean"])
    ok &= check("legibility", m >= GATE_LEGIBILITY,
                f"mean |corr| {m:.3f}  gate {GATE_LEGIBILITY:.2f}  "
                f"(min {run['legibility']['min']:.3f}, max {run['legibility']['max']:.3f})")
    c = run["cloud"]
    # `in_unit` counts the exact range; float32 puts a handful of pixels a few
    # 1e-8 outside it, which is the clamp's rounding and not a cloud that left
    # [0,1]. The gate checks the tolerance and reports the exact count.
    v = np.array(run["per_sampler"]["ddim"]["samples"], dtype=F)
    inside_eps = int(((v >= -1e-6) & (v <= 1 + 1e-6)).sum())
    ok &= check("cloud", (inside_eps == c["n"] and abs(c["mean"] - DATA_MEAN) < 0.1
                          and abs(c["std"] - DATA_STD) < 0.1),
                f"mean {c['mean']:+.3f} (data {DATA_MEAN:.3f})  std {c['std']:.3f} "
                f"(data {DATA_STD:.3f})  in [0,1] exactly {c['in_unit']}/{c['n']}, "
                f"within 1e-6 {inside_eps}/{c['n']}")

    print(f"\n{'PASS' if ok else 'FAIL'}: the ViT's gates")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
