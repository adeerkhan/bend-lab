"""The 2D rectified flow against PyTorch: same data, levels, init and steps.

  reference/.venv/bin/python reference/compare_flow2d.py

`examples/diffusion/flow_matching/main.bend` trains a 3-layer MLP (3 -> hidden -> hidden -> 2, SiLU) full-batch for
6,000 steps with AdamW at 3e-3 on the two moons. This trains the same network in torch with the example's own pieces:

  data    `flow_results.moons_norm()` — the example's normalised moons
  levels  the example's U32 Weyl level, `(s*2654435761 + i*2246822519) / 2^32` per point
  init    the example's `params()` pattern: `((3r + 5c + seed) % 7 * 0.1 - 0.2) / sqrt(cols)`, biases 0
  noise   `noise_of(2000000, 256)`, the same starting cloud the sampler uses

so the first loss is a like-for-like check of the arithmetic, and the final loss is a like-for-like check of the
optimizer. It compares with `results/flow-matching/run.json` and prints the deltas; exit 0 = within tolerance.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from flow_results import moons_norm
from test_diffusion_digits import noise_of

F = torch.float32
STEPS, EULER, LR = 6000, 32, 3e-3
TOL_FIRST, TOL_FINAL = 1e-3, 0.02          # the first loss is the arithmetic; the final is the optimizer


def hidden():
    """The example's width, read from its source so a change there moves the comparison."""
    src = open(os.path.join(ROOT, "examples/diffusion/flow_matching/main.bend")).read()
    import re
    m = re.search(r"def hidden\(\) -> Nat:\n  (\d+)n", src)
    return int(m.group(1))


def weights(h):
    """The example's deterministic init, part for part (w, b per layer)."""
    def mat(rows, cols, seed):
        r = torch.arange(rows).reshape(-1, 1).float()
        c = torch.arange(cols).reshape(1, -1).float()
        v = (((3 * r + 5 * c + seed) % 7) * 0.1 - 0.2) / np.sqrt(cols)
        return v.to(F)
    return [mat(h, 3, 1), torch.zeros(1, h), mat(h, h, 2), torch.zeros(1, h), mat(2, h, 3), torch.zeros(1, 2)]


def levels(s, n):
    a, c = 2654435761, 2246822519
    j = torch.arange(n, dtype=torch.int64)
    mixed = (s * a + j * c) & 0xFFFFFFFF
    return (mixed.double() / 2.0 ** 32).to(F)


def main():
    h = hidden()
    data = torch.tensor(moons_norm().reshape(-1, 2), dtype=F)
    n = data.shape[0]
    nz = torch.tensor(np.asarray(noise_of(2000000, 2 * n), dtype=np.float32).reshape(n, 2), dtype=F)
    p = [t.clone().requires_grad_(True) for t in weights(h)]
    opt = torch.optim.AdamW(p, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    curve = []
    for s in range(STEPS):
        t = levels(s, n).reshape(-1, 1)
        xt = (nz * (1 - t) + data * t)
        inp = torch.cat([xt, t], 1)
        h1 = Fn.silu(inp @ p[0].T + p[1])
        h2 = Fn.silu(h1 @ p[2].T + p[3])
        out = h2 @ p[4].T + p[5]
        loss = ((out - (data - nz)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 25 == 0:
            curve.append(float(loss.detach()))
    final = float(loss.detach())

    # 32 Euler steps from the same cloud, the example's sampler
    x = nz.clone()
    with torch.no_grad():
        for i in range(EULER):
            t = torch.full((n, 1), i / EULER, dtype=F)
            inp = torch.cat([x, t], 1)
            h1 = Fn.silu(inp @ p[0].T + p[1])
            h2 = Fn.silu(h1 @ p[2].T + p[3])
            x = x + (1.0 / EULER) * (h2 @ p[4].T + p[5])
    pts = x.numpy()

    rec = json.load(open(os.path.join(ROOT, "results/flow-matching/run.json")))
    rows = [
        ("first loss", rec["first_loss"], curve[0], TOL_FIRST),
        ("final loss", rec["final_loss"], final, TOL_FINAL),
        ("last 100", rec["last100"], float(np.mean(curve[-100:])), TOL_FINAL),
        ("sample mean x", rec["sample_mean"][0], float(pts[:, 0].mean()), 0.05),
        ("sample mean y", rec["sample_mean"][1], float(pts[:, 1].mean()), 0.05),
    ]
    ok = True
    for name, bend, torch_v, tol in rows:
        d = abs(bend - torch_v); good = d <= tol; ok = ok and good
        print(f"{'ok  ' if good else 'FAIL'} {name}: bend {bend:.6f}  torch {torch_v:.6f}  delta {d:.2e}  tol {tol:g}")
    out = os.path.join(ROOT, "results/flow-matching/compare-torch.json")
    json.dump({"hidden": h, "steps": STEPS, "lr": LR, "first_loss": curve[0], "final_loss": final,
               "last100": float(np.mean(curve[-100:])), "curve": curve,
               "sample_mean": [float(pts[:, 0].mean()), float(pts[:, 1].mean())],
               "sample_std": [float(pts[:, 0].std()), float(pts[:, 1].std())]},
              open(out, "w"), indent=2)
    print(f"wrote {os.path.relpath(out, ROOT)}")
    print("PASS: the flow matches torch" if ok else "FAIL: the flow against torch")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
