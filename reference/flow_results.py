"""Turn a flow-matching run into the evidence the example is judged on.

  reference/.venv/bin/python reference/flow_results.py /tmp/opencode/flow.log

Reads the example's stdout and writes

  results/flow-matching/run.json    the numbers, one object
  results/flow-matching/loss.svg    the loss curve
  results/flow-matching/scatter.svg data and samples, side by side
  results/flow-matching/REPORT.md   the report card

The example normalises the two-moons data to [-1, 1] on both axes, so the
data's box is the square and the box count needs no data file. The reference
is `reference/flow_probe.py`, the NumPy probe written before the port.
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from vit_results import loss_svg
import checkpoints
from experiment import bend_version, git_state, packages

OUT = os.path.join(ROOT, "results", "flow-matching")
STEPS = int(os.environ.get("FLOW_STEPS") or 6000)
EULER = 32


def moons_norm():
    """The example's data: the DDPM moons, mapped to [-1, 1] per axis."""
    n = 64
    k = np.arange(n, dtype=np.float64) + 0.5
    th = np.pi * k / 64.0
    upper = np.stack([np.cos(th), np.sin(th)], 1)
    lower = np.stack([1.0 - np.cos(th), 0.5 - np.sin(th)], 1)
    x = np.concatenate([upper, lower]).astype(np.float32)
    x[:, 0] = (2.0 * x[:, 0] - 1.0) / 3.0
    x[:, 1] = (2.0 * x[:, 1] - 0.5) / 1.5
    return x


def parse(text):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", text, re.M)]
    final = float(re.search(r"^final_loss: ([0-9.eE+-]+)$", text, re.M).group(1))
    pts = {}
    for k in ("data_mean", "data_std", "sample_mean", "sample_std"):
        m = re.search(rf"^{k}:\s*(.+)$", text, re.M)
        pts[k] = [float(v) for v in m.group(1).split()]
    samples = [float(v) for v in
               re.search(r"^samples: (.*)$", text, re.M).group(1).split()]
    return losses, final, pts, np.array(samples, dtype=np.float32).reshape(-1, 2)


def scatter_svg(path, data, samples):
    w, h, pad = 360, 360, 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="#fdfcf8"/>']
    def sx(x):
        return pad + (x + 1.4) / 2.8 * (w - 2 * pad)
    def sy(y):
        return h - pad - (y + 1.4) / 2.8 * (h - 2 * pad)
    for x, y in data:
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.6" fill="#8b7bd8" opacity="0.55"/>')
    for x, y in samples:
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.8" fill="#2f2f2f" opacity="0.55"/>')
    out.append(f'<text x="{pad}" y="14" font-family="sans-serif" font-size="11" '
               f'fill="#2f2f2f">data (violet) and samples (dark)</text>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        raise SystemExit("usage: flow_results.py <log>")
    text = open(path).read()
    losses, final, pts, samples = parse(text)
    curve = losses + [final]
    data = moons_norm()
    inside = int(((samples >= -1).all(1) & (samples <= 1).all(1)).sum())

    record = {
        "task": "flow-2d",
        "program": "examples/diffusion/flow_matching/main.bend",
        "steps": STEPS,
        "euler": EULER,
        "hidden": 128,
        "first_loss": curve[0],
        "final_loss": final,
        "last100": float(np.mean(curve[-100:])),
        "curve": curve,
        "data_mean": [float(v) for v in data.mean(0)],
        "data_std": [float(v) for v in data.std(0)],
        "sample_mean": pts["sample_mean"],
        "sample_std": pts["sample_std"],
        "inside_box": inside,
        "samples": samples.reshape(-1).tolist(),
        "box": [-1.0, 1.0],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "run.json"), "w") as f:
        json.dump(record, f)
        f.write("\n")
    loss_svg(os.path.join(OUT, "loss.svg"), curve)
    scatter_svg(os.path.join(OUT, "scatter.svg"), data, samples)

    card = f"""# flow-2d: rectified flow on two moons

`examples/diffusion/flow_matching/main.bend`, three linear layers with SiLU
(128 wide), trained full-batch for {STEPS} steps with AdamW at 3e-3. The data
is the DDPM example's two moons, normalised to [-1, 1] per axis. The sample is
Euler over {EULER} steps of `linspace(0, 1)`.

| | value |
|---|---|
| first loss | {curve[0]:.4f} |
| final loss | {final:.4f} |
| last 100 mean | **{(float(np.mean(curve[-100:]))):.4f}** |
| data mean / std | {data.mean(0)[0]:+.3f} / {data.mean(0)[1]:+.3f}  ·  {data.std(0)[0]:.3f} / {data.std(0)[1]:.3f} |
| sample mean / std | {pts['sample_mean'][0]:+.3f} / {pts['sample_mean'][1]:+.3f}  ·  {pts['sample_std'][0]:.3f} / {pts['sample_std'][1]:.3f} |
| inside the data's box | {inside}/{len(samples)} |

The NumPy probe (`reference/flow_probe.py`, 4096 points, 6000 steps) measured
loss 2.1253 -> 0.8662, samples mean [-0.111, 0.150] / std [0.546, 0.745]
against its data's [-0.002, -0.000] / [0.583, 0.741], and 465/512 inside the
box. This example runs the same method on the repository's own 128-point data,
so the means and the box count are compared with that data.

![loss](loss.svg)

![scatter](scatter.svg)
"""
    with open(os.path.join(OUT, "REPORT.md"), "w") as f:
        f.write(card)
    # what a resume would have to match: the model, the config, the packages
    identity = checkpoints.identity_only(
        {"layers": [3, 128, 128, 2], "act": "silu", "euler": EULER},
        {"steps": STEPS, "lr": 0.003, "batch": 128, "optimizer": "adamw"},
        packages(), bend=bend_version(), git=git_state().get("commit"))
    with open(os.path.join(OUT, "checkpoint.json"), "w") as f:
        json.dump(identity, f, indent=2)
        f.write("\n")
    print(f"identity: results/flow-matching/checkpoint.json"
          f" (model {identity['model']['structure'][:12]})")
    print(card)


if __name__ == "__main__":
    main()
