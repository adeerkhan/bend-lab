#!/usr/bin/env python3
"""Final v4 plots: 3 samples x 3 model columns, plus comparison plots.

  python3 reference/hit_plot_final.py

No ceiling lines, no spectrum panels. Draws are regenerated from IDENTICAL
starting noise for all three arms using the saved checkpoints, so the columns
are a fair visual comparison.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from flowers_report import legibility  # noqa: E402
EDGE = 64

ROOT = "results/hit-v4"
ARMS = ["vit", "jit", "hit"]
NAME = {"vit": "ViT", "jit": "JiT / DiT", "hit": "HiT"}
COL = {"vit": "#7a7a7a", "jit": "#c07736", "hit": "#2f5d94"}
runs = {a: json.load(open(f"{ROOT}/{a}/run.json")) for a in ARMS}
floor = runs["hit"]["noise_floor"]

if "--sample" in sys.argv:
    import torch
    from hit_v4 import HiT, Dense, EDGE, sample
    torch.manual_seed(20261122)
    z = torch.randn(8, EDGE, EDGE)
    out = {}
    for arm in ARMS:
        m = HiT() if arm == "hit" else Dense(adaln=(arm == "jit"))
        m.load_state_dict(torch.load(f"{ROOT}/{arm}/model.pt", weights_only=True))
        m.eval()
        out[arm] = sample(m, z).numpy()
    np.save(f"{ROOT}/draws_common.npy", np.stack([out[a] for a in ARMS]))
    print("wrote draws_common.npy")
    sys.exit(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stack = np.load(f"{ROOT}/draws_common.npy")
draws = {a: stack[i] for i, a in enumerate(ARMS)}

# ---------------------------------------------------------- sample grid 3x3
fig, axes = plt.subplots(3, 3, figsize=(8.4, 8.4))
for c, arm in enumerate(ARMS):
    for r in range(3):
        axes[r, c].imshow(draws[arm][r], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            axes[r, c].set_title(NAME[arm], fontsize=13, fontweight="bold")
        if c == 0:
            axes[r, c].set_ylabel(f"sample {r+1}", fontsize=10)
fig.suptitle("Three models, same starting noise — three samples each",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.012, "Saved v4 checkpoints, 32-step sampler, flowers64, 1200 steps. CPU torch reference.",
         ha="center", fontsize=9, color="#555555")
plt.tight_layout(rect=(0, 0.035, 1, 0.94))
fig.savefig(f"{ROOT}/samples_3x3.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- overview
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
vals = [runs[x]["legibility_held"] for x in ARMS]
bars = a.bar([NAME[x] for x in ARMS], vals, color=[COL[x] for x in ARMS], width=0.55)
a.axhline(floor, color="#aaaaaa", ls=":", lw=1.2, label=f"noise floor {floor:.3f}")
for b, v in zip(bars, vals):
    a.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontweight="bold")
a.set_ylim(0, 0.7); a.set_ylabel("held-out legibility")
a.set_title("Held-out legibility — 1200 steps, same metric"); a.legend(fontsize=8)

a = ax[0, 1]
for k in ARMS:
    a.plot(runs[k]["losses"], color=COL[k], label=NAME[k], alpha=0.9, lw=1.2)
a.set_xlabel("step"); a.set_ylabel("training loss"); a.set_title("Training loss")
a.legend(fontsize=8)

a = ax[1, 0]
x = np.arange(3); w = 0.35
a.bar(x-w/2, [runs[k]["legibility_train"] for k in ARMS], w, color="#c9c9c9", label="training set")
a.bar(x+w/2, [runs[k]["legibility_held"] for k in ARMS], w, color="#5a86b8", label="held-out set")
a.set_xticks(x, [NAME[k] for k in ARMS]); a.set_ylabel("legibility")
a.set_title("Train vs held-out generalization"); a.legend(fontsize=8)

a = ax[1, 1]
tr = np.fromfile(os.path.expanduser("~/.cache/bend-lab/flowers/flowers64-trainfull-gray.u8"), np.uint8).reshape(-1,4096)[:1024].astype(np.float32)/255
data = [legibility(draws[k].reshape(-1, 64*64), tr) for k in ARMS]
bp = a.boxplot(data, positions=[1,2,3], widths=0.55, patch_artist=True,
               medianprops=dict(color="black", lw=1.5))
for p, k in zip(bp["boxes"], ARMS):
    p.set_facecolor(COL[k]); p.set_alpha(0.55)
for i, k in enumerate(ARMS):
    a.scatter(np.random.default_rng(i).normal(i+1, .05, len(data[i])), data[i],
              s=15, color=COL[k], zorder=3, alpha=.8)
a.axhline(floor, color="#aaa", ls=":", lw=1)
a.set_xticks([1,2,3], [NAME[k] for k in ARMS]); a.set_ylabel("best |corr| to a train flower")
a.set_title("Per-sample legibility (8 draws, same noise)")

fig.suptitle("ViT vs JiT(DiT) vs HiT — v4 faithful port", fontsize=14, fontweight="bold")
fig.text(0.5, 0.012,
         "BANNERS: CPU torch reference, flowers64, 1200 steps. All three exceed a real-flower benchmark "
         "on this gameable metric; inspect the sample grid before trusting any single number.",
         ha="center", fontsize=8.5, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.035, 1, 0.95))
fig.savefig(f"{ROOT}/overview.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------- convergence
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
a = ax[0]
for k in ARMS:
    s = np.convolve(runs[k]["losses"], np.ones(50)/50, mode="valid")
    a.plot(s, color=COL[k], label=NAME[k], lw=1.5)
a.set_xlabel("step"); a.set_ylabel("smoothed loss (50-step)")
a.set_title("Training dynamics"); a.legend(fontsize=8)
a = ax[1]
for k in ARMS:
    a.scatter([runs[k]["params"]/1e6], [runs[k]["legibility_held"]],
              s=140, color=COL[k], label=NAME[k], zorder=3)
a.set_xlabel("parameters (millions)"); a.set_ylabel("held-out legibility")
a.set_title("Quality vs capacity"); a.legend(fontsize=8)
fig.suptitle("Convergence and capacity", fontsize=13, fontweight="bold")
fig.text(0.5, 0.012, "HiT's advantage is convergence per unit compute, not per-step wall time — "
         "see the paper's 2.4x cumulative-FLOP claim against JiT (not ViT).",
         ha="center", fontsize=8.5, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(f"{ROOT}/convergence.png", dpi=150)
plt.close(fig)

print("wrote samples_3x3.png, overview.png, convergence.png")
