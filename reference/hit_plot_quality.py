#!/usr/bin/env python3
"""Visible quality comparison: sample grids and per-sample legibility.

  python3 reference/hit_plot_quality.py

Uses the committed draws.npy from each v3 arm. Two figures:
  samples.png       16 draws per arm next to 16 real held-out flowers
  distributions.png per-sample legibility (not just the mean), with the
                    noise floor and real-flower ceiling as reference bands
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from flowers_report import legibility  # noqa: E402
from unet_flowers_eval import read_gray  # noqa: E402

ROOT = "results/hit-v3"
ARMS = ["vit", "jit", "hit"]
NAME = {"vit": "ViT", "jit": "JiT / DiT", "hit": "HiT"}
COL = {"vit": "#7a7a7a", "jit": "#c07736", "hit": "#2f5d94"}
runs = {a: json.load(open(f"{ROOT}/{a}/run.json")) for a in ARMS}
tr, ho = read_gray(os.path.expanduser("~/.cache/bend-lab/flowers/flowers64-trainfull-gray.u8")), read_gray(os.path.expanduser("~/.cache/bend-lab/flowers/flowers64-hold-gray.u8"))

# ------------------------------------------------------------------ samples
n = 16
fig, axes = plt.subplots(4, n, figsize=(18, 5.2))
rows = [("real held-out flower", ho[:n].reshape(-1, 64, 64))] + [(NAME[a], np.load(f"{ROOT}/{a}/draws.npy")[:n]) for a in ARMS]
for r, (label, imgs) in enumerate(rows):
    for c in range(n):
        ax = axes[r, c]
        ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(label, fontsize=11, fontweight="bold", labelpad=8)
fig.suptitle("What the models actually drew — 16 samples per arm vs 16 real held-out flowers",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.012,
         "Same starting-noise distribution, same 32-step sampler, 1200 training steps. "
         "CPU torch reference at flowers64 — not Bendlang.",
         ha="center", fontsize=9, color="#555555")
plt.tight_layout(rect=(0, 0.035, 1, 0.94))
fig.savefig(f"{ROOT}/samples.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------ distributions
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
a = ax[0]
data = []
for arm in ARMS:
    d = np.load(f"{ROOT}/{arm}/draws.npy").reshape(-1, 64 * 64)
    data.append(legibility(d, tr))
bp = a.boxplot(data, positions=[1, 2, 3], widths=0.55, patch_artist=True,
               medianprops=dict(color="black", lw=1.5))
for patch, arm in zip(bp["boxes"], ARMS):
    patch.set_facecolor(COL[arm]); patch.set_alpha(0.55)
for i, arm in enumerate(ARMS):
    a.scatter(np.random.default_rng(i).normal(i + 1, 0.06, len(data[i])), data[i],
              s=16, color=COL[arm], zorder=3, alpha=0.8)
a.axhline(runs["hit"]["noise_floor"], color="#aaaaaa", ls=":", lw=1.2, label="noise floor")
a.axhline(runs["hit"]["ceiling"], color="#7e9a5e", ls="--", lw=1.2, label="real flower ceiling")
a.set_xticks([1, 2, 3], [NAME[a] for a in ARMS])
a.set_ylabel("best |corr| to a training flower")
a.set_title("Per-sample legibility (32 draws each)")
a.legend(fontsize=8)

a = ax[1]
for arm in ARMS:
    losses = np.array(runs[arm]["losses"])
    # smoothed with a 50-step window so the trend is visible
    smooth = np.convolve(losses, np.ones(50)/50, mode="valid")
    a.plot(smooth, color=COL[arm], label=NAME[arm], lw=1.5)
a.set_xlabel("step"); a.set_ylabel("smoothed loss (50-step window)")
a.set_title("Training dynamics — 1200 steps")
a.legend(fontsize=8)

fig.suptitle("Quality is a distribution, not a single number", fontsize=13, fontweight="bold")
fig.text(0.5, 0.012,
         "BANNERS: CPU torch reference, flowers64, 1200 steps. The means (0.473 / 0.495 / 0.430) "
         "hide the spread; no FID or paper result is claimed.",
         ha="center", fontsize=8.5, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.045, 1, 0.94))
fig.savefig(f"{ROOT}/distributions.png", dpi=150)
plt.close(fig)
print("wrote samples.png and distributions.png")
