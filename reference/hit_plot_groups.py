#!/usr/bin/env python3
"""Figure-4-style learned-group plots from a HiT reference run.

  python3 reference/hit_plot_groups.py results/hit-reference/hit

Produces PNG panels: for each flower, the clean image with group boundaries
(top) and the hard group map (bottom), with one group highlighted, as in
Figure 4 of arXiv:2611.22962. Uses the committed run's groups.json, so the
plot is reproducible without retraining.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else "results/hit-reference/hit"
g = json.load(open(os.path.join(run, "groups.json")))
imgs = np.array(g["images"])                    # (3,64,64)
hard = np.array(g["hard1"])                     # (3,64) level-1 assignments
A = np.array(g["A1"])                           # (3,64,K) soft weights
GRID = 8

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
cmap = plt.get_cmap("tab20")
for i in range(3):
    lab = hard[i].reshape(GRID, GRID)
    # highlight the group whose members are most spatially scattered
    best_g, best_span = 0, -1
    for k in range(lab.max() + 1):
        ys, xs = np.where(lab == k)
        if len(xs) == 0:
            continue
        span = (xs.max() - xs.min()) + (ys.max() - ys.min())
        if span > best_span:
            best_span, best_g = span, k
    # top: clean image with group boundary lines
    ax = axes[0, i]
    ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
    for gy in range(1, GRID):
        row = lab[gy - 1]
        row2 = lab[gy]
        for gx in range(GRID):
            if row[gx] != row2[gx]:
                ax.plot([gx - 0.5, gx + 0.5], [gy - 0.5, gy - 0.5], "w-", lw=0.7, alpha=0.8)
    for gx in range(1, GRID):
        for gy in range(GRID):
            if lab[gy, gx - 1] != lab[gy, gx]:
                ax.plot([gx - 0.5, gx - 0.5], [gy - 0.5, gy + 0.5], "w-", lw=0.7, alpha=0.8)
    ys, xs = np.where(lab == best_g)
    for y, x in zip(ys, xs):
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                                   edgecolor="gold", lw=2.0))
    ax.set_title(f"flower {i}: clean + group outlines", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    # bottom: hard group map, highlighted group in gold
    ax = axes[1, i]
    rgb = np.zeros((GRID, GRID, 3))
    for k in range(lab.max() + 1):
        rgb[lab == k] = np.array(cmap(k % 20)[:3])
    rgb[lab == best_g] = np.array([1.0, 0.84, 0.0])
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(f"hard map: {lab.max()+1} groups, gold = scattered group", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("HiT learned visual organization on flowers64 (t=0.75, reference model)",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.01,
         "Group structure emerges from denoising alone; no segmentation labels. "
         "Outlines and maps come from the run's saved hard assignments.",
         ha="center", fontsize=9, color="#555555")
plt.tight_layout(rect=(0, 0.03, 1, 0.96))
out = os.path.join(run, "group_maps.png")
plt.savefig(out, dpi=150)
print("wrote", out)
