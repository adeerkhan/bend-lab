#!/usr/bin/env python3
"""Figure-4-style learned-group panels from a HiT reference run.

  python3 reference/hit_plot_groups.py results/hit-reference/hit

Two rows like arXiv:2611.22962 Figure 4: top = clean image with group
outlines and one highlighted group; bottom = the corresponding hard group
map. Groups are computed at t=0.75 (groups4.json), no segmentation labels.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else "results/hit-reference/hit"
g = json.load(open(os.path.join(run, "groups4.json")))
imgs = np.array(g["images"])                    # (3,64,64) clean
hards = np.array(g["hard1"])                    # (3,64) level-1 groups at t=0.75
K1 = g["K1"]
GRID = 8

fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
cmap = plt.get_cmap("tab20")

for i in range(3):
    lab = hards[i].reshape(GRID, GRID)           # 8x8 token grid
    # The paper shows three cases; vary the highlighted group across panels:
    # 0 = an object-sized group, 1 = the scattered background, 2 = nonlocal.
    sizes = [(k, int((lab == k).sum())) for k in range(int(lab.max()) + 1)]
    if i == 0:                       # object part: a compact mid-sized group
        cand = [k for k, n in sizes if 2 <= n <= 8]
        best_g = cand[len(cand)//2] if cand else max(sizes, key=lambda z: z[1])[0]
    elif i == 1:                     # background: largest group
        best_g = max(sizes, key=lambda z: z[1])[0]
    else:                            # nonlocal: most scattered, not the largest
        largest = max(sizes, key=lambda z: z[1])[0]
        scattered = [(k, (np.where(lab == k)[1].max() - np.where(lab == k)[1].min())
                      + (np.where(lab == k)[0].max() - np.where(lab == k)[0].min()))
                     for k, n in sizes if k != largest and n >= 2]
        best_g = max(scattered, key=lambda z: z[1])[0] if scattered else largest

    # ---- top: clean image with group outlines at pixel scale ----
    ax = axes[0, i]
    ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    # boundaries between tokens with different groups, drawn at 8-pixel scale
    for gy in range(1, GRID):
        y = gy * 8 - 0.5
        for gx in range(GRID):
            if lab[gy - 1, gx] != lab[gy, gx]:
                ax.plot([gx*8, gx*8 + 8], [y, y], "w-", lw=0.9, alpha=0.85)
    for gx in range(1, GRID):
        x = gx * 8 - 0.5
        for gy in range(GRID):
            if lab[gy, gx - 1] != lab[gy, gx]:
                ax.plot([x, x], [gy*8, gy*8 + 8], "w-", lw=0.9, alpha=0.85)
    # highlighted group: gold rectangle per token
    ys, xs = np.where(lab == best_g)
    for y, x in zip(ys, xs):
        ax.add_patch(plt.Rectangle((x*8, y*8), 8, 8, fill=False,
                                   edgecolor="#f5b700", lw=2.2))
    ax.set_title(f"flower {g['indices'][i]} · clean + group outlines", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # ---- bottom: hard group map, same colors, highlighted group in gold ----
    ax = axes[1, i]
    rgb = np.zeros((GRID, GRID, 3))
    for k in range(int(lab.max()) + 1):
        rgb[lab == k] = np.array(cmap(k % 20)[:3])
    rgb[lab == best_g] = np.array([0.96, 0.72, 0.0])
    ax.imshow(np.repeat(np.repeat(rgb, 8, 0), 8, 1), interpolation="nearest")
    ax.set_title(f"hard map · {K1[i]} groups · gold = scattered group", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("HiT learned visual organization on flowers64 (groups at t = 0.75)",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.012,
         "Outlines/maps come from the trained model's hard assignments — no segmentation labels. "
         "Gold collects spatially disconnected patches that share appearance (paper: background / nonlocal similarity).",
         ha="center", fontsize=9, color="#555555")
plt.tight_layout(rect=(0, 0.04, 1, 0.94))
out = os.path.join(run, "group_maps.png")
plt.savefig(out, dpi=150)
print("wrote", out)
