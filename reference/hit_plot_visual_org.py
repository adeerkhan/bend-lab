#!/usr/bin/env python3
"""Figure-4-style visual organization: L1 and L2 group boundaries + highlights.

  python3 reference/hit_plot_visual_org.py

Two-row comparison from results/hit-v4/hit/groups4.json:
  top:   level-1 hard group outlines overlaid on the clean flowers
  bottom: level-2 hard group map (the compact bottleneck), same flowers
Three flowers, one highlighted group per flower (object / background / nonlocal).
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = "results/hit-v4/hit"
g = json.load(open(os.path.join(RUN, "groups4.json")))
imgs = np.array(g["images"])
h1 = [np.array(x) for x in g["hard1"]]; h2 = [np.array(x) for x in g["hard2"]]
cmap = plt.get_cmap("tab20")
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

for i in range(3):
    for row, (lab_flat, K) in enumerate(((h1[i], g["K1"][i]), (h2[i], g["K2"][i]))):
        grid = int(np.ceil(np.sqrt(len(lab_flat))))
        size = len(lab_flat)
        lab = np.full(grid*grid, -1, dtype=np.int64); lab[:size] = lab_flat
        lab = lab.reshape(grid, grid)
        scale = max(1, 64 // grid)
        ax = axes[row, i]
        if row == 0:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            # white boundaries between different groups on the clean image
            for gy in range(1, grid):
                for gx in range(grid):
                    if lab[gy-1, gx] != lab[gy, gx]:
                        ax.plot([gx*scale, gx*scale+scale], [gy*scale-.5, gy*scale-.5],
                                "w-", lw=1.0, alpha=0.9)
            for gx in range(1, grid):
                for gy in range(grid):
                    if lab[gy, gx-1] != lab[gy, gx]:
                        ax.plot([gx*scale-.5, gx*scale-.5], [gy*scale, gy*scale+scale],
                                "w-", lw=1.0, alpha=0.9)
            sizes = [(int((lab == k).sum()), k) for k in range(int(lab.max())+1)]
            best = max(sizes)[1] if i == 1 else sorted(sizes)[len(sizes)//2][1]
            ys, xs = np.where(lab == best)
            for y, x in zip(ys, xs):
                ax.add_patch(plt.Rectangle((x*scale, y*scale), scale, scale, fill=False,
                                           edgecolor="#f5b700", lw=2.4))
            ax.set_title(f"flower {g['indices'][i]} · level 1 ({K} groups)", fontsize=10)
        else:
            rgb = np.zeros((grid, grid, 3))
            for k in range(int(lab.max())+1):
                rgb[lab == k] = np.array(cmap(k % 20)[:3])
            sizes = [(int((lab == k).sum()), k) for k in range(int(lab.max())+1)]
            best = max(sizes)[1] if i == 1 else sorted(sizes)[len(sizes)//2][1]
            rgb[lab == best] = np.array([0.96, 0.72, 0.0])
            ax.imshow(np.repeat(np.repeat(rgb, scale, 0), scale, 1), interpolation="nearest")
            ax.set_title(f"level 2 ({K} groups) · gold = highlighted", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("HiT learned visual organization — two grouping levels on flowers64 (t = 0.75)",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.012,
         "Outlines come from the trained model's own forward. No segmentation labels. "
         "Level 2 compresses to ~12 group tokens — the computational bottleneck.",
         ha="center", fontsize=9, color="#555555")
plt.tight_layout(rect=(0, 0.04, 1, 0.94))
out = os.path.join(RUN, "visual_organization.png")
plt.savefig(out, dpi=150)
print("wrote", out)
