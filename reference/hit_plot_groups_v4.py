#!/usr/bin/env python3
"""Figure-4-style learned group boundaries for the v3 HiT run."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else "results/hit-v4/hit"
g = json.load(open(os.path.join(run, "groups4.json")))
imgs = np.array(g["images"])
hards = np.array(g["hard1"])
GRID = 8
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
cmap = plt.get_cmap("tab20")

for i in range(3):
    lab = hards[i].reshape(GRID, GRID)
    sizes = [(k, int((lab == k).sum())) for k in range(int(lab.max()) + 1)]
    # three cases as in the paper: object part, background, nonlocal
    if i == 0:
        cand = [k for k, n in sizes if 3 <= n <= 9]
        best = cand[len(cand)//2] if cand else max(sizes, key=lambda z: z[1])[0]
    elif i == 1:
        best = max(sizes, key=lambda z: z[1])[0]
    else:
        largest = max(sizes, key=lambda z: z[1])[0]
        scat = [(k, (np.where(lab == k)[1].ptp() if hasattr(np.where(lab == k)[1], 'ptp') else np.ptp(np.where(lab == k)[1]))
                   + np.ptp(np.where(lab == k)[0])) for k, n in sizes if k != largest and n >= 2]
        best = max(scat, key=lambda z: z[1])[0] if scat else largest

    ax = axes[0, i]
    ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for gy in range(1, GRID):
        for gx in range(GRID):
            if lab[gy-1, gx] != lab[gy, gx]:
                ax.plot([gx*8, gx*8+8], [gy*8-0.5, gy*8-0.5], "w-", lw=0.9, alpha=0.85)
    for gx in range(1, GRID):
        for gy in range(GRID):
            if lab[gy, gx-1] != lab[gy, gx]:
                ax.plot([gx*8-0.5, gx*8-0.5], [gy*8, gy*8+8], "w-", lw=0.9, alpha=0.85)
    ys, xs = np.where(lab == best)
    for y, x in zip(ys, xs):
        ax.add_patch(plt.Rectangle((x*8, y*8), 8, 8, fill=False, edgecolor="#f5b700", lw=2.2))
    ax.set_title(f"flower {g['indices'][i]} · clean + group outlines", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, i]
    rgb = np.zeros((GRID, GRID, 3))
    for k in range(int(lab.max()) + 1):
        rgb[lab == k] = np.array(cmap(k % 20)[:3])
    rgb[lab == best] = np.array([0.96, 0.72, 0.0])
    ax.imshow(np.repeat(np.repeat(rgb, 8, 0), 8, 1), interpolation="nearest")
    ax.set_title(f"hard map · {g['K1'][i]} groups · gold = highlighted", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("HiT v4 learned visual organization on flowers64 (groups at t = 0.75)",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.012,
         "Outlines/maps come from the trained v3 model's hard assignments at t=0.75, thresholds frozen — "
         "no segmentation labels. At 64 tokens the groups are coarse (~45 level-1 groups).",
         ha="center", fontsize=9, color="#555555")
plt.tight_layout(rect=(0, 0.04, 1, 0.94))
out = os.path.join(run, "group_maps.png")
plt.savefig(out, dpi=150)
print("wrote", out)
