#!/usr/bin/env python3
"""One-page post figure: legibility, capacity, training loss, training data.

  python3 reference/hit_plot_onepager.py

Reads the four committed run.json files. No positive_control lines, no spectrum panels.
Training data size and setup are shown as the fourth panel.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [
    ("ViT", "results/hit-v4/vit/run.json", "#7a7a7a"),
    ("JiT / DiT", "results/hit-v4/jit/run.json", "#c07736"),
    ("HiT", "results/hit-v4/hit/run.json", "#2f5d94"),
    ("Dimba hybrid", "results/dimba/hybrid/run.json", "#7e9a5e"),
]
R = [(n, json.load(open(p)), c) for n, p, c in ARMS]
names = [n for n, _, _ in R]
cols = [c for _, _, c in R]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# 1. legibility
a = ax[0, 0]
vals = [r["legibility_test"] for _, r, _ in R]
bars = a.bar(names, vals, color=cols, width=0.58)
for b, v in zip(bars, vals):
    a.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.3f}", ha="center", fontweight="bold")
a.set_ylim(0, 0.72)
a.set_ylabel("test-set legibility")
a.set_title("Quality — same test flowers")

# 2. capacity
a = ax[0, 1]
vals = [r["params"]/1e6 for _, r, _ in R]
bars = a.bar(names, vals, color=cols, width=0.58)
for b, v in zip(bars, vals):
    a.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.2f}M", ha="center", fontsize=9)
a.set_ylim(0, 9)
a.set_ylabel("parameters (millions)")
a.set_title("Capacity")

# 3. training loss
a = ax[1, 0]
for n, r, c in R:
    a.plot(r["losses"], color=c, label=n, lw=1.1, alpha=0.9)
a.set_xlabel("step"); a.set_ylabel("training loss")
a.set_title("Training loss — 1,200 steps")
a.legend(fontsize=8, loc="upper right")

# 4. training data / setup as a table
a = ax[1, 1]
a.axis("off")
a.set_title("Training set and setup")
cells = [
    ["Training flowers", "2,048"],
    ["Test flowers", "102"],
    ["Image", "64x64 grayscale"],
    ["Patch / tokens", "8x8 / 64"],
    ["Steps x batch", "1,200 x 16"],
    ["Samples seen", "19,200"],
    ["Seed", "0"],
    ["Wall (Hybrid)", "1,705 s vs ~500 s"],
]
a.set_xlim(0, 1); a.set_ylim(0, 1)
for i, (k, v) in enumerate(cells):
    y = 0.93 - i * 0.115
    if i % 2 == 0:
        a.axhspan(y - 0.06, y + 0.055, color="#f5f5f5", zorder=0)
    a.text(0.03, y, k, fontsize=10.5, va="center", ha="left")
    a.text(0.97, y, v, fontsize=10.5, va="center", ha="right", fontweight="bold")

fig.suptitle("ViT vs JiT(DiT) vs HiT vs Dimba-style hybrid on flowers64",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.012,
         "CPU torch reference models in bend-lab, not Bendlang ports. One seed, 1,200 steps; "
         "legibility favours smooth images, so read it as a relative ranking, not image quality.",
         ha="center", fontsize=8.5, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.04, 1, 0.94))
out = "results/dimba/one_page.png"
fig.savefig(out, dpi=150)
print("wrote", out)
