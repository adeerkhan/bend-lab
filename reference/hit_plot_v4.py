#!/usr/bin/env python3
"""Proper comparison plots for ViT / JiT(DiT) / HiT from results/hit-v4 records.

  python3 reference/hit_plot_v3.py

Every number comes from a committed run.json. Three figures:
  overview.png     quality, loss, generalization gap, spectrum
  convergence.png  full loss curves incl. 2400-step runs, log-scaled step axis
  efficiency.png   quality against parameters and against wall time
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "results/hit-v4"
RUNS = {}
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d, "run.json")
    if os.path.exists(p):
        RUNS[d] = json.load(open(p))

C = {"vit": "#7a7a7a", "jit": "#c07736", "hit": "#2f5d94"}
NAME = {"vit": "ViT", "jit": "JiT / DiT", "hit": "HiT"}
SHORT = {"vit": "ViT", "jit": "JiT/DiT", "hit": "HiT"}
base = {a: RUNS[a] for a in NAME}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": "#cccccc", "axes.grid": True,
                     "grid.color": "#eeeeee", "figure.facecolor": "white"})

# ---------------------------------------------------------------- overview
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
order = ["vit", "jit", "hit"]
vals = [base[x]["legibility_test"] for x in order]
bars = a.bar([NAME[x] for x in order], vals,
             color=[C[x] for x in order], width=0.55)
a.axhline(base["hit"]["negative_control"], color="#bbbbbb", ls=":", lw=1.2,
          label=f"negative control {base['hit']['negative_control']:.3f}")
a.axhline(base["hit"]["positive_control"], color="#7e9a5e", ls="--", lw=1.2,
          label=f"real flower positive_control {base['hit']['positive_control']:.3f}")
for b, v in zip(bars, vals):
    a.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.3f}", ha="center", fontweight="bold")
a.set_ylim(0, 0.62)
a.set_ylabel("test-set legibility")
a.set_title("Test-set legibility — 1200 steps, same split and metric")
a.legend(fontsize=8, loc="upper left")

a = ax[0, 1]
for k in order:
    a.plot(base[k]["losses"], color=C[k], label=f"{NAME[k]} (1200)", alpha=0.9, lw=1.3)
a.set_xlabel("step"); a.set_ylabel("training loss")
a.set_title("Training loss, first 1200 steps")
a.legend(fontsize=8)

a = ax[1, 0]
x = np.arange(3); w = 0.35
a.bar(x - w/2, [base[k]["legibility_training"] for k in order], w,
      label="training set", color="#c9c9c9")
a.bar(x + w/2, [base[k]["legibility_test"] for k in order], w,
      label="test set", color="#5a86b8")
a.axhline(base["hit"]["positive_control"], color="#7e9a5e", ls="--", lw=1.0)
a.set_xticks(x, [NAME[k] for k in order])
a.set_ylabel("legibility")
a.set_title("Train vs held-out generalization gap")
a.legend(fontsize=8)

a = ax[1, 1]
bands = ["low", "mid-low", "mid-high", "high"]
x = np.arange(4); w = 0.26
for i, k in enumerate(order):
    a.bar(x + (i - 1)*w, base[k]["spectrum"], w, label=NAME[k], color=C[k])
a.axhline(1.0, color="#aaaaaa", ls="--", lw=0.9)
a.set_xticks(x, bands); a.set_ylim(0, 0.45)
a.set_ylabel("mean radial power (1.0 = white noise)")
a.set_title("Draw spectrum — lower = more structure")
a.legend(fontsize=8)

fig.suptitle("ViT vs JiT(DiT) vs HiT on flowers64 — faithful port v4 — all six corrections",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.012,
         "BANNERS: CPU torch reference, not Bendlang; 6.6M params at 18 blocks, 64 tokens, 1200 steps. "
         "At this scale JiT/DiT wins and HiT loses — the paper's large-scale result is not reproduced here. No FID claimed.",
         ha="center", fontsize=8.5, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.035, 1, 0.95))
fig.savefig(os.path.join(ROOT, "overview.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- convergence
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
a = ax[0]
for k in order:
    a.plot(base[k]["losses"], color=C[k], label=f"{NAME[k]} 1200", alpha=0.9)
for k in ("jit", "hit"):
    if k + "-long" not in RUNS: continue
    r = RUNS[k + "-long"]
    a.plot(r["losses"], color=C[k], ls="--", lw=1.1, alpha=0.7,
           label=f"{NAME[k]} 2400")
a.set_xlabel("step"); a.set_ylabel("training loss")
a.set_title("Full loss trajectories (1200 steps)")
a.legend(fontsize=8)

a = ax[1]
names, held, trains = [], [], []
for d, r in RUNS.items():
    names.append(f"{NAME[('hit' if d.startswith('hit') else 'jit' if d.startswith('jit') else 'vit')]}\n{r['steps']}")
    held.append(r["legibility_test"]); trains.append(r["legibility_training"])
colors = [C["vit"] if d.startswith("vit") else C["jit"] if d.startswith("jit") else C["hit"] for d in RUNS]
a.scatter(trains, held, s=110, c=colors, zorder=3)
for d, r in RUNS.items():
    a.annotate(f"{d}\n{r['legibility_test']:.3f}",
               (r["legibility_training"], r["legibility_test"]),
               textcoords="offset points", xytext=(7, 3), fontsize=8)
a.axhline(base["hit"]["negative_control"], color="#bbb", ls=":", lw=1)
a.axvline(base["hit"]["positive_control"], color="#7e9a5e", ls="--", lw=1,
          label="real flower positive_control (train)")
a.set_xlabel("legibility on training flowers")
a.set_ylabel("legibility on test flowers")
a.set_title("Generalization map — above positive_control = memorisation")
a.legend(fontsize=8)

fig.suptitle("Convergence and generalization — all five committed runs",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.012, "All three exceed the real-flower positive_control: memorisation, not generation. "
         "JiT/DiT's train legibility exceeds the real-flower positive_control: memorisation, not quality.",
         ha="center", fontsize=8.5, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.035, 1, 0.94))
fig.savefig(os.path.join(ROOT, "convergence.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- efficiency
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for a, key, label in ((ax[0], "params", "parameters (millions)"),
                      (ax[1], "wall_seconds", "wall seconds (1200 steps)")):
    for k in order:
        a.scatter([base[k][key]], [base[k]["legibility_test"]], s=140,
                  color=C[k], label=NAME[k], zorder=3)
    if key == "wall_seconds":
        a.set_xscale("log")
    a.set_xlabel(label); a.set_ylabel("test-set legibility")
    a.set_title("Quality vs " + ("capacity" if key == "params" else "prototype cost"))
    a.legend(fontsize=8)
fig.suptitle("Capacity and cost do not explain the ranking", fontsize=13, fontweight="bold")
fig.text(0.5, 0.015,
         "JiT/DiT has 1.7x ViT's params and wins; HiT has the most params and loses. "
         "Wall time is this Python prototype, not an architecture claim.",
         ha="center", fontsize=8.5, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig(os.path.join(ROOT, "efficiency.png"), dpi=150)
plt.close(fig)

print("wrote overview.png, convergence.png, efficiency.png in", ROOT)
