#!/usr/bin/env python3
"""Three-arm comparison: ViT vs HiT vs JiT(DiT) from results/hit-v2."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = "results/hit-v2"
R = {a: json.load(open(os.path.join(root, a, "run.json"))) for a in ("vit", "jit", "hit")}
colors = {"ViT": "#888888", "JiT/DiT": "#c07736", "HiT": "#2f5d94"}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

# 1. test-set legibility

keys = [(None, R["hit"]["negative_control"]), ("vit", R["vit"]["legibility_test"]),
        ("jit", R["jit"]["legibility_test"]), ("hit", R["hit"]["legibility_test"])]
names = ["noise\nfloor", "ViT\n0.213", "JiT/DiT\n0.215", "HiT\n0.221"]
vals = [v for _, v in keys]
ax[0].bar(names, vals, color=["#cccccc", colors["ViT"], colors["JiT/DiT"], colors["HiT"]])
ax[0].axhline(R["hit"]["positive_control"], color="#7e9a5e", ls="--", lw=1, label=f"real flower positive_control {R['hit']['positive_control']:.3f}")
for i, v in enumerate(vals):
    ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax[0].set_ylabel("test-set legibility"); ax[0].set_ylim(0, 0.62); ax[0].legend(fontsize=8)
ax[0].set_title("Quality: HiT > JiT/DiT > ViT")

# 2. loss curves
for a in ("vit", "jit", "hit"):
    ax[1].plot(R[a]["losses"], alpha=0.85, color=colors[{"vit": "ViT", "jit": "JiT/DiT", "hit": "HiT"}[a]], label=a)
ax[1].set_xlabel("step"); ax[1].set_ylabel("training loss"); ax[1].set_title("1200 steps, batch 16, 2048 flowers"); ax[1].legend()

# 3. params and wall
x = np.arange(3)
ax[2].bar(x - 0.18, [R[a]["params"]/1e6 for a in ("vit", "jit", "hit")], 0.36, label="M params", color="#7e9a5e")
ax[2].bar(x + 0.18, [R[a]["wall_seconds"]/60 for a in ("vit", "jit", "hit")], 0.36, label="wall minutes", color="#8b83b5")
ax[2].set_xticks(x, ["ViT", "JiT/DiT", "HiT"]); ax[2].set_title("Cost (reference prototype)"); ax[2].legend()

fig.suptitle("ViT vs HiT vs DiT(JiT) on flowers64 — faithful small port, 1200 steps",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.012,
         "BANNERS: CPU torch reference, not Bendlang; small scale and 1200 steps (paper: 960M / 600 epochs). "
         "HiT thresholds retuned once mid-training (frozen early calibration was stale at this scale). "
         "At 64 tokens grouping is weak (~64->45->39); the paper's advantage is large-scale. No FID claimed.",
         ha="center", fontsize=8, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.06, 1, 0.94))
out = os.path.join(root, "comparison.png")
plt.savefig(out, dpi=150)
print("wrote", out)
