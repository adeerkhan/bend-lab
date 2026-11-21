#!/usr/bin/env python3
"""Comparison chart for the HiT vs matched-ViT flower reference runs."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = "results/hit-reference"
h = json.load(open(os.path.join(root, "hit/run.json")))
v = json.load(open(os.path.join(root, "vit/run.json")))

fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))

# 1. legibility with reference points
ax[0].bar(["noise", "ViT", "HiT", "real flower\n(ceiling)"],
          [h["noise_floor"], v["legibility_held"], h["legibility_held"], h["ceiling"]],
          color=["#cccccc", "#c07736", "#2f5d94", "#7e9a5e"])
for i, x in enumerate([h["noise_floor"], v["legibility_held"], h["legibility_held"], h["ceiling"]]):
    ax[0].text(i, x + 0.012, f"{x:.3f}", ha="center", fontsize=9)
ax[0].set_ylabel("held-out legibility"); ax[0].set_ylim(0, 0.62)
ax[0].set_title("Quality: HiT +0.010 over ViT")

# 2. sample efficiency: legibility vs wall seconds
ax[1].scatter([v["wall_seconds"]], [v["legibility_held"]], s=120, color="#c07736", label="ViT")
ax[1].scatter([h["wall_seconds"]], [h["legibility_held"]], s=120, color="#2f5d94", label="HiT")
ax[1].annotate(f"{v['wall_seconds']:.1f}s", (v['wall_seconds'], v['legibility_held']),
               textcoords="offset points", xytext=(8, -4), fontsize=9)
ax[1].annotate(f"{h['wall_seconds']:.1f}s", (h['wall_seconds'], h['legibility_held']),
               textcoords="offset points", xytext=(8, -4), fontsize=9)
ax[1].set_xlabel("wall seconds for 300 steps (log)"); ax[1].set_ylabel("held-out legibility")
ax[1].set_xscale("log"); ax[1].set_title("Prototype wall time — NOT architecture efficiency"); ax[1].legend()

# 3. loss curves
ax[2].plot(h["losses"], alpha=0.85, color="#2f5d94", label="HiT")
ax[2].plot(v["losses"], alpha=0.85, color="#c07736", label="ViT")
ax[2].set_xlabel("step"); ax[2].set_ylabel("training loss")
ax[2].set_title("Same data, loss, 300 steps"); ax[2].legend()

# 4. spectrum
bands = ["low", "mid-low", "mid-high", "high"]
x = np.arange(4); w = 0.35
ax[3].bar(x - w/2, h["spectrum"], w, label="HiT", color="#2f5d94")
ax[3].bar(x + w/2, v["spectrum"], w, label="ViT", color="#c07736")
ax[3].set_xticks(x, bands); ax[3].set_ylim(0, 1)
ax[3].axhline(1.0, color="#999999", ls="--", lw=0.8)
ax[3].set_ylabel("radial power (1.0 = white noise)"); ax[3].set_title("Draw spectrum"); ax[3].legend()

fig.suptitle("HiT vs matched ViT on flowers64 — small reference models, 300 steps",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.012,
         "BANNERS: CPU torch prototype — wall time is implementation, not architecture. "
         "HiT does 0.64x the token-blocks of ViT at matched blocks. ViT is a flowers twin; "
         "(our Bend ViT is 8x8 digits). At 300 steps neither reaches the real-flower "
         "ceiling. No paper FID / ImageNet result is claimed here.",
         ha="center", fontsize=8.5, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.05, 1, 0.94))
out = os.path.join(root, "comparison.png")
plt.savefig(out, dpi=150)
print("wrote", out)
