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

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
# legibility vs floors
ax[0].bar(["noise", "ViT (held)", "HiT (held)", "real hold\nflower (ceiling)"],
          [h["noise_floor"], v["legibility_held"], h["legibility_held"], h["ceiling"]],
          color=["#cccccc", "#c07736", "#2f5d94", "#7e9a5e"])
ax[0].set_ylabel("legibility (best |corr| to a train flower)")
ax[0].set_title("Held-out legibility — same split, same metric")
for i, x in enumerate([h["noise_floor"], v["legibility_held"], h["legibility_held"], h["ceiling"]]):
    ax[0].text(i, x + 0.01, f"{x:.3f}", ha="center", fontsize=9)
ax[0].set_ylim(0, 0.6)

# loss curves
ax[1].plot(h["losses"], alpha=0.8, color="#2f5d94", label="HiT")
ax[1].plot(v["losses"], alpha=0.8, color="#c07736", label="ViT")
ax[1].set_xlabel("step"); ax[1].set_ylabel("training loss")
ax[1].set_title("300 steps, batch 8, same flowers")
ax[1].legend()

# spectrum
bands = ["low", "mid-low", "mid-high", "high"]
x = np.arange(4); w = 0.35
ax[2].bar(x - w/2, h["spectrum"], w, label="HiT", color="#2f5d94")
ax[2].bar(x + w/2, v["spectrum"], w, label="ViT", color="#c07736")
ax[2].set_xticks(x, bands); ax[2].set_ylim(0, 1)
ax[2].axhline(1.0, color="#999999", ls="--", lw=0.8)
ax[2].set_ylabel("radial power (1.0 = white noise)")
ax[2].set_title("Draw spectrum — flatter is noisier")
ax[2].legend()

fig.suptitle("HiT vs matched ViT on flowers64 — small reference models, 300 steps",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.01,
         "BANNERS: CPU torch reference, not Bendlang; ViT is a matched flowers twin "
         "(our committed Bend ViT is 8x8 digits). At 300 steps neither model reaches the "
         "real-flower ceiling. No FID or ImageNet claim is imported from the paper.",
         ha="center", fontsize=8.5, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.05, 1, 0.94))
out = os.path.join(root, "comparison.png")
plt.savefig(out, dpi=150)
print("wrote", out)
