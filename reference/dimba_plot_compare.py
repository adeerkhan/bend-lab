#!/usr/bin/env python3
"""Four-arm comparison: ViT / JiT(DiT) / HiT / Dimba-style hybrid."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = {}
for name, p in (("ViT","results/hit-v4/vit/run.json"),("JiT / DiT","results/hit-v4/jit/run.json"),
                ("HiT","results/hit-v4/hit/run.json"),("Dimba hybrid","results/dimba/hybrid/run.json")):
    R[name] = json.load(open(p))
C = {"ViT":"#7a7a7a","JiT / DiT":"#c07736","HiT":"#2f5d94","Dimba hybrid":"#7e9a5e"}
names = list(R)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
a = ax[0]
vals = [R[n]["legibility_held"] for n in names]
bars = a.bar(names, vals, color=[C[n] for n in names], width=0.55)
for b, v in zip(bars, vals):
    a.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontweight="bold")
a.set_ylim(0, 0.72); a.set_ylabel("held-out legibility"); a.set_title("Quality: hybrid leads")

a = ax[1]
for n in names:
    a.plot(R[n]["losses"], color=C[n], label=n, alpha=0.85, lw=1.2)
a.set_xlabel("step"); a.set_ylabel("training loss"); a.set_title("1200 steps, same flowers"); a.legend(fontsize=8)

a = ax[2]
x = np.arange(4)
a.bar(x, [R[n]["params"]/1e6 for n in names], color=[C[n] for n in names], width=0.55)
a.set_xticks(x, [n.replace(" / ", "/") for n in names], rotation=20, ha="right")
a.set_ylabel("parameters (millions)"); a.set_title("Capacity")

fig.suptitle("ViT vs JiT(DiT) vs HiT vs Dimba-style hybrid — flowers64, 1200 steps",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.012,
         "BANNERS: CPU torch reference, 4.7-7.6M params at 64 tokens. Dimba arm is a structural analogue "
         "(DiT blocks + Mamba SSM, no VAE/T5/captions), not the released 0.9B text-to-image model. "
         "Hybrid took 1705 s vs ~500 s for the others (recurrent scan). No FID claimed.",
         ha="center", fontsize=8, color="#8b2f2f")
plt.tight_layout(rect=(0, 0.06, 1, 0.93))
out = "results/dimba/comparison.png"
fig.savefig(out, dpi=150)
print("wrote", out)
