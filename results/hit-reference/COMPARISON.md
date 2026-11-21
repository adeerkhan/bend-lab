# HiT vs a matched ViT on flowers64 — small reference models

A first, faithful-scale port of the mechanism in **arXiv:2611.22962** (HiT),
run on our pinned flowers64 split. Source: `reference/hit_ref.py` (both arms),
`reference/hit_plot_groups.py`, `reference/hit_plot_compare.py`. The PNGs in
this directory are produced from the committed `run.json` / `groups.json`, so
they reproduce without retraining.

## What is actually being compared

| | HiT | ViT arm |
|---|---|---|
| params | 368,128 | 343,168 |
| features | patch 8 → 64 tokens, width 64, 10 blocks | same patch, width, block count |
| structure | two 2× grouping stages (64→55→31 at t=0.75), soft pooling/ungroup, compression loss | plain transformer, no grouping |
| training | 300 steps, batch 8, AdamW 1e-3, same flowers, seed 0 | identical |

Both use the paper's clean-image velocity loss (Eq. 12: `MSE(pred,x)/max(1-t,0.05)^2`)
and rectified interpolant `x_t = t·x + (1-t)·ε`. HiT implements the exact
equation set: symmetric affinity (2), average-linkage hard groups (8),
softmax soft weights (9), column-normalized pooling `Z = PᵀH` and ungroup
`H̃ = A·M(Z) + HWᵀ` (3), and the ratio compression loss with sigmoid soft group
count (10/11) at λ=0.01, Tr=0.05, Tb=0.1.

## Numbers (from `hit/run.json` and `vit/run.json`)

| metric | HiT | ViT | reference points |
|---|---|---|---|
| held-out legibility | **0.220** | 0.210 | noise floor 0.051, real held-out flower 0.521 |
| legibility vs training set | 0.249 | 0.259 | — |
| final training loss | 0.132 | 0.099 | — |
| wall (300 steps) | 44.2 s | 5.6 s | CPU torch 2.15 |

HiT is **+0.010 held-out legibility** over the matched ViT at the same step
budget and slightly fewer parameters, but is ~8× slower per step because the
average-linkage tree is rebuilt every forward pass. At 300 steps neither arm
is close to the real-flower ceiling (0.521), so this is a wiring-and-direction
result, not a quality verdict.

## Learned groups (Figure-4 analogue)

`group_maps.png` shows three flowers at t=0.75 (groups4.json, thresholds
recalibrated on the trained model at the visualization timestep, then frozen). The top row overlays group
boundaries on the clean image; the bottom row is the hard group map with one
spatially scattered group highlighted in gold. On flowers 1 and 2 the gold
group collects the **background** — disconnected patches that share appearance
— while the bright flower is split into many fine groups. This is the paper's
reported behaviour (object region / background / nonlocal similarity) emerging
from denoising alone, without segmentation labels. The small reference only
has 8×8 patches, so the "object part" detail is limited by resolution.

## Banners — do not read past these

- This is a **CPU torch reference**, not a Bendlang implementation. No Bend
  build or gate is implied by these numbers.
- Our committed Bend ViT (`examples/diffusion/vit`) is an **8×8 digits**
  denoiser; the ViT arm here is a **matched flowers twin** at the same patch,
  width and block count, not that program recompiled on flowers.
- The paper's FID/ImageNet numbers (1.48 FID, 40% JiT FLOPs) are **not our results**
  and are not claimed here.
- 300 steps and 1,024 training flowers are far short of the paper's 80–600
  epochs on ImageNet; group quality and legibility should be expected to change
  with a longer run.
- Group counts vary with image content and diffusion timestep, as in the
  paper; the 55/31 reported are for the three plotted flowers at t=0.75.
