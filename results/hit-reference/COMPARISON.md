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
| test-set legibility | **0.220** | 0.210 | negative control 0.051, real test flower 0.521 |
| legibility vs training set | 0.249 | 0.259 | — |
| final training loss | 0.132 | 0.099 | — |
| wall (300 steps) | 44.2 s | 5.6 s | CPU torch 2.15 — **see banner 6** |

HiT is **+0.010 test-set legibility** over the matched ViT at the same step
budget and slightly fewer parameters, but is ~8× slower per step because the
average-linkage tree is rebuilt every forward pass. At 300 steps neither arm
is close to the real-flower positive control (0.521), so this is a wiring-and-direction
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

## Correction: the wall-time number is not an architecture result

I first wrote "HiT is ~8× slower per step". That was **wrong to state as an
architecture comparison**. Profiling on this reference (`hit_ref.py`, batch 8):

| component | time |
|---|---|
| HiT forward | 32.9 ms |
| ViT forward | 3.5 ms |
| scipy linkage alone | 0.8 ms |
| my Python tree + merge replay | 11.8 ms |
| full HiT grouping stage | 16.9 ms |

The official release (`Nicholas0228/HiT-Released`, `hnet/linkage.py`,
`hnet/runtime/affinity.py`) runs the same exact average linkage through
**numba JIT** (`njit`, parallel, outside the GIL) and the soft-mean/readback
through fused **Triton** segmented sums. My reference does the merge replay as
hundreds of small Python/torch ops, so its wall time measures my implementation,
not HiT. The architecture-level cost is token-blocks at matched 10 blocks:

| | token-blocks (K1=31, K2=16) | relative |
|---|---:|---:|
| HiT (2×64 + 2×31 + 2×16 + 2×31 + 2×64) | 412 | **0.64×** |
| ViT (10×64) | 640 | 1.00× |

This agrees with the paper's HiT-B 22.2 GFLOPs vs JiT-B 25.2 GFLOPs. So the
honest statement is: **HiT is structurally cheaper, my reference is slower
only because it is a Python prototype.** Any real efficiency claim needs the
numba/Triton implementation or a FLOP count, not this wall time.

## Banners — do not read past these

- This is a **CPU torch reference**, not a Bendlang implementation. No Bend
  build or gate is implied by these numbers.
- Our committed Bend ViT (`examples/diffusion/vit`) is an **8×8 digits**
  denoiser; the ViT arm here is a **matched flowers twin** at the same patch,
  width and block count, not that program recompiled on flowers.
- The paper's FID/ImageNet numbers (1.48 FID, 40% JiT FLOPs) are **not our results**
  and are not claimed here.
- The 44 s vs 6 s wall time is a prototype artifact (see correction above),
  not evidence that ViT is the more efficient architecture.
- 300 steps and 1,024 training flowers are far short of the paper's 80–600
  epochs on ImageNet; group quality and legibility should be expected to change
  with a longer run.
- Group counts vary with image content and diffusion timestep, as in the
  paper; the 55/31 reported are for the three plotted flowers at t=0.75.
