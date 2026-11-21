# ViT vs HiT vs DiT(JiT) on flowers64 — faithful small port

A three-arm comparison built by reading `Nicholas0228/HiT-Released` line by line,
not by approximating it. Source: `reference/hit_v2.py`, plots from
`reference/hit_plot_v2.py`. Records in `vit/`, `jit/`, `hit/`.

## How this maps to the released code

| released file | what we ported |
|---|---|
| `hnet/model.py` `CleanJiT` | same five-stage structure for HiT (encoder/mid/core/dec-mid/decoder); the released JiT baseline is the dense DiT-style arm (JiT arm here) |
| `hnet/layers.py` | RMSNorm, QK-RMSNorm, SwiGLU MLP, 6-way adaLN shift/scale/gate; SpatialRoPE omitted at this scale |
| `hnet/grouping.py` `SoftGrouping` | exact average linkage via scipy's pinned `nn_chain`; **vectorized** soft reader with `scatter_add` group means of q and k — this fixed my v1's 30× Python-replay slowdown |
| `hnet/config.py` | `reader_temperature 0.05`, `budget_temperature 0.1`, `budget_weight 0.01`, ratios (2,2) |
| `docs/RECIPE.md` | AdamW betas (0.9,0.95), lr 2e-4, thresholds calibrated on fixed noisy examples then frozen |
| `scripts/calibrate.py` | pooled merge scores over several noise levels, tau chosen so total groups = population/ratio |

Arms: **ViT** = plain adaLN-free blocks (100k-scale, no context). **JiT/DiT** =
adaLN blocks with class/time context tokens at mid-depth (the released baseline).
**HiT** = same blocks in a 1/1/2/1/1 hierarchy with two routers, skips W1/W2,
and the ratio compression loss. All three use the same clean-image velocity
loss, 1200 steps, batch 16, 2,048 flowers, seed 0.

## Numbers (from the three run.json files)

| | ViT | JiT/DiT | HiT |
|---|---:|---:|---:|
| params | 805,824 | 1,400,256 | 1,498,560 |
| held-out legibility | 0.213 | 0.213 | **0.221** |
| final loss | 0.085 | 0.092 | 0.082 |
| wall (1200 steps) | 92 s | 84 s | 109 s |

So at this scale **HiT is best on held-out legibility**, by +0.006 over JiT/DiT
and +0.008 over ViT, with more parameters. JiT/DiT and ViT are tied at this scale. The earlier v1 result (HiT worse)
was caused by a stale one-batch threshold calibration and too-high LR — this
version follows the release recipe and adds one mid-run threshold retune.

## Why this is still not the paper's result

- **Scale:** 1.5M params / 1200 steps vs the paper's 960M params / 600 epochs.
  At 64 image tokens grouping is weak (64 → ~45 → ~39 rather than 2× per level);
  the paper's hierarchy advantage is a large-scale, long-training effect.
- **Threshold retune:** the release freezes thresholds calibrated on the
  *initial* model; at our scale those became stale. Retuning once mid-training
  was necessary to make grouping active — a deviation from the recipe, noted.
- **No EMA:** the release evaluates EMA 0.9999 over ~125k updates; at 1200 steps
  that leaves EMA weights near initialization, so sampling uses the live weights.
- **RoPE, packed compact batching, numba/Triton kernels, CFG, FID** are not
  implemented here. No FID or ImageNet number is claimed as ours.
- **Not Bendlang:** all runs are CPU torch. `examples/diffusion/hit` (a Bend port)
  is future work, as is a Bend DiT — BendTorch's `Attention` is causal-only
  (ISSUE-011) and adaLN is not a BendTorch layer.
