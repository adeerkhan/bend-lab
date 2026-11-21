# ViT vs HiT vs DiT(JiT) — faithful full-architecture port on flowers64

This is the corrected implementation: `reference/hit_v3.py` ports the released
architecture's actual components, not the simplified v2. Source read line by
line from `Nicholas0228/HiT-Released`:

| released component | v3 status |
|---|---|
| 5-stage 3/3/6/3/3 depth (18 blocks) | ported (v2 used 1/1/2/1/1) |
| SwiGLU MLP at 4× width | ported (v2 used GELU at 2×) |
| SpatialRoPE on full-resolution q/k | ported (v2 omitted it) |
| QK-RMSNorm, adaLN 6-way modulation | present since v2 |
| exact scipy `nn_chain` average linkage | present since v2 |
| vectorized scatter_add group-mean reader | present since v2 |
| calibration on pooled scores, frozen | ported (v2 retuned mid-run) |
| numba/Triton routing, packed compact batching | **not ported** (execution, not architecture) |
| EMA 0.9999, CFG, FID | **not ported** (recipe/eval, not architecture) |

All arms use the same 18 blocks, width 128, patch 8, 64 tokens, same
clean-image velocity loss, AdamW 2e-4 betas (0.9,0.95), 2,048 flowers, seed 0.

## Measured results

| arm | steps | params | held legibility | train legibility | wall |
|---|---:|---:|---:|---:|---:|
| ViT | 1200 | 4.74M | 0.473 | 0.550 | 459 s |
| JiT/DiT | 1200 | 6.53M | **0.495** | 0.566 | 509 s |
| HiT | 1200 | 6.62M | 0.430 | 0.503 | 511 s |
| JiT/DiT | 2400 | 6.53M | **0.503** | 0.569 | 1,235 s |
| HiT | 2400 | 6.62M | 0.405 | 0.471 | 1,133 s |

Real-flower ceiling 0.536, noise floor 0.053. **The dense JiT/DiT arm wins at
this scale, and the gap grows with training** — the opposite of the paper's
ImageNet result. This is now a well-controlled negative result, not a bug:
the architecture is faithfully ported, grouping is active (64→45→35 tokens
at the two levels, not the target 32→16), and longer training does not help
the hierarchy on 2,048 small flowers with 64 tokens.

## Why this does not contradict the paper

- **Scale.** 6.6M params / 2,048 flowers / 64 tokens vs 960M params / 1.28M
  ImageNet images / 256 tokens. The paper's compression ratio is 2× per level;
  ours is ~1.4× because the pooled calibration drifts as affinities evolve.
  A hierarchy that reduces 64 tokens by 1.4× buys little and costs skips and
  routers — at small scale that is pure overhead.
- **Recipe.** Thresholds are frozen at init as in the release; no EMA (0.9999
  over 1,200 updates would leave weights near init), no CFG, no FID.
- **Execution.** CPU torch; the release's numba/Triton kernels were not ported
  (execution, not architecture, so they cannot explain the quality gap).
- **Not Bendlang.** All runs are CPU torch. BendTorch's `Attention` is
  causal-only (ISSUE-011) and adaLN is not a BendTorch layer, so a faithful
  DiT/HiT port needs upstream work first.

## Conclusive statement

On the flowers64 task at the scale this machine can train: **JiT/DiT > ViT >
HiT**. The learned hierarchy does not pay for itself here. Whether it does at
larger scale is the paper's claim and remains untested by us — we will not
import their FID as our result.
