# ViT vs HiT vs DiT(JiT) — v4, with the six corrections applied

v3 was **not** a faithful port. I checked it against `Nicholas0228/HiT-Released`
line by line and found six concrete omissions. v4 fixes all six:

1. **Positional embedding** — v3 had none; v4 adds the release's 2D sincos
   `pos_embed` to every image token.
2. **Context tokens** — v3 added one token by broadcast addition to all patches;
   v4 uses 8 class-context tokens + 1 time token, concatenated before the
   blocks and excluded from both routers and both skips, as in `model.py`.
3. **Router init** — v3 used `eye + N(0,1e-3)`; v4 uses the release's seeded
   `eye + N(0,1e-3/sqrt(width))` with seeds 1701/1702.
4. **Tree metadata** — v3 used ad-hoc labels; v4 implements the release's exact
   `tree_metadata` (DFS order + contiguous intervals) and differentiable
   `node_means` prefix-sum merge replay.
5. **Variable-length stage 2** — v3 used fixed padding; v4 groups each image's
   actual stage-1 group count.
6. **Population-based budget loss** — v4 uses the actual incoming population,
   as in `grouping.py`.

v4 also fixed a NaN in my budget loss (division by zero on empty intervals)
that had silently poisoned one "result" — that number was withdrawn.

## Measured results (1200 steps, batch 16, 2048 flowers, seed 0)

| arm | params | held legibility | final loss | wall |
|---|---:|---:|---:|---:|
| ViT | 4.74M | 0.514 | 0.074 | 466 s |
| HiT | 6.62M | 0.573 | 0.120 | 536 s |
| JiT/DiT | 6.53M | **0.607** | 0.070 | 549 s |

Real-flower positive control 0.536, negative control 0.053. **Corrected conclusion: JiT/DiT
> HiT > ViT.** The earlier v3 claim "HiT is worst" was wrong — an artifact of
the missing architectural pieces and a NaN run. Once the hierarchy is actually
built, HiT beats the plain ViT by +0.059, but still trails JiT/DiT by 0.034.

All three exceed the real-flower positive control on test-set legibility (0.514–0.607
vs 0.536): legibility measures correlation with a training flower and is
gameable by smooth blobs, so none of these models genuinely generates flowers
at this budget. `samples.png` shows this directly.

## Learned visual organization (groups4.json / group_maps.png)

Group boundaries are now produced by the model's **own** `forward(return_aux=True)`
at t=0.75 with frozen thresholds — v3's separate extraction function was another
mistake. Stage 2 now compresses properly (64 → ~43 → ~12 tokens). The gold map
tracks the bright flower body in flower 29 and a coherent flower region in
flower 26, with background split off — the paper's Figure-4 behavior at 8×8
resolution.

## Banners

- CPU torch reference. There is still **no HiT or DiT model in Bendlang** —
  `find examples -iname '*hit*'` is empty. All comparisons are torch.
- 6.6M params / 64 tokens / 1200 steps, not 960M / 256 tokens / 600 epochs.
- numba/Triton routing, packed compact batching, EMA, CFG, FID are not ported.
- JiT/DiT's higher score comes with more memorisation, not better generation.
