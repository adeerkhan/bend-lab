# The trained 64×64 UNet denoiser on flowers

`model.bend` is the gate example's U trained instead of checked: **five levels**, 64 → 32 → 16 → 8 → 4 → **2** → 4
→ 8 → 16 → 32 → 64, 64 channels at each level, an `up2` upsample at every level, a residual skip at 32×32 and another
at 16×16, a 1×1 convolution as the head, and a **learned linear bypass** from the input beside the U. There are no
normalisations. The objective is rectified flow with the **clean image** as the target:

    x_t = noise*(1-t) + data*t      target x0 = data      MSE
    sample: v = (x0hat - x)/(1-t), then x <- x + v*dt, Euler over linspace(0,1,steps+1)

The input has five channels — the image and four time features, `sin(2πt)`, `cos(2πt)`, `sin(4πt)`, `cos(4πt)` — and
the output is one channel with no activation on top, so the prediction can be negative. The data is
`flowers64-trainfull-gray.u8`: **7,067 flowers at 64×64**, one grayscale byte a pixel, sha256 `87ee501c…`, the training
split of the pinned Oxford parquet (`reference/flowers_export.py`) in a fixed permutation (numpy `default_rng(0)`).
The 102 images of `flowers64-hold-gray.u8` (`f08730fe…`) are never trained on and are what the held-out legibility is
scored against. Last recorded 2026-11-08.

## What a run produced

| | value |
|---|---|
| parameters | **298,439** |
| steps | **3,600** (2 images a step, AdamW at 1e-3) — 1.02 passes over the 7,066 |
| loss, first → final | **15.3262 → 0.0259** (minimum 0.0091); the first value is the untrained head on the first batch — step 1 is 0.1343 |
| a predictor of the data's mean scores | 0.0539 |
| draws | 64 at 64×64, Euler 32 |
| draws, mean / std | **+0.325 / 0.179** |
| data, mean / std | +0.384 / 0.232 |
| **legibility**, 7,067 reference flowers | 0.633 — *above the ceiling: the draws memorise* |
| **legibility held out**, 102 unseen flowers | **0.547** |
| ceiling: a real held-out flower scores | 0.574 |
| floor: noise scores | 0.058 |
| of the span (legibility − floor)/(ceiling − floor), held out | **94.8%** |
| spectrum, radius 1-4 / 5-10 / 11-20 / 21-32 | 0.002 / 0.001 / 0.000 / 0.034 |
| the data's spectrum, same bands | 0.005 / 0.009 / 0.029 / 0.137 |

![loss](../../../results/diffusion-unet/trained/loss.svg)

![64 draws](../../../results/diffusion-unet/trained/samples.svg)

![drawn against data](../../../results/diffusion-unet/trained/stats.svg)

The five-level is the record by the rule set before it ran: it must beat the four-level's held-out span by three points,
and it does. The four-level's committed draws, re-scored today with the same evaluator, read 0.605 / **0.501** / 85.8%
(the published record said 0.498 / 85.4%); the five-level reads 0.633 / **0.547** / 94.8%. That is +4.6 points of
held-out legibility and +9.0 points of span.

The improvement is in the flower's **global identity**, not in its texture or its tone. The five-level's raw draws are
lower contrast than the four-level's (+0.325 / 0.179 against +0.381 / 0.238), and their top spectrum band is slightly
lower (0.034 against 0.039). A deeper network fits a more confident conditional mean, and an MSE objective shrinks an
uncertain mean toward gray; the held-out correlation, which is scale-invariant, rewards the better structure. The
four-level's pictures look punchier; the five-level is the better denoiser by the held-out metric. Both statements
belong to the record.

More Euler steps do not change that. On the same 8 draws, 32 and 128 steps give spread 0.185 and 0.174, top band 0.033
and 0.027, and legibility 0.656 / 0.575 against 0.673 / 0.568 trained / held out — differences inside the draw-count
noise below. The sampler is not the cause; 32 stays.

## How it got there: six levers, each measured

| arm | change | legibility | held out | of the span, held out |
|---|---|---|---|---|
| first record | — | 0.259 | 0.206 | 29.6% |
| objective | `v` → `x₀` | 0.426 | 0.330 | 54.2% |
| corpus | 512 → 3,874 images | 0.426 | 0.330 | 54.2% |
| third level | bottleneck 8×8 | 0.560 | 0.439 | 73.8% |
| corpus again | 3,874 → 7,067 | 0.424 → (no effect on two levels) | — | — |
| fourth level | bottleneck **4×4** | 0.603 | 0.498 | 85.4% |
| **fifth level** | bottleneck **2×2** | **0.633** | **0.547** | **94.8%** |

Reach is the lever for flower identity. The objective was the first one and the largest; every level of depth has paid;
the corpus alone was measured twice and did nothing on its own (73.1% → 72.6% at three levels); a wider stack (CH = 96,
336,679 parameters) was run as the alternative and did not beat depth. Group norms and batch size were measured and
rejected earlier: 0.233 with norms against 0.274 without, and a batch of 26 bought with resolution moved the held-out
span by +0.5. The fifth level is where depth stops paying for what the eye wants: it bought identity, not petals, and
cost contrast.

## Measuring it honestly

Three traps, all fixed in the tools or in the reading:

- **The reference set must be the one the model trained on.** `reference/unet_flowers_eval.py` used to default to the
  3,874-image subset, which is *inside* the 7,067; that inflates both `legibility` and the ceiling (0.562 against
  0.574). It now defaults to the full export, prints the reference set's size beside the ceiling, and prints a note when
  the trained legibility passes the ceiling. The numbers above are against the full set.
- **Memorisation is visible now.** At 298,439 parameters on 7,067 images the drawn `legibility` of 0.633 is above the
  ceiling of 0.574: the draws remember training flowers. The held-out figure is the only one that describes the model.
- **A max-over-correlations statistic moves with the draw count.** The five-level's first 8 draws score 0.656 trained /
  0.575 held out, 100.3% of the span; all 64 score 0.633 / 0.547, 94.8%. The 8-draw figure is not a better model, it is
  a smaller sample. The record is the 64-draw number, and every arm or sampler comparison must hold the draw count
  fixed. This cost one false +3-point "win" for 128 Euler steps on the four-level before it was caught.

## The gates that guard it

| gate | result | tolerance |
|---|---|---|
| `reference/test_unet_flowers.py` (NumPy, whole forward, two images) | **1.621e-05** | 2e-05 |
| `reference/compare_unet_flowers_torch.py` (PyTorch, whole forward) | **2.098e-05**, 0.83× the allowance | 2e-05 + 1e-6·\|v\| |
| `reference/test_unet_flowers_checkpoint.py` (save, resume) | **bit-identical**, 298,439 weights | exact |

The forward tolerance moved from `1e-5` to `2e-5` when the model gained its fourth level: the fifth adds two more
576-term convolutions and the measured error rose to 1.62e-05 (NumPy) and 2.10e-05 raw on torch, 0.83× the allowance
once its `|v|` term is included. That is the measured reason written beside the gate, and the can-fail still fails on a
0.5 error.

The twins read the recorded probe: `train.bend --probe --steps 0` prints the flat weights and one whole forward at a
fixed input, and each twin rebuilds it from those weights, so neither side's initialisation is trusted. The five-level
probe is 298,439 values in 22 blocks; the twins' intermediate level names are `l4`/`l2` because `b2`/`b3` are the biases
of the same names.

## Reproduce

    BEND_MEM=9G scripts/bend-guard.sh bend -o .scratch/unet_train examples/diffusion/unet/train.bend
    .scratch/unet_train --save /tmp/unet.bin                 # 3,600 steps
    .scratch/unet_train --steps 0 --resume /tmp/unet.bin     # the 64 draws
    reference/.venv/bin/python reference/unet_flowers_results.py <log>
    reference/.venv/bin/python reference/unet_flowers_eval.py <log>
    reference/.venv/bin/python reference/test_unet_flowers.py
    reference/.venv/bin/python reference/test_unet_flowers_checkpoint.py
    reference/.venv/bin/python reference/compare_unet_flowers_torch.py

The compile is the expensive part (9 GB under the guard); the step is 0.92 s wall on the CUDA lane at this size (55 min
11 s for 3,600 steps; 38 min 13 s of it user time). The CPU lane was not timed for five levels. `--steps N`, `--start N`, `--save PATH`, `--resume PATH`,
`--no-sample` and `--probe` work as in `flow_matching`. The draws of a checkpoint come from `--steps 0 --resume`, which
trains nothing.

## Bugs this example cost, and what now guards them

- **A permutation.** The training pair and the sampler built the five input channels in opposite orders, so the field
  was learned on one permutation and applied to another: the loss fell while the draws stayed at the noise's spread.
- **Hardcoded initialisation offsets** from the first (two-channel) layout zeroed real weights and one-filled
  unrelated cells. The offsets follow the model's forward order now and are checked before every run: `--probe --steps
  0` must show every bias at 0 and no stray 0 or 1 elsewhere.
- **A batch that does not divide the image count** reads past the end of the export, and only on the last batch.
- **A corpus read in less than one pass** trains on a prefix of the classes, because the parquet is grouped by label.
  3,600 steps at batch 2 is 1.02 passes; the earlier 2,000-step budget on the full corpus would have seen 58 of 102.

## Why there is no report card

`reference/experiment.py` tasks carry a reference arm and a gate on a held-out metric. This example has the metric — 102
unseen flowers, ceiling 0.574 — but no reference arm inside the task machinery. The honest comparison is
`examples/diffusion/flowers_flow`, whose row-token model reaches **0.536 trained / 0.490 held out** with 413,440
parameters. This UNet's held-out **0.547** is above it with 298,439 parameters; its trained 0.633 is above the ceiling
and is not comparable.

## Limits

The draws are soft, and the fifth level did not fix that: the spectrum is steeper than the data's at every band, the top
band moved from 0.039 at four levels to 0.034 at five, and the raw draws lost contrast. The bottleneck is now 2×2, four
numbers per image, so another level has nothing left to remove. The next lever is a mechanism that adds detail and
contrast to the conditional mean — a different sampler, a loss that does not shrink the amplitude, or a texture stage —
not a sixth level. What is measured and closed: the objective, the normalisation, the corpus on its own, the batch,
Euler steps (32/128/256 on the four-level, 32/128 on the five-level), and five levels of depth.
