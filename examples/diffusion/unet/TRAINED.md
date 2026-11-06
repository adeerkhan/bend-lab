# The trained 64×64 UNet denoiser on flowers

`model.bend` is the gate example's U trained instead of checked: 64 → 32 → 16 → 32 → 64, 64 channels at each level,
an `up2` upsample at each level, the 32×32 skip, a 1×1 convolution as the head, and a **learned linear bypass** from
the input beside the U. There are no normalisations. The objective is rectified flow with the **clean image** as the
target:

    x_t = noise*(1-t) + data*t      target x0 = data      MSE
    sample: v = (x0hat - x)/(1-t), then x <- x + v*dt, Euler over linspace(0,1,steps+1)

The input has five channels — the image and four time features, `sin(2πt)`, `cos(2πt)`, `sin(4πt)`, `cos(4πt)` — and
the output is one channel with no activation on top, so the prediction can be negative. The data is
`flowers64-train-gray.u8`: **3,874 flowers at 64×64**, one grayscale byte a pixel, the training split of the pinned
Oxford parquet (sha256 `1ff86747…`). The 102 images of `flowers64-hold-gray.u8` (`f08730fe…`) are never trained on and
are what the held-out legibility is scored against. Last recorded 2026-11-06.

## What a run produced

| | value |
|---|---|
| parameters | **76,871** |
| steps | **4,000** (2 images a step, AdamW at 1e-3) |
| loss, first → final | **0.6551 → 0.0390** (minimum 0.0081) |
| a predictor of the data's mean scores | **0.0534** (the image's variance) |
| draws | 64 at 64×64, Euler 32 |
| draws, mean / std | **+0.383 / 0.107** |
| data, mean / std | +0.398 / 0.226 |
| **legibility**, 3,874 reference flowers | **0.426** (0.313 – 0.609) |
| **legibility held out**, 102 unseen flowers | **0.330** |
| ceiling: a real held-out flower scores | **0.562** |
| floor: noise scores | 0.056 |
| of the span (legibility − floor)/(ceiling − floor) | **73.1% trained, 54.2% held out** |
| spectrum, radius 1-4 / 5-10 / 11-20 / 21-32 | **0.003 / 0.002 / 0.001 / 0.060** |
| the data's spectrum, same bands | 0.006 / 0.010 / 0.031 / 0.142 |

![loss](../../../results/diffusion-unet/trained/loss.svg)

![64 draws](../../../results/diffusion-unet/trained/samples.svg)

![drawn against data](../../../results/diffusion-unet/trained/stats.svg)

The draws are **flowers, blurred**: a bright mass on a dark background, with petal structure at the right layout and
about half the data's contrast. They are not speckle, which is what every earlier run of this example produced.

## The finding: the objective, not the model

Every run before 2026-11-06 predicted the velocity `v = data − noise`. That target is 95% noise: `E[(d−n)²] = 1.053`
against the image's own `E[x₀²] = 0.203`, so a model can halve the loss by learning the noise statistics alone, and
the loss says nothing about whether a flower comes out. The same model, data, batch, parameters and steps, with only
the target changed:

| | v target | **x₀ target** |
|---|---|---|
| final loss (not comparable across targets) | 0.1581 | 0.0390 |
| legibility | 0.274 | **0.426** |
| held out | 0.216 | **0.330** |
| of the span | 43.1% / 31.6% | **73.1% / 54.2%** |
| spectrum, 1-4 / 21-32 | 0.68 / 0.68 | 0.003 / 0.060 |
| the draws | speckle | **blurred flowers** |

Three other levers were measured and ruled out on the way, each judged by legibility and by the draws' spectrum
rather than by the loss:

| lever | measured effect |
|---|---|
| **group norms** (removed) | the model with them drew speckle at legibility 0.233; without them 0.274, on fewer steps. A norm over all of an image's pixels removes the amplitude the model needs to know how much noise to remove |
| **batch size** (2 → 8 → 26, bought with resolution) | span 43.1% → 47.3% → 50.6% trained, 31.6% → 33.5% → **34.0%** held out: the last 3.25× of batch buys +0.5 points, and no resolution's draws stopped being speckle |
| **capacity** (76,871 → 301,191) | span 73.1% → 74.4% trained, 54.2% → 54.9% held out: +1.3 points for 4× the parameters |

A batch cannot grow past 2 at 64×64: a batch is `B*4096` rows in the model's types, and the checker cannot equate a
type-level `Nat` past about 16,384 on this machine (`8,192` checks, `16,384` only on an empty stack, `32,768`
overflows — bendlang/bend#1071). A larger batch has to buy its rows with resolution, which is why the batch arms run
at 16×16 and 32×32.

## The gates that guard it

| gate | result |
|---|---|
| `reference/test_unet_flowers.py` (NumPy, whole forward, two images) | **9.54e-07**, gate 1e-05 |
| `reference/compare_unet_flowers_torch.py` (PyTorch, whole forward) | **1.61e-06**, worst cell 0.16× the allowance (1e-5 + 1e-6·\|v\|) |
| `reference/test_unet_flowers_checkpoint.py` (save, resume, same weights) | **bit-identical**, 76,871 weights, 922,460-byte file |

The twins read the recorded probe: `train.bend --probe --steps 0` prints the flat weights and one whole forward at a
fixed input, and each twin rebuilds it from those weights, so neither side's initialisation is trusted. The forward
agrees to `9.54e-07` now that there are no norms to divide small variances up; the gate stays at `1e-5`.
`reference/unet_flowers_eval.py` computes the legibility and spectrum numbers above with
`reference/flowers_report.py`'s `legibility` — the same function the `flowers-flow` task gates with, so the report and
the task cannot disagree.

The recorded run used the CUDA lane (`bendtorch-gpu@2.0.0.0` in place of `bendtorch@2.0.0.0`, the same file
otherwise). The two lanes agree to the last float32 digits on the same program: 200 steps give CPU
`0.046285428 / 0.04591473`, CUDA `0.046285436 / 0.045914732`, final `0.025204565` against `0.025204562`.

## Reproduce

    BEND_MEM=9G scripts/bend-guard.sh bend -o .scratch/unet_train examples/diffusion/unet/train.bend
    .scratch/unet_train --save /tmp/unet.bin                 # 4,000 steps, ~30 minutes
    .scratch/unet_train --steps 0 --resume /tmp/unet.bin     # the 64 draws
    reference/.venv/bin/python reference/unet_flowers_results.py <log>
    reference/.venv/bin/python reference/unet_flowers_eval.py <log>
    reference/.venv/bin/python reference/test_unet_flowers.py
    reference/.venv/bin/python reference/test_unet_flowers_checkpoint.py
    ../bendtorch/reference/.venv/bin/python reference/compare_unet_flowers_torch.py

The compile is the expensive part (9 GB under the guard); the step is 0.25 s on the CPU lane and 0.57 s on the CUDA
lane at this size. `--steps N`, `--start N`, `--save PATH`, `--resume PATH`, `--no-sample` and `--probe` work as in
`flow_matching`. The draws of a checkpoint come from `--steps 0 --resume`, which trains nothing.

## Bugs this example cost, and what now guards them

- **A permutation.** The training pair and the sampler built the five input channels in opposite orders, so the field
  was learned on one permutation and applied to another: the loss fell while the draws stayed at the noise's spread.
- **Hardcoded initialisation offsets** from the first (two-channel) layout zeroed real weights and one-filled
  unrelated cells, so every run started scrambled. The offsets now follow the model's forward order and are checked
  before every run: `--probe --steps 0` must show every bias at 0, every gain at 1, and no stray 0 or 1 elsewhere.
- **A batch that does not divide the image count** reads past the end of the export, and only on the last batch.
  3,874 = 149 × 26 and 3,872 = 484 × 8 were chosen to avoid it; the 64×64 record uses batch 2, which divides 3,874.

## Why there is no report card

`reference/experiment.py` tasks carry a reference arm and a gate on a held-out metric. This example has a held-out
metric now — the 102 unseen flowers, with a ceiling of 0.562 — but no reference arm inside the task machinery: the
honest comparison is `examples/diffusion/flowers_flow`, whose row-token model reaches **0.536 trained / 0.490 held
out** on the same reference set with 413,440 parameters. The UNet is behind it, and is the best UNet here.

## Limits

The draws are blurred. Two levels with a 4×4 bottleneck carry 256 numbers per image into the widest representation, so
the model can say *a flower is here, this bright, this layout* and cannot say *these petals, this edge*. Capacity and
steps are measured not to be the limit either (+1.3 points for 4× parameters), so the next lever is reach, not width:
a third level, whose bottleneck is 2×2 numbers per image.
