# The trained 64×64 UNet denoiser on flowers

`model.bend` is the gate example's U trained instead of checked: **three levels**, 64 → 32 → 16 → **8** → 16 → 32 → 64,
64 channels at each level, an `up2` upsample at each level, a residual skip at 32×32 and another at 16×16, a 1×1
convolution as the head, and a **learned linear bypass** from the input beside the U. There are no normalisations. The
objective is rectified flow with the **clean image** as the target:

    x_t = noise*(1-t) + data*t      target x0 = data      MSE
    sample: v = (x0hat - x)/(1-t), then x <- x + v*dt, Euler over linspace(0,1,steps+1)

The input has five channels — the image and four time features, `sin(2πt)`, `cos(2πt)`, `sin(4πt)`, `cos(4πt)` — and
the output is one channel with no activation on top, so the prediction can be negative. The data is
`flowers64-trainfull-gray.u8`: **7,067 flowers at 64×64**, one grayscale byte a pixel, sha256 `87ee501c…`, the training
split of the pinned Oxford parquet (`reference/flowers_export.py`) in a fixed permutation (numpy `default_rng(0)`).
The 102 images of `flowers64-hold-gray.u8` (`f08730fe…`) are never trained on and are what the held-out legibility is
scored against. Last recorded 2026-11-07.

## What a run produced

| | value |
|---|---|
| parameters | **150,727** |
| steps | **3,600** (2 images a step, AdamW at 1e-3) — 1.02 passes over the 7,066 |
| loss, first → final | **0.5542 → 0.0241** (minimum 0.0124) |
| a predictor of the data's mean scores | **0.0539** (the image's variance) |
| draws | 64 at 64×64, Euler 32 |
| draws, mean / std | +0.302 / **0.233** |
| data, mean / std | +0.384 / 0.232 |
| **legibility**, 7,067 reference flowers | **0.545** (0.373 – 0.706) |
| **legibility held out**, 102 unseen flowers | **0.438** |
| ceiling: a real held-out flower scores | **0.562** |
| floor: noise scores | 0.056 |
| of the span (legibility − floor)/(ceiling − floor) | **96.6% trained, 75.5% held out** |
| of the ceiling | **97.0% trained, 77.9% held out** |
| spectrum, radius 1-4 / 5-10 / 11-20 / 21-32 | 0.002 / 0.001 / 0.000 / 0.038 |
| the data's spectrum, same bands | 0.006 / 0.010 / 0.031 / 0.142 |

![loss](../../../results/diffusion-unet/trained/loss.svg)

![64 draws](../../../results/diffusion-unet/trained/samples.svg)

![drawn against data](../../../results/diffusion-unet/trained/stats.svg)

The draws are **flowers**: a bright mass, a dark surround, and a layout and scale that read as a flower head. They are
still soft — the spectrum is *steeper* than the data's, i.e. the model leaves the fine, high-frequency texture out, and
the mean is 0.302 against the data's 0.384, so they run a little dark. They are no longer blobs or speckle.

## How it got there: four levers, each measured

| arm | model | data | steps | legibility | held out | of the span |
|---|---|---|---|---|---|---|
| first record | 2 levels, 76,871 | 3,874 | 4,000 | 0.426 | 0.330 | 73.1% / 54.2% |
| **third level** | 3 levels, 150,727 | 3,874 | 2,000 | 0.485 | 0.376 | 84.8% / 63.2% |
| **full corpus** | 2 levels, 76,871 | **7,067** | 3,600 | 0.424 | 0.330 | 72.6% / 54.2% |
| **both** (this record) | 3 levels, 150,727 | **7,067** | 3,600 | **0.545** | **0.438** | **96.6% / 75.5%** |

**Reach is the lever; the corpus alone is not.** Doubling the data on the two-level model moved nothing (73.1% → 72.6%,
held out identical); the third level moved 11.7 and 9.0 points on its own, and *together* they moved 23.5 and 21.3.
The row-token reference behaves the other way round (512 → 3,874 images was worth 73% → 87% of its ceiling), which is
the point: the bottleneck is what a model can represent, and a two-level U with a 4×4 bottleneck cannot carry texture
however much it is shown.

Two levers were measured and ruled out earlier, and are worth not re-trying:

- **group norms**: the same model with them drew speckle at legibility 0.233 against 0.274 without. A norm over all of
  an image's pixels removes the amplitude the model needs in order to know how much noise to remove.
- **batch size**: 2 → 8 → 26, bought with resolution, moved the span 43.1% → 47.3% → 50.6% trained and 31.6% → 33.5% →
  34.0% held out, and every resolution's draws stayed speckle. A batch cannot exceed 2 at 64×64 anyway: a batch is
  `B*4096` rows in the model's types, and the checker cannot equate a type-level `Nat` past about 16,384 on this
  machine (bendlang/bend#1071; 8,192 checks, 16,384 only on an empty stack, 32,768 overflows).

## The finding that mattered most: the objective

Every run before 2026-11-06 predicted the velocity `v = data − noise`. That target is 95% noise — `E[(d−n)²] = 1.053`
against the image's own `E[x₀²] = 0.203` — so a model halves the loss by learning the noise statistics alone, and the
loss says nothing about whether a flower comes out. The same model, data, batch, parameters and steps, with only the
target changed: legibility **0.274 → 0.426**, held out **0.216 → 0.330**, and the draws went from speckle to blurred
flowers.

## The gates that guard it

| gate | result |
|---|---|
| `reference/test_unet_flowers.py` (NumPy, whole forward, two images) | **2.38e-06**, gate 1e-05 |
| `reference/compare_unet_flowers_torch.py` (PyTorch, whole forward) | **3.93e-06**, worst cell 0.36× the allowance (1e-5 + 1e-6·\|v\|) |
| `reference/test_unet_flowers_checkpoint.py` (save, resume, same weights) | **bit-identical**, 150,727 weights, 1,808,732-byte file |

The twins read the recorded probe: `train.bend --probe --steps 0` prints the flat weights and one whole forward at a
fixed input, and each twin rebuilds it from those weights, so neither side's initialisation is trusted.
`reference/unet_flowers_eval.py` computes the legibility, the held-out legibility, the ceiling, the noise floor and the
draws' spectrum with `reference/flowers_report.py`'s `legibility` — the same function the `flowers-flow` task gates
with, so this report and that task cannot disagree.

The recorded run used the CUDA lane (`bendtorch-gpu@2.0.0.0` in place of `bendtorch@2.0.0.0`, the same file
otherwise), because the same program is 1.5× faster there at this size. The two lanes agree to the last float32 digits:
200 steps give CPU `0.041913982 / 0.050086714`, CUDA `0.04191398 / 0.050086714`, and `final_loss` is `0.02286052` in
both.

## Reproduce

    BEND_MEM=9G scripts/bend-guard.sh bend -o .scratch/unet_train examples/diffusion/unet/train.bend
    .scratch/unet_train --save /tmp/unet.bin                 # 3,600 steps
    .scratch/unet_train --steps 0 --resume /tmp/unet.bin     # the 64 draws
    reference/.venv/bin/python reference/unet_flowers_results.py <log>
    reference/.venv/bin/python reference/unet_flowers_eval.py <log>
    reference/.venv/bin/python reference/test_unet_flowers.py
    reference/.venv/bin/python reference/test_unet_flowers_checkpoint.py
    reference/.venv/bin/python reference/compare_unet_flowers_torch.py

The compile is the expensive part (9 GB under the guard); the step is 0.57 s on the CPU lane at this size. `--steps N`,
`--start N`, `--save PATH`, `--resume PATH`, `--no-sample` and `--probe` work as in `flow_matching`. The draws of a
checkpoint come from `--steps 0 --resume`, which trains nothing.

## Bugs this example cost, and what now guards them

- **A permutation.** The training pair and the sampler built the five input channels in opposite orders, so the field
  was learned on one permutation and applied to another: the loss fell while the draws stayed at the noise's spread.
- **Hardcoded initialisation offsets** from the first (two-channel) layout zeroed real weights and one-filled
  unrelated cells, so every run started scrambled. The offsets now follow the model's forward order and are checked
  before every run: `--probe --steps 0` must show every bias at 0, every gain at 1, and no stray 0 or 1 elsewhere.
- **A batch that does not divide the image count** reads past the end of the export, and only on the last batch.
- **A corpus read in less than one pass** trains on a prefix of the classes, because the parquet is grouped by label.
  3,600 steps at batch 2 is 1.02 passes over the 7,066; the record's earlier 2,000-step budget on the full corpus would
  have seen 58 of 102 classes.

## Why there is no report card

`reference/experiment.py` tasks carry a reference arm and a gate on a held-out metric. This example now has the metric —
102 unseen flowers, with a ceiling of 0.562 — but no reference arm inside the task machinery. The honest comparison is
`examples/diffusion/flowers_flow`, whose row-token model reaches **0.536 trained / 0.490 held out** on the same
reference set with 413,440 parameters: this UNet is now *ahead* of it on the training-set legibility (0.545 against
0.536) and behind on the held-out one (0.438 against 0.490), which is the more useful number.

## Limits

The draws are soft, a little dark (mean 0.302 against 0.384), and carry no fine texture: the spectrum is steeper than
the data's at every band, so the model reproduces the shape and layout of a flower and not its petals. The next lever is
a **fourth level** (the bottleneck is 8×8, i.e. 64 numbers per image) or a wider stack, not more images — and after
that, a sampler that adds detail instead of integrating a mean-ish field.
