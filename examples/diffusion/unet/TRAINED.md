# The trained 64×64 UNet denoiser on flowers

`model.bend` is the gate example's U trained instead of checked: 64 → 32 → 16 → 32 → 64, 16 channels at each level,
a group norm before every activation, an `up2` upsample at each level, the 32×32 skip, and a 1×1 convolution as the
head. The objective is rectified flow on the 512 flowers of `flowers64-512-gray.u8` (64×64, one grayscale byte a
pixel):

    x_t = noise*(1-t) + data*t     target v = data - noise     MSE
    sample: x <- x + v(x,t)*dt, Euler over linspace(0,1,steps+1)

The input has five channels — the image and four time features, `sin(2πt)`, `cos(2πt)`, `sin(4πt)`, `cos(4πt)` — and
the output is the velocity in one channel, with no activation on top, so the field can be negative. Last recorded
2026-11-05.

## What a run produced

| | value |
|---|---|
| parameters | 5,489 |
| steps | 3,000 (2 images a step, AdamW at 1e-3) |
| loss, first → final | **1.1284 → 0.8364** |
| a predictor of zero scores | **1.0645** (the data's own velocity variance) |
| draws | 16 at 64×64, Euler 32 |
| draws, mean / std | **+0.477 / 1.012** |
| data, mean / std | +0.398 / 0.226 |

![loss](../../../results/diffusion-unet/trained/loss.svg)

![16 draws](../../../results/diffusion-unet/trained/samples.svg)

![drawn against data](../../../results/diffusion-unet/trained/stats.svg)

## What that means

**The gates are exact and the training is not good enough to draw flowers.** The loss falls 1.1284 → 0.8364, 21%
below a predictor of zero, so the model learns part of the velocity field; but the 16 draws are noise. Their spread
(1.012) is the noise's (1.003) rather than the data's (0.226): the sampler barely contracts.

## The measured variants, and what they taught

Every one of these is a run on this machine, with the same data and the same gates; the number is the final loss on
the same scale.

| variant | final loss | what it showed |
|---|---|---|
| five channels in, one out, group norms, 1×1 head (this record) | **0.8364** | the shape the deliverable asks for, and the best stable loss |
| the same without the group norms | 0.90 at 600 steps | the norms are not the whole story |
| the same with an outer skip (`v = head + x`) | diverged (std 2.68 = e¹) | the field at `t = 0` is `-x`, not a small correction: a residual there makes the head learn twice the input's size |
| four convs, no norms, no skip | 0.18 at 3,000 steps | the best loss of all, and it proves the stack trains |
| a single 1×1 convolution, five inputs, **linear** | **0.086** at 300 steps | the marginal conditional field is linear; a linear model finds it, the deep stack does not |
| the same, learning rate 1e-2 | diverged after ~1,000 steps | the rate, not the steps |
| the same as this record, learning rate 3e-3 | 0.81 at 6,000 steps, flat | the rate again: 1e-3 is the stable one |

Two bugs were found by those runs and fixed before this record:

- **the parameter init was scrambling the weights.** `init_arr` zeroed and one-filled hardcoded offsets from the
  first (two-channel) layout, so every run before it started with corrupted weights: the head's rows zeroed and
  unrelated cells set to one. The offsets now match the model's forward order, and the biases start at zero.
- **the draws were halved and scrambled** by reading the sampler's state in pairs. Fixed; the pictures before that
  were wrong for two reasons, and only one was the model.

## The gates that guard it

| gate | result |
|---|---|
| `reference/test_unet_flowers.py` (NumPy, whole forward, two images) | **1.01e-06**, gate 1e-05 |
| `reference/compare_unet_flowers_torch.py` (PyTorch, whole forward) | **1.13e-06**, gate 1e-05 + 1e-6·\|v\| |
| `reference/test_unet_flowers_checkpoint.py` (save, resume, same weights) | **bit-identical**, 5,489 weights, 65,876-byte file |

The twins read the recorded probe: `train.bend --probe --steps 0` prints the flat weights and one whole forward at a
fixed input, and each twin rebuilds it from those weights, so neither side's initialisation is trusted. The checkpoint
gate is a 40-step run against 20 steps saved and 20 resumed, with a 39-step run as the can-fail.

The forward tolerance is `1e-5`, not the gate example's `1e-6`: the measured agreement on these 8,192 cells is
`1.01e-06` with two group norms at `eps = 1e-5` (a norm divides small variances up, which amplifies float32
differences), so the gate sits about ten times above the measurement.

## Reproduce

    BEND_MEM=9G scripts/bend-guard.sh bend -o .scratch/unet_train examples/diffusion/unet/train.bend
    .scratch/unet_train --steps 3000 --save /tmp/unet.bin    # 3,000 steps and 16 draws, ~10 minutes
    reference/.venv/bin/python reference/unet_flowers_results.py <log>
    reference/.venv/bin/python reference/test_unet_flowers.py
    reference/.venv/bin/python reference/test_unet_flowers_checkpoint.py
    ../bendtorch/reference/.venv/bin/python reference/compare_unet_flowers_torch.py

The compile is the expensive part (9 GB under the guard); the step is 0.2 s. `--steps N`, `--start N`, `--save PATH`,
`--resume PATH`, `--no-sample` and `--probe` work as in `flow_matching`.

## What is left

The learning, and it is measured: a **learned linear bypass** (a 5→1 linear map added to the head's output) should
reproduce the linear model's 0.086 and let the body improve on it, because that is the only model that fits the
marginal field this input allows. After that, 32 channels, 8 images a step, and 20,000 steps. `ISSUE-065` carries the
list.

## Why there is no report card

`reference/experiment.py` tasks carry a reference arm and a gate on a held-out metric. This example has neither: the
only held-out signal would be legibility on unseen flowers, and there is no reference run to compare a training loss
against. A task here would have to invent both, so the record keeps the gates it can defend.
