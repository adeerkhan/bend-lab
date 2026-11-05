# The trained 64×64 UNet denoiser on flowers

`model.bend` is the gate example's U trained instead of checked: 64 → 32 → 16 → 32 → 64, 16 channels at each level,
a group norm before every activation, an `up2` upsample at each level, the 32×32 skip, a 1×1 convolution as the head,
and a **learned linear bypass** from the input beside the U. The objective is rectified flow on the 512 flowers of
`flowers64-512-gray.u8` (64×64, one grayscale byte a pixel):

    x_t = noise*(1-t) + data*t     target v = data - noise     MSE
    sample: x <- x + v(x,t)*dt, Euler over linspace(0,1,steps+1)

The input has five channels — the image and four time features, `sin(2πt)`, `cos(2πt)`, `sin(4πt)`, `cos(4πt)` — and
the output is the velocity in one channel, with no activation on top, so the field can be negative. Last recorded
2026-11-05.

## What a run produced

| | value |
|---|---|
| parameters | 5,495 |
| steps | 6,000 (2 images a step, AdamW at 1e-3) |
| loss, first → final | **1.2687 → 0.1652** |
| a predictor of zero scores | **1.0645** (the data's own velocity variance) |
| draws | 16 at 64×64, Euler 32 |
| draws, mean / std | **+0.359 / 0.262** |
| data, mean / std | +0.398 / 0.226 |

![loss](../../../results/diffusion-unet/trained/loss.svg)

![16 draws](../../../results/diffusion-unet/trained/samples.svg)

![drawn against data](../../../results/diffusion-unet/trained/stats.svg)

The loss is **84% below** the zero predictor, and the draws' mean and spread now match the data's (`+0.359 / 0.262`
against `+0.398 / 0.226`). The 16 pictures show structure — light and dark regions in the right proportion — but they
are blurry, not recognisable flowers: 5,495 parameters and 6,000 steps is a small model, and the next levers are in
`ISSUE-065` (32 channels, 8 images a step, 20,000 steps).

## The bug that made the first records noise

The training pairs and the sampler built the five input channels in **opposite orders**: the pair builder consed
`[xt, f1, f2, f3, f4]` and then reversed the whole list, which yields `[f4, f3, f2, f1, xt]` per pixel, while the
sampler built `[x, f1, f2, f3, f4]`. The model was trained on one permutation and sampled with another. Before the fix
the draws' spread stayed at the noise's (`1.01` against the data's `0.226`); after it, `0.262`. Two smaller bugs were
fixed in the same pass: `init_arr` zeroed hardcoded offsets from the first two-channel layout (corrupting the weights),
and the draw loop read the sampler's state in pairs (halving and reordering every picture).

## The gates that guard it

| gate | result |
|---|---|
| `reference/test_unet_flowers.py` (NumPy, whole forward, two images) | **1.01e-06**, gate 1e-05 |
| `reference/compare_unet_flowers_torch.py` (PyTorch, whole forward) | **1.13e-06**, gate 1e-05 + 1e-6·\|v\| |
| `reference/test_unet_flowers_checkpoint.py` (save, resume, same weights) | **bit-identical**, 5,495 weights, 65,948-byte file |

The twins read the recorded probe: `train.bend --probe --steps 0` prints the flat weights and one whole forward at a
fixed input, and each twin rebuilds it from those weights, so neither side's initialisation is trusted. The checkpoint
gate is a 40-step run against 20 steps saved and 20 resumed, with a 39-step run as the can-fail.

The forward tolerance is `1e-5`, not the gate example's `1e-6`: the measured agreement on these 8,192 cells is
`1.01e-06` with group norms at `eps = 1e-5` (a norm divides small variances up, which amplifies float32 differences).

## Reproduce

    BEND_MEM=9G scripts/bend-guard.sh bend -o .scratch/unet_train examples/diffusion/unet/train.bend
    .scratch/unet_train --save /tmp/unet.bin               # 6,000 steps and 16 draws, ~30 minutes
    reference/.venv/bin/python reference/unet_flowers_results.py <log>
    reference/.venv/bin/python reference/test_unet_flowers.py
    reference/.venv/bin/python reference/test_unet_flowers_checkpoint.py
    ../bendtorch/reference/.venv/bin/python reference/compare_unet_flowers_torch.py

The compile is the expensive part (9 GB under the guard); the step is 0.3 s. `--steps N`, `--start N`, `--save PATH`,
`--resume PATH`, `--no-sample` and `--probe` work as in `flow_matching`.

## Why there is no report card

`reference/experiment.py` tasks carry a reference arm and a gate on a held-out metric. This example has neither: the
only held-out signal would be legibility on unseen flowers, and there is no reference run to compare a training loss
against. A task here would have to invent both, so the record keeps the gates it can defend.
