# Rectified flow at 64x64 on Oxford Flowers

A flow model in Bend on BendTorch: noise to a flower in 32 Euler steps, one
row of 64 pixels per token, so the image is read in raster order and nothing
indexes backwards.

```
make doctor                       # what this machine can compile
make flow64 PRESET=medium         # build inside a cgroup, run, report
```

## The data

512 flowers at 64x64, one grayscale byte per pixel, made from the pinned Oxford
Flowers parquet by `reference/flowers_export.py`:

| file | bytes | sha256 |
|---|---|---|
| `flowers64-512-gray.u8` | 2,097,152 | `6b9c4c66…` |
| `flowers256-512-gray.u8` | 33,554,432 | `c58eab33…` |

The 64x64 RGB export the first version read is 12,288 bytes per image; the
tokeniser takes one byte per pixel, so it was reading a scrambled raster until
the grayscale conversion. `papers/TEM/data.py` is where the parquet's hash and
the reduction are recorded.

The directory is `MLBEND_DATA` when the experiment runner sets it
(`reference/.cache/datasets/flowers/`) and `~/.cache/bend-lab/flowers`
otherwise, so the program is not tied to one home directory. The task
`flowers-flow` pins both exports by sha256 and refuses a different file.

## The model

`config.bend` fixes the shapes before the compile, because a Bend program's
sizes are type-level constants: images per step, width, resolution, steps and
Euler count. `scripts/flow-config.py` writes it for a named preset, and
`scripts/system-profile.py` says which presets the machine can pay for.

| preset | images/step | width | rows | compile |
|---|---|---|---|---|
| `tiny` | 1 | 64 | 64 | 926 MB, 19 s |
| `medium` | 2 | 128 | 128 | 926 MB, 19 s |
| `full` | 8 | 128 | 512 | 926 MB, 19 s |
| `hi` | 1 | 128 | 256 (256x256) | 940 MB, 18 s |

The compile number does not move with the model, which is the finding recorded
in `docs/NOTES.md`: a `match` on three values cost 4.8 GB of codegen where the
same function matching two costs 6 MB. `scripts/bend-guard.sh` runs every build
in a memory cgroup with swap off, so a runaway compile is killed instead of
taking the desktop with it.

## The gates

Two references cover this example. The task's gate is measured: `params` is
exact and the held-out legibility is within 0.05 of 87% of the ceiling.
Beside it, `reference/twin_flow64.py` is a NumPy twin of the model and the
sampler, and `reference/test_flow64.py` compares it elementwise with
`reference/probe_flow64.bend`: the frozen model's own `Layer.dump`, one forward
at t = 0.5, one Euler step at t = 0, and the 8-, 16- and 32-step endpoints.

| gate | measured maxabs |
|---|---|
| the first two tokens' features (pixels, position, time) | 1.19e-07 |
| the forward at t = 0.5 | 2.68e-07 |
| one Euler step | 3.58e-07 |
| 8 / 16 / 32 steps | 3.58e-07 / 4.77e-07 / 4.77e-07 |

Every gate sits at 1e-5, about 20x above its measurement: the deltas are
float32 summation order and `F32.show`'s round trip, not a different
computation. `bench/flow64_first` is the same model in a 4-second run (52 s
with the build) and `make matrix` records it as the `flow64-first` cell, so the
forward or the sampler cannot drift from the twin without a cell failing.

## Measured

All runs: 2000 steps, 32 Euler steps, 28 cores. `legibility` is the best
absolute correlation of a sample with any training flower; random images score
0.048.

| model | attention | final loss | legibility |
|---|---|---|---|
| 1 block, 128 wide | causal | 0.328 | 0.419 |
| 1 block, 128 wide | full | 0.328 | 0.427 |
| 3 blocks, 128 wide | causal | 0.325 | **0.450** |
| 3 blocks, 128 wide | full | **0.316** | 0.418 |
| 3 blocks, 128 wide, 3,874 images | causal | 0.289 | trained 0.536 / held out 0.490 |

Full attention is the right layer for an image: with row-tokens, causal
attention lets row 40 see rows 0..40 only. At one block it is worth 0.008
legibility for 2.2x the time, so the restriction was not what limited the
model; depth was, at 0.031. At three blocks full attention gives the best loss
of the four and the lowest legibility of the pair, the same inversion the
2-image and 8-image runs showed. A training-set legibility proxy cannot rank
models, which is what the holdout below is for.

Samples are blurred flowers: the colour and the layout are right, the detail is
not. That is what a 1-3 block model at 64 tokens on 512 images should look like,
and it is not comparable to a paper's FID.

## What it learned, and what it memorised

Legibility is a max over a reference set, so it must be read as a fraction of
what a **real** unseen flower scores in the same comparison. On the 3,874/102
split (one held-out image per class):

```
ceiling   real held-out flowers vs the training set   0.562
floor     noise                 vs the training set   0.046
3,874-images model   trained 0.536   held out 0.490   87% of the ceiling
512-images model     trained 0.450   held out 0.414   73% of the ceiling
```

Two things follow. The corpus was a constraint: 512 to 3,874 images moves unseen
flowers from 73% to 87% of the ceiling, for 0.046 of memorisation gap. And the
earlier "held out 0.227, gap +0.301" was measured against 12 images of one class
and is not comparable with these; it made the model look far more memorising than
it is.

The 13% that remains is the next question, and it is not answered by more of the
same data.

## Not done

- **A per-class held-out split.** Every run above trained on all 512 images, so
  legibility measures memory of the batch. The parquet is grouped by class (the
  first 512 are classes 0..11), so "the last 12" is one class, not a random
  sample; the 500/12 split measures unseen images of a seen class and is
  recorded as such. A proper split takes one image per class and needs a
  re-export.
- **256x256.** It builds (940 MB) but the builders advance on the literal
  `64n`, so a 256 preset needs a builder that never names the resolution.
- **Full attention in the GPU lane.** `FullAttention` is in `torch/main.bend`;
  the device lane still has the causal-only `Attention`.
