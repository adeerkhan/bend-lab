# Changelog

What changed for the people who run the lab. Internal refactors are not listed.

## Unreleased

### The UNet draws flowers: the objective was the wrong one

- **`examples/diffusion/unet/{model,train,config}.bend`**: the gate example's two levels, 64 -> 32 -> 16 -> 32 -> 64, at
  **64 channels**, with no normalisations, a skip at 32x32 and a learned linear bypass from the input. Trained on the
  **3,874** flowers of the training split (not the 512 subset) with rectified flow whose **target is the clean image**
  instead of the velocity; the sampler converts, `v = (x0hat - x)/(1-t)`. Flags as in `flow_matching`: `--steps`,
  `--start`, `--save`, `--resume`, `--no-sample`, `--probe`.
- **Why the target changed**: `v = data - noise` is 95% noise (`E[(d-n)^2] = 1.053` against the image's `E[x0^2] =
  0.203`), so a model halves the loss by learning the noise statistics alone - which every earlier run of this example
  did, and none of them drew anything but speckle. With `x0`, every gradient is about the flower.
- **What it reaches**: 4,000 steps take the loss **0.6551 -> 0.0390** (a predictor of the data's mean scores 0.0534);
  the 64 draws are **blurred flowers**; legibility **0.426** against a held-out ceiling of 0.562, and **0.330** on the
  102 flowers the model never saw - 73.1% and 54.2% of the span above the noise floor. The previous record was 0.259 /
  0.206, and its draws were speckle.
- **Ruled out, each measured**: group norms (0.233 with them against 0.274 without, on more steps); batch size 2 -> 8
  -> 26 bought with resolution (+10.5 / +4.4 points of span, saturating, still speckle at every resolution); capacity
  76,871 -> 301,191 parameters (+1.3 points).
- **New script**: `reference/unet_flowers_eval.py` prints legibility, held-out legibility, the ceiling, the noise floor
  and the draws' radial spectrum, using `reference/flowers_report.py`'s `legibility` so the report and the
  `flowers-flow` task cannot disagree.
- **Gates**: `reference/test_unet_flowers.py` (NumPy, whole forward, **9.54e-07** against 1e-5),
  `reference/compare_unet_flowers_torch.py` (PyTorch, **1.61e-06**, 0.16x the allowance),
  `reference/test_unet_flowers_checkpoint.py` (40 steps == 20 + 20 resumed, **bit-identical**, 76,871 weights).
- **The bugs that cost the earlier runs**: the training pair builder and the sampler built the five input channels in
  opposite orders, so the field was learned on one permutation and applied to another; `init_arr` zeroed and one-filled
  hardcoded offsets from the first two-channel layout, scrambling every run's weights. Both are fixed, and the init is
  now checked before every run: `--probe --steps 0` must show every bias at 0, every gain at 1, and no stray 0 or 1.
