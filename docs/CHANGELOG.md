# Changelog

What changed for the people who run the lab. Internal refactors are not listed.

## Unreleased

### The UNet draws flowers: four levels, and an objective that was the wrong one

- **`examples/diffusion/unet/{model,train,config}.bend`**: the gate example's U with **four levels** (64 -> 32 -> 16 ->
  8 -> 4 -> 8 -> 16 -> 32 -> 64), 64 channels each, residual skips at 32x32 and 16x16, a linear bypass from the input,
  and no normalisations. Trained on the **7,067** flowers of the training split with rectified flow whose **target is
  the clean image**, and a sampler that converts, `v = (x0hat - x)/(1-t)`.
- **What it reaches**: 3,600 steps, loss **1.1831 -> 0.0298** (a predictor of the data's mean scores 0.0539); 64 draws
  whose mean and spread are the data's (+0.379 / 0.246 against +0.384 / 0.232); **held-out legibility 0.498** against a
  ceiling of 0.574, i.e. **85.4%** of the span above the noise floor. The first record of this example was
  0.259 / 0.206 with speckle for draws.
- **Five levers, each measured**: the objective (v -> x0) 0.274 -> 0.426; the third level 0.439 held out; the corpus
  alone **+0.0**; the fourth level 0.439 -> **0.498** (bottleneck 8x8 -> 4x4); a wider stack (CH 96) did not beat it.
  Group norms and batch size were rejected earlier by measurement (0.233 with norms against 0.274 without; a batch of
  26 for +0.5 points of held-out span).
- **Measurement fixed, twice**: the eval used to score against the 3,874-image subset while the model trained on 7,067,
  inflating both the legibility and the ceiling - it now defaults to the full export, prints the reference set's size,
  and says so when the trained legibility passes the ceiling (memorisation). The forward gates moved from 1e-5 to 2e-5
  when the fourth level added two convolutions: measured 9.15e-06 (NumPy) and 1.18e-05 (torch), with the reason in
  each file.
- **New scripts**: `reference/unet_flowers_eval.py` prints legibility, held-out legibility, the ceiling, the noise floor
  and the draws' spectrum through `flowers_report.py`'s `legibility`.
- **The bugs that cost the earlier runs**: the pair builder and the sampler built the five input channels in opposite
  orders; `init_arr` zeroed hardcoded offsets from the first two-channel layout; a batch that does not divide the image
  count reads past the export; and a run shorter than one pass trains on a class-ordered prefix. All four are fixed or
  gated.
