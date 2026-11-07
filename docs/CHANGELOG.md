# Changelog

What changed for the people who run the lab. Internal refactors are not listed.

## Unreleased

### The UNet draws flowers: a third level, the full corpus, and an objective that was the wrong one

- **`examples/diffusion/unet/{model,train,config}.bend`**: the gate example's U with **three levels** (64 -> 32 -> 16 ->
  8 -> 16 -> 32 -> 64), 64 channels each, residual skips at 32x32 and 16x16, a linear bypass from the input, and no
  normalisations. Trained on the **7,067** flowers of the training split (not the 512 subset) with rectified flow whose
  **target is the clean image**, `x0 = data`, and a sampler that converts, `v = (x0hat - x)/(1-t)`.
- **What it reaches**: 3,600 steps, loss **0.5542 -> 0.0241** (a predictor of the data's mean scores 0.0539); 64 draws
  that are **flowers**; legibility **0.545** against a held-out ceiling of 0.562 - 96.6% of the span above the noise
  floor - and **0.438 held out** on the 102 flowers the model never saw, 75.5% of the span. The first record of this
  example was 0.259 / 0.206 and its draws were speckle.
- **Four levers, each measured**: the objective (v -> x0: 0.274 -> 0.426); the third level (+11.7 points of span on
  3,874 images); the full corpus (+0.0 points on two levels, +4.6 on three); both together (96.6% / 75.5% of the span).
  Group norms and batch size were measured and rejected earlier: 0.233 with norms against 0.274 without, and a batch of
  26 bought with resolution moved the held-out span by +0.5.
- **New script**: `reference/unet_flowers_eval.py` prints legibility, held-out legibility, the ceiling, the noise floor
  and the draws' radial spectrum, using `reference/flowers_report.py`'s `legibility` so the report and the
  `flowers-flow` task cannot disagree.
- **Gates**: `reference/test_unet_flowers.py` (NumPy, whole forward, **2.38e-06** against 1e-5),
  `reference/compare_unet_flowers_torch.py` (PyTorch, **3.93e-06**, 0.36x the allowance),
  `reference/test_unet_flowers_checkpoint.py` (40 steps == 20 + 20 resumed, **bit-identical**, 150,727 weights).
- **The bugs that cost the earlier runs**: the training pair builder and the sampler built the five input channels in
  opposite orders; `init_arr` zeroed and one-filled hardcoded offsets from the first two-channel layout, scrambling
  every run's weights; a batch that does not divide the image count reads past the export; and a run shorter than one
  pass trains on a class-ordered prefix, because the parquet is grouped by label. All four are fixed or gated.
