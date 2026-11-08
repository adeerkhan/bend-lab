# Changelog

What changed for the people who run the lab. Internal refactors are not listed.

## Unreleased

### GLM-5.3-Flash-from-scratch: Stage A, and the linear-attention forward

- **`reference/glm53_flash/`**: the educational 25.7M model vendored at commit `96d9c7c`, MIT,
  with `VENDORED.md` stating that it is not the 320B release. Its 12 tests pass under
  `reference/.venv-rf` (torch 2.11.0+cu128).
- **Stage A** (`examples/language/glm53_flash/model_a.bend`): a 74,144-parameter dense-attention
  harness, width 32, four blocks, four heads of 8, 64 positions, SwiGLU 128, tied head, built
  from BendTorch's existing `Embedding`, `RMSNorm`, `RoPE`, `Attention`, `SiLU`, `Tie` and
  `CrossEntropy`. `probe_a.bend` prints whole logits and all 74,144 parameter gradients.
- **The gate** (`reference/test_glm53_layers.py`, six checks): logits **2.38e-07**, gradients
  **2.24e-08**, loss 5.5654626, causality (first half exact, future changed), the Stage B
  linear-attention forward **9.77e-04** on values to 4,096 (relative < 1e-6), and RoPE
  **2.44e-04** on values to 2,000 (relative < 1e-7). The RoPE bridge is explicit: the reference's
  interleaved pairs, BendTorch's half-split pairs, and the q/k row permutation
  `[0,4,1,5,2,6,3,7]`.
- **Stage B forward** (`lin_core.bend`): ELU+1, the causal prefix of `k` and `k⊗v`, the
  numerator, denominator and 1e-6 clamp, in list form. **Not built:** the linear-attention
  backward, Stage C, the MoE, the four-stream HyperConnection, RL and the 25.7M run.

### The rectified-flow port gets a whole-array gate, and the bug it caught

- **`reference/test_rectified_flow_layers.py`** (in `make check`): builds
  `reference/probe_rectified_flow_layers.bend` once and compares fourteen whole
  outputs with a NumPy twin — `Down2` forward and its exact-inverse backward
  (4,096 values each), the 2x2 max pool and its arg-max scatter (1,024 and
  4,096), the zero-query XAttnRes mix (1,024), the learned-query mix on equal
  and unequal keys (1,024 each), its saved weights (256), its five gradients
  (`dvh`, `dxc`, `dkh`, `dkx`, `dq`), and `Softmax(2,2)`'s axis. The
  permutation, pool, scatter and zero-query gates are exact (0.0); the learned
  mix and its backward are under 5e-3.
- **The mix backward**: `dvh`, `dxc`, `dkh`, `dkx` and the query gradient,
  with BendTorch's own `Softmax` backward on the saved weights. The
  query-gradient gate caught a missing recursion in `dq_sum` while it was
  being written (the sum stopped after row 0; max |diff| 689).
- **The bug it caught**: `dot8` indexed the 1x8 query with the row offset, so
  every row after the first read past the query's end, scored 0 and silently
  averaged 50/50. The row-only probes had passed because row 0 reads offset 0;
  the `kx = 2 * kh` whole-array gate failed immediately. `dot8` now reads the
  query at 0..7 and moves only the key/value row with `off`.
- **`examples/diffusion/rectified_flow/README.md`**: the 16x16, dim-8,
  one-hour target, the layer/probe inventory, the mix formula, and what is
  next (XAttn wrapper, memory-KV attention, DAG, `rf16`).

### Lab survey: the build environment, the checkpoint default, portable caches

- **`reference/check_all.py`** supplies the local `libLLVM` path to the build
  checks only, so `make check` builds from a fresh shell; `--verdict` still
  runs without it, as Lean needs its own clang.
- **`reference/rectified_flow_compare.py`**: `--sample` without `--checkpoint`
  loads the newest checkpoint under `--checkpoints`; `--resume` is now an
  honest warm start (the vendored `Trainer` has no step resume, and the old
  `checkpoint.latest.pt` was never written).
- **`examples/diffusion/{unet,flowers_flow}`** fall back to
  `$HOME/.cache/bend-lab` instead of one machine's absolute path.
- **Docs and links**: stale private-ledger and absolute-home references removed,
  three broken Markdown links fixed, the README's `make check` count and
  two-level UNet sentence brought up to date, `.understand-anything/tmp/`
  ignored, and `results/lucidrains-rectified-flow/` added to `.gitignore`.


### The UNet's fifth level: better flower identity, no more texture

- **`examples/diffusion/unet/{model,train,config}.bend`**: a **fifth level**, 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 4 -> 8
  -> 16 -> 32 -> 64, 64 channels, no normalisations, the clean image as the flow target, 7,067 flowers. 298,439
  parameters, 3,600 steps; loss **15.3262 -> 0.0259** (the first value is the untrained head on the first batch; step 1
  is 0.1343).
- **What it reaches**: 64 draws at Euler 32, mean/spread **+0.325 / 0.179** (the data's +0.384 / 0.232); **held-out
  legibility 0.547** against a ceiling of 0.574, **94.8%** of the span above the noise floor. The four-level record,
  re-scored today with the same evaluator, is 0.605 / **0.501** / 85.8% (published 0.498 / 85.4%), so the fifth level is
  +4.6 points of held-out legibility and +9.0 points of span. The trained legibility is 0.633, above the ceiling:
  memorisation, stated as such.
- **What it did not do**: the gain is global identity, not texture. The raw draws are lower contrast than the
  four-level's (+0.325 / 0.179 against +0.381 / 0.238) and the top spectrum band is 0.034 against 0.039 (the data's is
  0.137). On the same 8 draws, 128 Euler steps give spread 0.174 and top band 0.027, so the sampler is not the cause;
  **32 steps stays**. The depth line stops here; the next lever is detail and contrast, not a sixth level.
- **The draw-count trap**: a max-over-correlations statistic depends on how many draws are averaged. The five-level's
  first 8 draws score 0.575 held out (100.3% of the span) against 0.547 for all 64 (94.8%); an earlier 128-step run on
  the four-level looked like +3 points for the same reason. Comparisons hold the draw count fixed.
- **Gates**: the twins cover nine convolutions and 22 weight blocks; NumPy **1.621e-05** at 2e-05 and torch
  **2.098e-05**, 0.83x its allowance; the checkpoint resumes **bit-identical** at 298,439 weights.
- **The trap paid in the twins**: the new level outputs were first named `b2`/`b3`, which are the biases of the same
  names; the levels are `l4`/`l2` now.

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
