# Notes

Every finding, every measurement and every discarded hypothesis, by date. Read the
relevant entry before changing performance-sensitive code or a law.

## 2026-11-01 — the CIFAR fills: the target, not the input

- **`diffusion-vit` is green again, and the gap was two different things**
  (ISSUE-032, ISSUE-040). The task's gate compared `twin.first_loss()` — one
  timestep for the whole batch, the pre-ISSUE-016 wiring — against a program
  that runs one level per image: 1.0069675 against 1.0091865, 2.1e-3. With the
  gate on `loss_at(0)` the delta was still 1.15e-4, and a 1-second probe
  (`bench/vit_first`: the example's own `init_params` and `do_step(0)`) named
  the second cause: `ab_img`/`te_img` reverse a list that is already ascending,
  so image i is trained at image 63-i's level. Mirroring that in the twin gives
  **6.3e-09**, the first rows match value for value, and `TOL_FIRST` went from
  1e-3 to **1e-5** (tightened, from the measurement). The example is unchanged:
  the reversal is a harmless permutation of which image draws which level, and
  removing it would move the published 0.7875/0.8110 for no gain.
- **`cifar-train` is the first CIFAR task whose loss is a training
  measurement.** `examples/cifar_train` is the same denoiser over the same 256
  images with a level from {16, 32, 48} each step and lr 1e-3, 600 steps:
  `first_loss` **1.0022775**, `final_loss` **0.9960275**, `loss_drop`
  **1.0062749** against the twin's 1.0069602 (delta 6.85e-04, gated at 2e-3
  relative, which is the measured delta and not a widened one). The fall is
  0.6% in 600 steps — a falling curve, not a trained model: ISSUE-036's table
  said a repeating level is what makes a CIFAR loss move, and 0.6% is what the
  scheduled run gives at this budget (the 0.529 in the same table is one fixed
  noise draw, i.e. memorising a single pair, which is not a denoiser).
- **The step split at these sizes** (`bench/plumbing_cifar`, three runs,
  renamed to the keys the stage reader reads): input **41 ms**, forward
  **37**, target **4**, loss + backward + optimizer **45.7**, one step
  **127.7**, sampler 322-327. The floor `cifar_step` puts 807,168 parameters against 256
  rows: 413 GFLOP a forward, 827 with the backward, and the achieved numbers
  are 111% and 180% of those floors (`make experiment TASK=cifar-train`).
- **The first `cifar-train` split was measured on an empty dataset**
  (ISSUE-039). `reference/stages.py` ran the stage program without
  `MLBEND_DATA`, so the probe's `Datasets.csv_of` fell back to the hub, failed
  and returned `Nil{}`; the record then said input `<1 ms` and a step of
  80.7 ms where the probe reads 41 and 126. The cache is now in the stage
  environment and a run that prints `the dataset failed` is refused rather than
  timed. Any task whose stage program reads the cache was exposed to this.
- **The stage reader now speaks the CIFAR probe's keys.** `bench/plumbing_cifar`
  prints `step_in_ms`, `step_tgt_ms` and `forward_ms` (the names
  `reference/stages.py` reads), and the forward is timed on a second
  `params()` (3.1 s, outside every timer) because a parameter record is affine.
  The numbers are the ones item 1 measured under `input`/`target`.

- **`Train.step`'s config overrides `Train.new`'s, and that is the whole of
  ISSUE-037** (upstream: [nuxyel/BendTorch#24](https://github.com/nuxyel/BendTorch/issues/24)).
  `Train` is `Train{p, g, st, p1, p2, loss}` — no config — and `step` derives
  the optimizer from `+cfg` at the call site (`Opt.at(cfg, q1, q2)`,
  `OptCfg.c1/c2`), so the lr `Train.new` got reaches only `OptState.new` and
  `q_start`. `bench/train_cfg`: three `Train`s built at lr 0.001, 0.01 and 0.1
  and all stepped at 0.001 read **6.5033846** three times; stepped at their own
  rates they read 6.5033846 / 1.2917502 / 0.4104244. The issue proposes
  carrying the config in the `Train` and dropping the argument, or refusing a
  step whose lr/b1/b2/wd differ; we would take the first, because the bug is
  two sources for one value, and removing the parameter removes the class.
  Locally `do_step` now takes the config, and the numbers are bit-identical.

- **The fill crosses the version clash.** `bend-ml-tensor-gpu@0.1.5.0` imports
  `bend-ml-tensor-array@0.1.18.0` and this project pins 0.1.14.0; both resolve
  (`bendenv lock`: two `bend-ml-tensor-array` entries, 0.1.14.0 and 0.1.18.0)
  and a Mat of one is not a Mat of the other. What crosses is the return of
  `fill_gauss`: a Base `Array<F32>`, which is no package's type.
  `bench/fill_cross` builds `TA.Mat<2,4>{TG.fill_gauss(1, 8, 3)}` from the
  0.1.14 every example uses, prints the eight values beside `G.gauss_at`,
  multiplies it, and reads **max_diff 2.0861626e-7**. So the blocker the plan
  feared is not a blocker, and the stack bump is not a workaround we need: it
  stays its own change, for the packages that want one version.
- **What the fill is worth, measured** (`bench/plumbing_cifar`, three runs;
  the numbers are one run's, the run-to-run spread is ±15 ms on the step):
  the target's noise was `G.noise_of` (786,432 `gauss_at` plus a list) at
  **29 ms** and `Mat.from_list` at 3; as one fill it is **4-5 ms**. The step
  falls from **143 ms** (426/430/441 for three) to **128** (382/385/393). The
  input is unchanged at **40-41 ms** and the input cells do not move:
  0.5865267, 1.0214669, 0.9998779 at cells 0, 73728, 788479 before and after.
- **The input keeps `gauss_at`, and that is a measured negative.** Noising a
  pixel reads the noise by index, so `input_of` needs a list, not an array.
  Built from the fill (`arr_list_go` over the `Array`'s leaves, 17-28 ms for
  1,048,576 cells: `cap_depth` rounds 786,432 up to 2^20) the input reads
  **70 ms** against **40**; with the list built outside the timer the walk
  alone is 53-58 against 40, i.e. matching two lists per pixel costs more than
  the `gauss_at` it replaces. `Array.to_list` is not the way out either
  (32-34 ms, and its `List<F32>` is a Type that cannot ride in a record with
  `+` reuse).
- **On the CUDA lane the fill is slower, not faster** (`CUDA_HOME` and
  `LD_LIBRARY_PATH` at `~/.local/cuda-nvrtc`, `ldd` resolving
  `libnvrtc.so.13`, `nvidia-smi --query-compute-apps` showing the process at
  104 MiB, so it executes there): the first fill is **33-41 ms**, the warm one
  17, and the step **159-165 ms** against 128 on the CPU threads. One bang
  over 786,432 values is kernel work too small to pay the launch, and the
  values still have to come back to the host for `Mat.from_list` and the row
  walk. This is ISSUE-018's shape again: the device lane is not where this
  step's seconds are.
- **The gate held without a twin change**: `first_loss` 1.0014538 against the
  twin's 1.0014538764 (7.65e-08) and `final_loss` 1.0163177, so the fill
  mirrored nothing and no tolerance moved. `reference/test_cifar.py` prints
  `ok the first loss agrees: 1.0014538 vs 1.0014538764`.
- **The lock was already stale.** `bendenv lock --check` on the committed tree
  said "the lock is not current" before this change: four local packages'
  content hashes and `bend-ml-tensor-fork` (0.1.4.0, from `*`) had drifted. The
  relock here carries that, and tensor-fork moves to 0.1.5.0; no import in the
  tree uses it.

## 2026-10-29 (later) — the ViT reaches the gate, and the card is not the bottleneck

- **M1 is met.** The ViT denoiser, with **one timestep per image** (the
  reference's training distribution) and the Weyl schedule kept, reaches
  **0.783 DDIM / 0.807 ancestral** nearest-digit legibility in 3000 steps at
  batch 64, nearest-digit MSE 0.0618 / 0.0569, cloud mean/std
  **+0.3007 / 0.3450** against the data's 0.3076 / 0.3767. The torch arm of the
  comparison, same architecture and budget, is **0.790 / 0.779**; the lucidrains
  UNet is 0.932 / 0.940. The gap that remains is the convolution, not the
  training.
- **The last variable was the loop, and the isolation table said so.** One
  timestep per batch: 0.359 / 0.344 (deterministic), 0.392 / 0.441 (random),
  0.409 (Weyl). One timestep per image: 0.770 / 0.778. The first attempt at that
  table was **miswired** — the "random" arm drew a timestep *per image* — which
  is why ISSUE-016's first version is in the ledger with a correction above the
  table. Read the arm's code before its number.
- **The gate moved into `make check`**: `reference/test_diffusion_vit.py`, six
  measured checks (first loss tol 1e-3 from the 1.27e-4 delta between the twin's
  `loss_at(0)` and the run; forward 1.19e-07; step 4.77e-07; DDIM 6.38e-06;
  eta = 0 exactly 0; legibility ≥ 0.75; cloud within 1e-6 of [0,1]). **92/92.**
- **`Nat` fail-stops past 2^48−1; it does not wrap** (ISSUE-017). The per-image
  Weyl product `(i + s·64)·a` crossed it at `s ≈ 1653` and killed a 3000-step run
  with `bend: a Nat past the largest immediate 2^48-1` and no location. `U32` is
  the right type for a mixer, and the wrap is bit-free: `(x − k·2^32) >> 26`
  differs from `x >> 26` by a multiple of 64.
- **The GPU lane executes, and buys nothing here** (ISSUE-018). `bendtorch@1.1.1.0`
  + `bendtorch-gpu@1.1.1.0`, eight `LinearNBF` and one `F.Attention`: the binary
  links `libnvrtc`, `nvidia-smi` shows the process with **104 MiB on the card**
  during the run, the frozen values are **bit-identical** to the CPU lane — and
  the wall time is **unchanged** (5m44 both ways). The reason is where the time
  goes: seven `init_params()` calls, 1024 rows built per forward, and the list
  conversions are host work the card cannot touch. **A training step's seconds
  need splitting before any CIFAR decision**: plumbing against matmul.
- **Flow matching is much smaller than DDPM.** `reference/flow_probe.py`
  (rectified flow on two moons, 2×128×128×2, batch 256, 6000 steps, Euler 32):
  loss **2.1253 → 0.8662**, samples mean [-0.111, 0.150] / std [0.546, 0.745]
  against the data's [-0.002, -0.000] / [0.583, 0.741], 465/512 inside the data's
  box. The method is four lines — `x_t = noise·(1−t) + data·t`, `v = data −
  noise`, MSE, Euler — with no schedule, posterior, sigma, eta, clamp or rungs.
- **`#1894` closed as a known limit**, not a bug: the JS run path is bounded by
  the host engine's call stack, the native build is the path for data-sized
  recursion, and replacing host recursion in the JS emitter is not planned. The
  house rule stands: data-sized recursion runs from a native binary, under
  `setsid`, with `stdbuf`.
- **Two claims of mine were wrong and are corrected where they were made:**
  ISSUE-014 (the GPU lane is a separate package, not absent) and ISSUE-018's
  first version ("linking is not execution" — the witness says it is). Both
  corrections are above the original text, not replacements of it.

## 2026-10-28 — the ViT denoiser: the residual, causality, and the gates

- **The twin disagreed with Bend at O(1), and the ladder named the operation.**
  With Bend's own `Layer.dump` fed to the twin, `eps` differed by maxabs 0.343
  (1.41 relative). One probe printed five checkpoints and the twin printed the
  same: the staging row, `emb0`, `ln0` (block 1's norm) and `att0` (block 1's
  attention output) were **identical to the last float32**, and `blk0` (the
  block's output) was not. So the staging, the weights, the traversal, the first
  linear, the norm, the scores, the causal mask and the output projection were
  all already right, and the disagreement began in the block's residual
  (ISSUE-010).
- **What it was.** `AttBlk` was `Seq(LayerNorm, Residual(Att))`. `Seq` runs left
  then right, so that computes `ln(x) + att(ln(x))` — the residual taken on the
  *normalized* value. The intended block is `x + att(ln(x))`, which is what the
  twin and the torch probe implement. Four probe outputs were spent ruling out
  the attention (six Q/K/V permutations, all six `wo` positions, causal and
  not, three scales); the answer was in the wiring, not the arithmetic.
  Feeding the dump through Bend's own wiring reproduces `blk0` to **7.5e-08**;
  the pre-LN wiring is 0.64 away. The block is now `Residual(x, Seq(ln, f))` on
  both halves, parameter order unchanged, count still 100,736.
- **Why it mattered beyond the gate.** The stream was re-normalized twice per
  block and kept no identity path: every half threw away the incoming residual
  and passed on only its norm. That is not the architecture the torch probe
  measured, and it is what "trains to a plateau" looks like from the inside.
- **BendTorch's only attention is causal** (ISSUE-011). `qrow` builds `i + 1`
  scores for query `i`, `prow` softmaxes exactly those, and the layer takes no
  mask — the three trailing arguments are erased shape facts. A causal twin
  reproduces Bend's `att0` exactly; a bidirectional one cannot. So the 16 patch
  tokens are attended as a sequence (token 0 sees only itself, token 15 sees
  all), the torch arm of the comparison runs the same mask to be the same
  architecture, and every earlier ViT number in this file (0.809, 0.798, 0.778)
  is **bidirectional** and not comparable: the causal numbers are 0.782 (lr
  3e-3, batch 64), 0.779 ancestral, and 0.721 at the MLP's lr 1e-2.
- **The twin's cloud was one image wide.** `frozen_eps` reshaped the flat 4096
  into (64, 64) before patching, but `ddim_cloud` and the gate's `start` called
  `patch()` on the flat list, which reads the first 64 numbers as one image. The
  DDIM gate would have compared 64 values with 4096 and "passed" only because
  the shapes never met. Both now reshape first. Same class as the moons
  interleaving (2026-10-27): row 0 agrees, the rest does not.
- **The gates, measured** (`reference/vit_gate.py`, float32, different
  summation orders over 64 terms): frozen `eps` **maxabs 1.19e-07**, one
  ancestral step **4.77e-07**, the 16-step DDIM cloud **6.38e-06**, eta = 0
  moves the cloud by exactly **0** (the sampler's draw is inert). The M2 gate
  file is `reference/test_diffusion_vit.py` — the twin's first loss against the
  run's, those three, the eta invariance and legibility >= 0.75 from the
  recorded run.
- **The learning rate is not shared with the MLP** (ISSUE-012). Same architecture,
  same 3000 steps, torch: 1e-2 with batch 64 → 0.721, 3e-3 with batch 64 →
  **0.782**, 3e-3 with batch 128 → 0.774. The example runs 3e-3.
- **The minibatch was static, not rotating** (ISSUE-013). `step_in` took the
  first 64 digits every step, so Bend trained on 1/16 of the data while the
  torch arm sampled all 1024 — and the loss said so: **0.00546 at step 275**
  against torch's **0.0608 after 3000**. The batch is now chunk `s mod 16` of
  the sixteen 64-digit chunks, which is what "deterministic minibatch" was
  meant to be: reproducible and covering every digit. The first loss is
  unchanged, because step 0 is chunk 0 — the 64 digits the twin's
  `first_loss` already used.
- **Correction, same day: the gap is one t per batch, not the schedule.** The
  first reading of the isolation table blamed the periodic `t = 1 + s mod 64`.
  The PyTorch arm that "randomized t" was miswired — it drew a t *per image*.
  Written correctly, one t per batch scores 0.392/0.441 whether the level is
  deterministic, random or Weyl; one t per image scores 0.770/0.778. So the fix
  is per-image timesteps (a per-image time embedding and per-image `a`/`b` in
  the input mix), not randomness. Bend's Weyl run measures 0.409, with the
  schedule randomized and the loop unchanged — the number the corrected table
  predicts. `ISSUE-016` in the local issue ledger carries the corrected table and says
  plainly that the first version of the entry was wrong.
- **The loss curve is a sawtooth, and that is the timestep, not the optimiser.**
  The example draws **one t for the whole 64-digit batch** (`t = 1 + s mod 64`),
  so the printed loss is the loss at *that* t: ~1.0 at low t, where x_t is
  almost clean and the noise cannot be recovered from the input (the best
  predictor of ε is 0, so the loss is E[ε²] = 1), falling to ~0.002 at high t.
  Three runs at lr 1e-3, 2e-3 and 3e-3 spike at the *same* printed steps —
  which is the tell: it is the data's timestep, not divergence. It also means
  the per-step loss is not comparable between Bend (one t per batch) and the
  torch arm (a t per image, so its printed loss is an average over t); the
  legibility of the cloud is the comparison, not the loss.
- **A 64-wide torch model wants few threads.** The same 3000-step run took
  **21 minutes on 16 threads and 4 minutes on 3** — the matmuls are too small
  to pay for the synchronisation. `THREADS` is an environment variable in both
  torch arms now.
- **The three arms, on the same 1024 digits, the same starting cloud
  (`noise_of(2000000, 4096)`) and the same step counts**, nearest-digit
  correlation and nearest-digit MSE:

  | arm | DDIM 16 | ancestral 64 | DDIM MSE | ancestral MSE |
  |---|---|---|---|---|
  | torch-ViT (causal, 1 head) | 0.786 | 0.779 | 0.0602 | 0.0633 |
  | lucidrains UNet | 0.932 | 0.940 | 0.0202 | 0.0183 |

  The Bend arm's row is added by the run below; `reference/compare_vit.py`
  writes all three to `results/compare/`. (With the arms' own starting
  draws the two rows were 0.782/0.779 and 0.926/0.938; on the one cloud above
  they are these.) **PyTorch on a 64-wide model wants few threads:** the same
  run took 21 minutes and 5 GB of thread-time at 16 threads and 4 minutes at
  3 — the matmuls are too small to pay for the synchronisation.
- **The 0.809/0.798/0.778 numbers, re-read.** They were measured with a
  bidirectional mask, four (or one) heads and a learned position table. The
  causal, one-head, fixed-position model that Bend can express is the 0.782
  row above — a 0.03 difference, not a 0.4 one, and the gap to the reference
  UNet (0.926) is the architecture (convolutions) rather than the training.

## 2026-10-27 (later) — the digits denoiser, rebuilt against the reference

- **Why a reference at all.** The NumPy twin agrees with the Bend example *by
  construction* — it is a mirror I wrote — so it cannot tell me whether the
  example is right, only whether the two agree. `lucidrains/denoising-diffusion-pytorch`
  is what the field actually runs, so it became the second opinion. Vendored
  verbatim (`reference/lucidrains/`, commit `c8155b1d`, MIT, sha256 manifest,
  its own torch venv; nothing in `make check` imports it). Its `__init__`
  imports every submodule, so "the necessary files" is the whole package.
- **The comparison, on the same task** (1024 optdigits, T = 64, noise
  prediction, linear schedule): the reference reaches **0.106 in 300 steps**
  and **0.078 in 3000** (batch 128), and its sampled clouds sit at
  **0.2774/0.3400** (DDIM) and **0.2838/0.3515** (ancestral) with **4096/4096**
  pixels in [0,1], against the data's 0.3076/0.3767. Bend, at the time:
  **1.0258 after 3000 full-batch steps**, cloud mean 1.2e17.
- **The cause was the initialiser, and the ablation says so.** Our exact MLP,
  optimizer and data, one factor at a time (`reference/digits_ablation.py`):
  the example's init `[-0.2, +0.5]` ends at **1.3766/1.1827**, PyTorch's
  zero-mean init at **0.1455** — while [0,1] vs [−1,1] input range and our
  schedule vs the reference's move it only between 0.1455 and 0.1787. The
  positive mean (+0.100, not zero) adds `72 × 0.100 × 0.329 ≈ +2.4` to every
  pre-activation of the 72-wide first layer, and each following layer multiplies
  that common mode by `fan_in × 0.100 = 6.4`: measured z1 +2.36 → z2 +13.8 →
  output +88.4 rms, against a target of rms 1.0. That is the first loss,
  **8062.6895**, against the reference's 0.963. Fixed: a 97-value hash on [−1, 1), scaled by
  `1/sqrt(fan_in)` at the row builder — the first loss became **1.0039**
  (against the twin's 1.003894329, a delta of 2.9e-08) and the final **0.0417**.
- **Learning was half of it; the sampler was the other half.** After the fix
  the loss fell to **0.0417** (better than the reference's 0.078) but the cloud
  was still **std 35.8**, mean −1.39. The reference clamps the predicted
  start to its data range before forming the posterior mean, and re-derives ε̂
  from the clamped start for DDIM; we did neither. Isolated on one trained MLP
  (`reference/digits_sample_probe.py`): DDPM **32.79 → 0.3005**, DDIM
  **17.77 → 0.3614**, pixels in [0,1] **123/4096 → 4096/4096**. Both steps now
  reconstruct `x0 = (x − √(1−ᾱ)ε̂)/√ᾱ`, clamp to [0,1], and form the mean from
  it. The Bend clouds are now **0.3846/0.3549** (ancestral) and
  **0.4153/0.4059** (DDIM), 4096/4096 in range — the digits example finally
  shows what it claims.
- **The twins' backward pass was wrong, and only the fixed Bend arm could
  reveal it.** Both NumPy twins evaluated the SiLU derivative at the
  *activated* value instead of the pre-activation (BendTorch's `silu_d` takes
  the pre-activation). It was invisible while both arms sat at the mean —
  wrong and right gradients are both ~0 there — and it became visible the
  moment Bend learned and the twin did not: **identical first loss**
  (1.0038943 vs 1.003894329) and finals of **0.044 against 1.031**. The gated
  number is one step with no update, so `make check` could not see it; that is
  now written down in `docs/AUDIT.md`. After the fix the mirror is far better
  than it had ever been measured: the 2d final loss over 4000 steps agrees to
  **1.4e-08** and its clouds to ~1e-6, and the digits final loss to **1 %**
  (0.04408 vs 0.04449).
- **What the reference did right that we did not, in order of effect**:
  zero-mean init (fixed), x̂₀ clamp (fixed), data normalised to [−1,1]
  (measured as not the cause; still a fidelity option), the schedule's
  `1000/T` rescale (measured as not the cause; ours is a different linear
  schedule — ISSUE-002), and the sinusoidal embedding injected as per-block
  scale/shift rather than concatenated as 8 input columns (not the cause; a
  model-quality question — H4).
- **The denoiser is the remaining gap, and the route is measured.** The
  reference's UNet scores **0.926** on the legibility statistic (best
  correlation with any training digit), the dense MLP **0.408** (random
  images 0.326) — and the MLP has the *lower* training loss, 0.042 against
  0.078, so the loss was flattering it. `Conv2d` does not exist in BendTorch
  (issue #15), so the route is attention: the torch probe
  (`reference/digits_vit_probe.py`, 16 tokens of 2×2 patches, dim 64, 4 heads,
  3 blocks, 103k parameters, 3000 steps) reaches **0.798** with the cloud at
  0.260/0.299 and 4096/4096 in range. The port to Bend is
  [adeerkhan/MLBend#16](https://github.com/adeerkhan/MLBend/issues/16); the
  digits test now reports the statistic (`sample_corr`, 0.408 today) so the
  gate can be set when that denoiser lands.
- **The port's design is fixed, and the end-to-end numbers now exist.** Two
  things had to be settled before writing Bend code. (1) BendTorch has no
  per-row position table, so a learned position embedding is not expressible;
  the same ViT with fixed sinusoidal position *features* (8 values per token,
  beside the 8 time-embedding values) scores **0.809** — against 0.798 for the
  learned table — so the port uses features: 4 pixels + 8 time + 8 position =
  20 per token, embedded to 64. (2) The pipelines were compared end to end on
  the *same* architecture (the MLP), each trained and sampled by its own side,
  legibility = best correlation with any training digit, random 0.326:

  | arm | DDPM | DDIM |
  |---|---|---|
  | Bend MLP (full batch, AdamW 0.01) | 0.408 | 0.407 |
  | torch MLP (batch 128, Adam 1e-3) | 0.566 | 0.365 |

  Same order, different recipes; neither sampler is the limiter. The quality
  ceiling is the denoiser: ViT **0.809**, the reference's UNet **0.926**.
- **M1 starts: the token staging and the attention block (2026-10-28).** The
  ViT needs the digit as 16 tokens of 2x2 pixels. `reference/probe_patch.bend`
  is the staging and its inverse; the permutation matches the NumPy side
  exactly (the printed patch is the expected one up to the last printed
  digit) and the round trip returns the original at both ends. The checker
  taught two things worth keeping: a self-call must put its shrinking
  argument *first* (a growing counter before it is refused), and a
  `List<&2, F32>` read several times in one frame needs `+` on the
  parameter. The model then compiles as `LayerNorm -> Residual(Att())` over
  1024 token rows of width 64 with single-head attention
  (`Attention(1024, 64, 64, 64, 16, 1, 1, 64, 0.125, {==}, {==}, {==})`), and
  the two type facts that cost the iterations are: `LayerNorm`'s parameter is
  `T.LN<d>` (not `Mat<1,d>`; `RMSNorm` is the one that takes a row), and
  `Residual` nests its types as `Two<Two<Unit, P>, Unit>`. A single head is
  enough to clear the gate: 0.778 legibility against 0.809 for four heads in
  the torch probe, so the port starts single-head.
- **M1: the ViT compiles and runs (2026-10-28).** `reference/probe_vit_model.bend`
  is the whole denoiser — embed 20->64, three blocks of
  `LayerNorm -> Residual(Attention)` and `LayerNorm -> Residual(64->128->64)`,
  head 64->4 per token — **100,608 parameters**, over 1024 token rows (64
  digits x 16 tokens), single head (0.778 legibility in torch against 0.809
  for four). A zero-parameter forward returns 4096 values (1024 tokens x 4),
  so the shapes and the layer wiring are right end to end. Three borrowing
  facts, learned the hard way: `Mat` is a Type, so a bound matrix cannot take
  `+`; `IO(Unit)` is the signature but `IO<Unit>` is the `do` heading; and
  `Mat.to_list` returns the `ML` wrapper, not a list.
- **M1: the ViT trains in Bend (2026-10-28).** The probe now stages the token
  input (`20480` values = 1024 rows x 20: 4 patched pixels, 8 time, 8
  position), builds the target the same way (`patch(noise)`, so input and
  target share the permutation), initialises through the layer's own loader
  (`Layer.count` -> random `Array` -> `Layer.load`), and steps `T.Train`.
  Two runs, 400 steps each, one fixed window of 64 digits per step: with
  N(0,1) weights the loss falls **5803 -> 11.0**; with the weights scaled by
  0.1 (a 64-wide layer wants ~1/sqrt(64)) it starts at 0.93 and sits at the
  mean predictor. Neither is a surprise: 400 x 64 = 25.6k samples against the
  torch probe's 3000 x 128 = 384k, and 0.75 s a step means the full 3000
  steps is ~37 minutes. The next run is that long one, then the sampler.
  A step is 0.75 s, so the groundwork (staging, init, step, loss) is real;
  what is not yet shown is a loss below 1.
- **M1: the sampler path is written (2026-10-28).** The probe now carries the
  token-ordered cloud through the reverse loop — the start is
  `patch(noise)`, the model is fed the same 20-feature rows, each step forms
  the clamped start and the posterior mean (`c1`, `c2`, `sa`, `sm`, `sig`),
  and the result is unpatched only for the grid and the legibility
  statistic. It compiles; it has not been run against trained weights yet —
  the 3000-step training run was still going when this was written. The
  sampler needed the same three borrowing rules as the training path (a
  `match` on a computed `VS`/`PA`, `+` on a `Type`, `U32.from_nat`), all
  already in `AGENTS.md`.
- **M1's measurement tool (2026-10-28).** `reference/vit_report.py` reads a
  probe run and prints the loss curve, the cloud's statistics and the
  legibility next to the four references (random 0.326, MLP 0.408, torch ViT
  0.809, UNet 0.926), with the M1 threshold (>= 0.70) stated as PASS or FAIL.
  It is read-only and writes nothing, so a run's numbers can be re-read
  without re-running it.
- **M5's compute decision has evidence now (2026-10-28).** The installed
  `bendtorch@1.0.1.0` carries only the CPU layers: `.bend/env/<hash>/` holds
  `main.bend`, `tensor.bend`, `dtype.bend`, `spec.bend`, `LAWS.bend`,
  `PROOF.bend`, `package.bend` — there is no `torch/gpu.bend`, so `Attention`
  has no device mirror in the published package. A ViT step is already 0.75 s
  at 1024 tokens of width 64; CIFAR (64 tokens, wider dim, more rows) would be
  several times that, so a 10k-step run on the CPU is 10+ hours. CIFAR-10 on
  the CUDA lane therefore needs a GPU-capable layer package first (the
  upstream repo's `torch/gpu.bend` is not in what we install), or a much
  smaller model/steps budget. This is the "compute decision" the milestone
  asks for, answered with the directory listing and the measured step time.
- **The two DoD decisions, settled with what was measured (2026-10-28).**
  (1) *The schedule.* Ours is `lin_beta_at(t, T) = 1e-4 + 0.2799·t/T`, the
  reference's is `linspace(1000/T·1e-4, 1000/T·0.02, T)`: at T = 64 that is
  β 0.00447..0.28 and ᾱ₆₄ 4.085e-05 against β 0.00156..0.3125 and
  ᾱ₆₄ 1.224e-05, a 65% worst-step gap. It is a *different schedule*, not a
  rescaling of ours, and both examples and their twins use ours consistently;
  the ablation put it at 0.1787 against 0.1755 for the reference's, so it is
  not what any of the failures turned on. Kept and named, not adopted.
  (2) *The 2d clamp.* The reference clamps the predicted start in every
  sampler; the digits example needs it (ISSUE-005: 32.8 → 0.30 std). The 2d
  example does not have it and does not need it: its cloud is already inside
  the data's range (std 0.599/0.424 against the data's 0.866/0.494, means
  0.385/0.028 against 0.5/0.25), and adding it would move a task whose gates
  are green without a measured reason. Recorded here; if the 2d model is ever
  trained harder or the sampler is shared, the clamp comes with it.
- **Its DDIM step was already right.** `sigma_of`/`step_of`/`ddim` expand to
  the reference's `sigma` and `c` term for term; reading the vendored file
  confirmed the algebra rather than changing it.
- **Checker rules paid for here.** A pattern variable read twice needs the
  plus *inside the pattern* (`case Con{+xv, xt} …`, as `drop64_go` does); a
  record of `F32`s (`RC`) can be a `+` parameter and be reused; `F32.clamp`
  exists in Base and takes the bounds as ordinary arguments.
- **What ran:** the vendored import and a smoke train; the reference
  comparison at 300 and 3000 steps; the 5-variant ablation; the 4-way clamp
  probe; the Bend example before and after each fix (binary, build + 3000
  steps); `--check-only` (only the accepted 315-defs message). **What did
  not:** `make check` and the digits test after the clamp (running now), the
  2d example's sampler (its cloud is already in range; left as a fidelity
  item), and any GPU lane.

## 2026-10-27

- **The DDIM sampler, and the two sampling bugs it exposed (issues #12, #14,
  #15).** `mlbend-generative` 0.1.1.0 adds the DDIM algebra — `sigma_of` (the
  posterior's σ), `step_of` (the five coefficients of one step), `ddim` (the
  element-wise fold) and the strided rung walk (`strided_from`, `groups_of`) —
  with two new laws: `strided_length` (the every-k-th walk emits exactly the
  count `groups_of` says, so the rung lists run in lockstep) and
  `ddim_length` (one step over n elements is n elements). Both examples now
  run the same trained weights down all 64 rungs (ancestral) and down every
  4th rung (DDIM, **16 model calls instead of 64**), and at η = 0 the step
  reads no noise at all: `reference/probe_digits.bend` draws the same cloud
  bit for bit with the draw on, another salt and 4096 elements, and
  `test_ddim.py` gates it on the 2d model too. That invariance is a **test, not
  a law**, and `docs/AUDIT.md` says why: it is a claim about which terms the
  loop evaluates, and the loop lives in an example.
- **#14: the ancestral step's noise was wrong twice.** It used σ = √β_t *inside*
  the 1/√α scale, where Ho et al.'s Algorithm 2 puts the posterior's
  σ = √β̃_t *outside* it. Measured on the linear schedule the examples use:
  the effective noise at t = 64 was **0.6236 against the posterior's 0.5291**,
  at t = 1 **0.0670 where the posterior is exactly 0** (so the last step drew
  noise it must not), and over the whole walk the noise variance was
  **11.29 against 8.914** — 21% too much. The 2d cloud responded: its spread
  against the data's 0.866/0.494 went from **0.363/0.259** (pre-fix report) to
  **0.599/0.424**, while the gate (first loss 152.954, delta 1.5e-5) did not
  move, because training never sees σ.
- **#15: the 2d step divided by the cosine schedule's ᾱ while training used the
  linear one.** `G.alpha_at(t, 64n)` (cosine) and `G.lin_alphas(64n)` (linear)
  disagree by up to **1.9312×** (t = 4) and agree at the top rung; the table is
  in issue #15, with the committed pre-fix report gap that led to it. The fix
  is `abar(t)` everywhere, and the same `abar_below(t)` the digits example got.
- **#13: the digits array's order, and what the probe can and cannot see.** The
  consing `rev_go` handed back a reversed array; one step showed it
  (Bend vs the in-order twin 27.0, vs the reversed twin 3.6e-7). Two things I
  had to measure before choosing a tolerance: the *statistics* (mean, std over
  the flat array) are **order-blind**, so only element-wise reads can catch a
  reorder — hence the probe — and Bend's own `eps` differs from the twin's by
  **7 ulps** (BendTorch's `dotr` accumulates in four partial sums; numpy's BLAS
  does not), with `x` at 3 ulps and `z` at 2. So the probe's tight gate feeds
  **Bend's model output to both sides**: what it gates is the sampler's
  arithmetic and the array's order, not the matmul. Measured, with that gate:
  the step **3.81e-06** absolute (one FMA ulp of the 30-magnitude values), the
  frozen clouds **5.05e-04** (ancestral) and **3.91e-04** (DDIM) relative
  against a reversed array's **1.98**. The gates are `TOL_STEP = 1e-5` and
  `TOL_CLOUD = 5e-3`, ~5× over their own measurement. One correction worth
  keeping: for an **even** number of steps the old alternating orientation
  cancelled, so the digits *cosine* cloud was not reversed in the end; what
  was reversed was every intermediate and every odd-count probe. The 2d walk
  never had the bug (it appends in ascending index order).
- **The digits report's cloud, and an f32 variance that overflowed.** With the
  corrected step the trained digits sampler's cloud is deterministic (two runs
  bit for bit: final loss 1.0258427) and reaches **1.08e20** — the underfit
  denoiser's known geometric expansion (still the residual-head follow-up, not
  a gate). The example's `std_of` summed squares in f32, so 4096 values of
  1e20 overflowed the sum and the report printed `sample_std: inf`, which also
  breaks every parser in the layer. I made the variance overflow-safe instead
  of tolerating `inf`: it factors each deviation by the largest one, sums the
  squares of *those*, and multiplies the scale back — the same variance, finite
  at any magnitude (**1.0752e19** here). The underfit denoiser itself is not
  touched: `sample_mean` 1.2e17 against the data's 0.31 is the debt the README
  already names.
- **Measured in `reference/test_ddim.py`** (the 91st check): the posterior's
  variance against its closed form a **8.196e-08**, η = 0 reads exactly **0.0**
  of noise, one 2d step **9.54e-07**, the rung positions exact (the
  skip = (T−1) mod k anchor lands the top at ᾱ_T for every k), and the four
  clouds walk **2.7e-06, 7.07e-06, 5.53e-06, 2.0e-06** from their twins — the
  same order as the twin-to-twin baseline (6.0e-06), so the 16-call walk costs
  no more than the 64-call one.
- **Three checker rules I paid for again.** A `match` cannot scrutinize a local
  binder (bind the `Bool` and hand it to a def that matches on it — `std_of`
  does); a let that is read twice needs `+` (even when one read is only the
  guard's `is_eq`); and a parameter whose type is a `Type` (the trained
  `B4P()`, `T.Train`) cannot be marked `+` and cannot be used twice, so the
  digits report hands the weights to the second sampler through the `PS` state
  exactly as the 2d example does.

## 2026-10-26

- **The digits layout bug, found and fixed (commit `2f2631b`).** The last
  open difference in `examples/diffusion_digits` was the staged input build.
  `row_of` returned `tail` — the list after *one* pixel — instead of the
  remainder after the row's 64 pixels, so every row from 1 on was shifted by
  a pixel; and `rows_go` consed, so the rows arrived newest-first while the
  twin staged them oldest-first. Two changes fixed it: `drop64_go` consumes
  exactly the 64 pixels of a row (`Con{+h, +tail}` so the head survives the
  copy), and `rows_rev` puts row 0 back first — in both `input_of` and
  `sinput_of`. After it: first loss **8062.6895 against 8062.689453125** and
  data mean **0.30756474 against 0.30756473541259766**. The tell was always
  the gate: order-blind statistics (mean over 65536 pixels) barely moved
  while the loss did.
- **What the printed losses mean** (measured, not assumed). The loop prints
  `loss:` at i = 1, 99, 199, …, 2999 and each line carries the loss of the
  state *entering* step i−1, so the first printed line is the twin's
  `curve[0]` — verified: Bend 8062.6895, `curve[0]` 8062.689453125, while
  `curve[1]` is already 4111.5. The base case's `final_loss:` line carries
  the twin's `curve[-1]`. `loss.tsv`'s step column ({0, 98, 198, …}) matches
  the print schedule exactly, so the pairing was right and the visible drift
  at later steps (Bend 18.330002 vs 18.466026306152344 at step 98) is two
  optimizers rounding differently over 99 AdamW steps — which is why only
  the first loss is gated.
- **The task is wired and passes** (`diffusion-digits`, commit `855af5b`):
  first run **PASS**, first loss delta **4.69e-05** (relative 5.8e-9), data
  mean delta **4.6e-9**; build 124 s, run 298 s, reference 35 s. The
  report-mode fields say the sampler is the weak part: final loss 1.02584 vs
  1.07458, `sample_mean` **50256.9** against the data's 96.7,
  `sample_std` **748808** against 753.9 — an underfit denoiser whose
  geometric expansion through 64 reverse steps is the known follow-up (a
  residual head, `T.Residual`), not a pass condition.
- **The GPU lane's `libnvrtc.so.13` failure was an environment bug, not a
  compiler bug.** `bend -o` could not load `libnvrtc.so.13: cannot open
  shared object file` because the default `CUDA_HOME` named a nonexistent
  `~/.local/cuda12`, the libraries live in `lib64` (not `lib`), and
  `LD_LIBRARY_PATH` was unset. The toolkit here is three RPMs —
  `~/nvrtc.rpm`, `~/nvrtc-devel.rpm`, `~/cudart-devel.rpm` — unpacked into
  `~/.local/cuda-nvrtc`, which holds `include/nvrtc.h`, `include/cuda.h`
  and `lib64/libnvrtc.so.13`. The fix is committed in **bendtorch**
  (`6ee6721`): `CUDA_HOME ?= $(HOME)/.local/cuda-nvrtc`,
  `CUDA_LIB := $(if $(wildcard $(CUDA_HOME)/lib64),$(CUDA_HOME)/lib64,$(CUDA_HOME)/lib)`,
  and every GPU recipe and demo run exports `LD_LIBRARY_PATH=$(CUDA_LIB)`.
  With it, `make gpu` and `scripts/gpu_expcr.sh` pass: `gpu_test` all
  eight zeros,
  `gpu_layers` SiLU 4 ulps, cross-entropy 4, attention 176128 at the small
  sizes and 7680 at the large ones — their documented ulp policy.
  (Earlier sessions ran it with `PY=/home/adeerkhan/github/fatima/.venv/bin/python`
  only because no `reference/.venv` existed anywhere on this machine;
  `scripts/setup.sh` builds one with `torch==2.14.1+cpu` and that is what
  the verified run used — the oracles are CPU-side.)
  **Upstreamed on 2026-10-26**: BendTorch issue **#10** (both failure
  modes reproduced from scratch: the hardcoded `lib` vs `lib64`, and the
  gate printing "on the GPU" for a CPU build when no toolkit is found —
  `ldd` shows zero CUDA libraries) + PR **#11** with the detection/lib64/
  guard fix (`make laws`, `make gpu`, `make check` green on the branch),
  and the recipe doc issue **bendlang/bend#1863**.
- **`make check` is 90/90**: `reference/test_diffusion_digits.py` is the
  90th check (440 s, the slowest — Bend build + 3000-step run + the NumPy
  twin). Both diffusion examples now have a README with their saved picture
  (`scatter.svg`, `grid.svg`); regenerating them refreshed crossval prose
  that predated the shared folds (both sides 0.9307) and heart naive Bayes'
  number that predated the `var_smoothing` fix (0.95).

- **One `~/.bend/bin/bend`, two pins: MLBend 2.0.38 vs BendTorch
  2.0.39.** BendTorch's `scripts/setup.sh` SHA-checks **2.0.39** and
  installs to the *same* global `$HOME/.bend` that MLBend pins at
  **2.0.38** — so running the former in the BendTorch worktree silently
  replaced the global compiler, and MLBend's `check_all.py` toolchain
  gate caught it exactly as designed: `ERROR: bend 2.0.38 is required
  (found bend 2.0.39)`. Restored with MLBend's own SHA-verified
  `scripts/setup.sh`. **The recipe going forward**: BendTorch's setup
  honors `BEND_HOME`, so run it as
  `BEND_HOME=$HOME/.bend-2.0.39 scripts/setup.sh` and keep its pin
  beside the global one; give each repo its own `BEND_BIN`. The gate
  turning red within minutes of the flip is the system working — a
  number checked is a number caught.

- **CIFAR-scale steps are host-bound: the GPU lane does not help
  (measured, both lanes).** A probe built from the digits example at
  CIFAR shapes (n=256, d=3072, hidden 128, T=64, 16 samples), run on the
  RTX 5060 with Bend 2.0.38 on an idle machine, three runs each:

  | config | CPU (`torch/main.bend` + layers) | GPU (`torch/gpu.bend` layers) |
  |---|---|---|
  | 0 steps (x0 noise + stats + 64-step sampler) | 214.3 / 216.1 / 217.0 → **215.8 s** | 225.5 / 225.6 / 229.7 → **226.9 s** |
  | 1 step (S1 − S0) | **~7.4 s/step** (S1: 223.1 / 223.1 / 223.4) | **~8.7 s/step** (S1: 234.0 / 234.5, one 255.0 outlier) |
  | full 5-step run | 192.8 s | 201.6 s |
  | `final_loss` after 5 steps | 71710.04 | 71710.03 (1 ulp — the lane's bits hold) |

  **The GPU is never faster: +5 % on the sampler, +18 % on the step.**
  The reason is arithmetic: a step's matrix work is ≈1.2 GFLOP
  (0.1–0.3 s at BendTorch's own 10 GFLOP/s bench rate) against ~7 s of
  **host** work — the counter noise (`G.noise_of` builds 786 432
  Gaussians per step, each `gauss_at` recomputing *both* uniforms of its
  pair and the `log`/`sqrt` radius: ~100 host graph reductions ≈ µs each)
  plus the staged list input. The bangs make the ~2 % device share free
  and leave the 98 % where it was; the sampler is ~215 s of pure host
  noise in both lanes. Implications: (a) a CIFAR task must be sized by
  *steps × 7.4 s* until the noise can be filled on the device — the
  candidate fix is a counter-RNG fill bang in `bend-ml-tensor-gpu`
  (filed as bend-ml#23); (b) MLBend's examples stay on the proved
  `main.bend` layers with no loss, because the proved lane is the fast
  lane here too; (c) the probe files live in `/tmp/opencode/` and
  `bendtorch/.scratch/` (throwaway), the patch script is
  `patch_cifar_probe.py`.

- **The retraction lesson (BendTorch#12, closed within the hour).** I
  filed "the published package ships neither `gpu.bend` nor `fast.bend`"
  from listing the `bendtorch@` core hash's files. Wrong:
  `scripts/package.sh` assembles **four** hub packages — the law-clean
  core plus the `bendtorch-fast` / `bendtorch-gpu` / `bendtorch-load`
  satellites, all live at **1.1.1.0** — and a consumer imports the lanes
  by name. The lesson for the next issue: read the packaging pipeline's
  *outputs* and ask the hub itself
  (`curl https://hub.bend-lang.com/name/<name>`) before claiming what
  does not exist; a hash's file list is one package, not the design.
  What survived: #13 (a `main.bend`-written program still needs the
  `Linear -> LinearF` renames — the aliases PR is BendTorch#13's).

## 2026-10-25

- **The first diffusion model in Bend** (`examples/diffusion`, the task
  `diffusion-2d`): a three-linear-layer denoiser with SiLU trained by
  `T.Train.step`/AdamW to predict the noise of a 32-step DDPM forward
  process over two moons, then sampled by the ancestral loop. The gate is
  the **first training loss, 6.542548, agreeing with a float32 NumPy twin
  to 1.8e-7**: every "random" number is a pure function of a counter —
  murmur3's fmix32 (the hash shape bendtorch's dropout mask uses, though
  theirs is lowbias32) over the step salt and the index, Box-Muller over
  uniforms on a 24-bit grid (`(m24+1)/2^24`, exactly representable, never
  zero, so `log(u)` is safe). Base has `F32.sin/cos/log/exp` — they are
  laws with builtin bodies, and they were the difference between a
  sinusoidal time embedding plus real Gaussians and a much weaker design.
- **The affine rule owns the sampling loop.** Model parameters hold
  arrays, so they cannot be reused: a first draft passed `p` into 32
  reverse steps and the checker refused ("p consumed more than once"), and
  `+p` is refused too (a `+` needs Data, and an `Array` is not one). The
  answer is bendtorch's own: `T.Layer.fwd` **returns the parameters** in
  `Fwd{p, y, s}`, and nuxyel's eval loop (smol.bend's `ev_loop`) threads
  them through a record into a continuation. Our sampler does the same
  with a `PS{p, x}` record; `Mat.dup` exists for one-shot duplication but
  a loop wants the threading shape. House rule for the future: *a Bend
  inference loop returns its weights.*
- **Arrays are linear; lists are the data plane.** The example was drafted
  over raw `Array<F32>` and every multi-read walk failed (two `Array.get`
  on one array is a double consume; `Array.get` returns the array beside
  the value, so kernels thread it through tuples). The rewrite moved the
  data, the noise, the embeddings and the samples to `List<&2, F32>`
  (duplicable), touching `TA.Mat` only at the edges via `Mat.from_list` /
  `Mat.to_list`. That is also the knn/attention house style.
- **The one real bug the exact comparison caught**: the moons list is
  *interleaved* (u0, l0, u1, l1, ...) — the walk appends one point of each
  moon per step — and the NumPy twin first wrote the upper moon whole,
  then the lower. Row 0 agreed, every later row did not; the first loss
  sat 2 % off (6.5425 vs 6.4183) while every scalar stage (hash, uniform,
  Gaussian, ᾱ, embedding, weights, single forward outputs) matched to an
  ulp. The data's mean and std still agreed — order-blind statistics hide
  layout bugs; only the gated loss caught it. Fixed in the twin; the Bend
  side was right.
- **The checker's rules the file cost the most to satisfy**, on top of the
  known ones: no lets inside a `do` block after the first statement (put
  computation in defs); a match cannot scrutinize a computed value, so
  every unwrap of a `Fwd`/`LossOut`/`Pt` is its own def taking the value
  as a parameter; a binder read twice in one match arm needs `++`; a Nat
  match needs a `case _` catch-all; U32 literals are plain decimals with
  `: U32` annotations (no `1u`, no hex); `Seq`'s state type parameter is
  the *saved* state — `Linear` saves its input (`Mat<n,i>`), an activation
  its output; and a Data record cannot hold arrays (`type PS is Type:` is
  the affine record, the shape smol.bend uses for the same purpose).
- **`mlbend-report@0.1.0.0` is written** (issue #9): the confusion matrix and
  the per-class report — precision, recall, F1, support, the macro and the
  weighted averages — over `Nat` labels, on `mlbend-metrics`' counts. Two laws
  are proved: `matrix_rows` (n rows of the matrix are n rows) and
  `row_length` (n cells of a row are n cells), stated over the counted forms
  (`matrix_from`, `row_from`) because those are the recursions the checker can
  follow — the checker refused the law stated over the `k`-classes wrapper,
  because the walker keeps the *total* `k` fixed (every row is still `k` long)
  while only the rows-remaining counter shrinks. The wrapper `matrix` is the
  law at `n := k`. The debt — the cells add up to the number of samples — is
  the value-dependent wall of issues #1 and #8.
- **The numbers match scikit-learn exactly.** `reference/test_report.py`
  compares the matrix, every per-class score, the macro and weighted averages
  and the accuracy of `report/tests/ok.bend` (seven samples, three classes
  present, one absent — a zero-support class exercises `zero_division=0`)
  with `confusion_matrix` and `classification_report(labels=…, zero_division=0)`:
  every check agrees within F32 precision. `cancer-knn` now prints the report
  beside its accuracy (`examples/knn/main.bend`), and `reference/test_knn.py`
  checks the matrix `[[100, 2], [1, 34]]` and the per-class scores against
  scikit-learn's kNN on the same predictions. The weighted F1 lands at
  0.59999996 where scikit-learn says 0.6 — F32 rounding in the sum, exactly
  why the values are tested and never proved.
- **The checker's rules the package cost the most time to satisfy**, all
  already recorded in these notes and all confirmed: a list consumed twice
  needs `+`; a record has no field access (match on a parameter instead); a
  match cannot scrutinize a local binder, so an unwrap is its own def; a def
  must come before the defs that use it; an F32 binder read three times needs
  `+n = F32.from_nat(k)`.
- **Two upstream issues filed.** `bendlang/bend#1848`: the two spellings of an
  effect — a return type written `IO<Unit>` reports "a declared datatype
  (unknown: IO)" (the compiler reads an application as a declaration) and a
  `do IO(Unit)` reports `expected '<', observed '('`; neither names the pair.
  Both repros verified on 2.0.38 — and the first attempt at the repro
  reproduced *neither* until `import Base` was added, which itself is a
  finding: without the import, even the correct `IO(Unit)` return type reports
  "expected: a defined name, observed: IO". `nuxyel/bend-ml#20`: the design
  question the frozen `bend-ml-autograd` leaves — re-base autodiff on the
  maintained `tensor-array`, or declare BendTorch the answer and standardize
  the conversion seam; the leaning stated there is (2), with the conversion
  surface as the cheaper, law-carrying artifact.
- **The toolchain on this machine caught up.** `~/.bend/bin/bend` was still
  2.0.37 (the 2.0.38 move of 2026-10-24 had landed in the repo but not here)
  and the Python venv predated the repository's move to `github/bend/MLBend`
  (its interpreter pointed at the old path). `scripts/setup.sh` rebuilt both;
  Lean 4.34.0 was already installed, so the kernel `--verdict` runs locally
  from now on — and it proves the two new laws here, not only in CI
  (`make check` is 81/81 on this machine).

## 2026-10-24

- **The evidence stage (2026-10-24).** Three things changed under the
  corpus. The datasets are pinned to a HuggingFace revision (`resolve/<sha>`)
  and each task carries the expected SHA-256; the runner verifies the bytes it
  reads. `MLBEND_DATA` is now authoritative: a miss is an error, not a quiet
  fetch, so a manifest can never hash one file while the program reads
  another. `mlbend-datasets@0.1.6.0` carries both changes.
- **The same folds, and five seeds (2026-10-24).** `examples/crossval` prints
  the fold of every row and takes the seed as its first argument; the runner
  builds once, runs seeds 42–46, and the NumPy reference reads that assignment
  and runs the same 60 SGD steps. The means agree to **5.9e-8** and the folds to
  **2e-8**, instead of the old 0.03 tolerance for two different partitions.
  `RESULTS.md` now reports mean ± sample deviation; every seed must pass.
- **The naive Bayes variance (2026-10-24).** The exact comparison found the
  heart baseline at **0.9333** against scikit-learn's **0.95**. The cause was
  not the split or the data: scikit-learn adds `var_smoothing = 1e-9 ×
  max(variance)` to every variance, and the Bend examples did not. With the
  same smoothing both baselines agree exactly: heart **0.95**, breast
  **0.98540145**. The heart test now gates at 1e-6.
- **Upstream (2026-10-24).** `bendlang/bend#1845` makes `File.read` fill its
  buffer instead of answering one `read(2)`; the repro is a pipe that writes
  three bytes, sleeps, then three more, and a six-byte read that used to
  answer three. The PR has a FIFO regression on both lanes. The bend-ml
  finding — `bend-ml-tensor` is the only `Mat` autograd trains, while
  `bend-ml-tensor-array` is the faster one — is architectural and already
  documented as frozen/superseded; it is not a PR-shaped change.
- **The dataset cache (2026-10-24).** `mlbend-datasets@0.1.5.0` reads a
  local file when `MLBEND_DATA` is set: the directory is laid out as
  `<repo>/<path>`, the same shape as the hub (`mstz/iris/iris.data`), and a
  missing file falls back to the network. The experiment runner fills
  `reference/.cache/datasets` with the exact bytes it hashes and sets the
  variable before the Bend run, so a warm task records `fetch: 0.0` and its
  run time is compute. A cold first run still pays the fetch; run the corpus
  twice, or warm the cache once, before reading the timing column. The
  iris-crossval run fell from **0.85 s** to **0.098 s** of compute. The
  loader reads the file at its real size (`File.size`, then `File.read`)
  rather than a fixed cap, so a larger dataset is not silently truncated;
  the cache is gitignored and the SHA-256 stays in the manifest.
- **The structural graph (2026-10-24).** `/understand` built a fresh graph
  at `05eacbd`: 102 files, 958 nodes, 905 edges, 6 layers, 7 tour steps and
  no validation issues. It is structural, not LLM-enriched per file, and it
  is at `.ua/knowledge-graph.json`; the point is orientation, not evidence.
- **The foundation correction (2026-10-24).** The “BendTorch has no softmax”
  finding was true only of `bendtorch-core@0.1.0.0` (`aaaxn/bendtorch`, the old
  line). The maintained library is `bendtorch@1.0.1.0` (`nuxyel/BendTorch`),
  which has `Attention`, `RMSNorm`, `Embedding`, `CrossEntropy`, `AdamW` and
  `Train`, and targets Bend 2.0.38. MLBend now pins **Bend 2.0.38** and depends
  on the real `bendtorch`. The two artifacts opened against the wrong tree were
  closed; no upstream issue is open for a gap that does not exist.
- **The first trained transformer (2026-10-24).** `examples/transformer` is a
  one-block causal transformer on the real library: embedding, RMSNorm,
  single-head causal attention, output projection, RMSNorm, linear head,
  AdamW, 200 steps on next-token prediction over the periodic sequence 0..7.
  The loss falls from **2.3591** to **0.0021**; a NumPy twin with the same
  deterministic weights starts at **2.3591** (the two agree to 6.8e-7) and ends
  at **0.1351**. The Bend update moves faster after the first step; the curves
  and the table are in `results/transformer/` (`loss.svg`, `loss.tsv`,
  `run.json`), and the task is `transformer-periodic` in the experiment layer.
  The plain-Bend `attention` example remains the library-free forward parity
  check (1.5e-8).
- **Bend 2.0.38 (2026-10-24).** `scripts/setup.sh`, the check pin, the Makefile
  and the docs moved from 2.0.37; the SHA256s are the release tarballs'. The three old examples (`tape`, `regression`, `stacks`) were retired on
  2026-10-24, and `bendtorch-core` left `bend.toml` with them.

## 2026-10-23 — the experiment layer, and what a law is for

- **The merge (2026-10-23).** `devel` went into `main` as a fast-forward (55
  commits) and the work continues on `main`. The old branch stays for history;
  the rules in `AGENTS.md` and the issue tracker now say `main`.
- **The identity (2026-10-23).** MLBend is the experiment layer, not another
  model library. bend-ml owns the tensors and the laws, BendTorch owns the
  training, and this repository owns the data, the splits, the transformations,
  the selection, the metrics, the reports and the manifests. The boundary rule
  is now in `AGENTS.md`, `README.md` and `docs/PLAN.md`: a model enters through
  an adapter; no tensor, no autodiff, no optimizer, no second trainer.
- **The runner (2026-10-23).** `reference/experiment.py` turns a task into a
  report card: it builds and runs the program, parses the number, computes the
  reference with NumPy or scikit-learn, compares them within a declared
  tolerance, and writes the manifest — git commit and dirtiness, Bend version,
  every `mlbend-*`, `bend-ml-*` and `bendtorch-*` package with its hub hash,
  machine, seed, build time, run time and reference time. The reports are
  committed under `results/`; `--wandb` sends the same manifest to Weights &
  Biases, offline without an API key.
- **The timings are two different questions (2026-10-23).** A task's Bend run
  and its Python reference both fetch the dataset from HuggingFace, so their
  ratio compares the whole program, not the kernel. The compute-only benchmark
  is the second answer: fifty sequential 128 x 128 float32 products take
  **18.4 ms** in Bend over `bend-ml-tensor-array` and **1.0 ms** in NumPy —
  11.4 against 218.4 GFLOP/s, NumPy 19x faster, one Bend thread against BLAS.
  The two numbers belong in different tables, and `results/` keeps them apart.
- **The two new examples (2026-10-23).** Gaussian naive Bayes over breast
  cancer scores **0.98540145**, the same as scikit-learn's `GaussianNB` on the
  same split; PCA's first component explains **0.275443** of the variance,
  against scikit-learn's **0.2754426**. Neither needs standardising (a Gaussian
  is affine-invariant) or a trainer.
- **The pipeline, and the leakage it removes (2026-10-23).** `mlbend-pipeline`
  fits the scaler on the training rows of a fold and transforms both sides;
  `examples/crossval` had standardised the whole matrix first, so the test rows
  were in the means and the deviations. The law `pipeline_preserves_rows`
  proves one output row per input row; the numbers are compared with
  scikit-learn's `StandardScaler` and with `Pipeline(StandardScaler,
  LogisticRegression)` on the same folds. The iris mean is unchanged (0.9333)
  because standardising everywhere is nearly harmless there — the API is what
  changed, and the leakage is now impossible by construction.
- **CI hit its 30-minute ceiling (2026-10-23).** The run on `42ad0f4` was
  cancelled at 30m16s: 58 checks, each building a program, do not fit in 30
  minutes on a 4-vCPU runner. `.github/workflows/ci.yml` now allows 60, and
  the retired `devel` branch is out of the push filter.
- **The two spellings of an effect (2026-10-23).** A return type written
  `IO<Unit>` rather than `IO(Unit)` reports `a declared datatype (unknown:
  IO)`, and the reverse (`do IO(Unit)`) reports `expected '<', observed '('`.
  Neither names the pair or the position; the minimal repro and the suggested
  message are for `bendlang/bend` (the `#1632` diagnostic series).
- **The first transformer block, and the op it needs (2026-10-23).** A
  single-head self-attention block over four tokens of eight features — Q, K,
  V, scaled dot-product scores, row softmax, the weighted sum of V — is written
  in plain Bend (`examples/attention`) and agrees with PyTorch to **1.5e-8**
  on every output number (`reference/test_attention.py`). The weights are a
  fixed formula, so the task is `attention-parity` and the checksum agrees to
  3e-9.
  **The gap is training.** BendTorch's tape records add, sub, mul, matmul,
  relu, slice, gather, cross-entropy and MSE; it has no recorded `exp`, no
  broadcast `div`, and therefore no softmax backward. A transformer cannot be
  trained on it until those operations (and their VJPs) exist. The forward is
  pure Bend over lists for now; the next upstream step is a BendTorch issue for
  `softmax` (or `exp`/`div`/`sum_dims` with gradients), then a trained attention
  task here.
- **A function is a Type, so a builder cannot be passed and reused (2026-10-23).**
  Writing one `row(builder, ...)` and passing `x_cell`, `wq_cell`, ... needs
  `+f`, which the checker refuses: `+x` requires a `Data` type, and a function
  is a `Type`. The example builds by a `kind` number instead
  (`cell(kind, i, j)`), which is also what made the termination check simple:
  the decreasing `n` must sit before the increasing column index in the
  recursive call.
- **Laws: what they are for here (2026-10-23).** A law belongs to a pure
  structural claim — a count, a partition, a cover, a composition — and the
  kernel proves it. A number (`F32`), a wall time, a dataset fetch and a W&B
  log are not laws: they are tested against a reference and recorded in a
  manifest. So the experiment runner adds no laws, the two examples add none
  (they are examples, not packages), and the next Bend package — the
  no-leakage pipeline — owes one: *fitting sees only the training rows of the
  fold*. Until that proof exists, the claim is an issue, not a package.

## 2026-10-23

- **The tape accumulates gradients until `zero_grad` (2026-10-23).** A
  backward pass in bendtorch adds its contribution to each working slot, and
  `Sgd.step` keeps the slot after it steps (`Backward.zero_grad` is what
  empties it). A training loop that calls `backward` and `step` without
  `zero_grad` therefore applies the sum of every gradient so far: the first
  regression run printed 0.9000 after 200 steps at 0.1 and 0.9293 after 1000
  steps at 0.3 — more steps, worse loss, because the step was growing. With
  `Backward.zero_grad(tape)` after each step the same run reaches **0.7078516**
  where scikit-learn's `LinearRegression` gives **0.7078631**
  (`reference/test_regression.py`). The one-step `examples/tape` had hidden
  this: one backward, one step, nothing to accumulate.
- **The two-layer network is the rectifier's chain (2026-10-23).** A hidden
  layer of eight rectified units over iris, trained with `AG.linear`,
  `AG.mask.rows`, `AG.linear`, `AG.ce_loss`, `AG.ce_grad`,
  `AG.linear_bwd`, `AG.relu_bwd` and `AG.linear_bwd` again, reaches accuracy
  **0.9733** after 400 steps at 0.5, the same accuracy scikit-learn's
  `MLPClassifier(hidden_layer_sizes=(8,))` reaches on the same data. The
  starting weights are a fixed small pattern: all-zero weights leave the
  hidden units with a zero gradient, because the gradient of the rectifier at
  zero is zero.
- **k-means is a package now (2026-10-23).** `mlbend-cluster@0.1.0.0` carries
  the Lloyd rounds, the assignment and the inertia from `examples/kmeans`,
  with the law `cluster_covers` (one label per row). The example that read the
  wine dataset keeps only the data and the standardising, and the numbers did
  not move: inertia 50645.64, sizes 2384 990 3123.

## 2026-10-22

- **48/48 on the consolidated tree (2026-10-22).** `make check` after the six
  packages became one repository: every proof checks, the two programs that
  must be refused are refused, the pinned outputs match, and the NumPy and
  scikit-learn numbers match. The kernel `--verdict` was skipped for want of
  Lean 4.34.0 here; CI runs it. Verification is the same command as before —
  consolidation changed where the packages live, not what they compute.
- **One repository for the six packages (2026-10-22).** The per-package
  `mlbend-*` repositories were a mistake in shape, not in content: the packages
  were always developed side by side in the workspace `bend.toml`, and the
  release script existed only to copy each directory into a repository of its
  own. A package's identity is its name and version on BendHub, not a
  repository; the commit and the tag `<name>-v<version>` here record where the
  published bytes came from. `bend-env` keeps its repository because it is a
  tool, not a package, and is consumed by checkout rather than by the hub.
- **The one file that had drifted.** The datasets README was newer in the
  standalone clone than in `datasets/`: the four-dataset table, the `Table`
  type and the `drop` argument were only there. The package's `main.bend` was
  identical in both places, so the hub hash was never in question; the README
  is merged now.

- **bendtorch's shapes are settled (2026-10-22).** The question the
  handoff carried — the weight orientation and the rank of the bias — is
  answered by bendtorch's own source, not by guessing:
  - `src/tensor/matmul.bend`: "x @ W^T for x of [rows, inner] and W of
    [columns, inner]: cell (row, column) adds x[row, k] * W[column, k]".
    So the weight is **[out, in]** (PyTorch's layout) and the input is
    [rows, in].
  - `src/autograd/tensor_backward.bend`: "Linear: records x @ W^T with W
    of [out, in], PyTorch's layout, then adds the bias b of [out] to
    every row." So the bias is a **vector of [out]**.
  - The MNIST demo agrees: `weight1 [128, 784]`, `bias1 [128]`,
    `weight2 [10, 128]`, `bias2 [10]`.
  - The tape example now builds the weights as `[3, 4]` (three classes,
    four features) and the bias as `[3]`, and standardises the features
    through `mlbend-preprocess` so it trains the same model as
    `examples/logistic`.
  - **The 106 was the shape, after all.** bendtorch's `linear` records
    `x @ W^T`, so the inner dimensions are x's columns and W's columns:
    with the data of [150, 4] the weights must be [3, 4] and not [4, 3].
    The bias is [3], a vector of the outputs. With the shapes right the
    refusal is gone.
  - **A node's value is fixed when it is recorded.** The loss *after* the
    step is therefore a second forward pass on the stepped tape: the step
    writes the new weights into the leaves, and a recording after it
    reads them ("Recording an operation after a step reads the new
    values", as bendtorch's sgd.bend says). bendtorch's own MNIST demo
    rebuilds the tape each step for the same reason.
  - **The two stacks agree.** `examples/tape` (bendtorch) and
    `examples/logistic` (bend-ml) train the same model on the same
    standardised iris data with a learning rate of 0.5, and the losses
    agree to F32 precision: 1.0986128 before, 0.7287295 and 0.7287296
    after (`reference/test_tape.py`). Two independent libraries, one
    gradient step.

- **`mlbend-preprocess` (v0.3, issue #5).** The transformers of
  scikit-learn over **bend-ml's** matrix: `Mat<r, c>` from
  `bend-ml-tensor`, the one bend-ml's autodiff takes. MLBend writes no
  matrix of its own.
  - The standardiser is bend-ml's own arithmetic: a column-wise
    normalisation is `Mat.transpose`, then `Mat.layernorm` (which
    centres a row and scales it by its own standard deviation), then
    `Mat.transpose` back. The layernorm's variance is the population
    one (divided by the count), the same as scikit-learn's.
  - The numbers match scikit-learn's `StandardScaler`,
    `MinMaxScaler`, `Normalizer` (l1 and l2) and `np.eye` on the same
    matrix, to the precision of an F32 (`reference/test_preprocess.py`).
  - What is proved: the one-hot matrix has one row for every label
    (`one_hot_count_length`, an induction over the labels). What is
    only tested: the values, because F32 is opaque to the checker.
    "The standardised column has mean 0 and variance 1" is the debt of
    the issue.
  - `Mat.transpose` and `Mat.layernorm` take the *width* as a runtime
    argument, so a generic transformer over `Mat<r, c>` has to declare
    `+r: Nat, +c: Nat` (linear) — the dimensions are erased type
    parameters, and a transformer that needs them at run time cannot
    have them erased.
- **The logistic example standardises its features (2026-10-22).** With
  the standardiser the learning rate of 0.5 is sane: the loss goes
  1.0986128, 0.7287295, 0.6035278, 0.5403903. On the raw features (which
  run from 0.1 to 7) that same rate overshoots and the loss *rises* —
  the step overshooting, not a wrong gradient.
- **Issue #2 (the containers) is answered by the work.** The matrix is
  bend-ml's `bend-ml-tensor` `Mat`: it is a list of rows with the
  dimensions in the type, it is the one `bend-ml-autograd` takes (so
  the estimators can feed a model without a copy), and it has
  `col_sums`, `to_rows`, `transpose` and `layernorm` — everything the
  transformers need. The flat `bend-ml-tensor-array` is a *different*
  `Mat` type: using it would mean a second matrix in the ecosystem,
  which `docs/PLAN.md` forbids.

- **The bendtorch tape trains (2026-10-22).** The 106 was the shape after
  all: bendtorch's `linear` records `x @ W^T`, so the weights are
  `[out, in]` — three rows of four for the iris data, not four rows of
  three — and the bias is a vector of `[out]`. The second half of the
  puzzle was that a node's value is fixed when it is recorded: the loss
  *after* the step is a second forward pass on the stepped tape, because
  the step writes the new weights into the leaves. `examples/tape` now
  trains, and `reference/test_tape.py` checks it against
  `examples/logistic`: bendtorch and bend-ml are two implementations of
  the same gradient step, and they agree to F32 precision (1.0986128
  before, 0.7287295 and 0.7287296 after).
- **The cross-validation example now sits on the packages (2026-10-22).** It
  used to shuffle and deal the rows out itself; the folds come from
  `mlbend-select` now, so the split is written in one place. The
  assignment changed with it (the package deals the rows of a class out
  in turn, the example dealt them out in blocks), so the per-fold
  accuracies moved: 0.9667, 0.8333, 0.9667, 0.9333, 0.9667. The **mean is
  the same, 0.9333**, and scikit-learn's is 0.9533, so
  `reference/test_crossval.py` still agrees within 0.02.
- **The plan's v0.9 is reached (2026-10-22).** The iris dataset goes
  through standardize -> logistic regression -> cross-validation, and
  the accuracy matches scikit-learn's:
  - the Bend program (5 stratified folds, seed 42) scores 0.9667, 0.9,
    0.9333, 0.9, 0.9667 — a mean of 0.9333;
  - scikit-learn's LogisticRegression over its own stratified folds
    scores a mean of 0.9533.
  - The folds are *not* the same folds: scikit-learn shuffles each class
    with numpy's generator, the Bend program with bend-kit-random's
    xoshiro128**. What agrees is the procedure, to within the spread of
    two random partitions of 150 rows.
- **`mlbend-preprocess` (v0.3, issue #5) is published** as
  `mlbend-preprocess@0.1.0.0` (`0xa8182854f375cd61135ccd32993993cc`).
  The transformers run over bend-ml's `Mat<r, c>` — bend-ml's matrix,
  not one of ours — and the standardiser is bend-ml's own arithmetic:
  `Mat.transpose`, then `Mat.layernorm` (which centres a row and scales
  it by its own standard deviation), then `Mat.transpose` back. The
  numbers match scikit-learn's `StandardScaler`, `MinMaxScaler` and
  `Normalizer` to the precision of an F32.
- **The PRNG question is answered.** `bend-kit-random@0.1.0.1` is on the
  hub: xoshiro128**, pure (a draw returns the value beside the next
  state), with unbiased ranges and Fisher-Yates shuffles. It is someone
  else's package, so the split uses it rather than a generator of our
  own. The plan's open question about the PRNG is closed.
- **`mlbend-split` (v0.4) is written and published** as
  `mlbend-split@0.1.0.0` (`0x97faf074f5fec7518a7b67151e3be4f7`), on the
  hub and on GitHub under the tag `v0.1.0.0`. The package deals the rows
  of a class out over the folds in turn, so every fold holds the same
  proportion of every class. The shuffle is `bend-kit-random`'s
  xoshiro128**, seeded: the same seed gives the same split, and another
  seed gives another one. The law `fold_covers` says every row of the
  dataset is placed in a fold, so the folds partition the rows.
- **The fold sizes and the class counts match scikit-learn's exactly**
  (150 rows, three classes of fifty, five folds: 30 rows a fold, ten of
  each class a fold, on both sides). The two programs do not agree on
  *which* rows go where — scikit-learn shuffles with numpy's generator,
  the package with bend-kit-random's — so the comparison is of the shape
  of the split, not of its contents.
- **The sum law is the same wall as the accounting law (2026-10-22).** I
  tried to prove "the rows of a fold and the rows outside it add up to all
  the rows" (`Nat.add(count_in(xs, f), count_out(xs, f)) ==
  List.length(xs)`) and the checker named the shape it wants:
  `Nat.add(count_in_go(t, f, same_of(Nat.is_eq(h, f))),
  count_out_go(t, f, same_of(Nat.is_ne(h, f)))) == 1n+_`. The head's
  contribution is *data-dependent* — `same_of` is a `Bool` the checker
  cannot reduce — so the tail's two counts do not line up with the
  hypothesis's `Nat.add(count_in_go(t, f, 0n), count_out_go(t, f, 0n))`.
  This is the wall issue #1 records for the metrics' accounting law: a
  count that depends on the *value* of an element cannot be lifted to a
  `1n + ...` shape. The way out is a count that walks with the test
  decided by the caller, or arithmetic over `Nat` that the checker can
  reassociate — `bend-ml-nat-lemmas` has `add_comm` and `add_assoc`, but
  the hypothesis shape has to be the goal's shape, and neither lemma
  matches it.
- **`mlbend-select` (v0.5) is written and published** as
  `mlbend-select@0.1.0.0` (`0xf6594aa2392363c8ab774bc81fb20b9d`), on the
  hub and on GitHub under the tag `v0.1.0.0`. It hands out the folds
  `mlbend-split` makes, so the two agree by construction, and adds the
  walk over the folds: the test rows of each fold, the training rows,
  and the sizes. The law `select_covers` says every row of the dataset is
  in exactly one fold. The fold sizes and the class counts match
  scikit-learn's `StratifiedKFold` exactly (150 rows, three classes of
  fifty, five folds: 30 rows a fold, ten of each class a fold, on both
  sides).
- **`mlbend-split` is at 0.1.1.0** (`0xe8b7953f71ec5b441a90a08e0961269b`):
  it gained `test_of` and `train_of`, which take the fold list instead of
  the labels, so the folds are computed once and asked of as many times
  as needed.
- **The two new packages' laws are checked, not certified (2026-10-22).**
  `mlbend-split` and `mlbend-select` depend on `bend-kit-random`, which
  carries `entropy` and `from_os` — two defs that call the operating
  system. Neither package calls them (both seed the generator), but
  Bend's verdict refuses a book that holds any foreign code, and it
  cannot tell that nothing calls them. So the two laws are *checked* by
  Bend's checker, and the verdict is a refusal to certify — the same
  shape as the HTTP packages. The metrics package's law has no such
  dependency and is certified. The price of not writing a generator of
  our own is a weaker guarantee, and the READMEs and `docs/AUDIT.md` say
  so.
- **Bend facts paid for today.**
  - A `def` with no return type *fills a law*: the checker looks for a
    law of the same name, so a helper that is not a proof needs a
    return type.
  - A self-call has to pass its arguments unchanged until one of them
    shrinks, and the shrinking one comes first. A walk that counts
    *up* therefore has to be written with the `1n+rest` pattern and the
    recursion on `rest`, and the fold it handles is `rest`, not `f`.
  - A `case` binder read twice needs `++`: `case 1n++rest:`. The `+`
    on a *def* parameter is a different marker.
  - A hypothesis `%f(args) : {shape}` must have the *goal's* shape, with
    `_` marking the part the call fills in — not the lemma's own
    shape.
  - A match cannot scrutinize a computed value — not even
    `Nat.is_eq(fold, f)`; the comparison needs its own def, or the
    value has to arrive as a parameter.
  - A record has no field access: `l.before` is not a term. The record
    is destructured by a match, and the fields come out as binders.
  - `Bool.pick` evaluates both branches, so it cannot be used to choose
    between two lists without building both.
  - The decreasing argument of a self-call has to be *structurally*
    smaller. A `Nat` counter does not shrink by arithmetic that the
    checker can see (`Nat.sub(k, 1n)` is refused); it shrinks by being
    unpacked (`case 1n+rest` and passing `rest`), the way
    `bend-ml-nat-lemmas` proves `add_comm`. A list shrinks by its tail.
  - `List.range(n)` takes the count alone, and it counts *down*:
    `List.range(3n)` is `[2n, 1n, 0n]`.
  - `List.append(a, -A, xs, ys)` takes the element type as its second
    argument, and its two list arguments are the accumulator and the
    element.
  - The PRNG's shuffle returns a pair: the list and the generator that
    comes after it, which has to be threaded by hand.

## 2026-10-21

- **The CI was red before this session, and the reason was the verdict.**
  `bend hf/main.bend` and `bend datasets/main.bend` end in "SOME PROOFS
  FAIL: 312 defs rely on unsafe or foreign code". A file with no `main`
  is checked by the *verdict* path (`cli_verdict` in `bend2/main.ts`),
  and the verdict refuses a book that holds any foreign code — the
  socket bindings of the http package are foreign code. So a package
  that does IO can never have a clean verdict, and the check now reads
  it as a pass when the *only* complaint is the foreign code (a type
  error has "expected" and "observed" in it).
  - The laws of `mlbend-metrics` are still proved by the kernel — that
    package is pure and its verdict is clean.
- **clang lives in the tree, not in /tmp (2026-10-21).** The wrapper at
  `~/.local/bin/clang` pointed its `LD_LIBRARY_PATH` at a directory whose
  libraries were symlinks into `/tmp/opencode`. `/tmp` is cleaned, and
  one morning the builds failed with "bend needs clang 14 or newer
  (found no clang)". The wrapper now names the tree's own libraries, and
  `scripts/setup.sh` asks for clang — the checks build every program
  before it runs it.
- **k-nearest neighbours over the breast cancer dataset
  (`examples/knn/main.bend`).** Plain Bend again: the neighbours are
  sorted by distance with `List.sort`, and the three nearest vote. The
  accuracy over the last fifth of the rows is 0.9781022, and
  scikit-learn's `KNeighborsClassifier(n_neighbors=3)` over the same
  split gives 0.9781021897810219 (`reference/test_knn.py`).
  - Two Bend details cost the most here. `List.sort` takes its
    comparator as a *template*: `List.sort(~Neighbour, ~nearer, xs)`,
    with the `~` at the call site.
  - A vote that counts the labels must count them over the whole list,
    not over the tail of the recursion: the count is computed once and
    the recursion only walks to the end.
- **The datasets are the UCI sets, not scikit-learn's subsets.** The `mstz`
  mirrors on HuggingFace carry the whole file: wine is 6497 rows (not 178),
  breast cancer 683 (not 569), heart failure 299. The loader is now one
  generic CSV reader instead of a function per dataset: `mlbend-datasets`
  fetches the file over the hub client, drops the header, reads every cell
  as a number (`True` is 1, `False` is 0) and takes the last column as the
  label. The labels are `F32`, not `Nat`: a class label is the index of the
  class and a regression label is the value, and the model decides which.
- **k-means over the wine dataset (`examples/kmeans/main.bend`).** Plain
  Bend: no gradients, no library, no tensors. The features are standardised
  first (the wine columns run from 0 to 160), the first three rows are the
  centres, and ten rounds of Lloyd's algorithm follow.
  - The result matches NumPy: the cluster sizes are the same (1071, 717,
    4709) and the inertia agrees to the fifth digit (57343.812 against
    57344.120). `reference/test_kmeans.py` runs the Bend program and
    `reference/kmeans_wine.py` is the NumPy side.
  - The arithmetic is done on whole rows, never by the place of a column:
    a fold over the rows adds the columns up (`zip_add`), a fold over a row
    adds the cells up. A row may then be short, which the wine file's last
    row is.
- **A counter is never a decreasing argument (2026-10-21).** The checker
  reads the arguments of a self-call left to right and wants the first
  changed argument to *shrink*: the parameter has to be a constructor and
  the argument a subterm of it (`term_descend` in `bend2/bend.ts`). So
  `f(Nat.sub(k, 1n), ...)` is refused: `Nat.sub(k, 1n)` is a call, not a
  smaller term. A loop over a number is written with `List.range(n)` and a
  loop over a list with the tail, and the index rides along as an
  accumulator. This is the same rule as "no forward references" and it
  costs the same number of round trips.
- **A def body has one result.** `def f(): e1` and then `e2` is a parse
  error: the second statement wants a new `def`. Several effects go in a
  `do IO<Unit>:` block, and a `match` heads a def body, so it cannot be a
  statement inside a `do` block: it goes in a def of its own.
- **The interpreter's stack is a few hundred frames.** `bend file.bend`
  overflows on a fold over 300 rows (`memory fault (machine stack
  overflow?)`); the same program built with `bend -o` runs. Every test in
  `reference/check_all.py` is therefore built and then run.
- **`scripts/reuse.py` and `scripts/sort_defs.py` (2026-10-21).** Bend
  consumes a value once unless it is bound with `+`, and the error for a
  value read twice ("consumed more than once") came up once per def.
  `scripts/reuse.py` marks the parameters, the pattern binders and the
  binds that the body reads twice; it is a script over the source, not a
  change to the language. `scripts/sort_defs.py` now keeps a comment with
  the def it describes: the comments at the end of a block belong to the
  next one.

## 2026-10-20

- **v0.1: `mlbend-metrics`.** The first package: the evaluation surface (MSE,
  MAE, RMSE, accuracy, precision, recall, F1) and the counts they rest on. The
  ten metrics of `metrics/tests/ok.bend` match NumPy and scikit-learn within a
  relative 1e-5 (`reference/test_metrics.py`). One law proved so far:
  `count_length`, the number of samples is the length of the label list.
- **The plan** is in `docs/PLAN.md`: the packages, the order and the stop
  criteria. The order is metrics -> linalg -> preprocess -> split -> linear ->
  cluster -> pipeline -> datasets -> select.
- **Bend traps this package paid for** (they are in `AGENTS.md`): `match` only
  inspects a parameter, so every comparison is computed by the caller and passed
  in (`same_len_of(Nat.is_eq(n, m), n, m)`); there is no `let` keyword (`x = v`,
  and `+x = v` to reuse); a def cannot be used before it is defined, so the
  helpers come first; `Result` is a `Type` and cannot be reused with `+`; the
  comparison operators are not terms (`Nat.is_eq`, `F32.is_eq`); a negative
  literal is not a term (`(0.0 - 0.5 : F32)`).
- **Debt, not in the package.**
  - The accounting law ("the samples predicted right and the samples predicted
    wrong add up to the number of samples") is not in v0.1. It needs the
    hypothesis that the two label lists have the same length, and the proof
    needs the counts to come out of one walk, so the statement can name both.
    It is v0.2's first law, with the `mlbend-linalg` inner-product lemmas.
  - The classification metrics are stated per class (`precision(ys, ps, k)`);
    scikit-learn also has a macro and a weighted average over the classes.
    Those come with the confusion matrix, in v0.2.
  - `r2` and the log loss are not here: both need the variance of the targets
    (a second sum) and a zero-denominator case of their own. v0.2.
- **The process, and where the memory lives.** An agent's durable context is
  this repository, not a transcript: `AGENTS.md` (the rules, the read order, the
  traps), `docs/agents/handoff.md` (where we are, what was run, what was not),
  `docs/agents/domain.md` (the vocabulary and the design rules),
  `docs/agents/issue-tracker.md` (how the issues carry the work), `NOTES.md`
  (what was measured and what failed), `docs/AUDIT.md` (the trust base) and the
  issues themselves (the debt). The laws are the real memory: a claim that
  survives the Lean kernel needs no log to stay true.
- **The bend repository stays where it is.** It is not vendored here: it is a
  fork with two remotes, it moves daily, and the copy would be wrong tomorrow.
  The pinned install (`~/.bend`) is the versioned reference (the guide and
  `base.bend` come with it), the hub cache (`~/.bend/lib`) is an offline mirror
  of the published packages, and `make sync-bend` fetches the pinned tag into
  `../bend-ref` when a read-only copy of the compiler is wanted.
- **Housekeeping on the bend fork**: the `/understand` knowledge graph (`.ua/`)
  is generated and must never be committed — it is now in that repository's
  `.git/info/exclude`, because it would otherwise fail bend's own repo gate (the
  allow list) the first time an agent ran `git add -A`.
- **The base layer exists (2026-10-20).** The plan was wrong about
  `mlbend-linalg`: the ecosystem already has the containers, the CSV reader,
  the bytes, the integers, the soft float, the proved lemmas and the parser
  combinators (`bend-kit-csv`, `-bytes`, `-int`, `-f64`, `-lemmas`, `-parse`),
  and bend-ml has the tensors and the reverse-mode autodiff. MLBend writes the
  *machine-learning* surface and imports the rest. A second `Mat` and a second
  set of lemmas would be the third copy of the same thing in the ecosystem, so
  the `mlbend-linalg` package is off the plan.
- **The numbers gap, and the ask (2026-10-20).** Base can print a number but
  not read one: no `U32.of_string`, no `F32.of_string`, and `bend-kit-f64`
  says so outright ("Decimal text is not here"). The data layer needs it, so
  the ask is upstream: bendlang/bend#1789, with the shape (Base's own
  `Result<&1, &1, U32 & String, A>`) and the law (`U32.of_string
  (U32.show x) == x`, provable). Until it lands, v0.2 carries a small decimal
  parser and drops it when Base has one.
- **The environment (2026-10-20).** The repository is a workspace now: a
  root `bend.toml` names the members (`metrics`), one `bend.lock` holds
  every package in the closure, and one `.bend/env` holds them all.
  `bendenv` is at <https://github.com/adeerkhan/bend-env>; it is a Rust
  tool that takes uv's shape (a global content-addressed cache, hard
  links into the environment, parallel downloads, one lock per
  workspace). `make env` builds it, `make env-check` verifies it and
  runs `examples/stacks/main.bend` over it.
  - **The example is the proof.** One program, one environment, two
    stacks: `bend-ml-tensor-array@0.1.14.0` computes a 2x2 product
    (`[12, 12, 12, 12]`) and `bendtorch-core@0.1.0.0` builds a tensor
    of four words. Both come from the lock.
  - **The ecosystem's tensors do not mix, and the tool says so.** A
    Bend type belongs to the package that declares it, so
    `bend-ml-tensor` and `bend-ml-tensor-array` are two different
    `Mat`, and so are the `G` of `bend-ml-autograd` and of
    `bend-ml-tensor-array`. `bendenv` reports every type two packages
    of the closure declare. The consequence: **the flat tensor library
    has no autodiff behind it** — `bend-ml-autograd@0.1.1.0` is written
    against the list-based `bend-ml-tensor@0.1.0.0`. That is the gap
    the deep-learning half has to close.
  - **The versions skew too.** `bend-ml-autograd@0.1.1.0` imports
    `bend-ml-nat-lemmas@0.1.0.0` while this repository asks for
    `0.1.1.0`; both are in the lock, and the tool says so. Not a bug:
    a Bend source names its dependencies exactly, so two versions can
    live side by side.
  - **The publish path is verified.** `bendenv hash <file.bend>`
    computes the hash a package would publish under, and on
    `aaaxn/bendtorch`'s checkout it gives
    `0x1689e5deb278c16abd3bd45eac101ef3` — the hash the hub holds for
    `bendtorch-core@0.1.0.0`, over the same 196 files. So the file
    set, the paths, the digests and the hash are the compiler's, byte
    for byte.
- **The datasets come from HuggingFace (2026-10-21).** Two new packages:
  `mlbend-hf` (a HuggingFace client: the dataset page, the file tree, a
  file's bytes, and the rows server's JSON rows) and `mlbend-datasets`
  (the datasets of scikit-learn, fetched at run time). The iris dataset
  comes from the file the dataset's own repository holds, which holds the
  same numbers scikit-learn ships: 150 rows of four features and the class
  of each, as a `Mat<150n, 4n>` and a list of the labels. Nothing but
  Bend: `bend-kit-http` for the HTTP, `bend-kit-json` for the JSON, and a
  parser of my own for the rows.
  - **The shape of the code is Bend's, not mine.** The checker will not
    let a match scrutinize a computed value, so every helper that
    unwraps a `Maybe` takes it as a parameter and matches on that, and a
    caller gets one call each. A def must also come before the defs that
    use it, which is a rule I broke often enough to write a sorter for.
  - **The toolchain is up.** `clang` 22 (the Fedora RPMs, unpacked into
    `~/opt/llvm-tree`, no root) and a native `bend` 2.0.37 at
    `~/.bend/bin/bend` (a compiled bun binary; a build is 0.8 s). Without
    clang the bend compiler cannot build anything.
- **The ecosystem's other package manager does not compile (2026-10-21).**
  `ezx` is on the hub: "ez, the package manager for Bend 2", by
  noah-emp, v1.5.0.0, 74 files. Its manifest is `ez.toml`, a ledger of
  every dependency with its hash and its origin (a hub name or a git
  commit with a NAR hash), it vendors into `.ez/lib`, and it pins tools.
  It does not compile against bend 2.0.37: `emerging-ezhttp` declares
  `Body.Text{text: String}` and Base's `Event` declares
  `Event.Text{text: String}`, and a program that imports both has two
  `Text` constructors in scope. `bendenv lock` now says so before you
  try:
  `Text: 0xf1c957a470368870a6d1d62a8c0cbe32, bend2/base.bend, ezx`.
  That is the second time the collision report has earned its place: Base
  is not on the hub and it moves, so a constructor it gains can break a
  package published before it.
  - **ezx pins its dependencies by hash, not by name.** All five of them.
    The resolver follows those now.
- **The tensors and the autodiff are bend-ml's and bendtorch's (2026-10-21).**
  Two packages of our own, `mlbend-nn` and `mlbend-torch`, re-implemented
  what bend-ml and bendtorch already provide, and they are gone. The
  examples use the libraries directly: the forward pass is `AG.linear`,
  the loss is `AG.ce_loss`, the gradient of the loss is `AG.ce_grad`, the
  backward of the layer is `AG.linear_bwd`, the step is `AG.sgd`, and on
  the other side the tape is `Backward.leaf`, `Backward.linear`,
  `Backward.cross_entropy`, `Backward.backward` and `Sgd.step`.
  - What this repository adds is the classic-ML surface around them:
    the metrics, the datasets, and the estimators that come next.
  - The logistic example over the online iris dataset loses 1.0986 to
    1.0672 in three steps. The learning rate is 0.01 because the
    features are not standardised: at 0.5 the loss *rises* (1.099 to
    1.537 to 6.88), which is the step overshooting and not a wrong
    gradient. `mlbend-preprocess`, which standardises, is what makes a
    larger rate sane.
  - The bendtorch path compiles and runs -- the data, the weights and
    the bias are built, the leaves go on the tape, the loss reads back --
    and the `linear` node still refuses the operands (error 106, an inner
    dimension mismatch). The shape convention of the weights (PyTorch's
    `[out, in]` or the mathematical `[in, out]`) and the rank of the
    bias are not settled yet; bendtorch's own MNIST demo is the
    reference.
- **The data (2026-10-20).** iris (150 rows, 4 features) and breast cancer
  (569 rows, 30 features) are in `mlbend-datasets/data/`, taken from
  scikit-learn's copies (which come from UCI). A hub package is one file, so
  the data lives with the program that uses it, not in the package: the loaders
  take a path.

## The 64x64 build's memory wall (2026-10-30)

`bend -o examples/diffusion/flowers_flow/main.bend` does not finish. On a 15 GB
machine it reached **9.3 GB RSS at 29 s** and was still climbing when a 15-minute
timeout killed it; the desktop went into swap and froze twice.

Measured with a cgroup (`MemoryMax=10G`, `MemorySwapMax=0`) and a 1 s RSS
sampler, bisecting the file by prefix:

| prefix | result |
|---|---|
| model only (up to the data section) | 0.6 s, builds |
| model + data (512 rows, 3 blocks) | **4.3 GB peak**, 7 s |
| + `cfg`, `mse`, `init_params`, `load_p` | past 6 GB, OOM-killed |
| + the training step | past 8 GB, OOM-killed |

The control is `examples/diffusion/vit/main.bend`, the same architecture at the
same 512 rows: it builds in **82 s under 8 GB**, into a 7.8 MB binary. So the
shape is buildable on this machine and ours is heavier by more than 2x, with the
training step's instantiation the largest single term.

`ulimit -v` is not a usable guard: Bun reserves address space, so a 4 GB cap
aborts the compiler at 2.3 GB RSS and the abort looks like a program bug. A
cgroup with swap disabled is the guard that works; `scripts/bend-guard.sh` wraps
any command in one (`BEND_MEM`, default 6G) and was verified to kill a 1 GB
overflow while the host stayed up.

Nothing was trained. The data export, the typecheck verdict, the example source
and `reference/flowers_report.py` all predate the wall and are unaffected.

### After the config split, and the guard (2026-10-30, later)

The example now takes its shapes from `config.bend`, the data is one grayscale
byte per pixel (`reference/flowers_export.py`, sha256 `6b9c4c66…`), and every
build runs under `scripts/bend-guard.sh`. The wall did not move with the model:

| build | result |
|---|---|
| `tiny` (1 image, 64 wide, 1 block), full program | **past 7.5 GB in 11 s**, killed |
| `tiny`, training only (sampler removed from the file) | past 7.5 GB in 10 s, killed |
| `tiny`, model + data + the sampler-input builders | 7.0–7.5 GB, 13 s, builds |
| `tiny` via the interpreter (`bend file.bend`, no `-o`) | past 4 GB, killed |
| ViT control, 512 rows, 3 blocks | **82 s under 8 GB**, 7.8 MB binary |

So shrinking the model 8x did not shrink the compile: the cost tracks how many
large recursive builders sit in one program, not the dimensions. The 4.3 GB
figure above is the pre-split file, whose model+data prefix stopped before the
sampler-input builders; adding those is what takes it to 7 GB at one block.

The next experiment is therefore structural, not numerical: **one program per
phase**, training and sampling built separately, each with only the builders it
uses. The ViT example proves the architecture fits in 8 GB when the program is
shaped that way.

### The wall was a three-value match (2026-10-30, found)

`tok_go`'s state machine matched three values at once:

```bend
match img row col:
  case 8n _ _:      ...
  case _ 64n _:     ...
  case _ _ 64n:     ...
  case _ _ _:       ...
```

That one shape costs **4.8 GB** of codegen. The same function with `match img
col:` (two values, three cases, same behaviour, `row` still a parameter) costs
**6 MB**. Measured on the same file with a `main` that prints `hi`, in a cgroup:

| build | peak |
|---|---|
| model + data, `match img row col` | 4.8 GB, 7 s |
| model + data, `match img col` | 6 MB, 1 s |
| full program, three-value match | past 9.5 GB, killed |
| full program, two-value match | **926 MB, 19 s** |

The cost is in the number of values a `match` scrutinises, not in the model
depth (1 and 3 blocks cost the same), not in the image size (32x32 dies like
64x64), not in the fuel (a runtime-equal fuel changes nothing), and not in the
feature builder (~400 MB of the 4.8).

With the two-value match the flow trains and samples:

```
medium (2 images, 128 wide, 1 block), 2000 steps, 32 Euler
  loss 1.316 -> 0.328
  samples  mean -0.128  std 0.363   against the data's -0.204 / 0.453
           8136 of 8192 in [-1, 1]
  legibility 0.419 (random baseline 0.048)
```

`legibility` is the best absolute correlation of a sample with any training
flower: 0.419 is a blurred flower, not noise. The run takes 2m03s on 28 cores.

### The 8-image run (2026-10-30)

`make flow64 PRESET=full` (8 images per step, 128 wide, one block, 2000 steps,
32 Euler) builds at 926 MB in 19 s and runs in 12m on 28 cores:

```
loss 1.369 -> 0.203 (min 0.185)
samples  mean -0.165  std 0.355   against the data's -0.204 / 0.453
         32466 of 32768 in [-1, 1]
legibility 0.344 (worst 0.233)    random baseline 0.049
```

Note the two runs disagree on the legibility proxy in the direction the loss
does not: the 2-image run reaches 0.419 with a final loss of 0.328, the 8-image
run 0.344 with 0.203. More images per step gives the better loss and the less
memorised samples, which is what a legibility proxy built on the training set
should do. Neither is a quality measurement beyond "these are blurred flowers".

### The 256x256 attempt is void (2026-10-30, corrected the same hour)

`PRESET=hi` builds in 940 MB / 18 s and ran 2000 steps in 2m33, but the run
does not measure a 256x256 model. The token builders advance on the literal
`64n`:

```bend
match img col:
  case _ 64n:   # the row is full
```

A pattern cannot be `TK()` (patterns are constructors and literals), so the row
length is baked into the builder, and at RES=256 the builder takes 64 pixels per
row and 64 rows per image: a quarter of each row, a quarter of the image. The
sample printer is hardcoded the same way, which is why the block came back as
256 lines of 64 numbers instead of 256 of 256.

The legibility 0.060 and the std 1.02 in the commit before this one are therefore
about a miscut sample block, not about 256x256. The build number (940 MB) is
real and is the useful part: resolution does not move the compile.

The fix for a real 256 preset is a builder that never matches a resolution: count
down from `TK()` and match only `0n`, and print with `TK()`.

### Attention and depth at 64x64 (2026-10-31)

Three runs, same budget (2000 steps, 2 images per step, 128 wide, 32 Euler,
28 cores), all on the 512-image grayscale export:

| model | attention | final loss | legibility | wall |
|---|---|---|---|---|
| 1 block | causal | 0.328 | 0.419 | 2m03 |
| 1 block | full | 0.328 | 0.427 | 4m30 |
| 3 blocks | causal | 0.325 | **0.450** | 17m02 |
| 3 blocks | full | **0.316** | 0.418 | 11m22 |

Full attention costs 2.2x the time and buys 0.008 legibility at one block: at 64
tokens the causal restriction was not what limited the model, even though it is
the wrong layer for an image (row 40 cannot see rows 41..63). Depth is the
lever: two more blocks buy 0.031 for 8x the time.

At three blocks full attention gives the best loss of the four (0.316) and the
lowest legibility of the pair (0.418 against 0.450): the same inversion the
2-image and 8-image runs showed, and the reason a training-set legibility proxy
cannot rank models. The wall times are machine variance, not the layer (the
11m22 full run and the 17m02 causal one are the same program shape).

The full-attention run imports the local `feat/full-attention` branch of
BendTorch (`examples/diffusion/flowers_flow/full.bend` points at
`../../../../bendtorch/torch/`), so it exercises the layer before the PR lands.

### The data is ordered by class, so a contiguous holdout is biased (2026-10-31)

`train-00000-of-00001-12de94e121bdbead.parquet` is grouped by label: the first
512 images are 12 classes (0..11), and the last 12 images are all class 11. A
holdout taken as "the last N" is therefore one class, and the training split
keeps only 9 of that class in its first 500.

The 500/12 split still measures *unseen images of a seen class*, which is what
the sample question needs, but it is not a per-class holdout and the number must
not be read as one. A proper split takes one image per class: a re-export that
writes the 500 training images and the 12 held-out ones as two pinned files.

### What the model actually learned (2026-10-31)

The 500/12 run, three blocks, causal, 2000 steps:

```
final loss 0.2625
legibility  trained 0.529 (min 0.525)   held out 0.227 (min 0.212)
            random baseline on the held-out set 0.032
            gap +0.301
```

Read the three numbers together. The samples resemble unseen flowers at 0.227
against a 0.032 baseline, so the model learned something transferable. They
resemble the training flowers at 0.529, so **more than half of the training-set
legibility is memorisation**. Every "0.450" in the table above is therefore an
upper bound on quality, not an estimate of it; the honest figure for unseen
images is 0.227.

The practical consequence is that 512 images is the binding constraint, not the
step count: longer training on the same 512 would raise 0.529 and leave 0.227
roughly where it is. The next lever is data (a larger export, or augmentation),
not more blocks, more steps or a different attention.

Caveats recorded with the number: the holdout is one class (the parquet is
grouped by label), and `final_loss` is the last batch's loss, not a held-out
loss.

### Legibility is a fraction of a ceiling, not a number (2026-10-31)

The proxy is "the best absolute correlation of a sample with any training
flower", so it depends on **how many images the reference set holds** and on what
a *real* image scores in the same comparison. Measured on the 3,874/102 split:

```
real held-out flowers vs the training set   0.562   <- the ceiling
noise                     vs the training set   0.055
the 512-trained model's samples, held out       0.414   = 73% of the ceiling
the same samples, against the 512 it trained on 0.450
```

So the earlier "held out 0.227, gap +0.301" was measured against **12** images,
one class, and is not comparable with 0.414 against **102**. The samples were
never as bad as that number suggested: on unseen classes they reach 73% of what a
real unseen flower scores.

Rules that follow, and are now in the reporter:

- report the **ceiling** beside the sample score, and read the sample as a
  fraction of it;
- never compare two legibility numbers taken against reference sets of different
  sizes;
- the random baseline stays as the floor (0.046–0.055), since it is the other end
  of the same scale.

### The corpus run, 3,874 images (2026-10-31)

Exported from the pinned parquet, evenly across classes, by
`reference/flowers_export.py --split`: 40 training and 1 held-out image per
class, 64x64 grayscale, all 102 classes.

| file | images | bytes | sha256 |
|---|---|---|---|
| `flowers64-train-gray.u8` | 3,874 | 15,867,904 | `1ff86747f9b070c3ed7b2058dd0c1f88317dfea931ed883379513f121385566e` |
| `flowers64-hold-gray.u8` | 102 | 417,792 | `f08730fe6730aacf440526420dfd645cd78f2ed1ef3fa82dacb1d90f2b978cf8` |

`make flow64 PRESET=corpus` (3 blocks, 2 images per step, 6,000 steps, 32 Euler)
builds at 926 MB and runs in 47 minutes. Loss 0.289 at the last step.

```
legibility  trained 0.536   held out 0.490   gap +0.046
ceiling     real held-out flowers vs the training set 0.562  -> samples reach 87%
floor       noise vs the training set 0.046
```

Against the 512-image model (73% of the same ceiling), the corpus moves unseen
flowers up 14 points and costs 0.046 of memorisation. The corpus was a
constraint; the remaining 13% is not answered by more of the same data.

### The proof shape the checker accepts (2026-10-31)

A law whose return type calls a recursive function does not reduce its branch
goal when the recursion is made inside a two-list match:

```
law accounting:
  {Nat.add(right(ys, ps), wrong(ys, ps)) == walked(ys, ps) : Nat}
```

`case Con{y, yt} Con{p, pt}:` leaves the goal at `right(y <> yt, ps)`, while the
recursive call returns the statement over `yt, pt`; the checker will not rewrite
one into the other, and six other shapes (a record threaded through the walk, an
accumulator induction, `+` binders, a `let` for the Bool) each failed at a
different point.

The shape that works is `tensor-array`'s, at `main.bend:1943-1980` in the
installed package: the helper (`ASet.node_t`) is defined **before** the recursion
(`ASet.facts`), takes the subtree's fact as a **parameter**, and states its
return type over the **constructor** (`ASet.Facts(T, ANode{xs, ys}, ...)`), so
the branch body is a call whose type already is the goal. Applied here:

```
def split_pair(... , hf: {Nat.add(right(yt, pt), wrong(yt, pt)) == walked(yt, pt)}) ->
  {Nat.add(right(Con{y, yt}, Con{p, pt}), wrong(...)) == walked(...)}

def split_ok(+ys, +ps) -> {Nat.add(right(ys, ps), wrong(ys, ps)) == walked(ys, ps)}:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      split_pair(y, p, yt, pt, Nat.is_eq(y, p), split_ok(yt, pt))
    case Nil{} _:
      {==}
    case Con{y, yt} Nil{}:
      {==}
```

The `_ _` base case does not refine `ps`, so the two lists need their three
cases written out. The same template carries `law rows_labels` in
`mlbend-datasets`: `rows_break` is defined before `rows_ok`, the accumulator's
invariant arrives as `h`, and `rows_step` is a closed Nat lemma over
`length_append`.

`law rows_labels` is type-checked but not kernel-certified: `mlbend-datasets`
holds the HuggingFace socket bindings, and the verdict refuses a book with
foreign code (314 definitions listed, before and after the law). It is the
trust base, not the proof, and `datasets/README.md` says so.

### cifar-10: the 3072-output DDPM (2026-10-31)

`examples/cifar/main.bend` is the digits skeleton at CIFAR shapes, and
`reference/test_cifar.py` is its NumPy twin.

The data is `arwics/cifar10` at revision `f44619ab`, `train.csv`, 54,902,548
bytes, sha256
`80f3ca81e56946a0eabc65f68b69558b4fdc741e1aafb0d9f552fcee768b3161`: 5,000 rows of
3,072 pixels (0..255, channel-height-width) with the label last. The first 256
rows are the training set, the pixels scaled by 255.

`n=256`, `hidden=128`, `inwidth=3080`, `steps=40`, `samples=8`, the 64-step
linear schedule, full-batch AdamW, the ancestral loop and DDIM.

| measured | value |
|---|---|
| build | 50.8 s |
| run | 65.4 s |
| twin | 5.4 s |
| fetch | 0.069 s |
| `first_loss` | 1.00143 vs 1.00145 (-2.518e-05) |
| `data_mean` | 0.474855 vs 0.474855 (1.689e-09) |

121 s against the 410 s estimate the issue sized from the probe, so the config
stayed where it was; the run's wall time in the manifest and `RESULTS.md` is
this one. `final_loss` and the sample statistics are report mode: 40 steps is
a pipeline, not a trained model. The legibility report reads 0.071 against
random images' 0.055.

Two findings came out of the gates, not out of reading:

- **`drop` counts from the right, and the issue's `1n` was wrong.** With
  `csv_of(..., False{}, 1n)` the label is the second-to-last column and one
  pixel column is dropped, so the rows hold 3,071 features and the data mean
  reads 0.47471285. `0n` is the label-last call the file's shape wants; it
  reads 0.47485462 and the 1e-5 gate passes. The issue text says drop=1 takes
  the label off the end; it does not.
- **The data mean is a sum-order number.** The example walks the pixels with one
  float32 running sum; `numpy.mean` adds pairwise, and over 786,432 values the
  two orders differ by 9.6e-5, ten times the gate. The twin therefore mirrors
  the running sum and prints the exact mean beside it; the tolerance is
  unchanged.

### The flat cifar-10 loss is the schedule, not the step (2026-11-01)

The task trains 40 steps and its loss barely moves (`1.0014539` after the first
update, `1.0163207` at the end), which reads as a step that does nothing. It is
not: the same program, with the noise level held fixed, trains.

`bench/cifar_learning/main.bend` steps the example's own model and its own
`Train.step`, with two schedules and three learning rates. Bend 2.0.38, the
cached data, one run each:

| run | steps | lr | at step 50 | last |
|---|---:|---:|---:|---:|
| A. t = 1..40 cycling (the example) | 600 | 1e-2 | 1.3992742 | 1.0026302 |
| B. fixed t = 32 | 600 | 1e-2 | 0.9728233 | 0.7803922 |
| C1. fixed t = 32 | 150 | 1e-3 | 0.8405294 | **0.5791386** |
| C2. fixed t = 32 | 150 | 1e-1 | 1.0839473 | 0.9990001 |
| D. fixed t = 32 | 600 | 1e-3 | 0.8405294 | **0.5286835** |
| E. t = 1..40 cycling | 600 | 1e-3 | 1.0016015 | 1.0006951 |

Three things follow.

- **The step works.** At t = 32 (`x_t = 0.7 x0 + 0.7 eps`) the loss falls from
  0.841 at step 50 to 0.529 at step 600, starting near 1.0 (1.0015 after the
  first update of the 1e-2 run). Gradients reach the weights, the optimizer
  updates them, and the input carries the target's information: a misaligned
  noise pair would hold the loss at 1.0, and it does not.
- **The schedule is what does not.** Cycling t = 1..40 visits each noise level
  15 times in 600 steps, and the loss stays at 1.0007, better only than the
  1e-2 cycle (which spikes to 1.399 and settles back to 1.003). One gradient
  step per level in sequence, each pulling a different way, leaves the weights
  where they started.
- **The example's lr is high for its budget.** At step 50 the 1e-3 runs are at
  0.841 where the 1e-2 runs are at 0.973 (fixed level) and 1.399 (cycle): the
  1e-2 step overshoots for the first tens of steps. The example is a 1e-2 run,
  so its printed losses carry that overshoot.

`final_loss` is report mode because 40 steps is a pipeline; that stays true. What
this entry adds is that the number is not a measurement of whether the model
learns, and that a task wanting to show learning needs a repeating level and a
1e-3-scale lr, not 40 single visits. Recorded as ISSUE-036 in the local ledger.

### The device lane trains the CIFAR step, 1.4e-3 from the CPU lane (2026-11-01)

The same probe on BendTorch's device lane: `bench/cifar_learning_gpu/main.bend`
keeps the example's parameter types and swaps the layer values for
`bendtorch-gpu`'s `LinearFirstF`/`LinearF`/`SiLU` and its `MSE`. Fixed
t = 32, lr 1e-3, 600 steps, the same init and the same inputs:

| step | CPU lane (`torch/main.bend`) | device lane, CPU threads | device lane, RTX 5060 |
|---:|---:|---:|---:|
| 100 | 0.64049715 | 0.64049715 | 0.64050174 |
| 200 | 0.5624825 | 0.5624825 | 0.5634144 |
| 300 | 0.5463061 | 0.5463061 | 0.5547442 |
| 400 | 0.53720737 | 0.53720737 | 0.53706175 |
| 500 | 0.5315328 | 0.5315328 | 0.53129405 |
| 600 | 0.5286835 | 0.5286835 | 0.52727836 |

Two things, one expected and one worth writing down.

- **The layer swap is exact on the CPU threads** — every printed loss is the
  CPU lane's to the last digit — so `bendtorch-gpu` and `torch/main.bend` are
  interchangeable as a training model, not only per layer.
- **On the GPU the curve tracks and does not match**: 0.6405 against 0.6405 at
  step 100, 0.5273 against 0.5287 at 600, a 1.4e-3 gap that compounds. This is
  the documented exp-ulp policy, not a defect: `SiLU` is `x/(1+exp(-x))` and
  the device's `exp` is a device function; `demo/gpu_layers_test.bend` already
  prints 176128 ulps for its SiLU case, and 600 steps of 807,168 weights carry
  that into the loss. A task that runs on the device lane therefore needs a
  measured tolerance, not bit equality.

**The device lane is not faster at this size.** 600 steps took 101.4 s on the
CPU lane and 136.7 s on the RTX 5060 (1.35x slower; the forward is 0.2 GFLOP a
step, and 600 steps of bangs cost more in launches than they save). That is the
same shape as bend-ml#23's table: at CIFAR scale with batch 256, the device
buys nothing until the work per step is much larger.
