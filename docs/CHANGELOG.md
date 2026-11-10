# Changelog

What changed for the people who run the lab. Internal refactors are not listed.

## Unreleased

### GLM-5.3-Flash: the sparse core is dense, batched and 65x faster

- **`sp_dense.bend`** replaces the per-query `List.set` scatter with a dense T x T score matrix,
  the reference's window-16 / stride-16 causal mask, one row softmax, and four matmuls for the
  forward and every gradient. The gathered list core in `attn_c.bend` stays as the oracle.
- **The gate** (`reference/test_glm53_sp_dense.py`, 8 checks) runs both cores at 32 and 128
  positions on the same q, k, v and dy. The largest differences are **8.941e-08** (y), 1.192e-07
  to 8.941e-08 (dq), 1.192e-07 to 3.576e-07 (dk) and 3.576e-07 to 5.960e-07 (dv), against a
  **1e-06** gate.
- **The sparse core's own forward and backward**: 32 and 128 positions, four heads, one pass
  each, `probe_time_cores.bend` on `cc2a7ca` and on this tree, median of three:
  **1,976 + 17,136 = 19,112 ms** old against **122 + 172 = 294 ms** new, a **65.0x** speedup.
- **The full model step is 7.0x, not 20x.** `train_h128 --steps 1` (the same command, old binary
  from `cc2a7ca` against the new one): **24.83 s -> 3.56 s** (median of three), peak RSS
  **15,940 -> 15,500 KiB**. The remaining time is the linear core and the per-head gather/rope/
  join, not the sparse core. Read the step-3 gate as the sparse core's own step; the model step
  is reported honestly at 7.0x.
- **`make check`** is **129/129** with the vectorized core in the model, including
  `test_glm53_attn_bwd.py` (12), `test_glm53_h_bwd.py` (33) and `test_glm53_sp_dense.py` (8).
- **Two bugs the forward comparison caught, both invisible to the gradients.** The first mask
  row omitted s = t, which moved y by 0.30 at 4 positions and 0.48 at 128 while every gradient
  stayed right; `y_of` also forgot to scale q by hd^-1/2, which changed y while the backward
  (which does scale) matched. This is why the gate checks y and every gradient.

### GLM-5.3-Flash: the four-block hybrid's full gradient is gated at N=8

- **`probe_model_bwd.bend`** runs one masked-cross-entropy backward through the whole
  74,144-parameter hybrid at N=8 and writes every parameter gradient in the parameter tree's
  dump order.
- **The gate** (`reference/test_glm53_h_bwd.py`, 33 checks) compares the 74,144 gradients per
  tensor against PyTorch autograd on the vendored modules (`reference/glm53_h_bwd.py`, golden
  `reference/glm53_h_bwd_golden.npz`). The loss is **5.5453690** against **5.5453691**
  (**2.673e-08** relative); the worst tensor is `b2/down` at **6.761e-07** relative, and the
  largest whole-tensor difference is **1.788e-07**. The moved-label gradient is rejected at
  **1.286** relative and matches its own bad-label golden at **3.701e-07**.
- **`make check`** gains the gate (one test file, 33 internal checks).

### GLM-5.3-Flash: the four-head attention backward is gated at N=8

- **`probe_attn_bwd.bend`** runs the whole `LinAttention` and `SpAttention` layer at N=8:
  the qkv projection, the split, RoPE, the four-head core and the out projection. It
  prints `y`, `dx`, the qkv weight gradient and the out weight gradient for each core,
  and the same for a deliberately moved `dout`.
- **The gate** (`reference/test_glm53_attn_bwd.py`, 12 checks) compares all eight tensors
  against PyTorch autograd on the vendored modules
  (`reference/glm53_attn_bwd.py`, golden `reference/glm53_attn_bwd_golden.npz`). The
  largest relative error is **3.231e-07** (sparse/dqkv); the linear ones are
  **1.58e-07 .. 3.17e-07**. The moved-`dout` gradients are rejected at **3.212e-01**,
  **1.763e-01** and **4.601e-01** relative, and match their own golden at **1.992e-07**,
  so the can-fail is the comparison's and not a broken probe.
- **`make check`** gains the gate (one test file, 12 internal checks).

### GLM-5.3-Flash: a trainable dense Stage A baseline, and its 400-step record

- **`examples/language/glm53_flash/train_a.bend`**: the 74,144-parameter Stage A model under
  `T.Train`, AdamW (lr 3e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.1) and the reference's
  masked cross-entropy. The data is the reference's synthetic Python corpus through its byte
  tokenizer: two fixed 64-byte rows, written once and reused, 128 tokens per step.
- **The 20-step gate** (`reference/test_glm53_dense.py`): every step's loss within **5.599e-07**
  relative of the PyTorch-autograd twin (`reference/glm53_dense_twin.py`), the 74,144 final weights
  at **3.455e-07** and the 74,144 final gradients at **2.157e-07** whole-tensor relative (the
  weights/gradients gate is 1e-3 because 20 AdamW steps amplify last-bit rounding). Can-fail runs
  with one weight moved (+2.0) and one input byte changed diverge at step 1 at **3.887e-03** and
  **4.485e-04**, and the same gate rejects them.
- **The 400-step record** (`results/glm53-dense/`): first loss **5.595126**, final **1.724139**,
  last-100 mean **2.0280423**, loss at the final weights **1.716634**, 51,200 tokens, build
  **86.059 s**, run **7.13 s**, peak RSS **57,040 KiB**. The fixed batch makes this an overfit
  baseline, not a corpus pass.
- **Greedy dev/confirm** (`gen_a.bend`, `model_eval.bend` at 192 positions,
  `reference/glm53_dense_eval.py`, the vendored evaluator): **0/64 solved, 0/192 tests** on dev
  (57 token-cap hits) and **0/64 solved, 0/192 tests** on confirm (62 token-cap hits); every family
  is 0/8. The manifest is frozen in `reference/glm53_dense_eval_golden.json` and gated.
- **`make check`** gains the dense gate (one test file, ten internal checks); it is **121/121**.
  The Stage B backward and Stage C remain open, and no upstream issue was filed.

### GLM-5.3-Flash: the 4-block hybrid is implemented, the corpus run is blocked by the sparse core

- **`attn_mh.bend` / `model_h.bend` / `weights_h.bend` / `train_h128.bend`**: the 4-block hybrid,
  three linear blocks and one sparse, width 32, four heads, 128 positions, SwiGLU 128, tied head,
  **74,144 parameters**. The multi-head wrapper gathers each head, runs the gated single-head
  fast linear core or the gated sparse core, and joins the four head outputs; the parameters and
  loader are separate from the dense Stage A tree.
- **The forward check found two bugs before any training conclusion.** A four-head forward at N=8
  against the vendored `LinearAttention`/`SparseAttention` caught an 8-wide/32-wide split (the
  q/k/v chunks are 32 wide, not one head) and a `Lin,Sp,Lin,Lin` block order (the reference is
  sparse on every fourth block; four blocks are `Lin,Lin,Lin,Sp`). After both fixes:
  linear **8.84e-08**, sparse **1.23e-07**, and the full four-block model **1.22e-07** relative
  against the NumPy twin (`reference/test_glm53_mh.py`, 4/4). The one-step training smoke had
  passed both bugs, which is why the forward check matters.
- **`reference/test_glm53_mh.py`** (4 checks) is in `make check`: the two four-head forward
  checks, the can-fail linear-vs-sparse check, and the full four-block model forward. `make
  check` becomes **126/126**.
- **The one-step smoke (fixed model)**: loss **5.5634737**, val **5.4198546**, **18.97 s** for
  128 tokens, peak RSS **16,956 KiB**, build **30.915 s** (`results/glm53-hybrid128/smoke.json`).
- **The blocker**: 128 tokens a step makes the dense baseline's 1,638,400-token budget a
  **12,800-step, 67.4-hour** job. A 400-step / 51,200-token smoke would take 2.1 hours and is
  32x below the comparison budget, so it is not a baseline. The sparse core's `List.set` scatter
  and the per-head gather/scatter dominate; the linear core is already fast.
- **Not claimed**: no 192-position accuracy is reported. The next step is a vectorized sparse
  core and a batched multi-head model, before the hybrid can enter the 1.6M-token comparison.

### GLM-5.3-Flash: a held-out family is not learned

- **Train on seven families, evaluate the eighth.** The 74,144-parameter dense model at 128
  positions, batch 32, 1,638,400 tokens, with `reverse` removed from the pretraining stream
  (`corpus_stream_holdout`): loss **5.5683017** -> **0.29021987**, held-out **0.27888998**; run
  **112.0 s**, peak RSS **440,836 KiB**.
- **Result**: dev **56/64** (168/192 tests), confirm **56/64** (168/192); `reverse` is
  **0/8 solved, 0/24 tests** in both splits, and all seven seen families are **8/8, 24/24**.
  The model memorises the seven templates and does not generalise to the unseen one.
- **Why it matters**: the seen-family benchmark is saturated, so MoE/HyperConnection/25.7M can
  only be justified by this kind of headroom or by efficiency. The held-out-family split is now
  a recorded benchmark and a frozen evaluation manifest.
- **Gate**: `reference/test_glm53_corpus.py` now validates the held-out data, both manifests and
  a corrupted copy, 19/19 internal checks; `make check` stays **125/125**.

### GLM-5.3-Flash: the prefix-state linear core

- **`examples/language/glm53_flash/lin_fast.bend`**: the prefix-state causal linear-attention
  core, `K_t/KV_t` forward and `N_s/Q_s` suffix backward. On the same 64 x 8 inputs as the direct
  core, y is bit-identical and dq/dk/dv agree at **6.07e-07 / 4.40e-07 / 2.75e-07** relative;
  the fast-only probe runs in **0.024 s** against the direct core's **11.4 s**. At 128 positions
  y is exact and dq/dk/dv agree at **3.27e-07 / 3.38e-07 / 4.23e-07**; the direct run takes
  **73.8 s**.
- **The integrated cores use it**: `attn_c.bend` now calls the prefix-state linear core, while the
  sparse core is unchanged. The Stage C 20-step gate still matches the PyTorch twin at
  **2.852e-07** on the loss, **3.705e-08** on the final weights and **1.487e-07** on the final
  gradients; the test takes **37 s** instead of 3 m 25 s.
- **The gate** (`reference/test_glm53_bwd.py`, now 28 checks): fast vs direct at 64 at 1e-5, plus
  the direct t=2/t=64 and integrated layer gates. `make check` stays **125/125**.

### GLM-5.3-Flash: the dense baseline is real, 61/64 -> 64/64 with the token budget

- **A corpus-trained dense baseline at 128 positions** (`train_dense128.bend`, `model_128.bend`):
  74,144 parameters, batch 8, 400 steps, **409,600 tokens**. Loss 5.574197 -> **0.32177755**,
  last-100 mean **0.3530558**, held-out **0.30741644**; build **53.786 s**, run **27.28 s**, peak
  RSS **157,436 KiB**. Greedy dev/confirm: **61/64 and 60/64 solved**, **186/192 and 184/192
  tests**, 64/64 valid programs, no cap hits. The three dev failures and four confirm failures are
  all `square` completions that return `x * 2` instead of `x * x`; the other seven families are
  8/8 in both splits.
- **The token budget closes them.** At batch 32, 400 steps, **1,638,400 tokens** (the reference's
  400-step budget): loss 5.5704083 -> **0.29012972**, last-100 **0.3012568**, held-out
  **0.27868035**; build **75.499 s**, run **108.47 s**, peak RSS **445,252 KiB**. Greedy
  dev/confirm: **64/64 and 64/64 solved**, **192/192 and 192/192 tests**, every family 8/8, no cap
  hits. The square confusion was a data/compute effect, not a formatting or capacity defect.
- **The earlier 0/64 was the fixed-batch, out-of-context run**, not a property of the dense model.
  The corpus baseline is the number to beat; the 74k dense model is already at the ceiling of this
  synthetic task at 1.6M tokens. Architecture work now has to be justified by efficiency or a
  harder task, not accuracy on a solved corpus.
- **The gates**: `reference/test_glm53_corpus.py` (14 checks) regenerates both window sets, checks
  next-byte alignment and hashes, reruns the 400-step batch-8 training and requires the loss curve
  to match at 1e-5, validates both dev/confirm manifests against their frozen goldens and the
  reference evaluator, and rejects a corrupted batch at step 1. `make check` is **125/125**.

### GLM-5.3-Flash: the masked-label gate and the training canary

- **The ignored-label branch is now gated** (`reference/test_glm53_mask.py`, 4 checks): a 2 x 3
  logits matrix with row 0 labelled 0 and row 1 ignored. The masked loss is **1.2019429**
  (expected 1.20194285), the ignored row's gradient is **exactly zero**, and the mean uses one
  valid row, not two. The 64-byte training batch has no ignored labels, so this branch was
  previously untested; a padded corpus batch will use it.
- **The learning canary** (`results/glm53-c/canary-16.json`): the Stage C model overfits the
  first 16 labels at lr 0.03 for 50 steps. Bend: **0.04192061** final loss, **0.03682779** at the
  final weights, build **22.869 s**, run **446.4 s**. The PyTorch twin: **0.04192125** and
  **0.03683004**, a relative difference of **1.5e-05 / 6.1e-05**. This proves the Bend loss,
  backward, AdamW and data can drive a loss down, not only match a twin.
- **The learning-rate sweep** (PyTorch twin, same fixed 64-byte batch, 400 steps): lr 3e-4
  **3.955**, 1e-3 **2.236**, 3e-3 **0.175**, 1e-2 **0.00263**. On a 16-label overfit, 200 steps:
  lr 3e-4 **4.651**, 3e-3 **0.723**, 1e-2 **0.00286**, 3e-2 **0.00041**. The 3e-4 rate is the
  reference's 25.7M rate; the 20-step parity gate is not a learning benchmark.
- **`make check`** is **124/124**.

### GLM-5.3-Flash: Stage C, the integrated linear + sparse model at 20 steps

- **`examples/language/glm53_flash/{attn_c,model_c,train_c}.bend`**: one batch, 64 positions, one
  head of width 8, **linear attention then sparse attention** (window 16, stride 16), each with a
  SwiGLU hidden 32, a final RMSNorm and the tied head: **4,168 parameters**. The sparse indices
  are exactly the reference's `s <= t and (s % 16 == 0 or t - s < 16)`; the sparse backward is the
  softmax backward on the gathered scores, with the scatter into dk/dv.
- **The 20-step gate** (`reference/test_glm53_c.py`, 7 checks): every step's loss within
  **2.852e-07** relative of the PyTorch twin (`reference/glm53_c_twin.py`), the 4,168 final
  weights at **3.705e-08** and the 4,168 final gradients at **1.487e-07** whole-tensor relative
  (gate 1e-3). The loss falls 5.5484023 -> 5.4986544. Can-fail runs with one weight moved (+2.0)
  and one input byte changed diverge at step 1 at **7.860e-03** and **6.585e-04**.
- **The layer cores** are also gated directly (`reference/test_glm53_bwd.py`, now 20 checks):
  RoPE + the linear core and RoPE + the sparse core on the same 64 x 8 inputs, against
  PyTorch-autograd goldens, all at **1.5e-07 .. 1.9e-06** relative.
- **`make check`** is **123/123**. Record: `results/glm53-c/run.json`.

### GLM-5.3-Flash: the Stage B linear-attention backward, exact at t=2 and 64

- **`examples/language/glm53_flash/lin_bwd.bend`**: the exact single-head causal linear-attention
  backward in direct O(T²d) form. With `c_t = clamp(den_t, 1e-6)`, `dnum_t = dy_t/c_t`,
  `dden_t = -dy_t·num_t/c_t²` and `a_{t,s} = dnum_t·v_s + dden_t`, the gradients are
  `dq_t = Σ_{s≤t} a_{t,s} k~_s`, `dk_s = Σ_{t≥s} a_{t,s} q~_t` and
  `dv_s = Σ_{t≥s} dnum_t (q~_t·k~_s)`, then the ELU chain.
- **The gate** (`reference/test_glm53_bwd.py`, 12 checks): at t=2, one head, hd=8, y/dq/dk/dv
  match a NumPy twin at **1.19e-07 / 1.49e-06 / 4.52e-07 / 6.47e-08** relative, and a one-hot
  dout at t=0 gives **dq = dk = dv = 0 exactly** for every t>0. At t=64 the 512-value tensors
  match the PyTorch-autograd golden at **1.51e-07 / 1.19e-06 / 1.17e-06 / 2.75e-07**. The
  one-hot case is the can-fail the old 64-step list implementation lacked (local ISSUE-090,
  closed).
- **`make check`** is **122/122**.

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
