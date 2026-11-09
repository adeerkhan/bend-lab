# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-11-09 — GLM Stage A becomes a trainable dense baseline

**Done and in the tree.** `examples/language/glm53_flash/train_a.bend`: the 74,144-parameter
Stage A stack under `T.Train`, AdamW (lr 3e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.1)
and the reference's masked cross-entropy (`loss_masked.bend`). The data is the reference's
synthetic Python corpus through its byte tokenizer: two fixed 64-byte rows, written once and
reused, 128 tokens a step. `reference/glm53_dense_data.py` writes the NumPy weights, the batch and
the frozen dev/confirm prompts; `reference/glm53_dense_twin.py` is the PyTorch-autograd twin.

**The 20-step gate.** `reference/test_glm53_dense.py` runs Bend and compares with the twin:
every step's loss max relative **5.174e-07** (gate 1e-4); the 74,144 final weights **3.455e-07**
and the 74,144 final gradients **2.157e-07** whole-tensor relative (gate 1e-3, because 20 AdamW
steps amplify last-bit rounding). Can-fail runs with one embedding weight moved by +2.0 and one
input byte changed diverge at step 1 at **3.887e-03** and **4.485e-04**; the same gate rejects
them.

**The 400-step record** (`results/glm53-dense/`, one fixed batch, so overfit by design): first
loss **5.595126**, final **1.724139**, last-100 mean **2.0280423**, loss at the final weights
**1.716634**, 51,200 tokens, build **86.059 s**, run **7.13 s**, peak RSS **57,040 KiB**.

**Greedy dev/confirm** (`gen_a.bend` and `model_eval.bend` at 192 positions,
`reference/glm53_dense_eval.py`, the vendored evaluator): **0/64 solved, 0/192 tests** on dev
(57 cap hits) and **0/64, 0/192** on confirm (62 cap hits), every family 0/8. That is the honest
result for a 74k model trained on one fixed 64-byte batch and evaluated beyond its training
length; it is not the 25.7M reference. The manifest is frozen in
`reference/glm53_dense_eval_golden.json` and gated.

**Gate count.** `env -u LD_LIBRARY_PATH make check` is **121/121** (one new test file; it prints
10 internal gates). The masked-CE gradient bug found during the work: `mask_one` called
`T.ce_pick` but not `T.ce_row2`, so the gradient was the softmax, not `(p - onehot)/n`; a 2x3
probe with an exact `T.CrossEntropy` comparison caught it before the 20-step gate.

**What is not built.** Stage B backward (the old 64-step list implementation is open as local
ISSUE-090; do not debug it), Stage C, the MoE, the four-stream HyperConnection, RL and the 25.7M
run. ISSUE-091 (BendTorch feature gaps: ELU, causal cumsum, top-k/gather, masked CE, clipping,
per-parameter scopes) remains **not filed**; no upstream issue was opened for this work.

## 2026-11-08 (latest) — GLM-5.3-Flash-from-scratch: Stage A and the linear-attention forward

**Done and in the tree.** `reference/glm53_flash/`: the educational 25.7M model vendored at
`96d9c7c`, MIT, with `VENDORED.md`; its 12 tests pass under `reference/.venv-rf` (torch
2.11.0+cu128). `examples/language/glm53_flash/`: Stage A, a **74,144-parameter** dense-attention
harness (width 32, four blocks, four heads of 8, 64 positions, SwiGLU 128, tied head) built from
BendTorch's existing `Embedding`, `RMSNorm`, `RoPE`, `Attention`, `SiLU`, `Tie` and
`CrossEntropy`; `model_a.bend`, `probe_a.bend`, `probe_rope.bend`, and `lin_core.bend`, the
Stage B linear-attention forward.

**The gate.** `reference/test_glm53_layers.py` runs under `make check` with no PyTorch: it
generates the weights and batch with NumPy, runs the probes, and compares a frozen
PyTorch-autograd golden. Six checks: Stage A logits **2.38e-07**; all 74,144 gradients
**2.24e-08**; loss 5.5654626 (golden 5.5654621); causality (first half exact, future changed);
Stage B linear forward **9.77e-04** on values to 4,096 (relative < 1e-6); RoPE **2.44e-04** on
values to 2,000 (relative < 1e-7). The RoPE bridge is explicit: the reference's interleaved
pairs, BendTorch's half-split pairs, and the q/k row permutation `[0,4,1,5,2,6,3,7]`. The
gradients are compared in the file layout, so the permutation is part of the gate.
`make check` is **120/120**.

**What is not built.** Stage B backward (dq, dk, dv); Stage C gathered sparse attention and its
backward; the MoE, the four-stream HyperConnection, RL and the 25.7M run. The list-form core's
reverse pass is the next work: the forward already matches, and the same prefix lists are the
state it needs. No upstream issue has been filed; this is local implementation, not a Bend
defect.

## 2026-11-08 (earlier) — the rectified-flow layer gate, its backward, and the bugs they caught

**Done and in the tree.** `reference/test_rectified_flow_layers.py` builds
`reference/probe_rectified_flow_layers.bend` once and compares fourteen whole outputs with a NumPy twin: `Down2`
forward and its exact-inverse backward (4,096 values each), the 2x2 max pool and its arg-max scatter (1,024 and
4,096), the zero-query XAttnRes mix (1,024), the learned-query mix on equal keys and on `kx = 2 * kh` (1,024 each),
its saved weights (256) and its five gradients (`dvh`, `dxc`, `dkh`, `dkx`, `dq`), and `Softmax(2,2)`'s axis. Every
gate passed; the permutation, pool, scatter, zero-query and forward mix gates are **0.0**, and the backward is under
5e-3 (`dq` 1.2e-4). The mix backward uses BendTorch's own `Softmax` backward; its `dq_sum` first stopped after row 0,
which the query-gradient gate caught at max |diff| 689 before it could ship.
`make check` is **119/119**, run from a shell with no exported `LD_LIBRARY_PATH` (the new build-env split in
`check_all.py` supplies the local `libLLVM` path, and `--verdict` still runs without it).

**The bug it caught.** `dot8` indexed the 1x8 query with the row offset, so every row after the first read past the
query's end, scored 0 and silently averaged 50/50. The row-0-only probes had passed because row 0 reads offset 0; the
`kx = 2 * kh` whole-array gate failed at once (max |diff| 1,023). Fixed in `model.bend`; only the whole-array twin can
see it, which is the lesson for every later matrix layer.

**Survey fixes, all in the same pass.** `check_all.py` gives the build checks the local `libLLVM` path only.
`rectified_flow_compare.py` resolves the newest `checkpoint.*.pt` under `--checkpoints` and documents `--resume` as a
warm start (the vendored `Trainer` has no step resume; the old `checkpoint.latest.pt` was never written). Both flower
examples fall back to `$HOME/.cache/bend-lab` instead of an absolute home path. The new
`examples/diffusion/rectified_flow/README.md` records the 16x16, dim-8, one-hour target and the layer inventory.
`reference/lucidrains/README.md` and `requirements-rf.txt` document `.venv-rf` and the rectified-flow vendor. The
README's gate count and its stale two-level UNet sentence are current; three broken Markdown links, the remaining
public `.local`/absolute-path references, the upstream table (#39, #36, #34, #33, #29, bend-ml #35, merged #32) and
`.gitignore` are fixed.

**Verification pass (2026-11-08, `93ce3ac`).** Clean tree; `env -u LD_LIBRARY_PATH make check` **119/119** and the
layer test **14/14**. The regression proof in a temporary worktree: the old `dot8` fails **7/14** (`mix2`, `mixw`, and
the five `mb_*` gates), the old `dq_sum` fails `mb_dq` at **6.890e+02** (13/14); the worktree was removed and the main
tree reran **14/14**. The defect-class audit over **78** tracked `.bend` files found one `1n+p` case without a
self-call, `flow_matching train_steps`, which delegates to `loop`; the only offset dot is `rectified_flow dot8`, and
the other dots are full-list. The docs scan found **0** missing relative links in **62** Markdown files and no
absolute home path in `*.md`, `*.bend` or `*.py`.

**The k=4 parse error is ruled out.** `do IO(T.Train<ModelP()>):` is invalid: the `do` form takes `IO<T>`, not
`IO(T)`, hence `expected '<' observed '('`. The scratch copy also calls a `do_step` it does not define, so the arm is
not runnable, but there is no Bend defect and no upstream issue. The rule is now in `AGENTS.md`'s pitfall list.

**What was not run.** The PyTorch 33M reference is still smoke-only (stopped deliberately); there is no reference
number for the 16x16 target. The XAttn wrapper, the memory-KV `FullAttention`, the DAG and the `rf16` task are the
next work.

**Open, in order.** 1. the XAttn wrapper (normalization, the LoRA key, the projection, the query as a parameter);
2. the memory-KV attention; 3. the DAG U-Net; 4. `rf16` plus the texture band. The k=4 arm is not runnable because
its scratch file lacks `do_step`; that is not Bend. The PyTorch 33M reference remains smoke-only.

## 2026-11-08 (earlier) — the UNet's fifth level: the metric paid, the texture did not

**Done and in the tree.** `examples/diffusion/unet/{model,train,config}.bend`: **five levels** (64 -> 32 -> 16 -> 8 ->
4 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64), 64 channels, no normalisations, the clean image as the target, 7,067 flowers.
Record `results/diffusion-unet/trained/` regenerated from the five-level run, write-up `TRAINED.md`, changelog entry.

**The numbers.** 298,439 parameters, 3,600 steps: loss **15.3262 -> 0.0259** (step 1 is 0.1343; the first value is the
untrained head on the first batch), 64 draws at Euler 32 with mean/spread **+0.325 / 0.179** against the data's
+0.384 / 0.232; **held-out legibility 0.547** against a ceiling of 0.574, **94.8%** of the span. The four-level record,
re-scored with the same evaluator, is 0.605 / **0.501** / 85.8% (published 0.498 / 85.4%), so the fifth level is +4.6
points of held-out legibility and +9.0 points of span. Trained legibility 0.633, above the ceiling: memorisation.

**The finding that matters.** The fifth level bought **global flower identity**, not petals or tone. The raw draws are
lower contrast than the four-level's (+0.325 / 0.179 against +0.381 / 0.238) and the top spectrum band is 0.034 against
0.039 (data 0.137). A deeper net fits a more confident conditional mean and MSE shrinks an uncertain mean toward gray;
the held-out correlation is scale-invariant, so it rewards the better structure. On the same 8 draws, 128 Euler steps
change spread 0.185 -> 0.174 and top band 0.033 -> 0.027, so the sampler is not the cause. **32 steps stays; the depth
line stops; the next lever is detail and contrast, not a sixth level.**

**The draw-count trap, again.** The five-level's first 8 draws score 0.575 held out (100.3% of span) against 0.547 for
all 64 (94.8%); the 128-step four-level run looked like +3 points for the same reason. A max-over-correlations statistic
depends on the sample size, so comparisons hold the draw count fixed. This is in TRAINED.md and the issue ledger.

**Gates.** NumPy **1.621e-05** at 2e-05; torch **2.098e-05**, 0.83x its allowance (2e-05 + 1e-6|v|); checkpoint
**bit-identical** at 298,439 weights (40 steps == 20 + 20, final loss 0.23739263 both). The twins cover nine
convolutions / 22 weight blocks.

**Open, in order.** 1. The contrast/texture mechanism: a sampler that adds detail, a loss that does not shrink the
amplitude, or a texture stage; a sixth level has nothing left to remove (bottleneck 2x2). 2. The four-different-batch
k=4 accumulation arm still is not running: k=1 `accumulate`+`flush` is **bit-identical** to `Train.step` on this UNet
(same 8,192-cell probe, loss 0.340704, probe md5 `d63d0fa3d2c716922ccae43a6847850b`), and four micro-batches with the
loss divided by 4 land on the **same weights**; the loop that reads four different batches hit a Bend syntax error
(`expected '<' observed '('`), ruled out in the later section as invalid `do IO(T)` syntax. 3. A report-card task still needs a reference arm; `flowers_flow` is
0.536 / 0.490 with 413,440 parameters, so the UNet's 0.547 is above it at 298,439.

**Upstream.** #34 (`tile`, 6aa3428) and #33 (`rope2d`, d82365e) are rebased, force-pushed to `fork`; `make check-tile`
PASS 0.000e+00, `make check-rope2d` PASS 1.192e-07, `PROOF.bend --verdict` ALL PROOFS CHECK on both. #32 is merged; #36
is tried and reported (see Open). Comments are on #32, #33, #34, #29, #27, #26, #36 and bendlang#1781. The next move is
nuxyel's merge.

**Environment.** Builds need `LD_LIBRARY_PATH=$HOME/.local/lib/llvm22:$HOME/.local/cuda-nvrtc/lib64`; `--verdict` must
run **without** it (Lean's own clang). Bend 2.0.40 is installed at `~/.bend-240/bend/bin/bend`; bend-lab still pins
2.0.38. Both venvs are on python 3.15; the five-level CUDA run took 55 min 11 s wall for 3,600 steps (0.92 s/step;
38 min 13 s user).

**Traps paid for.** A max-over-correlations statistic moves with the draw count. The eval's reference set must be the
training set. `Seq` returns a pair at every level, so nesting two Seqs nests the pairs. A batch must divide the image
count. A run shorter than one pass trains on a class-ordered prefix. A gate is never widened without the measured number
beside it.

## 2026-11-08 (earlier) — the UNet draws flowers: a fourth level, and the measurement fixed

**Done and in the tree.** `examples/diffusion/unet/{model,train,config}.bend`: **four levels** (64 -> 32 -> 16 -> 8 ->
4 -> 8 -> 16 -> 32 -> 64), 64 channels, no normalisations, the clean image as the flow target, 7,067 flowers. Record
`results/diffusion-unet/trained/`, write-up `TRAINED.md`, changelog entry.

**The numbers.** 224,583 parameters, 3,600 steps: loss **1.1831 -> 0.0298**; 64 draws with the data's mean and spread
(+0.379 / 0.246 against +0.384 / 0.232); **held-out legibility 0.498** against a ceiling of 0.574 — **85.4%** of the
span above the noise floor. The trained legibility is 0.603, above the ceiling: memorisation, stated as such. Gates:
NumPy **9.15e-06** and torch **1.18e-05** at 2e-05 (widened from 1e-5 when the fourth level added two convolutions;
measured value beside each), checkpoint **bit-identical** at 224,583 weights.

**The measurement, fixed twice this session.** The eval scored against the 3,874-image subset while the model trained
on 7,067, so `legibility` and the ceiling were both inflated and the previous record's "7,067 reference" label was
wrong (0.545 was against 3,874; the true figures were 0.560 / 0.439). The eval now defaults to the full export, prints
the reference set's size beside the ceiling, and prints a memorisation note when the trained legibility passes the
ceiling.

**Open, in order.** 1. The softness: the spectrum is steeper than the data's at every band, the bottleneck is 4x4
(16 numbers per image), so the levers are a **fifth level** or a sampler that adds detail. 2. The width arm
(CH = 96, 336,679 parameters, three levels) is running: if it beats 85.4% of the span it replaces depth, otherwise
depth is settled. 3. A report-card task still needs a reference arm; the comparison is `flowers_flow`
(0.536 / 0.490 with 413,440 parameters), and this UNet's held-out 0.498 is level with it at half the parameters.

**Upstream.** bendlang/bend#1071 (a comment with the measured thresholds) and nuxyel/BendTorch#36 (gradient
accumulation) are open as filed. The three BendTorch PRs are answered: #34 and #33 got their reference checks wired
into `make check`, their NOTES and limitations edits; #32 carries the #29 doc edits. The remaining obligation is the
**rebase and law recount** for #33/#34 once #32 merges.

**Environment.** Builds need `LD_LIBRARY_PATH=$HOME/.local/lib/llvm22:$HOME/.local/cuda-nvrtc/lib64` (the restored
`libLLVM` lives in `~/.local/lib/llvm22`; `~/opt/llvm-tree/usr/lib64` is read-only). Both venvs were rebuilt on python
3.15 after the system moved; bendtorch's uses its pinned torch 2.14.1+cpu.

**Traps paid for.** `Seq` returns a pair of parameters at every level, so nesting two Seqs nests the pairs. A batch must
divide the image count. A run shorter than one pass trains on a class-ordered prefix. The eval's reference set must be
the training set. A gate is never widened without the measured number written beside it.

## 2026-11-07 — the UNet draws flowers: a third level and the full corpus

**Done and in the tree.** `examples/diffusion/unet/{model,train,config}.bend`: **three levels** (64 -> 32 -> 16 -> 8 ->
16 -> 32 -> 64), 64 channels, no normalisations, the clean image as the flow target, and the **7,067** flowers of the
training split (a fixed permutation, sha256 `87ee501c…`). Record `results/diffusion-unet/trained/`, write-up
`TRAINED.md`, `reference/unet_flowers_eval.py`, changelog entry.

**The numbers.** 150,727 parameters, 3,600 steps: loss **0.5542 -> 0.0241** (the data's mean scores 0.0539); 64 draws
that are **flowers**; legibility **0.545** (96.6% of the span above the noise floor, 97.0% of the ceiling) and **0.438
held out** (75.5% of the span). Gates: NumPy **2.38e-06**, torch **3.93e-06**, checkpoint **bit-identical** at 150,727
weights.

**The four levers, measured apart.** Objective v -> x0: 0.274 -> 0.426. Third level on 3,874 images: +11.7 points of
span. Full corpus on two levels: **+0.0** (72.6% against 73.1%). All three together: 96.6% / 75.5%. Group norms
(0.233 with, 0.274 without) and batch size (2 -> 8 -> 26 for +0.5 points held out) were measured and rejected earlier.

**Open, in order.** 1. The softness: the spectrum is steeper than the data's at every band and the mean is 0.302 against
0.384, so the next lever is a **fourth level** (the bottleneck is 8x8 = 64 numbers per image) or a wider stack - not more
images, which is measured to do nothing on its own. 2. A report-card task for `unet-flowers` still needs a reference
arm; the honest comparison is `flowers_flow` (0.536 / 0.490 with 413,440 parameters), which this UNet now beats on
trained legibility and trails on held out. 3. The batch cannot exceed 2 at 64x64 (bendlang/bend#1071); accumulation is
requested as nuxyel/BendTorch#36.

**Upstream.** bendlang/bend#1071 (a comment with the measured thresholds) and nuxyel/BendTorch#36 (gradient
accumulation) are both open, as filed on 2026-11-06.

**Environment, repaired this session.** Python moved 3.14 -> 3.15 and orphaned `reference/.venv`; it was rebuilt on
3.15 (requirements + pillow, pyarrow, torch-cpu). `libLLVM.so.22.1` had been deleted from `~/opt/llvm-tree`; the real
147 MB file came back out of the cached `llvm-libs-22.1.8-4.fc44.x86_64.rpm` into **`~/.local/lib/llvm22`** (the tree's
`usr/lib64` is read-only). **Any build in this checkout now needs**
`LD_LIBRARY_PATH=$HOME/.local/lib/llvm22:$HOME/.local/cuda-nvrtc/lib64`.

**Traps paid for.** `Seq` returns a *pair* of parameters at every level, so dropping a layer keeps `Unit` and nesting
two `Seq`s nests the pairs (the third level cost three failed checks to this). A type error whose types carry deep
literals crashes the renderer (bendlang/bend#1291's symptom) - the workaround is a **batch-1 twin with small literals**.
`Up2`'s rows are the batch times the level's own rows. A batch must divide the image count or the last read runs past
the export.

## 2026-11-06 — the UNet draws flowers: the objective was the wrong one

**Done and in the tree.** `examples/diffusion/unet/{model,train,config}.bend`: 64 channels at each level, **no
normalisations**, the **clean image** as the flow target instead of the velocity, and the 3,874 flowers of the training
split. Record `results/diffusion-unet/trained/{run.json,loss.svg,samples.svg,stats.svg}`, write-up `TRAINED.md`, new
`reference/unet_flowers_eval.py`, changelog entry.

**The numbers.** 76,871 parameters, 4,000 steps: loss **0.6551 -> 0.0390** (a predictor of the data's mean scores
0.0534); the 64 draws are **blurred flowers**; legibility **0.426** (73.1% of the span above the noise floor) and
**0.330 held out** (54.2%) against a ceiling of 0.562. The previous record was 0.259 / 0.206 and its draws were
speckle. Gates: NumPy **9.54e-07**, torch **1.61e-06** (0.16x the allowance), checkpoint **bit-identical** (76,871
weights).

**What was measured and ruled out.** Group norms (0.233 with, 0.274 without, on more steps); batch 2 -> 8 -> 26 at
three resolutions (+10.5 / +4.4 points of span, saturating, still speckle at every resolution); capacity 76,871 ->
301,191 parameters (+1.3 points). The loss is not a progress measure here: the arm with the best loss had the worst
picture, twice.

**Open, in order.** 1. The blur: two levels with a 4x4 bottleneck carry 256 numbers per image, so a **third level** is
the next lever (not width - measured). 2. A report-card task for `unet-flowers` still needs a reference arm; the honest
comparison now is `flowers_flow` (0.536 / 0.490 with 413,440 parameters). 3. The batch cannot exceed 2 at 64x64
(bendlang/bend#1071); accumulation is requested as nuxyel/BendTorch#36.

**Upstream.** bendlang/bend#1071 — a comment with the measured 2.0.38/Linux thresholds (8,192 checks, 16,384 only on
an empty stack, 32,768 overflows) and the observation that the checker's budget is shared across module imports, so
splitting a file does not raise it. nuxyel/BendTorch#36 — a feature request for gradient accumulation, with the rows
argument and the numbers from this example.

**Traps paid for.** A batch that does not divide the image count reads past the export, and only on the last batch
(3,874 = 149 x 26, 3,872 = 484 x 8). A type error whose expected/observed types carry deep literals crashes the error
renderer (bendlang/bend#1291's symptom, no new issue) — the workaround is a **batch-1 twin with small literals**, which
prints the real mismatch. `Seq`'s parameter pair keeps `Unit` where a layer is dropped, and `Up2`'s rows are the batch
times the level's own rows.

## 2026-11-05 — the UNet trains and denoises; the channel order was the bug

`examples/diffusion/unet/{model,train}.bend`: the two-level U (64→32→16→32→64, 16 channels, a group norm before every
activation, the 32×32 skip, a 1×1 linear head) plus a **learned linear bypass** from the input, trained on 512 flowers
by rectified flow, with `--steps/--start/--save/--resume/--no-sample/--probe`. Record:
`results/diffusion-unet/trained/{run.json,loss.svg,samples.svg,stats.svg}` from 6,000 steps; write-up
`examples/diffusion/unet/TRAINED.md`.

**The numbers:** loss **1.2687 → 0.1652** against a zero predictor at 1.0645 (84% below); draws mean/std
**+0.359 / 0.262** against the data's +0.398 / 0.226. Structure in the right proportion, blurry rather than
recognisable; the levers are 32 channels, 8 images a step, 20k steps.

**The bug that cost the day:** the training pair builder and the sampler built the five input channels in opposite
orders (`[f4,f3,f2,f1,xt]` against `[x,f1,f2,f3,f4]`), so the field was learned on one permutation and applied to
another — the draws stayed at the noise's spread while the loss fell. Fixed; the spread fell from 1.01 to 0.262. Two
more of ours were fixed with it: `init_arr` zeroed offsets from the first two-channel layout, and `drawn` read the
sampler's state in pairs.

**Gates, exact:** NumPy twin 1.01e-06 (gate 1e-5), torch twin 1.13e-06, checkpoint bit-identical (5,495 weights).
`make check` green.

## 2026-11-05 (latest) — UNet level 2, the flow re-measured, item 4 opened

**Done and pushed.** The UNet's two-level U (`2e02ece`): 64→32→16→32→64, four `Conv2d`, four `GroupNorm`, two
`up2`, two skips; gates **5.96e-08** (NumPy) and **1.19e-06 / 7.63e-06** (torch, whole plane). The flow's "width win"
was a batch-size bug (`n()` had become 256, so half the targets were zero): with `n()` restored, **no setting beats the
baseline** (0.8360 final, 0.8859 last 100), and `compare_flow2d.py` now **passes** — first loss 1.476769 vs 1.476769
(8.97e-08) after fixing the data order, the per-step noise and the init scale (`0f6898a`). `reference/flow_config.py`
is the one reader of `hidden()`/`steps()`/`euler()` (`b4c1b49` for the docs). `make check` **116/116**, fast 31/31.

**Item 3 is blocked:** BendTorch PR #34 (`Tile`) is OPEN, so `LayerNormMod` + the gated residual wait. **Item 5,
nothing to do:** #32/#33/#34 are MERGEABLE with 0 reviews.

**Item 4, where it actually starts.** The four U32 facts are a `Word` bit-vector library: Base has only
`U32.add_comm`, and the facts (`J(j,2^d) == j`, `shr(2^(d+1)) == 2^d`, `Z` is bit `d-1`, the subtraction bound) each
need word inductions. **Six lemmas are proved** in a local scratch file (not committed), `--verdict` green: A
(`shr(shl(w)) == ctop(w)`), B (`ctop(w) == w` when `top(w) == False`), C1 (`init(shl(w)) == shl.put(False, init(w))`),
`top_shl`, and the canonical `2^d` — `p2(d,n)` (`d` zeros, a one, then zeros) with `shl_p2`. The builder had to match
`d` first and `n` inside the branches: matching `d` after `n` is the match-order trap, and two defs are mutual
recursion, which is forbidden. `init_p` and `shl.put` are the same recursion at different lengths.

**C is proved structurally:** `top_p2` gives `top(p2(d, d+2+j)) == False` for every `j, d` (induction on `d`; the
step is `p2(d+1,n) = F :: p2(d,n-1)` and `top_cons`). Only the 32-bit instantiation is left — `top(32, p2(d,32))`
needs `d + (2+j) == 32`, i.e. the `d <= 30` bound, which the *array* law must carry (a U32 holds `2^d`, so `d <= 31`;
`d = 31` is the top-bit edge). **D/F are a separate layer, not a longer induction:** Base has no law for `Word.and`, `Word.sub` or `Word.cmp`
(only `Word.adc`, `Word.add_comm`), so D needs `sub(2^d,1) = ones(d)`, a mask law and a comparison law; F needs
subtraction-without-borrow. Then `ASet.Apart` on perfect trees, `Mat.to_list(Mat.of_list xs) == xs`, and #29's adjoint.
Oracle: `tensor-array/tests/apart.bend`, 21,590 pairs, 0 failures.

**Posted upstream:** the six U32 lemmas and the three missing pieces are **bendlang/bend#2059** (2026-11-05), with the
program inline; the offer text is in a local scratch file (not committed).

**Lean:** 4.34.0 **is installed** (`~/.elan/bin/lean`), just not on the shell PATH — use
`PATH=$HOME/.elan/bin:$PATH bend FILE --verdict`. The earlier "no Lean here" note was the PATH.

## 2026-11-05 (earlier) — both examples fixed and re-recorded

**ISSUE-040 FIXED** (see below) and **ISSUE-044 FIXED**. 044's last bug was the image boundary: after a row's cells the
loop reset `col` and consumed the *next row's* cells before the transition, so the second image lost its last 16-value
`feats` block (the probe's 10224 vs 10240, drift at row 64). The transition now happens in the row-append step; a
control-flow simulation gave `B*TK` appends for `(64,64,2), (4,3,2), (64,64,1), (8,5,3)` before the code was touched.
`tok_go`/`stok_go` are one chain each (cells, rows, images; fuel first), each image takes `level(s, img)`, and
`twin_flow64.tokens_of` reads positions `k % TK`.

Verified: `reference/probe_flow_tokens.bend` prints 10240 values matching `tokens_of` to `5.96e-08`, no row over
tolerance; `test_flow64.py` **PASS** (forward 2.98e-07, euler 3.58e-07; it was 0.568 before); the 6,000-step record
passed — `legibility_heldout 0.484633` vs 0.489027 (delta -0.0044, tol 0.05), `final_loss 0.291029`, `params 413440`
exact, `results/latest/flowers-flow.json` updated.

**Next in the queue:** the U32 bit-lemma library (the four facts), then `ASet.Apart` on perfect trees and
`Mat.to_list(Mat.of_list xs) == xs` (bend-ml#26), then #29's adjoint. PRs #32, #33, #34 are still open upstream.

## 2026-11-05 (later still) — 040 fixed and re-recorded; 044 is one missing block

**ISSUE-040 is FIXED** (committed with its record). The probe showed the `List.reverse` in `ab_img`/`te_img` was
correcting **three** things at once — image order, each pixel's `(a,b)` pair, and each 8-value `temb` block. The fix
does them at the source: drop both reverses, cons `(sqrt(abar), sqrt(1-abar))` in `ab64_rev`, reverse each `temb_of`
block in `te_rev`; `twin_vit.levels_of(s) = t_of(s, i)`. Re-recorded through the guard (a plain `bend` run of the
example dies with `machine stack overflow`): `BEND_MEM=9G scripts/bend-guard.sh bend -o /tmp/vit_bin …`, run it,
`reference/vit_results.py /tmp/vit_ln.txt`, then the gate — `first loss delta 2.40e-08` (tol 1e-5), forward 1.19e-07,
legibility 0.795 (gate 0.75), PASS. The recorded first loss moved 1.0091865 → 1.0093014, as the level assignment should.

**ISSUE-044 is localised to one block.** `reference/probe_flow_tokens.bend` (new) prints the flow's `sample_input(0.5,
synthetic)`: 10224 values against the twin's 10240 — **one 16-value `feats` block missing**, the drift starting at
index 5184 = row 64, column 64, the **first row of the second image** (rows 0..63 match value for value). So the row
reset is right and the image-boundary transition appends 63 blocks for the second image instead of 64. The example is
reverted to green; fix the transition, verify with the same probe, then `make experiment TASK=flowers-flow` and
`reference/test_flow64.py`.

## 2026-11-05 (later) — 040 solved on paper, 044 localised, both blocked on re-records

**ISSUE-040.** `reference/probe_vit_tokens.bend` (new) prints `step_in(0, synthetic)` so the token order is read, not
inferred. It shows row 0 = image 0's pixels with level `t_of(0,63)`'s coefficients and `temb`, **no pair swap** — so the
`List.reverse` in `ab_img`/`te_img` is correcting three things at once: image order, `(a,b)` pair order, and the 8-value
`temb` block. Fix: drop both reverses, cons `(sqrt(abar), sqrt(1-abar))` in `ab64_rev`, reverse each `temb_of` block in
`te_rev`. Probe-verified: all 1024 rows match `twin_vit`'s reconstruction to the last float digit. **Not done: the
recorded run.** A direct `bend examples/diffusion/vit/main.bend` dies with `machine stack overflow`; it must run as
`BEND_MEM=9G scripts/bend-guard.sh bend examples/diffusion/vit/main.bend > log`, then `reference/vit_results.py log`
writes `run.json`, then `reference/test_diffusion_vit.py`. Reverted until that lands.

**ISSUE-044.** The images → rows → cells rewrite (fuel first, because Bend needs one shrinking argument) compiles and
`twin_flow64.tokens_of` uses `k % TK`, but `test_flow64.py` gives `ok input` (first two tokens) and `FAIL forward
0.568`: rows 64..127 still disagree. Next instrument: print the example's `inp` at rows 64/65 inside
`probe_flow64.bend` and compare with `tokens_of`'s rows, then re-record with `make experiment TASK=flowers-flow` and
`reference/test_flow64.py`. Reverted to green.

## 2026-11-05 — our own debt: one fixed, one found deeper, the rest recorded

**Fixed and pushed (`bend-lab` main).** **ISSUE-042**: a generated dataset's pin now lives in the task
(`dataset.pin: roots + name -> sha256`), `check_pins` enforces it and `run_bend` calls it **before the build**;
`--check-dataset TASK` runs that check alone. Tested both ways — `flowers-flow: 2 pinned file(s) ok`, and a
one-byte-flipped copy in a temporary root refused with both digests. `_flowers_paths()` now calls the same function, so
the pre-run check and the parse cannot drift.

**Recorded.** ISSUE-011 closed (`FullAttention`, v1.3.0); ISSUE-026 narrowed to `LayerNormMod` + the gated residual
(RoPE2d is PR #33, Tile is PR #34); ISSUE-038 accepted with the decision written down (the target's fill landed; the
input's path needs a list, which a fill cannot produce).

**ISSUE-040 is deeper than the ledger said, and was reverted.** Dropping the two `List.reverse`s and setting
`twin_vit.levels_of(s) = t_of(s, i)` takes the twin's step-0 loss to 1.0093014 against Bend's 1.00921 — a **9.2e-5**
gap, where the pre-fix pair matched to **6.3e-9**. Reversing a 128-value row also swaps each pixel's `(a, b)` pair, and
`twin_vit`'s token order was derived from the reversed rows. So the fix needs `twin_vit.tokens_of` re-derived first,
then the example re-run and `vit_results.py` re-recorded. The tree is green (`first loss: ... delta 6.27e-09`) and the
finding is in ISSUE-040.

**ISSUE-044 is the one left, with its plan in the ledger**: rewrite `tok_go`/`stok_go` as images → rows → columns
chains (killing the stale `case 8n _` guard, restarting the row per image), `pair_of` passing `level(s, img)`, the twin's
`tokens_of`/`feats` following, then `make experiment TASK=flowers-flow` (6,000 steps) and `reference/test_flow64.py`.
Do 040 and 044 in one change so the two decisions are consistent.

## 2026-11-04 (later still) — three PRs are up; next is the U32 bit-lemma library

**Where things stand.** Three PRs against `nuxyel/BendTorch:devel`, all from `adeerkhan/BendTorch`, all green on CI
and on the local gates:

- **#32** `im2col-roundtrip` (4 commits): `conv_roundtrip_ks`, `conv_roundtrip_window`, `divmod_go_spec`/`divmod_spec`,
  the decode. 37 laws in all.
- **#33** `rope2d` (3 commits): `RoPE2d`, its three laws, its `Nat` addressing defs, `reference/rope2d_check.py`. 38.
- **#34** `tile` (1 commit): `Tile`, its two laws (`tile_row`, `tile_sum`), `reference/tile_check.py`, and the timings
  comment on #19 (`#issuecomment-6638085831`). 37.

**#19's timings, which closed its open question:** at 512 x 128, 20,000 iterations, medians of three — one elementwise
pass 1.49 s (74.5 µs a step), `Tile` then the same pass 1.45 s (72.5 µs a step). The broadcast is not measurably
slower, so a fused `Modulate` buys nothing at this shape; #19's cost was the host staging, and `Tile` removes it.

**The conflict to expect:** all three touch `LAWS.bend`, `PROOF.bend` and the same docs lines. Whichever merges second
needs a one-line law-count recount (37/38/37 in all, and the badge), and third needs two. All three PR bodies say so.

**Next item in the queue:** the **U32 bit-lemma library** (the four facts: `ASet.J(j,2^d) == j`, `U32.shr(2^d) ==
2^(d-1)`, `ASet.Z(j,2^d)` is bit `d-1`, `U32.sub(j,2^(d-1)) < 2^(d-1)`), as a standalone file first in the pattern of
nuxyel's `docs/upstream/array_set_laws.bend`; then `ASet.Apart` on perfect trees and `Mat.to_list(Mat.of_list xs) == xs`
for bend-ml#26; then #29's adjoint. After that: the UNet's second level, and the lab's ISSUE-040/044 fixes.

## 2026-11-04 — everything we filed is closed; the pin is 2.0.0

**Where things stand.** `bendtorch@2.0.0.0` is on BendHub with its three lanes
(v2.0.0, 2026-11-03). #27 merged (the config lives in the `Train`), #22 merged
(`Conv2d`), #24 and #15 closed, 35 laws, 167 mutants; the release notes credit
the Conv2d work to "by or with @adeerkhan". Our two remaining upstream issues
are both performance findings from yesterday: **#26** (launches cannot be
counted) and **#28** (the attention block is 3.2x its arithmetic).

- **The next change is the 2.0.0 migration, and its scope is measured, not
  guessed**: 21 files import `bendtorch@1.3.0.0`, 13 `Train.step` calls lose an
  argument, 59 `T.Train{…}` patterns gain `c` after `st`. `tensor-array` stays
  at `0.1.18.0`; nothing else moves. Two files (`examples/cifar/main.bend`,
  `bench/cifar_learning`) are already ported on branch `devel-200` against a path
  import — they need the import swapped to the hub package.
- **Owed to nuxyel, twice asked**: the CIFAR sweep on the released
  `2.0.0.0` (we ran it on `devel@211ae84`: 0.5790425 / 0.87949044 / 0.99899477
  against 1.3.0.0's 0.84698695 three times).
- **The UNet is unblocked** and is the next task after the migration:
  `Conv2d` (2.0.0), `GroupNorm` and `FullAttention` (1.3.0) are all in the pin;
  #15 is closed with "an unconditional UNet is expressible".
- **New gate**: `reference/corpus/qwenbend/` + `reference/test_corpus.py` (200
  model-written programs, 116/116 values, 84/84 diagnostics, three negative
  controls). `make check` is **113/113**.

**Next, in order.** 1. The 2.0.0 migration in one change (21 imports, 13 calls,
59 patterns), then the sweep on the hub package, posted on #24. 2. The UNet task,
twin and cell first. 3. `FullAttention` at `AttBlk`'s shape for #28. 4. bendlang
#1632 items 9–10, with the corpus's 84 rejections as the input.

## 2026-11-04 (night, later) — the UNet's forward runs; one channel is 1.88e-4 out

**Where things stand.** `examples/diffusion/unet/main.bend` now composes and runs
the smallest honest U (64x64 -> 32x32 -> 64x64, 2 channels, one additive skip,
80 parameters: `Conv2d` s2 -> `GroupNorm` -> `SiLU` -> `Up2` -> `Conv2d` s1 ->
`GroupNorm` -> `SiLU` -> `+ x`). `reference/unet_gate.py` rebuilds it in NumPy.

- **What matches**: channel 0 of every cell, exactly. That pins the flat row
  order, the channels-last patch, the stride-2 surplus, `Up2`'s order and the
  skip — an error in any of them would move both channels.
- **What does not**: channel 1, by **1.86e-4**, whose `GroupNorm` scale is
  `-0.01` so its normalised value is ~19.9 and residuals are amplified ~100x.
  ISSUE-051 lists the three candidates: the group's variance/`rstd` order, the
  conv's 18-term accumulation, and where the bias enters.
- **It is a probe, not a gate** — the file is `unet_gate.py`, not `test_unet.py`,
  because a tolerance wide enough to pass would hide an unexplained difference.
  `make check` stays at **113/113**; `test_up2.py` (the decoder's upsample) is
  the UNet's one green gate, with the adjoint identity exact.
- **BendTorch #29's two proofs are not started.** Lean 4.34.0 is on this machine
  (`~/.elan/bin/lean`, put it on `PATH`); `bend PROOF.bend --verdict` reads ALL
  PROOFS CHECK on the current branch, so the proofs can be written and checked
  here rather than in CI.

**Next, in order.** 1. Close ISSUE-051 (the 1.86e-4), then rename `unet_gate.py`
to `test_unet.py` and let it into the gate. 2. The rest of the UNet: the second
down/up level, the bottleneck, a `Layer` composition (Seq/Fork/Both/Add2/Id, so
the backward comes free), then the probe/twin/cell and a flowers64 run. 3. BendTorch #29: **done, at any stride and padding** — `conv_roundtrip_window`
(BendTorch `a2732c7`) says slot `(ch, ky, kx)` of `(oy, ox)` is image cell
`(oy·s + ky − p, ox·s + kx − p)`, and `conv_roundtrip_ks` is its instance. What is
left in the conv line is Conv2d's adjoint (#15), which needs `torch/spec.bend` to
grow indexed accumulation (`nth`/`set_at`) first: `col2im` writes into an array
by index and the checker cannot follow that (`docs/limitations.md`).

## 2026-11-04 (night) — the UNet's decoder block exists and is gated

**Where things stand.** The UNet is started at its hardest point and stopped at a
tested boundary, not mid-file.

- **`Up2`** (`examples/diffusion/unet/main.bend`): the 2x nearest-neighbour
  upsample a UNet's decoder needs, because BendTorch has `Conv2d` but no
  transposed convolution and no channel concat. Forward repeats each pixel 2x2
  (vertical pass then horizontal, over the flat row-major `Mat`); backward is the
  exact adjoint, summing the four cells.
- **`reference/test_up2.py`** (in `make check`): forward equals NumPy's repeat on
  a 2x2 image of 2 channels, `<up(x), y> == <x, down(y)>` exactly (816 = 816),
  and a one-cell error is rejected — the gate can fail.
- **What is left for the model**: compose `Conv2d(1->16, stride 2)` →
  `GroupNorm` → `SiLU` → `Conv2d(16->32, stride 2)` → bottleneck (`Conv2d` +
  `FullAttention` at 16x16) → `Up2` + `Conv2d(32->32)` + an **additive** skip
  (channels must match, since there is no concat) → `Up2` + `Conv2d(32->16)` +
  skip → `Conv2d(1x1, 16->1)`. Then the probe, the twin, the cell
  (`bench/unet_first` + a `unet-first` task, matrix 19 → 20) and a flowers64 run.
- **BendTorch #29 is not started**: two proofs (the general `k = s` round trip
  and Conv2d's adjoint), with Lean 4.34.0 available locally so `--verdict` can
  check them here. It is the next session's first item.

**Next, in order.** 1. The UNet's model composition (the block list above), then
its probe/twin/cell. 2. BendTorch #29's two proofs. 3. bendlang #1632 items 9–10.

## 2026-11-04 (later still) — 2.0.0 landed, #28 answered, the UNet's first fact checked

**Done this session.**

- **The pin is `bendtorch@2.0.0.0`**: 22 files, 21 imports, 13 `Train.step` calls
  without their configuration, 59 record patterns with `c`, in one commit.
  `make check` **113/113** and `make matrix` **19 cells, 0 failed** on the
  migrated tree — no recorded number moved. The sweep on the hub package reads
  **0.5790425 / 0.87949044 / 0.99899477** (posted on #24, closed).
- **#28 is answered**: masked full attention is **9.55 ms** a pass against
  causal's **10.65 ms** at the block's shape, so the mask is not the cost — the
  core is 64 tiny 16x16 products, and the lever is fewer, larger calls. Posted
  with the caveat that today's machine is slower: compare within a run.
- **The UNet's hardest fact is checked**: `Conv2d(1, 1, 8, 3, 63, 63, 1, 1, 0)`
  on one 64x64 image is `Mat<4096,1>` -> `Mat<1024,8>` (oh=ow=32) with an
  im2col state of `Mat<1024,9>`, and it runs: a constant image gives **0.4** at
  the first cell, exactly 4 of the 9 weights, which is what a padded corner
  should read. The UNet itself is the next session's work, not this one's.

**Next, in order.** 1. The UNet: probe (`reference/probe_unet.bend`, frozen
weights, forward + one Euler step + the 8/16/32 endpoints), twin, gate, cell
(`bench/unet_first`, `unet-first` task, matrix 20 cells), then a flowers64 run.
Start at `tiny` (base 16, two downs, two ups) and let `make doctor` pick the
preset before the build. The block is `Conv2d(k=3, hs=63, ws=63, p=1, s1=1)` ->
`GroupNorm` -> `FullAttention`; the corner-cell check above is the shape's own
unit test. 2. bendlang #1632 items 9-10 from the corpus's 84 rejections.

## 2026-11-04 (later) — on 2.0.0.0, matrix pending, UNet next

**Where things stand.** The pin is `bendtorch@2.0.0.0` and the migration is
done: 22 files, 21 imports, 13 `Train.step` call sites, 59 record patterns, in
one commit. `make check` **113/113**; the learning-rate sweep on the hub package
reads 0.5790425 / 0.87949044 / 0.99899477 (posted on #24, closed). `make matrix`
was started at the end of the session — **check `results/matrix/SUMMARY.md` for
whether the 19 cells finished** before quoting it.

- **#28 answered** with a measurement: masked full attention is **9.55 ms** a
  pass against causal's **10.65 ms** at the block's shape, so the causal mask is
  not the cost; the core is 64 tiny 16x16 products and the lever is fewer,
  larger calls. Posted.
- **The one thing left from the plan is the UNet**, and it is 1-2 sessions:
  `Conv2d` (2.0.0), `GroupNorm` and `FullAttention` are all in the pin, #15 is
  closed with "an unconditional UNet is expressible", and the shape is
  `k=3, s=2, p=1` at 64x64 with `hs = ws = 63` (the release notes' own example).
- **No upstream issue of ours is open on a decision**: #26 and #28 are the two
  performance findings, both answered or waiting on their reading; bendlang
  #1632 items 9-10 are ours to write, with the corpus's 84 rejections as input.

**Next, in order.** 1. The UNet: `reference/probe_unet.bend` (frozen weights, one
forward, one Euler step, the 8/16/32 endpoints), `reference/twin_unet.py`,
`reference/test_unet.py`, `bench/unet_first` + a `unet-first` task and cell, then
a flowers64 run. Start at `tiny` (base 16, two downs, two ups) — a 3-block flow
reached 9 GB to compile, so `make doctor` picks the preset before the build.
2. The matrix, if it did not finish. 3. A diagnostics spine from the corpus.

## 2026-11-04 — the QwenBend corpus is a gate now

**Where things stand.** `reference/corpus/qwenbend/` holds 200 Bend programs
written by Qwen3.5-0.8B and checked against real Bend2 upstream; `make check`
now runs them through the Bend we pin. All four checks agree (116/116 verdicts,
84/84 rejections, 116/116 values, 84/84 diagnostics), and the test's own
negative control corrupts a verdict, an output and a diagnostic to prove the
comparison can fail. 36 s.

- Why it matters: it is the first conformance test of the *checker* that we did
  not write, and it fails on a Bend bump that changes the static rules or the
  error messages.
- Provenance: Apache-2.0, pinned by sha256, re-projectable from
  `bendtune/logs/tuned400_test.jsonl`; the snippet is in the corpus README.
- Not taken from that project: the in-weights executor, the LoRA, the 4B
  checkpoint. Their benchmark is honest that it is less accurate than asking the
  model directly (50% vs 70%); the win there is exactness and a replayable trace.

**Next, in order.** 1. Watch #22; on merge, re-run our gates and start the UNet.
2. On BendTorch 2.0.0's release, finish `devel-200` (11 call sites) and re-run
the CIFAR sweep as acceptance. 3. The `FullAttention`-at-the-same-shape
comparison for #28. 4. Consider a spine for the diagnostics: the corpus's 84
rejections are a ready-made input for bendlang #1632 items 9-10.

## 2026-11-03 (night, later) — #27 lands, #22 is rebased, 2.0.0 is staged

**Where things stand.** nuxyel answered all three open threads in one go.

- **#24 → #27, merged to `devel`, shipping as 2.0.0.** `Train` carries its
  config, `Train.step` drops the argument, `Train.with_cfg` changes it, and
  `Opt.guard` refuses a kind mismatch with a NaN optimizer (four laws). He built
  all four candidate fixes on branches before choosing ours, and found the worse
  half we had missed: a kind mismatch used to keep the loss and raise nothing.
- **Our sweep on it** (`devel@211ae84`, 150 steps, one lr at `Train.new`):
  **0.5790425 / 0.87949044 / 0.99899477** where 1.3.0.0 read 0.84698695 three
  times. The rows match the old per-step form to the digit, so adopting it moves
  no number. Posted on #27 with three review notes (the guard is the only shape
  a pure step can have; `with_cfg` keeping `p1`/`p2` is what makes a mid-run
  change safe; and a question about `dump`/`restore` carrying `cfg`, so a 1.3.0
  checkpoint and a 2.0.0 one are different payloads).
- **#22 is rebased** on `devel@211ae84` and pushed as `a876e57`: `LAWS.bend`
  keeps both law sets (34 laws), the README says `159/159` and **34** — devel's
  badge said 33 while its file holds 32 obligations, so 32 + our 2 = 34. He runs
  the gates and merges; that is what unblocks `Conv2d` and the UNet.
- **#15**: he closes it when #22 merges, and opens a follow-up for the general
  `k = s` round trip and the adjoint law.
- **Staged, not merged**: branch `devel-200` (pushed) ports the two files the
  sweep needed to the 2.0.0 API, importing `../bendtorch` by path. 11 call sites
  remain; the whole thing lands the day 2.0.0 is on the hub.

**Next, in order.** 1. Watch #22: on merge, re-run our gates and start the UNet
(`GroupNorm`, `FullAttention` and `Conv2d` are all in then). 2. On 2.0.0's
release, finish `devel-200` (11 call sites), switch the import to
`bendtorch@2.0.0.0`, and re-run the sweep as the acceptance. 3. The
`FullAttention`-at-the-same-shape comparison for #28. 4. bendlang #1632 items
9-10.

## 2026-11-03 (night) — the floor was 64x too small; the attention is where the ms are

**Where things stand.** Two more fixes and the first located performance finding.
`make check` reads **112/112** (99 + 13 bench programs, which the gate now
type-checks).

- **ISSUE-049: `vit_step` counted `tokens` (16, one image) where `cifar_step`
  and `flow64_step` count `rows` (1024 = 64 images x 16).** The forward's dense
  floor was 64x too small, so the page read "0.5% of its floor" for a forward at
  **32.6%**. Fixed: forward floor **20.55 ms** against 63, `loss_backward_opt`
  **43.1%**, "how much is on the table" **96.7 ms** of kernels (was 151.4).
  `bounds_for` now reads the task's sidecar (the declaration) before the
  record's copy, so a corrected formula reaches every page.
- **ISSUE-050: `make call-cost`** (`bench/call_cost/main.bend`) times the
  model's own shapes: `Mat.matmul(1024x64 . 64x64)` **0.9 ms** against its
  0.84 ms floor; the same through `Layer.fwd(LinearNBF)` **0.7 ms**; one
  `LayerNorm` **0.18**; one `Mlp` **3.2** against 3.3; one `AttBlk` **11.2**
  against 3.5 — **3.2x**. The per-call-overhead hypothesis is dead, and the
  forward's 42.5 ms of headroom is the attention block's score/softmax path
  (~7.5 ms per block, ~8,704 `exp` calls per batch). Filed as BendTorch#28 with
  the table and the repro.
- **ISSUE-045 fixed**: `bench/cifar_learning`, `bench/cifar_learning_gpu` and
  `bench/fill_cross` type-check again (a `Mat` wrapped in `Mat.from_list`, and
  `Array.to_list`'s `&1` read at `&2`), and `check_all.py` type-checks every
  `bench/**/main.bend`, so rot is a failing check now.
- **ISSUE-046 fixed**: `write_report` routes the `latest` card through
  `artifacts.save_json`; a re-run that only moved the clock leaves the file
  byte-identical (verified). The timestamped record is still a new file per run,
  by design.
- **Upstream today**: #26 (launches cannot be counted), #28 (the attention
  block), the bend-ml#23 comment with the fill numbers.

**Next, in order.** 1. `FullAttention` at the same shape as `AttBlk` — it is the
comparison #28 asks for, and it decides whether the causal per-query path is the
cost or the block's wiring. 2. The UNet task, twin and cell first (`Conv2d`
arrives with #22; `GroupNorm` and full attention are in 1.3.0.0). 3. Watch: #22,
#24, #26, #28.

## 2026-11-03 (later still) — the floors were the fixture, so the worklist changed

**Where things stand.** Burn's shape is fully in the lab (record v3, the page,
`--against`, bounds, matrix, `make validate`, CI), and reading it against the
numbers exposed two defects in the *denominators*, both fixed here. `make check`
reads **99/99**.

- **The peaks were one cold pass of 16 MB under a 1 ms clock** (medians 5.0 /
  2.0 / 3.0 ms). `bench/peaks` now repeats every phase 200 times inside one timed
  region: samples 115 / 333 / 279 ms, `checks.sample_enough`. Measured after:
  read **12.027 GB/s** (copy-charged), write **29.178**, copy **24.053**, and the
  reducing walk published separately as `reduce_gbs` **10.076**. The old numbers
  were 3.355 / 8.389 / 11.185 — a cold pass reads a third of the warm rate.
- **Nothing recorded the thread count**, so a stage could be scored against a
  ceiling measured at another. `peaks.threads`, `flops_threads` and
  `variant.threads` are recorded, `--threads` reaches the runtime
  (`+RTS --threads N -RTS`), the matrix has a `threads` column and declares
  `attention-parity` at 1 and 16, and `bounds.floor_us` refuses a peaks block
  whose machine, Bend build or thread count differs — the page prints
  `peaks unavailable: <reason>`.
- **The worklist no longer points upstream at an unmeasured number.** A leaf
  without a floor prints `unavailable` with the reason; a scored floor says what
  it counts (`matmul flops only` for the ViT's forward and backward).
- **`make plumbing` writes one record per machine** and the page prints one row
  per machine: the same step measured 131 / 173 / 134 ms on three kernels, with
  the 1.3.0.0 bump in between and no effect from it.
- **Upstream, filed today.** BendTorch#26: a step's kernel launches cannot be
  counted from a program (only under `nsys`), so `launches` and `unclaimed` are
  permanently `unavailable`; asks for one count per `Train.step`. bend-ml#23:
  the owed end-to-end comment — target noise **4 ms** against 29 + 3, the three
  input cells and `first_loss` unchanged (-7.65e-08), posted with the command.
- **The open question this leaves**: the ViT forward is still ~0.5% of a
  10 GFLOP/s ceiling that `bench/matmul` measures at 128x128, and its bound
  counts matmul flops only. Per-call cost or elementwise work is unresolved; the
  probe that isolates one matmul inside the model is the next measurement, and
  until it runs, "the kernels are slow" is not a claim we can file.

**Next, in order.** 1. The matmul-inside-the-model probe (one `LinearNBF` of
64x64 over 64 rows, timed alone) — it decides whether the 63 ms is overhead or
arithmetic, and therefore whether anything is upstream's to fix. 2. The UNet
task with its twin and cell, `Conv2d` arriving with BendTorch#22. 3. ISSUE-045
(three `bench/` programs that do not type-check; `bench/` is in no gate) and
ISSUE-046 (the runner rewrites `results/latest` with a new clock when no number
moved). 4. Watch: #22 merge, #24's shape answer, #26.

## 2026-11-03 (later) — the pin is on 1.3.0.0, and #24 never blocked it

**Where things stand.** The bump is done and it is 45 import lines: `bendtorch`
and `bendtorch-gpu` to `1.3.0.0` in 22 files, `bend-ml-tensor-array` to
`0.1.18.0` in 23. `make check` on this tree reads **99/99**.

- **The handoff's last entry predicted 31 `Train.step` call sites.** That was
  wrong: `type Train`, `Train.new` and `Train.step` are byte-identical between
  `1.1.1.0` (901/919/946) and `devel@1a37d12` (902/920/947) -- a diff of the
  block is 0 lines -- and the repository has 13 call sites. BendTorch#24 is
  real and still open, but it blocks only the signature change the issue
  proposes, not the bump.
- **Measured inert.** `test_flow64.py` passes with the same deltas to the digit;
  `flow64-first` reads -1.877e-08 / -2.404e-07 / -1.049e-06; `vit-first`, which
  runs `Train.step`, reads `loss_at_0` **-6.271e-09**, the same as on 1.1.1.0.
- **What it buys.** `FullAttention` and `GroupNorm` are in 1.3.0, so after
  BendTorch#22's `Conv2d` there is no upstream dependency left for an
  unconditional UNet. RoPE2d (#18) and adaLN (#19) are for a conditional one.
- **Upstream today.** #23 closed as agreed (v1.3.0 did it differently); #22 is
  mergeable, the three wording notes are in `88d3961` and answered, and it waits
  on nuxyel's merge; #24 got a paste-runnable repro (also green on `devel`) and
  one question -- is shape (1) still what he would take -- so the signature is
  settled before we edit call sites.
- **Found on the way.** ISSUE-045: `bench/cifar_learning`,
  `bench/cifar_learning_gpu` and `bench/fill_cross` do not type-check, and did
  not before the bump; `bench/` is in no gate. Also, `make experiment` still
  rewrites `results/latest/*.json` with a new clock when no number moved, which
  ISSUE-033 closed for the five tests but not for the runner.

**Next, in order.** 1. **The UNet** (ours, unblocked): `examples/diffusion/unet`
with its twin and cell first, the way `flow64` landed; `Conv2d` arrives with
#22, `GroupNorm` and full attention are already here. 2. Answer-driven: if
nuxyel takes shape (1) on #24, send the PR with the repro as its test, then
adopt the new signature across the 13 call sites in one change. 3. Then
`~`-parameter message on bendlang (#1632 items 9-10), and the three rotted
bench programs (ISSUE-045) either fixed or deleted.

## 2026-11-03 — the 64x64 flow has a twin, so 99/99

**Where things stand.** The handoff's first item is closed (ISSUE-043): the flow
was the one task gated by a measured expectation rather than a second
implementation, and it now has both. `make check` reads **99/99** on this tree
(the run took 47 minutes here; the machine was swapping).

- **What exists now.** `reference/probe_flow64.bend` (frozen dump in eight
  parts, the first two tokens' features, the forward at t = 0.5, one Euler step,
  the 8/16/32 endpoints), `reference/twin_flow64.py` (float32, same operation
  order, `from_flat` on the probe's dump), `reference/test_flow64.py` (five
  elementwise gates), `bench/flow64_first` and the `flow64-first` task with its
  frozen reference and matrix cell.
- **Measured.** Features **1.19e-07**, forward **2.68e-07**, one Euler step
  **3.58e-07**, the 8/16/32 endpoints **3.58e-07 / 4.77e-07 / 4.77e-07**, and
  `params` 413,440 on both sides. Gates are at 1e-5, ~20x the measurement. The
  gate costs 9 s on a built binary and 56 s with the build (cached by mtime in
  `/tmp/mlbend-flow64-frozen`). The cell is 3.8 s of run and 48.6 s of build,
  52 s total; its three mean squares agree to 1.0e-06 relative against a 1e-4
  gate. The matrix is 18 cells.
- **Two shapes the twin had to read off the code**, and the reason it matched on
  its first build: 128 rows (not 2 x 64), and per token the pixels, then
  position, then time. `feats` conses time in front of position and `lr_of`
  reverses, so both blocks come out ascending, position first. The attention is
  causal in two windows of 64 rows with `sc = 0.125` (not `1/sqrt(128)`), and
  `upd`'s `dt` is the literal 0.03125, so the 8- and 16-step runs are prefixes
  of the 32-step one. Details in `docs/NOTES.md` under 2026-11-03.
- **Left as measured, not fixed.** ISSUE-044: `tok_go` never increments `img`,
  so the `8n` branch is dead, rows 64..127 read positions 1.0..1.98, and both
  images take `level(s, 0)`. The twin mirrors it (the forward agrees), so the
  published run trained on this pairing; changing it is a behaviour change with
  a re-run, not a cleanup. The latent `1/EULER()` in `upd` is the same kind of
  thing: bit-identical at the only preset (32), so it can be cleaned up in a
  change that moves no number, and it was not touched here.

**Next, in order.** 1. The `~`-parameter message: the ten-line repro, then file
on bendlang (about an hour, no dependencies). 2. Optionally, `upd`'s `dt` from
`EULER()` — bit-identical at 32, so a no-number cleanup, worth doing the next
time the example is open. 3. Watch, not poll: bend-lab pins
`bendtorch@1.1.1.0`; the bump waits on BendTorch#24, and #22/#23 wait on
review.

## 2026-11-02 (later still) — a checkpoint writes, resumes and refuses

**Where things stand.** ISSUE-034 (B9) is closed. The identity layer and
BendTorch's `Train.save` / `Train.restore` existed but no example used them.
`examples/diffusion/flow_matching` now takes `--save PATH`, `--resume PATH`,
`--start N`, `--steps N` and `--no-sample`; without flags it prints the same
6,000-step curve as before. `reference/flow_checkpoint.py` writes and checks the
sidecar (model, config, packages, weights sha256, step), and
`reference/test_flow_checkpoint.py` gates the path in `make check`.

- **Measured.** 200 steps in one run against 100 steps + a resume at step 100:
  the segments print the single run's 200 losses **bit for bit** (the stopped
  run ends 0.9704682; the resumed run's first number is 0.9746805, the single
  run's value). The checkpoint is **207,392 bytes** = 3 × 17,282 parameters + 2,
  float32, and the sidecar's sha256 matches it.
- **The refusal is before the first step.** A moved lr prints
  `config.lr: 0.003 -> 0.004`; a width moved 128 -> 64 prints
  `model.declared`. The test asserts both and never launches the program on a
  mismatch.
- **The upstream links** are collected in `docs/agents/upstream.md`: 12 open
  (6 BendTorch, 2 bend-ml, 4 bendlang) and 6 closed.
- **The default path is unchanged.** The 6,000-step curve and the sample
  mean/std equal `results/flow-matching/run.json` exactly; only the log's F32
  `data_mean`/`data_std` differ from the record's float64 ones, which were
  never the record's source.
- **Traps paid for, for the next example.** A `String` is linear
  (`String.eq` consumes it: match `SNil`/`SCon` and rebuild); a `match` cannot
  be a second match on an earlier binder (match in parameter order; match a
  list head before its tail); a self-call's shrinking argument must be a direct
  tail; a `Nat` reused twice needs `+`; `do IO<Train<..>>` is angle-bracketed
  while the return type is `IO(Train<..>)`.

**Next, in order.** 1. `flow64`'s twin: `probe_flow.bend` + `twin_flow.py` for
one forward and one Euler step, then the Euler 8/16/32 comparison; the task is
still gated by a measured expectation. 2. The `~`-parameter message: a
ten-line repro, then file on bendlang. 3. Watch: bend-lab pins
`bendtorch@1.1.1.0`. (The "31 call sites" this line used to predict was
measured on 2026-11-03 and is wrong: the `Train` API is byte-identical between
1.1.1.0 and 1.3.0, the bump is 45 import lines, and it does not wait on #24 --
see the newest entry.)

## 2026-11-02 (later) — the gate leaves the tree clean

**Where things stand.** ISSUE-033 is fixed. `reference/artifacts.py` is the
only writer for the five tests that write `results/<task>/` (`test_cifar`,
`test_cifar_train`, `test_transformer`, `test_diffusion`,
`test_diffusion_digits`): `save_text` compares the tables and figures exactly,
and `save_json` compares the record after dropping the fields that are not
results (`created_at`, `git`, `machine`, `process`, `seconds` and
`*_seconds`, `speed_ratio`, `repeat`). A moved number writes the whole record;
a new clock alone writes nothing. Each test prints `wrote` or `unchanged`, and
on this tree all five print `unchanged`.

- **Verified.** `BEND_MEM=9G scripts/bend-guard.sh make check` → **97/97**
  (`reference/test_artifacts.py` is the 97th check), `git status --short
  results` lists no tracked file afterwards, and the five tests take 6 / 58 /
  185 / 127 / 196 s. Before the fix the same gate left five modified records
  and cost two hand `git checkout --` reverts.
- **The guard is gated.** `test_artifacts.py` copies each of the five committed
  records to a temporary directory and checks that an identical record writes
  nothing, a moved `bend.first_loss` writes, and a new `created_at` and
  `seconds` alone does not; a loss table is checked the same way. That is what
  keeps a wrong key in the volatile list from either dirtying the tree again or
  swallowing a real change.
- **Not touched.** `results/matrix/matrix.json` already keeps its `created_at`
  on a `--summary-only` re-render (the note under ISSUE-033), and the other
  `reference/*_results.py` writers run under `make`, not under the gate.
- **The ledger** is 45 entries: 21 fixed, 1 open (037 upstream), 8 upstream
  requests, 15 other.

**Next after that.** The checkpoint work above closed ISSUE-034; the order from
there is in the newest section.

## 2026-11-02 — flowers-flow is a task, and the layer now carries 64x64

**Where things stand.** `main` is green. `flowers-flow` is a task: 6,000 steps,
32 Euler, the `corpus` preset (413,440 parameters, 128 rows a step), **PASS**
with `params` **413440 exact**, `legibility_heldout` **0.490093** against the
reference's 0.489027 (0.87 × the 0.5621 ceiling), `final_loss` 0.289049. Those
are the 2026-10-31 hand numbers to three decimals, which proves the port that
made the export path portable (`MLBEND_DATA`, machine cache as fallback; the
config generator writes the bare file name) moved nothing.

- **Cost, from the record**: `build 28 s + compute 1,176 s`, a step 137.7 ms
  (input 1, forward 32, loss + backward + optimizer 104.7), the 32-Euler sample
  inside the run. So `flowers-flow` is a `NO_CELL` line (1,204 s a cell), and
  NOTES' 47 minutes for the same preset is a load difference — quote the
  record's `compute_seconds` from now on.
- **The gate is a measured expectation, not a twin**: the ceiling from the 102
  held-out images and the 87% of it the corpus reached, plus `params` exact to
  pin the preset. The flow's frozen probe (one forward, one Euler step) is still
  owed and is the next real measurement on this arm.
- **The split**: `bench/plumbing_flow64` (batch read, pair build, forward, step,
  one Euler step) with `floors.flow64_step`; `reference/stages.py` now handles a
  program that builds its input and target in one pass (no `step_tgt`) and
  honours a per-task `stages.note`.
- **The matrix refuses silence in code**: `matrix.silent()` is checked by
  `make matrix`, so a task that is neither a cell nor a `NO_CELL` line fails the
  run instead of leaving a gap a reader has to notice. ISSUE-042 records the one
  weakness this task carries: a generated export cannot be hashed by the
  runner, so the pin is enforced by the parse, after the run.

## 2026-11-01 — the fills land for the target, cifar-train is real, and the config is upstream's

**Where things stand.** `main` is green: `make validate --quick` passes (16 s,
env-check and `check_all --fast`) and **`make check` 95/95** on this tree
(`BEND_MEM=16G scripts/bend-guard.sh make check PY=reference/.venv-lucidrains/bin/python`;
it rewrote five `results/*/run.json` timestamps, reverted per ISSUE-033). Five
commits, plus a sixth for the regression cells and the matrix (below):

- `7cdd515` — the CIFAR target's noise is bend-ml#23's `fill_gauss`
  (`bend-ml-tensor-gpu@0.1.5.0`). A step **143 ms → 128** (`bench/plumbing_cifar`,
  three runs: `full_step_x3_ms` 426/430/441 → 376/382/393). The target's noise
  goes from 29 ms of `gauss_at` plus 3 ms of staging to **4 ms**. The input keeps
  `gauss_at`: the fill plus the list the row walk needs measured **70 ms**
  against **40** (ISSUE-038). The version clash does not block it — a `Mat`
  cannot cross `tensor-array` 0.1.14/0.1.18, but the `Array<F32>` that
  `fill_gauss` returns can (`bench/fill_cross`, max_diff **2.09e-7** against
  `gauss_at`, gate 1.3e-6) — so the stack bump stays its own change and was not
  needed. Comment posted on bend-ml#23 with both splits, the CUDA numbers and
  the commands.
- `7962882` — `examples/cifar`'s `do_step` takes the config, because
  `Train.step` derives the optimizer from the config it is handed and `Train`
  carries none (ISSUE-037). The numbers are bit-identical. Filed upstream as
  [nuxyel/BendTorch#24](https://github.com/nuxyel/BendTorch/issues/24) with
  `bench/train_cfg` (three Trains at lr 0.001/0.01/0.1, all stepped at 0.001,
  all read **6.5033846**) and two candidate fixes; we said we would take
  "carry the config in the `Train` and drop `step`'s argument".
- `91adea2` — `examples/cifar_train` + task `cifar-train`: the same model and
  the same 256 images, a level from {16, 32, 48} each step, lr 1e-3, 600 steps.
  `first_loss` **1.0022775**, `final_loss` **0.9960275**, `loss_drop`
  **1.0062749** against the twin's 1.0069602 (delta 6.85e-04, gated at 2e-3
  relative). `final_loss` is report-only. `cifar-10` is unchanged and its
  description now says it is a plumbing check.

**Every finding from this session is now a cell, and the matrix names every
task** (the plan's B5 and appendix D; ISSUE-041). `vit-first` is a new task over
`bench/vit_first/main.bend` — the ViT example's own `init_params` and
`do_step(0)`, **1 s** — whose reference is the twin's `loss_pieces(0)` and whose
three metrics are `loss_at_0` and `tgt_sq` at 1e-6 relative and **`out_sq`** at
1e-4, and the last is the point: the loss is dominated by the target's mean
square and moved only 1.15e-4 while `out_sq` moved 2.5e-3. With the pre-fix twin
pairing the cell FAILs with exactly those deltas (the failing record is
committed: `results/matrix/cells/vit-first-20261101T160503Z.json`) and passes
restored. `reference/test_stages.py` is the cell for ISSUE-039 in 0.3 s: the
stage environment carries `MLBEND_DATA` and a program that prints
`the dataset failed` is refused. `make matrix` now reads **17 cells run, 17
records, 0 failed**, with `diffusion-digits` (226 s) and `diffusion-vit`
(724 s) as `no_cell` lines that name their cost and their test, and four `gpu`
lanes declared absent. A failing cell now names its metric, not `run`.

**`diffusion-vit` is green again** (ISSUE-032, ISSUE-040). The task's reference
called `twin.first_loss()` — one timestep for the whole batch, the
pre-ISSUE-016 wiring — against a program that runs one level per image
(1.0069675 vs 1.0091865, 2.1e-3); it now calls `loss_at(0)`. The remaining
1.15e-4 was the twin not mirroring the example's level reversal: `ab_img` and
`te_img` reverse a list that is already ascending, so image *i* runs at image
*63 − i*'s level (a harmless permutation; the example is unchanged). Mirroring
it in `twin_vit.levels_of` gives **6.3e-09**, `TOL_FIRST` is **tightened 1e-3 →
1e-5**, the one-second probe is `bench/vit_first`, and
`make experiment TASK=diffusion-vit` PASSes (first_loss 1.00919, final 0.0711534,
legibility 0.787/0.806 as recorded). The record is committed: the file that had
been untracked since the FAIL is now `results/latest/diffusion-vit.json`.

**What the fills were worth at CIFAR scale: 15 ms of a 143 ms step.** The
generator was half the staging but the *list* is what the input pays for, so
the fill pays off where a `Mat` can take the array directly (the target) and not
where a list must be walked (the input). On the CUDA lane it is **worse**: the
first fill 33-41 ms, a warm one 17, the step **159-165 ms** against 128 on the
CPU threads, with `ldd` resolving `libnvrtc.so.13` and `nvidia-smi` showing the
pid at 104 MiB. See ISSUE-038 and `docs/NOTES.md` for the table.

**The step now has a name for every part** (`bench/plumbing_cifar`, three runs,
the record's medians): input **41 ms**, forward **37**, target **4**, loss +
backward + optimizer **45.7**, one step **127.7**, sampler 322-327. The stage reader
(`reference/stages.py`) reads these keys, so `cifar-train`'s page has a real
split and `bounds.cifar_step` scores it (114% and 203% of the flops floors).

**Upstream, untouched by us:** BendTorch#22 and #23 are both `MERGEABLE`/`CLEAN`
against `devel`. nuxyel reviewed #22 and asked for three things (close the
`k > h + 2p` edge in the signature, a stride > 1 end-to-end case, and the
PyTorch comparisons moved into `make check`); we answered on the PR with the two
signature shapes and the note that his `7x5` example has an even surplus, so
`7x6` goes in as well. **Nothing was pushed**: no new PRs, and #24 stays an
issue until he answers.

**One bug worth remembering (ISSUE-039).** The first `cifar-train` record's
stage split was measured on an empty dataset: `reference/stages.py` ran the
stage program without `MLBEND_DATA`, the probe's `Datasets.csv_of` fell back to
the hub, failed, and timed 256 rows of embedding only (input `<1 ms`, step
80.7). The stage environment now carries the runner's cache and a run whose
output says `the dataset failed` is refused instead of timed.

**A correction to the session's brief (worth knowing):** `reference/.venv` is
**not** empty — it is Python 3.14.8 with numpy 2.5.4, and every `make` target
here ran over it without an override. `reference/.venv-lucidrains/bin/python`
was used for the twin runs as asked, and `make check`/`make experiment` were run
with `PY=reference/.venv-lucidrains/bin/python`.

**Traps paid for this session.** A `Mat` and a `List` cannot share one `Array`
(`Array.to_list` consumes it), so a fill that feeds both costs two bangs; the
copy from an `Array` to a `List<&2, F32>` is ~18 ns a cell and a walk that
matches two lists a pixel costs more than the `gauss_at` it replaces; a DoF
record cannot hold a `List<F32>` (a `Type`), and a `do` block cannot hold a
`+x = v` let, so a value reused twice becomes a parameter of its own def; and
`T.Fwd<P, Y, S>` takes `Y` second, not `X`.

## 2026-10-30 — PR #17 merged; the corpus is renamed and the README is real

**Where things stand.** `main` is green, **`make check` 92/92** on the
merged tree. PR #17 landed the structure: thirteen flat examples became four
areas, every path followed, and the local folder is now
`~/github/bend/bend-lab` with `origin` at `adeerkhan/bend-lab`. The README is
rewritten around `docs/assets/bend-hero.png`, the measured three-arm table and the real
layout; `examples/README.md` is a 13-task index instead of a stale second copy.

**The one red, and its fix (ISSUE-019).** `examples/diffusion/2d` could not be
imported: Bend refuses an import segment that starts with a digit, and
`probe_ddim.bend` imports its example. The directory is now
`examples/diffusion/two_moons`; the task id stays `diffusion-2d`. The nine
generated example commands and the result links the move had broken were
regenerated, and the ViT example README now carries the committed 0.787/0.811.

**The plumbing split, and the fix (2026-10-30).** A step was **543 ms**, with
77% in input and target construction and `init_params` at 25.3 s. The builders
appended into a growing list (O(n²)) and the tokeniser scanned each digit with
`at()` 64 times. Cons + one reverse and a linear 16-pixel tokeniser bring a
step to **131 ms** (40% forward, 54% loss and backward, 5% build) and
`init_params` to **6 ms**, with the full row list and the first three losses
bit-identical. `results/PLUMBING.md` and `bench/plumbing/main.bend`.

**ISSUE-020 found and fixed (2026-10-30).** `mix_img` returned 4033 values for
64×64, so the last digit in every batch was its first pixel and 63 zeros. The
fix repeats each image's `(a, b)` pair for its 64 pixels and mixes three flat
lists in step. Retrained 3000 steps: legibility **0.7875** (was 0.7829),
ancestral **0.8110** (was 0.8070), final loss 0.0712 (was 0.0896). The gate
passes at a 1.15e-4 first-loss delta.

**Flow matching (2026-10-30).** `examples/diffusion/flow_matching` runs
rectified flow on the DDPM example's two moons, normalised to [-1, 1]:
3 -> 128 -> 128 -> 2 with SiLU, full batch, 6000 AdamW steps, Euler 32.
Last-100 loss **0.8859** (the probe measured 0.8662), sample mean/std
**+0.049 / -0.048 · 0.610 / 0.675** against the data's 0 / 0 · 0.577 / 0.659,
and **112/128** inside the box. `make flow` rebuilds it and rewrites
`results/flow-matching/`. The frozen probe and its NumPy twin are still owed.

**The measured state (unchanged, 2026-10-29).** The diffusion objective's
M1–M4 are done on measured numbers; M5 (CIFAR) and M6 (UNet) are not.

| | DDIM 16 | ancestral 64 | DDIM MSE | ancestral MSE |
|---|---|---|---|---|
| **Bend-ViT, per-image t, 3000 steps** | **0.787** | **0.811** | 0.0598 | 0.0561 |
| torch-ViT, same architecture | 0.790 | 0.779 | 0.0583 | 0.0633 |
| lucidrains UNet | 0.932 | 0.940 | 0.0202 | 0.0183 |

Cloud mean/std +0.2957 / 0.3395 against the data's 0.3076 / 0.3767. The gate is
`reference/test_diffusion_vit.py` (six checks; first-loss tolerance 1e-3 from a
measured 1.27e-4 delta). Evidence: `results/diffusion-vit/{run.json, REPORT.md,
loss.svg, grid.svg}` (run.json carries the loss `curve` and both samplers'
samples) and `results/compare/{compare.json, index.html}` — torch-vit.json also
carries its curve, so both arms can be plotted from the JSONs alone.

**The finding that closed M1: one timestep per image, not per batch** — worth
**+0.36**. Isolation table, all 3000 steps, same model: one t per batch
0.359/0.344 (deterministic), 0.392/0.441 (random), 0.409 (Weyl); one t per image
0.770/0.778. Implemented as `mix_img` (per-image a/b), `rows_img` (per-image
8-value time embedding) and `t_img`. **The first version of that table was
miswired** (the "random" arm drew t per image) — ISSUE-016 keeps the correction
above the table.

**The GPU lane: it runs; the no-speed result was plumbing-bound**
(ISSUE-018).
`bendtorch@1.1.1.0` + `bendtorch-gpu@1.1.1.0` are in `bend.toml` on branch
**`lane/gpu`**; the ViT there uses eight `LinearNBF` and one `F.Attention`. The
binary links `libnvrtc`, `nvidia-smi` shows the process with **104 MiB on the
card**, the frozen values are **bit-identical** to the CPU lane — and the wall
time was unchanged (5m44 both ways) against the old data path. After the
builder fix a step is 131 ms and 95% of it is forward and backward, so the
card is worth re-measuring.

**Papers and 64x64 (2026-10-30).** The TEM entry is closed as a negative
result: the 64x64 contrastive arm collapses to a constant energy, the score
arm needs 12.8M images rather than 32k, and a faithful Bend port needs full
attention, RoPE and adaLN. The 16x16 lite score arm trains (loss 10.76,
nearest-test 13.72) but is texture, not a 64x64 result. The next Bend-native
64x64 target is a rectified-flow ViT on flowers: 64 tokens, width 128, three
blocks, existing layers only. The device lane is verified on current main
(104 MiB on the card, CPU and GPU probe outputs bit-identical) and the data
path is proven: 512 images read in 44 ms and fill an array in 45 ms at depth
23 (ISSUE-027). The 512-image export is pinned at
`02385f3ae6c9aa9c0cbe294b45e9758e347cb04abda529352952cd8b7b7075a8`.

## What is next, in order

1. **The 64x64 flow's own gate, then its lanes** — `flowers-flow` is a task now
   (done 2026-11-02), but it is gated by a measured expectation, not a twin:
   write `probe_flow.bend` at the fixed init and `twin_flow.py` for one forward
   and one Euler step, the way the ViT's gates work. Then the Euler-step
   comparison at 8/16/32, a sample grid, and the lane question (the example
   imports `bendtorch` only, so there is no device path to link `libnvrtc`).
2. **Second migration** — package sources under `packages/`, `reference/` under
   `apparatus/`; `bend.toml`'s members, the intra-package imports,
   `check_all.py`'s package list and the Makefile. One commit, `make check`
   behind it.
3. **Flow matching's frozen gate**: a `probe_flow.bend` at the fixed init and
   a `twin_flow.py` for one forward and one Euler step, the way the ViT gates
   work. Then flowers at batch 32 / 11k steps, dataset pinned by sha256.
4. **`FullAttention`** — the PR we accepted on BendTorch#17: separate def, share
   `qrow`/`prow` with the key length, its own three laws over all t,
   `demo/ops_test.bend` + `reference/test_ops.py` against
   `F.scaled_dot_product_attention`, `Attention`'s bits untouched. Then
   `Im2col` (#15: `Im2col(n,h,w,c,k,s,p)`, gather backward, device bangs).
5. **CIFAR-10** only behind (1), a real step time and a re-measured device lane.
6. **Ledger hygiene**: 22 entries — 12 fixed, **0 open**, 1 upstream, 9 other.
   Collapse resolved-and-reported entries to one line in an archive; ISSUE-014
   and ISSUE-018 keep their corrections above the original text on purpose.

## Open threads upstream

- **bendlang/bend#1894** — closed as a known limit; PR 1921 carries the
  diagnostic (`Refs`), nothing owed.
- **PR 1820** — poll period now 64 (our `6b9c405f`); waiting on the maintainers;
  we cannot re-request review there (no permission).
- **BendTorch#15** — design agreed; `Im2col` offered, after `FullAttention`.
- **BendTorch#17** — our issue; we took the `FullAttention` PR.
- **bend-ml#26** — laws landed on `devel` (`ASet.*`); our **PR #27** (64x64 test
  half) is open; the corollary law is next.
- **BendTorch `make gpu`** — its demo step calls `reference/.venv/bin/python`,
  absent from the checkout (Error 127); a one-line fix we could PR.

## Traps paid for this session

`Nat` fail-stops past 2^48-1 (use `U32` for mixers); a `match` cannot scrutinise
a computed value, and a binder cannot be matched after a later one; `List`s are
affine unless the parameter is `+`; stdout is block-buffered (use `stdbuf -oL`
and `setsid`, as `scripts/vit.sh` does); linking `libnvrtc` is not proof of
execution — `nvidia-smi --query-compute-apps` is; a GPU run is only faster
if the work is matmul and not plumbing; an import path segment must start
with a letter, so a directory named `2d` cannot be imported (`two_moons` can);
and `List.append` into a growing list is O(n²), as is an `at()` scan inside a
loop — cons and reverse once, and walk the list linearly.
