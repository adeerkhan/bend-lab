# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

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
`bendtorch@1.1.1.0`; a bump to 1.3.x inherits `Train.step`'s change and needs
31 call sites edited, in one change and after #24 is answered.

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
