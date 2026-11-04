# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-11-04 (later) — `RoPE2d` is in; its law is blocked, and why

**Where things stand.** Item 2 of the queue (#18) has its layer, exact against a hand-built 2D rotation
(`reference/rope2d_check.py`: `1.192e-07` forward, `1.192e-07` backward, tolerance `1e-6`). `make laws` (37 laws) and
`make build` are green on the BendTorch branch `test/im2col-ks`.

- **Not done, and recorded as ISSUE-055:** the four laws #18 asks for. Three are `F32` statements, and laws here are
  `Nat`/`Bool` only. The one address law (agreement with `RoPE` at `tw = 1`, `half_x = 0`) needs
  `U32.mod(x, k) < k` — absent. Two honest routes are in the ledger.
- **Also not done:** the device twin (`gpu.bend`) — no CUDA toolkit on this machine, so it was deferred rather than
  committed untested; and `make check` (it trains; the layer is covered by the new reference check and the build).
- **Next item in the queue:** `Tile` (#19) — promised after `RoPE2d`, `Mat<b,c>` to `Mat<b*t,c>`, backward sums the
  `t` rows, two laws, PyTorch `repeat_interleave` + autograd, then the `Tile; Mul2` vs fused `Modulate` timings to
  report on the issue. #19's laws are `Nat`/`Bool` (row counts and which rows are summed), so they do not hit
  ISSUE-055's wall.

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
