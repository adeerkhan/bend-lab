# Changelog

What changed for the people who run the lab. Internal refactors are not listed.

## Unreleased

### The flow's settings were re-measured, and the torch comparison closes

- **No setting beats the baseline.** With the batch restored to 128 points: baseline 0.8360 final / 0.8859 last 100,
  lower lr 0.8412, 12,000 steps 0.8630, 256 wide 0.8471, 256 wide + 12,000 0.8614. The "0.4318 at 256 wide" this
  repository recorded for a day was a batch-size bug: the width edit also changed `def n()`, so half the batch's
  targets were zero. The confounded run stays in `variants/zeropad-artifact.json` and `RESULTS.md` says what it was.
- **`reference/compare_flow2d.py` now passes**: first loss 1.476769 vs 1.476769 (8.97e-08), final 0.835969 vs 0.846537,
  last 100 0.885889 vs 0.893301. Three corrections in the comparator: the data order (`moons()` alternates the moons),
  the per-step noise (`noise_of(s)`), and the init scale (`1/sqrt(cols - j)` per column). The printed loss is the
  step's own loss with its pre-step weights; the convention was never wrong (ISSUE-060).
- **`reference/flow_config.py`** is the one reader of the example's `hidden()`/`steps()`/`euler()`, used by the twin,
  the torch comparison and the checkpoint sidecar, so a width change cannot leave a gate expecting the old weights
  (ISSUE-062).

### The UNet gains a second level, gated whole-plane against PyTorch

- **`examples/diffusion/unet/main.bend`** is now a two-level U: 64x64 -> 32x32 -> 16x16 -> 32x32 -> 64x64, with four
  `Conv2d`, four `GroupNorm`, two `Up2` upsamples and two additive skips. `--dump` prints the whole planes, so a
  comparison is 2,048 cells per intermediate and 8,192 for the output instead of eight.
- **`reference/test_unet.py`** is the two-level NumPy twin: first eight cells of `y`, `z` and `out` at **5.96e-08**
  against `1e-6`, a can-fail that rejects a cell moved by 0.5, and the `Up2` adjoint. **`reference/compare_unet_torch.py`**
  rebuilds the same deterministic weights and forward in torch and checks every cell: `y`/`z` **1.19e-06**, `out`
  **7.63e-06**, all inside `2e-6 + 1e-6*|value|` (the last cell is one float32 ulp at the skip's +-82 ramp).
- **`reference/unet_results.py`** writes the record: `run.json`, the U diagram, the planes as heatmaps and the
  per-stage magnitudes under `results/diffusion-unet/`. The write-up, with the numbers and the two traps, is
  `examples/diffusion/unet/RESULTS.md`.
- **The trap worth keeping:** a conv's `Fwd` state type is the **im2col matrix** (`CST1()` here, `Mat<1024,18>`), not
  the output image (`MID()`, `Mat<1024,2>`).

### The 2D flow has an element-by-element gate

- **`reference/probe_flow2d.bend` + `reference/twin_flow2d.py` + `reference/test_flow2d.py`**: the 2D rectified flow
  was the one example measured only end to end — its README said the frozen probe and its NumPy twin were still owed.
  The probe prints the model's own `Layer.dump`, the 128 input rows, one forward step at `t = 0.5`, one Euler step at
  `t = 0` and the 8/16/32-step states; the twin rebuilds all of them from that dump. Six gates, each under `4e-7`
  against `1e-5`. `reference/check_all.py` runs it with the rest.

### A wrong dataset is refused before the run, not after

- **`reference/experiment.py`**: a task's *generated* dataset can now carry a `pin` (`roots` to search and
  `name -> sha256`), and `run_bend` checks it **before the build and the run**. `flowers-flow` is pinned this way, so a
  stale or corrupted export costs a second instead of a 19.6-minute run (ISSUE-042). `_flowers_paths()` calls the same
  check, so the pre-run pin and the parse cannot disagree, and `--check-dataset TASK` runs the check and exits — which
  is how it is tested without a run.

### The UNet's forward, composed and confronted with a twin

### The conv round trip at any stride and padding

- **`conv_roundtrip_window`** in BendTorch (`a2732c7`): for every kernel, stride
  and padding, slot `(ch, ky, kx)` of output position `(oy, ox)` is image cell
  `(oy·s + ky − p, ox·s + kx − p)`, channel `ch` — the gather and the scatter name
  the same cell, `p_src` both times. That is the geometry the UNet runs
  (`k = 3, s = 2, p = 1`), and the numbers agree with PyTorch exactly on three
  shapes.
- It sits on the mixed-radix decode and on `divmod_go_spec`, Bend's division
  meeting its spec; the narrower `conv_roundtrip_ks` (`s = k`, `p = 0`) is now a
  kernel-checked instance of it. `make laws` and `--verdict` are green, 37 laws.
- **Still owed:** Conv2d's adjoint (#15), which needs a spec model with indexed
  accumulation; `docs/limitations.md` says exactly what is missing.

- **`examples/diffusion/unet/main.bend`** is the smallest honest U: `Conv2d`
  stride 2 down, `GroupNorm`, `SiLU`, the `Up2` block, `Conv2d` stride 1,
  `GroupNorm`, `SiLU`, and an additive skip from the input — `Mat<4096,2>` in,
  `Mat<4096,2>` out, 80 parameters, all deterministic.
- **`reference/unet_gate.py`** rebuilds that forward in NumPy: the strided
  window, the `(c, kh, kw)` patch order, the group over both channels, `SiLU`,
  the 2x2 repeat, the skip. **Channel 0 of every cell matches exactly; channel 1
  is 1.86e-4 apart**, so it is a probe, not a gate, and the file is deliberately
  not named `test_*` (ISSUE-051).
- **What that already proves**: the flat row order, the channels-last patch, the
  stride-2 surplus, `Up2`'s vertical-then-horizontal order and the additive skip
  are all right — an error in any of them would move both channels.
- **Why channel 1 is the hard one**: its `GroupNorm` scale is `-0.01`, so the
  normalised value there is ~19.9 and any residual is amplified 100x; the
  candidates are the group's variance order, the conv's 18-term accumulation, and
  where the bias enters.

### The UNet's decoder block, invented and gated

BendTorch 2.0.0 has `Conv2d`, `GroupNorm`, `SiLU` and `FullAttention`, but no
transposed convolution and no channel concatenation, so a UNet's decoder needs an
upsampler that does not exist in the package.

- **`examples/diffusion/unet/main.bend`** holds `Up2`: a parameterless layer
  whose forward repeats every pixel 2x2 (vertical pass, then horizontal) and
  whose backward is its exact adjoint, the sum of the four cells that came from
  one input cell. Both work on the flat `Mat` layout Conv2d uses (rows are
  pixels, channels last).
- **`reference/test_up2.py`** gates it without a tolerance: the forward must
  equal NumPy's repeat on a 2x2 image of 2 channels, the adjoint identity
  `<Up2(x), y> == <x, down(y)>` must hold exactly (816 = 816), and one cell moved
  in the expectation must be rejected, so the gate can fail.
- **Why it matters**: this is the only piece of a UNet that BendTorch cannot
  express today; with it, `Conv2d` (stride 2 down, stride 1 up), `GroupNorm` and
  additive skips compose the decoder, and the model is the next change.

### The pin is `bendtorch@2.0.0.0`

`bendtorch` 2.0.0 (2026-11-03) moves the optimizer configuration into the
`Train` and adds `Conv2d`. Both breaking changes are mechanical here.

- **21 imports, 13 `Train.step` call sites, 59 `T.Train{…}` patterns** across 22
  files, in one change: the pin moves, the step loses its configuration
  argument, and the record gains `c` after `st`.
- **No number moved.** `make check` 113/113, and the learning-rate sweep on the
  released package reads the same three rows it read on `devel`:
  **0.5790425 / 0.87949044 / 0.99899477** where 1.3.0.0 read `0.84698695` three
  times. Posted on BendTorch#24, which is now closed.
- **The trap is not `Train.step`.** A wrapper that passes the configuration one
  call up (`M.do_step(0n, cfg(), tr, x0)`) hides it from `grep Train.step`;
  `--check-only` names each one with `expected T.Train<…>` / `observed T.OptCfg`.
- **`Conv2d` arrives with it**, which with `GroupNorm` and `FullAttention`
  leaves an unconditional UNet with no upstream dependency (BendTorch#15 is
  closed for that reason).

### `Attention` against `FullAttention` at the block's shape

BendTorch#28 asked whether the causal per-query softmax is what costs. It is
not: at 1024 rows of width 64, 64 images of 16 tokens, masked full attention is
**9.55 ms** a pass against causal's **10.65 ms** — 10% *cheaper*. The cost is the
core at this size, 64 tiny 16x16 products rather than one big one, so the lever
is fewer, larger calls rather than the mask. `make call-cost` measures it.

### An outside corpus for the compiler, with a gate that can fail

- **`reference/corpus/qwenbend/`**: 200 small Bend programs written by
  Qwen3.5-0.8B, each one independently checked against real Bend2 by the author
  of [qwenbend](https://github.com/voidash/qwenbend), with its verdict, its value
  and its first diagnostic line. Apache-2.0, pinned by sha256, and we wrote none
  of them.
- **`reference/test_corpus.py`** asks whether the Bend we pin behaves like the
  language everyone else checked against: it accepts exactly the 116 programs
  Bend2 accepted, rejects the 84 it rejected, runs the accepted ones to the same
  number, and matches the first rule line of every rejection. 36 s, in
  `make check`.
- **A gate that cannot fail is not a gate**: the last three checks corrupt one
  expectation each (a verdict, an output, a diagnostic) and require the
  comparator to report it.

### The ViT's floor was 64x too small, and the attention is where the time is

- **`vit_step` counted `tokens` where every other bound counts `rows`.** A
  forward multiplies every weight against all 1024 rows of the batch, so its
  dense floor is **206.3 MFLOP (20.55 ms)**, not 3.2 MFLOP (0.32 ms). The page
  now reads the forward at **32.6%** of its floor and loss+backward at
  **43.1%**, and "how much is on the table" is 96.7 ms of kernels, not 151.4.
- **`bounds_for` reads the sidecar first.** A bound is a declaration over the
  task's shapes, not a measurement, so a corrected formula reaches every page;
  the record keeps its copy as evidence.
- **`make call-cost`** (`bench/call_cost/main.bend`) times the model's own
  shapes: one `Mat.matmul(1024x64 · 64x64)` at **0.9 ms** against a 0.84 ms
  floor, the same through `Layer.fwd(LinearNBF)` at **0.7 ms**, one `AttBlk` at
  **11.2 ms** against 3.5, and one `Mlp` at **3.2 ms** against 3.3. The per-call
  cost is nothing; the attention block is 3.2x its arithmetic, which is where
  the forward's headroom lives.

### A floor is measured on its own machine, at its own thread count

The page's percentages were computed from a peaks file whose samples were 2-5 ms
long under a 1 ms clock, and from a flops rate whose thread count nobody
recorded. Two of the report's totals were therefore artefacts of the fixture,
not of the model.

- **`bench/peaks` repeats every phase 200 times inside one timed region**, so a
  sample is 115-333 ms instead of 2-5 ms, and the file records both the total it
  measured and the per-pass figure a stage is charged. A sample under 100 ms is
  flagged in `checks.sample_enough`.
- **The numbers moved by 3x, and the old ones were cold passes.** Measured on
  this machine: read **3.355 -> 12.027 GB/s**, write **8.389 -> 29.178**,
  copy **11.185 -> 24.053**; the reducing walk is published separately as
  `reduce_gbs` (**10.076**), because it pays an F32 add chain and is not a
  bandwidth. The first pass of a freshly allocated array carried the page faults,
  which is why one cold sample read as a third of the warm rate.
- **A peak scores only its own machine, Bend build and thread count.**
  `bounds.floor_us` refuses a mismatch and the page prints
  `peaks unavailable: peaks measured with bend X, this record with Y` rather
  than a floor.
- **The worklist no longer points upstream at an unmeasured number.** A leaf
  without a floor prints `unavailable` with the reason; a scored floor says what
  it counts (`matmul flops only`, for the ViT's forward and backward).
- **`--threads` is real.** The experiment runner passes it to the runtime
  (`+RTS --threads N -RTS`) and the record carries it; `make matrix` has a
  threads column and declares `attention-parity` at 1 and 16 threads.
- **`make plumbing` writes one record per machine** (`results/plumbing-<machine>.json`)
  and the page prints one row per machine, because the same program measured
  131, 173 and 134 ms across three kernels and two loads. A row is the context;
  two rows are not a trend.

### The pin moves to BendTorch 1.3.0.0, with no number moving

The repository imported `bendtorch@1.1.1.0` and `bend-ml-tensor-array@0.1.14.0`,
which predates full attention and `GroupNorm`.

- **24 files, 45 import lines**: `bendtorch` and `bendtorch-gpu` to `1.3.0.0`,
  `tensor-array` to `0.1.18.0`. `tensor-gpu` and `tensor-fork` were already at
  `0.1.5.0`, the pair 1.3.0 uses.
- **No call site changes.** `Train`, `Train.new` and `Train.step` are
  byte-identical between the two versions, so the "31 call sites" the handoff
  expected were not there; the repository has 13, and none moves.
- **Measured inert.** `reference/test_flow64.py` passes with the same deltas to
  the digit, the `flow64-first` cell reads the same three mean squares, and
  `vit-first`'s trained window reads `loss_at_0` 1.00919 to **-6.271e-09** — the
  same delta as on 1.1.1.0.
- **What it buys**: full attention and `GroupNorm`, which with the `Conv2d` in
  BendTorch#22 leaves an unconditional UNet with no upstream dependency.
- **Found on the way** (ISSUE-045): three `bench/` programs do not type-check and
  did not before the bump; `bench/` is in no gate.

### The 64x64 flow has a twin, and its forward and sampler are gated

`flowers-flow` was the only task gated by a measured expectation rather than a
second implementation: its `params` and its held-out legibility say a run
reproduced a number, not that the forward and the Euler sampler are the same
computation (ISSUE-043).

- **`reference/probe_flow64.bend`** prints the frozen model's flat
  `Layer.dump` (413,440 weights in eight parts), the first two tokens' features,
  one forward at t = 0.5 on the fixed noise, one Euler step at t = 0, and the
  8-, 16- and 32-step endpoints. No dataset, so it is cheap: the build is the
  preset's, `reference/test_flow64.py` caches the binary by mtime and the whole
  gate is 9 s on a built binary (56 s with the build).
- **`reference/twin_flow64.py`** mirrors the model and the sampler in float32 at
  the same operation order: 128 rows of 64 pixels with 8 position and 8 time
  features, embed 80 -> 128, three blocks (LayerNorm, causal attention over two
  windows of 64 rows, scale 0.125, then 128 -> 256 -> 128 GELU), head LayerNorm
  and 128 -> 64, then `x <- x + v/32` at `t = i/32`. `from_flat` consumes the
  probe's dump, so neither side's initialisation is trusted.
- **The gates are elementwise, at 1e-5**, ~20x above the measurement: the first
  two tokens' features **1.19e-07**, the forward **2.68e-07**, one Euler step
  **3.58e-07**, and the 8/16/32 endpoints **3.58e-07 / 4.77e-07 / 4.77e-07**.
  The 8- and 16-step states are prefixes of the 32-step one (their distances to
  it are 0.477 and 0.324), because `step_vs` divides the level by the preset's
  `EULER`, not by the number of steps it is handed.
- **The task has a cheap cell.** `bench/flow64_first` prints the count and three
  mean squares (0.0288806, 1.0198, 1.0434) in 3.8 s, 52 s with the build, and
  `make matrix` records it as `flow64-first/cpu/0` against a frozen reference in
  `results/reference/flow64-first.json`. The matrix is 18 cells.
- **`flowers-flow` stays out of the matrix at 1,204 s** and its `NO_CELL` line
  now says so with the twin beside it; the `gpu` line for it was corrected to
  what was measured (the example's build links no `libnvrtc`, so there is no
  device lane to record), rather than the stale "imports bendtorch only".
- **One observation, mirrored rather than fixed** (ISSUE-044): the token builder
  never increments `img`, so its `8n` branch is unreachable, rows 64..127 of the
  second image read positions 1.0..1.98 rather than 0..0.98, and both images
  take `level(s, 0)`. The twin reproduces that to 1.19e-07 on the input; fixing
  it would move every published number for a model that trained on this
  pairing.

### The flow example writes and resumes a checkpoint

`examples/diffusion/flow_matching` trained from a fixed init and kept no
weights, so a checkpoint had an identity layer (`reference/checkpoints.py`) but
no example that wrote one (ISSUE-034, B9).

- **`--save` and `--resume` on the example.** `--save PATH` writes the training
  state through `Train.save` (parameters, Adam's moments, `1 - b1^t`,
  `1 - b2^t`); `--resume PATH` reads it back through `Train.restore`, and
  `--start N` sets the global index of the first step. `--steps N` and
  `--no-sample` make a short probe possible. Without flags the program prints
  the same 6,000-step curve as before.
- **The sidecar is written by `reference/flow_checkpoint.py`**, which names the
  model, the config, the packages, the weights' sha256 and the step. Its
  `--check` refuses a resume whose lr, model or packages moved, naming the
  field, before the program is launched.
- **The contract is the gate.** `reference/test_flow_checkpoint.py` runs 200
  steps, then 100 + a resume for 100: the two segments print exactly the single
  run's 200 losses, bit for bit. The checkpoint is 207,392 bytes (3 × 17,282
  parameters + 2, float32), and a moved lr is refused at `config.lr`.

### A check that leaves the tree clean

`make check` rewrote five committed records on every run. The numbers were
identical and the `created_at` and wall `seconds` were not, so the gate dirtied
the tree it had just verified, and an agent that committed what it found could
commit a re-timed artifact as a result (ISSUE-033).

- **`reference/artifacts.py` is the writer.** `save_text` compares the loss
  tables and the SVG figures byte for byte. `save_json` compares the record
  after dropping the fields that are not results (`created_at`, `git`,
  `machine`, `process`, `seconds` and `*_seconds`, `speed_ratio`, `repeat`);
  when a number moved it writes the whole record, clock included.
- **The five tests say what happened**: `wrote` when a number moved,
  `unchanged` when the committed file already holds it. On this tree all five
  print `unchanged`.
- **`reference/test_artifacts.py` gates the guard**: an identical record writes
  nothing, a moved number writes, a clock moved alone does not, on each of the
  five records and on a loss table.

### `flowers-flow`: the 64x64 flow is a task

`examples/diffusion/flowers_flow` was the lab's newest generative result and the
experiment layer could not see it: measured by hand with `make flow64`, no pin,
no page, no gate. It is now the task `flowers-flow`.

- **The data is pinned and portable.** The 3,874-image training export and the
  102-image holdout are named by sha256 (`1ff86747…`, `f08730fe…`), checked
  before the numbers are read, and the program takes its directory from
  `MLBEND_DATA` when the runner sets it with `~/.cache/bend-lab/flowers` as the
  fallback, so the absolute path one machine had is gone.
  `scripts/flow-config.py` writes the bare file name, so re-picking a preset
  cannot re-pin a path.
- **One implementation of the statistics.** `reference/flowers_report.py`'s
  `answer()` is what the task parses and what the human report prints:
  `legibility` against the training set, `legibility_heldout` against the 102
  images the model never saw, and `ceiling`, what a real held-out flower scores
  against the same reference set.
- **The gate is the measured expectation.** `params` is **exact** — the model's
  shape as one number, so a preset change fails loudly instead of measuring a
  different model — and `legibility_heldout` is within 0.05 of **87% of the
  ceiling**, the ratio the corpus run measured. `ceiling` and `final_loss` are
  report mode. There is now a NumPy twin of the 3-block flow's forward and
  sampler (`reference/twin_flow64.py`, gated by `reference/test_flow64.py`); the
  section above has the measurements.
- **The step has a split and a floor.** `bench/plumbing_flow64` times the batch
  read, the pair build, the forward, the whole step and one Euler step, and
  `reference/bounds.py`'s `flow64_step` gives the page its floors for the 64x64
  shapes (413,440 parameters against 128 rows a step).

### A cell per fixed finding, and a matrix that names every task

The plan's rule is that a fixed issue becomes a cell that failed before and
passes after. This session's three findings have one each.

- **`vit-first`** is a new task: `bench/vit_first/main.bend` runs the ViT
  example's own `init_params` and `do_step(0)` in **one second** and prints the
  loss, the target's mean square and the model's; the reference is the twin's
  `loss_pieces(0)`; `loss_at_0` and `tgt_sq` are gated at 1e-6 relative and
  `out_sq` at 1e-4, the tolerance the plan set for it. It is the cell for
  ISSUE-032 and ISSUE-040: with the pre-fix twin pairing it reads
  `loss_at_0` **-1.149e-04** and `out_sq` **-2.5e-3 relative**, and with the fix
  it reads 6.3e-09.
- **`reference/test_stages.py`** is the cell for ISSUE-039, in 0.3 s inside
  `make check`: the stage environment carries the runner's `MLBEND_DATA`, and a
  program whose output says `the dataset failed` is refused rather than timed.
- **The matrix names every task.** Sixteen tasks are now 15 cells and two
  `no_cell` lines with their measured cost and the test that covers them
  (`diffusion-digits` 226 s, `diffusion-vit` 724 s), plus the four `gpu` lanes
  that have no measured tolerance yet. `SUMMARY.md` prints both lists, so no
  task is silent. A cell that fails now names the metric, not `run`.

### `diffusion-vit` passes again: its gate compared the wrong twin window

The task's reference called `twin_vit.first_loss()`, which uses **one timestep
for the whole batch** — the wiring from before ISSUE-016 — while the program
runs **one level per image**. That is 1.0069675 against 1.0091865, 2.1e-3, and
the task FAILed. The reference now calls `twin_vit.loss_at(0)`, and the twin
mirrors the example's level pairing as well: `ab_img` and `te_img` reverse a
list that is already ascending, so image *i* is trained at image *63 − i*'s
level (a harmless permutation of which image draws which level). With both
fixed the twin reads **1.0091865062713623** against the run's 1.0091865, a
delta of **6.3e-09**, and `reference/test_diffusion_vit.py`'s first-loss
tolerance is **tightened from 1e-3 to 1e-5** from that measurement. The
task's own 1e-4 is unchanged, the example is unchanged, and its record is
committed again.

### `cifar-train`: a CIFAR run whose loss can fall

`examples/cifar_train` is `examples/cifar`'s denoiser over the same 256 images
with the two things ISSUE-036 says the plumbing run lacks: a level drawn from
{16, 32, 48} each step (200 visits a level in 600 steps, against one in forty)
and lr 1e-3. The twin is `reference/test_cifar.py`'s loop with
`train(levels=..., lr=...)`, and the task is `cifar-train` in
`reference/experiment.py`: the same dataset pin, `bench/plumbing_cifar` as its
stage split, `reference/bounds.py`'s `cifar_step` as its floors, `loss_drop`
(first loss over last) gated at 2e-3 relative, `final_loss` report-only.

```
cifar-train   loss_drop   1.00627  vs 1.00696   delta -6.85e-04   ok
              final_loss  0.996027 vs 0.99535   (report)
```

The loss falls by 0.6% in 600 steps: enough to be a falling curve and to agree
with the twin, not a trained model. `cifar-10` stays as it was and now says in
its description that it is a plumbing check. The step's split at these sizes
(`bench/plumbing_cifar`, three runs, the stage capture's medians): input
41 ms, forward 37, target 4, loss and backward 45.7, one step 127.7. The gate is the measured delta, not a widened
tolerance: the two implementations' arithmetic differs by ulps and 600 AdamW
steps carry that to 6.85e-4 on the ratio.

### The CIFAR step takes its config, so a task can name one

`examples/cifar/main.bend`'s `do_step` called `Train.step` with `cfg()` in its
own body, so a learning rate given to `Train.new` never reached the step:
BendTorch's `Train` carries no config and `Train.step` builds the optimizer
from the config it is handed (ISSUE-037, filed upstream as
[nuxyel/BendTorch#24](https://github.com/nuxyel/BendTorch/issues/24)). The
config is now a parameter, and the loop passes `cfg()`, so the numbers are
unchanged bit for bit: `first_loss` 1.0014538, `final_loss` 1.0163177, and a
step is still 126-129 ms in `bench/plumbing_cifar`. `bench/train_cfg` is the
20-line repro: three `Train`s built at lr 0.001, 0.01 and 0.1 and stepped at
0.001 all read 6.5033846.

### The CIFAR target's noise is one counter fill

`examples/cifar/main.bend` asked for 786,432 Gaussians one at a time
(`G.noise_of`) and handed the list to `Mat.from_list` on every step. The target
is now bend-ml#23's `fill_gauss` (`bend-ml-tensor-gpu@0.1.5.0`), one bang over
the same counter, wrapped as this project's `Mat`. The values are the same to a
few ulp (`bench/fill_cross`: 2.09e-7 on the first eight, the gate 1.3e-6), and
the gated numbers do not move:

| metric | before | after | twin |
|---|---:|---:|---:|
| `first_loss` | 1.0014539 | 1.0014538 | 1.0014538764 |
| `final_loss` | 1.0163207 | 1.0163177 | 1.0167508 |
| `sample_std` | 0.42896575 | 0.42897448 | 0.428963 |

The step (`bench/plumbing_cifar`, three runs; `full_step_x3_ms` 426/430/441
before against 382/385/393 after, i.e. **143 ms a step against 128**) drops by
the target's noise: 29 ms plus 3 ms of staging becomes 4 ms. The input keeps
`gauss_at`: noising a pixel reads the noise by index, so that path needs a list,
and turning the fill's 1,048,576-cell array into one costs 17-28 ms against the
40 ms the whole input already costs with `gauss_at` inline (measured both ways:
70 ms with the fill, 40 without). The version clash is real but does not block
the fill: `bend-ml-tensor-array` 0.1.14.0 and 0.1.18.0 both resolve, and the
`Array<F32>` that `fill_gauss` returns belongs to neither.

### The CIFAR example trains on the whole image

`examples/cifar/main.bend` built its batch with the digits example's 64-pixel
rows: 73,728 numbers went into a 256 x 3080 matrix and `Mat.from_list` padded
the remaining 90.7% of every row with zeros. The task passed anyway, because
its gated first loss is mostly the target's mean square. It now builds one
image a row, so the numbers move onto the twin's (`final_loss` 1.01675 against
1.01675, was 1.002612 against 1.01675):

| metric | before | after | reference |
|---|---:|---:|---:|
| `first_loss` | 1.0014287 | 1.0014539 | 1.0014539 |
| `final_loss` | 1.002612 | 1.0163207 | 1.0167508 |
| `sample_mean` | 0.496275 | 0.493898 | 0.493899 |
| `sample_std` | 0.403198 | 0.428966 | 0.428963 |

Records from before this change are not comparable on the report-only metrics.
The row builder and the weight builder also cons instead of appending, which
took `params()` from 6.4 s to 3.4 s of the run, and the step bench
(`bench/plumbing_cifar`) reads 149 ms a step, 74 of them the host staging.

### The run record is schema 3: stages, peaks, bounds, variant, notes

`reference/experiment.py` writes one JSON that now carries the stage split of
the run, the machine's measured peaks, the task's declared bounds, the variant
(trainer, lane, threads, Bend) and one line per caveat. A task declares its
split with `"stages": {"program": ...}`; a task without one writes `[]`.

Schemas 1 and 2 still render everywhere: `reference/report_run.py` and the
report card key on the fields, not on the string. Nothing needs migrating; a
record simply renders with fewer sections.

### One page per run, and `--against`

`make report TASK=<name>` prints `task`/`record`/`run`, the stage table with
shares and floors, the totals, the ladder, the worklist, the metric table and
PASS/FAIL. `make report TASK=<name> AGAINST=<record>` prints the comparison
keyed on the stage path; a trainer, lane, threads, Bend, machine or dataset
change is a banner and the rows say `lane changed` instead of a delta. A path
only one side has prints `added` or `gone`, never a 0.

### Measured peaks, and a floor per task stage

`make peaks` measures read, write and copy bandwidth and takes the compute rate
from `bench/matmul`, writing `results/peaks-<machine>.json` (launch latency
stays `unprobed` until a bang is available). `make bounds TASK=<name>` writes
`results/bounds-<name>.json` from the task's declared shapes. The page prints a
floor, the achieved percentage and the gain per stage; a stage with no bound is
`unavailable` and says why, never 0.

### The worklist

The page ends its timing section with the five largest recoverable
milliseconds, each millisecond once, and names the lever: `less fetch`,
`less preprocess`, `less conversion`, `less host work` are ours, `a faster
kernel` says bend-ml/BendTorch. `fewer launches` prints `unavailable` unless a
record counts launches and the wait between them.

### Frozen references, and a task gate without NumPy

`make references` writes `results/reference/<name>.json` — the reference column
of the task's latest run, per seed, with its source record and the time it was
frozen. `make experiment TASK=<name> REFERENCES=frozen` compares against that
file and never calls a reference function, so the gate runs on a machine
without NumPy; `make references-refresh` recomputes the files with the live
functions. `make check` now fails when a task has no frozen reference, because a
skipped check reads like a passing one.

### The matrix

`make matrix` runs the cells declared in `reference/matrix.py` — `(task, lane,
seed)` — writes one record per cell under `results/matrix/raw/` and a summary in
`results/matrix/SUMMARY.md` whose failures name `task/lane/seed/metric`. A lane
is a build fact: `gpu` builds with `CUDA_HOME` set and the binary must link
`libnvrtc`; `cpu` must not, and a cell whose label and binary disagree fails.

### One gate: `make validate`

`make validate` runs the environment check, the cheap gate (`--fast`: the proofs
and the frozen references, about 20 seconds), the full `make check`, and one
staged task end to end; `QUICK=1` stops after the cheap gate and `LONG=1` adds
the corpus, the benchmarks, the plumbing split and flow64. CI runs
`make validate` on pull requests and `make validate LONG=1` weekly, uploading
`results/`.

### The plain report card is now the page

`make experiment` prints the same text `make report` prints. Scripts that
scraped the old card's layout should read the record instead: the JSON is the
stable interface, the page is for people.
