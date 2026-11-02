# Changelog

What changed for the people who run the lab. Internal refactors are not listed.

## Unreleased

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
  report mode. There is no NumPy twin of the 3-block flow yet; the flow's frozen
  probe is its own change, and the description says so.
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
