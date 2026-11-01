# Changelog

What changed for the people who run the lab. Internal refactors are not listed.

## Unreleased

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
