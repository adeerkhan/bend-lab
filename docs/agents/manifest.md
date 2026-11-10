# The run record

Every experiment writes one JSON: `results/raw/<task>-<stamp>.json` (every run,
never overwritten) and `results/latest/<task>.json` (the last one; the NumPy
trainer writes `results/latest-numpy/`). The report card is a reading of this
file, and so is `reference/report_run.py`.

The schema is `mlbend.experiment/3`. A reader keys on the fields, not on the
string: a `1` record has no `stages`, `variant`, `peaks`, `bounds`, `counts` or
`notes`, and still renders.

## Fields

| field | what it is |
|---|---|
| `schema` | `mlbend.experiment/3` |
| `created_at` | UTC time |
| `task` | `name`, `description`, `program`, `dataset` (files, bytes, `sha256`, `expected_sha256`, `fetch_seconds`, cache), `seed` |
| `seeds` | every seed the run covered |
| `trainer` | `bend` or `numpy` |
| `variant` | `trainer`, `lane` (`cpu`, `gpu`), `threads` (`--threads` passed to the runtime; `null` when the program ran at the runtime's default), `bend`, and any extra the task declares (e.g. `batch`, `positions`) |
| `git` | `commit`, `branch`, `dirty` |
| `bend` | `version` |
| `packages` | every `mlbend-*`, `bend-ml-*` and `bendtorch-*` package at its hub hash |
| `machine` | platform, machine, processor, CPU count, Python and NumPy versions |
| `run` | `build_seconds`, `seconds_min`, `seconds_all`, `repeat`, `fetch_seconds`, `compute_seconds`, `reference_seconds`, `speed_ratio` |
| `stages` | the pre-order stage list (below); `[]` when the task declares no split |
| `stages_source` | `kind` (`plumbing`), `program`, `repeats`, `note`; `null` without stages |
| `counts` | per stage path, the named numbers that are not milliseconds (`rows_ms`, `from_list_ms`, and later `launches`) |
| `peaks` | the machine's measured rates, `{}` until `make peaks` writes them; a block is only used when its machine, Bend build and `threads` match the record's |
| `bounds` | per stage path, `{read, write, flops}`, `{}` until the task declares them |
| `notes` | one line per caveat a reader must see |
| `seed_runs` | per seed: the metric table and the run/reference times |
| `metrics` | the aggregate metric table (mean ± sd over the seeds) |
| `passed` | every metric passed |

## Stages

`stages` is pre-order: a parent comes before its children. `ms` is **inclusive**
(a parent's milliseconds contain its children's), so the reader computes each
share from the root and never sums children to get a parent. `depth` is the
number of `/` in `path`. `kind` is one of:

- `wall` — a stage the host waited on, device work included;
- `host` — host-only work (a row build, a target build, a fetch);
- `once` — process-level work, outside the repeated step (parameter init);
- `device` — a stage timed on the device (none yet; the lanes report wall).

A stage below one millisecond keeps `ms: 0` and the report prints `<1 ms`. No
stage is rounded up to look measurable.

### Where the split comes from

A task declares it in `TASKS`:

```python
"stages": {"program": "bench/plumbing/main.bend"},
```

The runner builds that program once, runs it three times, keeps each key's
median, and turns the `<name>_ms` lines into the stages of appendix A's shape.
The program is a benchmark, not the task's run; `stages_source.note` says so,
and so does a note in the record. A task without `stages` writes `[]`, and every
existing task keeps working.

### The trace and the dashboard

A task whose program prints one JSON object per step gets a live trace:

```console
$ make trace TASK=glm53-hybrid128 STEPS=20
{"step": 1, "tokens": 128, "loss": 5.5634737, "ms": 1208}
```

The four keys are `step` (1-based), `tokens`, `loss` and `ms`; `reference/trace.py`
refuses a line that is not JSON or misses a key. `ms` is one loop iteration: for
the GLM task it is the batch read and `T.Train.step` under one timer, so the
stage split's `load + step` is the same work. The trace is written and flushed
line by line, which is what `make dash TASK=… WATCH=1` follows; the trace is an
observation, not the record, and no gate reads it.

The split a trace is held against declares `"tree": "glm53"`:

```python
"stages": {"program": "bench/plumbing_glm53/main.bend", "tree": "glm53"}
```

`run` is `init + train_new + load + step`, the parts of `run/step` are
`forward`, `loss`, `backward` and `update`, and the note prints the parts' sum
against the step. The parts are timed on a fresh `Train` state from the same
weights, so the sum is close to, not equal to, the step; a difference is
measurement, and the note says which way it went.

### What is not measurable from a Bend program

`IO.now` counts whole milliseconds: a stage below it prints `<1 ms`, and a
per-op span at these shapes would be all zeros. BendTorch exposes no device
timestamps to a Bend program, so a device timeline and a launch count are
`unavailable`, never zero. Per-op attribution would need call-site wrappers or
an upstream observer; the dashboard reports the stages a task declares, not
every op the library ran.

## Example

One real record (`diffusion-vit`, 2026-10-31), trimmed to the new fields:

```json
{
  "schema": "mlbend.experiment/3",
  "variant": {"trainer": "bend", "lane": "cpu", "threads": null, "bend": "2.0.38"},
  "stages": [
    {"path": "run", "depth": 0, "ms": 486.333, "kind": "wall"},
    {"path": "run/init", "depth": 1, "ms": 14.0, "kind": "once"},
    {"path": "run/step", "depth": 1, "ms": 472.333, "kind": "wall"},
    {"path": "run/step/input", "depth": 2, "ms": 17.0, "kind": "host"},
    {"path": "run/step/forward", "depth": 2, "ms": 166.0, "kind": "wall"},
    {"path": "run/step/target", "depth": 2, "ms": 2.0, "kind": "host"},
    {"path": "run/step/loss_backward_opt", "depth": 2, "ms": 287.333, "kind": "wall"}
  ],
  "stages_source": {"kind": "plumbing", "program": "bench/plumbing/main.bend",
                    "repeats": 3, "note": "one step from this program, not the task's whole run"},
  "counts": {"run/step/input": {"rows_ms": 17.0, "from_list_ms": 0.0}},
  "peaks": {},
  "bounds": {},
  "notes": [
    "clock: IO.now is milliseconds; a stage below 1 ms prints <1 ms",
    "init_params 14 ms once per process (in run/init)",
    "rows 17 ms and the matrix conversion 0 ms sit inside step_in",
    "parameters 100736",
    "the stage split is one step, not the whole run"
  ]
}
```

`run` is `init + step`; the step's children (input, forward, target,
loss/backward/optimizer) sum to the step. The shares are the reader's: 166 /
486.333 = 34.1 % for the forward.

## Peaks: what makes a ceiling usable

`make peaks` writes `results/peaks-<machine>.json` from `bench/peaks`, and the
record carries it under `peaks`. Three rules decide whether a floor may be
printed from it (`bounds.floor_us` refuses, and the page prints
`peaks unavailable: <reason>`):

- **the machine** (`platform`, `machine`, `cpu_count`) must be the record's;
- **the Bend build** must be the record's (`peaks.bend` against `bend.version`);
- **the thread count** must be the record's: `peaks.threads` is the runtime's
  `--threads` the probe ran with, and `variant.threads` is the program's. A
  record at 1 thread is not scored against a peak at 16.

The file also records what makes the number a measurement rather than a guess:
`clock_ms` (Bend's `IO.now` counts whole milliseconds), `passes` (every phase
is repeated inside one timed region), `total_ms` (what was timed) and `pass_ms`
(what a stage is charged), and `checks` (`read <= write`, `reduce <= read`,
`sample_enough`: every sample at least 100 ms). A single cold pass of 16 MB is
a 3 ms number; the same work repeated 200 times is a 300 ms one, and only the
second can carry a floor.
