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
| `variant` | `trainer`, `lane` (`cpu`, `gpu`), `threads` (or `null` when not recorded), `bend` |
| `git` | `commit`, `branch`, `dirty` |
| `bend` | `version` |
| `packages` | every `mlbend-*`, `bend-ml-*` and `bendtorch-*` package at its hub hash |
| `machine` | platform, machine, processor, CPU count, Python and NumPy versions |
| `run` | `build_seconds`, `seconds_min`, `seconds_all`, `repeat`, `fetch_seconds`, `compute_seconds`, `reference_seconds`, `speed_ratio` |
| `stages` | the pre-order stage list (below); `[]` when the task declares no split |
| `stages_source` | `kind` (`plumbing`), `program`, `repeats`, `note`; `null` without stages |
| `counts` | per stage path, the named numbers that are not milliseconds (`rows_ms`, `from_list_ms`, and later `launches`) |
| `peaks` | the machine's measured rates, `{}` until `make peaks` writes them |
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
