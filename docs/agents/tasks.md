# Adding a task

A task is a dataset, a program and the numbers that program must produce, with
a tolerance for each. This is the whole path, in the order the runner needs it.
`docs/agents/adapters.md` says what a task *is*; this says how to add one.

## 1. The program

`examples/<area>/<name>/main.bend`. It prints one `name: value` line per metric
(the grammar the parse function reads), and it is deterministic: the seed is its
first argument, and `MLBEND_SEED` is set in its environment.

- Do not read an absolute path: the runner sets `MLBEND_DATA` to the pinned
  cache file and the program reads that.
- Nothing heavy at import (no `init_params` in a timed phase unless it is
  declared `once`).
- A phase that returns `Unit` is erased by the compiler and times as `0`; keep
  the work inside a value that is used.

## 2. The dataset pin

`dataset` names `name`, `rows`, `features`, `source`, and `files` — the exact
URLs — with `expected_sha256`. The runner fetches into
`reference/.cache/datasets/`, hashes what it read, and records the bytes and the
hash in the manifest. A warm cache is `fetch_seconds: 0`; a second run must not
touch the network.

## 3. The parse

`parse(stdout) -> {metric: value}` in `reference/experiment.py`. Anchor on a
full-line pattern (`^name: ([0-9.eE+-]+)$`), never on a substring: a metric that
moved into another line must fail loudly, not parse the wrong number.

## 4. The reference

`reference(context=None) -> {metric: value}` with the same metric names, in
NumPy or scikit-learn. `context` is what the program parsed, so a reference can
read the program's own partition (iris-crossval reads the fold assignment) and
compare numbers instead of procedures. When the reference is a second
implementation of the training, say in a comment which part is comparable and
which is `report` mode.

## 5. The metrics and the tolerances

```python
"metrics": {"mean_accuracy": {"tolerance": 0.000001, "mode": "absolute"}},
```

`mode` is `absolute`, `relative`, `exact` or `report`. `report` prints the two
numbers and passes without comparing: use it only for a number a different
convergence path cannot reproduce, and say why in the description. A tolerance
is set from a measured delta, never widened to make a run pass.

## 6. The seeds

`"seed": 0` for a deterministic program; `"seeds": [42, 43, 44, 45, 46]` when
the task shares a fold or initialisation sweep. The report is mean ± sample
deviation and passes only if every seed passes.

## 7. The stages (optional)

```python
"stages": {"program": "bench/plumbing/main.bend"},
```

points at a program that prints `<phase>_ms: value` lines; the runner builds it
once, runs it three times and keeps each key's median. The split is a benchmark,
not the task's run, and `stages_source.note` says so in the record. A task
without `stages` writes `[]` and every page still renders.

## 8. The bounds (optional)

Add a function to `reference/bounds.py` for the task's shapes (Appendix C of the
measurement plan: read, write, flops; two per multiply-add), then declare
`"bounds": floors.<function>`. The page prints a floor, the achieved percentage
and the gain for each stage whose path the function returns, and `unavailable`
for the rest. A bound is a **lower** bound: a stage measured below it means the
declaration is wrong, not that the kernel is magic.

## 9. Register and run

```sh
make experiment TASK=<name>          # card + record; PASS expected
make report TASK=<name>              # the page: stages, floors, worklist, metrics
```

Fix a failure in the program or the reference; do not widen the tolerance.

## 10. Freeze the reference

```sh
make references                      # results/reference/<name>.json, from the latest record
make references REFRESH=1            # recompute the file from its source record
make references-refresh              # recompute with the live functions (runs the corpus)
```

The gate compares against the frozen file, so it runs on a machine without
NumPy. A task without a frozen file is a gate error — `make check` says which.

## 11. Declare its cells

Add the task to `CELLS` in `reference/matrix.py` (and to `ABSENT` with a reason
when a lane cannot run it). A lane is a build fact: `gpu` links `libnvrtc`,
`cpu` must not; the matrix fails a cell whose label and binary disagree.

## 12. Finish

```sh
make matrix TASK=<name>              # the declared cells, one record each
make validate --quick                # the cheap gate (under a minute)
make check                           # the full gate
```

Then a line in `docs/CHANGELOG.md` under `Unreleased`, and the numbers in the
card. A task is done when its Bend number and its reference agree within the
declared tolerance, the page renders its stages and floors, and the gate is
green.
