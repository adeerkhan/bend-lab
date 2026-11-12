# Track review, 2026-11-12

The lab now has five model tracks. This is the review matrix and the order to
re-run them on one lane. It is not a new benchmark; every cell names the
command that produced its last number.

## Current baseline

| check | command | last result |
| --- | --- | --- |
| proofs, frozen references | `make check-fast` | `31/31`, 2026-11-12 |
| full gate | `make check` | `139/139`, 2026-11-11 |
| Mamba-3 SISO and full LM | `make mamba3` | `MAMBA3 PASS`, 2026-11-12 |

Short tests run on 2026-11-12, all on the current `2.0.0.0` lane:

| track | command | result |
| --- | --- | --- |
| classical | `make experiment TASK=cancer-knn` | PASS, accuracy `0.978102`, task `0.381 s`, build `59.977 s` |
| diffusion | `make experiment TASK=diffusion-2d` | PASS, task `103.705 s`, build `8.961 s` |
| diffusion | `make experiment TASK=vit-first` | PASS, task `1.275 s`, build `121.388 s` |
| flow | `make experiment TASK=flow64-first` | PASS, task `4.839 s`, build `103.179 s` |
| GLM | `reference/trace.py --task glm53-hybrid128 --steps 2` | 2 lines, median `1476 ms`, run `2.979 s`, build `49.8 s` |
| Mamba-3 | `make mamba3` | PASS, `46.5 s` total; train probe `9.81 s` (the probe is noisy: `5.25-10.50 s` over three runs) |

## The lane is not one lane

Before any feature test, the versions have to agree. Right now:

| piece | `bend.toml` | `.bend/env` and examples | latest upstream |
| --- | --- | --- | --- |
| Bend | `2.0.38` | `2.0.38` | `2.0.40` |
| BendTorch | `1.1.1.0` | `2.0.0.0` | `2.1.0` |
| tensor-array | `0.1.14.0` | `0.1.18.0` | `0.1.21.0` |
| tensor-gpu | `0.1.5.0` | `0.1.5.0` | `0.1.5.3` |

A `make check` on this tree tests the old imports, not the new features. The
first refix is a **single review lane**: Bend 2.0.40, BendTorch 2.1.0,
tensor-array 0.1.21.0, tensor-gpu 0.1.5.3, tensor-fork 0.1.7.2, with every
`import` and `bend.toml` entry saying the same thing. Add a `make versions`
that prints the pin, the env, the imports and the hub hash; a manifest already
records them per run.

## The matrix

| track | gate | short test | new feature to try | review action |
| --- | --- | --- | --- | --- |
| classical ML | `make experiment TASK=...` | `iris-crossval`, `wine-pca`, `transformer-periodic` | none; `Mat.reshape` only if a preprocess needs it | align the lane, rerun the report cards, compare to frozen references |
| diffusion | `reference/test_*.py`, `make experiment` | `diffusion-2d`, `vit-first`, `cifar-10` | `Conv2dNB`, `GroupNorm`, `Tile`, `RoPE2d`, GPU lane | adopt the layers that replace local copies; rerun the gates |
| flow matching | `reference/test_flow64.py`, `make experiment` | `flow64-first`; the 6,000-step `flowers-flow` only when a gate or claim needs it | `Conv2dNB`, `GroupNorm`, GPU lane | check the forward/sampler gate; report legibility as measured, not gated |
| GLM-5.3 Flash | `make experiment TASK=glm53-hybrid128` | one hybrid step; 192 evaluation only after the list core changes | attention triangle and parallel forward, gradient accumulation, `Mat.reshape`, Mat/Array core | make the list core an array core first; then rerun the 192 lane |
| Mamba-3 | `make mamba3` | the SISO and full-LM gates; the `q.S^T` product benchmark | `Mat.matmul_nt`, `T.Layer`/`T.Train`, GPU lane | the Mat qS inside `scan_head` passed every gate but was slower (`0.84-1.18 s` against `0.71-0.73 s` for 20 forwards); next is the array core and a chunked scan, then `T.Layer` |

## Order

1. **Align the lane** and add `make versions`. Run `make check-fast`; then the
   full `make check`. A version bump is its own commit with a `NOTES.md`
   entry; no feature work rides on it.
2. **Baseline the five tracks** with the short tests above, using the frozen
   references. Write the date, version, command and result into the table.
3. **Refix what the baseline finds**: stale import versions, a gate that no
   longer runs, a report card that drifted, a doc claim that no longer holds.
4. **Adopt one feature per track**, each with a before/after:
   - Diffusion/flow: `Conv2dNB`, `GroupNorm`, `Tile`, `RoPE2d`.
   - GLM: triangle attention, gradient accumulation, then the array core.
   - Mamba-3: the array core, then a chunked scan, then `T.Layer`/`T.Train`;
     the one-call Mat qS was measured slower and is recorded in
     `examples/language/mamba3/COMPARISON.md`.
   - Classical: version bump only; it is the control.
5. **Upstream**: review #42 (`MaxPool2d`); finish the #26 `Mat` round-trip
   corollary. Add `Rewire` or attention backward blocks only when a second
   user or a measurement asks for them.

## The browser page, and the one-hour budget

`make dashboard TASK=<task> PORT=8787 [PID=<pid>]` serves
`reference/dashboard.py`: the same trace and record as the terminal
`reference/dash.py`, as HTML on a local port. It shows the loss, step time,
tokens/s, resident and peak memory, the stage split, the recent steps and any
extra numeric fields in a trace line. Extras are the hook for model internals:
write `lr`, `grad_norm`, `state_norm` or `gate_mean` and they become small
charts. No CDN, no build step, standard library only.

Every run lives in `results/runs.sqlite3` (`reference/runs.py`): one row per
run and one per step, plus the run's own trace under
`results/<task>/runs/`. The page lists all runs in a selector, so a run stays
clickable after the next one overwrites `results/<task>/trace.jsonl`; the
database and those traces are observations and are ignored by git. A missing
run prints `unavailable`, never a zero. `trace.py` writes the database while
the program runs and marks the run `done` or `aborted` at the end;
`experiment.py` records a finished report card as a run with its record but no
per-step trace, and the page labels it `record only`. On an empty database the
dashboard imports every `results/latest/<task>.json` and the task's latest
trace; `reference/runs.py --import-latest` does the records by hand, and
`MLBEND_RUNS_DB` points the store at another file for tests.

`make trace` now stops a run after one hour by default
(`reference/trace.py --max-seconds 3600`; `MAX_SECONDS=0` disables it). The
page shows the loop time against the same budget in red when over. The
existing `glm53-hybrid128` trace is **3,733.97 s, 62 min**, over the budget;
that is the reason the guard exists.

The one-hour rule is about wall time, not a target: a run that needs longer
is split, resumed from a checkpoint, or not run.

## What not to do

- Do not rerun the 6,000-step flow or the 5.6-hour GLM 192 evaluation just to
  see a green number; run the gate that covers the claim.
- Do not claim a new feature is tested when it is only available in a version
  the tree does not import.
- Do not put only the dense layers on the Bend GPU lane around a CPU scan;
  the transfer and launch cost can be larger than the win.
- Do not add `Rewire` or a batched attention backward without a measured
  shape that needs it.
