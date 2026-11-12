# Track review, 2026-11-12

The lab now has five model tracks. This is the review matrix and the order to
re-run them on one lane. It is not a new benchmark; every cell names the
command that produced its last number.

## Current baseline

| check | command | last result |
| --- | --- | --- |
| proofs, frozen references | `make check-fast` | `31/31`, 2026-11-12 |
| full gate | `make check` | `140/140`, 2026-11-12, `54m39 s`, on the migrated lane |
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
| GLM | `make trace TASK=glm53-hybrid128 STEPS=30` | 30 steps, loss `5.5635 -> 3.2657`, eval `3.2838`, loop `72.9 s`, 118 system samples |
| GLM | `make experiment TASK=glm53-hybrid128` | PASS, one step, `5m55` |
| Mamba-3 | `make session TASK=mamba3 CMD="make mamba3"` | PASS, `22.2 s` |
| classical | `make experiment TASK=iris-crossval` | PASS, `1m05` |
| gate | `make session TASK=check-fast CMD="make check-fast"` | `31/31`, `23.6 s` |
| GPU | `make session TASK=gpu-probe CMD="reference/.venv-rf/bin/python reference/gpu_probe.py"` | `50.1 s`, GPU `52-100%`, process `326 MB` |

## One lane

The versions now agree (`reference/versions.py` prints one version per
package):

| piece | `bend.toml` | `.bend/env` | imports |
| --- | --- | --- | --- |
| Bend | `2.0.38` | `2.0.38` | n/a |
| BendTorch | `2.1.0.0` | `2.1.0.0` | `2.1.0.0` |
| BendTorch GPU | `2.1.0.0` | `2.1.0.0` | `2.1.0.0` |
| tensor-array | `0.1.20.0` | `0.1.20.0` | `0.1.20.0` |
| tensor-gpu | `0.1.5.2` | `0.1.5.2` | `0.1.5.2` |

The latest tensor-array `0.1.21.0` and tensor-gpu `0.1.5.3` are **not** a lane
with BendTorch 2.1.0.0: the trainer imports `0.1.20.0` and `0.1.5.2`, and the
two `Mat` types are nominally different even with the same shape
(ISSUE-118). The compatible lane is BendTorch's own closure; the unused
tensor-fork wildcard was removed because its latest pulls `0.1.21.0`.

The Bend pin remains 2.0.38: it is the installed binary and the Makefile pin.
2.0.40 needs its own toolchain change.

On this lane: `make check-fast` `31/31`, full `make check` `140/140` in
`54m39 s`, and `make mamba3` PASS. The migration touched 107 `.bend` files
and `bend.toml`/`bend.lock`; the env was purged and synced from the lock
(ISSUE-119).

| track | command | result |
| --- | --- | --- |
| flow | `make experiment TASK=flow64-first` | PASS, task `3.951 s`, build `53.784 s` |
| diffusion | `make experiment TASK=vit-first` | PASS, task `3.380 s`, build `245.884 s` |
| GLM | `make experiment TASK=glm53-hybrid128` | PASS, task `3.017 s`, build `91.818 s` |
| classical | `make experiment TASK=iris-crossval` | PASS, task `0.208 s`, build `119.090 s` |
| Mamba-3 | `make mamba3` | PASS, `23.35 s` |
| UNet science | `make unet-metric` | PASS · held `0.547`, ceiling `0.574`, span `94.9%` · can-fail white noise and mean · `2.4 s` |

## The matrix

| track | gate | short test | new feature to try | review action |
| --- | --- | --- | --- | --- |
| classical ML | `make experiment TASK=...` | `iris-crossval`, `wine-pca`, `transformer-periodic` | none; `Mat.reshape` only if a preprocess needs it | align the lane, rerun the report cards, compare to frozen references |
| diffusion | `reference/test_*.py`, `make experiment`, `make unet-metric` | `diffusion-2d`, `vit-first`, `cifar-10`, the UNet held-out gate | `Conv2dNB`, `GroupNorm`, `Tile`, `RoPE2d`, GPU lane | adopt the layers that replace local copies; rerun the gates |
| flow matching | `reference/test_flow64.py`, `make experiment` | `flow64-first`; the 6,000-step `flowers-flow` only when a gate or claim needs it | `Conv2dNB`, `GroupNorm`, GPU lane | check the forward/sampler gate; report legibility as measured, not gated |
| GLM-5.3 Flash | `make experiment TASK=glm53-hybrid128` | one hybrid step; 192 evaluation only after the list core changes | attention triangle and parallel forward, gradient accumulation, `Mat.reshape`, Mat/Array core | make the list core an array core first; then rerun the 192 lane |
| Mamba-3 | `make mamba3` | the SISO and full-LM gates; the `q.S^T` product benchmark | `Mat.matmul_nt`, `T.Layer`/`T.Train`, GPU lane | the Mat qS inside `scan_head` passed every gate but was slower (`0.84-1.18 s` against `0.71-0.73 s` for 20 forwards); next is the array core and a chunked scan, then `T.Layer` |

## Order

1. **The lane is aligned** (2026-11-12): BendTorch 2.1.0.0, tensor-array
   0.1.20.0, tensor-gpu 0.1.5.2, Bend 2.0.38. `make check-fast` `31/31` and
   full `make check` `140/140` in `54m39 s`. Keep new feature work off the
   lane change; the next item is the science/performance batch.
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

`make dashboard PORT=8787` serves `reference/dashboard.py` as **Bend-Lab**,
and `./scripts/dashboard.sh start` leaves it running under setsid+nohup (with
`status` and `stop`): one page for **every** run in the store, not one task at
a time, with a sun/moon light/dark switch. The overview
shows the number of live sessions, host CPU, RAM, GPU utilization and GPU
memory, and a table of runs with task, kind, status, PID, steps, loss,
parameter count, per-run CPU as host share plus cores, RAM, GPU memory and
loop time. Selecting a run
opens its loss and step time, up to four extra signals, CPU/RAM/GPU samples,
its configuration (program, arguments, lane, threads, seed, git, versions),
recent steps and the record it wrote. Several sessions can be live at once:
the sampler walks each run's process tree, so a Python test that spawns Bend is
sampled as one run.

`make session TASK=<name> CMD="make check-fast"` wraps any test command in the
same store (`reference/session.py`), with its log under
`results/<task>/runs/`. A training run that writes steps uses `make trace`
instead, so one run does not become two rows. `nvidia-smi` supplies the GPU
numbers; without it the page prints `unavailable`. No CDN, no build step,
standard library only.

Every run lives in `results/runs.sqlite3` (`reference/runs.py`): one row per
run and one per step, plus the run's own trace under
`results/<task>/runs/`. The page lists all runs in a selector and a clickable
table, so a run stays clickable after the next one overwrites
`results/<task>/trace.jsonl`; the
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
