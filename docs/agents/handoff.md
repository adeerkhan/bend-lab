# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-11-16 — numerical lanes, the correctness campaign, and the run-record v3

The goal was taken through every row. A1–A3, B1–B6, C1–C4 and D1 are green
and committed, each row with its own gate, Makefile target, batch row,
fresh session run, ROADMAP/CHANGELOG entry and commit; all pushed to
`origin` (`adeerkhan/bend-lab`, `main`). `make check-fast` is **31/31** and
`make validate QUICK=1` passes. The full `make check` did **not** complete
in this session — two runs timed out on heavy gates under a loaded host;
see "Not verified" below. No tolerance, timeout or gate was loosened to
make a run fit. `.local/` was never committed; mimo and the architecture
comparison remain untouched.

### A. Numerical lanes (`docs/mamba3-lanes.md`)

- **A1** (`84d4c9f`): the SISO block at T = 1, 2, 4, 8, 16, 64 through the
  F32 twin (oracle), the Triton kernel with IEEE dots, with TF32 dots,
  and the public bf16 module. Forward, kernel backward and module
  backward tables. `make mamba3-lanes`: ieee ≤ 1e-05 (measured 2.6e-06),
  tf32 ≤ 5e-03 (measured 2.4e-03), module ≤ 5e-02 (measured 2.4e-02).
  `MAMBA3 LANES PASS`.
- **A2** (`43ed45a`): the byte-corpus tiny-LM for 20 steps in the CUDA F32
  twin lane and the public bf16 module lane; per-step losses and grad
  norms in `docs/mamba3-tiny-lanes.md`. bf16 stays within 5.7e-05 per
  step; both lanes land 512/2540 train hits. A lane measurement, not a
  verdict. `MAMBA3 BF16 CURVE PASS`.
- **A3** (`ea655e3`): T = 1 skip, T = 2 state, T = 64 chunk; closed forms
  match the twin exactly, IEEE kernel within 4.1e-07, DA_CS vs
  `cumsum(ADT*log2 e)` at 8.5e-08, bf16 module within 6.5e-03.
  `MAMBA3 CASES PASS`.

### B. Correctness campaign

- **B1** (`f9df70f`): finite differences at seqlen 4, δ = 1e-2, worst rel
  8.6e-05 vs the bumped-input twin, bound 1e-2. `MAMBA3 FD PASS`.
- **B2** (`1628d0b`): the exact-step decoder's state round-trip: 30 steps
  from fresh vs 10 + 20 resumed from the returned state, identical token
  for token (31 tokens); first 16 == the `gen_tokens` golden. Driver
  `probe_mamba3_state.bend` shares `S.layer_row`. `MAMBA3 STATE PASS`.
- **B3** (`0826520`): the seeded golden input never crosses 2π (~0.22 rad
  max), so a synthetic case fills `dt_bias` to 10.0 (~7 crossings/head,
  ~44 rad). The exact loop matches the twin at angle 1.1e-06 / y 3.6e-07;
  both decode paths emit the twin's 16 tokens exactly; wrap variants
  agree. `MAMBA3 WRAP PASS`.
- **B4** (`015f04d`): A at `-A_floor`, trap at exactly 0/1, softplus at
  large/small dd_dt. The ladder found and fixed the port's `softplus`
  overflow to `inf` past x ≈ 88.7 (torch's threshold=20 semantics);
  `make mamba3` green after the change. The twin's `heavy_tail` has a
  knife-edge nan gradient at `dd_A = 1.0` (torch `where` evaluates both
  branches); recorded, never fired in a gated run. `MAMBA3 CLAMPS PASS`.
- **B5** (`c13ac34`): three source corruptions built at run time — missing
  `shifted` (y rel 3.2e-01), wrong rope sign (6.6e-02), zeroed `D`
  (7.6e-01) — each blows the forward gate; the good program still
  matches at 5.5e-07. `MAMBA3 CANFAIL PASS`.
- **B6** (`3c0437e`): two separate builds print bitwise-identical seqlen-8
  forward tags; sumsq identical to 1e-9. `MAMBA3 DET PASS`.

### C. Run record v3

- **C1** (`ebc6623`): the landed schema is `mlbend.experiment/3`, a
  superset of the goal's "schema: 2"; `docs/agents/manifest.md` documents
  it with an example. `iris-crossval` (live reference) and `vit-step`
  (frozen) both write and PASS; the old `/1` records render.
- **C2** (`26c43b8`): `make report TASK=vit-step` renders the full Burn
  page — stages with floors, ladder (415 ms, 3 of 5 leaves), worklist,
  totals, metrics, PASS; AGAINST a cpu record prints `mismatch lane:
  cpu -> gpu` and `lane changed` in every delta row, never a percentage.
- **C3** (`c99da7d`): fixture stage refused (existing `test_stages.py`),
  a moved floor refused by `bounds_for` (new `test_bounds_for.py`:
  sidecar wins per path, record covers the rest, stale path dropped with
  a named refusal), and a mismatch banner gate (`test_against.py`:
  banner, `lane changed`, `added`, `gone`). Existing floors unchanged.
- **C4** (`8a15733`): the tiny-LM is an experiment task. One live run
  (1965 s, build 6.7 s, reference 9.1 s) PASSES: final loss -9.9e-07,
  bits/byte -1.6e-06, hits exact (1279, 512), w_sumsq -7.3e-04 abs vs
  the twin. Reference frozen from that record. `make report` rendered
  once into `results/mamba3-tiny-lm-page.txt`; ROADMAP §4 links the page,
  the card and the record. The first run's parser missed `show_tag`'s
  trailing space and the new task tripped the frozen-reference gate;
  both were fixed by measuring, not by loosening anything.
- **D1** (`a98cf97`, plus the generator follow-up): the card's variant
  (`bend`/`cpu`/threads unavailable/Bend 2.0.38, batch 1, positions 128)
  and `bounds: unavailable` (no stage split — never a zero or an
  estimate) are generated from the record by `test_mamba3_tiny_lm_run.py`
  so they survive regeneration. No new run, no architecture comparison.

### Not verified — the one honest delta

`make check` (the full gate) did not complete in this session. Two runs
each hit the **unchanged** 1800 s per-test timeout in
`reference/check_all.py`: the first at `test_mamba_vit_train.py`, the
second at `test_mamba3_tiny_lm_run.py` (141 ok checks before the second
failure). The host was running Dota 2 at ~248% CPU across both windows;
a timed-out gate also leaves its child Bend binary running (killed by
hand; recorded in `.local/issues-hit.md`). `make validate` inherits the
same failure at its `check` stage. No timeout, tolerance or gate was
loosened; the earlier recorded `158/158` from 2026-11-15 was not
re-verified in this session. `make check-fast` 31/31 and `make validate
QUICK=1` are green.

### Out of budget / blocked (unchanged)

- mimo: no vendored MIMO kernel or TileLang on this lane; untouched.
- Architecture comparison: no Bend Transformer, pinned flowers data or
  equal-parameter arms; recorded out of budget, not estimated.
- Full-corpus tiny-LM pass: `unavailable` at this budget; see the card.

### What changed in the code

- New gates: `reference/test_mamba3_{lanes,bf16_curve,cases,fd,state,wrap,
  clamps,canfail,det,bounds_for,against}.py`, `reference/mamba3_lanes.py`,
  `reference/mamba3_cases.py`, `reference/mamba3_wrap.py`.
- New probes: `probe_mamba3_state.bend`, `probe_mamba3_clamps.bend`.
- Behavior change: `mamba3_siso.bend`'s `softplus` uses torch's
  threshold=20 semantics; `reference/bounds.py`'s `bounds_for` merges
  per path and refuses a stale/moved floor.
- `reference/experiment.py` gains the `mamba3-tiny-lm` task; the tiny-LM
  card writer emits its variant/bounds from the record.
- Full measurement ledgers: `docs/mamba3-lanes.md`,
  `docs/mamba3-tiny-lanes.md`, `results/mamba3-tiny-lm.md`,
  `results/mamba3-tiny-lm-page.txt`; `make issues` re-rendered (145
  entries, `.local/` only).

### Last fix after the handoff draft

A full-check run saw a transient Triton `OutOfResources` in the lanes
gate (autotune chose a config needing 122384 B of shared memory on a
101376 B GPU). `reference/mamba3_lanes.py` now caps autotuned
`num_stages` to <= 2 in the dump subprocess — a compile-resource cap,
not a tolerance change. Fresh lane maxima: ieee 2.490e-06, tf32
2.392e-03, module 2.396e-02; `MAMBA3 LANES PASS`, check-fast 31/31.
The full `make check` delta above still stands (the two 1800 s
timeouts were not retried with any gate change).

## 2026-11-17 — follow-ups after the goal

- **Upstream:** BendTorch PR #42 (our MaxPool2d) merged into devel
  `8999c69`; #39 closed. nuxyel added the GPU twin (0 ulps at four
  shapes), seven mutants and the docs. Our follow-up PR #44 guards
  geometry by construction on both lanes (image as kernel + surplus,
  stride minus one, Conv2d's convention); `check-maxpool2d` and
  `check-gpu-layers` PASS. `make laws` reaches ALL PROOFS CHECK but its
  `fast.bend` defs check fails on `Array.get_unchecked` — reproduced on
  clean upstream devel, an environment/toolchain mismatch. PR #44 is
  waiting on the maintainer.
- **bend-lab #2:** the tiny-LM is instrumented. `make report
  TASK=mamba3-tiny-lm` now shows stages and bounds from the fresh
  2026-11-17 record: `run/step` 48,585 ms = forward 3,177 ms (6.5%) +
  loss_backward_opt 45,408 ms (93.5%); the matmul floors are ~1-3 ms,
  so overhead dominates this lane — the page's "a faster kernel" label
  is a worklist default, not a diagnosis. The card generator reads
  variant+bounds from the record, so regeneration cannot erase them.
- **bend-lab #3:** `make oracle-boundary` (static run-path check) and
  `make lane-matrix` (task/lane compliance page, 20/20) are committed;
  `record-compat` still renders schema /1 and /3 records; check-fast
  31/31.
- The compiler issue bendlang/bend#1403 was closed as not worth fixing
  (literal-Nat match memory; succ-pattern rows are the workaround).
- **Not done:** the full `make check` was deferred by the user; no
  full-gate number was recorded for this period. bend-ml#26's round-trip
  proof and BendTorch#36's four-batch run remain outstanding upstream
  promises; no ONNX work was added — the run path stays oracle-free.

## 2026-11-17 — research and upstream follow-ups

- The tiny-LM's step was decomposed with its own bench: optimizer walks
  0.06 s/step, array add of the same 71,952 cells 0.02 s, forward 3.1-3.7 s,
  scan ~7-8 s per two blocks, preprocessing weights ~14-80 s per two
  blocks depending on machine load. A synthetic 128-iteration loop of the
  per-token in_proj accumulation (outer + 19,456-cell flat add + transpose
  matvec) took 52.9 s in one run, two thirds of that phase: the hot loop is
  our flat-list gradient accumulation in `pw_fin2`, not the scan, the
  optimizer or a library op. Absolute times vary ~3x with CPU contention;
  within-run ratios only. `examples/language/mamba3/COMPARISON.md` has the
  tables and caveats.
- `make pages` rendered all 20 task pages from committed records into
  `results/pages/`; no runs were made for them.
- bend-ml#26: a one-cell round-trip law is proved on the fork branch
  (`feat/mat-roundtrip` 66bb483); the general law needs counter bits,
  `new_apart` and two-sided induction. BendTorch#36's four-batch UNet run
  is blocked by the checker's 16,384-row ceiling; reported on both issues.
- Full `make check` was not completed: resource-capped runs were stopped
  externally twice and once timed out on test_cifar with a 4 GiB cap that
  its 4.34 GiB peak could not fit. At 5 GiB that single gate passes in
  1:53; the full run was then externally terminated at 171 ok / 0 errors,
  only `test_vit_checkpoint.py` left unchecked. No gate or timeout was
  loosened.
