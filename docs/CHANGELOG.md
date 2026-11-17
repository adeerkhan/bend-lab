# Changelog

What changed for the people who run the lab. Internal refactors are not listed.

## Unreleased

### Lanes gate: the autotune cap follows upstream #997

- `reference/mamba3_lanes.py` now keeps only `num_stages=1` configs. The
  vendored mamba commit predates state-spaces/mamba#997, which found that
  stages 2/3 silently corrupt `mamba3_siso_fwd` on Blackwell (sm_120) and
  that stage 3 can ask for 122384 B of shared memory on a 101376 B GPU.
  The earlier `<= 2` cap fixed the OOM but allowed an upstream-unsafe
  config; this one is the documented-correct stage. Fresh lane maxima are
  unchanged (2.490e-06 / 2.392e-03 / 2.396e-02); `MAMBA3 LANES PASS`.


### Upstream after the fetch: PR #42 is mergeable again

- BendTorch devel moved to `4528d08` (AdaIN #43), which made our MaxPool2d
  PR #42 conflict in the Makefile and CHANGELOG. Rebased our fork branch
  `feat/maxpool2d` onto it (`6e4e1f2`), keeping both AdaIN's and
  MaxPool2d's checks and entries; GitHub reports `mergeable=true`.
  Re-verified: `check-maxpool2d` PASS, `check-adain` PASS, `PROOF.bend
  --verdict` ALL PROOFS CHECK. Evidence on the PR.
- bendlang/bend now tags **v2.0.40** upstream; this lab stays on 2.0.38 —
  a version change is a banner, never a delta.


### The full check's timeout now kills the gate's whole subtree

- `reference/check_all.py`'s `run()` starts each gate in its own process
  group and, on the unchanged 1800 s timeout, sends SIGTERM then SIGKILL
  to the group. A timed-out gate no longer leaves its Bend binary at
  99% CPU loading the next gates into their own timeouts. The timeout
  and every gate tolerance are unchanged; a timeout is still a FAIL.
- Verified with a fixture that spawns a sleeping child: the parent gate
  prints `timed out after 2s; the process group was killed`, exits 1,
  and the child PID is gone. `make check-fast` stays 31/31.


### The lanes gate under a loaded GPU: a compile-resource cap

- A full-check run raised `OutOfResources` (Triton autotune selected a
  num_stages=3 config needing 122384 B of shared memory on a 101376 B
  GPU). `reference/mamba3_lanes.py` now caps autotuned stages to <= 2
  in its dump subprocess — a compile-resource cap, not a tolerance
  change. Re-measured from fresh dumps: ieee 2.490e-06, tf32 2.392e-03,
  module 2.396e-02, all within the committed bounds; `MAMBA3 LANES PASS`,
  check-fast 31/31. Ledger entry added.


### D1 follow-up: the card keeps its variant/bounds across regeneration

- `reference/test_mamba3_tiny_lm_run.py` now generates the card's
  `variant` / `bounds` section itself, reading the variant from
  `results/latest/mamba3-tiny-lm.json` and printing `bounds: unavailable`
  with the reason (or `unavailable` + the command when no record exists).
  A hand-edit used to vanish the next time `make check` ran this gate;
  the regenerated card now keeps it. No number is invented by the writer.
- Full `make check` did **not** complete in this session: two runs each
  hit the unchanged 1800 s per-test timeout (first
  `test_mamba_vit_train.py`, then `test_mamba3_tiny_lm_run.py`, with 141
  ok checks before the second failure). The host was running Dota 2 at
  ~248% CPU; a timed-out gate also left its Bend binary running and was
  killed by hand. No timeout, tolerance or gate was loosened. See
  `docs/agents/handoff.md` and `.local/issues-hit.md`. `make check-fast`
  and `make validate QUICK=1` are green (31/31).


### Mamba-3 tiny-LM card, row D1: variant and bounds, no new run

- `results/mamba3-tiny-lm.md` gains a `variant` / `bounds` section read
  from the C4 run record: trainer bend, lane cpu, threads unavailable,
  Bend 2.0.38, batch 1, positions 128; `bounds: unavailable` because the
  task declares no stage split (no plumbing program for this two-layer
  LM), never a zero or an estimate. The card links the rendered page,
  the record and the frozen reference; no new run, no new number, no
  architecture comparison.


### The run record v3, row C4: the tiny-LM page and one gate

- `reference/experiment.py` gains the `mamba3-tiny-lm` task: the two-layer
  SISO LM on the byte corpus, 20 steps, batch 1, 127 next-byte targets,
  the fixed seeds 101/102/20261115, compared against the twin retrained
  live (CPU F32). Run 2026-11-16: bend vs twin final loss -9.9e-07,
  bits/byte -1.6e-06, held-out/train hits exact (1279, 512), w_sumsq
  -7.3e-04 abs (a 2.6e-07 relative difference); run 1965 s, build 6.7 s,
  reference 9.1 s; PASS. The frozen reference is
  `results/reference/mamba3-tiny-lm.json`.
- Two real failures were fixed, not waved through: the first run's parser
  missed `show_tag`'s trailing space (all metrics printed `-`), and the
  new task then tripped the frozen-reference gate (`make validate QUICK=1`
  went red). The parser regex now accepts the trailing space, the second
  run passes, and the reference was frozen from that passing record —
  no tolerance was loosened. The failed raw record is not published.
- `make report TASK=mamba3-tiny-lm` was rendered once into
  `results/mamba3-tiny-lm-page.txt` (no stages declared; six metrics and
  PASS), linked from ROADMAP §4 with the report card and the record.
- `make validate` is the one gate for the lab; its full run's number is
  recorded in `docs/agents/handoff.md` when it completes.


### The run record v3, row C3: the refusal gates

- Fixture stage: existing behavior pinned by `reference/test_stages.py` —
  a stage program that prints `the dataset failed` is refused by the
  capture (`run_program` raises); a normal split is not refused.
- Moved floor: `reference/bounds.py`'s `bounds_for` merged per stage path
  with a refusal for sidecar paths that are not stages of this run — the
  moved floor is dropped, the refusal is printed once naming the sidecar
  and the path, and every other path still scores. New gate
  `reference/test_bounds_for.py` (no sidecar / overlap / extends / single
  moved floor / all-moved sidecar). Existing sidecars are unchanged and
  still render (`make report TASK=vit-step` shows the same floors).
- Mismatch banner: new gate `reference/test_against.py` synthesizes two
  records (cpu -> gpu, one renamed stage, one new stage) and asserts the
  page prints `mismatch lane: cpu -> gpu`, `lane changed` in every
  overlapping delta row (never a percentage), `added` for the new stage,
  `gone` for the vanished one.


### The run record v3, row C2: the one-page report, verified

- `reference/report_run.py` prints Burn's order — header (task,
  trainer/lane/threads/Bend/machine/cpus), the run, the stages with
  path/ms/share/kind/bound/achieved/floor ms/gain ms, totals
  (unaccounted/unclaimed), the ladder ("how much is on the table"),
  the worklist, the metric table, PASS/FAIL.
- `make report TASK=vit-step` renders the latest record (2026-11-16):
  stages `run` 458.7 ms total; `run/step/forward` 205.0 ms at 44.7 %
  share, 10.0 % of its floor (20.55 ms, gain 184.45);
  `run/step/loss_backward_opt` 213.7 ms, 19.2 % of its floor (41.11 ms,
  gain 172.56); ladder 415.0 ms with 3 of 5 leaves floored; worklist
  ranks the two kernels first (184.4 ms, 172.6 ms), then run/init
  (20.0 ms), run/step/input (17.0 ms, less preprocess), host 58.0 ms;
  unaccounted 0.0 ms, unclaimed unavailable.
- `make report TASK=vit-step AGAINST=results/raw/vit-step-20261101T083152Z.json`
  (cpu record): `mismatch lane: cpu -> gpu` prints as a banner and every
  delta row says `lane changed` — never a percentage. Not a regression
  verdict; a variant banner.
- The page is unchanged; this row is the verification and the record of it.


### The run record v3, row C1: schema and docs verified, not rewritten

- The landed schema is `mlbend.experiment/3` (superset of the goal's
  `schema: 2`; documented in `docs/agents/manifest.md`). Verified on this
  lane, 2026-11-16: `make experiment TASK=iris-crossval` (5-seed live
  reference, PASS) and `reference/experiment.py --task vit-step --seeds 0
  --references frozen --no-latest` (PASS against the frozen reference,
  the command `make validate` runs); the old `mlbend.experiment/1`
  records render with `reference/report_run.py` (checked on
  results/raw/attention-parity-20261023, prints `task`, `run`, metric
  table, PASS).
- No experiment, timing number, or bound was changed. Verification runs
  did not commit new raw records; the checked-in 5-seed iris record is
  untouched.


### The correctness campaign, row B6

- `make mamba3-det` (`reference/test_mamba3_det.py`): two separate
  builds of `mamba3_siso.bend` print bitwise-identical seqlen-8
  forward tags (out_v, y_inner, y); sumsq identical to 1e-9.
  `MAMBA3 DET PASS`.


### The correctness campaign, row B5

- `make mamba3-canfail` (`reference/test_mamba3_canfail.py`): three
  corruptions of the Bend source, applied at run time to
  .scratch/canfail/ copies — missing `shifted` (y rel 3.2e-01), wrong
  rope sign (6.6e-02), zeroed `D` (7.6e-01). Each must exceed the
  forward gate 1e-5, i.e., the mutation would fail `make mamba3`; the
  good program still matches the golden at 5.5e-07.
  `MAMBA3 CANFAIL PASS`.


### The correctness campaign, row B4

- `make mamba3-clamps` (`reference/test_mamba3_clamps.py`): A at
  -A_floor exactly (clamp knife edge and below), trap saturating to
  exactly 0/1, softplus at large/small dd_dt. Found and fixed in the
  same row: the port's `softplus` computed `log(1+exp(x))`, which
  overflows F32 past x ~ 88.7 where torch yields x; fixed to torch's
  threshold=20 semantics, `make mamba3` green after the change.
  `MAMBA3 CLAMPS PASS`.
- Also recorded: the twin's `heavy_tail` yields a nan gradient at
  dd_A == 1.0 exactly (torch `where` evaluates both branches; the
  reciprocal branch hits 0*inf). Never fired in the gated runs; ledger.


### The correctness campaign, row B3

- `make mamba3-wrap` (`reference/test_mamba3_wrap.py`): a synthetic
  sequence whose angle cumsum crosses 2pi (~7 crossings per head;
  dt_bias filled to 10.0, the seeded golden input never crosses). The
  exact loop matches the twin at angle 1.1e-06 and y 3.6e-07; the
  full-prefix pass and the exact step both emit the twin's 16 tokens
  exactly; the two wrap variants agree at F32. `MAMBA3 WRAP PASS`.


### The correctness campaign, row B2

- `make mamba3-state` (`reference/test_mamba3_state.py`): the exact-step
  decode run twice — 30 steps from a fresh state vs 10 steps then 20
  resumed from the returned state — token for token identical (31
  tokens), and the first 16 equal the `gen_tokens` golden. Driver:
  `examples/language/mamba3/probe_mamba3_state.bend`, which shares
  `S.layer_row` rather than copying it. `MAMBA3 STATE PASS`.


### The correctness campaign, row B1

- `make mamba3-fd` (`reference/test_mamba3_fd.py`): dL/du by autograd
  through the twin vs central finite differences (delta = 1e-2) on the
  bumped-input twin at seqlen 4, worst rel 8.6e-05, bound 1e-2.
  `MAMBA3 FD PASS`.


### The kernel's explanatory cases: T = 1, T = 2, T = 64 (A3)

- `make mamba3-cases` (`reference/test_mamba3_cases.py`) re-derives the
  closed forms: at T = 1 the output is the `D + gamma*(q.k)` skip (SSM
  term is zero); at T = 2 token 0 writes the state and token 1 reads it;
  at T = 64 (one full chunk) the kernel's in-chunk cumulative decay
  `DA_CS` equals `cumsum(ADT * log2 e)` (8.5e-08). The twin matches its
  own closed forms exactly (0.0); the IEEE kernel lane lands within
  4.1e-07 of the twin; the bf16 module within 6.5e-03. Short table in
  `docs/mamba3-lanes.md` (the A3 section). `MAMBA3 CASES PASS`.


### The tiny-LM 20-step curve in two lanes (A2)

- `make mamba-bf16-curve` (`reference/test_mamba3_bf16_curve.py`) trains
  the byte-corpus tiny-LM identically in the CUDA F32 twin lane
  (explicit t-by-t loop) and the public bf16 `Mamba3` module lane (same
  data, same seeds 101/102/SEEDX 20261115, same AdamW + clip + warmup /
  cosine schedule, 20 steps, 127 targets per example). The bf16 curve
  stays within 5.7e-05 of the F32 curve per step (max at step 19), the
  gradient norms within 6.7e-05, and both lanes land 512/2540 train
  hits. The CUDA F32 record agrees with the recorded CPU F32 run
  (`twin.json`) to 4.8e-07 per step — device is a banner, never a delta.
  A lane measurement, not a verdict; see `docs/mamba3-tiny-lanes.md`.
- The gate pins the recorded CUDA F32 curve (deterministic per build),
  asserts bf16-vs-f32 ≤ 5e-04 (measured 5.7e-05), the hits equality, and
  finiteness. `MAMBA3 BF16 CURVE PASS`.


### Mamba-3 numerical lanes: the SISO block through four lanes (A1)

- `make mamba3-lanes` (`reference/test_mamba3_lanes.py`) dumps the SISO
  block at T = 1, 2, 4, 8, 16, 64 through four lanes — the F32 twin
  (oracle), the vendored Triton kernel with IEEE dots, the same kernel
  with TF32 dots, and the public bf16 `Mamba3` module — and asserts the
  per-lane tripwires recorded in `docs/mamba3-lanes.md`: ieee ≤ 1e-05
  (measured 2.56e-06), tf32 ≤ 5e-03 (measured 2.39e-03), module ≤ 5e-02
  (measured 2.40e-02). `MAMBA3 LANES PASS`.
- `docs/mamba3-lanes.md` is the table: forward (out_v, q, k*sc, qk,
  scale, gamma, final state), kernel backward (dQ/dK/dV-equivalent input
  grads), module backward (du, dW_in, dW_out, the norm weights, biases).
  The kernel lane stores SSM states in bf16 by contract
  (`mamba3_siso_fwd.py:604`) and everyone inherits that rounding; the F32
  lanes cannot go through `mamba3_siso_combined` (it always casts to
  bf16), so they call the low-level kernels and replicate the backward
  step by step; `TRITON_F32_DEFAULT` is read at Triton import, so the
  ieee/tf32 lanes run as separate processes.
- Two accounting notes are documented in the doc because they look like
  precision bugs and are not: the twin's DT gradient includes the
  `ADT = A*DT` path while the kernel returns it separately as dADT, and
  the kernel's incoming gradient is `silu(z) * (dy @ W_out)` because the
  twin's `y` is post-`out_proj`.
- T = 64 is also the chunk boundary. The bf16 module remains a reported
  lane and is never the oracle.

### Mamba-3 tiny-LM: AdamW + clipping + warmup/cosine against the twin (B2)

- `examples/language/mamba3/train_mamba3_tiny.bend` trains the two-layer SISO
  Mamba-3 LM on the byte corpus: the first 20 train examples, batch 1 in file
  order, 127 next-byte targets per example (the last position is dropped — the
  corpus is one stream, so no target ever crosses an example boundary).
  AdamW (lr 1e-3, weight_decay 0.01, betas (0.9, 0.999), eps 1e-8, bias
  correction, decoupled decay), gradient clipping by global norm at 1.0,
  warmup 5 steps linear then cosine from 1e-3 to 1e-4; the eps, the clipping
  formula and the schedule are identical on both sides.
- The weights travel as one flat 71,952-value list in the manifest order and
  the AdamW moments as flat lists, so the update never round-trips a record.
  The embedding gradient is scattered by token id into the 127x64 table (a
  token's gradient is the sum of the per-position gradients of every position
  that used it) — the copy-task SGD path applies the (8, 64) table-row
  gradient directly, which is only valid because the copy task is a
  permutation.
- `make mamba3-tiny-opt` (`reference/test_mamba3_tiny_opt.py`) regenerates the
  twin dump (`reference/mamba3_tiny_twin.py`, float32 CPU, fixed seeds
  101/102/20261115), runs the same schedule in both, and checks: the loss at
  every step at 1e-3 (worst ~1e-07), the gradient norm at every step at 1e-3
  (both sides clip at 1.0 several steps), the train hits (512/2540 on both
  sides) and the final per-tensor sums of squares at 1e-3.
  `MAMBA3 TINY OPT PASS`. Measured wall time ~17 min for 20 steps and the
  held-out evaluation on this lane.
- The full corpus (64 examples, one pass) is not the budget of this row: a
  128-position forward+backward is what it costs, and the run uses the first
  20. The held-out evaluation is after the last step, which is "every N
  steps" with N = steps; mid-training evaluation is unavailable at this cost.
- B3 (the held-out compare and the report card) lands with its own commit.

### Mamba-3 tiny-LM: the held-out run is a report card, not a win (B3)

- `make mamba3-tiny-lm` (`reference/test_mamba3_tiny_lm_run.py`) runs the same
  twin dump/train and the same Bend trainer as the B2 gate, then compares the
  held-out metrics on the 4,096-byte split (4,064 next-byte targets): loss
  curve 1e-3, gradient norms 1e-3, train hits within 2 examples, final weights
  at 1e-3, bits/byte within 0.02 and token accuracy within 2 points.
- `results/mamba3-tiny-lm.md` records bits/byte, token accuracy, step count,
  schedule and seed, and the baselines (unigram 4.8093, bigram 3.3569
  bits/byte). The number is what 20 steps of 1e-3 AdamW on 20 train examples
  reach: the twin and the Bend trainer agree to ~1e-07 per step, and **both**
  numbers are far above the bigram baseline — the report card says so and
  does not claim a language-model result. The full corpus pass is unavailable
  at this budget.

### Mamba-3 decode step: varlen, GQA and outnorm breadth rows (C1/C2/C3)

- `examples/language/mamba3/step_mamba3_varlen.bend`: the two-layer LM
  decodes the packed prompt `[0,1,2,3|0,1,2,3]`, the state reset at row 4
  (angle, SSM array and the trapezoid's shifted term). The gate
  (`reference/test_mamba3_step_varlen.py`, `make mamba3-step-varlen`) compares
  step vs full-prefix vs the SISO-config twin's golden at 16 and 64 tokens,
  exactly: all three agree on `0 1 2 3 0 1 2 3 3 3 ...`.
- `examples/language/mamba3/step_mamba3_gqa.bend` (C2): the step at ngroups 2,
  rope_fraction 1.0 (per-v-head B/C biases, group-indexed B/C reads, 8 angle
  pairs). `reference/test_mamba3_step_gqa.py` (`make mamba3-step-gqa`) checks
  step == full-prefix == the GQA twin's golden at 16 and 64 tokens: all three
  agree on `0 1 2 3 0 3 0 3 ...` at 16 (the model, a fresh random-weight LM,
  settles into a 0/3 cycle; the golden pins it down).
- `examples/language/mamba3/step_mamba3_outnorm.bend` (C3): the step with
  `is_outproj_norm=True` (out_v ungated, then `y_inner = rms32(out_v) *
  norm.weight * silu(z)` and out_proj). `reference/test_mamba3_step_on.py`
  (`make mamba3-step-outnorm`) checks step == full-prefix == the outnorm
  twin's golden at 16 and 64 tokens: all three agree on `0 1 2 3 0 3 0 3 ...`
  at 16 and at 64.
- The probe compares against `reference/mamba3_cfg_lm.py`'s dump and greedy
  golden (`.scratch/<cfg>_lm_data/`), never against itself; the gate regenerates
  the dump and goldens before building. The probes found a wire bug: the SISO
  decode is the single-block model (no residual, no final RMSNorm,
  weight-tied layers), and a naive port decodes a different model; the
  full-prefix pass disagreed and the probes were rewired to the residual+RMS
  model (`h = u + y` per layer, RMSNorm before the head).
- `reference/mamba3_cfg_lm.py` writes the dumps and the goldens for the three
  breadth configs; the packed prompt's reset matches the reference's varlen
  semantics (per-segment independent passes, concatenated).

### Mamba-3 tiny-LM: next-token loss with explicit targets is gated (B1)

- `loss_tgt_go`/`dy_all_tgt_go`/`dWh_tgt_go` in `mamba3_siso.bend` are the
  byte-LM forms of `loss_go`/`dy_all_go`/`dWh_go`: they read the target per
  position from a list instead of using the row index, and take the vocab
  and the mean-over-targets scale as parameters. The copy-task functions
  are untouched.
- `make mamba3-lm-tgt` (`reference/test_mamba3_lm_tgt.py`) dumps fixed y
  (8x64), Wh (8x64) and targets (3, 1, 4, 1, 5, 7, 2, 6), builds
  `probe_mamba3_lm_tgt.bend`, and compares against a float64 NumPy twin at
  1e-5: loss rel `2.6e-08`, dy `1.9e-07`, dWh `2.0e-07`.
  `MAMBA3 LM TGT PASS`; build 2.6 s, run 0.1 s.

### MambaViT: the subset trainer, parity with the twin (A3)

- `examples/language/mamba_vit/train_mamba_vit.bend` trains the model A1b and
  A2 define: one flat 83,050-value parameter list, cross entropy with the
  mean-over-batch gradient, global-norm clipping at 1.0 and AdamW in torch's
  order, batch 32 in file order, constant lr.
- `reference/mamba_vit_train_data.py` writes the fixed MNIST slice (the
  first 2,048 train and 512 test images as 0..1 float32), the seed-0 init as
  108 per-tensor files plus one flat file, and a manifest with every sha256
  and flat offset. `reference/mamba_vit_twin_train.py` runs the twin on the
  same data and schedule in float64 and records `twin.json`.
- `make mamba-vit-train` (`reference/test_mamba_vit_train.py`) checks the
  manifest hashes, the loss curve every step at 1e-3 relative, both
  accuracies within 2 points and the 108 final per-tensor sums of squares at
  1e-3: **`MAMBA-VIT TRAIN PASS`**, and the batch row runs it.
- **The full epoch is unavailable at this budget and the gate says so.** The
  twin's **94.61 % after one 60,000-image epoch** needs 1,875 steps of batch
  32. The measured Bend run (2026-11-15, `make mamba-vit-train`) is
  1869.8 s wall for the 160 training + 256 held-out images (4.49 s per
  image), so the fixed 2,048-image subset is on the order of hours and a
  60,000-image epoch on the order of days; the run trains 160 images (5 steps
  of 32) and evaluates 256 held-out ones. The twin's 94.61 % after one full
  epoch has no Bend counterpart here, and the gate prints that instead of a
  number.

### MambaViT: the backward is gated, and the scan's slot order was reversed (A2)

- `probe_mamba_vit_bwd.bend` adds a hand-written backward through the head,
  the mean pool, the final LayerNorm, the eight blocks (out_proj, the SiLU
  gate, the 64x16 scan running t backwards, `dt = softplus`, `x_proj`, the
  conv's three padded taps, `in_proj`, the block LayerNorm), the residual
  stream, the position/cls add and the patch embedding. It prints
  `mv_g_*` for every gradient the dump carries: 108 parameter tensors and
  `g_emb`, `g_tokens`, `g_block0..7`, `g_final`, `g_pooled`.
- `reference/mamba_vit_bins.py` also writes the A2 oracle: `dy`, the ten
  values fixing `loss = sum(logits * dy)`, and the gradient of that scalar
  for every parameter and those twelve intermediates. The 130 forward files
  are byte-identical to the ones A1a/A1b were gated on.
- `make mamba-vit-bwd` (`reference/test_mamba_vit_bwd.py`) compares all 120
  at 1e-4 relative: worst `g_layers_4_mixer_A_log` at `5.67e-06`, median
  `3.8e-07`, 22 above `1e-06` and all of those the gradients of the SSM
  tensors, whose true values are `1e-11` to `1e-15`. Build 6 s, run 11 s;
  `MAMBA-VIT BWD PASS`, and the batch row runs it.
- **The scan's 16 slots were stored in reverse.** `upd_c` reversed each
  channel's row before prepending it and then reversed the whole list, so
  the two reversals cancelled across rows but not inside them. The model was
  self-consistent and the forward gate could not see it (the state is
  `~1e-7` of `y`; `D * x` carries the rest), but the three gradients that
  read the state (`dC`, `dA`, `d(dt)`) were 90 to 100 % off while `dB`,
  `dD`, `in_proj`, `out_proj`, `conv1d` and the norms matched. Deleting the
  inner reverse fixed it: the state now matches a NumPy replica at
  `1.6e-06` relative and the forward's block tags improved fourfold (worst
  `block5` from `2.16e-06` to `4.15e-07`).

### MambaViT: the whole forward is gated (A1b)

- `probe_mamba_vit_emb.bend` now runs the eight Mamba-1 blocks, the final
  LayerNorm, the mean pool over the 50 tokens and the head, loading each
  block's twelve tensors inside one IO recursion and printing
  `mv_block0..7`, `mv_block0..7_res`, `mv_final`, `mv_pooled` and
  `mv_logits` beside the A1a tags.
- `make mamba-vit-fwd` compares all twenty-three tags against
  `reference/mamba_vit_bins.py`'s dump at 1e-5: the worst is `block5` at
  rel `2.2e-06`, the residual rows and `tokens` at `2.0e-07`, `logits` at
  `1.8e-07`. `MAMBA-VIT EMB PASS` covers the two embedding tags and
  `MAMBA-VIT FWD PASS` everything; the batch row runs the gate instead of
  `not run`.
- The block keeps the 64x16 scan as a flat list rebuilt per token, so no
  `Array<F32> & F32` threading is needed; the notes in the README record
  why, and the three Bend idioms (no destructuring of a computed value, no
  pure `let` in a `do` block, one flat list per pass) that cost rewrites.

### Mamba-3 GQA: forward and backward at ngroups 2, rope_fraction 1.0

- `forward_gqa` and `backward_gqa` in `mamba3_siso.bend`: the in_proj row is
  340 wide, B/C are per group (2) while the B/C biases stay per v-head,
  v-head h reads group h//2, the angle state is per v-head with 8 pairs (all
  16 q/k dims rotated), and the backward carries the angle widths (4->8),
  the group-indexed dCn/dBn and dW_norm, and the 340-row dW_in.
- `make mamba3-gqa` matches `reference/mamba3_gqa_twin.py`'s CPU twin at
  every forward intermediate (y rel `6.7e-07`, gate `1e-5`) and at every
  gradient (all `<= 1.1e-06`, gate `1e-4`; the angle gradient summed over
  tokens, as the SISO gate does). Build 4.6 s, run 0.44 s; `MAMBA3 GQA PASS`.

### MambaViT: the patch embedding and tokens are gated (A1a)

- `probe_mamba_vit_emb.bend` implements the ViT half of the reference's
  forward: the 4x4 patch extraction, `LN -> Linear -> LN` embedding, the cls
  token and the position embedding. `make mamba-vit-emb` compares `emb`
  (49x32) and `tokens` (50x32) against `reference/mamba_vit_bins.py`'s
  dump at 1e-5: rel `2.8e-07` and `2.0e-07`.
- The eight Mamba-1 blocks, the final norm, the pool and the head (A1b) are
  not built; the gate prints that and the batch row stays `not run`.

### Mamba-3 tiny-LM: the corpus subset and its read-back are gated

- `reference/mamba3_tiny_data.py` builds a deterministic tiny subset of the
  lab's byte corpus: the first 64 train examples (8,192 uint8 tokens) and
  the corpus's own held-out split (4,096 tokens), vocab 127, with sha256s in
  `.scratch/mamba3_tiny/manifest.json`.
- `make mamba3-tiny-data` verifies the hashes, the split identity and that
  Bend reads both files back with the same first sixteen tokens:
  `MAMBA3 TINY-LM DATA PASS`. The trainer (next-token loss, AdamW + weight
  decay, clipping, LR schedule, held-out eval, report card, twin) is not
  built yet; the batch row stays `not run`.

### Mamba-3 mimo: blocked on its oracle (recorded, no gate)

- The reference MIMO forward is the TileLang kernel `mamba3_mimo_combined`,
  which is not vendored, and `tilelang` is not installed, so the reference
  cannot run the MIMO config on this lane. Only the rotary Q/K step
  (`mamba3_mimo_rotary_step.py`) is vendored, which does not cover the rank
  projections, the headwise norm or the SSM update. Per the goal's rule the
  row ships no self-referential gate: it stays `not run` in the batch with
  that reason, and the ROADMAP records what would unblock it.

### Mamba-3 deep: depth-4 stack forward, backward and generation

- `lm4_forward`, `lm4_step` and `lm4_gen` (`L4W`/`LM4` records): four
  independent blocks with a residual around each, the final RMSNorm and the
  head; the backward chain composes four `backward` calls with the
  residual-sum propagation and per-layer SGD.
- `make mamba3-deep` matches `reference/mamba3_deep_twin.py` at the prompt
  (logits rel `2.6e-07`, gate `1e-5`), at the ten-step training run
  (rel `1.7e-07`, gate `1e-4`) and at sixteen-token greedy generation
  (exactly equal). Build 9.6 s, run 34.4 s; `MAMBA3 DEEP PASS`.
- Depths 8 and 16 are `unavailable` (not built) and the gate says so.

### Mamba-3 outnorm: is_outproj_norm gated grouped RMSNorm, forward and backward

- `forward_on`/`backward_on` in `mamba3_siso.bend`: the kernel output stays
  ungated, then `rms` per 32-element group × `norm_w` × `silu(z)`, then
  out_proj; the backward runs the grouped RMS chain into out_v, keeps the
  norm weight's gradient in `dW_on` (own `BWON`/`GRADON` accumulators, the
  shared records unchanged).
- `make mamba3-outnorm` matches `reference/mamba3_outnorm_twin.py` at every
  forward intermediate (rel `<= 1.1e-06`, gate `1e-5`) and every gradient
  (rel `<= 2.9e-06`, gate `1e-4`), `dW_on` included. `MAMBA3 OUTNORM PASS`.

### Mamba-3 varlen: the packed input resets at the boundary

- `forward_vl` in `mamba3_siso.bend` runs the row-independent `pass_a` once
  over a packed `[A ; B]` input and each segment's angle/SSM passes on its
  own slice, with the segment's own zero-supplemented DT/trap row — a true
  reset of the angle state, the SSM array and the trapezoid's shifted term.
  `make mamba3-varlen` (`probe_mamba3_varlen.bend`, 16 packed rows, boundary
  at 8) reproduces the twin-gated 8-row golden in both segments: y rel
  `5.5e-07`, angle `1.5e-07`, states `5.0e-07`, gate `1e-5`; build 2.6 s,
  run 0.10 s. `PROBE_MAMBA3`'s 8-row golden is unchanged, so `make mamba3`
  and the other gates are untouched.

### Mamba-3 decode step: implemented and gated at 16/64/256 tokens

- `examples/language/mamba3/step_mamba3.bend` is a real online decode step for
  the SISO LM: one row per token, state threaded unmatched through a
  single self-recursive `step_go` (`SR` state: two layer states with the
  2048-float SSM `Array`, the 16-float angle block and a one-row
  pending `x`/`k`/`DT`/`trap`). The trapezoid's dependence of row `t`'s
  `scale` on row `t+1`'s `DT`/`trap` is handled by folding every row with
  `scale = 0` and applying an additive rank-one SSM correction
  `S[v, q] += (scale * x_v) * k_q` when the next row's `DT`/`trap` arrive,
  in the same float op order as the reference pass — exact parity at 16
  tokens, not an approximation.
- `make mamba3-step` runs `reference/test_mamba3_step.py`: the sixteen
  greedy tokens equal the golden `[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1]` exactly (build 2.2 s, run 0.08 s). 64- and 256-token parity
  were then verified against the compiled full-prefix binaries — 64 exact
  (step 0.45 s, reference 17.7 s) and 256 exact (step 1.2 s, reference
  29 m 17 s); `bend file.bend` dies in the interpreter at seq len >= 64,
  the compiled path works at both lengths.
- `make check` was re-run after the step landed: 143/143 PASS at that
  moment (37 m 48 s, user 92 m 54 s); after
  `reference/test_mamba3_train_curve.py` landed the gate reports 144/144,
  its row auto-included; the fresh batch flagged my stale 143/143 pin and
  the pin was corrected to 144/144.
- The earlier "decode placeholder" claim (identical to the probe, reported
  as PASS) is superseded by this; the placeholder was deleted and the row
  had been recorded as not run. See `results/mamba3-step-gate.md` and the
  ISSUE-127 entry in the local ledger.
- `make mamba3-train-curve` (ISSUE-126 record): `reference/test_mamba3_train_curve.py`
  writes `results/mamba3-train-curve.md` — the custom trainer's ten losses
  (`2.1568611 → 2.1401443`), the layer's final loss (`2.1404066`, rel `1.23e-04`,
  gate `1e-3`), tokens equal, heads ≤ `8.9e-07`; the per-step layer losses remain
  unreadable (`bendtorch#86`).
- The Stage-C learning canary was re-run on this lane (16 labels, lr 0.03,
  50 steps): bend `final_loss` 0.04192596 (twin 0.0419212505),
  `grad_loss` 0.03683044 (twin 0.0368300416), weights rel 5.5e-05, grads
  rel 1.2e-03; record `results/glm53-c-canary-repro-20261113.md`. The
  before-RL 0/24 claim remains unreproduced (no checkpoints in the tree);
  the third-party published README was not edited.
- `docs/bend-2.0.38-pitfalls.md` records the ISSUE-120..128 frictions hit
  on this port; `docs/agents/oss-needs.md` records what Base does and does
  not ship (no scan/cumsum anywhere; softmax/CE/argmax live in
  tensor-array and bendtorch, not in Base). Three language asks are filed as
  `bendlang/bend#2146` (records), `bendlang/bend#2147` (forward
  references and mutual recursion) and `bendlang/bend#2150` (encoding for
  evolving state across an online loop) — diagnostics and repros only, no
  code pushed. `aaaxn/bendtorch#86` asks whether the last-step loss is meant
  to be all of `T.Train`.


### Mamba-3 decode placeholder: full-prefix twin parity passes

> Superseded by "Mamba-3 decode step: implemented and gated at 16 tokens"
> (2026-11-13): the placeholder was deleted, and this section is kept as
> history only.

- `make mamba3-step` runs `reference/test_mamba3_step.py` against
  `examples/language/mamba3/step_mamba3.bend` (identical to the probe's
  full-prefix generator) and compares the generated tokens to the twin's
  16-token golden: PASS.
- The proper per-token step decoder (angle/SSM/K/V state carried across
  calls) is not yet implemented in Bend — it remains the next step.
- `scripts/batch.sh` now runs this as the `mamba3-step` row.

### Mamba-3's SISO block is wired into BendTorch `T.Layer`/`T.Train`

- `make mamba3-layer` builds `examples/language/mamba3/mamba3_layer.bend` and
  runs `reference/test_mamba3_layer.py`: the block registered as a `T.Layer`
  (hand-written backward, `Fwd`/`Bwd`/`Step`, zeros gradient, SGD-only
  `step`) is trained ten `T.Train` steps on the copy task and compared
  against the custom ten-step trainer on the same weights: tokens exactly
  (`[3, 6, 0, 5, 0, 7, 0, 7]`), final loss `2.1404066` against the custom
  `2.1401443` (`1.2e-04` relative, gate `1e-3`), weight heads `<= 1.3e-05`
  relative, wall `26.8 s`.
- Bend did not make that wiring easy. See ISSUE-120..126 in the local ledger:
  matches cannot scrutinize computed values (defs must take them as
  parameters), multi-scrutinee matches need the full cross product, decreasing
  self-calls read arguments left to right, shared (`+`) bindings need the
  `Data` sort, tuple types are `A & B` with the `Tuple{...}` constructor, `IO`
  binds require typed `x : T <- ...` lines and explicit `IO.pure(T, x)`, and
  `T.Train`'s per-step loss list is not readable from outside the loop. The
  per-step losses therefore cannot be compared: the gate checks the final
  step's loss and the tokens.
- `make mamba3` still PASS. `scripts/batch.sh` now runs this as the
  `mamba3-layer` row.

### One batch driver for the matrix

- `scripts/batch.sh` (and `make batch`) runs every row through `make session`
  and writes `results/batch-<date>.md`: status, seconds, sample count and the
  first failing line. A missing sample count is `unavailable`. The `mamba3-layer`
  row is included: `MAMBA3 LAYER PASS`, 2026-11-13.
- The first full run is 14/14 PASS: Mamba-3 gates, the GPU probe, versions,
  `check-fast` 31/31, the full gate 141/141 in 2460 s, the UNet gate, the GLM
  decoder and 192 evaluation, and the four short tracks. The Mamba-3 decode
  step, `T.Layer`/`T.Train`, varlen, GQA, outnorm, deep, MIMO and the tiny LM
  are recorded `not run` with their reasons.
- The fresh 2026-11-13 batch (`results/batch-20261113-210547.md`,
  `-220714.md`) records `mamba3-step` and `mamba3-train-curve` PASS; the full
  gate reports **144/144** after `test_mamba3_train_curve.py` landed (a stale
  `143/143` pin in the runner honestly flagged the change; re-pinned). Four
  rows of the 220714 batch failed with `ENOSPC`/`EDQUOT` on the shared
  `/tmp` (a concurrent session was flooding it); every failed row passed on
  re-run with the disk free (`results/batch-20261114-190144.md` vit 112 s,
  classical 50 s, glm53 155 s in `batch-20261113-231337.md`;
  `results/batch-20261114-190453.md` full 2890 s). Varlen/GQA/outnorm/deep
  remain `not run` (not implemented); MIMO and the tiny LM remain
  unavailable.

### The hybrid 192 evaluation is complete

- `step_h192.bend` decodes the four-block hybrid one token at a time: the
  embedding, three linear blocks with their prefix state, the sparse block
  with its K/V cache, the final norm and the tied head. `make glm-decode`
  reproduces episode 0's 48 tokens exactly (3.35 s/episode, about 130x the
  full-prefix generator).
- The full evaluation, `make glm-h192-step`: dev **0/64 solved, 0/192 tests,
  0 valid, 64 at the token cap** in **306.997 s**; confirm the same in
  **311.955 s**. The first four dev token lists match the full-prefix
  four-episode run; these are the full 192 results (ISSUE-101).

### The GLM 192 step cache protocol

- `make glm-step` runs `probe_glm_step.bend`: the linear attention's prefix
  state and the sparse attention's post-RoPE K/V cache reproduce the
  full-prefix 192-position rows (`lin max|d| 0`, `sp max|d| 0`). Synthetic
  q/k/v, no weights.
- `make glm-block-step` runs the fourth block on the trained weights: the
  full `SpBlock` on 192 rows against the one-row attention + SwiGLU path with
  the K/V cache is `max|d| 0` (`GLM BLOCK STEP PASS`, `16.1 s`).
- `make glm-lin-block` does the same for the linear block on the trained
  weights, with the prefix state `K`/`KV`: `lin block max|d| 0`
  (`GLM LIN BLOCK PASS`). The four-block decode is next.

### The GLM evaluation can run a named subset

- `gen_a.bend`, `gen_h192.bend` and `reference/glm53_dense_eval.py` take
  `--episodes N`; the output is `eval-<split>-epN.json` and the frozen golden
  is untouched. The first four hybrid dev episodes took `1749.7 s (29.2 min)`,
  solved `0/4`, passed `0/12` tests and hit the token cap on all four.
- This is a named subset, not the 192 evaluation; the full 64 episodes
  extrapolate to about `7.8 h`, over the one-hour rule.

### Mamba-3's scan keeps its state in an array

- `scan_head` copies each head's 512-float state into one `Array<F32>`, updates
  the 32x16 cells in place and reads it back, instead of rebuilding 32 rows per
  token. Twenty forwards at seqlen 8 are `0.37-0.38 s` against `0.71-0.73 s`;
  the ten-step train probe is `3.64-3.67 s` against `9.81-10.50 s`.
- The scan now keeps one 2048-float `Array<F32>` across all four heads and all
  tokens (`scan_rows_st`, `sc_go_st`). In a same-session, interleaved A/B (not
  comparable with the per-head figures above, which are from an earlier state),
  twenty forwards are `0.41-0.42 s` against `0.53-0.55 s` for the per-head
  array core (medians `0.41 s` against `0.54 s`; a second session gave
  `0.37-0.45` against `0.49-0.56`), and the train probe is equal within noise; `make mamba3` passes
  forward, backward, training, generation and the full LM.
- `make mamba3-bench` builds and runs the 20-forward benchmark and prints its
  checksum and wall time (`19.323364`, `0.44 s` in the recorded session). The
  list loop stays in the tree as the reference (ISSUE-108).

### The UNet has a held-out gate

- `make unet-metric` runs `reference/test_unet_flowers_metric.py` on the
  recorded 64-draw run: held-out legibility `0.547`, floor `0.058`, ceiling
  `0.574`, span `94.9%`, spectrum `0.002 / 0.001 / 0.000 / 0.034`.
- The gate fails if held-out legibility is below `0.40`, below 60% of the
  span, above the ceiling by more than `0.02`, or the low/high spectrum bands
  are white. White noise (`0.041`, high band `1.002`) and the data mean
  (`0.000`, span `-11.3%`) are the can-fail arms; both fail. The training loss
  stays report-only (ISSUE-066).

### One version lane

- The tree now imports BendTorch `2.1.0.0`, BendTorch GPU `2.1.0.0`,
  tensor-array `0.1.20.0` and tensor-gpu `0.1.5.2`; `bend.toml`, `bend.lock`,
  `.bend/env` and every `.bend` import agree. Bend stays at the installed
  `2.0.38` pin.
- The latest tensor-array `0.1.21.0` and tensor-gpu `0.1.5.3` are **not**
  compatible with BendTorch 2.1.0.0: the `Mat` types are nominally different
  even at the same shape. The compatible lane is BendTorch's own closure; the
  unused tensor-fork wildcard was removed.
- The full gate passes `141/141` in `34m26 s` and `make check-fast` `31/31`;
  `make mamba3`, `flow64-first`, `vit-first`, `glm53-hybrid128` and
  `iris-crossval` pass on the lane. `bendenv sync` was purged first because it
  keeps old package directories (ISSUE-119).

### The dashboard is the in-house W&B

- `reference/dashboard.py` serves one page for every run in the store. The
  overview shows live sessions, host CPU, RAM, GPU utilization and GPU memory,
  and a table with task, kind, status, PID, steps, loss, parameter count,
  per-run CPU/RAM/GPU and loop time. A selected run opens loss, step time,
  extra signals, CPU/RAM/GPU samples, configuration, recent steps and its
  record.
- The sampler walks each run's process tree, so several sessions can be live
  at once and a Python test that spawns Bend is sampled as one run. GPU
  numbers come from `nvidia-smi`; without it the page prints `unavailable`.
- The sessions table scrolls with a sticky header; `running` is green,
  `failed` red, `lost` amber. Charts draw a marker and a level for a single
  point, floor the y axis at zero, label both axes, and the loss has `train`,
  `eval` and `log` toggles. CPU, RAM, GPU utilization and per-process GPU
  memory are separate charts. A finished run shows its peak sample; a run with
  no steps or samples hides those cards. `?run=ID` opens a run directly, and
  the sessions table has a filter for task, kind or status.
- CPU is a run's **share of the host** with its core count (`27% · 7.4c`), not
  a sum of per-core percentages; RAM is PSS over the process tree, not summed
  RSS; the GPU axis is capped at 100. The page is named **Bend-Lab**, uses
  the Bend cream/charcoal/purple theme, and its sun/moon button switches
  between light and dark (saved in the browser; `?theme=dark` for a link).
  The README embeds `docs/assets/dashboard.png`.
- The GLM trace writes a final held-out `val_loss`; `trace.py` stores it as
  `runs.eval_loss` and the loss chart shows it at the last step beside train.
  `make trace TASK=glm53-hybrid128 STEPS=10` measured train `4.5521`, eval
  `4.4326`; the 30-step repeats measured train `3.2657`, eval `3.2838`, with
  PSS max `1507 MB` after the sampler change.
- `./scripts/dashboard.sh start` leaves the server running under
  setsid+nohup (`status` and `stop` too); `make dashboard` stays foreground.
- `make session TASK=<name> CMD="..."` wraps any test command in the store
  (`reference/session.py`), with its log under `results/<task>/runs/`.
  `experiment.py` registers a run before the build and finishes it after;
  `trace.py` registers the trace process and swaps to the child PID.
- The store has `pid`, `command`, `config`, `params`, `kind`, `host`,
  `started`, `finished`, `exit_code`, `error`, `git_commit`, `versions` and a
  `samples` table. `runs.reconcile()` marks a run whose process is gone as
  `lost`; `runs.backfill()` fills params/config on older rows.
- The full gate (`make check`) passed `140/140` in `49m35 s` through a
  dashboard session, with `2,811` system samples and a `5.6 GB` peak RSS.
  `reference/gpu_probe.py` is the CUDA probe: a 15 s loop shows `52-100%` GPU
  utilization and a `326 MB` per-process GPU memory line.

### The run store, and ISSUE-058 / ISSUE-072

- `reference/runs.py` keeps every run in `results/runs.sqlite3`: one row per
  run, one per step, with the run's own trace under
  `results/<task>/runs/`. `reference/trace.py` writes it live and the
  dashboard selects any run by id, in a selector and a clickable table, so a
  later run no longer replaces an earlier one in the page. The database and the per-run traces are ignored by
  git; the latest single-file trace can be imported. A finished
  `reference/experiment.py` report card is stored as a run with its record and
  no steps, and the page marks it `record only`. On an empty database the
  dashboard imports every latest record and the task's trace;
  `reference/runs.py --import-latest` imports the records by hand, and
  `MLBEND_RUNS_DB` selects another database for a test.
- `trace.py` now keeps every numeric field in a line, not only `step`,
  `tokens`, `loss` and `ms`, so `lr`, `grad_norm`, `state_norm` and
  `gate_mean` reach the extra-signal charts.
- The ViT example takes `--steps`, `--start`, `--save`, `--resume` and
  `--no-samples` (ISSUE-058). A 2-step checkpoint restores for one more step
  to the fresh 3-step loss, `0.96344185`; `reference/test_vit_checkpoint.py`
  runs that gate and the zero-step draw in about two minutes.
- The flow examples print `final_loss: unavailable` on a `--steps 0` resume
  instead of a zero (ISSUE-072): the checkpoint does not carry the loss.

### The Mat `q·S^T` product does not transfer to the scan

- Replacing the 32 list `dot` calls in Mamba-3's `scan_head` with one
  `TA.Mat.matmul_nt(1, 16, 32, ...)` kept the checksum and passed every gate,
  but twenty forwards took `0.84-1.18 s` against `0.71-0.73 s` for the list
  path. The row extraction is still needed for the state update, so the
  isolated 65x is not the scan's cost. The change was reverted; the record is
  in `examples/language/mamba3/COMPARISON.md` and `ROADMAP.md`.

### A browser dashboard, and a one-hour training budget

- `reference/dashboard.py` serves a training trace and record as HTML on a
  local port (`make dashboard PORT=8787`): loss, step time, tokens/s,
  resident and peak memory, the stage split, recent steps, and any extra
  numeric fields in a trace line as small charts. The terminal
  `reference/dash.py` remains; this is the page an agent run can leave open.
- `reference/trace.py` stops a run after one hour by default
  (`--max-seconds 3600`; `MAX_SECONDS=0` disables it). The existing
  `glm53-hybrid128` trace is `3733.97 s` (62 min), over the budget; the page
  shows it in red.
- `make dashboard` joins `make trace` and `make dash`; the server is standard
  library only and polls `/api/state`.

### The five-track review starts with the version lane

- `docs/agents/review.md` records the matrix for classical ML, diffusion,
  flow matching, GLM-5.3-Flash and Mamba-3, with a short gate and a short test
  for each.
- The first finding is version skew: `bend.toml` pins BendTorch `1.1.1.0` and
  tensor-array `0.1.14.0`, `.bend/env` and the examples use `2.0.0.0` and
  `0.1.18.0`, and upstream is `2.1.0` and `0.1.21.0`; Bend is `2.0.38` against
  `2.0.40`. The new features are therefore not the versions the gate imports.
- `make check-fast` is `31/31` on the current env; the review starts by
  agreeing the lane, adding `make versions`, and re-running the short tests.
  The 6,000-step flow and the 5.6-hour GLM 192 run are not part of the first
  pass.

### Mamba-3 SISO: the reference block, forward and backward, in Bend

- **Vendored** `state-spaces/mamba` at `e9594ce1c732d97440f0332fdc43170a2294dbfa`
  (Apache-2.0): the SISO module, the combined/fwd/bwd Triton kernels, the angle
  cumsum, `layernorm_gated.py`, and the generation and varlen tests. The tree is
  importable with `PYTHONPATH=reference/mamba3`.
- **A pure-PyTorch twin** of the kernel math, as an explicit loop over `t`
  (`reference/mamba3_twin.py`). The public module is a bf16 lane (`2.5e-03`
  relative against the twin); the forward kernel called directly with float32
  and IEEE dots agrees to `2.0e-07` on `out_v` and `4.0e-07` on the state.
- **The Bend block** (`examples/language/mamba3/`): the recurrent scan over
  `List<&2,F32>`, gated RMSNorm, the `DT`-scaled angle cumsum, the trapezoid
  state, the `D + gamma*(q.k)` skip, the SiLU gate and the projections.
- **Gates, all against the F32 twin**: forward at seqlen 8 max relative
  `9.2e-07` (gate `1e-05`), at seqlen 4 and 16 `7.4e-07` and `9.6e-07`;
  backward on every parameter and retained tensor max relative `2.9e-06`
  (gate `1e-04`); the moved-`dy` can-fail matches its own golden and changes
  `du`.
- **Ten-step copy task**: loss max relative `1.1e-07`, final tokens
  `[3, 6, 0, 5, 0, 7, 0, 7]`, 80 tokens. The gate's `bend` process took
  `4.20 s` in one run (`4.2` to `8.1 s` observed under load) and `354264 KiB`
  including the compiler and checker; the native C binary of the same probe
  takes `5.77 s` and `6.0 MB`, and the PyTorch CPU twin `0.048 s`.
- **Sixteen-token generation**, two weight-tied layers: exact
  `[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`; `1.17 s` native,
  `0.039 s` PyTorch CPU twin.
- **The comparison** (`examples/language/mamba3/COMPARISON.md`, from
  `reference/mamba3_compare.py`): Bend is `25x` the same-lane CPU twin on the
  forward and `99x` on forward+backward; the public bf16 Triton module is
  `0.596 ms` and `2.5e-03` relative, the fastest and least accurate lane. At
  this shape the CUDA twin is launch-bound and slower than the CPU twin.
- **The full two-layer LM** (`probe_mamba3_lm.bend`,
  `reference/mamba3_lm_twin.py`): two independent blocks with a residual
  around each, a final RMSNorm and an untied head, `56,720` parameters. The
  forward `lm_hn` is `1.03e-07` relative, the ten-step loss `1.77e-07`, and
  both the final tokens `[4, 2, 6, 3, 3, 5, 0, 4]` and the generation
  `[0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]` are exact. The native
  probe is `17.68 s` against `0.145 s` for the PyTorch CPU twin's training
  plus generation.
- **The BendTorch GPU lane is real.** `probe_bendtorch_gpu.bend` runs a
  `LinearNBF(8, 64, 304)` forward and backward once on the CPU and once on the
  GPU: **0 differing words** in `y`, `dX` and `dW`; the process is visible in
  `nvidia-smi`. At that size the GPU is slower (`0.28-0.36 s` against under
  `0.01 s`), which is launch-bound. Mamba-3's scan is not on the lane yet:
  BendTorch has no selective-scan layer, so the block stays CPU until the scan
  becomes a chunked GPU bang. `examples/language/mamba3/GPU.md` records the
  build and the caveat. The upstream branches were fetched at `65c8038`
  (BendTorch devel) and `8af6052` (bend-ml).
- **The bend-ml Mat product is 65x the list product.** The `q.S^T` product
  of one Mamba-3 head, 20,000 times: `2.59 s` with `take_row` + `dot`,
  `0.04 s` with `TA.Mat.matmul_nt(1,16,32, ...)`, same output to about
  `5e-8` per product. This is the substrate lever; it is a component, not yet
  the full scan. `bench_qs_list.bend` and `bench_qs_mat.bend`.
- **The roadmap** (`examples/language/mamba3/ROADMAP.md`): array core and
  chunked scan, step decode, batch/varlen/GQA, a real tiny LM, MIMO, and the
  numerical and ablation questions, in that order.
- The gates run from `reference/check_mamba3_bend.py`, and `make check` calls
  them through `reference/check_all.py`; the bins come from the committed
  `reference/mamba3_golden.npz` with NumPy only. `make mamba3` is the same
  gate on its own.

### The hybrid step is 1 s, and the lab has a trace and a dashboard

- **`--trace` on `train_h128.bend`**: one JSON line per iteration,
  `{"step": 1, "tokens": 128, "loss": 5.5634737, "ms": 1018}`, with the batch read and
  `T.Train.step` under one timer. A traced and an untraced run print the same loss, and two
  traced runs agree within 2%.
- **The 18.97 s smoke was the loop, not the sparse core.** `step_res(step(...))` matched the pure
  `Train` value at every field projection and rebuilt it for the return, so one logical step ran
  about eighteen times. Binding it once with `IO.pure` gives **0.99 s** with `--no-val --no-save`
  and **1.45 s** with validation and the weight save. Three steps print identical losses in both
  builds, and the canary's weight sha256 `1856080434…c71c3ce` is unchanged.
- The 1,638,400-token budget is therefore **12,800 steps / about 3.5 hours** at the measured step
  (an extrapolation, not a run), not 67.4 hours. The vectorization target comes from the stage
  split.
- **The stage split** (`bench/plumbing_glm53/main.bend`, the `glm53` tree in
  `reference/stages.py`): load, forward, loss, backward and update at the real shape. One step
  **1135 ms**: forward **514 ms (45.0%)**, backward **729 ms (63.8%)**, loss and update below
  **1 ms** each. The parts sum to **1243 ms (+9.5%)**, timed on a fresh state; the page prints
  `unaccounted in step -108.0 ms` rather than hiding the gap.
- **Checkpoints**: `train_h128 --save-every N` writes `out/step_<i>.bin` every N steps in the
  same flat layout as the final save; a four-step run proves `step_4.bin` is byte-identical to
  `final_weights.bin`. A 12,800-step run can now be interrupted without losing the model.
- **The task and the record**: `glm53-hybrid128` in `TASKS` runs one step with
  `--no-val --no-save` and writes schema 3 with `variant.batch` and `variant.positions`.
  `results/glm53-hybrid128/run.json` replaces the hand-written `smoke.json` and carries the
  forward check and the projection as notes; `make report TASK=glm53-hybrid128` prints the split
  and `PASS`.
- **`make trace` and `make dash`**: `reference/trace.py` streams the program's JSON lines to
  `results/<task>/trace.jsonl` as they arrive, and `reference/dash.py` renders the live page: the
  loss and step sparklines, tokens/s, the record's stage table, and with `PID` the `/proc` RSS,
  peak and CPU. Standard library only, and it decides nothing; the trace is ignored by git.

### RLOO, implemented from the reference's description

- **`reference/glm53_rloo.py`**: 96 prompts across `increment`, `double` and `even` (the `rl`
  split, seed 3907), 16 sampled completions each, reward **1 / 0 / -0.1**, leave-one-out
  advantages, one AdamW update per prompt, 96 updates from checkpoint 100. The reference's RL code
  is not in the vendored tree, so the hyperparameters are inferred from its report.
- **It learns, then collapses.** At lr **3e-6**: confirm three trained families **3/24 -> 12/24**,
  all-eight greedy **15/64 -> 21/64**, pass@8 at T=0.8 **10/64 -> 13/64**. The dev curve peaks at
  **15/24** (step 80) and falls to 12/24; at lr 1e-5 it collapses by step 32 (confirm 11/24 at
  step 16, 9/24 at 96).
- **The published magnitude is not reproduced**: the report's 0/24 -> 16/24 and 25/64 -> 34/64
  need a stable run and a sampling temperature the summary does not state. At the policy's own
  temperature (0.35) pass@8 is **30/64** after RL against the baseline's **33/64**, so the RL gain
  is in greedy trained-family decoding, not in sampled pass@8.
  Record: `results/glm53-reference/rloo-summary.json`.

### The reference itself, on this machine

- **Pretraining reproduced**: 400 steps, 1,381,760 tokens, **78.758 s**, **2.143 GiB** peak on an
  RTX 5060 Laptop (the reference's README reports 63.1 s and 2.139 GiB on a 3080 Ti, a lane
  banner). `results/glm53-reference/run.json` carries the run and both checkpoint evaluations.
- **Checkpoint 400 solves the benchmark**: **24/24** on the three trained families and **64/64**
  on all eight (dev **63/64**). Checkpoint 100 is **3/24** and **15/64**; the README's before-RL
  0/24 is that earlier checkpoint and does not reproduce exactly (ISSUE-102).
- **The held-out family is not learned at either scale.** With `reverse` truly removed from
  pretraining (`family.name`, not the object), checkpoint 400 scores **56/64** and **0/8 reverse**
  on confirm and dev; checkpoint 200 is also **0/8**. The 74,144-parameter port has the same
  pattern: **56/64** and **0/8**. A sweep at **1.9M, 4.3M and 7.6M** parameters (dim 64/96/128,
  eight layers) is **0/8** too, so pretraining does not learn the family at any scale tested. The
  first attempt filtered `Family` objects against the string `"reverse"`, kept the family, and its
  8/8 was invalid (ISSUE-103); the docs here are corrected.

### The hybrid trains, and its evaluation is the next wall

- **The partial run**: 1,398 steps (178,944 tokens) at 128 positions, loss **5.5634737 -> 0.3845**
  at step 1,000 and **0.3953** at the end, median step **1,917 ms**
  (`results/glm53-hybrid128/training-partial.json`). It is stopped, not a baseline: the
  checkpoint is `run/step_1000.bin`.
- **The 192 wall**: a hybrid forward is **1.22 s** at 128 positions and **6.48 s** at 192, so a
  64-episode split is **5.6 h**, and the frozen prompts do not fit 128 (`dev-absolute-000` needs
  prompt+48 > 128). The first hybrid evaluation waits for the vectorized attention (ISSUE-101);
  the model is about **200x** the dense model per token.
- **The evaluation path**: `model_h192`, `gen_h192`, `gen_h128`, `gen_a128` and
  `reference/glm53_hybrid_data.py` (74,144 values, checked against a checkpoint). The evaluator
  takes `--positions` and `--layout`, and the eval prompts are no longer always 192 wide; the
  dense 409,600-token model still scores **61/64 dev, 60/64 confirm** at 192, so the path did not
  regress. Generators emit 64 episodes, so `--per-family` is refused unless it is 8.

### GLM-5.3-Flash: the 128 gate, the canary and the evaluation-leak tests

- **128 full-model gradient** (`test_glm53_h_bwd128.py`, 34 checks): one masked-CE backward
  through `model_h.bend`, all 74,144 gradients per tensor. Worst tensor **b3/ln2 at 4.837e-07**;
  loss **5.5634737** against torch **5.563475**; a moved label is rejected at **1.534e-01** and
  matches its own golden at **1.672e-07**. Two identical probe runs have the same gradient
  sha256, so the 128 path is deterministic.
- **Bit-identical step canary** (`test_glm53_canary.py`, 4 checks): two `train_h128 --steps 1
  --no-val` runs print **5.5634737** and the same final-weight sha256
  `1856080434…c71c3ce`; a corpus copy with one input token changed prints **5.5687194** and a
  different hash. This is the gate that catches a row-order or scale bug a tolerance hides.
- **Four-head 128 forward/backward** (`test_glm53_attn_bwd128.py`, 12 checks): the wrapper at
  128 against the vendored modules; the largest relative error is **4.389e-07**, and the moved
  dout is rejected at **3.676e-01**.
- **Mask golden** (`test_glm53_sparse_mask.py`, 3 checks): 16/16 and 32/32 match `_indices` in
  **0** of 4,096 cells; the 8/8 can-fail differs in **436** cells.
- **Evaluation leakage** (`test_glm53_eval_leak.py`, 6 checks): the holdout stream contains
  **0 of 263** reverse tasks, the full stream contains them, dev and confirm each have **8/8**
  reverse tasks, and the frozen manifest records **0/8** on both.
- **Twenty-step drift** (`test_glm53_train20.py`, 33 checks): 20 AdamW steps at 128 against
  the torch twin. Final loss **3.7224956** vs **3.7224958**; the worst final weight is
  `b0/qkv` at **5.673e-05** relative (gate **1e-03**); the lr 0.004 can-fail is rejected at
  **4.937e-02**. A single step did not show this.
- **Two-sequence isolation** (`test_glm53_iso2.py`, 3 checks): a 2 x 128 model with a
  block-diagonal mask. Changing sequence 0 leaves sequence 1's logits at **0** and its parameter
  gradient at **0**, while sequence 0's logits move by **0.53063375**. The first gradient version
  leaked **13.299267** because the denominator term was not masked; fixing it brought the leak to
  zero.
- **4096-row limit** (`probe_mask4096.bend`): one block mask materialises **16,777,216** cells in
  **159.60 s** and peaks at **589,356 KiB**. Batch 32 is not practical with this builder; step 4
  stops at the measured limit.
- **`make check` is 137/137** with all seven new files, including `test_glm53_iso2.py`. A
  shared `/tmp` probe cache between the N=8 and N=128 tests made the first run 135/137; the 128
  caches now have their own names and the full check passes.

### GLM-5.3-Flash: the linear core is dense, and the model step is 1.5 s

- **`lin_dense.bend`** replaces the per-position list prefix with dense causal matmuls:
  `q~ = ELU(q)+1`, `k~ = ELU(k)+1`, `S = q~k~^T` masked to `s <= t`, `den` the row sums,
  `num = S V`, `y = num / clamp(den, 1e-6)`; backward `dnum`, `dden`, `dS`, `dq~`, `dk~`,
  `dV`, then the ELU derivative. The list prefix in `lin_fast`/`lin_bwd` stays as the oracle.
- **The gate** (`reference/test_glm53_lin_dense.py`, 5 checks, in `make check`): y/dq/dk/dv
  at 128 positions are **7.451e-08 / 4.718e-08 / 1.192e-07 / 3.576e-07** relative (gate
  **1e-6**); removing the causal mask moves y by **8.934e-01**, so the mask is load-bearing.
- **The model step**: `train_h128 --steps 1 --no-val`, median of three, **1.12 s** (from
  **24.83 s**, **22.2x**), peak RSS **16,956 KiB**; the gate is **1.24 s**, so it passes.
  With the validation pass and weight save (`--steps 1`), the run is **1.49 s** (16.7x);
  `--no-val` is the option that isolates the step. The 1.6M-token budget at 1.12 s is
  12,800 steps / **3.98 h**.
- **`make check`** is **130/130** with the dense linear core in the model, including
  `test_glm53_lin_dense.py` (5 checks), `test_glm53_sp_dense.py` (8),
  `test_glm53_attn_bwd.py` (12) and `test_glm53_h_bwd.py` (33).
- **The four-head wrapper's builders** now cons and descend in index order instead of appending
  and reversing; a flat `List.reverse` had reversed the numbers inside each row. The loss is
  bit-identical: **5.5634737**, val **5.4198546**, as before.

### GLM-5.3-Flash: the sparse core is dense, batched and 65x faster

- **`sp_dense.bend`** replaces the per-query `List.set` scatter with a dense T x T score matrix,
  the reference's window-16 / stride-16 causal mask, one row softmax, and four matmuls for the
  forward and every gradient. The gathered list core in `attn_c.bend` stays as the oracle.
- **The gate** (`reference/test_glm53_sp_dense.py`, 8 checks) runs both cores at 32 and 128
  positions on the same q, k, v and dy. The largest differences are **8.941e-08** (y), 1.192e-07
  to 8.941e-08 (dq), 1.192e-07 to 3.576e-07 (dk) and 3.576e-07 to 5.960e-07 (dv), against a
  **1e-06** gate.
- **The sparse core's own forward and backward**: 32 and 128 positions, four heads, one pass
  each, `probe_time_cores.bend` on `cc2a7ca` and on this tree, median of three:
  **1,976 + 17,136 = 19,112 ms** old against **122 + 172 = 294 ms** new, a **65.0x** speedup.
- **The full model step is 7.0x, not 20x.** `train_h128 --steps 1` (the same command, old binary
  from `cc2a7ca` against the new one): **24.83 s -> 3.56 s** (median of three), peak RSS
  **15,940 -> 15,500 KiB**. The remaining time is the linear core and the per-head gather/rope/
  join, not the sparse core. Read the step-3 gate as the sparse core's own step; the model step
  is reported honestly at 7.0x.
- **`make check`** is **129/129** with the vectorized core in the model, including
  `test_glm53_attn_bwd.py` (12), `test_glm53_h_bwd.py` (33) and `test_glm53_sp_dense.py` (8).
- **Two bugs the forward comparison caught, both invisible to the gradients.** The first mask
  row omitted s = t, which moved y by 0.30 at 4 positions and 0.48 at 128 while every gradient
  stayed right; `y_of` also forgot to scale q by hd^-1/2, which changed y while the backward
  (which does scale) matched. This is why the gate checks y and every gradient.

### GLM-5.3-Flash: the four-block hybrid's full gradient is gated at N=8

- **`probe_model_bwd.bend`** runs one masked-cross-entropy backward through the whole
  74,144-parameter hybrid at N=8 and writes every parameter gradient in the parameter tree's
  dump order.
- **The gate** (`reference/test_glm53_h_bwd.py`, 33 checks) compares the 74,144 gradients per
  tensor against PyTorch autograd on the vendored modules (`reference/glm53_h_bwd.py`, golden
  `reference/glm53_h_bwd_golden.npz`). The loss is **5.5453690** against **5.5453691**
  (**2.673e-08** relative); the worst tensor is `b2/down` at **6.761e-07** relative, and the
  largest whole-tensor difference is **1.788e-07**. The moved-label gradient is rejected at
  **1.286** relative and matches its own bad-label golden at **3.701e-07**.
- **`make check`** gains the gate (one test file, 33 internal checks).

### GLM-5.3-Flash: the four-head attention backward is gated at N=8

- **`probe_attn_bwd.bend`** runs the whole `LinAttention` and `SpAttention` layer at N=8:
  the qkv projection, the split, RoPE, the four-head core and the out projection. It
  prints `y`, `dx`, the qkv weight gradient and the out weight gradient for each core,
  and the same for a deliberately moved `dout`.
- **The gate** (`reference/test_glm53_attn_bwd.py`, 12 checks) compares all eight tensors
  against PyTorch autograd on the vendored modules
  (`reference/glm53_attn_bwd.py`, golden `reference/glm53_attn_bwd_golden.npz`). The
  largest relative error is **3.231e-07** (sparse/dqkv); the linear ones are
  **1.58e-07 .. 3.17e-07**. The moved-`dout` gradients are rejected at **3.212e-01**,
  **1.763e-01** and **4.601e-01** relative, and match their own golden at **1.992e-07**,
  so the can-fail is the comparison's and not a broken probe.
- **`make check`** gains the gate (one test file, 12 internal checks).

### GLM-5.3-Flash: a trainable dense Stage A baseline, and its 400-step record

- **`examples/language/glm53_flash/train_a.bend`**: the 74,144-parameter Stage A model under
  `T.Train`, AdamW (lr 3e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.1) and the reference's
  masked cross-entropy. The data is the reference's synthetic Python corpus through its byte
  tokenizer: two fixed 64-byte rows, written once and reused, 128 tokens per step.
- **The 20-step gate** (`reference/test_glm53_dense.py`): every step's loss within **5.599e-07**
  relative of the PyTorch-autograd twin (`reference/glm53_dense_twin.py`), the 74,144 final weights
  at **3.455e-07** and the 74,144 final gradients at **2.157e-07** whole-tensor relative (the
  weights/gradients gate is 1e-3 because 20 AdamW steps amplify last-bit rounding). Can-fail runs
  with one weight moved (+2.0) and one input byte changed diverge at step 1 at **3.887e-03** and
  **4.485e-04**, and the same gate rejects them.
- **The 400-step record** (`results/glm53-dense/`): first loss **5.595126**, final **1.724139**,
  last-100 mean **2.0280423**, loss at the final weights **1.716634**, 51,200 tokens, build
  **86.059 s**, run **7.13 s**, peak RSS **57,040 KiB**. The fixed batch makes this an overfit
  baseline, not a corpus pass.
- **Greedy dev/confirm** (`gen_a.bend`, `model_eval.bend` at 192 positions,
  `reference/glm53_dense_eval.py`, the vendored evaluator): **0/64 solved, 0/192 tests** on dev
  (57 token-cap hits) and **0/64 solved, 0/192 tests** on confirm (62 token-cap hits); every family
  is 0/8. The manifest is frozen in `reference/glm53_dense_eval_golden.json` and gated.
- **`make check`** gains the dense gate (one test file, ten internal checks); it is **121/121**.
  The Stage B backward and Stage C remain open, and no upstream issue was filed.

### GLM-5.3-Flash: the 4-block hybrid is implemented, the corpus run is blocked by the sparse core

- **`attn_mh.bend` / `model_h.bend` / `weights_h.bend` / `train_h128.bend`**: the 4-block hybrid,
  three linear blocks and one sparse, width 32, four heads, 128 positions, SwiGLU 128, tied head,
  **74,144 parameters**. The multi-head wrapper gathers each head, runs the gated single-head
  fast linear core or the gated sparse core, and joins the four head outputs; the parameters and
  loader are separate from the dense Stage A tree.
- **The forward check found two bugs before any training conclusion.** A four-head forward at N=8
  against the vendored `LinearAttention`/`SparseAttention` caught an 8-wide/32-wide split (the
  q/k/v chunks are 32 wide, not one head) and a `Lin,Sp,Lin,Lin` block order (the reference is
  sparse on every fourth block; four blocks are `Lin,Lin,Lin,Sp`). After both fixes:
  linear **8.84e-08**, sparse **1.23e-07**, and the full four-block model **1.22e-07** relative
  against the NumPy twin (`reference/test_glm53_mh.py`, 4/4). The one-step training smoke had
  passed both bugs, which is why the forward check matters.
- **`reference/test_glm53_mh.py`** (4 checks) is in `make check`: the two four-head forward
  checks, the can-fail linear-vs-sparse check, and the full four-block model forward. `make
  check` becomes **126/126**.
- **The one-step smoke (fixed model)**: loss **5.5634737**, val **5.4198546**, **18.97 s** for
  128 tokens, peak RSS **16,956 KiB**, build **30.915 s** (`results/glm53-hybrid128/run.json`).
  The 18.97 s and the 67.4-hour projection below were the loop's re-evaluation (see the entry at
  the top of this file); the corrected step is **0.99 s**, and the record is now schema 3.
- **The blocker as measured then**: 128 tokens a step made the dense baseline's 1,638,400-token
  budget a **12,800-step, 67.4-hour** job. The loop fix removed that figure; the stage split now
  puts **63.8%** in the backward and **45.0%** in the forward, with the optimizer below a
  millisecond.
- **Not claimed**: no 192-position accuracy is reported. The next step is a vectorized sparse
  core and a batched multi-head model, before the hybrid can enter the 1.6M-token comparison.

### GLM-5.3-Flash: a held-out family is not learned

- **Train on seven families, evaluate the eighth.** The 74,144-parameter dense model at 128
  positions, batch 32, 1,638,400 tokens, with `reverse` removed from the pretraining stream
  (`corpus_stream_holdout`): loss **5.5683017** -> **0.29021987**, held-out **0.27888998**; run
  **112.0 s**, peak RSS **440,836 KiB**.
- **Result**: dev **56/64** (168/192 tests), confirm **56/64** (168/192); `reverse` is
  **0/8 solved, 0/24 tests** in both splits, and all seven seen families are **8/8, 24/24**.
  The model memorises the seven templates and does not generalise to the unseen one.
- **Why it matters**: the seen-family benchmark is saturated, so MoE/HyperConnection/25.7M can
  only be justified by this kind of headroom or by efficiency. The held-out-family split is now
  a recorded benchmark and a frozen evaluation manifest.
- **Gate**: `reference/test_glm53_corpus.py` now validates the held-out data, both manifests and
  a corrupted copy, 19/19 internal checks; `make check` stays **125/125**.

### GLM-5.3-Flash: the prefix-state linear core

- **`examples/language/glm53_flash/lin_fast.bend`**: the prefix-state causal linear-attention
  core, `K_t/KV_t` forward and `N_s/Q_s` suffix backward. On the same 64 x 8 inputs as the direct
  core, y is bit-identical and dq/dk/dv agree at **6.07e-07 / 4.40e-07 / 2.75e-07** relative;
  the fast-only probe runs in **0.024 s** against the direct core's **11.4 s**. At 128 positions
  y is exact and dq/dk/dv agree at **3.27e-07 / 3.38e-07 / 4.23e-07**; the direct run takes
  **73.8 s**.
- **The integrated cores use it**: `attn_c.bend` now calls the prefix-state linear core, while the
  sparse core is unchanged. The Stage C 20-step gate still matches the PyTorch twin at
  **2.852e-07** on the loss, **3.705e-08** on the final weights and **1.487e-07** on the final
  gradients; the test takes **37 s** instead of 3 m 25 s.
- **The gate** (`reference/test_glm53_bwd.py`, now 28 checks): fast vs direct at 64 at 1e-5, plus
  the direct t=2/t=64 and integrated layer gates. `make check` stays **125/125**.

### GLM-5.3-Flash: the dense baseline is real, 61/64 -> 64/64 with the token budget

- **A corpus-trained dense baseline at 128 positions** (`train_dense128.bend`, `model_128.bend`):
  74,144 parameters, batch 8, 400 steps, **409,600 tokens**. Loss 5.574197 -> **0.32177755**,
  last-100 mean **0.3530558**, held-out **0.30741644**; build **53.786 s**, run **27.28 s**, peak
  RSS **157,436 KiB**. Greedy dev/confirm: **61/64 and 60/64 solved**, **186/192 and 184/192
  tests**, 64/64 valid programs, no cap hits. The three dev failures and four confirm failures are
  all `square` completions that return `x * 2` instead of `x * x`; the other seven families are
  8/8 in both splits.
- **The token budget closes them.** At batch 32, 400 steps, **1,638,400 tokens** (the reference's
  400-step budget): loss 5.5704083 -> **0.29012972**, last-100 **0.3012568**, held-out
  **0.27868035**; build **75.499 s**, run **108.47 s**, peak RSS **445,252 KiB**. Greedy
  dev/confirm: **64/64 and 64/64 solved**, **192/192 and 192/192 tests**, every family 8/8, no cap
  hits. The square confusion was a data/compute effect, not a formatting or capacity defect.
- **The earlier 0/64 was the fixed-batch, out-of-context run**, not a property of the dense model.
  The corpus baseline is the number to beat; the 74k dense model is already at the ceiling of this
  synthetic task at 1.6M tokens. Architecture work now has to be justified by efficiency or a
  harder task, not accuracy on a solved corpus.
- **The gates**: `reference/test_glm53_corpus.py` (14 checks) regenerates both window sets, checks
  next-byte alignment and hashes, reruns the 400-step batch-8 training and requires the loss curve
  to match at 1e-5, validates both dev/confirm manifests against their frozen goldens and the
  reference evaluator, and rejects a corrupted batch at step 1. `make check` is **125/125**.

### GLM-5.3-Flash: the masked-label gate and the training canary

- **The ignored-label branch is now gated** (`reference/test_glm53_mask.py`, 4 checks): a 2 x 3
  logits matrix with row 0 labelled 0 and row 1 ignored. The masked loss is **1.2019429**
  (expected 1.20194285), the ignored row's gradient is **exactly zero**, and the mean uses one
  valid row, not two. The 64-byte training batch has no ignored labels, so this branch was
  previously untested; a padded corpus batch will use it.
- **The learning canary** (`results/glm53-c/canary-16.json`): the Stage C model overfits the
  first 16 labels at lr 0.03 for 50 steps. Bend: **0.04192061** final loss, **0.03682779** at the
  final weights, build **22.869 s**, run **446.4 s**. The PyTorch twin: **0.04192125** and
  **0.03683004**, a relative difference of **1.5e-05 / 6.1e-05**. This proves the Bend loss,
  backward, AdamW and data can drive a loss down, not only match a twin.
- **The learning-rate sweep** (PyTorch twin, same fixed 64-byte batch, 400 steps): lr 3e-4
  **3.955**, 1e-3 **2.236**, 3e-3 **0.175**, 1e-2 **0.00263**. On a 16-label overfit, 200 steps:
  lr 3e-4 **4.651**, 3e-3 **0.723**, 1e-2 **0.00286**, 3e-2 **0.00041**. The 3e-4 rate is the
  reference's 25.7M rate; the 20-step parity gate is not a learning benchmark.
- **`make check`** is **124/124**.

### GLM-5.3-Flash: Stage C, the integrated linear + sparse model at 20 steps

- **`examples/language/glm53_flash/{attn_c,model_c,train_c}.bend`**: one batch, 64 positions, one
  head of width 8, **linear attention then sparse attention** (window 16, stride 16), each with a
  SwiGLU hidden 32, a final RMSNorm and the tied head: **4,168 parameters**. The sparse indices
  are exactly the reference's `s <= t and (s % 16 == 0 or t - s < 16)`; the sparse backward is the
  softmax backward on the gathered scores, with the scatter into dk/dv.
- **The 20-step gate** (`reference/test_glm53_c.py`, 7 checks): every step's loss within
  **2.852e-07** relative of the PyTorch twin (`reference/glm53_c_twin.py`), the 4,168 final
  weights at **3.705e-08** and the 4,168 final gradients at **1.487e-07** whole-tensor relative
  (gate 1e-3). The loss falls 5.5484023 -> 5.4986544. Can-fail runs with one weight moved (+2.0)
  and one input byte changed diverge at step 1 at **7.860e-03** and **6.585e-04**.
- **The layer cores** are also gated directly (`reference/test_glm53_bwd.py`, now 20 checks):
  RoPE + the linear core and RoPE + the sparse core on the same 64 x 8 inputs, against
  PyTorch-autograd goldens, all at **1.5e-07 .. 1.9e-06** relative.
- **`make check`** is **123/123**. Record: `results/glm53-c/run.json`.

### GLM-5.3-Flash: the Stage B linear-attention backward, exact at t=2 and 64

- **`examples/language/glm53_flash/lin_bwd.bend`**: the exact single-head causal linear-attention
  backward in direct O(T²d) form. With `c_t = clamp(den_t, 1e-6)`, `dnum_t = dy_t/c_t`,
  `dden_t = -dy_t·num_t/c_t²` and `a_{t,s} = dnum_t·v_s + dden_t`, the gradients are
  `dq_t = Σ_{s≤t} a_{t,s} k~_s`, `dk_s = Σ_{t≥s} a_{t,s} q~_t` and
  `dv_s = Σ_{t≥s} dnum_t (q~_t·k~_s)`, then the ELU chain.
- **The gate** (`reference/test_glm53_bwd.py`, 12 checks): at t=2, one head, hd=8, y/dq/dk/dv
  match a NumPy twin at **1.19e-07 / 1.49e-06 / 4.52e-07 / 6.47e-08** relative, and a one-hot
  dout at t=0 gives **dq = dk = dv = 0 exactly** for every t>0. At t=64 the 512-value tensors
  match the PyTorch-autograd golden at **1.51e-07 / 1.19e-06 / 1.17e-06 / 2.75e-07**. The
  one-hot case is the can-fail the old 64-step list implementation lacked (local ISSUE-090,
  closed).
- **`make check`** is **122/122**.

### GLM-5.3-Flash-from-scratch: Stage A, and the linear-attention forward

- **`reference/glm53_flash/`**: the educational 25.7M model vendored at commit `96d9c7c`, MIT,
  with `VENDORED.md` stating that it is not the 320B release. Its 12 tests pass under
  `reference/.venv-rf` (torch 2.11.0+cu128).
- **Stage A** (`examples/language/glm53_flash/model_a.bend`): a 74,144-parameter dense-attention
  harness, width 32, four blocks, four heads of 8, 64 positions, SwiGLU 128, tied head, built
  from BendTorch's existing `Embedding`, `RMSNorm`, `RoPE`, `Attention`, `SiLU`, `Tie` and
  `CrossEntropy`. `probe_a.bend` prints whole logits and all 74,144 parameter gradients.
- **The gate** (`reference/test_glm53_layers.py`, six checks): logits **2.38e-07**, gradients
  **2.24e-08**, loss 5.5654626, causality (first half exact, future changed), the Stage B
  linear-attention forward **9.77e-04** on values to 4,096 (relative < 1e-6), and RoPE
  **2.44e-04** on values to 2,000 (relative < 1e-7). The RoPE bridge is explicit: the reference's
  interleaved pairs, BendTorch's half-split pairs, and the q/k row permutation
  `[0,4,1,5,2,6,3,7]`.
- **Stage B forward** (`lin_core.bend`): ELU+1, the causal prefix of `k` and `k⊗v`, the
  numerator, denominator and 1e-6 clamp, in list form. **Not built:** the linear-attention
  backward, Stage C, the MoE, the four-stream HyperConnection, RL and the 25.7M run.

### The rectified-flow port gets a whole-array gate, and the bug it caught

- **`reference/test_rectified_flow_layers.py`** (in `make check`): builds
  `reference/probe_rectified_flow_layers.bend` once and compares fourteen whole
  outputs with a NumPy twin — `Down2` forward and its exact-inverse backward
  (4,096 values each), the 2x2 max pool and its arg-max scatter (1,024 and
  4,096), the zero-query XAttnRes mix (1,024), the learned-query mix on equal
  and unequal keys (1,024 each), its saved weights (256), its five gradients
  (`dvh`, `dxc`, `dkh`, `dkx`, `dq`), and `Softmax(2,2)`'s axis. The
  permutation, pool, scatter and zero-query gates are exact (0.0); the learned
  mix and its backward are under 5e-3.
- **The mix backward**: `dvh`, `dxc`, `dkh`, `dkx` and the query gradient,
  with BendTorch's own `Softmax` backward on the saved weights. The
  query-gradient gate caught a missing recursion in `dq_sum` while it was
  being written (the sum stopped after row 0; max |diff| 689).
- **The bug it caught**: `dot8` indexed the 1x8 query with the row offset, so
  every row after the first read past the query's end, scored 0 and silently
  averaged 50/50. The row-only probes had passed because row 0 reads offset 0;
  the `kx = 2 * kh` whole-array gate failed immediately. `dot8` now reads the
  query at 0..7 and moves only the key/value row with `off`.
- **`examples/diffusion/rectified_flow/README.md`**: the 16x16, dim-8,
  one-hour target, the layer/probe inventory, the mix formula, and what is
  next (XAttn wrapper, memory-KV attention, DAG, `rf16`).

### Lab survey: the build environment, the checkpoint default, portable caches

- **`reference/check_all.py`** supplies the local `libLLVM` path to the build
  checks only, so `make check` builds from a fresh shell; `--verdict` still
  runs without it, as Lean needs its own clang.
- **`reference/rectified_flow_compare.py`**: `--sample` without `--checkpoint`
  loads the newest checkpoint under `--checkpoints`; `--resume` is now an
  honest warm start (the vendored `Trainer` has no step resume, and the old
  `checkpoint.latest.pt` was never written).
- **`examples/diffusion/{unet,flowers_flow}`** fall back to
  `$HOME/.cache/bend-lab` instead of one machine's absolute path.
- **Docs and links**: stale private-ledger and absolute-home references removed,
  three broken Markdown links fixed, the README's `make check` count and
  two-level UNet sentence brought up to date, `.understand-anything/tmp/`
  ignored, and `results/lucidrains-rectified-flow/` added to `.gitignore`.


### The UNet's fifth level: better flower identity, no more texture

- **`examples/diffusion/unet/{model,train,config}.bend`**: a **fifth level**, 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 4 -> 8
  -> 16 -> 32 -> 64, 64 channels, no normalisations, the clean image as the flow target, 7,067 flowers. 298,439
  parameters, 3,600 steps; loss **15.3262 -> 0.0259** (the first value is the untrained head on the first batch; step 1
  is 0.1343).
- **What it reaches**: 64 draws at Euler 32, mean/spread **+0.325 / 0.179** (the data's +0.384 / 0.232); **held-out
  legibility 0.547** against a ceiling of 0.574, **94.8%** of the span above the noise floor. The four-level record,
  re-scored today with the same evaluator, is 0.605 / **0.501** / 85.8% (published 0.498 / 85.4%), so the fifth level is
  +4.6 points of held-out legibility and +9.0 points of span. The trained legibility is 0.633, above the ceiling:
  memorisation, stated as such.
- **What it did not do**: the gain is global identity, not texture. The raw draws are lower contrast than the
  four-level's (+0.325 / 0.179 against +0.381 / 0.238) and the top spectrum band is 0.034 against 0.039 (the data's is
  0.137). On the same 8 draws, 128 Euler steps give spread 0.174 and top band 0.027, so the sampler is not the cause;
  **32 steps stays**. The depth line stops here; the next lever is detail and contrast, not a sixth level.
- **The draw-count trap**: a max-over-correlations statistic depends on how many draws are averaged. The five-level's
  first 8 draws score 0.575 held out (100.3% of the span) against 0.547 for all 64 (94.8%); an earlier 128-step run on
  the four-level looked like +3 points for the same reason. Comparisons hold the draw count fixed.
- **Gates**: the twins cover nine convolutions and 22 weight blocks; NumPy **1.621e-05** at 2e-05 and torch
  **2.098e-05**, 0.83x its allowance; the checkpoint resumes **bit-identical** at 298,439 weights.
- **The trap paid in the twins**: the new level outputs were first named `b2`/`b3`, which are the biases of the same
  names; the levels are `l4`/`l2` now.

### The UNet draws flowers: four levels, and an objective that was the wrong one

- **`examples/diffusion/unet/{model,train,config}.bend`**: the gate example's U with **four levels** (64 -> 32 -> 16 ->
  8 -> 4 -> 8 -> 16 -> 32 -> 64), 64 channels each, residual skips at 32x32 and 16x16, a linear bypass from the input,
  and no normalisations. Trained on the **7,067** flowers of the training split with rectified flow whose **target is
  the clean image**, and a sampler that converts, `v = (x0hat - x)/(1-t)`.
- **What it reaches**: 3,600 steps, loss **1.1831 -> 0.0298** (a predictor of the data's mean scores 0.0539); 64 draws
  whose mean and spread are the data's (+0.379 / 0.246 against +0.384 / 0.232); **held-out legibility 0.498** against a
  ceiling of 0.574, i.e. **85.4%** of the span above the noise floor. The first record of this example was
  0.259 / 0.206 with speckle for draws.
- **Five levers, each measured**: the objective (v -> x0) 0.274 -> 0.426; the third level 0.439 held out; the corpus
  alone **+0.0**; the fourth level 0.439 -> **0.498** (bottleneck 8x8 -> 4x4); a wider stack (CH 96) did not beat it.
  Group norms and batch size were rejected earlier by measurement (0.233 with norms against 0.274 without; a batch of
  26 for +0.5 points of held-out span).
- **Measurement fixed, twice**: the eval used to score against the 3,874-image subset while the model trained on 7,067,
  inflating both the legibility and the ceiling - it now defaults to the full export, prints the reference set's size,
  and says so when the trained legibility passes the ceiling (memorisation). The forward gates moved from 1e-5 to 2e-5
  when the fourth level added two convolutions: measured 9.15e-06 (NumPy) and 1.18e-05 (torch), with the reason in
  each file.
- **New scripts**: `reference/unet_flowers_eval.py` prints legibility, held-out legibility, the ceiling, the noise floor
  and the draws' spectrum through `flowers_report.py`'s `legibility`.
- **The bugs that cost the earlier runs**: the pair builder and the sampler built the five input channels in opposite
  orders; `init_arr` zeroed hardcoded offsets from the first two-channel layout; a batch that does not divide the image
  count reads past the export; and a run shorter than one pass trains on a class-ordered prefix. All four are fixed or
  gated.
