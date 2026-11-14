# mamba3-step gate — 2026-11-13

Session record for the Mamba-3 decode step (ISSUE-127) and the `T.Train`
per-step loss question (ISSUE-126). All numbers below are from commands run
in `/home/adeerkhan/github/bend/bend-lab` on this lane (Bend 2.0.38,
RTX 5060, `reference/.venv/bin/python`); Bend ran with
`LD_LIBRARY_PATH=$HOME/.local/lib/llvm22`, `BEND_NO_TELEMETRY=1`.
This file is committed.

## ISSUE-127 — the decode step (ISSUE-127 → IMPLEMENTED, 16 tokens)

`make mamba3-step`:

```text
reference/.venv/bin/python reference/test_mamba3_step.py
MAMBA3 STEP PASS   (build 2.2 s, run 0.08 s)
step_tokens:       [0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
golden gen_tokens: [0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   (exact, all 16)
```

Golden source: `reference/mamba3_golden.npz` `gen_tokens`, the same golden
the full-prefix probe (`probe_mamba3_gen.bend`, gated by `make mamba3`) must
match. `make mamba3` was re-run this session and PASSes.

What the step is. `examples/language/mamba3/step_mamba3.bend` decodes one
token at a time: a two-layer weight-tied block driven by one self-recursive
`step_go`, state carried unmatched through a `r: SR` parameter
(`ST & (List<&2, Nat> & Nat)` became an `SR` record after an inference
failure — see `docs/bend-2.0.38-pitfalls.md`). Carries, per layer:

- `ssm`: 2048 floats in one `Array<F32>` (4 heads × 32 v-rows × 16 qk dims),
  the same array-state substrate the full pass uses;
- `angle`: 16 floats (4 heads × 4);
- `pend_x` / `pend_k` / `pend_dt` / `pend_trap`: the previous row's `x` (128),
  roped `k` (64), `DT` (4), `trap` (4).

The trapezoid wrinkle. The full-prefix pass computes row `t`'s `scale` from
row `t+1`'s `DT` and `trap` (`scale = DT_t*trap_t + DT_{t+1}*(1 - trap_{t+1})`),
and the final row of a pass gets `scale = 0` (its `shift` row is `Nil`,
`nth` defaults to `0.0`). An online decoder cannot know `DT_{t+1}` when it
emits row `t`. The step therefore folds **every** row with `scale = 0` (the
final-row rule, uniformly) and, when the next row's `DT`/`trap` arrive,
applies an additive rank-one correction
`S[v, q] += (scale * x_v) * k_q` to the carried array — in the same float op
order as the reference's `row_set`, so the state is the same F32 computation,
not an approximation. The angle has no future dependency and threads
directly.

Parity at 64 and 256 tokens — **both exact** (checked this session, after
the 16-token gate cleared):

```text
$ sed -e 's/12n), 16n)/60n), 64n)/' examples/language/mamba3/probe_mamba3_gen.bend > .scratch/probe_mamba3_gen64.bend
$ sed -e 's/}, 16n), 16n)/}, 64n), 64n)/' -e 's/    case 15n:/    case 63n:/' \
    examples/language/mamba3/step_mamba3.bend > .scratch/step_mamba3_64.bend
$ sed -i 's#import ./mamba3_siso.bend as M#import ../examples/language/mamba3/mamba3_siso.bend as M#' \
    .scratch/probe_mamba3_gen64.bend .scratch/step_mamba3_64.bend
$ BEND_MEM=12G scripts/bend-guard.sh $HOME/.bend/bin/bend -o .scratch/probe64.bin .scratch/probe_mamba3_gen64.bend
$ time ./.scratch/probe64.bin
gen_tokens: 0 1 2 3 1 ... 0 1 4 7 0 1   # 64 tokens, run 17.7 s
$ time $HOME/.bend/bin/bend .scratch/step_mamba3_64.bend
step_tokens: 0 1 2 3 1 ... 0 1 4 7 0 1   # 64 tokens, run 0.45 s; equal to full-prefix, exactly
```

256 tokens:

```text
$ sed -e 's/12n), 16n)/252n), 256n)/' examples/language/mamba3/probe_mamba3_gen.bend > .scratch/probe_mamba3_gen256.bend
$ sed -e 's/}, 16n), 16n)/}, 256n), 256n)/' -e 's/    case 15n:/    case 255n:/' \
    examples/language/mamba3/step_mamba3.bend > .scratch/step_mamba3_256.bend
$ sed -i 's#import ./mamba3_siso.bend as M#import ../examples/language/mamba3/mamba3_siso.bend as M#' \
    .scratch/probe_mamba3_gen256.bend .scratch/step_mamba3_256.bend
$ BEND_MEM=12G scripts/bend-guard.sh $HOME/.bend/bin/bend -o .scratch/probe256.bin .scratch/probe_mamba3_gen256.bend
$ time ./.scratch/probe256.bin
gen_tokens: 0 1 2 3 1 ... 6 6 6        # 256 tokens, run 29 m 17 s
$ python3  # compare step256_out.txt vs gen256_out.txt
step len 256 gen len 256 equal 256: True
```

Note on the failure mode: `bend file.bend` (the interpreted path) dies with
`bend: memory fault (machine stack overflow?)` at seq len >= 64; the
compiled path (`bend -o` then running the binary) works at 64 and 256.
`ulimit -s unlimited` does not help the interpreter. This is the same class
of limit as ISSUE-007 (the Bend VM stack), not a step-decoder problem.

The step's wall time is 0.08 s at 16 tokens, 0.45 s at 64, 1.2 s at 256 —
linear in the token count, as an online decoder should be; the full-prefix
reference is quadratic (1 s / 17.7 s / 29 m 17 s).

`make check` was re-run after the step landed: at that moment **143/143
checks passed** (37 m 48 s); after `reference/test_mamba3_train_curve.py`
landed the gate reports **144/144** (the curve test is auto-included; the
fresh batch's `full` row honestly flagged the stale `143/143` pin and the
pin was corrected to 144/144). The full gate includes both Mamba-3 test
files. Dashboard: the 16-token gate is run 114, the curve gate is run in the
20261113-205808 session (`make session TASK=mamba3-train-curve`).

## ISSUE-126 — `T.Train` cannot report the per-step loss

`make mamba3-layer` compares only what is observable from outside the
training loop, by design of `T.Train`:

- tokens exact: `[3, 6, 0, 5, 0, 7, 0, 7]`;
- final-step loss: custom trainer `2.1401443` vs `T.Train` layer `2.1404066`,
  relative `1.2e-04`, gate `1e-3` (loss of earlier steps is overwritten
  inside the loop; only `loss: F32` survives on the `Train` record);
- weight heads (`w_in`, `E`, `Wh`, `w_out`) ≤ `1.3e-05` relative.

The recorded artifact now exists: `make mamba3-train-curve`
(`reference/test_mamba3_train_curve.py`) records the custom curve and the
final-step comparison into `results/mamba3-train-curve.md` — custom ten losses
`2.1568611 … 2.1401443`, layer final `2.1404066`, rel `1.23e-04` (gate 1e-3),
tokens equal, heads ≤ `8.9e-07`. No curve from the layer path exists: there is
no curve to record — `Train.loss` is one scalar updated in the loop, and no
accessor is exposed.
Per the goal, the upstream question was asked, not patched:

- `aaaxn/bendtorch#86` (linked): *Question: is keeping only the last-step
  loss in `T.Train` intentional?* (filed 2026-11-13; bendtorch is the repo we
  pin at 2.1.0.0; no code pushed, no PR).

No per-step gate was written: it would be a comparison of two curves that
cannot both be read, i.e. decoration. The layer gate's limits are stated in
its output and in `mamba3_layer.bend`'s doc.

## ISSUE-102 — the published before-RL 0/24 does not reproduce

Previously measured on this lane, recorded with commands in
`results/glm53-c/canary-16.json` (canary: overfit 16 labels of the fixed
batch, 50 steps at lr 0.03; Bend final loss `0.04192061` vs twin
`0.04192125`, relative `1.5e-05`; weights relative `6.1e-05`) and in
`results/glm53-reference/rloo-summary.json` (RLOO from the report's
description, 96 prompts/16 samples/96 updates, 1240 s; confirm-three-families
greedy **3/24 → 12/24**, all-eight **15/64 → 21/64**; the published
`0/24 → 16/24` and `25/64 → 34/64` are not reproduced in magnitude).
Our documents state this (`docs/CHANGELOG.md`, `docs/agents/handoff.md`,
`docs/agents/upstream.md`). This session did not re-run the RL canary: the
25.7M before-RL checkpoint is not in this tree and retraining is out of
scope for one session; the numbers above stand as measured. The published
third-party README (`reference/glm53_flash/README.md`) was not edited.

## Bend language asks (tracked elsewhere)

- `bendlang/bend#2146` (linked): record field projection / match on a local
  binder / Data-shared records — ISSUE-104/105.
- `bendlang/bend#2147` (linked): forward references / mutual recursion — ISSUE-107.
- `bendlang/bend#2150` (linked): the intended encoding for evolving state
  across an online loop — ISSUE-127; the 126/127 minimal examples are
  `bendtorch#86` (pointer + limits) and #2146/#2147/#2150.
- The design asks behind the pitfalls doc's ISSUE-120/121/122/123/124/125/128
  sections: `docs/bend-2.0.38-pitfalls.md` and `docs/agents/oss-needs.md`.
