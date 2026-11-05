# The 2D flow: three settings, and the comparison with PyTorch

`main.bend`, two moons (128 points), AdamW at 3e-3, full batch, 32 Euler steps. Every number below is from a run on
this machine and is recorded under `results/flow-matching/` (each variant keeps its own `run.json`, `loss.svg` and
`scatter.svg` in `variants/`); the official record is the winner's. Floor: a model that predicts nothing scores
**1.4665**.

## The three settings

| run | hidden | lr | steps | first loss | final loss | last 100 |
|---|---|---:|---:|---:|---:|---:|
| **baseline (and the record)** | **128** | 3e-3 | 6,000 | 1.4768 | **0.8360** | **0.8859** |
| lower lr | 128 | **1e-3** | 6,000 | 1.4768 | 0.8412 | 0.8866 |
| more steps | 128 | 3e-3 | **12,000** | 1.4768 | 0.8630 | 0.9028 |
| wider | **256** | 3e-3 | 6,000 | 1.5622 | 0.8471 | 0.8887 |
| wider + more steps | 256 | 3e-3 | 12,000 | 1.5622 | 0.8614 | 0.9033 |

**None of the three beats the baseline.** A lower learning rate changes nothing within noise; 12,000 steps at 128 wide
make it *worse* (0.8630 against 0.8360); 256 wide is slightly worse at both step counts, and its heavier start (1.5622)
does not pay back. The design probe's plateau (0.8662) and these runs agree: the model is at the capacity of this
setup, and the next lever is not one of these three.

![loss, the record](../../../results/flow-matching/loss.svg)

![samples, the record](../../../results/flow-matching/scatter.svg)

The record's scatter is the two-moons shape drawn from noise in 32 Euler steps: sample mean `[0.049, -0.048]`, std
`[0.610, 0.675]` against the data's `[0.000, 0.000]` and `[0.577, 0.659]`. The spread is right and the mean is close.

## The 0.4318 that was not real

The first version of this file recorded **0.4318** for "wider", and it was wrong. Changing the width was done by
replacing lines that were exactly `  128n`, which also hit `def n() -> Nat:` — the batch size. Those runs trained a
256-row batch against a 128-point dataset, and `at()` returns `0.0` past the end, so **half of every target was zero**.
The easier target explains the whole drop. With `n()` restored to 128 the same 256-wide run reads 0.8471.

The confounded run is kept as `variants/zeropad-artifact.json` (first 0.9280, final 0.4318, last 100 0.4465) so the
mistake is on the record rather than deleted.

## Against PyTorch

`reference/compare_flow2d.py` trains the same network in torch and now **agrees**:

| | Bend | torch | delta | tolerance |
|---|---:|---:|---:|---:|
| first loss | 1.476769 | 1.476769 | 8.97e-08 | 1e-3 |
| final loss | 0.835969 | 0.846537 | 1.06e-02 | 2e-2 |
| last 100 | 0.885889 | 0.893301 | 7.41e-03 | 2e-2 |
| sample mean x | 0.048676 | 0.046436 | 2.24e-03 | 0.05 |
| sample mean y | -0.048143 | -0.070823 | 2.27e-02 | 0.05 |

The first loss agreeing to `9e-08` is the strong part: it says the initial weights, the data order, the U32 levels and
the per-step noise are the same on both sides. The end-of-run residual (about 1%) is AdamW trajectory drift over 6,000
steps, not a model difference.

Getting there took three corrections, all in the comparator (ISSUE-060):

- **the data order**: `two_moons.moons()` alternates the moons (`upper0, lower0, upper1, ...`), and the flow's `norm`
  reads consecutive pairs; `moons_norm()` concatenated them (all upper, then all lower), so every noise row was paired
  with a different point and the first loss moved by 0.05;
- **the noise**: the example's `do_step(s, …)` draws `noise_of(s, 256)` **per step**, not one cloud reused;
- **the init scale**: the example's `row_vals` divides column `j` by `sqrt(cols - j)` (the recursive helper passes the
  remaining column count), not by `sqrt(cols)`; the uniform scale moved the first loss by 0.03.

The earlier suspicion that Bend prints the loss of the *wrong step* was wrong: with the three corrections the printed
first loss is torch's pre-step loss at step 0 to `9e-08`, so the printed value is the step's own loss with its pre-step
weights.

## Gate on the forward

`reference/test_flow2d.py` compares the frozen probe against `reference/twin_flow2d.py` element by element; all six
gates pass, the worst `3.576e-07` against `1e-5`. The twin reads the width from the example (`hidden()`), so a width
change moves the gate instead of leaving it expecting the old weights (ISSUE-062).

## Reproduce

    BEND_LIB=$(pwd)/.bend/env bend -o .scratch/flow_bin examples/diffusion/flow_matching/main.bend
    .scratch/flow_bin                       # the record: hidden 128, 6,000 steps, ~45 s
    reference/.venv/bin/python reference/flow_results.py <log>
    reference/.venv/bin/python reference/test_flow2d.py
    ../bendtorch/reference/.venv/bin/python reference/compare_flow2d.py

The program prints its own `hidden:`/`steps:`/`euler:` and `flow_results.py` reads them, so the record cannot describe a
configuration other than the one it ran (ISSUE-059).
