# The tiny-LM step across changes

One row per change that touches what a step costs, measured with
`scripts/bench_mamba_tiny.sh` (the gate) or `scripts/bench_ab.sh` (two builds
of the same bench, alternated in shuffled rounds). Both refuse to measure when
the one-minute load average is above `MAX_LOAD` (1.0 by default); `FORCE=1`
overrides, and a row measured that way is a reference, not an idle
measurement, with the load in its column.

`ms_per_step` is the program's own clock, `full_step_x3_ms / 3`, three steps
from the same weights. A row is comparable only with another row of the same
Bend pin and the same model shape; the gate fails if the loss or the backward
and scan heads moved.

| date | change | a ms/step | b ms/step | rounds | load |
|---|---|---|---|---|---|
| 2026-11-18 | list accumulator → array accumulator (FORCE=1, reference) | 10860 | — | 1 | 2.9 |
| 2026-11-19 | array accumulator, first idle record | 11163 | — | 1 | 0.52 |
