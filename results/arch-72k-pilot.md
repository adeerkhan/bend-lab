# The 72k two-arm pilot

`reference/arch_compare.py --steps 200`; 25600 tokens per arm, the first 200 windows of corpus128-b32, byte-identical for both (checked).

| arm | params | steps | first loss | last loss | wall | load |
|---|---|---|---|---|---|---|
| Mamba-3 | 71952 | 200 | 4.838168 | 3.178023 | 7539.0 s | 3.76 to 2.10 |
| GLM hybrid | 74144 | 200 | 5.563474 | 1.789081 | 377.9 s | 2.10 to 1.98 |

## Banners

- vocabulary: Mamba 127, GLM 260 (133 classes never used), so the two cross-entropies are not one measurement
- optimizer: Mamba AdamW 0.9/0.999 wd 0.01 clip 1.0; GLM CAdamW 0.9/0.95 wd 0.1 clip 0.05
- schedule: for --steps N both peak at 3e-4, but Mamba warms up 5 steps and cosines to 3e-5 over N while the GLM is constant
- held-out: Mamba's 4,096-byte split, GLM's val batch, different slices

## What a 200-step matched run needs

- the GLM arm needs the Mamba optimizer (0.9/0.999, wd 0.01, clip 1.0) and the same warmup/cosine over the shared budget (its LR is constant today)
- the GLM head is 260 wide; a matched run needs 127 (or a restricted eval), which changes its parameter count and its gates

The pilot validates the wiring and the data alignment only; no quality verdict is read from it.
