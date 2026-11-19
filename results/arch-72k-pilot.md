# The 72k two-arm pilot

`reference/arch_compare.py --steps 20`; 2560 tokens per arm, the first 20 windows of corpus128-b32, byte-identical for both (checked).

| arm | params | steps | first loss | last loss | wall | load |
|---|---|---|---|---|---|---|
| Mamba-3 | 71952 | 20 | 4.838168 | 4.333053 | 392.7 s | 2.12 to 1.49 |
| GLM hybrid | 74144 | 20 | 5.563474 | 3.722496 | 31.5 s | 1.49 to 1.57 |

## Banners

- vocabulary: Mamba 127, GLM 260 (133 classes never used), so the two cross-entropies are not one measurement
- optimizer: Mamba AdamW 0.9/0.999 wd 0.01 clip 1.0; GLM CAdamW 0.9/0.95 wd 0.1 clip 0.05
- schedule: Mamba warmup 5 + cosine 1e-3 to 1e-4 over 20 steps; GLM constant 3e-4
- held-out: Mamba's 4,096-byte split, GLM's val batch, different slices

## What a 200-step matched run needs

- Mamba train_tokens.bin holds 64 windows; 200 needs 200
- Mamba trainer has no --steps/--lr/--data flags; its schedule is fixed at 20 steps
- the GLM arm needs the Mamba optimizer and a warmup/cosine over the same budget
- the GLM head is 260 wide; a matched run needs 127 (or a restricted eval), which changes its parameter count and its gates

The pilot validates the wiring and the data alignment only; no quality verdict is read from it.
