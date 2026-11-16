# The tiny-LM training curve in two lanes (A2)

The byte-corpus tiny-LM — two SISO Mamba-3 layers (d_model 64, d_state 16,
expand 2, headdim 32) + embedding + final norm + head, vocab 127, the
first 20 train examples, batch 1, 127 next-byte targets per example, 20
steps — trained identically in two lanes:

| lane | what it is |
|---|---|
| `f32` | the twin path on CUDA: `W.twin_forward` (explicit t-by-t loop, F32 everywhere) — the same lane as the CPU F32 record `twin.json`; the device differs (CUDA vs CPU), a banner, never a delta |
| `bf16` | the public `Mamba3` module on CUDA (internally bf16: the wrapper casts Q/K/V/trap/angles before the kernel) — everything else (embedding, head, norm, cross-entropy) stays F32 |

Same data, same seeds, same schedule: AdamW (lr warmup 5 then cosine
1e-3→1e-4, betas (0.9, 0.999), eps 1e-8, weight_decay 0.01), gradient
clipping by global norm at 1.0, torch step order, bit-identical inits
loaded from `.scratch/mamba3_tiny/w_init/` (seeds 101/102, extra 20261115).

Command: `reference/.venv-rf/bin/python reference/mamba3_tiny_lanes.py
--lane both`; gate: `reference/.venv/bin/python
reference/test_mamba3_bf16_curve.py` (`MAMBA3 BF16 CURVE PASS`).

## Per-step losses

| step | f32 loss | bf16 loss | |Δ| | f32 gn | bf16 gn |
|---|---|---|---|---|---|
| 0 | 4.838168 | 4.838161 | 7.15e-06 | 0.9846 | 0.9846 |
| 1 | 4.849788 | 4.849782 | 6.68e-06 | 0.8674 | 0.8675 |
| 2 | 4.797366 | 4.797346 | 2.00e-05 | 0.9354 | 0.9354 |
| 3 | 4.769843 | 4.769840 | 2.38e-06 | 0.9232 | 0.9232 |
| 4 | 4.775183 | 4.775199 | 1.62e-05 | 0.8704 | 0.8703 |
| 5 | 4.713411 | 4.713420 | 9.06e-06 | 0.9293 | 0.9293 |
| 6 | 4.679996 | 4.680008 | 1.24e-05 | 0.8751 | 0.8750 |
| 7 | 4.607713 | 4.607717 | 4.29e-06 | 1.0466 | 1.0466 |
| 8 | 4.581757 | 4.581771 | 1.38e-05 | 1.0014 | 1.0014 |
| 9 | 4.567843 | 4.567830 | 1.38e-05 | 0.9511 | 0.9511 |
| 10 | 4.462538 | 4.462583 | 4.53e-05 | 1.0237 | 1.0237 |
| 11 | 4.444792 | 4.444825 | 3.29e-05 | 1.0266 | 1.0266 |
| 12 | 4.452862 | 4.452889 | 2.72e-05 | 0.9769 | 0.9769 |
| 13 | 4.464232 | 4.464180 | 5.29e-05 | 0.9412 | 0.9412 |
| 14 | 4.469739 | 4.469711 | 2.81e-05 | 0.9256 | 0.9256 |
| 15 | 4.318074 | 4.318052 | 2.19e-05 | 1.0131 | 1.0131 |
| 16 | 4.344019 | 4.344061 | 4.24e-05 | 1.1053 | 1.1053 |
| 17 | 4.355888 | 4.355869 | 1.91e-05 | 1.0042 | 1.0041 |
| 18 | 4.370821 | 4.370852 | 3.15e-05 | 0.9906 | 0.9906 |
| 19 | 4.333054 | 4.333111 | 5.67e-05 | 1.0452 | 1.0452 |

## Summary

| | f32 | bf16 |
|---|---|---|
| train hits | 512 / 2540 | 512 / 2540 |
| max \|Δloss\| vs the other lane | — | 5.67e-05 (step 19) |
| max \|Δgn\| vs the other lane | — | 6.74e-05 (step 19) |
| final-step loss | 4.333054 | 4.333111 |

Cross-check, CUDA F32 vs the recorded CPU F32 (`twin.json`,
`mamba3_tiny_twin.py --mode train`, same seeds): max per-step loss
difference **4.8e-07** — the same lane on a different device, agreeing
to F32 noise.

## What this does not say

- It is a lane measurement, not a verdict.  The bf16 lane tracked the
  F32 curve within 5.7e-05 for 20 steps at lr ≤ 1e-3 on 20 train
  examples; nothing here is a claim about longer horizons, larger
  corpora or other precisions.  At 20 steps the loss is still far above
  the bigram baseline (see `results/mamba3-tiny-lm.md`); no
  language-model claim is made in either lane.
- The CPU and CUDA F32 records differ by ≤ 4.8e-07 per step; CUDA vs
  CPU is a banner, never a delta.
- Mid-training evaluation is unavailable at this budget; the curve is
  the 20-step training losses only.
- A lane, device or version change is a banner, never a delta.
