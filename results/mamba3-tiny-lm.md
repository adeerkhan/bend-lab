# Mamba-3 tiny-LM — the held-out run

Recorded by `reference/test_mamba3_tiny_lm_run.py` (regenerated on every run).

Command: `make mamba3-tiny-lm` (`reference/test_mamba3_tiny_lm_run.py`)

## Configuration

- Model: two-layer SISO Mamba-3 LM, d_model 64, d_state 16, expand 2, headdim 32.
- Vocabulary 127 (the corpus's observed maximum + 1); positions 128; targets 127 per example.
- Data: `.scratch/mamba3_tiny/` (built by `reference/mamba3_tiny_data.py`):
  the first 64 corpus examples (8,192 tokens) for training, the corpus's own
  held-out split (32 examples, 4,096 tokens, 4,064 next-byte targets).
- Schedule: batch 1 in file order (no shuffling), one example per step, the
  first 20 train examples, 20 steps. AdamW lr = warmup (5 steps linear 0 -> 1e-3)
  then cosine from 1e-3 to 1e-4 over steps 5..19; weight_decay 0.01,
  betas (0.9, 0.999), eps 1e-8, gradient clipping by global norm at 1.0,
  bias correction, decoupled decay, torch's step order.
- Seed: l1 101, l2 102, E/Wh/nw 20261115 (`reference/mamba3_tiny_twin.py`).
- Bend: float32; the twin: float32 on the CPU.

## Numbers

| | Bend | twin (float32, CPU) |
|---|---|---|
| train hits (pre-update, of 2540) | 512 | 512 |
| train accuracy (pre-update) | 0.2016 | 0.2016 |
| held-out CE, nats (of 4,064 targets) | 17599.3 | 17599.3 |
| held-out bits/byte | **6.2476** | **6.2476** |
| held-out token accuracy | **0.3147** (1279/4064) | **0.3147** (1279/4064) |
| final weights (19 per-tensor sums of squares), worst rel | 7.858e-06 | reference |

## Against the baselines (the same 4,096 held-out bytes)

- unigram 4.8093 bits/byte; bigram **3.3569** bits/byte (the corpus statistics, unchanged by this run);
- the twin after 20 steps: 6.2476 bits/byte; the Bend trainer: 6.2476.

The run is 20 steps of 127-target cross entropy on 20 train examples; the
held-out split was not seen in training. Both numbers above are what those
20 steps reach. **The full corpus pass and a convergence claim are not made
by this card**: the Bend 128-position fwd+bwd measures seconds per image on
this lane, so the full 8,192-token pass is not this session's budget, and the
comparison to the bigram baseline is recorded, not met.

## Gates on this artifact

- `reference/test_mamba3_tiny_data.py` (`make mamba3-tiny-data`): the data.
- `reference/test_mamba3_tiny_opt.py` (`make mamba3-tiny-opt`): optimizer parity, same run.
- `reference/test_mamba3_tiny_lm_run.py` (`make mamba3-tiny-lm`): this card.

MAMBA3 TINY-LM: PASS
