# The GLM-5.3-Flash-from-scratch port, Stages A/B and the dense baseline

This is the start of the port of
[`vukrosic/glm-5.3-flash-from-scratch`](../../../reference/glm53_flash/README.md),
the **educational 25.7M-parameter** model, not the 320B released model. The
vendored source is under `reference/glm53_flash/` at commit `96d9c7c`.

## What is built

**Stage A, the dense-attention harness.** Width 32, four blocks, four heads of
8, 64 positions, SwiGLU hidden 128, tied embedding/head: **74,144
parameters**. It is composed from BendTorch's `Embedding`, `RMSNorm`, `RoPE`,
`Attention`, `SiLU`, `Tie` and `CrossEntropy`.

- `model_a.bend` — the model at 64 positions.
- `weights_a.bend` — the loader for the twin's tensors.
- `probe_a.bend` — loads the twin's tensors and prints the whole logits, the
  loss and all 74,144 parameter gradients.
- `probe_rope.bend` — RoPE alone, in BendTorch's half-split layout.
- `reference/test_glm53_layers.py` — the Stage A/B gate: a NumPy-generated
  weight set, a PyTorch-autograd golden (`reference/glm53_golden_a.npz`), and
  6 checks.

Measured by `reference/.venv/bin/python reference/test_glm53_layers.py`:

| check | result |
|---|---|
| Stage A logits (33,280 values) | max abs diff **2.38e-07** |
| Stage A gradients (74,144 values) | max abs diff **2.24e-08** |
| Stage A loss | 5.5654626 (golden 5.5654621) |
| causality | first 64 rows exact, future 64 changed |
| Stage B linear-attention forward | max abs diff **9.77e-04** on values to 4,096 (relative <1e-6) |
| RoPE forward | max abs diff **2.44e-04** on values to 2,000 (relative <1e-7) |

The RoPE bridge is tested explicitly: the reference rotates interleaved
even/odd pairs, BendTorch rotates half-split pairs, and the q/k weight rows are
permuted by `[0,4,1,5,2,6,3,7]` on the way in. The Stage A gradients are
compared in the file layout, so the permutation is part of the gate.

## The dense Stage A baseline

`train_a.bend` is the Stage A model under `T.Train`: AdamW at
**lr 3e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.1**, and the
reference's cross-entropy with the ignored-label mask (`loss_masked.bend`).
The data is the reference's synthetic Python corpus, encoded with its byte
tokenizer: the first 129 tokens, in **two fixed rows of 64** inputs and the
next-token labels. The batch is written once by
`reference/glm53_dense_data.py` and reused; 128 tokens per step.

**The 20-step gate.** `reference/glm53_dense_twin.py` is the PyTorch-autograd
twin: the same NumPy weights, the same batch, the same optimizer, 20 steps.
`reference/test_glm53_dense.py` runs the Bend program on the same files and
compares every step's loss and the final weights and gradients:

| check | tolerance | measured |
|---|---|---|
| loss at every step | 1e-4 relative | max **5.174e-07** |
| final weights (74,144) | 1e-3 relative | **3.455e-07** |
| final gradients (74,144) | 1e-3 relative | **2.157e-07** |

The 1e-3 gate on the weights and gradients is deliberate: 20 AdamW steps
amplify the last-bit rounding of the two implementations' matrix products.
The loss itself still agrees at 5.2e-07. Two can-fail runs (one embedding
weight moved by +2.0, one input byte changed) diverge at step 1 at
**3.887e-03** and **4.485e-04** relative, and the same gate rejects them.

**The 400-step run** (one fixed batch, so this is an overfit baseline, not a
corpus pass), measured by `reference/glm53_dense_report.py`:

| quantity | value |
|---|---|
| parameters | 74,144 |
| steps / tokens seen | 400 / 51,200 |
| first loss | 5.595126 |
| final loss (step 400) | 1.724139 |
| mean of the last 100 | 2.0280423 |
| loss at the final weights | 1.716634 |
| build | 86.059 s |
| run | 7.13 s |
| peak RSS | 57,040 KiB |

**Greedy dev/confirm evaluation.** `gen_a.bend` runs the same layer stack at
192 positions (`model_eval.bend`) because a prompt is 76-95 tokens and the
evaluator allows 48 new tokens. `reference/glm53_dense_eval.py` applies the
vendored `glm53_flash/evaluator.py` to 8 identities per family (64 tasks),
the reference's prompts, greedy decode:

| split | solved | tests | valid programs | token-cap hits |
|---|---:|---:|---:|---:|
| dev | **0/64** | **0/192** | 0 | 57 |
| confirm | **0/64** | **0/192** | 0 | 62 |

Every family is 0/8 solved and 0/24 tests in both splits. This is the honest
result for a 74k model trained on one fixed 64-byte batch and then asked to
generate at positions it never trained on; it is **not** the reference 25.7M
model and not a corpus-trained baseline. What it does establish is that the
whole path (data, optimizer, backward, checkpoint, generation, evaluator)
runs and is gated.

The evaluation manifest is frozen in
`reference/glm53_dense_eval_golden.json`; `make check` re-checks every prompt
against the reference tasks, every verdict against the reference evaluator,
the four summary counts, and the frozen 48-token sequence.

## What is not built

- Stage B backward (dq, dk, dv).
- Stage C, the gathered sparse attention, and its backward.
- The MoE, the four-stream HyperConnection, RL and the 25.7M model. Those are
  later stages.

No upstream issue has been filed: the remaining work is in this repository,
not a Bend defect.

## Run

```console
$ reference/.venv/bin/python reference/glm53_dense_data.py --prepare
$ reference/.venv-rf/bin/python reference/glm53_dense_twin.py --steps 20 \
      --golden reference/glm53_dense_golden.npz
$ reference/.venv/bin/python reference/glm53_dense_report.py
$ reference/.venv/bin/python reference/glm53_dense_eval.py \
      --weights results/glm53-dense/final_weights.bin --out results/glm53-dense
$ reference/.venv/bin/python reference/test_glm53_dense.py
$ reference/.venv/bin/python reference/test_glm53_layers.py
$ make check
```
