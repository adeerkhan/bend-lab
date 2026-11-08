# The GLM-5.3-Flash-from-scratch port, Stage A/B

This is the start of the port of
[`vukrosic/glm-5.3-flash-from-scratch`](../../../reference/glm53_flash/README.md),
the **educational 25.7M-parameter** model, not the 320B released model. The
vendored source is under `reference/glm53_flash/` at commit `96d9c7c`.

## What is built

**Stage A, the dense-attention harness.** Width 32, four blocks, four heads of
8, 64 positions, SwiGLU hidden 128, tied embedding/head: **74,144
parameters**. It is composed from BendTorch's `Embedding`, `RMSNorm`, `RoPE`,
`Attention`, `SiLU`, `Tie` and `CrossEntropy`.

- `model_a.bend` — the model.
- `probe_a.bend` — loads the twin's tensors and prints the whole logits, the
  loss and all 74,144 parameter gradients.
- `probe_rope.bend` — RoPE alone, in BendTorch's half-split layout.
- `reference/test_glm53_layers.py` — the gate: a NumPy-generated weight set,
  a PyTorch-autograd golden (`reference/glm53_golden_a.npz`), and 6 checks.

Measured by `reference/.venv/bin/python reference/test_glm53_layers.py`:

| check | result |
|---|---|
| Stage A logits (33,280 values) | max abs diff **2.38e-07** |
| Stage A gradients (74,144 values) | max abs diff **2.24e-08** |
| Stage A loss | 5.5654626 (golden 5.5654621) |
| causality | first 64 rows exact, future 64 changed |
| Stage B linear-attention forward | max abs diff **9.77e-04** on values to 4,096 (relative < 1e-6) |
| RoPE forward | max abs diff **2.44e-04** on values to 2,000 (relative < 1e-7) |

The RoPE bridge is tested explicitly: the reference rotates interleaved
even/odd pairs, BendTorch rotates half-split pairs, and the q/k weight rows are
permuted by `[0,4,1,5,2,6,3,7]` on the way in. The Stage A gradients are
compared in the file layout, so the permutation is part of the gate.

**Stage B, linear attention, forward only.** `lin_core.bend` implements the
reference's `ELU+1`, causal prefix of `k` and `k⊗v`, numerator `q·KV`,
denominator `q·K` and the `1e-6` clamp, in list form. Its forward matches the
NumPy twin; **the backward is not built yet**.

## What is not built

- Stage B backward (dq, dk, dv).
- Stage C, the gathered sparse attention, and its backward.
- The MoE, the four-stream HyperConnection, RL and the 25.7M model. Those are
  later stages.

The next step is the linear-attention reverse pass, then the sparse attention
in the same list form. No upstream issue has been filed: the remaining work is
in this repository, not a Bend defect.

## Run

```console
$ reference/.venv/bin/python reference/test_glm53_layers.py
$ make check        # runs it with the rest
```
