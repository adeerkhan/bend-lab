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

**Stage B, linear attention.** `lin_core.bend` implements the reference's
`ELU+1`, causal prefix of `k` and `k⊗v`, numerator `q·KV`, denominator `q·K`
and the `1e-6` clamp, in list form; its forward matches the NumPy twin.
`lin_bwd.bend` is the exact backward, single head, direct O(T²d): with
`c_t = clamp(den_t, 1e-6)`, `dnum_t = dy_t/c_t`,
`dden_t = -dy_t·num_t/c_t²` and `a_{t,s} = dnum_t·v_s + dden_t`,
`dq_t = Σ_{s≤t} a_{t,s} k~_s`, `dk_s = Σ_{t≥s} a_{t,s} q~_t` and
`dv_s = Σ_{t≥s} dnum_t (q~_t·k~_s)`, then the ELU chain. Measured by
`reference/test_glm53_bwd.py` (12 gates, no PyTorch in the gate):

| check | result |
|---|---|
| t=2, one head, hd=8: y | relative **1.19e-07** |
| t=2: dq / dk / dv | **1.49e-06 / 4.52e-07 / 6.47e-08** |
| t=2, one-hot dout at t=0 | dq = dk = dv = **0** for every t>0 |
| t=64, one head: y | relative **1.51e-07** |
| t=64: dq / dk / dv | **1.19e-06 / 1.17e-06 / 2.75e-07** |

The one-hot test is the can-fail the old 64-step list implementation lacked;
it is local ISSUE-090, now closed. The 64-position golden is
`reference/glm53_linear_bwd_golden.npz`, written by
`reference/glm53_linear_bwd.py` (PyTorch autograd on the same four files).

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
| loss at every step | 1e-4 relative | max **5.599e-07** |
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
generate at positions it never trained on. It is **not** the reference 25.7M
model and it is **not** the corpus baseline below.

## The corpus-trained dense baseline

`model_128.bend` and `train_dense128.bend` move the same 74,144-parameter
stack to **128 positions** and a real corpus stream: batch 8, 400 steps,
**409,600 tokens**, AdamW at lr 3e-3. The evaluation is the same vendored
evaluator, 8 identities per family.

| check | batch 8 / 409,600 tokens | batch 32 / 1,638,400 tokens |
|---|---:|---:|
| loss first -> final | 5.574197 -> 0.32177755 | 5.5704083 -> 0.29012972 |
| last-100 mean | 0.3530558 | 0.3012568 |
| held-out loss | 0.30741644 | 0.27868035 |
| build / run | 53.786 s / 27.28 s | 75.499 s / 108.47 s |
| peak RSS | 157,436 KiB | 445,252 KiB |
| dev solved / tests | **61/64 / 186/192** | **64/64 / 192/192** |
| confirm solved / tests | **60/64 / 184/192** | **64/64 / 192/192** |

At 409,600 tokens the only failures are `square` completions returning
`x * 2` instead of `x * x` (5/8 dev, 4/8 confirm); the other seven families
are 8/8. At the reference's 400-step token budget (1.6M) every family is 8/8
in both splits, with no token-cap hits and 64/64 valid programs. The square
confusion was a data/compute effect. The 74k dense model is at the ceiling of
this synthetic corpus; the 0/64 above is the fixed-batch, out-of-context
result, not a baseline.

The evaluation manifest is frozen in
`reference/glm53_dense_eval_golden.json`; `make check` re-checks every prompt
against the reference tasks, every verdict against the reference evaluator,
the four summary counts, and the frozen 48-token sequence.

## Stage C: the integrated linear + sparse model

`attn_c.bend` holds the split, RoPE, the linear and sparse cores, and the
attention layers; `model_c.bend` is one batch, 64 positions, one head of width
8, **linear attention then sparse attention** (window 16, stride 16), each with
a SwiGLU hidden 32, a final RMSNorm and the tied head: **4,168 parameters**.
The sparse indices are exactly the reference's
`s <= t and (s % 16 == 0 or t - s < 16)`; its backward is the softmax backward
on the gathered scores with the scatter into dk/dv.

`reference/test_glm53_c.py` (7 gates, no PyTorch) runs **20 AdamW steps**
against `reference/glm53_c_twin.py`:

| check | tolerance | measured |
|---|---|---|
| loss at every step | 1e-4 relative | max **2.852e-07** |
| final weights (4,168) | 1e-3 relative | **3.705e-08** |
| final gradients (4,168) | 1e-3 relative | **1.487e-07** |

The loss falls 5.5484023 -> 5.4986544. Can-fail runs with one weight moved
(+2.0) and one input byte changed diverge at step 1 at **7.860e-03** and
**6.585e-04**. The record is `results/glm53-c/run.json`.

The layer cores are also gated directly, RoPE + each core on the same 64 x 8
inputs against PyTorch-autograd goldens, at **1.5e-07 .. 1.9e-06** relative
(`reference/test_glm53_bwd.py`, 28 checks).

`lin_fast.bend` is the prefix-state linear core: `K_t/KV_t` forward and
`N_s/Q_s` suffix backward. Against the direct core at 64, y is exact and
dq/dk/dv agree at **6.07e-07 / 4.40e-07 / 2.75e-07** relative; the fast-only
probe runs in **0.024 s** against the direct core's **11.4 s**. At 128
positions y is exact and dq/dk/dv agree at **3.27e-07 / 3.38e-07 / 4.23e-07**;
the direct run takes **73.8 s**. The integrated attention layers use it, which
took the 20-step Stage C test from 3 m 25 s to **37 s** without moving the twin
gate.

### A held-out family

The seen-family benchmark is saturated, so the next test removes one family from the
pretraining stream and evaluates it. With `reverse` held out, the same 74,144-parameter
model at batch 32, 128 positions and 1,638,400 tokens reaches loss **0.29021987**, held-out
**0.27888998**, and:

| split | solved | tests | reverse | seven seen families |
|---|---:|---:|---:|---:|
| dev | **56/64** | 168/192 | **0/8, 0/24** | 8/8, 24/24 each |
| confirm | **56/64** | 168/192 | **0/8, 0/24** | 8/8, 24/24 each |

The model memorises the seven templates and does not generalise to the unseen one. Run
**112.0 s**, peak RSS **440,836 KiB**; the build is shared with the batch-32 baseline. This is a
real headroom benchmark for architecture work; the seen-family 64/64 is not.

## Diagnostics: the mask and the learning canary

The 64-byte training batch has no ignored labels, so the masked-loss branch was untested until
`reference/test_glm53_mask.py` (4 checks). A 2 x 3 case with row 0 labelled and row 1 ignored
gives loss **1.2019429** (expected 1.20194285), a gradient row 0 of
`[-0.6994, 0.3322, 0.3672]`, and an ignored row that is **exactly zero**; the mean uses one valid
row, not two.

The 20-step Stage C parity gate is a correctness gate, not a learning run: it uses the
reference's 25.7M learning rate, 3e-4, on a 4,168-parameter model. The learning canary overfits
the first 16 labels at lr 0.03 for 50 steps (`results/glm53-c/canary-16.json`):

| quantity | Bend | PyTorch twin |
|---|---:|---:|
| final loss | **0.04192061** | **0.04192125** |
| loss at final weights | **0.03682779** | **0.03683004** |
| build | 22.869 s | - |
| run | 446.4 s | - |

Relative difference: **1.5e-05** on the final loss, **6.1e-05** at the final weights. The Bend
loss, backward, AdamW and data can drive a loss down, not only match a twin.

The twin's learning-rate sweep on the full 64-byte batch, 400 steps:

| lr | final loss |
|---:|---:|
| 3e-4 | 3.955 |
| 1e-3 | 2.236 |
| 3e-3 | 0.175 |
| 1e-2 | 0.00263 |

On a 16-label overfit, 200 steps: 3e-4 **4.651**, 3e-3 **0.723**, 1e-2 **0.00286**,
3e-2 **0.00041**. The 3e-4 rate is the reference's; a small model needs a larger rate or many
more steps. This is the training variable, not the data, loss or init.

## The 4-block hybrid: implemented, corpus run blocked

`attn_mh.bend`, `model_h.bend`, `weights_h.bend` and `train_h128.bend` implement the 4-block
hybrid: three linear blocks and one sparse (`Lin,Lin,Lin,Sp`, the reference's every-fourth
schedule), width 32, four heads, 128 positions, SwiGLU 128, tied head, **74,144 parameters**.

The N=8 forward check (`reference/test_glm53_mh.py`, 4/4) is the important gate. It found an
8-wide/32-wide split bug and a `Lin,Sp,Lin,Lin` order before any training conclusion. After the
fixes, the four-head linear/sparse and the full four-block model match the NumPy twin at
**8.84e-08 / 1.23e-07 / 1.22e-07** relative. The NumPy twin itself matches the vendored torch
modules at 2e-08. One training step of the fixed model measures loss **5.5634737**, val
**5.4198546**, **18.97 s** for 128 tokens and **16,956 KiB** peak RSS.

The corpus run is blocked on throughput: 128 tokens a step makes the 1,638,400-token budget
**12,800 steps / 67.4 hours** (an extrapolation from the measured step, not a run). A 400-step
smoke would be 2.1 hours and 32x below the dense baseline's token count, so it is not a
comparable baseline. The list-based sparse backward (`List.set` scatter) and the per-head
gather/scatter dominate. The next step is the same one that took the linear core from 11.4 s to
0.024 s: a vectorized sparse core and a batched multi-head model, then the 192-position
evaluation at the 1.6M-token budget.

## What is not built

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
$ reference/.venv/bin/python reference/test_glm53_bwd.py
$ reference/.venv/bin/python reference/test_glm53_c.py
$ reference/.venv/bin/python reference/test_glm53_layers.py
$ make check
```
