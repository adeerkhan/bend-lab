# MambaViT — the reference, its twin on this lane, and the port plan

Reference: https://github.com/JLrumberger/MambaViT
(`src/mamba_vit.py`, commit `0236b08`). A ViT whose blocks are **Mamba-1**
(`mamba_ssm.Mamba` through `mixer_seq_simple.create_block`), not Mamba-3:
patch embedding (LayerNorm → Linear → LayerNorm), a cls token, learned
position embeddings, `n_layer` pre-norm blocks of
`residual += x; x = Mamba(LN(residual))`, one final LayerNorm, mean/cls pool
and a linear head.

## Why the reference cannot run here

The reference imports `mamba_ssm` (the Mamba-1 package) and `causal_conv1d`
for the depthwise causal convolution. Neither is installed in
`reference/.venv` or `reference/.venv-rf`, and the repo ships no source for
them (only the notebook and `src/mamba_vit.py`). So the reference's own code
cannot be executed on this lane as it stands.

## The twin (the oracle we do have)

`reference/mamba_vit_twin.py` reimplements the SAME model in plain torch:
the ViT shell above and the Mamba-1 block with `mamba_ssm`'s shapes and math
— `in_proj(d_model → 2·d_inner)`, a depthwise causal `conv1d(k=4)` + SiLU,
`x_proj(d_inner → dt_rank + 2·d_state)`, `dt_proj`, `dt = softplus(…)`,
`A = −exp(A_log)`, the per-timestep scan
`h_t = exp(dt·A) h_{t−1} + dt·B x_t`, `y = ⟨C, h_t⟩ + D x_t`, the SiLU gate
and `out_proj` — with `mamba_ssm`'s `dt_bias`/`A_log`/conv initialisation.

Measured on this lane (MNIST config: 28×28, patch 4, channels 1, dim 32,
`n_layer` 8, `d_state` 16, `d_conv` 4, expand 2, mean pool):

| run | params | result |
|---|---:|---|
| reference README's claim | — | ~90 % validation "in a couple of epochs" |
| twin, 1 epoch, batch 64, AdamW lr 1e-3, wd 0.01, clip 1.0, RTX 5060 | **83,050** | **val 94.61 %**, train 67.98 %, 4 m 05 s |

Run: `reference/.venv-rf/bin/python reference/mamba_vit_twin.py --epochs 1`
(downloads MNIST to `/tmp/opencode/mnist`). The twin is therefore a valid
stand-in: it is the same model, it learns, and it already matches the
reference's published behaviour.

## The port plan (one gate at a time, like the Mamba-3 rows)

1. **Patch embedding + Mamba-1 mixer, forward at 1e-5** — a Bend program
   `examples/language/mamba_vit/main.bend` reusing the lab's list/tensor
   helpers; gate `reference/test_mamba_vit.py` builds it, loads the twin's
   parameters, and compares every intermediate (patches, pos add, per-block
   norm/mixer output, final norm, logits) against `mamba_vit_twin.py`.
   The scan is the easy half: Mamba-1's recurrence has no trapezoid, no
   angle state and no B/C normalisation, so it is one `Array<F32>` of
   `d_inner × d_state` per layer plus `exp(dt·A)`.
2. **Backward at 1e-4** — hand-written through the conv1d, the causal
   scan, `dt = softplus`, and the gate; the same gradient-reversal pattern
   the Mamba-3 port uses (a scan backward that runs t backwards).
3. **Training parity** — train the same config on MNIST in Bend and compare
   the held-out accuracy against the twin (94.61 % after 1 epoch is the
   number to match), then a report card in `results/`.

The comparison to the reference is therefore two-layered and honest: the
model is theirs (its math, shapes and init), the oracle is our twin (which
we verified learns as they claim), and the port is judged against the twin,
not against a number copied from a README.
