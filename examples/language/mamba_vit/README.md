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

## The reference dump (done) and the Bend forward probe (next)

`reference/mamba_vit_bins.py` writes, for a fixed MNIST image and the
reference's default init, every parameter and every intermediate of the
8-layer forward into `.scratch/mamba_vit/*.bin` (130 files): the image, the
patch tokens, each block's output and its accumulated residual, the final
norm, the pooled vector and the logits. Command:

```
reference/.venv-rf/bin/python reference/mamba_vit_bins.py
```

The probe `examples/language/mamba_vit/probe_mamba_vit.bend` must load those
files and print, under `mv_*` tags, `emb`, `tokens`, `block0..7`,
`block0..7_res`, `final`, `pooled`, `logits` for comparison at 1e-5.

The shapes to load (all row-major float32) and the helpers they need:

| file | shape | helper |
|---|---|---|
| `img` | 784 | one row |
| `w_to_patch_embedding_0_weight` / `_bias` | 16 / 16 | 1×16 |
| `w_to_patch_embedding_1_weight` / `_bias` | 32×16 / 32 | 32×16, 32 |
| `w_to_patch_embedding_2_weight` / `_bias` | 32 / 32 | 1×32 |
| `w_cls_token` | 1×1×32 | 32 |
| `w_pos_embedding` | 50×32 | 50×32 |
| `w_layers_N_norm_weight` / `_bias` | 32 / 32 | 1×32 |
| `w_layers_N_mixer_in_proj_weight` | 128×32 | 128×32 |
| `w_layers_N_mixer_conv1d_weight` | 64×1×4 | 64×4 (squeeze the group axis) |
| `w_layers_N_mixer_conv1d_bias` | 64 | 1×64 |
| `w_layers_N_mixer_x_proj_weight` | 34×64 | 34×64 |
| `w_layers_N_mixer_dt_proj_weight` / `_bias` | 64×2 / 64 | 64×2, 1×64 |
| `w_layers_N_mixer_A_log` | 64×16 | 64×16 |
| `w_layers_N_mixer_D` | 64 | 1×64 |
| `w_layers_N_mixer_out_proj_weight` | 32×64 | 32×64 |
| `w_norm_f_weight` / `_bias` | 32 / 32 | 1×32 |
| `w_mlp_head_weight` / `_bias` | 10×32 / 10 | 10×32, 1×10 |

The block math (dim 32, d_inner 64, dt_rank 2, d_state 16, d_conv 4):

1. `res = x + res` (res is the running sum), `h = LN(res)` per token;
2. `xz = in_proj(h)` (128), `z = xz[64:]`, `x = silu(conv1d(xz[:64]))` with
   the causal conv over the 49 tokens (pad three zero tokens in front, then
   `y[t,c] = bias[c] + sum_k w[c,k]*xp[t+k,c]`);
3. `dbl = x_proj(x)` (34): `dt_raw = dbl[0:2]`, `B = dbl[2:18]`, `C = dbl[18:34]`;
   `dt[c] = softplus(dot(dt_proj_w[c], dt_raw) + dt_proj_b[c] + dt_bias[c])`
   (note: `dt_bias` is in the twin's `w_layers_N_mixer_dt_bias`);
4. `A[c,q] = -exp(A_log[c,q])`; scan `h[c,q] = exp(dt[c]*A[c,q])*h + dt[c]*B[q]*x[c]`,
   `y[c] = sum_q h[c,q]*C[q] + D[c]*x[c]`;
5. `y = y * silu(z)`, `out = out_proj(y)` (32); the block returns `(out, res)`.

The scan is 64×16 floats in one `Array<F32>` per layer, stepped 49 times —
the same shape the Mamba-3 port's `pass_c` uses, without its trapezoid.

## A1a is green; A1b's shape (the blocks)

`make mamba-vit-emb` PASS: `emb` rel `2.8e-07`, `tokens` rel `2.0e-07`
(commit `00c22aa`). The remaining A1b is the eight blocks, the final norm,
the mean pool and the head. Two implementation notes that save a rewrite:

1. **Load each block inside an IO recursion.** A `TA.Mat` is consumed once
   and there are 14 tensors per block, so `run_blocks(+left, +i, x, res)`
   loads block `i`'s files (paths from `Nat.show(i)`), computes, prints
   `blockN` / `blockN_res`, and recurses with `i+1` on the new lists. Eleven
   list values never need to travel as parameters.
2. **The scan threads `Array<F32> & F32` the way `row_set` does.** Per token
   and per inner channel `c`, `scan_q` walks the 16 state slots with
   `Array.get` passed as the next `cur`, `Array.set` for the update, and the
   dot with `C` accumulated into a `RA{h, acc}` record; `scan_head` carries
   `RA` out and recurses to `c+1`. The record match is legal because `RA` is
   a parameter of the continuation helper, not a local.

The three lists the block needs per token are `x` (49x64, `silu(conv1d)`),
`dt` (49x64, `softplus`), `B`/`C` (49x16) and the constants `A` (64x16),
`D` (64), plus `z` (49x64) for the gate.
