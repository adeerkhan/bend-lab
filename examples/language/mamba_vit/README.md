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

## The reference dump and the Bend forward probe

`reference/mamba_vit_bins.py` writes, for a fixed MNIST image and the
reference's default init, every parameter and every intermediate of the
8-layer forward into `.scratch/mamba_vit/*.bin` (130 files): the image, the
patch tokens, each block's output and its accumulated residual, the final
norm, the pooled vector and the logits. Command:

```
reference/.venv-rf/bin/python reference/mamba_vit_bins.py
```

The probe `examples/language/mamba_vit/probe_mamba_vit_emb.bend` loads those
files and prints, under `mv_*` tags, `emb`, `tokens`, `block0..7`,
`block0..7_res`, `final`, `pooled`, `logits`; `reference/test_mamba_vit.py`
compares them at 1e-5. The file keeps the `_emb` name it was created with in
A1a: it now carries the whole forward.

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
| `w_layers_N_mixer_dt_bias` | 64 | 1×64 (mamba_ssm's `dt_bias`) |
| `w_layers_N_mixer_A_log` | 64×16 | 64×16 |
| `w_layers_N_mixer_D` | 64 | 1×64 |
| `w_layers_N_mixer_out_proj_weight` | 32×64 | 32×64 |
| `w_norm_f_weight` / `_bias` | 32 / 32 | 1×32 |
| `w_mlp_head_weight` / `_bias` | 10×32 / 10 | 10×32, 1×10 |

The block math (dim 32, d_inner 64, dt_rank 2, d_state 16, d_conv 4):

1. `res = x + res` (res is the running sum), `h = LN(res)` per token;
2. `xz = in_proj(h)` (128), `z = xz[64:]`, `x = silu(conv1d(xz[:64]))` with
   the causal conv over the 50 tokens (pad three zero tokens in front, then
   `y[t,c] = bias[c] + sum_k w[c,k]*xp[t+k,c]`);
3. `dbl = x_proj(x)` (34): `dt_raw = dbl[0:2]`, `B = dbl[2:18]`, `C = dbl[18:34]`;
   `dt[c] = softplus(dot(dt_proj_w[c], dt_raw) + dt_proj_b[c] + dt_bias[c])`
   (note: `dt_bias` is in the twin's `w_layers_N_mixer_dt_bias`);
4. `A[c,q] = -exp(A_log[c,q])`; scan `h[c,q] = exp(dt[c]*A[c,q])*h + dt[c]*B[q]*x[c]`,
   `y[c] = sum_q h[c,q]*C[q] + D[c]*x[c]`;
5. `y = y * silu(z)`, `out = out_proj(y)` (32); the block returns `(out, res)`.

The scan is 64×16 floats per layer, stepped 50 times. The implementation
keeps it as a flat `List<&2, F32>` of 1024 values rebuilt per token (the
measured run below is fast enough that the array was not needed; the reason
is in the notes at the end of this file).

## A1a and A1b are green

`make mamba-vit-emb` PASS: `emb` rel `2.8e-07`, `tokens` rel `2.0e-07`
(commit `00c22aa`).

`make mamba-vit-fwd` PASS: the eight blocks, the final norm, the mean pool
and the head, against the same dump, at 1e-5 relative. Measured on this lane
(`reference/.venv/bin/python reference/test_mamba_vit.py`), worst tag first:

| tag | max abs diff | relative |
|---|---:|---:|
| `mv_block5` | 7.6e-10 | **2.16e-06** |
| `mv_block1` | 5.2e-10 | 1.66e-06 |
| `mv_block7` | 2.9e-10 | 8.88e-07 |
| `mv_block2` | 2.6e-10 | 7.79e-07 |
| `mv_block3` | 2.3e-10 | 6.89e-07 |
| `mv_block6` | 2.3e-10 | 6.55e-07 |
| `mv_block0` | 2.5e-10 | 6.48e-07 |
| `mv_block4` | 2.4e-10 | 6.45e-07 |
| `mv_emb` | 8.3e-07 | 2.84e-07 |
| `mv_pooled` | 7.5e-08 | 2.37e-07 |
| `mv_final` | 6.9e-07 | 2.08e-07 |
| `mv_block0..7_res` | 8.3e-07 | 1.99e-07 |
| `mv_tokens` | 8.3e-07 | 1.99e-07 |
| `mv_logits` | 5.6e-09 | 1.78e-07 |

The residual and token rows carry the largest absolute difference because
they are the only lists of order one (`|tokens|` max `4.20`); the mixer
outputs are `4e-4`, so their absolute error is four orders down. Every row is
under the 1e-5 gate with more than four times the margin at the worst tag.

## What the block cost, in Bend idioms

1. **A computed value cannot be taken apart.** `match` inspects a parameter
   or a pattern variable only: neither `match f(x):` nor `r = f(x)` then
   `match r:` nor the destructuring let is accepted (`a match cannot
   scrutinize a computed value` / `... a local binder`). So no stage of the
   block returns two values: each pass folds a whole batch of rows into one
   flat list (the 50 tokens for `inproj_go` and `conv_go`, the 64 channels
   for `upd_c` and `dotc`), and the one record that does travel (`CVW`,
   `BSW`) is the token loop's own recursion state, matched by the next
   iteration's parameter.
2. **A `do` block holds effects only.** A pure `let` inside a `do` block is a
   parse error (`expected a pattern`); a value computed between two loads has
   to come back through `x : T <- IO.pure(T, v)`, and `+x : T <- IO.pure(...)`
   when it is used twice. The eight-block recursion needs three of them per
   block (`res2`, the conv columns, the block output).
3. **Load each block inside the IO recursion.** A `TA.Mat` is consumed once
   and there are 12 tensors per block, so `blocks_io(+left, +i, x, res, ...)`
   loads block `i`'s files (paths from `Nat.show(i)`), prints
   `blockN` / `blockN_res`, and recurses with `i+1`. Only the final norm and
   head weights travel as parameters.
4. **Three vectors per token, three flat passes.** `xz = in_proj(LN(res2))`
   stays 128 wide in one list, so the conv pass can read `x` and `z` out of it
   (`z = xz[64:]`, `x = silu(conv(xz[:64]))`) without a second record; the
   conv carries its last three `x` rows in `CVW`, which makes the three-zero
   padding implicit (`y[t] = cb + c3*x[t] + c2*x[t-1] + c1*x[t-2] + c0*x[t-3]`).
5. **Threading the state.** Rebuilding the 64×16 state as a list per token is
   `upd_c`/`upd_q`: rows are prepended in reverse and the whole 1024-float
   list is reversed once at the end of the token (a `List.append` per row is
   quadratic and was removed). The `Array<F32> & F32` threading of
   `mamba3_siso.bend`'s `row_set` is not needed: its look-ahead exists so a
   linear array can be updated in place, and the list form has no such
   constraint.
6. **A missing activation is invisible in the shapes.** The first run matched
   `h`, `xz` and the conv exactly and was garbage from `x_proj` on: the SiLU
   after the conv (`self.act(self.conv1d(...))`) had been folded into the
   wrong place. Stage dumping (`mv_h`, `mv_xz`, `mv_cv`, `mv_dbl`,
   `mv_state`, `mv_y`) against a NumPy replica of the same formulas found it
   in one run; the shapes and the parameter count cannot.

