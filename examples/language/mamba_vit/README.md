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
   the Mamba-3 port uses (a scan backward that runs t backwards). Green:
   `make mamba-vit-bwd`, 120 gradients, worst rel `5.7e-06`.
3. **Training parity** — train the same config on MNIST in Bend and compare
   the held-out accuracy against the twin. Green on a fixed subset at a
   budget this lane can hold: `make mamba-vit-train`, `MAMBA-VIT TRAIN PASS`
   (see "A3" below). The twin's 94.61 % after one 60,000-image epoch is
   **unavailable** on Bend: the measured run (2026-11-15, `make mamba-vit-train`)
   is 1869.8 s wall for 160 training + 256 held-out images (4.49 s per image),
   so a full epoch is on the order of days and the fixed 2,048-image subset
   on the order of hours.

The comparison to the reference is therefore two-layered and honest: the
model is theirs (its math, shapes and init), the oracle is our twin (which
we verified learns as they claim), and the port is judged against the twin,
not against a number copied from a README.

## The reference dump and the Bend forward probe

`reference/mamba_vit_bins.py` writes, for a fixed MNIST image and the
reference's default init, every parameter and every intermediate of the
8-layer forward into `.scratch/mamba_vit/*.bin` (130 files): the image, the
patch tokens, each block's output and its accumulated residual, the final
norm, the pooled vector and the logits. It also writes the A2 backward's
oracle (121 more files): `dy`, the ten values that fix the scalar loss
`loss = sum(logits * dy)`, and the gradient of that scalar with respect to
every parameter (`g_<the same name as the weight>`) and to the intermediates
the probes compare (`g_emb`, `g_tokens`, `g_block0..7`, `g_final`,
`g_pooled`). `dy` is drawn from its own generator after the forward, so the
130 forward files are byte-identical to the ones A1a/A1b were gated on.
Command:

```
reference/.venv-rf/bin/python reference/mamba_vit_bins.py
```

The probe `examples/language/mamba_vit/probe_mamba_vit_emb.bend` loads those
files and prints, under `mv_*` tags, `emb`, `tokens`, `block0..7`,
`block0..7_res`, `final`, `pooled`, `logits`; `reference/test_mamba_vit.py`
compares them at 1e-5. The file keeps the `_emb` name it was created with in
A1a: it now carries the whole forward.

The probe `examples/language/mamba_vit/probe_mamba_vit_bwd.bend` repeats the
forward (it needs the per-token intermediates the forward gate never printed)
and walks back through it, printing `mv_g_*` for every file the dump carries;
`reference/test_mamba_vit_bwd.py` (`make mamba-vit-bwd`) compares all 120 at
1e-4 relative.

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
| `mv_block0` | 1.7e-10 | **4.57e-07** |
| `mv_block2` | 1.5e-10 | 4.33e-07 |
| `mv_block3` | 1.5e-10 | 4.31e-07 |
| `mv_block5` | 1.5e-10 | 4.15e-07 |
| `mv_block6` | 1.5e-10 | 4.09e-07 |
| `mv_block4` | 1.5e-10 | 3.91e-07 |
| `mv_block1` | 1.2e-10 | 3.69e-07 |
| `mv_block7` | 1.2e-10 | 3.55e-07 |
| `mv_emb` | 8.3e-07 | 2.84e-07 |
| `mv_pooled` | 7.5e-08 | 2.37e-07 |
| `mv_final` | 6.9e-07 | 2.08e-07 |
| `mv_block0..7_res` | 8.3e-07 | 1.99e-07 |
| `mv_tokens` | 8.3e-07 | 1.99e-07 |
| `mv_logits` | 5.6e-09 | 1.78e-07 |

The table is the run after A2 fixed the scan's slot order (below); the A1b
commit's worst tag was `block5` at `2.16e-06`, so the fix improved every
block row about four times and left the rest unchanged.

The residual and token rows carry the largest absolute difference because
they are the only lists of order one (`|tokens|` max `4.20`); the mixer
outputs are `4e-4`, so their absolute error is four orders down. Every row is
under the 1e-5 gate with more than twenty times the margin at the worst tag.

## A2 is green: the backward at 1e-4

`make mamba-vit-bwd` PASS: 120 gradients (108 parameter tensors and the
twelve intermediates `g_emb`, `g_tokens`, `g_block0..7`, `g_final`,
`g_pooled`), all under 1e-4 relative against PyTorch autograd. Measured on
this lane (`reference/.venv/bin/python reference/test_mamba_vit_bwd.py`),
worst first:

| tag | max abs diff | max of the reference | relative |
|---|---:|---:|---:|
| `g_layers_4_mixer_A_log` | 5.0e-19 | 8.8e-14 | **5.67e-06** |
| `g_layers_1_mixer_x_proj_weight` | 4.7e-17 | 1.5e-11 | 3.20e-06 |
| `g_layers_4_mixer_x_proj_weight` | 3.0e-17 | 1.0e-11 | 2.92e-06 |
| `g_layers_1_mixer_A_log` | 7.5e-19 | 3.1e-13 | 2.41e-06 |
| `g_layers_5_mixer_dt_proj_weight` | 7.4e-22 | 3.6e-16 | 2.08e-06 |
| `g_layers_6_mixer_dt_bias` | 9.7e-19 | 5.0e-13 | 1.96e-06 |
| `g_layers_7_mixer_dt_proj_bias` | 4.0e-19 | 2.2e-13 | 1.83e-06 |
| `g_to_patch_embedding_1_bias` | 3.8e-06 | 2.6e+01 | 1.47e-07 |
| `g_mlp_head_weight` | 1.8e-07 | 6.9e-01 | 2.61e-07 |
| `g_pooled` | 7.5e-09 | 1.1e-01 | 6.80e-08 |
| `g_mlp_head_bias` | 0 | 2.2e+00 | 0 |

The median of the 120 is `3.8e-07`; 22 are above `1e-06` and all of those are
the gradients of the SSM parameters (`A_log`, `x_proj`, `dt_proj`, `dt_bias`),
whose true values are `1e-11` to `1e-15` because `dt` is of order `1e-3` and
the state of order `1e-7` in this initialisation. A relative gate on a
quantity of `1e-15` is a gate on the ordering of the same float32 operations
on both sides, not on the magnitude: `g_layers_5_mixer_dt_proj_weight` agrees
at `2.1e-06` while its largest entry is `3.6e-16`.

## The scan's slot order was reversed, and only the backward could see it

`upd_c` builds the 64x16 state by prepending each channel's 16-slot row and
then reversing the whole list. Two reversals cancel, so each row came out in
the reverse of `A`/`B`/`C`'s slot order: the state was correct as a model
(the recurrence and the `y = <C, h>` read used the same reversed layout) but
not as the reference's model. The forward gate could not see it because in
this initialisation the state contributes about `1e-7` of `y` (`D * x`
carries the rest), which is the same order as the gate's own slack. The
backward could: `dC`, `dA` and `d(dt)` are the three gradients that read the
state, and all three failed (`g_x_proj`'s `C` rows were 100 % off, the
`A_log` rows 90 %) while `dB`, `dD`, `in_proj`, `out_proj`, `conv1d` and the
norms matched. Deleting the inner reverse in `upd_c` fixed the layout, made
the stored state match a NumPy replica at `1.6e-06` relative, and improved
the forward's block tags fourfold: the backward is a sharper test of a
recurrent forward than the forward's own outputs.

## What the block and its backward cost, in Bend idioms

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
   `upd_c`/`upd_q`: each channel's 16-slot row is prepended and the whole
   1024-float list is reversed once at the end of the token (a `List.append`
   per row is quadratic and was removed). The `Array<F32> & F32` threading of
   `mamba3_siso.bend`'s `row_set` is not needed: its look-ahead exists so a
   linear array can be updated in place, and the list form has no such
   constraint. Two reversals cancel *inside* a row as well as across rows:
   `upd_q` builds each row backwards, so the outer reverse of the whole list
   is what puts the rows in channel order, and a second reverse per row
   reverses the 16 slots (see the slot-order section above).
6. **A missing activation is invisible in the shapes.** The first run matched
   `h`, `xz` and the conv exactly and was garbage from `x_proj` on: the SiLU
   after the conv (`self.act(self.conv1d(...))`) had been folded into the
   wrong place. Stage dumping (`mv_h`, `mv_xz`, `mv_cv`, `mv_dbl`,
   `mv_state`, `mv_y`) against a NumPy replica of the same formulas found it
   in one run; the shapes and the parameter count cannot.
7. **Running the scan backwards costs one pass, not one pass per gradient.**
   At token `t` the whole per-token chain is local: `out_proj`, the SiLU
   gate, the scan step, `dt = softplus`, `x_proj` and the conv all read
   token `t`'s stored row plus the carry `G = dL/dh_t`. So one reverse loop
   accumulates the block's ten per-token parameter gradients (up to 4096
   floats each) and the per-token `d_xz` rows in the same walk; only the
   block LayerNorm, a within-row reduction, needs a second pass, and it adds
   the last two tensors. The conv's three padded taps need `d_y[t+1..t+3]`, which the
   same reverse order has already computed: the loop carries the last three
   `d_y` rows (fields `d1`, `d2`, `d3` of the carry record).
8. **A record can be taken apart once, so a probe hands it on.** The block's
   forward returns one 11-field record (`BF`: `h`, `xz`, the conv output,
   `xs`, `dbl`, `dt`, `z`, `y`, the gated `yg`, the 50 states, the output)
   and only a `def` that matches it can read it; the same value cannot be
   matched twice, so `bf_out(bf)` and `blk_bwd(bf, ...)` are two separate
   consumers of two separate values, and the tensors are converted from
   `TA.Mat` to lists once and passed to both as `+` parameters. A record is
   a `Type`, so `+bf : BF <- IO.pure(BF, ...)` is refused (`+bf can be used
   many times, so its type must be Data`); a field that is read twice needs
   `+field` in the pattern (`... hs (consumed more than once)` otherwise).
9. **No forward reference, not even for the small helper.** `emb_go` calling
   `emb_step` before its definition is `a filled definition (an unfilled law
   is a dead claim: live code cannot use it)`; the step has to come first.
10. **A relative gate on a tiny gradient is a gate on the operation order.**
   `dt = softplus(...)` is of order `1e-3` in this initialisation, so the
   state is `1e-7` and the gradients of `A_log`, `x_proj`, `dt_proj` and
   `dt_bias` land between `1e-11` and `1e-16`. They pass at `1e-06`
   relative only because both sides perform the same float32 operations in
   the same order; the absolute agreement is `1e-17` to `1e-22`. A stage
   that reorders a sum, e.g. folding `dt * B * x` differently, shows up as a
   percentage on these four tensors and nowhere else.


## A3 is green: subset training parity with the twin

`make mamba-vit-train` PASS (`reference/test_mamba_vit_train.py`,
`MAMBA-VIT TRAIN PASS`). `examples/language/mamba_vit/train_mamba_vit.bend`
is the trainer: the A1b forward and the A2 backward, one flat 83,050-value
parameter list in `named_parameters()` order, cross entropy with the
mean-over-batch gradient, clipping by global norm at 1.0 and AdamW in torch's
order (lr 1e-3, weight_decay 0.01, betas 0.9/0.999, eps 1e-8, bias
correction, decoupled decay), batch 32 in file order, constant lr. The twin
(`reference/mamba_vit_twin_train.py`, `MambaViT` from `mamba_vit_twin.py` in
float64 on the CPU) runs the same data, the same init and the same schedule,
and `twin.json` records the oracle.

The data is `.scratch/mv_mnist/` (`reference/mamba_vit_train_data.py`): the
first 2,048 MNIST training images and the first 512 test images as 0..1
float32 (`ToTensor`, not the A1b dump's centered image), the seed-0 init as
108 per-tensor files plus one flat file, and a manifest with every sha256 and
the flat offset of every tensor.

**The budget, and why it is a subset.** Measured on this lane 2026-11-15
(`reference/test_mamba_vit_train.py`): 1869.8 s wall for the 160 training +
256 held-out images (4.49 s per image), so the fixed 2,048-image subset is
on the order of hours and a 60,000-image epoch on the order of days. The
run therefore trains on the first 160 of the 2,048 images (5
steps of 32) and evaluates the first 256 held-out images, and the gate checks
parity at that budget: **the twin's 94.61 % after one full epoch is
`unavailable` on Bend**, and the accuracy below is what 5 steps at lr 1e-3
reach, not a claim on the epoch. The gate prints this in its own output.

| check | Bend | twin (float64) | gate |
|---|---|---|---|
| loss, every cstep | same curve | same curve | 1e-3 relative |
| gradient norm, every step | clipped at 1.0 | clipped at 1.0 | 1e-3 relative |
| train hits (pre-update) | 25/160 | 25/160 | 2 points |
| held-out hits | 35/256 | 35/256 | 2 points |
| final weights (108 sums of squares) | — | — | 1e-3 relative |

Two things the gates could not see, both found by this run:

- **The patch embedding's linear bias.** A1b/A2 ran on the seed-0 dump,
  where `to_patch_embedding.1.bias` is zero, so their forward adds nothing
  and their backward's LayerNorm reads the unbiased `pre`. A3 updates that
  bias on the first step, and a LayerNorm over the 16 patch values is not
  shift-invariant per feature, so both the forward and the backward need the
  bias. The probe is unchanged (its dump still has the zero bias); the
  trainer adds it in both directions.
- **The flat gradient's head order.** The first version prepended the patch
  embedding's six tensors in front of `pos`/`cls` while the flat parameter
  list is `pos`, `cls`, then the patch embedding. The loss curve could not
  see a rotation inside the first 2,272 values (the weights move by 1e-3 in
  the wrong tenors yet the loss by ~1e-6), and the per-tensor checksums
  localized it in one run: six tensors off by 1e-2 to 1e-4, the other 102
  under 1e-6. The order is now the manifest's.
