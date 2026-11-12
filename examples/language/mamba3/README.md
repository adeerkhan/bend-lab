# Mamba-3 SISO in Bend

One Mamba-3 SISO layer, the exact recurrent loop from the vendored Triton
kernel, with a hand-written backward. The port is correct and gated; it is a
float32 CPU reference, not a fast kernel.

Shapes: `d_model 64`, `d_state 16`, `expand 2`, `headdim 32` (4 heads),
`ngroups 1`, `rope_fraction 0.5` (4 angle pairs), batch 1, chunk 64. The
reference's `headdim 16` with `expand 2` would give eight heads; `headdim 32`
gives the four heads the rest of the work assumes.

| | |
| --- | --- |
| correctness | forward `9.2e-07`, backward `2.9e-06` relative against the F32 twin |
| speed, one block, T=8 | `19-22 ms` forward (the chunked array scan); `440.8 ms` forward+backward as measured on the list scan (the backward is still list) |
| ten-step copy task | `5.77 s` native, `0.048 s` PyTorch CPU twin, same loss and tokens |
| sixteen-token generation | exact match; `1.17 s` native, `0.039 s` PyTorch CPU twin |
| full two-layer LM | forward, ten-step training and generation exact; `17.68 s` native against `0.145 s` CPU twin |
| bend-ml Mat product | `65x` the list `q.S^T` product: `0.04 s` against `2.59 s` for 20,000 |
| block parameters | `27,816`; the full LM is `56,720` |

The full comparison, with the measurement boundaries and the lane caveats, is
in [COMPARISON.md](COMPARISON.md). The next steps are in
[ROADMAP.md](ROADMAP.md).

## The block

`in_proj` gives `[z, x, B, C, dd_dt, dd_A, trap, angle]`. Then

    A = -heavy_tail(dd_A), clamped at -1e-4
    DT = softplus(dd_dt + dt_bias), ADT = A*DT
    trap = sigmoid(trap_raw), gamma = DT*trap
    shifted_t = DT_{t+1}*(1 - trap_{t+1}), scale = gamma + shifted
    angle_t = mod2pi(angle_{t-1} + tanh(angle_raw)*pi*DT_t)
    q = rope(C_norm + C_bias), k = rope(B_norm + B_bias)
    o_t = exp(ADT_t)*(q_t S_t^T) + (D + gamma_t*(q_t . k_t))*x_t
    S_{t+1} = exp(ADT_t)*S_t + x_t (x) (scale_t*k_t)
    y = (o * silu(z)) @ out_proj^T

B and C are RMS-normalized (eps 1e-5) before their biases. The `q_t . k_t`
skip uses the pre-rope biased vectors, exactly as the kernel computes it
before applying rotary. The exact loop stays in the tree as the reference; the
live path (`pass_c`) is the chunked array scan.

## The full model

The block above is one layer. The full model in `probe_mamba3_lm.bend` is two
independent blocks with a residual around each, a final RMSNorm and an untied
`8 x 64` head:

    u    = E[tokens]
    h1   = u + block1(u)
    h2   = h1 + block2(h1)
    hn   = rms_norm(h2)
    out  = hn @ Wh^T

The training target is the same token sequence (`0..7`), the loss is mean
cross-entropy, and the update is plain SGD. The generation prompt is
`[0, 1, 2, 3]`, with twelve greedy tokens after it. The full model is
`56,720` parameters and is compared with `reference/mamba3_lm_twin.py`.

The two details that are easy to get wrong:

- **The trapezoid has two halves.** `gamma = DT*trap` is the diagonal skip in
  `o_t`; the state update uses `scale = gamma + shifted`, where `shifted` is
  the *next* token's `DT*(1 - trap)`. The output uses `gamma`, the state uses
  `gamma + shifted`. A port that uses `gamma` in both places is wrong at the
  next step.
- **The angle is a rate, not a position.** The kernel takes the raw angle
  projection, multiplies `tanh(raw) * pi * DT` and cumsums it, then reduces
  modulo `2π`. The cumsum carries across chunks. `F32.mod` in Bend follows C
  `fmod`, so a negative dividend stays negative; the port wraps the result to
  `[0, 2π)` before storing it.

## Commands

The goldens and the twin's kernel check:

    PYTHONPATH=reference/mamba3 reference/.venv-rf/bin/python reference/mamba3_twin.py

The gates:

    reference/.venv-rf/bin/python reference/check_mamba3_bend.py

The comparison table:

    reference/.venv-rf/bin/python reference/mamba3_compare.py

`make mamba3` runs the gates; `make mamba3-bench` builds and runs the
20-forward benchmark and prints its checksum and wall time. `make check` runs
the gates through `reference/check_all.py` after the package proofs. The bins
are regenerated from the committed `reference/mamba3_golden.npz` by
`reference/mamba3_bins.py`, so the gate needs NumPy and `bend`, not torch.

The pieces:

| file | what it does |
| --- | --- |
| `examples/language/mamba3/mamba3_siso.bend` | the forward, backward, ten-step copy task and two-layer greedy generation |
| `probe_mamba3_bwd.bend` | pass 1 of the backward: the output gate and the scan |
| `probe_mamba3_grad.bend` | the full backward |
| `probe_mamba3_bad.bend` | the same backward with one moved `dy` |
| `probe_mamba3_train.bend` | ten SGD steps on the copy task |
| `probe_mamba3_gen.bend` | sixteen greedy tokens, two weight-tied layers |
| `probe_mamba3_n.bend` | the forward at seqlen 4 and 16 |
| `probe_mamba3_lm.bend` | the full two-layer LM: forward, ten SGD steps, generation |
| `bench_mamba3_fwd.bend`, `bench_mamba3_bwd.bend` | quiet one-call timing probes |
| `bench_mamba3_fwd_n.bend`, `bench_mamba3_bwd_n.bend` | 20 and 10 calls in one process, for the slope |
| `bench_qs_list.bend`, `bench_qs_mat.bend` | the `q.S^T` product, list versus `Mat.matmul_nt` |
| `COMPARISON.md` | Bend beside the twin, the kernel, the public module and the full LM |
| `GPU.md` | the BendTorch CUDA lane, the linear test, and why the scan is still CPU |
| `ROADMAP.md` | what to build next and what each step teaches |

## The lanes

| lane | what it is | evidence against the twin |
| --- | --- | --- |
| F32 twin | the explicit loop, the oracle | the goldens |
| Bend F32 | the port, CPU | forward `9.2e-07`, backward `2.9e-06` |
| kernel, F32 + IEEE | `TRITON_F32_DEFAULT=ieee` | `2.0e-07` on `out_v`, `4.0e-07` on the state |
| kernel, F32 + TF32 | the default `tl.dot` | about `1.6e-03` |
| public module | bf16 Q/K/V/Trap/Angles/Z | `2.5e-03` |

The CPU scan keeps each head's 512-float state in one `Array<F32>` and updates
the 32x16 cells in place, instead of rebuilding 32 rows per token. Twenty
forwards at seqlen 8 are `0.37-0.38 s` against `0.71-0.73 s` for the list path,
and the ten-step train probe is `3.64-3.67 s` against `9.81-10.50 s`; every
gate still passes. The chunked scan then carries one 2048-float array across
all four heads and all tokens: in a same-session A/B (`scripts/mamba3_ab.sh`),
twenty forwards are `0.41-0.42 s` against `0.53-0.55 s` for the per-head array
core (medians `0.41 s` against `0.54 s`; a second session gave `0.37-0.45` and
`0.49-0.56`), with the train probe equal within noise.

The public `Mamba3` module is a bf16 lane: `mamba3_siso_combined` casts Q, K,
V, Trap, Angles and Z before it calls the kernel. The F32 kernel check is not
the public module path; it calls `mamba3_siso_fwd` directly with float32
inputs. The Bend port is held against the F32 twin, not against the bf16
module.

## The gates

| gate | result |
| --- | --- |
| forward at seqlen 8, 18 tensors | max relative `9.2e-07` (gate `1e-05`) |
| forward at seqlen 4 / 16 | `7.4e-07` / `9.6e-07` |
| backward, every parameter and retained tensor | max relative `2.9e-06` (gate `1e-04`) |
| moved-`dy` can-fail | `du_bad` at `6.9e-07`; `du` changes |
| ten-step copy task | loss max relative `1.1e-07`; tokens `[3, 6, 0, 5, 0, 7, 0, 7]` |
| sixteen-token generation | `[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`, exact |
| full LM forward, `lm_hn` / `lm_logits` / loss | `1.03e-07` / `3.48e-07` / `8.85e-08` |
| full LM ten-step loss / tokens | `1.77e-07`; `[4, 2, 6, 3, 3, 5, 0, 4]`, exact |
| full LM generation | `[0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]`, exact |

The copy task is `tokens 0..7 in, the same tokens out`: an embedding table
`8 x 64`, the block, a `64 x 8` head, mean cross-entropy, plain SGD at
`lr=1e-3`. The single-block generation probe uses two weight-tied layers and
greedy `argmax`; the full LM uses two independent layers. The prompt is
`[0, 1, 2, 3]`; twelve more tokens are generated.

## What the comparison says

- The port is numerically right, with a wide margin on every gate. The
  backward is right to `2.9e-06` relative, not merely close on the loss.
- The bend-ml `Mat.matmul_nt` product is about `65x` the list `q.S^T` product
  in a 20,000-call test. That is the substrate lever; the full scan still has
  the state update and allocations around it.
- The port is slow: `25x` the CPU twin on the forward and `99x` on
  forward+backward. The cause is the list substrate and the scalar
  per-element work, not the recurrence. The chunked kernel exists to remove
  exactly that work. The full two-layer LM is about `122x` the CPU twin on
  training plus generation.
- The public module is the fastest lane (`0.596 ms` forward) and the least
  accurate (`2.5e-03`). Speed and accuracy move in opposite directions here.
- At this tiny shape the GPU is launch-bound: the CUDA twin is slower than
  the CPU twin (`3.27` against `1.48 ms` forward). The fused Triton module
  wins because it uses far fewer launches, not because the GPU is intrinsically
  faster at eight positions.
- The native C binary is `5.5` to `6.4 MB` resident. The `354264 KiB` quoted
  in an earlier version of this file was the `bend` compile-and-run process,
  which includes the compiler and checker; it was not the program's memory.
  The CUDA twin's live allocation is `17.463 MB`; its `291.504 MB` reserved
  peak is the caching allocator's high-water mark, not live tensor memory.

## What is not here

MIMO, `ngroups > 1`, `rope_fraction 1.0`, `is_outproj_norm=True`, varlen,
batch greater than one, the 25.7M model, the Triton backward and Mamba-3 on
the BendTorch GPU lane. The BendTorch CUDA lane itself is tested on a linear
layer (0 differing words, `GPU.md`); the scan is still the CPU array code. The
full LM is two independent layers, but still one batch and eight positions. `COMPARISON.md` keeps those out of the measured table, and
`ROADMAP.md` puts them in an order.
