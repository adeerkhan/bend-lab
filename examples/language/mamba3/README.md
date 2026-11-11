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
| speed, one block, T=8 | `37.2 ms` forward, `440.8 ms` forward+backward, native C, CPU |
| ten-step copy task | `5.77 s` native, `0.048 s` PyTorch CPU twin, same loss and tokens |
| sixteen-token generation | exact match; `1.17 s` native, `0.039 s` PyTorch CPU twin |
| block parameters | `27,816`, plus a `512` embedding and a `512` head for the tiny LM |

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
before applying rotary. The scan is the loop, not a chunked or associative
form.

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

`make mamba3` runs the gates. `make check` runs them through
`reference/check_all.py` after the package proofs. The bins are regenerated
from the committed `reference/mamba3_golden.npz` by
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
| `bench_mamba3_fwd.bend`, `bench_mamba3_bwd.bend` | quiet one-call timing probes |
| `bench_mamba3_fwd_n.bend`, `bench_mamba3_bwd_n.bend` | 20 and 10 calls in one process, for the slope |
| `COMPARISON.md` | Bend beside the twin, the kernel and the public module |
| `ROADMAP.md` | what to build next and what each step teaches |

## The lanes

| lane | what it is | evidence against the twin |
| --- | --- | --- |
| F32 twin | the explicit loop, the oracle | the goldens |
| Bend F32 | the port, CPU | forward `9.2e-07`, backward `2.9e-06` |
| kernel, F32 + IEEE | `TRITON_F32_DEFAULT=ieee` | `2.0e-07` on `out_v`, `4.0e-07` on the state |
| kernel, F32 + TF32 | the default `tl.dot` | about `1.6e-03` |
| public module | bf16 Q/K/V/Trap/Angles/Z | `2.5e-03` |

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

The copy task is `tokens 0..7 in, the same tokens out`: an embedding table
`8 x 64`, the block, a `64 x 8` head, mean cross-entropy, plain SGD at
`lr=1e-3`. Generation uses the trained weights with two weight-tied layers and
greedy `argmax`. The prompt is `[0, 1, 2, 3]`; twelve more tokens are
generated.

## What the comparison says

- The port is numerically right, with a wide margin on every gate. The
  backward is right to `2.9e-06` relative, not merely close on the loss.
- The port is slow: `25x` the CPU twin on the forward and `99x` on
  forward+backward. The cause is the list substrate and the scalar
  per-element work, not the recurrence. The chunked kernel exists to remove
  exactly that work.
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
batch greater than one, independent two-layer weights, the 25.7M model, the
Triton backward and a Bend GPU lane. The port is one layer, one batch, eight
positions, one CPU. `COMPARISON.md` keeps those out of the measured table, and
`ROADMAP.md` puts them in an order.
