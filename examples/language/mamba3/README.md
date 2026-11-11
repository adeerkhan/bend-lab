# Mamba-3 SISO in Bend

One Mamba-3 SISO layer, the exact recurrent loop from the vendored Triton
kernel, with a hand-written backward. Shapes: `d_model 64`, `d_state 16`,
`expand 2`, `headdim 32` (4 heads), `ngroups 1`, `rope_fraction 0.5`
(4 angle pairs), batch 1, chunk 64.

`reference/mamba3/` is `state-spaces/mamba` at
`e9594ce1c732d97440f0332fdc43170a2294dbfa`, Apache-2.0; see its
`VENDORED.md`. `reference/mamba3_twin.py` is the pure-PyTorch twin of the
kernel's math and writes the goldens and the flat F32 bins the probes load.

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

## Commands

The goldens and the twin's kernel check:

    PYTHONPATH=reference/mamba3 reference/.venv-rf/bin/python reference/mamba3_twin.py

The gates:

    reference/.venv-rf/bin/python reference/check_mamba3_bend.py

`make check` runs the same gates after the package proofs, through
`reference/check_all.py`. The bins are regenerated from the committed
`reference/mamba3_golden.npz` by `reference/mamba3_bins.py`, so the gate needs
NumPy and `bend`, not torch.

The pieces:

| file | what it does |
| --- | --- |
| `examples/language/mamba3/mamba3_siso.bend` | the forward, the backward, the ten-step copy task and the two-layer greedy generation |
| `examples/language/mamba3/probe_mamba3_bwd.bend` | pass 1 of the backward: the output gate and the scan |
| `examples/language/mamba3/probe_mamba3_grad.bend` | the full backward |
| `examples/language/mamba3/probe_mamba3_bad.bend` | the same backward with one moved `dy` |
| `examples/language/mamba3/probe_mamba3_train.bend` | ten SGD steps on the copy task |
| `examples/language/mamba3/probe_mamba3_gen.bend` | sixteen greedy tokens, two weight-tied layers |
| `examples/language/mamba3/probe_mamba3_n.bend` | the forward at seqlen 4 and 16 |

## The lanes

The public `Mamba3` module is a bfloat16 lane: `mamba3_siso_combined` casts Q,
K, V, Trap, Angles and Z before it calls the kernel. Called directly with
float32 inputs and `TRITON_F32_DEFAULT=ieee`, the forward kernel agrees with
the F32 twin to `2.0e-07` relative on `out_v` and `4.0e-07` on the final
state; the public module differs from the twin by `2.5e-03` relative. The
Bend port is held against the F32 twin, not against the bf16 module.

## The gates

| gate | result |
| --- | --- |
| forward at seqlen 8, 18 tensors | max relative `9.2e-07` (gate `1e-05`) |
| forward at seqlen 4 / 16 | `7.4e-07` / `9.6e-07` |
| backward, every parameter and retained tensor | max relative `2.9e-06` (gate `1e-04`) |
| moved-`dy` can-fail | `du_bad` matches its golden at `6.9e-07`; `du` changes |
| ten-step copy task | loss max relative `1.1e-07`; tokens `[3, 6, 0, 5, 0, 7, 0, 7]` |
| sixteen-token generation | `[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`, exact |

The ten-step run processed 80 tokens in `4.20 s` wall in the gate
(`4.58 s` for the process under `/usr/bin/time -v`), with peak RSS
`354264 KiB`. The generation probe ran in `1.6 s` with peak RSS
`278456 KiB`. Those two numbers are from the commands above on the RTX 5060
(sm_120), with Triton 3.6.0 and torch in `reference/.venv-rf`.

The copy task is `tokens 0..7 in, the same tokens out`: an embedding table
`8 x 64`, the block, a `64 x 8` head, mean cross-entropy, plain SGD at
`lr=1e-3`. Generation uses the trained weights with two weight-tied layers and
greedy `argmax`.
