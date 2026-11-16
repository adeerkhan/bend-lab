# Mamba-3 numerical lanes

The same SISO block, the same weights, the same 64-token input (the first
T rows, seed fixed), through four lanes.  This is a lane measurement,
never a verdict on any lane: the F32 twin is the oracle, and the numbers
below are properties of each lane, not of the math.

Measured 2026-11-16.  Command:
`reference/.venv-rf/bin/python reference/mamba3_lanes.py` (dump all four
lanes and print the tables), `reference/.venv/bin/python reference/test_mamba3_lanes.py`
(the gate).  The gate's `reference/mamba3_lanes.py gate` prints the
max-over-T per lane and asserts the bounds at the bottom.

Machine: NVIDIA GeForce RTX 5060 Laptop GPU (sm_120), driver 615.78.08,
torch 2.11.0+cu128, Triton 3.6.0 (no Bend in this path).  Headdim 32, d_state 16, 4 heads,
num_rope_angles 4, chunk_size 64.

## The lanes

| lane | what it is |
|---|---|
| `twin` | the F32 oracle, `reference/mamba3_twin.py`, explicit t-by-t loop, F32 everywhere |
| `ieee` | the vendored Triton kernel called directly (`mamba3_siso_fwd` low-level forward, then `compute_dqkv` / `compute_dqktheta` / `compute_ddt_dtrap_dinput_states` / `angle_dt_bwd` for the backward) with `TRITON_F32_DEFAULT=ieee`. IEEE `tl.dot` |
| `tf32` | the same calls with `TRITON_F32_DEFAULT=tf32` (the Triton default on this GPU). TF32 10-bit-mantissa `tl.dot` |
| `module` | the public `Mamba3` module on CUDA. Its wrapper casts Q/K/V/trap/angles to bfloat16 before the kernel (`mamba3_siso_combined.py:390-397`), so this lane is bf16. Reported, never the oracle |

Two things had to be true to run this honestly:

1. `mamba3_siso_combined` always casts to bf16, so the F32 kernel lane
   cannot go through it; it replicates `_Mamba3Function.backward` step by
   step with F32 inputs instead.
2. `TRITON_F32_DEFAULT` is read once at Triton import time
   (`triton/knobs.py:484`, consumed in `triton/language/core.py:2024`), so
   the ieee and tf32 lanes run as separate processes, one dump each.

Shared conditions: `torch.manual_seed(71)`, `torch.set_num_threads(1)`,
weights from `Mamba3(d_model=64, d_state=16, expand=2, headdim=32,
ngroups=1, is_mimo=False).cuda().float()` in seed order, input
`randn(1, 64, 64) * 0.5` (`torch.manual_seed(123)`), upstream gradient
`dy = randn(1, 64, 64) * 0.1` (`torch.Generator("cuda").manual_seed(9)`).

## Forward (max rel error vs the twin)

| T | ieee out_v | tf32 out_v | module out | ieee q | tf32 q | ieee k*sc | tf32 k*sc | ieee qk | tf32 qk | ieee scale | tf32 scale | ieee gamma | tf32 gamma | ieee states | tf32 states |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.842e-08 | 7.708e-05 | 3.642e-03 | 7.752e-08 | 7.752e-08 | 7.389e-08 | 7.389e-08 | 1.874e-07 | 3.023e-04 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 5.615e-08 | 8.057e-04 |
| 2 | 4.842e-08 | 4.883e-04 | 3.454e-03 | 3.585e-07 | 3.585e-07 | 1.910e-07 | 1.910e-07 | 0.000e+00 | 3.023e-04 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.225e-07 | 8.960e-04 |
| 4 | 9.365e-08 | 6.602e-04 | 5.436e-03 | 3.549e-07 | 3.549e-07 | 5.125e-07 | 5.125e-07 | 1.874e-07 | 4.412e-04 | 1.194e-07 | 1.194e-07 | 9.747e-08 | 9.747e-08 | 4.816e-07 | 7.440e-04 |
| 8 | 1.224e-07 | 1.258e-03 | 3.976e-03 | 7.675e-07 | 7.675e-07 | 5.370e-07 | 5.370e-07 | 1.474e-07 | 4.533e-04 | 9.680e-08 | 9.680e-08 | 8.903e-08 | 8.903e-08 | 3.865e-07 | 7.270e-04 |
| 16 | 2.496e-07 | 1.297e-03 | 3.985e-03 | 7.245e-07 | 7.245e-07 | 5.540e-07 | 5.540e-07 | 1.022e-07 | 4.670e-04 | 9.060e-08 | 9.060e-08 | 8.903e-08 | 8.903e-08 | 5.599e-07 | 7.894e-04 |
| 64 | 4.049e-07 | 2.101e-03 | 6.495e-03 | 1.971e-06 | 1.971e-06 | 1.761e-06 | 1.761e-06 | 1.173e-07 | 4.433e-04 | 8.472e-08 | 8.472e-08 | 6.009e-08 | 6.009e-08 | 2.490e-06 | 7.593e-04 |

`k*sc` is `rope(K + bias) * scale` — what the kernel's `K_store` holds
(`mamba3_siso_fwd.py`: `k_pre_block *= scale[:, None]` before the store;
`K_store` is fed to `tl.dot` with Q).  The twin stores unscaled `k`; the
comparison applies the twin's own `scale`.

## Kernel backward (max rel error of the kernel's input grads vs the twin's retained grads)

Both lanes compute `dL/d(inputs)` for the same scalar
`L = sum(y * dy)`; with `y = (out_v * silu(z)) @ W_out^T` the kernel's
incoming gradient is `do = silu(z) * (dy @ W_out)` (headdim-major,
(1, T, 4, 32)), which is what the dumps pass to `compute_dqkv`.

| T | ieee dQ | tf32 dQ | ieee dK | tf32 dK | ieee dADT | tf32 dADT | ieee dDT* | tf32 dDT* | ieee dTrap | tf32 dTrap | ieee dAngles | tf32 dAngles | ieee dD | tf32 dD | ieee dQ_bias | tf32 dQ_bias | ieee dK_bias | tf32 dK_bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.588e-07 | 2.930e-04 | 3.479e-07 | 2.929e-04 | 0.000e+00 | 0.000e+00 | 1.833e-07 | 2.614e-04 | 1.854e-07 | 2.614e-04 | 0.000e+00 | 0.000e+00 | 1.312e-07 | 7.291e-04 | 1.134e-07 | 2.615e-04 | 1.905e-07 | 2.614e-04 |
| 2 | 1.612e-07 | 1.050e-03 | 2.513e-07 | 7.238e-04 | 2.537e-07 | 1.407e-03 | 1.123e-07 | 7.446e-04 | 2.574e-07 | 6.364e-04 | 8.875e-07 | 1.111e-03 | 7.643e-08 | 9.786e-04 | 1.341e-07 | 6.407e-04 | 1.240e-07 | 5.493e-04 |
| 4 | 2.199e-07 | 1.010e-03 | 3.897e-07 | 1.377e-03 | 4.912e-07 | 1.608e-03 | 6.917e-08 | 8.936e-04 | 3.417e-07 | 7.566e-04 | 2.844e-07 | 1.027e-03 | 2.352e-07 | 6.715e-04 | 1.351e-07 | 9.875e-04 | 1.942e-07 | 1.057e-03 |
| 8 | 2.829e-07 | 1.010e-03 | 5.785e-07 | 1.639e-03 | 1.125e-07 | 1.661e-03 | 2.489e-07 | 1.424e-03 | 2.241e-07 | 1.796e-03 | 3.980e-07 | 1.767e-03 | 1.804e-07 | 9.750e-04 | 3.774e-07 | 9.773e-04 | 2.167e-07 | 9.602e-04 |
| 16 | 4.596e-07 | 1.010e-03 | 9.429e-07 | 1.624e-03 | 3.759e-07 | 1.540e-03 | 2.140e-07 | 1.285e-03 | 1.928e-07 | 1.746e-03 | 6.883e-07 | 2.392e-03 | 1.179e-07 | 9.850e-04 | 3.020e-07 | 1.154e-03 | 3.828e-07 | 1.504e-03 |
| 64 | 7.086e-07 | 1.332e-03 | 9.404e-07 | 1.459e-03 | 3.343e-07 | 1.405e-03 | 4.070e-07 | 1.317e-03 | 3.290e-07 | 1.136e-03 | 2.560e-06 | 1.850e-03 | 1.408e-07 | 8.155e-04 | 5.894e-07 | 1.406e-03 | 1.115e-06 | 1.474e-03 |

\* `dDT` compares only the direct part of the twin's DT gradient; the
twin's `DT` gradient also carries the `ADT = A * DT` path, which the
kernel returns separately as `dADT`.  `g_DT_direct = g_DT - A * g_ADT`.
Without this, `dDT` appears to be off by ~5e-02–2e-01 and that is an
accounting difference, not a precision one.

The kernel's SSM states are stored in bfloat16 by contract
(`mamba3_siso_fwd.py:604`: `SSM_States = torch.zeros(..., dtype=torch.bfloat16)`),
in both kernel lanes.  The dumps cast them back to F32 for the backward
dotes; the bf16 rounding is baked into every `ieee`/`tf32` number above.
This is the only non-IEEE rounding in the `ieee` lane.

The rotary angle path: the kernel consumes **raw** angles, computes the
tanh- and dt-weighted cumsum internally, and does not store it; the
angle cumsum itself matches the twin to 9.5e-07 (checked separately).
`TF32` does not change the elementwise rotation either (no `tl.dot` in
it), which is why `tf32 q` equals `ieee q` exactly here.

## Module backward, public bf16 module (max rel error vs the twin's whole-module grads)

The same scalar `L`, backpropagated through `m(u)` (bf16 internally)
against the twin's F32 whole-module backward.

| T | du | dW_in | ddt_bias | dB_bias | dC_bias | dB_norm_w | dC_norm_w | dD | dW_out |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.995e-03 | 3.597e-03 | 2.245e-03 | 2.992e-03 | 2.033e-03 | 2.623e-03 | 3.352e-03 | 2.044e-03 | 5.606e-03 |
| 2 | 2.223e-03 | 1.487e-03 | 5.800e-03 | 5.160e-03 | 7.347e-03 | 5.838e-03 | 5.141e-03 | 5.502e-03 | 5.630e-03 |
| 4 | 3.719e-03 | 3.021e-03 | 6.680e-03 | 8.681e-03 | 8.854e-03 | 1.409e-02 | 6.608e-03 | 6.447e-03 | 3.916e-03 |
| 8 | 3.827e-03 | 3.110e-03 | 8.090e-03 | 9.077e-03 | 1.209e-02 | 2.396e-02 | 2.167e-02 | 4.613e-03 | 8.051e-03 |
| 16 | 3.630e-03 | 2.679e-03 | 5.337e-03 | 1.420e-02 | 9.269e-03 | 1.085e-02 | 1.132e-02 | 2.176e-03 | 5.705e-03 |
| 64 | 3.448e-03 | 5.410e-03 | 3.384e-03 | 1.114e-02 | 4.313e-03 | 5.739e-03 | 5.559e-03 | 3.759e-03 | 5.158e-03 |

## Summary (max over T = 1, 2, 4, 8, 16, 64)

| lane | max rel over all tensors |
|---|---|
| ieee (F32 kernel, IEEE dots) | 2.560e-06 (reruns observe 2.49e-06; the kernel autotunes) |
| tf32 (F32 kernel, TF32 dots) | 2.392e-03 |
| module (public bf16 module) | 2.396e-02 |

`unavailable`: the kernel does not store its rotary-angle cumsum, so
there is no `angle` cell; the kernel traps and the twin's bf16 module do
not expose per-head intermediates beyond what is listed.

## The angles path, checked separately

`angle_dt_fwd(raw_angles, DT)` vs the twin's
`cumsum(tanh(ang_raw) * π * DT) mod 2π`: max abs difference 9.5e-07 at
T = 8 (the kernel's `tanh.approx.f32` PTX instruction vs `torch.tanh`;
both F32).  Not gated here; recorded because the angle path is the one
place the kernel uses an approximate transcendental.

## Gate bounds (reference/test_mamba3_lanes.py)

| lane | bound | measured worst | reason |
|---|---|---|---|
| ieee | 1e-05 | 2.56e-06 (dAngles, T=64); reruns 2.49e-06 | IEEE `tl.dot`; the only rounding is the kernel's bf16 SSM-state store (`mamba3_siso_fwd.py:604`). 4× headroom for rerun noise |
| tf32 | 5e-03 | 2.392e-03 (dAngles, T=16) | TF32's 10-bit mantissa; ~1e-03 typical. 2× headroom; never widened without a new measurement |
| module | 5e-02 | 2.396e-02 (dB_norm_w, T=8) | the public module, bf16 internally. A tripwire, 2× above the measured worst. The bf16 lane is reported, never the oracle |

T = 64 is also the kernel's chunk boundary (`chunk_size=64`).  The T = 1,
T = 2 and T = 64 explanatory cases get their own short table in row A3 of
the correctness campaign, in this file when it lands.

## What this does not say

- It does not rank lanes for accuracy in training.  For that see A2
  (the F32 vs bf16 training curve) and only then a verdict.
- It does not compare CPU-Bend against CUDA PyTorch; the Bend port is
  checked against the same twin by `make mamba3` and friends.
- The `ieee` lane is the closest this GPU gives to the reference
  recurrence, not a bitwise guarantee.
- A lane/device/thread/version change is a banner, never a delta.

## The explanatory cases (A3)

The three T that explain the whole kernel.  Same tolerances as the lane
tables above; the closed forms are the twin's own math written out, so a
disagreement is a bug, not a lane difference.

| case | what it shows | closed form vs twin | kernel (ieee) vs twin | module (bf16) vs twin |
|---|---|---|---|---|
| T = 1 | the result is the `D + gamma*(q.k)` skip only; the SSM term `q @ state^T` is 0 (the initial state is zeros) | 0.000e+00 | 4.842e-08 | 3.642e-03 |
| T = 2 | the state enters: `state_1 = a_0 * state_0 + x_0 ⊗ (scale_0 * k_0)` with `scale_0 = DT_0*trap_0 + DT_1*(1-trap_1)`; `o_1 = a_1 * q_1 . state_1 + (D + qk_1) * x_1` | 0.000e+00 | 4.842e-08 | 3.454e-03 |
| T = 64 | one full chunk (`chunk_size=64`): the in-chunk cumulative decay `DA_CS == cumsum(ADT * log2 e)` over the block (kernel vs `torch.cumsum`: 8.517e-08) | 8.517e-08 | 4.049e-07 | 6.495e-03 |

`make mamba3-cases` (`reference/test_mamba3_cases.py`) re-derives all of
this; `MAMBA3 CASES PASS`.  The closed forms match the twin exactly
(measured 0.0 on 2026-11-16), which is the expected identity: the twin is
these closed forms nested in a loop.
