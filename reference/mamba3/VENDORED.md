# Vendored Mamba-3

`state-spaces/mamba` at commit
`e9594ce1c732d97440f0332fdc43170a2294dbfa` (2026-07-22T19:41:12Z),
Apache-2.0 (`LICENSE` in this directory).

The SISO path only. MIMO and the TileLang/CuTeDSL step kernels are outside this
work; the files below are the ones the SISO module imports, plus the two tests
that define the reference's generation and variable-length protocols.

| file | why |
| --- | --- |
| `mamba_ssm/modules/mamba3.py` | the module wrapper: `in_proj`, `heavy_tail`, `A`, `DT`, gated RMSNorm, the SISO call, `out_proj` |
| `mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py` | the autograd wrapper; the bf16 casts and the angle cumsum call |
| `mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py` | the chunked forward kernel; the trapezoid, rotary order and skip term |
| `mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py` | the backward kernels |
| `mamba_ssm/ops/triton/mamba3/angle_dt.py` | `cumsum(angle * DT) mod 2π` and its backward |
| `mamba_ssm/ops/triton/mamba3/grouped_head_reduction.py` | GQA reduction used by the wrappers |
| `mamba_ssm/ops/triton/mamba3/utils.py` | the PTX `tanh`, `sigmoid`, `cos` and `sin` approximations |
| `mamba_ssm/ops/triton/layernorm_gated.py` | `RMSNorm` for B and C (eps 1e-5) |
| `mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py` | the SISO decode step; imported by the generation test |
| `mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py` | imported by `mamba3.py` at module load even for SISO |
| `mamba_ssm/ops/triton/mamba3/mamba3_mimo_utils.py` | imported by the MIMO step module |
| `tests/test_generation.py` | the generation protocol: prompt, greedy loop, stop token |
| `tests/modules/test_mamba3_varlen.py` | the variable-length interface |

The `mamba_ssm` package `__init__.py` files are empty stubs, so the vendored
tree imports with `PYTHONPATH=reference/mamba3` and does not shadow an
installed `mamba_ssm`. `einops` is the only extra Python dependency; it is
installed in `reference/.venv-rf`.

The kernels were not edited. The probe calls the SISO forward kernel directly
with float32 inputs and `TRITON_F32_DEFAULT=ieee`; the public module path is a
bfloat16 lane because the combined wrapper casts Q, K, V, Trap, Angles and Z.
