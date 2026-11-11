# Mamba-3: what to do next, and why

The SISO block is correct and gated. This is the list of things that are worth
doing next, in the order that keeps the oracle intact. Each item says what it
changes, what it teaches, and how it would be accepted. None of it is a reason
to loosen a gate.

## 1. Make the scan fast without changing its math

**What.** Replace the exact `t` loop with the kernel's chunked form: split the
sequence into chunks of 64, compute the intra-chunk causal part as a small
matrix, carry the state across chunks, and use `exp2` on the cumulative
`ADT * log2(e)`. Keep the loop as the reference and gate the chunked form
against it at every tensor, not only at `y`.

**Why it matters.** The measured forward is `37.9 ms/call` and the
forward+backward `440.8 ms/call` for eight positions; the CPU PyTorch twin is
`1.48` and `4.47 ms`. The gap is the per-element list work and the repeated
`take_row`/`nth`, which is quadratic in the position count. A chunked scan
turns the intra-chunk work into a few dense operations and reduces the number
of state round-trips.

**Learn.** The kernel's `da_cs`, `da_cs_rev`, `exp2` and the `scale` factor
are not an optimization detail; they are the same trapezoid recurrence
re-associated. Writing the chunked form and proving it against the loop is the
best way to understand why the kernel is written the way it is.

**Accept.** The chunked forward matches the loop at `1e-5` relative for T =
8, 16, 64, 65 and 128, including the 64-boundary. The backward matches at
`1e-4` per tensor. The loop stays in the tree as the oracle.

**Before that.** An array-based core (`Array.get`/`Array.set`, O(1) per cell)
would remove the list traversal even in the loop. It is a smaller change and
it makes the chunked version easier to write.

## 2. A decode step and a real generation loop

**What.** Implement the reference's step path: one token at a time, carrying
the four states (`angle`, SSM, K, V). The current generation re-runs the whole
prefix for every token, which is `O(T^2)` in the prefix and `1.166 s` for
sixteen tokens. A step kernel is `O(1)` per token after the prompt.

**Why it matters.** Generation is the place where Mamba's recurrent state is
supposed to win: no KV cache that grows with the context, one state per layer.
The full-prefix loop hides that advantage.

**Learn.** The reference's `mamba3_siso_step.py` and
`tests/test_generation.py` show the state protocol: the angle state is the
running cumsum, the K state is post-bias and post-rotation but pre-scale, the
V state is the last value, and the SSM state is the `headdim x d_state`
matrix. The step must also apply the trapezoid's shifted term from the
previous token, which is why the state carries K and V separately.

**Accept.** For the same prompt and weights, the step-by-step 16 tokens equal
the full-prefix 16 tokens, and the generated tokens equal the twin's. Compare
the per-token wall time at 16, 64 and 256 tokens.

## 3. Broaden the model, in the reference's own order

**3a. Batch and varlen.** The current port is batch 1. Varlen is the
`cu_seqlens` path: concatenate sequences, reset the state at each boundary.
The reference has `tests/modules/test_mamba3_varlen.py` for exactly this.

**Accept.** Two sequences run separately and concatenated with `cu_seqlens`
give the same outputs and the same final states, at `1e-5`.

**3b. `ngroups > 1` (GQA) and `rope_fraction 1.0`.** The kernel supports
grouped Q/K heads and a full-width rotation. Both change the shape plumbing,
not the recurrence. Test the angle layout and the B/C bias broadcast.

**3c. `is_outproj_norm=True`.** The module can add a gated RMSNorm after the
block, with `group_size = headdim`. The SISO port currently uses the default
`False`. This is a small extra backward through a per-head RMSNorm and the
SiLU gate, and it is a good test of the norm chain already written.

**3d. MIMO.** The module's MIMO rank adds `mimo_x`, `mimo_z` and `mimo_o`
projections and a headwise norm. It is the largest remaining model feature and
it needs its own oracle: the SISO twin does not cover it. Do not start it
until batch/varlen and the step are stable.

**3e. Independent two-layer weights.** Generation currently uses two
weight-tied layers. An independent second layer is a parameter-plumbing
change, and it makes the "tiny LM" closer to a real stack.

## 4. Train a real tiny LM, not only a copy task

**What.** A small vocabulary, a real tokenized corpus, a multi-layer stack,
AdamW with weight decay, gradient clipping, a learning-rate schedule, and a
held-out split. The current ten-step copy task is a plumbing and gradient
test, not a language-model result.

**Why.** A ten-step loss curve proves the backward. It does not prove that the
block learns a useful sequence transformation, that the trapezoid helps, or
that the angle/RoPE parameterization behaves at longer lengths.

**Learn.** This is where the reference's training recipe matters: `dt_init`,
`A_floor`, the heavy-tail activation, the `D` skip initialization, and the
bf16 lane. Compare F32 and bf16 training at the same seed and report both
curves; a bf16 run is a lane, never a delta.

**Accept.** A held-out loss and token accuracy, a fixed seed, a manifest and a
report card in the `bend-lab` format, and a comparison against the twin's
training on the same data. The 25.7M model remains out of scope; a 1–2 layer,
64-dim model is enough to learn from.

## 5. Numerical lanes and the kernel truth

**What.** Treat the four lanes explicitly:

| lane | what it is | current evidence |
| --- | --- | --- |
| F32 twin | the explicit loop, the oracle | the goldens |
| F32 kernel + IEEE | the same math with `input_precision="ieee"` | `2.0e-07` on `out_v` |
| F32 kernel + TF32 | the default `tl.dot` lane | about `1.6e-3` |
| public module | bf16 Q/K/V/Trap/Angles/Z | `2.5e-3` |

**Why.** A future change can look correct in one lane and fail in another. The
public module is not the oracle; the F32 twin is. The kernel's default TF32
dot is another lane again.

**Learn.** Run the same comparison at T = 1, 2, 4, 8, 16, 64 and at a chunk
boundary. At T = 1 the result is the `D + gamma*(q.k)` skip only; at T = 2 the
state and the shifted term enter; at T = 64 the chunk boundary enters. Those
three cases explain the whole kernel.

**Accept.** A table of max relative errors per lane and per tensor, with the
tolerance and the command. No lane is silently promoted to the oracle.

## 6. A correctness campaign

- **Finite differences** for the scan and the preprocessing at a few
  coordinates, to catch a sign that happens to cancel in the loss.
- **State round-trip**: run T tokens, then continue from the returned state;
  the result must equal a single run of the concatenation.
- **Angle wrap**: a sequence whose cumsum crosses `2π` many times; the port
  already needed `wrap` because `F32.mod` follows C `fmod`.
- **Clamp boundaries**: `A` at the `-A_floor` clamp, `trap` at 0 and 1,
  `softplus` at large and small `dd_dt`.
- **Can-fails**: moved `dy` (already), missing `shifted`, wrong rope sign, a
  zeroed `D`, a transposed state, a chunk boundary off by one.
- **Determinism**: two runs of the same binary and two builds of the same
  source give the same checksums.

## 7. Integrate with the rest of the lab

The block is a standalone example. The next layer is a BendTorch layer and a
`bend-ml` task, so `make experiment` can write a report card with the same
manifest, metrics and replay as the other models. That requires an adapter,
not a new trainer: the tensors and the optimizer stay in BendTorch, and the
task owns the data, splits, selection and report.

A `make mamba3` target and the `make check` gate already exist. Add a small
CI lane for the C build (gcc is available; clang is not, so `bend -o` fails
and `bend -o .c` plus gcc is the working route).

## 8. Research questions worth the measurement

1. **Does the trapezoid matter?** Fix `trap = 0.5` or `trap = 1` and train the
   same tiny LM. The reference's claim is that the trapezoid improves
   stability and quality; a tiny copy task may not show it, a real corpus
   might.
2. **Does the data-dependent A and the heavy tail matter?** Compare
   `heavy_tail` against `softplus`, and sweep `A_floor`.
3. **Does the angle/RoPE parameterization extrapolate?** Train at 64 and
   evaluate at 128 and 256; watch the angle cumsum and the state norm.
4. **Mamba-3 against a small Transformer and a linear-attention block** on the
   same data and budget. The comparison needs the same parameter count, the
   same tokens and the same evaluation, or it is not a comparison.
5. **Chunked versus loop.** The chunked form is a performance change; verify
   it does not change the loss curve before using it in a research run.

## Order of work

1. Array-based core and a chunked scan, gated against the loop.
2. Step decode and the step-generation protocol.
3. Batch, varlen, `ngroups`, `rope_fraction`, `is_outproj_norm`.
4. A real tiny LM and a report card.
5. MIMO and the fused norm.
6. The numerical and research questions.

## What not to do

- Do not replace the loop before the chunked form is gated against it.
- Do not call the public bf16 module the oracle.
- Do not compare a CPU Bend number to a CUDA PyTorch number without saying so.
- Do not port MIMO before batch and varlen are stable.
- Do not widen a tolerance to make a run fit; bring the config down.
