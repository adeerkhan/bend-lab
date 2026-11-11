# Mamba-3 SISO: Bend beside the Python reference

This is the measured comparison behind `README.md`. The question is narrow:
the same SISO block, the same weights, the same `u` and `dy`, run through

1. the Bend port (`examples/language/mamba3/mamba3_siso.bend`),
2. the pure-PyTorch twin (`reference/mamba3_twin.py`), the F32 oracle,
3. the public `Mamba3` module, a bf16 Triton lane, and
4. the SISO forward kernel called directly with float32 and IEEE dots.

The numbers come from one command:

    reference/.venv-rf/bin/python reference/mamba3_compare.py

The machine is an Intel Core i7-14700HX and an NVIDIA GeForce RTX 5060 Laptop
GPU (sm_120), torch 2.11.0+cu128, Triton 3.6.0, Bend 2.0.38. The Bend rows are
native C binaries built from the `.bend` files with `bend -o .c` and
`gcc -O2`. The torch rows are warm in-process loops. The boundaries differ:
the Bend rows include process start, the `Mat` loads and the list conversion;
the torch rows exclude imports and model construction. The two are not one
lane, and the table keeps them apart.

## Correctness

All gates are in `reference/check_mamba3_bend.py`, against the F32 twin.

| path | lane | compared | max relative |
| --- | --- | --- | --- |
| Bend forward, seqlen 8, 18 tensors | F32, CPU | twin | `9.2e-07` |
| Bend forward, seqlen 4 / 16 | F32, CPU | twin | `7.4e-07` / `9.6e-07` |
| Bend backward, every parameter and retained tensor | F32, CPU | twin | `2.9e-06` |
| Bend moved-`dy` can-fail | F32, CPU | moved-`dy` twin | `6.9e-07`, and `du` changes |
| twin forward, `out_v` / final state | F32, CUDA | kernel, F32 + IEEE dots | `2.0e-07` / `4.0e-07` |
| public module, whole block | bf16, CUDA | twin | `2.5e-03` |

The order matters. The Bend port is a float32 computation and is closer to
the mathematical oracle than the public module, which casts Q, K, V, Trap,
Angles and Z to bfloat16 before the kernel. The kernel itself is not wrong:
called with float32 inputs and `TRITON_F32_DEFAULT=ieee`, it reproduces the
twin at `2.0e-07`. The bf16 gap is the wrapper's chosen lane, not the scan.

## Speed

One block, `d_model 64`, `d_state 16`, four heads, batch 1, seqlen 8. The Bend
rows include the load; the torch rows do not. The forward is one call; the
backward is `loss = sum(y * dy)`.

| path | lane | forward | forward+backward |
| --- | --- | --- | --- |
| Bend, native C, one process | F32, CPU | `37.23 ms` | `440.79 ms` |
| Bend, slope over 20 / 10 calls | F32, CPU | `37.91 ms/call` | `586.93 ms/call` |
| PyTorch twin | F32, CPU | `1.477 ms` | `4.473 ms` |
| PyTorch twin | F32, CUDA | `3.269 ms` | `11.034 ms` |
| public `Mamba3` module | bf16, CUDA | `0.596 ms` | unavailable |

The Bend slope is the fixed-cost-free per-call time, from running the block
20 times in one process for the forward and 10 times for the backward. The
forward slope and the one-call number agree (`37.9` ms), so the load is
negligible. The backward slope is larger than the one-call number
(`586.9` against `440.8` ms), which is allocation pressure: each call builds
a fresh 2,048-float state list and its gradient, and the process keeps the
old lists alive longer than one call does. That is a real property of the
list substrate, not a timing artifact.

Against the same-lane CPU twin, the Bend forward is **25x** slower and the
forward+backward is **99x** slower. Against the CUDA twin, the forward is
**11x** slower and the forward+backward **40x** slower, but that comparison
crosses the lane. The public bf16 module is **63x** faster than the Bend
forward, and it is the least accurate of the four: speed and accuracy move in
opposite directions here.

The CUDA twin is slower than the CPU twin at this shape (`3.27` against
`1.48` ms forward, `11.0` against `4.47` ms forward+backward). A batch of one
and eight positions cannot pay the kernel-launch and Python-dispatch cost.
The Triton module wins because it fuses the whole SISO block into a small
number of launches, not because the GPU is intrinsically faster at this size.

## Training and generation

The ten-step copy task is `tokens 0..7 in, the same tokens out`: an embedding
`8 x 64`, the block, a `64 x 8` head, mean cross-entropy, plain SGD at
`lr = 1e-3`. Generation is two weight-tied layers, greedy, from the prompt
`[0, 1, 2, 3]` to 16 tokens.

| path | lane | ten-step copy task | twelve generation steps | final tokens |
| --- | --- | --- | --- | --- |
| Bend, native C | F32, CPU | `5.766 s` | `1.166 s` | `[0, 1, 2, 3, 1, ...]` |
| PyTorch twin | F32, CPU | `0.048 s` | `0.039 s` | same |
| PyTorch twin | F32, CUDA | `0.104 s` | `0.072 s` | same |
| public `Mamba3` | bf16, CUDA | n/a | n/a | n/a |

The copy task has 80 token positions in ten steps. Bend is **120x** slower
than the CPU twin and **18x** slower than the CUDA twin. Generation is
**30x** slower than the CPU twin. The loss curves and the final tokens are
identical to the twin's; the Bend rows are slow, not wrong.

The CUDA twin is slower than the CPU twin on training too (`0.104` against
`0.048 s`) for the same launch-bound reason. At this scale the CPU is the
better PyTorch lane; the GPU becomes useful when the batch, the sequence or
the model grows, or when the fused Triton kernel replaces the explicit loop.

## Memory

| path | what is measured | value |
| --- | --- | --- |
| Bend native C, forward | process peak RSS | `5.5 MB` |
| Bend native C, forward+backward | process peak RSS | `6.4 MB` |
| Bend native C, ten-step training | process peak RSS | `6.0 MB` |
| Bend native C, generation | process peak RSS | `5.1 MB` |
| PyTorch twin, forward+backward | CUDA allocated peak | `17.463 MB` |
| PyTorch twin, forward+backward | CUDA reserved peak | `291.504 MB` |

The earlier `354264 KiB` in the README was the `bend` compile-and-run process,
not the program: the compiler, the checker and the JIT dominate the resident
set. The native C binary of the whole training probe is `6.0 MB`. The two
numbers are both true, but they measure different things, and the smaller one
is the one that belongs beside the CUDA allocated peak.

The CUDA reserved peak is the caching allocator's high-water mark, not live
tensor memory. The allocated peak (`17.46 MB`) is the comparable number.

## What the comparison says

- The port is numerically right. Every gate is at least two orders of
  magnitude inside its tolerance, and the backward is right to `2.9e-06`
  relative, not merely close on the loss.
- The port is slow because it is an explicit loop over lists on one CPU core.
  The 99x forward+backward gap to the CPU twin is the list substrate and the
  scalar per-element work, not the scan math. The chunked kernel and the
  fused Triton path exist precisely to remove that work.
- The public module's speed comes with a bf16 lane. At `2.5e-03` relative it
  is still a fine training lane, but it is not the F32 oracle, and a gate
  written against it would not prove the F32 block.
- At this tiny shape, the GPU is not automatically the fast lane. The CUDA
  twin is slower than the CPU twin for both the block and the ten-step task
  until the fused kernel or a larger batch/sequence pays the launch cost.
- The training and generation tokens match exactly, so the slow Bend lane can
  still be used as a correctness oracle for a faster future implementation.

## What is not compared

MIMO, `ngroups > 1`, `rope_fraction 1.0`, `is_outproj_norm=True`, varlen,
batch greater than one, independent two-layer weights, the 25.7M model, the
Triton backward, and a Bend GPU lane. Those are roadmap items, not
measurements. The comparison is one layer, one batch, eight positions, one
CPU, one GPU, and one pinned reference commit.
