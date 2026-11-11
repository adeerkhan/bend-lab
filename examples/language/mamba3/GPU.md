# The BendTorch GPU lane, and why Mamba-3 is not on it yet

Bend does have a GPU lane. BendTorch calls it `gpu.bend`: the same layers,
types and bits, with their products and element-wise work run as `!` bangs
through `bend-ml-tensor-gpu`. A program built with CUDA runs the bangs on the
device; the same binary with `+RTS --gpu off -RTS` runs them on the CPU
threads. My earlier "there is no Bend GPU lane" meant that the Mamba-3
example does not use this lane. It was wrong as a statement about Bend.

## What was measured

`probe_bendtorch_gpu.bend` builds a `LinearNBF(8, 64, 304)` layer, runs its
forward and backward twice, once with BendTorch's CPU linear layer and once
with the GPU layer, and prints the number of differing words in `y`, `dX` and
`dW`.

    CPU and GPU bits: 0 differing words

The process appears in `nvidia-smi --query-compute-apps` while it runs. With
`--gpu on` the tiny layer takes `0.28-0.36 s` including the GPU program load;
with `--gpu off` it takes under `0.01 s`. That is the expected result at this
size: the work is too small to pay the device context and launch cost. The
BendTorch README reports the same ordering for its tiny models and recommends
the CPU lane until the products are large.

## Why Mamba-3 is still a CPU model

The Mamba-3 block is not a composition of existing BendTorch layers. The dense
pieces are: `in_proj` and `out_proj` are `Linear`, B and C get `RMSNorm`, the
output gets `SiLU`, and the LM has `Embedding` and `CrossEntropy`. The part
that makes it Mamba-3 is the selective scan, and BendTorch has no scan layer.

The current scan is a `Nat` recursion over `t` over `List<&2, F32>`. That code
cannot become a GPU bang by changing an import: a bang runs one `!` call over
cells or rows, and the scan carries a `headdim x d_state` state from one `t` to
the next. Making it a GPU lane needs one of:

1. **A chunked scan**, the kernel's own form: the intra-chunk causal matrix,
   the carried chunk state, and matrix products through `tensor-gpu`. This is
   the right path and the one the roadmap already names first.
2. **A sequential scan bang**, one leaf per head with the `t` loop inside the
   leaf. It would run on the device, but with four leaves it would not use the
   GPU well.
3. **A custom CUDA/Triton kernel** called from Bend. That is outside the
   BendTorch layer set and outside this example's scope.

## What to do with BendTorch

BendTorch is still the right framework for the model, even before the scan is
on the GPU:

- Wrap the block as a `T.Layer<P, TA.Mat<8, 64>, TA.Mat<8, 64>, S>` so
  `T.Seq`, `Residual` and `T.Train.step` handle composition, losses,
  optimizers and checkpoints. The hand-written SGD and CE in
  `probe_mamba3_lm.bend` can then go away.
- Use `G.LinearF` / `G.RMSNorm` / `G.SiLU` for the dense pieces and keep the
  scan on the host, or move the scan to a chunked GPU bang. The first is a
  hybrid; the second is the real GPU model.
- Keep the Python twin as the oracle. BendTorch is the trainer and the GPU
  lane, not a reference implementation of Mamba-3's recurrence.

## Reproduce the GPU lane

The upstream `bendtorch` worktree was fetched at `65c8038` (devel, 2026-11-11)
and `bend-ml` `main` at `8af6052`. The package cache has
`bendtorch@2.0.0.0`, `bendtorch-gpu@2.0.0.0` and `bend-ml-tensor-gpu@0.1.5.0`;
the devel tree wants `tensor-gpu@0.1.5.2`, which is not in the cache, so the
probe uses the published 2.0.0.0 packages. The linear layer is the same lane.

In this environment the `clang` on PATH is broken and the CUDA toolkit is the
NVRTC package, not a full toolkit. The GPU build used the compiler's C output
and `gcc`:

    bun <bend>/bend2/main.ts probe_bendtorch_gpu.bend -o /tmp/gpu_linear.c
    gcc -O3 -DBEND_CUDA=1 -I<cuda-home>/include /tmp/gpu_linear.c \
        -L<cuda-home>/lib64 -lcuda -lnvrtc -lpthread -lm -o /tmp/gpu_linear
    LD_LIBRARY_PATH=<cuda-home>/lib64 ./gpu_linear +RTS --gpu-build
    LD_LIBRARY_PATH=<cuda-home>/lib64 ./gpu_linear +RTS --gpu on -RTS

`<cuda-home>` needs `include/nvrtc.h`, `include/cuda.h`, `lib64/libnvrtc.so`
(and `libnvrtc-builtins.so.13.0`), and `lib64/libcuda.so`. With a full CUDA
toolkit and a working `clang`, `CUDA_HOME=<cuda> bend probe_bendtorch_gpu.bend
-o probe` does the same build in one step.

The GPU lane is a real lane. It is not yet a Mamba-3 lane.
