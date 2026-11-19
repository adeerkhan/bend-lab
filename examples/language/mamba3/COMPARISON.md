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

The same input, first four values of `y` at token 0 and of `du` at the first
token:

| tensor | Bend | twin |
| --- | --- | --- |
| `y[0:4]` | `-0.02732083, -0.01525306, 0.03155780, 0.02665956` | `-0.02732083, -0.01525307, 0.03155781, 0.02665959` |
| `du[0:4]` | `-6.3932296e-03, -3.3832359e-04, -1.1846449e-02, -2.9656259e-05` | `-6.3932259e-03, -3.3832528e-04, -1.1846456e-02, -2.9659655e-05` |

The max absolute differences are `8.75e-08` on `y` and `2.61e-08` on `du`.

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
forward+backward is **99x** slower on this run. A second run of the same
command gave Bend `42.6 ms` and `482 ms` and the CPU twin `2.46 ms` and
`5.44 ms`; the ratio is about `17-25x` and `89-99x`, not a fixed constant.

The Bend rows above are the **list scan**. With the array core and the chunked
scan, the same 20-forward benchmark is `0.41-0.42 s` (about `20-21 ms/call`,
`make mamba3-bench`), so the forward ratio to the twin is about `13-15x`; the
backward still runs the list path and its numbers are unchanged.
Against the CUDA twin, the forward is **11x** slower and the forward+backward
**40x** slower, but that comparison crosses the lane. The public bf16 module is **63x** faster than the Bend
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

## The full two-layer LM

The block comparison above is one layer. The full model is two independent
blocks with a residual around each, a final RMSNorm and an untied `8 x 64`
head: `55,632` block parameters plus `512` embedding, `512` head and `64`
norm, `56,720` in total. The gate is `probe_mamba3_lm.bend`; the Python twin
is `reference/mamba3_lm_twin.py`.

| path | lane | forward | ten SGD steps | twelve generation steps |
| --- | --- | --- | --- | --- |
| Bend, native C, one probe | F32, CPU | included | included | included, `17.682 s` total |
| PyTorch full LM | F32, CPU | `3.370 ms` | `0.102 s` | `0.043 s` |
| PyTorch full LM | F32, CUDA | `6.034 ms` | `0.235 s` | `0.076 s` |

The Bend probe prints the initial forward (`lm_hn`, `lm_logits`, `lm_loss`),
runs ten SGD steps, then generates twelve tokens from the final weights. The
whole native binary is `17.682 s` and `6.9 MB` RSS; against the CPU twin's
`0.102 + 0.043 s` that is about **122x**, and against the CUDA twin's
`0.235 + 0.076 s` about **57x**. The probe's own forward is inside the total,
so the ratio is conservative.

Correctness of the full model:

| quantity | max relative |
| --- | --- |
| `lm_hn` | `1.03e-07` |
| `lm_logits` | `3.48e-07` |
| `lm_loss` | `8.85e-08` |
| ten-step loss curve | `1.77e-07` |
| final tokens | `[4, 2, 6, 3, 3, 5, 0, 4]`, exact |
| generation tokens | `[0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]`, exact |

This is the whole model, not a block: the residual, the final norm, the head,
the embedding and both layers' backward all enter the loss curve. The Python
twin uses autograd; the Bend side is the hand-written chain from the block,
the norm and the head.

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
- The port is slow because it is an explicit loop over arrays and lists on
  one CPU core. The 99x forward+backward gap to the CPU twin is the list
  substrate in the backward and the scalar per-element work, not the scan math. The chunked kernel and the
  fused Triton path exist precisely to remove that work.
- The public module's speed comes with a bf16 lane. At `2.5e-03` relative it
  is still a fine training lane, but it is not the F32 oracle, and a gate
  written against it would not prove the F32 block.
- At this tiny shape, the GPU is not automatically the fast lane. The CUDA
  twin is slower than the CPU twin for both the block and the ten-step task
  until the fused kernel or a larger batch/sequence pays the launch cost.
- The training and generation tokens match exactly, so the slow Bend lane can
  still be used as a correctness oracle for a faster future implementation.

## Why the reference is the Python twin, not BendTorch

BendTorch is the training framework, not an implementation of Mamba-3. It has
`Linear`, `RMSNorm`, `SiLU`, `Residual`, `Embedding`, `CrossEntropy`, four
optimizers and one `Train.step`, but no selective scan or state-space layer.
`bend-ml`'s `tensor-array` is used here for `TA.Mat`, the storage the probes
load. The Python twin is the oracle because it is the only implementation of
the recurrence that can be checked line by line against the kernel. The
hand-written backward was also the point of the original port: the goal asked
for it, not for `T.Train`.

The right use of BendTorch is the next step: wrap the block as a `T.Layer`,
use `T.Train.step` for the optimizer, and use the `gpu.bend` names for the
dense pieces. That is a framework change, not a replacement for the oracle.

## The Mat product

The scan's largest dense product is `q_t S_t^T`: one `1 x 16` query against
one `32 x 16` state. The list version drops a row for every one of the 32
outputs and calls `dot` on lists. The bend-ml version is one
`TA.Mat.matmul_nt(1, 16, 32, ...)`.

`bench_qs_list.bend` and `bench_qs_mat.bend` run the same product 20,000
times on the same query and state, extracted from the golden at `t=3`, head
0, and print a checksum:

| path | 20,000 products | per product |
| --- | --- | --- |
| list `take_row` + `dot` | `2.59 s` | `129 us` |
| `TA.Mat.matmul_nt` | `0.04 s` | `2 us` |

The Mat product is about **65x** faster. The two checksums differ by
`1e-3` over 20,000 products (about `5e-8` each); the printed 32 values agree
to the last one or two decimal digits. This is the substrate lever: one qS
product is a small part of the full scan, but the full scan has one per head
per token, and the state update and output have the same list overhead.

Putting the product inside `scan_head` did **not** carry the 65x over. The
substitution kept the checksum and passed every gate (forward, backward,
training, generation), but twenty full forwards at seqlen 8 took `0.84-1.18 s`
against `0.71-0.73 s` for the list path on the same machine (user time
`0.83-1.11 s` against `0.71-0.73 s`). `scan_head` still needs `take_row` for
each row of the state update, so most of the isolated benchmark's saving was
the row extraction, not the 16-element dot; the `Mat` conversions add
allocation on top.

The **array state** did carry. `scan_head` now copies the 512-float head state
into one `Array<F32>`, updates the 32x16 cells in place with `Array.set`, and
reads the state back once; the state is no longer rebuilt as 32 new rows per
token. Twenty forwards at seqlen 8 fell from `0.71-0.73 s` to `0.37-0.38 s`
(user time `0.37-0.38 s`), and the ten-step train probe from `9.81-10.50 s` to
`3.64-3.67 s`. The forward, backward, training, generation and full-LM gates
all pass.

The **chunked scan** carried it further. `pass_c` now builds one 2048-float
`Array<F32>` from the initial state and carries it across all four heads and
all tokens (`scan_rows_st`, `sc_go_st`), so the state is not copied into a fresh
array per head per token. In a same-session, interleaved A/B, twenty forwards
at seqlen 8 took `0.41-0.42 s` against `0.53-0.55 s` for the per-head array
core, medians `0.41 s` against `0.54 s` (about **24% less**); a second session
gave `0.37-0.45 s` against `0.49-0.56 s`, about 15%. The ten-step train probe
was equal within noise
(`5.58-5.97 s` against `5.74-5.78 s`, interleaved runs). The earlier
`0.37-0.38 s` figure is the array core in a faster machine state: the two
sessions are not compared as a delta.

## The BendTorch GPU lane

BendTorch does have a CUDA lane (`gpu.bend` over `bend-ml-tensor-gpu`), built
with NVRTC. A `LinearNBF(8, 64, 304)` forward and backward, run once with the
CPU layer and once with the GPU layer, gives **0 differing words** in `y`,
`dX` and `dW`; the process appears in `nvidia-smi` while it runs. At this size
the GPU run is slower (`0.28-0.36 s` including the GPU program load against
under `0.01 s` on the CPU), which is the launch-bound result BendTorch reports
for its tiny models.

Mamba-3 is not on that lane yet because its scan is not a BendTorch layer. The
linear, norm, SiLU, embedding and head pieces can move; the `t` recursion with
the `headdim x d_state` state is the part that needs a chunked GPU scan or a
custom bang. `GPU.md` has the build and the caveats. This is a missing
implementation, not a missing Bend GPU.

## What is not compared

MIMO, `ngroups > 1`, `rope_fraction 1.0`, `is_outproj_norm=True`, varlen,
batch greater than one, the 25.7M model, the Triton backward, and Mamba-3 on
the BendTorch GPU lane. The lane itself is tested on a linear layer; the scan
is not on it. The full two-layer LM is in the measured table; the variants
are not. Those are roadmap items, not
measurements. The comparison is one layer, one batch, eight positions, one
CPU, one GPU, and one pinned reference commit.

## The tiny-LM step: where the time goes (2026-11-17, CPU lane)

Measured with `bench/plumbing_mamba_tiny/main.bend`, which calls the
trainer's own functions; the run is a Bend 2.0.38 native binary on this
machine, no GPU and no Python in the timed region. One diagnostic run
printed:

| phase | measured | per step |
|---|---|---|
| forward (`tiny_parts`, two blocks + head) | 3,076 ms | 3.1 s |
| full step, three `tiny_io_example` calls | 155,213 ms / 3 | 51.7 s |
| AdamW's three list walks (m, v, p) over 71,952 cells | 167 ms / 3 | **0.06 s** |
| one array-style add of the same 71,952 cells (incl. conversions) | 20 ms | 0.02 s |
| two block backward passes, each with its own forward | 71,814 ms | 71.8 s / 2 blocks |

Subtracting the two forwards from the last row leaves roughly 65 s for the
two block backwards — the same order as the 51.7 s full step, so the scan
backward dominates. The optimizer's three passes over all 71,952 weights
are negligible; an array-style pass over the same cells is ~3x faster than
the list walks but still only tens of milliseconds, so that difference
cannot explain a 50 s scan.

These are diagnostics, not a clean partition: the backward probe uses a
constant upstream dy and includes its own forwards, while the full step
uses the real loss and dy. The same bench printed 45.8, 106 and 51.7 s per
step on three runs of this machine; other processes were using the CPU
during the 106 s run, so no single number here is a stable benchmark. The
conclusion — backward dominates, optimizer does not — held in every run.

No upstream issue was filed: the expensive code is our explicit backward
scan in `examples/language/mamba3/mamba3_siso.bend`, not a benchmarked
library operation, and the array-vs-list comparison does not pin the cost
on bend-ml or BendTorch. The next question is local: which part of that
scan backward (the recomputed forward, the per-token list indexing, or the
gradient assembly) accounts for the tens of seconds.

### The hot loop inside that backward (same probe, later run)

The backward was split into its two phases by timing `bw_start` (the scan)
and `S.backward` (scan + preprocessing weights) separately on the same
weights, and the per-token in_proj accumulation was timed on its own:

| phase (two blocks unless noted) | measured in one run |
|---|---|
| full step (three steps, divided by three) | 99.2 s |
| scan + preprocessing weights | 158.8 s |
| scan only | 78.4 s |
| preprocessing weights only (difference) | 80.4 s |
| 128 × (outer(dpre, u_t) + flat add over 19,456 cells + transpose matvec) | **52.9 s** |

The synthetic hot loop is 66% of the preprocessing phase in the same run,
so the dominant cost is the per-token in_proj gradient accumulation into a
19,456-cell linked list (`pw_fin2` in `mamba3_siso.bend`), not the scan and
not the optimizer. Absolute times in this table vary by ~3x between runs on
this machine (other processes using the CPU); only the within-run ratios
are the result. The loop is our code, not a library op — an upstream issue
would be a guess until someone measures a Mat-backed version of the same
accumulation.

### Dedicated measurement: list vs array accumulation

`bench/mamba_tiny_accum/main.bend` runs only the accumulation, 128 tokens of
a 304-wide dpre against a 64-wide u_t (19,456 cells). The list side is
`outer` + `add_lists`; the array side is `acc_outer`, the helper the trainer
now uses, with `arr_dW` converting the result back to the list once. The
values differ per cell, and the program ends by walking both 19,456-cell
results and printing the sum of the squared differences:

| loop | measured (one run, load average 3.0) |
|---|---|
| list `outer` + `add_lists`, 128 tokens | **6,719 ms** |
| array `acc_outer` + one conversion + the check sum, 128 tokens | **50 ms** |
| the conversion alone (19,456 cells) | 0 ms (below the 1 ms clock) |
| sum of squared differences, cell by cell | **0** |

The array form is ~130x faster for the same arithmetic. `cells_diff` is
exactly 0, which is the claim worth having: the array result is the list
result cell for cell, not a permutation with the same total. The first
version of this benchmark wrote the array side in column-major order, so
both sides summed to the same number (each cell is touched once per token
either way) while the lists differed; this check is what caught it before
the trainer was changed.

### The accumulator as an array (2026-11-18)

`G2.dW_in` is an `Array<F32>` now, added in place per token by `acc_outer`
and rendered as a list once by `arr_dW` at the `to_grad` boundary; the SISO
and GQA paths both changed. The per-cell add order is unchanged, so the
result is bit-identical: the check above reports 0, and on the real probe
`probe_mamba3_grad.bend` built against `275f057` and against this commit
prints the same 26 lines, `dW_in`'s 19,456 numbers included (`cmp` finds no
difference). The same bench, in one run at load average 3.4, timing the
trainer's own functions:

| phase (two blocks unless noted) | before (2026-11-17) | after, same process |
|---|---|---|
| full step | 48.6 s (recorded run) | 11.9 s |
| scan + preprocessing weights | 43.2 s / 158.8 s (two runs) | 10.7 s |
| scan only | 14.8 s / 78.4 s (two runs) | 12.2 s |
| the old list accumulation alone, 128 tokens | 52.9 s (one run) | 12.6 s |

The last two rows are the point: in the same process, the old accumulation
loop by itself (12.6 s, one block's worth of tokens) now costs more than the
entire new backward for two blocks (10.7 s), and the preprocessing phase has
fallen to the difference between two scans, which came out negative inside
the noise. Absolute times still move by ~3x with outside CPU load (a game
held 2.3 cores during the after run), so the before and after columns are
not one lane; the within-run comparison is. The `mamba3-tiny-lm` record
was re-made on 2026-11-19 at load 0.52: **11,163 ms per step** (forward
2,557 ms, backward and optimizer 8,606 ms) against the 2026-11-17
record's 48,585 ms per step, 4.3x. Both records are in `results/raw/`;
`bench/history.md` has the rows.

### The scan's own split (2026-11-19)

Four probes of one position of `bw_go`, 128 positions, one process at load
average 4.2 (the probes ran back to back, so their ratios are the result):

| piece of one position | per block |
|---|---|
| `take_row(states, t*2048, 2048)`, the old walk | **2,076 ms** |
| the same slice after reversing the states once | **79 ms** |
| `outer(dy_t, yi_t)` + `add_lists` into 8,192 cells (`dW_out`) | 1,070 ms |
| `scan_bw_all`, the four head scans and their combine | 514 ms |
| `matvec_t(w_out, dy_t, 64, 128)` for `dyi` | 114 ms |

The largest single piece was a walk, not arithmetic: `take_row` drops
`t * 2048` cells before taking the row, so the block was quadratic in the
number of positions. `bw_start` now reverses the states once (minus the final
state the backward never uses) and each position takes 2,048 cells and drops
2,048: 79 ms against 2,076 ms in the same process. The step gate prints the
same loss and heads (4.838168, 327157.12, 9.758908), and `make mamba3` and
`make mamba3-gqa` pass, so the numbers did not move.

The second row is the next target: the per-token outer product plus flat add
into the 8,192-cell `dW_out` accumulator is the same shape as `pw_fin2`
before the array accumulator, and the fix is the same one.
