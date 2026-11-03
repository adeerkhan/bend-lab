# The plumbing split of one ViT training step

`make plumbing` builds and runs `bench/plumbing/main.bend` 3 times and
keeps the median of each phase. It imports the real example (`examples/diffusion/vit`)
and calls its own functions, so these are the numbers the model actually pays.

Machine: Linux-7.2.14-200.fc44.x86_64-x86_64-with-glibc2.43 · 28 CPUs · Bend 2.0.38

| phase | ms |
|---|---:|
| `init_params` (once per process) | 6 |
| `rows` (1024 rows before the matrix) | 6 |
| `step_in` (batch, noise, mix, rows, matrix) | 7 |
| `Mat.from_list` (derived: step_in - rows) | 1 |
| `forward` (the model on that matrix) | 52 |
| `step_tgt` (noise, patches, matrix) | 1 |
| `train_new` (grad and optimizer buffers) | 1 |
| **one full step** | **134** |

Shares of the step: input and target construction **6%**,
forward 39%, loss + backward + optimizer
55%.

## Every machine measured, newest first

A step time is comparable only inside its own row: the kernel version, the CPU
governor and the load are part of the number, and a row is the whole context.

| machine | measured | step ms | forward ms | rows ms |
|---|---|---:|---:|---:|
| `Linux-7.2.14-200.fc44.x86_64-x86_64-with-glibc2.43` | 2026-11-03 | 134 | 52 | 6 |

The step is now arithmetic. `step_in` and `step_tgt` together are
8 ms of 134 ms (6%); the forward is
52 ms and loss + backward + optimizer are 74 ms.
`init_params` costs 6 ms once per process.

The history below is quoted with its machine: 7.2.9 on 2026-10-30, not this
one. Before the builder fix (that machine, the same program) a step was 543 ms, with
77% in the input and target build, and `init_params` was 25.3 s. The builders
appended into a growing list (O(n²)) and the tokeniser called `at()` 64 times
per digit. With cons + one reverse and a linear 16-pixel tokeniser the build
falls to 8 ms, and the values are bit-identical: the full 20,480
value row list and the first three losses were compared against the old
version.

The device is worth re-measuring now: the forward and backward are
94% of the step, and the host no longer
hides them.
