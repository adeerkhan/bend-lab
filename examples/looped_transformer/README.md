# Looped Transformer in Bend

A Bend port of the loop transformer experiment in
[`burn-lab/looped`](../../../../burn-lab/looped/README.md). The model has four
stored transformer blocks: one prelude block, two middle blocks reused for
several passes, and one coda block. The middle passes share their weights and
feed a gated recurrent message back into the next pass.

This is the Bend arm only. The PyTorch parity fixture and the trained Burn
reference live in `burn-lab`; this directory consumes their protocol, not their
code.

## Status

First milestone: a runnable Bend arm on the article's 16-note chain puzzle.
It trains, and the same binary takes `passes` at run time, so plain and looped
use exactly the same stored blocks and code path.

This v0 is deliberately reduced, and its numbers are **not** a verdict on the
article:

- the message is `start + RMSNorm(message)`; the learned `keep`/`let_in` gate
  is not in the Bend arm yet;
- the four-token depthwise K/V convolution is not in the Bend arm yet. The
  article's small-model notebook uses it to make one-hop lookup learnable, and
  the burn-lab port includes it;
- the model is `d=32`, four heads, one prelude block, two shared middle blocks,
  one coda block, batch 8, sequence 66, answer-only loss. Burn-lab uses
  `d=64`, batch 16 and a 192-wide MLP for the recorded comparison.

## First measured result

`train_eval`, 2,000 steps, 8 puzzles at hop 1 (blind guess 25%):

| arm | passes | stored blocks | layer runs | hop-1 accuracy |
|---|---:|---:|---:|---:|
| plain | 1 | 4 | 4 | 25.0% |
| looped | 4 | 4 | 10 | 12.5% |

The 8-puzzle sample is too small to separate the arms. The point of this run is
that the Bend loop is live: the same parameters are trained through one or four
middle passes, the loss falls from `3.66` (`log 39`) to about `1.7`/`2.4`, and
the binary runs both arms without recompiling. Burn-lab's 128-puzzle run is the
comparison to reproduce after the gate and short convolution land.

## Layout

```
examples/looped_transformer/
  main.bend                 model, puzzle batch reader, train/eval modes
  reference/make_data.py    deterministic puzzle batches (burn-lab RNG)
  data/                     generated batches (gitignored)
  README.md
```

## Data

The chain puzzle: sixteen shuffled notes form four chains; one question asks
for the digit at the end of a chain. Answering needs one hop per note.

```sh
python3 reference/make_data.py --out data --steps 500 --batch 16
```

Writes `data/train.bin` (one batch of 16 sequences per step) and
`data/eval-hop-{1..12}.bin` (one forced chain length per file).

## Run

```sh
# this machine's clang wrapper needs its LLVM libraries on the path
BEND_MEM=6G PATH=$HOME/.local/bin:$HOME/.bend/bin:$HOME/.elan/bin:$PATH \
  LD_LIBRARY_PATH=$HOME/.local/lib/llvm22 \
  scripts/bend-guard.sh bend -o .scratch/looped examples/looped_transformer/main.bend

.scratch/looped train_eval 2000 1 data/train.bin data/eval-hop-1.bin
.scratch/looped train_eval 2000 4 data/train.bin data/eval-hop-1.bin
```

The first comparison is plain (`passes 1`) against looped (`passes 4`) with the
same stored blocks and the same data.
