# Looped Transformer in Bend

A Bend port of the loop transformer experiment in
[`burn-lab/looped`](../../../../burn-lab/looped/README.md). The model has four
stored transformer blocks: one prelude block, two middle blocks reused for
several passes, and one coda block. The middle passes share their weights and
feed a recurrent message back into the next pass.

This is the Bend arm only. The PyTorch parity fixture and the trained Burn
reference live in `burn-lab`; this directory consumes their protocol, not their
code.

## Status

The Bend arm trains, and the same binary takes `passes` at run time. Plain and
looped use the same stored blocks and code path. Evaluation can run every hop
of a 128-puzzle sample in one process.

This v0 is deliberately reduced, and its numbers are **not** a verdict on the
article:

- the message is `start + RMSNorm(message)`; the learned `keep`/`let_in` gate
  is absent;
- the four-token causal depthwise K/V convolution is absent. The released
  `bendtorch@2.1.0.0` does not contain the grouped `Conv1d` merged upstream in
  PR #45;
- per-head Q/K RMSNorm before RoPE is absent;
- every parameter, including RMSNorm gains, is initialized uniformly in
  `[-0.5, 0.5)`;
- the model is `d=32`, four heads, batch 8, FF=64. Burn-lab's recorded
  comparison is `d=64`, batch 16, FF=192.

## Measured result

2,000 steps, 128 puzzles per hop, the same data for both arms:

| arm | passes | layer runs | final loss | seen 1-8 | unseen 9-12 |
|---|---:|---:|---:|---:|---:|
| plain | 1 | 4 | 2.0886405 | 16.5% | 18.6% |
| looped | 4 | 10 | 2.2999907 | 10.3% | 10.7% |

Both arms are below the 25% four-digit chance line at one hop (plain 20.3%,
looped 10.2%). This is an undertrained, answer-biased model, not an
architecture comparison. The earlier eight-puzzle smoke reported 25.0% and
12.5%; the larger sample changes those to 20.3% and 10.2%.

The exact commands, run ids, and per-hop table are in
[`results/looped-transformer/REPORT.md`](../../../results/looped-transformer/REPORT.md).

## Layout

```
examples/looped_transformer/
  main.bend                 model, puzzle reader, train/eval modes
  reference/make_data.py    deterministic puzzle batches (burn-lab RNG)
  reference/summarize_eval.py  aggregate the per-step evaluation log
  data/                     generated batches (gitignored)
  README.md
```

## Data

The chain puzzle: sixteen shuffled notes form four chains; one question asks
for the digit at the end of a chain. Answering needs one hop per note.

```sh
python3 reference/make_data.py --out data --steps 500 --batch 8 \
  --eval-all-count 128 --eval-all-batch 8 --eval-hops 12 --max-hops 8
```

Writes `data/train.bin`, one-step `eval-hop-{1..12}.bin` smoke files, and
`data/eval-all.bin`. The last is 12 hops × 16 steps × 8 sequences; each step
contains all of its inputs followed by all of its shifted targets.

## Run

```sh
# this machine's clang wrapper needs its LLVM libraries on the path
BEND_MEM=6G PATH=$HOME/.local/bin:$HOME/.bend/bin:$HOME/.elan/bin:$PATH \
  LD_LIBRARY_PATH=$HOME/.local/lib/llvm22 \
  scripts/bend-guard.sh bend -o .scratch/looped examples/looped_transformer/main.bend

.scratch/looped train_eval_all 2000 1 data/train.bin data/eval-all.bin
.scratch/looped train_eval_all 2000 4 data/train.bin data/eval-all.bin
python3 reference/summarize_eval.py results/<task>/runs/<run>.log
```

`train_eval` and `eval` remain for single eight-puzzle batches.
