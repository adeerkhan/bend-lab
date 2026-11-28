# Looped transformer, Bend v0 at 128 puzzles per hop

Local measurement of `examples/looped_transformer/main.bend`. The model is a
reduced Bend arm of `burn-lab/looped`: four stored transformer blocks (one
prelude, two shared middle blocks, one coda), runtime `passes`, answer-only
loss, and a 16-note chain puzzle.

## Build

    BEND_MEM=6G TMPDIR=/tmp/opencode scripts/bend-guard.sh make session \
      TASK=looped-build-eval128 \
      CMD="env PATH=$HOME/.local/bin:$HOME/.bend/bin:$HOME/.elan/bin:$PATH \
      LD_LIBRARY_PATH=$HOME/.local/lib/llvm22 \
      bend -o .scratch/looped examples/looped_transformer/main.bend"

Run 338, exit 0, 118.8 s. The build is Bend 2.0.38 over the pinned
`bendtorch@2.1.0.0` and `bend-ml-tensor-array@0.1.20.0` environment.

## Data

    BEND_MEM=6G TMPDIR=/tmp/opencode scripts/bend-guard.sh make session \
      TASK=looped-data-128 \
      CMD="python3 examples/looped_transformer/reference/make_data.py \
      --out examples/looped_transformer/data --steps 2000 --batch 8 \
      --eval-count 8 --eval-all-count 128 --eval-all-batch 8 \
      --eval-hops 12 --max-hops 8"

Run 327, exit 0, 0.6 s. `eval-all.bin` is 192 evaluation steps: 12 hops, 16
steps per hop, 8 puzzles per step, or 128 puzzles per hop. The same training
file is used by both arms.

## Training and evaluation

    BEND_MEM=6G TMPDIR=/tmp/opencode scripts/bend-guard.sh make session \
      TASK=looped-plain-2k-eval128 \
      CMD=".scratch/looped train_eval_all 2000 1 \
      examples/looped_transformer/data/train.bin \
      examples/looped_transformer/data/eval-all.bin"

Run 340, exit 0, 108.6 s. Final loss 2.0886405.

    BEND_MEM=6G TMPDIR=/tmp/opencode scripts/bend-guard.sh make session \
      TASK=looped-loop4-2k-eval128 \
      CMD=".scratch/looped train_eval_all 2000 4 \
      examples/looped_transformer/data/train.bin \
      examples/looped_transformer/data/eval-all.bin"

Run 341, exit 0, 258.6 s. Final loss 2.2999907.

Each evaluation step prints one `accuracy` line. The report aggregates each
block of 16 lines with:

    python3 examples/looped_transformer/reference/summarize_eval.py \
      results/looped-plain-2k-eval128/runs/20261128-114223-24340.log
    python3 examples/looped_transformer/reference/summarize_eval.py \
      results/looped-loop4-2k-eval128/runs/20261128-114416-24721.log

Runs 342 and 343, exit 0. The four-digit blind-guess line is 25% per hop.

## Result

| hop | plain, 1 pass | looped, 4 passes |
|---:|---:|---:|
| 1 | 20.3% (26/128) | 10.2% (13/128) |
| 2 | 15.6% (20/128) | 8.6% (11/128) |
| 3 | 18.0% (23/128) | 13.3% (17/128) |
| 4 | 15.6% (20/128) | 10.2% (13/128) |
| 5 | 19.5% (25/128) | 10.2% (13/128) |
| 6 | 14.1% (18/128) | 11.7% (15/128) |
| 7 | 8.6% (11/128) | 8.6% (11/128) |
| 8 | 20.3% (26/128) | 9.4% (12/128) |
| 9 | 19.5% (25/128) | 10.9% (14/128) |
| 10 | 16.4% (21/128) | 9.4% (12/128) |
| 11 | 17.2% (22/128) | 10.9% (14/128) |
| 12 | 21.1% (27/128) | 11.7% (15/128) |
| **seen 1-8** | **16.5%** | **10.3%** |
| **unseen 9-12** | **18.6%** | **10.7%** |

**No architecture conclusion.** Both arms are below the 25% chance line at
one hop and have not learned the task. The earlier eight-puzzle smoke reported
25.0% and 12.5%; at 128 puzzles those values are 20.3% and 10.2%. The larger
sample exposes an undertrained, answer-biased model rather than a small
sampling difference.

## Fidelity banners

The Bend v0 is not the burn-lab model:

- the message is `start + RMSNorm(message)`; the learned `keep`/`let_in` gate
  is absent;
- the four-token causal depthwise K/V convolution is absent. The released
  `bendtorch@2.1.0.0` does not contain the grouped `Conv1d` merged upstream in
  PR #45;
- per-head Q/K RMSNorm before RoPE is absent;
- every parameter, including RMSNorm gains, is initialized uniformly in
  `[-0.5, 0.5)`; burn-lab uses norm gains of one and library initialization;
- the Bend arm is `d=32`, batch 8, FF=64. Burn-lab's recorded comparison is
  `d=64`, batch 16, FF=192.

Burn-lab's recorded 2,000-step article run uses the full architecture, 128
puzzles per hop, and reaches 100% at one hop. Its seen/unseen means are 35.4% /
25.8% for plain, 35.1% / 25.4% for looped fixed four passes, and 37.3% / 26.4%
for deep. These are a different lane; the Bend numbers above must not be
reported as a reproduction of them.

The next fidelity work is the K/V short convolution and its forward/backward
parity gate, then the gate and initialization, then a controlled re-run.
