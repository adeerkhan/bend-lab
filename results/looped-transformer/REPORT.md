# Looped transformer, Bend v0 at 128 puzzles per hop

Local measurement of `examples/looped_transformer/main.bend`. The model is a
reduced Bend arm of `burn-lab/looped`: four stored transformer blocks (one
prelude, two shared middle blocks, one coda), runtime `passes`, answer-only
loss, and a 16-note chain puzzle.

## v0.2: the four-token K/V convolution (current lane)

The convolution is `examples/looped_transformer/short_conv.bend`: a local
depthwise causal Conv1d (kernel 4, padding 3, no bias), written as a dense patch
matrix because `bendtorch@2.1.0.0` has no grouped Conv1d. It is applied to K
and V in `KVL`.

    reference/.venv/bin/python reference/test_short_conv.py

Run 425, exit 0: **y, dx and dw0..dw3 match PyTorch at 0.017x the oracle
tolerance.** The probe is `examples/looped_transformer/short_conv_probe.bend`.

    BEND_MEM=6G TMPDIR=/tmp/opencode scripts/bend-guard.sh make session \
      TASK=looped-build-shortconv \
      CMD="env PATH=$HOME/.local/bin:$HOME/.bend/bin:$HOME/.elan/bin:$PATH \
      LD_LIBRARY_PATH=$HOME/.local/lib/llvm22 \
      bend -o .scratch/looped examples/looped_transformer/main.bend"

Build run 421, exit 0, 69.3 s.

| arm | run | wall | final loss | seen 1-8 | unseen 9-12 | hop 1 |
|---|---|---:|---:|---:|---:|---:|
| plain, 1 pass | 423 | 888.2 s | 2.1289926 | 23.2% | 24.4% | 21.9% |
| looped, 4 passes | 424 | 2261.5 s | 1.9377306 | 19.9% | 20.9% | 20.3% |

**The convolution did not unlock one-hop lookup.** Plain stays at the chance
line; looped stays below it. Against the v0.1 norm-init lane (plain 24.3% /
24.0%, looped 18.4% / 20.1%), the differences are within noise and the cost is
about **9.6x plain** and **9.9x looped** because the patch-matrix
implementation does a 4d-wide matrix product per layer. A direct shift/scale
convolution would remove that factor; the parity-gated math is already here.

## v0.1: RMSNorm gains initialized to one (previous lane)

The sections below are the pre-convolution lane. The title still says v0.

## Build

    BEND_MEM=6G TMPDIR=/tmp/opencode scripts/bend-guard.sh make session \
      TASK=looped-build-norminit \
      CMD="env PATH=$HOME/.local/bin:$HOME/.bend/bin:$HOME/.elan/bin:$PATH \
      LD_LIBRARY_PATH=$HOME/.local/lib/llvm22 \
      bend -o .scratch/looped examples/looped_transformer/main.bend"

Run 349, exit 0, 115.5 s. The build is Bend 2.0.38 over the pinned
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
      TASK=looped-norminit-plain-2k \
      CMD=".scratch/looped train_eval_all 2000 1 \
      examples/looped_transformer/data/train.bin \
      examples/looped_transformer/data/eval-all.bin"

Run 350, exit 0, 92.1 s. Final loss 2.1634831.

    BEND_MEM=6G TMPDIR=/tmp/opencode scripts/bend-guard.sh make session \
      TASK=looped-norminit-loop4-2k \
      CMD=".scratch/looped train_eval_all 2000 4 \
      examples/looped_transformer/data/train.bin \
      examples/looped_transformer/data/eval-all.bin"

Run 351, exit 0, 229.3 s. Final loss 2.1318007.

Each evaluation step prints one `accuracy` line. The report aggregates each
block of 16 lines with:

    python3 examples/looped_transformer/reference/summarize_eval.py \
      results/looped-norminit-plain-2k/runs/20261128-120642-30616.log
    python3 examples/looped_transformer/reference/summarize_eval.py \
      results/looped-norminit-loop4-2k/runs/20261128-120842-31070.log

Runs 352 and 353, exit 0. The four-digit blind-guess line is 25% per hop.

## Result: RMSNorm gains initialized to one

| hop | plain, 1 pass | looped, 4 passes |
|---:|---:|---:|
| 1 | 20.3% (26/128) | 25.0% (32/128) |
| 2 | 26.6% (34/128) | 18.0% (23/128) |
| 3 | 18.8% (24/128) | 16.4% (21/128) |
| 4 | 18.0% (23/128) | 21.1% (27/128) |
| 5 | 29.7% (38/128) | 18.0% (23/128) |
| 6 | 23.4% (30/128) | 19.5% (25/128) |
| 7 | 28.9% (37/128) | 18.0% (23/128) |
| 8 | 28.9% (37/128) | 10.9% (14/128) |
| 9 | 21.9% (28/128) | 18.0% (23/128) |
| 10 | 28.9% (37/128) | 24.2% (31/128) |
| 11 | 18.8% (24/128) | 21.1% (27/128) |
| 12 | 26.6% (34/128) | 17.2% (22/128) |
| **seen 1-8** | **24.3%** | **18.4%** |
| **unseen 9-12** | **24.0%** | **20.1%** |

**No architecture conclusion.** Plain sits at the chance line overall but is
still at 20.3% on one hop; looped is below chance overall and exactly at chance
on one hop. Neither arm has learned the lookup task. This is an undertrained,
answer-biased model, not a loop-transformer verdict.

### Previous initialization lane

The first 128-puzzle run initialized every parameter, including RMSNorm gains,
uniformly in `[-0.5, 0.5)`. It is retained as a separate lane, not as a delta:

| arm | build/run | final loss | seen 1-8 | unseen 9-12 | hop 1 |
|---|---|---:|---:|---:|---:|
| plain | 338 / 340 | 2.0886405 | 16.5% | 18.6% | 20.3% |
| looped | 338 / 341 | 2.2999907 | 10.3% | 10.7% | 10.2% |

The earlier eight-puzzle smoke (25.0% and 12.5%) was noise; the first
128-puzzle lane measured 20.3% and 10.2% at hop 1.

## Fidelity banners

The Bend v0 is not the burn-lab model:

- RMSNorm gains now initialize to one, but linear and embedding weights still
  use uniform `[-0.5, 0.5)` rather than library fan-in scaling;
- the message is `start + RMSNorm(message)`; the learned `keep`/`let_in` gate
  is absent;
- the four-token causal depthwise K/V convolution is present and parity-gated
  (0.017x), but its dense patch-matrix implementation costs about 10x and did
  not improve this lane;
- per-head Q/K RMSNorm before RoPE is absent;
- the Bend arm is `d=32`, batch 8, FF=64. Burn-lab's recorded comparison is
  `d=64`, batch 16, FF=192.

Burn-lab's recorded 2,000-step article run uses the full architecture, 128
puzzles per hop, and reaches 100% at one hop. Its seen/unseen means are 35.4% /
25.8% for plain, 35.1% / 25.4% for looped fixed four passes, and 37.3% / 26.4%
for deep. These are a different lane; the Bend numbers above must not be
reported as a reproduction of them.

The next fidelity work is the K/V short convolution and its forward/backward
parity gate, then the learned gate and fan-in initialization, then a controlled
re-run.
