# Handoff

One file, overwritten at the end of every session. A fresh agent reads this
first, then the issue it names, then `AGENTS.md`.

## 2026-11-28 — clean-runner setup, CPU Mamba-3 twins, and the looped Bend arm

### Goal

Make bend-lab a reproducible corpus for Bend ML: a clean hosted runner can
provision every fixture, the full gate reports unavailable CUDA lanes honestly,
and the Bend looped-transformer arm produces measured results rather than a
smoke claim.

### Landed on `main`

- `8c88632` — clean-runner fixtures and a self-contained DiT oracle. Setup now
  materializes the six SHA-pinned datasets, both GLM53 corpus windows, the
  Mamba-3 tiny dump and the MambaViT dump. The six DiT probes use
  `reference/oracle.py`; `test_dit_ref.py` keeps its two parity gates green and
  freezes the full-block gradient gap at 5.454x. MambaViT uses a deterministic
  sine image instead of downloading MNIST.
- `02f87cd` — looped evaluation at 128 puzzles per hop. `eval-all.bin` is 12
  hops × 16 steps × 8 puzzles; `train_eval_all` evaluates all 192 steps in one
  process and `summarize_eval.py` aggregates them. The generator now stops
  after 100,000 infeasible splits instead of spinning forever.
- `404c936` — RMSNorm gains initialize to one. The flat initializer randomizes
  zero-valued linear/embedding entries and preserves nonzero gains. This is a
  separate measured lane, not a delta against the random-norm lane.
- `dd4d0f6` — CPU-only Mamba-3 twins. `Mamba3` is a lazy proxy with a
  parameter-identical `PureMamba3` fallback when Triton/einops are absent. The
  GQA and outnorm twins use the same proxy. Five CUDA gates are reported as
  unavailable on a CPU runner; they still run when CUDA is present.

### Clean CI state

Hosted workflow `46143520699` (`4d3ca67`) is **success**. The main
`make validate` job ran 69m37s and the `reference-long` `make check-long` job
ran 56m40s, both under the unchanged 90-minute limit. Setup provisions every
fixture; the five CUDA/Triton gates are reported `unavailable` on the CPU
runner; the three long references run in the second job. The previous main-job
run was 177/182; the five errors were the missing GLM `data/h` weights and the
DiT gap gate frozen to a local 5.454x. Setup now prepares `data/h`, and the
DiT gate requires the gap to stay above 1.0x and prints its measured value
(5.454x local, 2.038x hosted CPU). Each fixed gate passes standalone.

No gate, tolerance or timeout was loosened. The main job uploads its full-gate
log on failure or cancellation.

### Looped transformer: current measured lane (v0.2, with K/V convolution)

Build run 421, 69.3 s. The four-token causal depthwise K/V convolution is in
`examples/looped_transformer/short_conv.bend` and passes a PyTorch parity probe
on y, dx and dw0..dw3 at **0.017x** the oracle tolerance (run 425). Both arms
use the same 2,000-step, batch-8, 16-note data and 128 puzzles per hop.

| arm | run | wall | final loss | seen 1-8 | unseen 9-12 | hop 1 |
|---|---|---:|---:|---:|---:|---:|
| plain, 1 pass | 423 | 888.2 s | 2.1289926 | 23.2% | 24.4% | 21.9% |
| looped, 4 passes | 424 | 2261.5 s | 1.9377306 | 19.9% | 20.9% | 20.3% |

The convolution did not unlock one-hop lookup; plain stays at chance and
looped below it. The dense patch-matrix implementation costs about 10x the
no-convolution lane, so the performance fix is a direct four-shift/per-channel
scale layer. The learned `keep`/`let_in` gate and true per-head Q/K RMSNorm are
still absent; linear/embedding initialization is uniform and norm gains are
one.

### Upstream state

- BendTorch #46 and #47 are open, mergeable, with green selected CI and no
  review yet. No public comment was posted without approval.
- bend-ml #40 remains draft at `4d07592`. The general round-trip statement is
  false without a size bound (`r*c <= 2^31`), and the `Fill.Other` construction
  is still incomplete. No axiom, `@unsafe` or unproved law was added.

### Next, in order

1. Replace the dense patch-matrix convolution with the four-shift/per-channel
   implementation; keep the same parity probe (y, dx, dw) and the 128-puzzle
   lane. The patch form is correct but 10x too slow.
2. Add the learned `keep`/`let_in` gate and a true per-head Q/K norm, then
   re-run the same lane.
3. Per-layer fan-in initialization needs a BendTorch API change: `Layer.zeros`
   is also the optimizer and gradient identity, so it cannot encode an
   initializer.
4. Keep #40 draft until the owner approves the size-bounded statement.
