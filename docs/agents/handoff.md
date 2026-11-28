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

Hosted CI run `46127994381` (`60fd377`) completed setup successfully and ran
the full gate until the workflow's 90-minute limit (cancelled at 10:30:33 UTC
while `mv_train.bin` was running). A local full `make check` with
`PYRF=reference/.venv/bin/python` finished at **184/185 passed, 5 unavailable**
before the MNIST fixture; `test_mamba_vit_train.py` was the only error, from a
missing `torchvision`. `reference/requirements.txt` now pins
`torchvision==0.29.2`; `reference/materialize_mnist.py` provisions the four
pinned MNIST archives; the gate passes standalone in 772.8 s (run 366, gate
1800 s).

The 90-minute job limit was not loosened. The three long references
(`test_mamba3_tiny_lm_run.py`, `test_mamba3_tiny_opt.py`,
`test_mamba_vit_train.py`) now run in a second `reference-long` job via
`make check-long`; the main job runs `make check-ci` through
`make validate CI_SPLIT=1`. Both jobs keep the same gates and the same
90-minute limit. The main job also uploads its full-gate log on cancellation.

### Looped transformer: current measured lane

Build run 349: 115.5 s. Both arms use the same 2,000-step, batch-8, 16-note data
and 128 puzzles per hop.

| arm | run | final loss | seen 1-8 | unseen 9-12 | hop 1 |
|---|---|---:|---:|---:|---:|
| plain, 1 pass | 350 | 2.1634831 | 24.3% | 24.0% | 20.3% |
| looped, 4 passes | 351 | 2.1318007 | 18.4% | 20.1% | 25.0% |

Plain reaches the chance line overall but does not learn one-hop lookup;
looped remains below chance overall. This is an undertrained, answer-biased
model, not an architecture verdict. The exact commands and per-hop table are
in `results/looped-transformer/REPORT.md`.

The Bend v0 still lacks the learned `keep`/`let_in` gate, the four-token causal
depthwise K/V convolution and per-head Q/K RMSNorm. Linear/embedding weights
are uniform `[-0.5, 0.5)` rather than fan-in scaled, and the model is `d=32`,
batch 8, FF=64 against burn-lab's `d=64`, batch 16, FF=192. The grouped
`Conv1d` is merged upstream in BendTorch PR #45 but is not in the released
`bendtorch@2.1.0.0`.

### Upstream state

- BendTorch #46 and #47 are open, mergeable, with green selected CI and no
  review yet. No public comment was posted without approval.
- bend-ml #40 remains draft at `4d07592`. The general round-trip statement is
  false without a size bound (`r*c <= 2^31`), and the `Fill.Other` construction
  is still incomplete. No axiom, `@unsafe` or unproved law was added.

### Next, in order

1. Read the final hosted `make validate` result and the local CPU full-check
   log. Fix setup/gate failures without loosening a tolerance or timeout.
2. If the CPU runner times out a long gate, reduce that gate's configuration
   with a measured reason; do not hide it.
3. For the looped arm, implement the K/V short convolution with a
   forward/backward PyTorch parity gate before integrating it. A viable local
   design is four causal row shifts plus per-channel scales; the released
   BendTorch has no grouped `Conv1d`. Then add the learned gate, per-head Q/K
   norm and fan-in initialization, and re-run the same 128-puzzle lane.
4. Keep #40 draft until the owner approves the size-bounded statement.
