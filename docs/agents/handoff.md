# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

## 2026-11-19 — the goal run: records, parity, proofs, and what is still owed

### The three differential runs were already settled

- The 200-step Mamba/GLM pilot **completed** (`/tmp/opencode/arch200.log`,
  `results/arch-72k-pilot.{md,json}`; Mamba 4.838168 -> 3.178023 in 7539.0 s,
  GLM 5.563474 -> 1.789081 in 377.9 s, load 3.76 -> 1.98 and 2.10 -> 1.98).
  It is wiring only, not a quality result. The report files were committed.
- The k=1 checkpoint run **completed**: `/tmp/opencode/unet_ckpt4.log`, 3,600
  steps, `final_loss: 0.025900224`, checkpoint `.scratch/unet_ckpt`.
- The "four-pair" run was a duplicate k=1: its checkpoint is byte-identical
  (md5 `87b64cfa21df975db396c5a216afdff6`) and the binary it used predates
  the `--four` source by three minutes. A new binary was built under the
  memory guard and a new session confirmed `--four` changes the trajectory
  (step-25 loss 0.014468994 vs k=1's 0.13427661).

### What landed in bend-lab (commits `580be26`, `063b187`)

- Fixed `reference/unet_record.py`: `c31dd1d` advertised `UNET_RUN`/
  `UNET_TASK` but left `NAME` and `DEFAULT_RUN` undefined and a fixed output
  path, so every registration died with `NameError`. Both are now defined
  from the environment, the record is named after `UNET_TASK`, and
  `reference/test_unet_record.py` gates both forms.
- GLM arm parity: `--matched` in `train_h128.bend` sets CAdamW 0.9/0.999,
  wd 0.01, c1 0.1, c2 0.001 and applies the Mamba warmup/cosine through
  BendTorch's `Train.with_cfg`. A 12-step direct smoke at load ~5 printed
  `5.5634737 -> 5.4061995` matched and `5.5634737 -> 5.344054` default.
  `reference/arch_compare.py --matched` writes `results/arch-matched.*`;
  `reference/test_arch_parity.py` checks the constants against the Mamba
  source. Vocabulary, held-out slices and gradient clipping are banners.
- Inference-only UNet task: `make unet-sample CKPT=<checkpoint>` resumes with
  `--steps 0`, draws 64 samples and re-scores them through the metric gate.
  The results parser now accepts an inference log with no loss lines;
  `reference/test_unet_sample.py` gates it. No sampled number was produced:
  the run reached 12/64 draws in 12 minutes at load ~7 and was stopped.
- Gates after the changes: `make check-fast` **33/33**; the three new tests
  pass individually. The full `make check` was still running at handoff
  (see below), not claimed as green.

### What landed upstream

- bend-ml **PR #39** (branch `feat/array-get-returns-array`, commit `21eafb5`):
  `get_returns_array` proves `Pair.fst(Array.get(T, a, i)) == a` on top of
  PR #38's `get_go_returns_array`, via `ArrayAt.fst` and `ASet.get_fst`
  (ASet.Copy's `b == a` and size equation). `--check-only` and `--verdict`
  (Lean 4.34.0) both `ALL PROOFS CHECK`; `list_laws.py --check-readme` says
  43 laws, README ok. Opened on nuxyel/bend-ml, stacked on #38.
- **PR #38** (`get.go` returns the array) remains OPEN, mergeable, CI
  SUCCESS, zero reviews and zero comments. It needs the maintainer's review;
  it was not merged by this session.
- Both PRs are linked to this OpenChamber session.

### Not verified — owed, with the reason

- **Four-pair legibility.** No trained --four checkpoint or draws exist. The
  restarted 3,600-step session reached 8 loss lines in 25m45s (≈8 steps/min
  at load 5.1), needs several hours, and was stopped before its final save.
  Re-run the same `make session TASK=unet-four` command on a quiet machine,
  then score and register with `make unet-record-four`.
- **tiny-LM step record.** `make bench-mamba-tiny` refuses above load 1.0.
  The host ran a game at ~162% CPU and a `diffusion world-train` job at
  ~75%; no idle measurement was possible, so no new step number is quoted.
- **Full `make check`.** Started as `make session TASK=check-full`; at handoff
  it was at `reference/test_cifar_train.py` with a Bend CIFAR build running.
  No completion or count is claimed until it finishes.
- **Matched 3,200-step run.** Parity code and wiring are in place; a 4-step
  `arch_compare.py --matched` attempt did not finish in 600 s under load 5-6.
  The long run needs a quiet machine and a multi-hour window.
- **Sampled legibility from the checkpoint.** Stopped at 12/64 draws, no
  record written.

### Local ledger

`bend-lab/.local/issues-hit.md` gained five 2026-11-19 entries: the undefined
record-writer names, the stale binary, the multi-hour four-pair window, the
load-bound jobs, and the missing sampled number. `make issues` renders it;
`.local/` was never committed, pushed or quoted publicly.

### Next, in order

1. On a quiet machine, one job at a time: full `make check`, then
   `make bench-mamba-tiny`, then the 3,600-step --four run with sampling.
2. Score the four-pair draws, `make unet-record-four`, `make results`.
3. Launch the matched 3,200-step run with `--matched` and report it against
   the banners; that is the only answer to "which is better".
4. Re-run `make unet-sample` for the sampled legibility number.
5. Wait for the maintainer's reviews on PRs #38 and #39.

## 2026-11-20 — local cleanup and the current blockers

- Unknown CLI flags now fail loudly in all three trainers (commit `febe298`);
  `--bogus` exits 1 with the flag named. No upstream issue was filed: a
  minimal Bend 2.0.38 program proved the runtime passes unknown flags
  through, so the wildcard in our own parsers was the whole cause.
- Full `make check` is still not green. The MambaViT parity test exceeds the
  unchanged 1800 s gate. Build is only 5 s; the binary is the cost. A reduced
  config (96 train + 64 held-out, 3 steps) runs in 1283 s when measured
  alone, but the whole test timed out at 1700 s while the host was under
  load 7 from a forge3d rustc build. The reduction is currently applied and
  unverified in the full gate.
- `make bench-mamba-tiny` refuses at load 1.40–7; no idle record was made.
- Upstream: PR #38 and #39 remain open with green CI and no reviews; PR #40
  (one-cell round trip) was opened and linked this pass.
- Paper ideas #1 and #3 are not documented anywhere in the workspace; their
  descriptions came from the user's status message only and must be supplied
  before any related-work pass can start.


## 2026-11-20 — BendTorch PR #45 reviewed and approved

- tryingET's general Conv1d (#43's first layer) was reviewed and approved.
  Verified locally on the PR head with Bend 2.0.40: `make laws` passed
  (ALL PROOFS CHECK, verdict, FAST_DEFS/GPU_DEFS pins, all bad programs
  rejected) and `make check-conv1d` passed all five cuts at 0.000e+00 for
  y/dx/dW/db, including grouped, stride, dilation and depthwise.
- Two non-blocking questions were left on the PR: whether the padded-surplus
  encoding `t = eff_k + ts - 2p` rejects over-large padding or truncates via
  Nat.sub, and whether the caller establishes the round-trip law's
  hypotheses. This is the #44 geometry-guard lens applied to 1-D.
- #43 is therefore unblocked and moving upstream; we do not write TTS layers.
  CUDA device execution was not run locally.

## 2026-11-21 — gate coverage complete, single-run claim not made

- The full `make check` was externally stopped at its last test with 157
  gates passed and 0 errors. The remaining test (`test_mamba_vit_train.py`)
  was re-run standalone and passed. All 158 gates are therefore covered, but
  not in one uninterrupted run, so no single-run "all green" is claimed.
- The tiny-LM idle step record is fresh: 8,180 ms/step at load 0.43
  (`make bench-mamba-tiny`), down from 11,163 ms/step before the state-walk
  and dW_out fixes. Recorded in `bench/history.md`.
- All Bend builds now run through a 4G memory cgroup; no bare `bend -o`
  remains in the Makefile or bench script.

## 2026-11-23 — Bendlang DiT denoiser, upstream issue, and toolchain state

**The first Bendlang DiT denoiser now trains.** `examples/diffusion/dit/main.bend`
(496 params) imports the verified conditioned block (`wide_block.bend`, width 8,
two kv heads) and wraps it with a token-feature embedding and output head,
taking a separate per-image condition vector. Gates:
`make check-dit` — attention 0.076x, LN+attention 0.954x, full tiny block 5.454x
frozen FAIL (gap isolated, tolerance not loosened); `make check-dit-denoiser` —
forward 0.225x, six AdamW losses 1286.7 -> 781.2, first matches torch exactly,
PASS. Registered as dashboard runs 245 (dit-gate) and 246 (dit-denoiser).
The full conditioned block's gradients drift (14.203x at width 8) — documented
in `results/dit/COMPARISON.md`, not gated. Not yet built: real dataset adapter,
noise schedule and sampler. Dims are illustrative (b=2, t=3, C=8).

**Upstream.** bend-ml#41 filed (masked scatter-add / segment-mean primitive with
forward/backward laws), linked to this session, after an audit found no
duplicate. BendTorch#45 (Conv1d, 715bcb9) was fully re-verified including the
device lane `make gpu` (exit 0 on an RTX 5060) and the existing comment edited
to report it. No BendTorch#43 comment was added. PRs bend-ml#38/#39/#40 and
BendTorch#45 remain OPEN with no reviews.

**Toolchain (traps paid for).** Bend drifted to 2.0.40 during PR verification;
restored to the pinned 2.0.38 via `scripts/setup.sh`, and `make check-fast`
is 33/33 again. clang 22 needs
`LD_LIBRARY_PATH=$HOME/.local/lib/llvm22` to build any Bend binary here.
CUDA_HOME must be exported for the device lane. The knowledge graph was
updated incrementally at HEAD 33cfcf4 (3,322 nodes / 4,565 edges, 13 tour steps)
in `.understand-anything/`; the older `.ua/` graph is unused.

**Repo hygiene.** README Status wall was replaced with a short pointer section.
All findings are in `.local/issues-hit.md` (37 entries, rendered by `make issues`).
