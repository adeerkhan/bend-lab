# MLBend

The experiment layer for machine learning in Bend 2: a dataset in, a
reproducible report out, over any Bend trainer. The tensors and the laws are
bend-ml's; the training is BendTorch's; the data, splits, transformations,
selection, metrics, reports and manifests are ours.

**The rule that keeps it distinct:** MLBend writes no tensor, no autodiff and
no optimizer. A model enters through a task and an adapter, not through a new
engine. The only models that live here are gradient-free baselines.

## Where things are

- `metrics/`, `preprocess/`, `split/`, `select/`, `cluster/`, `pipeline/`, `datasets/`, `hf/`: the eight
  packages (`mlbend-metrics`, `mlbend-preprocess`, `mlbend-split`,
  `mlbend-select`, `mlbend-cluster`, `mlbend-pipeline`, `mlbend-datasets`,
  `mlbend-huggingface`), one repository,
  one workspace (`bend.toml`), each published to BendHub. `bendenv` is the
  only piece that lives in a repository of its own: `adeerkhan/bend-env`.
- `reference/experiment.py`: the experiment runner — tasks, adapters,
  references, report cards, manifests and the W&B sink.
- `reference/test_transformer.py`: the trained transformer against its NumPy
  twin; it writes `results/transformer/loss.tsv`, `loss.svg` and `run.json`.
- The maintained training library is **`bendtorch@1.0.1.0`** (`nuxyel/BendTorch`,
  `upstream/devel`). `bendtorch-core@0.1.0.0` was the old `aaaxn` line; the
  three examples on it were retired on 2026-10-24.
- `docs/PLAN.md`: the layer, the roadmap and the stop criteria.
- `docs/AUDIT.md`: what is proved, what is tested, what is trusted.
- `docs/NOTES.md`: every finding and every discarded hypothesis, by date.
- `reference/`: the NumPy and scikit-learn comparisons, the per-task tests,
  and `experiment.py`, the runner that writes the reports.
- `reports/`: the generated report cards (ignored by git).
- `docs/agents/`: the map of the work for an agent.
  - `docs/agents/handoff.md`: **where we are, what was run, what was not, what
    is next.** It is overwritten at the end of every session; read it first.
  - `docs/agents/domain.md`: the vocabulary and the design rules.
  - `docs/agents/issue-tracker.md`: how the issues carry the work.
  - `docs/agents/adapters.md`: the task and adapter protocol, and the manifest.

## How an agent works here

1. **Read, in this order**: this file, `docs/agents/handoff.md`, the issue the
   handoff names, `docs/PLAN.md`, `docs/agents/domain.md`, then
   `docs/NOTES.md` for the part of the code you are about to touch.
2. **One issue per work item**, on `main`, in small commits. The acceptance
   criteria go in the issue (`docs/agents/issue-tracker.md`).
3. **Run `make check` before you believe anything.** It checks the proofs, the
   kernel, the programs that must be refused, and the numbers against NumPy and
   scikit-learn. A number you did not run is not a number.
4. **Say what you ran and what you did not**, in the pull request and in
   `docs/NOTES.md`. The kernel check (`--verdict`) needs Lean 4.34.0: if your machine
   has no Lean, the law is *checked* but not *proved*, and the pull request
   must say so.
5. **A law that cannot be proved yet is an issue**, not a package.
6. **Run the experiment layer too**: `make experiment TASK=<task>` writes the
   report card; `make experiments` runs the corpus. A task whose Bend number
   drifts from its reference is a failure, not a note.
7. **Write the handoff** before you stop: the next agent starts from it.

## Where the language lives

Not in this repository, and not as a copy. The language moves every day; a copy
here would be wrong tomorrow.

- **The guide**: `bend guide` (the installed one), or
  `~/.bend/guide/GUIDE.md`.
- **The Base source**: `~/.bend/bend2/base.bend` (the pinned 2.0.38 install) or
  `bend base <name>`.
- **The language source**: `github.com/bendlang/bend`. The work on the language
  itself (the compiler, the runtime, the checker) happens there, in its own
  repository, on its own rhythm. `make sync-bend` fetches the pinned tag into
  `../bend-ref` when a read-only copy is wanted.
- **The published packages**: `~/.bend/lib` is the hub cache; the source of
  every package this repository depends on is readable there, offline.

## Branches

Work on `main`: every commit and push goes there, and CI checks it. The
`devel` branch was merged into `main` on 2026-10-23 and is kept for history; it
is no longer the work branch. A *package* release (`make release PKG=<dir>`)
publishes to BendHub and tags the released commit `<name>-v<version>`.

## Rules

0. **No new trainer here.** No tensor, no autodiff, no optimizer, no second
   `linear`/`ridge`/`logistic` implementation. Those are BendTorch's or
   bend-ml's; this repository adapts them. Gradient-free baselines are the
   exception, because a baseline needs no trainer.
1. Bend is pinned to **2.0.38** (and Lean 4.34.0 for `--verdict`). A version
   bump is its own change, with an entry in `docs/NOTES.md`.
2. Run the checker on every change; it takes a fraction of a second.
3. **Every proof is real.** No `@unsafe`, no `?TODO`, no axioms. A law that
   cannot be proved yet is recorded in `docs/NOTES.md` as debt and left out of the
   package.
4. **Prove structure, test numbers.** Sizes, partitions and composition belong
   in the types and the laws; `F32` results are checked against NumPy and
   scikit-learn by the scripts in `reference/`.
5. **Every proved law is explained in plain language** in its package's README.
6. **Publishing to BendHub is public and permanent.** Four-number versions
   (`0.1.0.0`), names of 12 or more characters, a `LICENSE` next to the entry
   file. A published version never moves: a change is a new version.
7. **Benchmarks are honest**: the hardware, the versions, an idle machine, the
   losses and the noise, not only the best run.
8. Everything in the repository is in English. Commits are small, with clear
   messages and no attribution lines.
9. Public text (READMEs, `docs/`) is written in a neutral, professional voice:
   no first person, no dash dividers, no private ledger links.

## Measurement rules

- **A number comes from a command.** In a report, a README, a commit or a
  record, a number is quoted with the command that printed it. No estimates, no
  "about".
- **`unavailable` over 0.** A missing measurement prints `unavailable`, never a
  zero and never a guess; a stage below a millisecond prints `<1 ms`.
- **A lane, device, thread or version change is a banner, never a delta.**
  `make report TASK=... AGAINST=...` prints `lane changed` and keeps both
  numbers; the verdict does not pretend two lanes are one measurement.
- **No gate is loosened.** A config comes down, a tolerance is widened only with
  a measured reason written beside it, and a check is never skipped: a skipped
  check reads exactly like a passing one.
- **`.local/` is local-only.** Never committed, pushed, linked or quoted. Every
  finding goes to `bend-lab/.local/issues-hit.md`, rendered by `make issues`;
  a public issue is written only after the maintainer has the repro.
- **Read the plan before performance work**: `.local/burn-to-bend-plan.md`.
- **Heavy builds run under the guard**: `BEND_MEM=9G scripts/bend-guard.sh ...`,
  never a bare `bend -o`.

## Bend 2.0.38 pitfalls we paid for

The traps below are the ones this repository's code hit; they are the same
traps bend-ml paid for, and the fix is in each case.

- `match` only inspects a parameter or a pattern variable. A computed value
  (including a call) needs its own def, or a parameter computed by the caller:
  `same_len_of(Nat.is_eq(n, m), n, m)`, never `match Nat.is_eq(n, m):`.
- There is no `let` keyword: a let is `x = v`, and `+x = v` when the value is
  reused (and `Data`). A statement inside a case body needs the `;` that ends
  it, or the next `case` is parsed as part of it.
- No forward references: a def must be defined before it is used, so the
  helpers come first.
- Variables are affine; `+x` reuses a `Data` value. A parameter used in a type
  counts as a use. A `Result` is a `Type`, so it cannot be reused.
- `match` on two values takes two patterns: `case Con{y, yt} Con{p, pt}:` and
  `case _ _:` for the rest.
- `List<&2, F32>` is a list whose values may be used twice (the list is
  read-only); `List<&2, Nat>` likewise. `List.length(&2, F32, xs)` reads it
  without consuming it.
- `Nat` and `F32` are `Data`; `Result` and functions are `Type`.
- The comparison operators are not terms: use `Nat.is_eq`, `F32.is_eq` and
  `Nat.cmp`, and pass the `Bool` to a def that matches on it.
- A negative literal is not a term: write `(0.0 - 0.5 : F32)`.
