# Contributing to MLBend

Thanks for looking. Bug reports, numbers from other machines, fixes, new laws and
new operations are all welcome.

## Branches

- **Open pull requests against `devel`.** All work happens there; `main` only
  receives releases, through a pull request from `devel`, and the version tags
  live on `main`. GitHub proposes `main` by default, so change the base branch
  when you open the PR.

## Before you start

- For anything beyond a small fix (a change to a package's types or laws, a new
  package, performance work, a new dependency), open an issue first so we can
  agree on the approach before you spend time on it.
- Read [`NOTES.md`](docs/NOTES.md): it records every experiment, including the ones
  that did not pay off. [`docs/AUDIT.md`](docs/AUDIT.md) says what is proved,
  what is tested and what is trusted.
- Bend 2 is young and differs a lot from Bend 1. The "Bend 2.0.37 pitfalls" in
  [`AGENTS.md`](AGENTS.md) lists the traps; read it before writing Bend.

## Setup and checks

```bash
make setup   # the pinned Bend 2.0.37, Lean 4.34.0 and the Python venv
make check   # the proofs, the kernel (--verdict), the rejected programs, the NumPy comparison
make laws    # every law statement of every package
```

A pull request should pass `make check`; CI runs it on every push. If it changes
the numbers a package computes, say why in the PR and show that the comparison
with NumPy and scikit-learn still passes.

## Rules for code

- **Bend stays pinned to 2.0.37.** A version bump is its own change, with an
  entry in `NOTES.md`.
- **Every proof is real.** No `@unsafe`, no `?TODO`, no axioms. A law that
  cannot be proved yet goes into `NOTES.md` as debt, not into the package.
- **Prove structure, test numbers.** Sizes, partitions and composition belong
  in the types and laws; `F32` results are checked against NumPy and
  scikit-learn by the scripts in `reference/`.
- **Every proved law is explained in plain language** in its package's README
  and listed in `reference/list_laws.py`.
- Each package is published to BendHub as one file, so it imports only Base and
  other published packages, at pinned versions. Published versions are
  permanent: a change to a package becomes a new version.

## Publishing a package

A package is published to BendHub as one file, and **publishing is public and
permanent**: a published version never moves, so a change is a new version.

```bash
bend login                                   # once, with your BendAI key
bend metrics/main.bend --publish mlbend-metrics@0.1.0.0
```

- The name is 12 or more characters, a-z, 0-9 and `-`; the version is four
  numbers.
- A `LICENSE` file sits next to the entry file (the repository's MIT licence
  does).
- The publish prints, on stderr, that the hub is public and permanent.
- After publishing, point the demos and the other packages at the new version
  and record it in `NOTES.md`.

## Benchmarks

Performance claims need numbers someone else can reproduce: the hardware, the
Bend and compiler versions, an idle machine, runs alternated between the old and
the new version, and one-thread timings pinned to a single core. Report the
losses and the noise too, not only the best run.

## Commits and pull requests

- Small commits, one logical change each, with clear messages in English.
- Keep a pull request to one topic.
- Don't commit datasets, model weights or large logs.
- Using AI tools is fine. You are still the author: say in the PR what you ran
  yourself and what you did not, and keep the description short enough for a
  person to read.

## License

By contributing you agree that your contribution is licensed under the
[MIT License](LICENSE) of this repository.
