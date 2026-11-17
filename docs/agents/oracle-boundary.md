# The oracle boundary, and what we take from Burn

Four rules, learned from Burn (`burn.dev/books/burn`) and from issue
`nuxyel/BendTorch#43`. None of them is a new runtime dependency; all of them
are about keeping the measurement honest.

## 1. The run path has no oracle in it

A run is a compiled Bend program plus its weights. The oracle — NumPy,
PyTorch, ONNX Runtime — lives in the reference function, outside the run path,
and its time is charged to `run.reference_seconds`, never to the program.

This is what Burn does too: its models are Rust source compiled into a binary;
ONNX Runtime is not in the inference path. Its own troubleshooting guide even
says to compare outputs against ONNX Runtime when a result looks wrong — the
oracle is the check, not the runtime. Issue #43's TTS model already runs in
Bend with no ONNX Runtime at inference; ONNX Runtime is only the layer oracle.

A compiled language does not remove the need for an independent reference. It
removes the need to *ship* one. Keep the boundary: `reference/experiment.py`
calls `cfg["reference"]` only after the Bend binary has run.

## 2. Interop is a build-time edge, never a runtime layer

If we ever need to import a foreign model, the Burn pattern is the one to
copy: translate it to source ahead of time (their `burn-onnx` crate generates
readable Rust plus a separate weight file at build time), then compile the
result as an ordinary module. No ONNX interpreter at run time, and the
generated code can be read, edited and checked.

For us that means any future importer is a separate package or a build script
that emits Bend source — not a dependency of `bend-lab`, `bend-ml` or
BendTorch, and never a runtime call inside a gate. It is the same rule nuxyel
gave in #43 for signal-processing: FFT/STFT are a separate BendHub package,
not a BendTorch layer.

## 3. A lane is a typed, gated dimension

Burn ships no execution backend by default: the backend is an explicit Cargo
feature, and generated models have a compliance gate before they are trusted.

Our equivalent is `variant` in the run record (`trainer`, `lane`, `threads`,
`bend`), the gate suite a lane must pass, and the frozen reference a task must
carry. `reference/report_run.py --against` keys on the stage path and prints
`mismatch`/`lane changed` rather than subtracting numbers across variants.
A lane change is a banner; it is never a percentage and never a quiet
improvement. No number is published for a lane that has not passed its gate.

## 4. Records and weights are versioned, never rewritten silently

Burn keeps weights in a separate Burnpack file with explicit load strategies
(file, embedded, bytes), and the importer writes readable source so a model
survives a toolchain change. Our run record is versioned the same way:
`schema: mlbend.experiment/3`, with every run kept under `results/raw/` and
the last one in `results/latest/`. Readers key on fields, so a `/1` record
with no `stages`/`variant`/`peaks` still renders — it prints `none declared`,
not a fabricated zero.

Two consequences we hold to: a corrected floor goes in `bounds-<task>.json`
via `bounds_for` and a stale path is refused; and `reference/test_record_compat.py`
renders an old record on every full check so the compatibility claim is a
measured one, not a sentence in this file.

Rule 3 is now a page as well: `make lane-matrix` reads the committed records
and writes `results/lane-matrix.md`, a task/lane table with the frozen
reference and PASS state of each cell (20/20 compliant as generated). It runs
nothing and invents nothing. `make oracle-boundary` is the static gate for
rule 1: no task program calls an oracle, the runner's binary runs before its
reference function, and bend records charge `reference_seconds` separately.

## What this does not ask for

No ONNX package, no model importer, no new backend and no version bump. The
four rules describe boundaries we already keep; the gate is there so a future
change cannot quietly cross them.
