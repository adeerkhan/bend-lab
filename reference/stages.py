"""The stage split of a run: the pre-order list the report reads (B1).

A task may declare where its stage split comes from. `plumbing` builds a small
program that times the task's own step functions and prints
`<name>_ms: <milliseconds>` lines; the runner keeps the median of three runs
and turns those lines into stages. The program is the one in
`bench/plumbing/main.bend` unless the task names another.

The stages are inclusive: a parent's milliseconds contain its children's, and
the reader computes each share from the root. A stage below one millisecond is
kept as 0 and the report prints `<1 ms`; nothing here rounds a measurement up.
"""
import os
import re
import statistics
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

NAME_MS = re.compile(r"^([a-z_0-9]+)_ms: ([0-9.eE+-]+)$")
NAMED = re.compile(r"^([a-z_0-9]+): ([0-9.eE+-]+)$")


def named(text):
    """Every `name: number` line of a program's output, as floats."""
    values = {}
    for line in text.splitlines():
        match = NAMED.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def run_program(program, repeats=3):
    """Build `program` once, run it `repeats` times, and keep each key's median."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-stages")
    build = subprocess.run([BEND, "-o", binary, program],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"stages: {program} did not build:\n{build.stdout}{build.stderr}")
    runs = []
    for _ in range(max(1, repeats)):
        out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
        if out.returncode != 0:
            raise SystemExit(f"stages: {program} did not run:\n{out.stdout}{out.stderr}")
        runs.append(named(out.stdout))
    keys = sorted(set().union(*runs))
    return {key: statistics.median(run.get(key, 0.0) for run in runs) for key in keys}


def stage(path, depth, ms, kind):
    return {"path": path, "depth": depth, "ms": round(ms, 3), "kind": kind}


def tree(measure):
    """The stages of one measured step, from the plumbing program's own keys.

    `full_step_x3_ms` is three steps, so one step is a third. The root is the
    process's measured work: the one-time parameter initialisation plus one
    step. `loss_backward_opt` is what is left of the step once the input, the
    forward and the target are taken out, which is what the plumbing document
    already prints as the backward share.
    """
    step = measure["full_step_x3_ms"] / 3.0
    init = measure.get("init_params_ms", 0.0)
    step_in = measure.get("step_in_ms", 0.0)
    forward = measure.get("forward_ms", 0.0)
    target = measure.get("step_tgt_ms", 0.0)
    rest = max(0.0, step - step_in - forward - target)
    stages = [
        stage("run", 0, init + step, "wall"),
        stage("run/init", 1, init, "once"),
        stage("run/step", 1, step, "wall"),
        stage("run/step/input", 2, step_in, "host"),
        stage("run/step/forward", 2, forward, "wall"),
        stage("run/step/target", 2, target, "host"),
        stage("run/step/loss_backward_opt", 2, rest, "wall"),
    ]
    counts = {
        "run/step/input": {
            "rows_ms": round(measure.get("rows_ms", 0.0), 3),
            "from_list_ms": round(max(0.0, step_in - measure.get("rows_ms", 0.0)), 3),
        }
    }
    notes = [
        "clock: IO.now is milliseconds; a stage below 1 ms prints <1 ms",
        f"init_params {init:.0f} ms once per process (in run/init)",
        f"rows {measure.get('rows_ms', 0.0):.0f} ms and the matrix conversion "
        f"{counts['run/step/input']['from_list_ms']:.0f} ms sit inside step_in",
    ]
    if "params" in measure:
        notes.append(f"parameters {int(measure['params'])}")
    return stages, counts, notes


def capture(cfg, repeats=3):
    """The stages, counts and notes of a task that declares `stages`.

    Returns (stages, counts, notes, source). `source` is what the manifest says
    the split measured, so a reader never mistakes one step for the whole run.
    """
    spec = cfg["stages"]
    program = spec.get("program", "bench/plumbing/main.bend")
    measure = run_program(program, repeats)
    stages, counts, notes = tree(measure)
    source = {
        "kind": "plumbing",
        "program": program,
        "repeats": repeats,
        "note": "one step from this program, not the task's whole run",
    }
    return stages, counts, notes, source


def text(stages, counts=None):
    """The stage block of the report card, before the full page exists (B2)."""
    if not stages:
        return []
    root = stages[0]["ms"]
    lines = ["stages:"]
    for entry in stages:
        ms = entry["ms"]
        shown = "<1" if ms < 1.0 else f"{ms:.1f}"
        share = 100.0 * ms / root if root else None
        share_text = "unavailable" if share is None else f"{share:.1f}%"
        indent = "  " * (entry["depth"] + 1)
        lines.append(f"{indent}{entry['path']:<34}{shown:>8} ms  {share_text:>8}  {entry['kind']}")
    children = sum(entry["ms"] for entry in stages if entry["depth"] == 1)
    lines.append(f"{'  unaccounted':<36}{root - children:>8.1f} ms")
    return lines
