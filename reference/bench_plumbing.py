"""Where one ViT training step's seconds go.

  reference/.venv/bin/python reference/bench_plumbing.py

The GPU lane runs the model with the same wall time as the CPU lane
(ISSUE-018). This benchmark splits a real step into its parts, using the
example's own functions: parameter initialisation, the row build, the
list-to-matrix conversion, one forward, the target build, and the full step.
It writes `results/plumbing-<machine>.json` and `results/PLUMBING.md`.

A step time belongs to the machine and kernel it was measured on, so the record
is keyed by machine and the page prints one row per machine: two rows are two
machines, never a trend.
"""
import json
import os
import platform
import re
import glob
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

PROGRAM = os.path.join(ROOT, "bench/plumbing/main.bend")
REPEATS = 3

LINE = re.compile(r"^([a-z_0-9]+): ([0-9.eE+-]+)$")


def run_once(binary):
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the benchmark did not run:\n" + out.stdout + out.stderr)
    values = {}
    for line in out.stdout.splitlines():
        match = LINE.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def slug():
    return re.sub(r"[^A-Za-z0-9._-]+", "-", platform.platform()).strip("-")


def history():
    """Every machine's record, newest first, for the page's table."""
    rows = []
    for path in glob.glob(os.path.join(ROOT, "results", "plumbing-*.json")):
        try:
            with open(path) as f:
                found = json.load(f)
        except (OSError, ValueError):
            continue
        rows.append((found.get("created_at") or "", found))
    return [row for _, row in sorted(rows, reverse=True)]


def main():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-plumbing-bench")
    build = subprocess.run([BEND, "-o", binary, PROGRAM], capture_output=True, text=True,
                           env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the benchmark did not build:\n" + build.stdout + build.stderr)

    runs = [run_once(binary) for _ in range(REPEATS)]
    keys = sorted(runs[0])
    median = {key: statistics.median(run[key] for run in runs) for key in keys}

    step = median["full_step_x3_ms"] / 3.0
    from_list = max(0.0, median["step_in_ms"] - median["rows_ms"])
    backward = max(0.0, step - median["step_in_ms"] - median["step_tgt_ms"] - median["forward_ms"])
    plumbing = median["step_in_ms"] + median["step_tgt_ms"]
    derived = {
        "step_ms": step,
        "from_list_ms": from_list,
        "backward_update_ms": backward,
        "plumbing_ms": plumbing,
        "plumbing_share": plumbing / step,
        "backward_share": backward / step,
        "forward_share": median["forward_ms"] / step,
    }

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "program": "bench/plumbing/main.bend",
        "what": "one ViT training step, split by IO.now()", 
        "params": int(median["params"]),
        "repeats": REPEATS,
        "runs": runs,
        "median_ms": {key: value for key, value in median.items() if key != "params"},
        "derived_ms": derived,
        "machine": platform.platform(),
        "cpus": os.cpu_count(),
        "bend": "2.0.38",
    }
    machine_path = os.path.join(ROOT, "results", f"plumbing-{slug()}.json")
    with open(machine_path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")

    m = median
    rows = ["| machine | measured | step ms | forward ms | rows ms |", "|---|---|---:|---:|---:|"]
    for found in history():
        med = found.get("median_ms") or {}
        if not med:
            continue
        rows.append(f"| `{found.get('machine', '?')}` | {(found.get('created_at') or '')[:10]} | "
                    f"{(found.get('derived_ms') or {}).get('step_ms', 0):.0f} | "
                    f"{med.get('forward_ms', 0):.0f} | {med.get('rows_ms', 0):.0f} |")
    rows = "\n".join(rows)
    card = f"""# The plumbing split of one ViT training step

`make plumbing` builds and runs `bench/plumbing/main.bend` {REPEATS} times and
keeps the median of each phase. It imports the real example (`examples/diffusion/vit`)
and calls its own functions, so these are the numbers the model actually pays.

Machine: {platform.platform()} · {os.cpu_count()} CPUs · Bend 2.0.38

| phase | ms |
|---|---:|
| `init_params` (once per process) | {m['init_params_ms']:.0f} |
| `rows` (1024 rows before the matrix) | {m['rows_ms']:.0f} |
| `step_in` (batch, noise, mix, rows, matrix) | {m['step_in_ms']:.0f} |
| `Mat.from_list` (derived: step_in - rows) | {from_list:.0f} |
| `forward` (the model on that matrix) | {m['forward_ms']:.0f} |
| `step_tgt` (noise, patches, matrix) | {m['step_tgt_ms']:.0f} |
| `train_new` (grad and optimizer buffers) | {m['train_new_ms']:.0f} |
| **one full step** | **{step:.0f}** |

Shares of the step: input and target construction **{plumbing / step:.0%}**,
forward {m['forward_ms'] / step:.0%}, loss + backward + optimizer
{backward / step:.0%}.

## Every machine measured, newest first

A step time is comparable only inside its own row: the kernel version, the CPU
governor and the load are part of the number, and a row is the whole context.

{rows}

The step is now arithmetic. `step_in` and `step_tgt` together are
{plumbing:.0f} ms of {step:.0f} ms ({plumbing / step:.0%}); the forward is
{m['forward_ms']:.0f} ms and loss + backward + optimizer are {backward:.0f} ms.
`init_params` costs {m['init_params_ms']:.0f} ms once per process.

The history below is quoted with its machine: 7.2.9 on 2026-10-30, not this
one. Before the builder fix (that machine, the same program) a step was 543 ms, with
77% in the input and target build, and `init_params` was 25.3 s. The builders
appended into a growing list (O(n²)) and the tokeniser called `at()` 64 times
per digit. With cons + one reverse and a linear 16-pixel tokeniser the build
falls to {plumbing:.0f} ms, and the values are bit-identical: the full 20,480
value row list and the first three losses were compared against the old
version.

The device is worth re-measuring now: the forward and backward are
{(m['forward_ms'] + backward) / step:.0%} of the step, and the host no longer
hides them.
"""
    with open(os.path.join(ROOT, "results/PLUMBING.md"), "w") as f:
        f.write(card)

    print(card)


if __name__ == "__main__":
    main()
