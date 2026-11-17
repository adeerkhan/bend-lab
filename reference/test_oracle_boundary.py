"""Gate: no oracle in the run path (the Burn lesson, rule 1).

  reference/.venv/bin/python reference/test_oracle_boundary.py

A task's Bend program is the run path; the oracle (NumPy, PyTorch, ONNX
Runtime) is called only after the binary ran, and its time is charged to
`run.reference_seconds`, not to the program. This gate checks that boundary
statically, so a future convenience import cannot quietly move the oracle
into the run:

  - no task program imports or names an oracle;
  - `evaluate` runs the binary before calling `cfg["reference"]`;
  - a committed bend run records reference_seconds as a separate number.

It reads source and committed records; it writes nothing.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import experiment  # noqa: E402


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return bool(ok)


def main():
    ok = True
    # Bend programs legitimately *mention* PyTorch/NumPy in comments (the
    # oracle they were checked against), and `bendtorch` contains "torch".
    # Only an actual import or a call is a boundary violation.
    foreign = re.compile(
        r"(?:^\s*import\s+(?:torch|numpy|sklearn|onnxruntime)\b"
        r"|^\s*from\s+(?:torch|numpy|sklearn|onnxruntime)\b"
        r"|(?<!bend)\btorch\.|\bnp\.|\bonnxruntime\.)", re.M)

    # the run path: every task's Bend program, and the runner's run_binary body
    bad = []
    for name, cfg in experiment.TASKS.items():
        program = cfg.get("program")
        if not program or not program.endswith(".bend"):
            continue
        path = os.path.join(ROOT, program)
        if not os.path.isfile(path):
            bad.append(f"{name}: {program} missing")
            continue
        code = "\n".join(l for l in open(path) if not l.lstrip().startswith("#"))
        if foreign.search(code):
            bad.append(f"{name}: {program} calls an oracle")
    ok &= check("no task program names an oracle", not bad,
                "; ".join(bad) if bad else f"{len(experiment.TASKS)} task programs checked")

    runner = open(os.path.join(ROOT, "reference/experiment.py")).read()
    run_at = runner.find("output, times = run_binary(")
    ref_at = runner.find("reference_values = cfg[\"reference\"](parsed)")
    ok &= check("the binary runs before the oracle is called",
                run_at > 0 and ref_at > run_at,
                f"run_binary at {run_at}, reference at {ref_at}")

    # committed bend records keep the reference's time as its own field
    checked = 0
    missing = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results/latest/*.json"))):
        with open(path) as f:
            record = json.load(f)
        variant = record.get("variant") or {}
        if variant.get("trainer") != "bend":
            continue
        run = record.get("run") or {}
        if "reference_seconds" not in run or "seconds_min" not in run:
            missing.append(os.path.basename(path))
        checked += 1
    ok &= check("bend records charge reference time separately", not missing,
                f"{checked} records, {len(missing)} missing the field"
                + (": " + ", ".join(missing[:3]) if missing else ""))

    if not ok:
        print("FAIL: oracle boundary")
        sys.exit(1)
    print("PASS: the oracle stays outside the run path")


if __name__ == "__main__":
    main()
