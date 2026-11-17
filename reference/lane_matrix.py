"""The lane compliance matrix: one page, from committed records only.

  reference/.venv/bin/python reference/lane_matrix.py

Each task/lane cell says whether a run record exists, whether that record
passed, and whether a frozen reference is on file. Nothing here runs a task;
every number is read from `results/latest/*.json`. A lane with no frozen
reference or a failing record prints `unavailable`/`FAIL`, never a fabricated
zero. See `docs/agents/oracle-boundary.md` (rule 3): a gate still has to pass
before a lane's numbers are published.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import references  # noqa: E402

OUT = os.path.join(ROOT, "results", "lane-matrix.md")


def main():
    cells = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results/latest/*.json"))):
        with open(path) as f:
            record = json.load(f)
        variant = record.get("variant") or {}
        name = record.get("task", {}).get("name", os.path.basename(path))
        cells.append({
            "task": name,
            "lane": variant.get("lane") or "unrecorded",
            "trainer": variant.get("trainer") or record.get("trainer") or "?",
            "bend": variant.get("bend") or (record.get("bend") or {}).get("version", "?"),
            "schema": record.get("schema", "?"),
            "seeds": len(record.get("seeds") or []),
            "frozen": bool(references.read(name)),
            "passed": bool(record.get("passed")),
        })
    cells.sort(key=lambda c: (c["lane"], c["task"]))
    good = [c for c in cells if c["frozen"] and c["passed"]]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        "# Lane compliance matrix",
        "",
        f"Generated {now} by `make lane-matrix` from `results/latest/*.json` —",
        "no task was run to produce this page. A cell is compliant when its",
        "record passed and a frozen reference is on file; per-lane gate suites",
        "(for example `docs/mamba3-lanes.md`) must still pass before numbers",
        "are published. A mismatch across lanes is a banner, not a delta.",
        "",
        f"**{len(good)}/{len(cells)} cells compliant**",
        "",
        "| task | lane | trainer | bend | schema | seeds | frozen | record |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c['task']} | {c['lane']} | {c['trainer']} | {c['bend']} "
            f"| {c['schema']} | {c['seeds']} "
            f"| {'yes' if c['frozen'] else 'unavailable'} "
            f"| {'PASS' if c['passed'] else 'FAIL'} |")
    lines += ["", "## What this page does not say", "",
              "It does not rank lanes, replace the gates, or compare numbers across",
              "lanes. It says only that each recorded lane has an independent",
              "reference and a passing run.", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.relpath(OUT, ROOT)} — {len(good)}/{len(cells)} compliant")
    return 0 if len(good) == len(cells) else 1


if __name__ == "__main__":
    sys.exit(main())
