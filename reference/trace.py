"""Run a task's program with `--trace` and write one JSON line per step, live.

  reference/trace.py --task glm53-hybrid128 --steps 20

The trace is the live feed `reference/dash.py` reads and the evidence a record's
stage split is held against. The program prints one object per line and nothing
else in trace mode; this runner writes each line and flushes it as it arrives,
so `dash.py --watch` sees the run while it is running. A line that is not JSON
is refused rather than written.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import experiment  # noqa: E402

KEYS = ("step", "tokens", "loss", "ms")


def record_of(line, number):
    try:
        record = json.loads(line)
    except ValueError:
        raise SystemExit(f"trace: line {number} is not JSON: {line!r}")
    missing = [key for key in KEYS if key not in record]
    if missing:
        raise SystemExit(f"trace: line {number} misses {', '.join(missing)}: {line!r}")
    return {key: record[key] for key in KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(experiment.TASKS))
    parser.add_argument("--steps", type=int, help="override the task's --steps")
    parser.add_argument("--out", help="the trace path; default results/<task>/trace.jsonl")
    parser.add_argument("--threads", type=int, help="the runtime's thread count")
    args = parser.parse_args()

    cfg = experiment.TASKS[args.task]
    extra = list(cfg.get("args") or [])
    if args.steps is not None:
        extra += ["--steps", str(args.steps)]
    extra.append("--trace")

    out = args.out or os.path.join(ROOT, "results", args.task, "trace.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    binary, build_seconds = experiment.build_bend(args.task, cfg)

    start = time.perf_counter()
    proc = subprocess.Popen([binary, "0"] + extra + experiment.rts(args.threads),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=experiment.ENV, cwd=ROOT)
    records = []
    with open(out, "w") as f:
        for number, line in enumerate(proc.stdout, 1):
            if not line.strip():
                continue
            record = record_of(line.strip(), number)
            records.append(record)
            f.write(json.dumps(record) + "\n")
            f.flush()
    stderr = proc.stderr.read()
    code = proc.wait()
    seconds = time.perf_counter() - start
    if code != 0:
        raise SystemExit(f"trace: the program exited {code}:\n{stderr}")
    if not records:
        raise SystemExit("trace: the program printed no JSON lines (does it support --trace?)")

    ms = [record["ms"] for record in records]
    print(f"trace: {len(records)} lines, {records[0]['step']}..{records[-1]['step']}"
          f" · median {statistics.median(ms):.0f} ms")
    print(f"trace: build {build_seconds:.1f} s · run {seconds:.3f} s"
          f" · {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
