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
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import experiment  # noqa: E402
import runs  # noqa: E402

KEYS = ("step", "tokens", "loss", "ms")


def record_of(line, number):
    try:
        record = json.loads(line)
    except ValueError:
        raise SystemExit(f"trace: line {number} is not JSON: {line!r}")
    missing = [key for key in KEYS if key not in record]
    if missing:
        raise SystemExit(f"trace: line {number} misses {', '.join(missing)}: {line!r}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(experiment.TASKS))
    parser.add_argument("--steps", type=int, help="override the task's --steps")
    parser.add_argument("--out", help="the trace path; default results/<task>/trace.jsonl")
    parser.add_argument("--threads", type=int, help="the runtime's thread count")
    parser.add_argument("--max-seconds", type=float, default=3600.0,
                        help="stop the run after this wall time (default one hour; 0 disables)")
    args = parser.parse_args()

    cfg = experiment.TASKS[args.task]
    extra = list(cfg.get("args") or [])
    if args.steps is not None:
        extra += ["--steps", str(args.steps)]
    extra.append("--trace")

    out = args.out or os.path.join(ROOT, "results", args.task, "trace.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    run_trace = os.path.join(ROOT, "results", args.task, "runs", f"{stamp}-{os.getpid()}.jsonl")
    os.makedirs(os.path.dirname(run_trace), exist_ok=True)
    record_path = os.path.join(ROOT, "results", "latest", args.task + ".json")
    try:
        run_id = runs.start(
            args.task, run_trace, None, f"{stamp} · pid {os.getpid()}",
            pid=os.getpid(), command=" ".join(sys.argv),
            config={"program": cfg["program"], "args": extra, "seed": 0,
                    "steps": args.steps, "variant": cfg.get("variant"),
                    "threads": args.threads},
            params=cfg.get("params"), kind="trace")
    except Exception as exc:
        print(f"trace: run store unavailable: {exc}")
        run_id = None
    binary, build_seconds = experiment.build_bend(args.task, cfg)

    start = time.perf_counter()
    proc = subprocess.Popen([binary, "0"] + extra + experiment.rts(args.threads),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=experiment.ENV, cwd=ROOT)
    if run_id is not None:
        try:
            runs.set_pid(run_id, proc.pid)
        except Exception:
            pass
    records = []
    aborted = []
    timer = None
    if args.max_seconds and args.max_seconds > 0:
        def stop():
            aborted.append(True)
            proc.terminate()
        timer = threading.Timer(args.max_seconds, stop)
        timer.start()
    try:
        with open(run_trace, "w") as rf, open(out, "w") as f:
            for number, line in enumerate(proc.stdout, 1):
                if not line.strip():
                    continue
                record = record_of(line.strip(), number)
                records.append(record)
                line_out = json.dumps(record) + "\n"
                rf.write(line_out); rf.flush()
                f.write(line_out); f.flush()
                if run_id is not None:
                    try:
                        runs.add_step(run_id, record)
                    except Exception as exc:
                        print(f"trace: run store append unavailable: {exc}")
                        run_id = None
    finally:
        if timer is not None:
            timer.cancel()
    stderr = proc.stderr.read()
    code = proc.wait()
    seconds = time.perf_counter() - start
    if run_id is not None:
        try:
            # the trace program writes no record of its own; the latest
            # record belongs to another run and is imported as that run
            runs.finish(run_id, "aborted" if aborted else "done", None)
        except Exception as exc:
            print(f"trace: run store finish unavailable: {exc}")
    if aborted:
        print(f"trace: budget {args.max_seconds:.0f} s exceeded after {len(records)} lines"
              f" and {seconds:.1f} s; the run was stopped")
        return 124
    if code != 0:
        raise SystemExit(f"trace: the program exited {code}:\n{stderr}")
    if not records:
        raise SystemExit("trace: the program printed no JSON lines (does it support --trace?)")

    ms = [record["ms"] for record in records]
    print(f"trace: {len(records)} lines, {records[0]['step']}..{records[-1]['step']}"
          f" · median {statistics.median(ms):.0f} ms")
    print(f"trace: build {build_seconds:.1f} s · run {seconds:.3f} s"
          f" · {os.path.relpath(run_trace, ROOT)}")
    print(f"trace: run {run_id} · latest {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
