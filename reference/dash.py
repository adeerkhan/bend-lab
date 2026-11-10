"""The live page of a training trace: steps, loss, time, and the record's stages.

  reference/dash.py --task glm53-hybrid128
  reference/dash.py --task glm53-hybrid128 --watch
  reference/dash.py --task glm53-hybrid128 --watch --sample-pid 12345

Reads `results/<task>/trace.jsonl` (written by `reference/trace.py`) and the
task's latest record, and prints the layout of `reference/report_run.py` with
the live block on top: the loss curve, the step time and tokens/s, and, when a
record exists, its stage table and unaccounted line.

`--watch` redraws as lines arrive and stops when the process named by
`--sample-pid` exits, or when the file has not grown for `--idle` seconds.
`--sample-pid` also adds the process's resident and peak memory, read from
`/proc/<pid>/status`: the dashboard does not need the library to report it.

No dependency beyond the standard library, and nothing here decides pass/fail.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import report_run  # noqa: E402

BLOCKS = "▁▂▃▄▅▆▇█"  # eight levels, low to high


def read_trace(path):
    """Every complete JSON line; a partial last line while the file grows is skipped."""
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if all(key in record for key in ("step", "tokens", "loss", "ms")):
                    records.append(record)
    except OSError:
        return []
    return records


def spark(values, width=54):
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    low, high = min(values), max(values)
    if high <= low:
        return BLOCKS[0] * len(values)
    return "".join(BLOCKS[min(len(BLOCKS) - 1, int((value - low) / (high - low) * len(BLOCKS)))]
                   for value in values)


def sample(pid):
    """RSS, peak RSS and the CPU tick count from /proc, or None."""
    try:
        status = open(f"/proc/{pid}/status").read()
        stat = open(f"/proc/{pid}/stat").read()
    except OSError:
        return None
    values = {}
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            values[key.rstrip(":")] = int(value.split()[0])
    # the comm field can hold spaces and parentheses; the state is after the last ')'
    fields = stat[stat.rfind(")") + 2:].split()
    values["ticks"] = int(fields[11]) + int(fields[12])
    return values


def where(path):
    """The repository-relative path, or the absolute one when it is outside."""
    rel = os.path.relpath(path, ROOT)
    return rel if not rel.startswith("..") else path


def record_lines(record):
    stages = record.get("stages") or []
    if not stages:
        return ["stages   unavailable: the record declares no split"]
    root = stages[0]["ms"]
    source = record.get("stages_source") or {}
    lines = [f"stages   {report_run.ms_text(root)} ms · {source.get('kind', 'split')}"
             f" {source.get('program', '?')}"
             f" (median of {source.get('repeats', 1)}) · "
             f"{source.get('note') or 'a stage split'}"]
    lines.append(report_run.pad("component", 34) + report_run.pad("ms", 8, True)
                 + report_run.pad("share", 8, True) + report_run.pad("kind", 8, True))
    lines.extend(report_run.stage_rows(record, stages, root))
    lines.extend(line for line in report_run.totals(record, stages, root) if line)
    return lines


def page(task, trace_path, records, record, memory=None, cpu=None, rss_series=None):
    lines = [f"task     {task} · {len(records)} steps"
             + (f" · {sum(r['tokens'] for r in records):,} tokens" if records else "")]
    lines.append(f"trace    {where(trace_path)}"
                 + ("" if records else " · unavailable (run make trace TASK=" + task + ")"))
    if records:
        first, last = records[0], records[-1]
        losses = [r["loss"] for r in records]
        lines.append(f"loss     first {first['loss']:.6g} · last {last['loss']:.6g}"
                     f" · best {min(losses):.6g} · worst {max(losses):.6g}")
        lines.append("         " + spark(losses))
        ms = [r["ms"] for r in records]
        tokens = records[0]["tokens"] or 0
        rates = [r["tokens"] / (r["ms"] / 1000.0) for r in records if r["ms"]]
        lines.append(f"step     median {statistics.median(ms):.0f} ms · min {min(ms)}"
                     f" · max {max(ms)} · {tokens} tokens each")
        lines.append("         " + spark(ms))
        if rates:
            lines.append(f"rate     median {statistics.median(rates):.1f} tokens/s")
        lines.append(f"wall     {sum(ms) / 1000.0:.3f} s in the loop")
    else:
        lines.append("loss     unavailable")
        lines.append("step     unavailable")
        lines.append("rate     unavailable")
    if memory is None:
        lines.append("memory   unavailable: no --sample-pid")
    elif not memory:
        lines.append("memory   unavailable: /proc/<pid>/status is not readable")
    else:
        peak = f"{memory['VmHWM']:,} KiB peak" if "VmHWM" in memory else "peak unavailable"
        now = f"{memory['VmRSS']:,} KiB now" if "VmRSS" in memory else "now unavailable"
        used = f" · {cpu:.1f}% cpu" if cpu is not None else ""
        lines.append(f"memory   {now} · {peak}{used}")
        series = rss_series or ([memory["VmRSS"]] if "VmRSS" in memory else [])
        if len(series) > 1:
            lines.append("         " + spark(series))
    if record is None:
        lines.append("record   unavailable: results/latest/" + task + ".json")
    else:
        lines.append(f"record   {where(record_path_for(task))}"
                     f" · {'PASS' if record.get('passed') else 'FAIL'}")
        lines.extend(record_lines(record))
        stages = record.get("stages") or []
        step = next((entry for entry in stages if entry["path"] == "run/step"), None)
        if records and step is not None:
            traced = statistics.median([r["ms"] for r in records])
            lines.append(f"compare  trace median {traced:.0f} ms · record step"
                         f" {step['ms']:.0f} ms · {traced - step['ms']:+.0f} ms")
            lines.append("         different runs: the trace is one process, the split"
                         " another; not a delta")
    return "\n".join(lines)


def record_path_for(task):
    return os.path.join(ROOT, "results", "latest", task + ".json")


def read_record(task, path=None):
    path = path or record_path_for(task)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def alive(pid):
    return os.path.isdir(f"/proc/{pid}") if pid else False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--trace", help="the trace path; default results/<task>/trace.jsonl")
    parser.add_argument("--record", help="the record path; default results/latest/<task>.json")
    parser.add_argument("--watch", action="store_true", help="redraw as lines arrive")
    parser.add_argument("--sample-pid", type=int, help="add the memory of this process")
    parser.add_argument("--idle", type=float, default=3.0,
                        help="with --watch: stop after this many seconds without growth")
    args = parser.parse_args()

    trace_path = args.trace or os.path.join(ROOT, "results", args.task, "trace.jsonl")
    records = read_trace(trace_path)
    record = read_record(args.task, args.record)
    memory = (sample(args.sample_pid) or {}) if args.sample_pid and not args.watch else None
    block = page(args.task, trace_path, records, record, memory)
    print(block)
    if not records and not args.watch:
        return 1
    if not args.watch:
        return 0

    seen = len(records)
    last_change = time.monotonic()
    hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    memory, series, cpu = None, [], None
    previous, previous_at = None, time.monotonic()
    while True:
        time.sleep(0.5)
        records = read_trace(trace_path)
        record = read_record(args.task, args.record)
        if args.sample_pid:
            current = sample(args.sample_pid)
            now = time.monotonic()
            if current:
                memory = current
                if "VmRSS" in current:
                    series.append(current["VmRSS"])
                if previous and now > previous_at:
                    cpu = 100.0 * (current["ticks"] - previous["ticks"]) / hz / (now - previous_at)
                previous, previous_at = current, now
        if len(records) != seen:
            seen = len(records)
            last_change = time.monotonic()
            block = page(args.task, trace_path, records, record, memory, cpu, series)
            if sys.stdout.isatty():
                sys.stdout.write("\033[2J\033[H")
            print(block + ("\n" if not sys.stdout.isatty() else ""))
            sys.stdout.flush()
        if args.sample_pid and not alive(args.sample_pid):
            break
        if not args.sample_pid and time.monotonic() - last_change > args.idle:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
