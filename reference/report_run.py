"""One page per run: the report card and the timing, and `--against` (B2).

  reference/report_run.py --task diffusion-vit
  reference/report_run.py --task diffusion-vit --against results/raw/....json

The page is a reading of the run record `reference/experiment.py` writes; it
measures nothing. Its order is the reader's: who ran, what the run cost, the
stages (with floors once B3 has them), what is on the table, the worklist, then
the metric table and PASS/FAIL. `--against` keys on the stage path and refuses
to subtract across a lane, trainer, thread, Bend or machine change: those print
as a banner and the rows say `lane changed`, not a delta.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- text ----

def ms_text(ms):
    if ms is None:
        return "unavailable"
    if ms < 1.0:
        return "<1"
    return f"{ms:.1f}"


def share_text(ms, total):
    if not total:
        return "unavailable"
    return f"{100.0 * ms / total:.1f}%"


def show_metric(value, sd=None, digits=6):
    if value is None:
        return "-"
    if isinstance(value, float):
        text = f"{value:.{digits}g}"
        return text if not sd else f"{text} ± {sd:.2g}"
    return str(value)


def pad(text, width, left=False):
    text = str(text)
    return text.rjust(width) if left else text.ljust(width)


# ---- floors (filled by B3) ----

def floor_us(record, path):
    """The floor of a stage in microseconds, or None.

    B2 ships with empty bounds/peaks, so this returns None and the page prints
    `unavailable`; B3 fills `reference/bounds.py` and the task's `bounds`.
    """
    try:
        import bounds
    except ImportError:
        return None
    return bounds.floor_us(record, path)


# ---- the page ----

def page(record, against=None):
    lines = []
    variant = record.get("variant") or {}
    task = record["task"]
    threads = variant.get("threads")
    lines.append(
        f"task     {task['name']} · {variant.get('trainer', record.get('trainer', '?'))}"
        f" · {variant.get('lane', 'cpu')} · threads {threads if threads else 'unavailable'}"
        f" · bend {record.get('bend', {}).get('version', '?')}"
        f" · {record.get('machine', {}).get('cpu_count', '?')} cpus")
    git = record.get("git") or {}
    seeds = record.get("seeds") or []
    dataset = (task.get("dataset") or {}).get("sha256", "")
    lines.append(
        f"record   {git.get('branch', '?')} {git.get('commit', '?')}"
        + (" (dirty)" if git.get("dirty") else "")
        + f" · seeds {len(seeds)}" + (f" ({seeds[0]}..{seeds[-1]})" if len(seeds) > 1 else "")
        + (f" · dataset {dataset[:12]}" if dataset else ""))
    run = record.get("run", {})
    fetch = run.get("fetch_seconds", 0.0)
    lines.append(
        f"run      {run.get('seconds_min', 0.0):.3f} s for the task"
        f" · build {run.get('build_seconds', 0.0):.3f} s"
        f" · reference {run.get('reference_seconds', 0.0):.3f} s"
        f" · fetch {fetch:.3f} s" + (" warm" if not fetch else ""))

    stages = record.get("stages") or []
    source = record.get("stages_source")
    if stages:
        root = stages[0]["ms"]
        if source:
            lines.append(
                f"stages   {ms_text(root)} ms · {source.get('program', '?')}"
                f" (median of {source.get('repeats', 1)}) · {source.get('note', 'a stage split')}")
        else:
            lines.append(f"stages   {ms_text(root)} ms")
        lines.append("timings  wall from IO.now(); a stage below 1 ms prints <1 ms")
        peaks = record.get("peaks") or {}
        if not peaks:
            lines.append("peaks    not probed: no stage is scored, and no floor is printed")
        else:
            lines.append(
                f"peaks    {peaks.get('read_gbs', '?')} GB/s read, {peaks.get('write_gbs', '?')} GB/s written"
                f" · {peaks.get('flops_gflops', '?')} GFLOP/s"
                f" · one launch {peaks.get('launch_us', '?')} us (not charged)")
        lines.extend(notes(record))
        lines.append("")
        lines.append(pad("component", 34) + pad("ms", 8, True) + pad("share", 8, True)
                     + pad("kind", 8, True) + pad("bound", 8, True)
                     + pad("achieved", 10, True) + pad("floor ms", 9, True)
                     + pad("gain ms", 9, True))
        lines.append("-" * 94)
        lines.extend(stage_rows(record, stages, root))
        lines.append("")
        lines.extend(totals(record, stages, root))
        lines.extend(ladder(record, stages, root))
        lines.extend(worklist(record, stages))
    else:
        lines.append("stages   none declared: this task writes no stage split")
    lines.extend(metric_table(record))
    lines.append("")
    lines.append("PASS" if record.get("passed") else "FAIL")
    if against is not None:
        lines.append("")
        lines.extend(comparison(record, against))
    return "\n".join(lines)


def notes(record):
    out = []
    for note in record.get("notes") or []:
        out.append(f"         {note}")
    return out


def stage_rows(record, stages, root):
    bounds = record.get("bounds") or {}
    rows = []
    for entry in stages:
        path = entry["path"]
        ms = entry["ms"]
        indent = "  " * entry["depth"]
        bound = bounds.get(path)
        floor = floor_us(record, path)
        leaf = not any(other["path"].startswith(path + "/") for other in stages)
        if bound is None and floor is None:
            b, achieved, fl, gain = "-", "-", "-", "-"
        elif floor is None:
            b, achieved, fl, gain = "-", "unavailable", "-", "-"
        else:
            b = "both" if bound and bound.get("read") and bound.get("flops") else (
                "array" if bound and bound.get("flops") else "bytes")
            pct = 100.0 * floor / (ms * 1000.0) if ms else None
            if pct is None:
                achieved = "unavailable"
            elif pct > 100.0:
                achieved = f"*{pct:.1f}%"
            else:
                achieved = f"{pct:.1f}%"
            fl = f"{floor / 1000.0:.2f}"
            gain = ("unavailable" if not leaf else (
                f"{(ms * 1000.0 - floor) / 1000.0:.2f}" if ms * 1000.0 >= floor else "*"))
        rows.append(
            pad(indent + path, 34) + pad(ms_text(ms), 8, True)
            + pad(share_text(ms, root), 8, True) + pad(entry.get("kind", "wall"), 8, True)
            + pad(b, 8, True) + pad(achieved, 10, True) + pad(fl, 9, True) + pad(gain, 9, True))
    return rows


def totals(record, stages, root):
    host = sum(entry["ms"] for entry in stages if entry.get("kind") == "host")
    children = sum(entry["ms"] for entry in stages if entry["depth"] == 1)
    unaccounted = max(0.0, root - children)
    counts = record.get("counts") or {}
    launches = None
    for value in counts.values():
        if isinstance(value, dict) and "launches" in value:
            launches = (launches or 0) + value["launches"]
    unclaimed = None
    for value in counts.values():
        if isinstance(value, dict) and "unclaimed_ms" in value:
            unclaimed = (unclaimed or 0.0) + value["unclaimed_ms"]
    return [
        "",
        pad("on device", 30) + pad(ms_text(root), 8, True) + " ms   "
        + (f"{launches} launches" if launches is not None else "launches unavailable"),
        pad("host", 30) + pad(ms_text(host), 8, True) + " ms",
        pad("unaccounted", 30) + pad(f"{unaccounted:.1f}", 8, True) + " ms   stages that are not split",
        pad("unclaimed", 30) + pad("unavailable" if unclaimed is None else f"{unclaimed:.1f} ms", 8, True)
        + "   kernels no stage claimed",
    ]


def ladder(record, stages, root):
    floors = [floor_us(record, entry["path"]) for entry in stages if entry["depth"] > 0]
    if not record.get("peaks") or not record.get("bounds") or any(f is None for f in floors):
        return ["", "  how much is on the table   unavailable: a floor needs bounds and measured peaks"]
    host = sum(entry["ms"] for entry in stages if entry.get("kind") == "host")
    gains = sum(max(0.0, entry["ms"] * 1000.0 - (floor_us(record, entry["path"]) or 0.0))
                for entry in stages
                if entry["depth"] > 0
                and not any(other["path"].startswith(entry["path"] + "/") for other in stages)
                and floor_us(record, entry["path"]) is not None)
    return ["", f"  how much is on the table{ms_text((host * 1000.0 + gains) / 1000.0):>20} ms",
            f"  {'host':<12}{ms_text(host):>8}   less host work",
            f"  {'kernels':<12}{ms_text(gains / 1000.0):>8}   a faster kernel or a fuser"]


def worklist(record, stages):
    try:
        import worklist as work
    except ImportError:
        return ["", "worklist", "  unavailable: the levers are B4"]
    return work.section(record, stages)


def metric_table(record):
    lines = ["", f"{'metric':<18}{'bend':>28}{'reference':>28}{'delta':>22}  ok"]
    for name, row in (record.get("metrics") or {}).items():
        bend = show_metric(row.get("bend"), row.get("bend_sd"))
        want = show_metric(row.get("reference"), row.get("reference_sd"))
        delta = show_metric(row.get("delta"), row.get("delta_sd"), 4)
        ok = "ok" if row.get("pass") else "FAIL"
        lines.append(f"{name:<18}{bend:>28}{want:>28}{delta:>22}  {ok}")
    return lines


# ---- the comparison ----

def mismatches(now, was):
    out = []
    nv, wv = now.get("variant"), was.get("variant")
    if not nv or not wv:
        out.append("variant: not recorded on the older side" if not wv else "variant: not recorded here")
    else:
        for key in ("trainer", "lane", "threads"):
            if nv.get(key) != wv.get(key):
                out.append(f"{key}: {wv.get(key)} -> {nv.get(key)}")
    if (now.get("bend") or {}).get("version") != (was.get("bend") or {}).get("version"):
        out.append(f"bend: {(was.get('bend') or {}).get('version')} -> {(now.get('bend') or {}).get('version')}")
    if (now.get("machine") or {}).get("platform") != (was.get("machine") or {}).get("platform"):
        out.append("machine: different platform")
    if (now.get("task", {}).get("dataset") or {}).get("sha256") != \
            (was.get("task", {}).get("dataset") or {}).get("sha256"):
        out.append("dataset: different bytes")
    return out


def comparison(now, was):
    was_variant = was.get("variant") or {}
    lines = [f"against  {was.get('created_at', '?')}"
             f" ({was_variant.get('trainer', was.get('trainer', '?'))},"
             f" {was_variant.get('lane', 'not recorded')})"]
    bad = mismatches(now, was)
    if bad:
        lines.append("mismatch " + "; ".join(bad))
    now_stages = now.get("stages") or []
    was_stages = {entry["path"]: entry["ms"] for entry in (was.get("stages") or [])}
    if not now_stages:
        lines.append("no stages in this record to compare")
        return lines
    lines.append(pad("component", 34) + pad("was ms", 9, True) + pad("now ms", 9, True)
                 + pad("delta", 9, True) + "  note")
    lines.append("-" * 94)
    for entry in now_stages:
        path = entry["path"]
        old = was_stages.get(path)
        indent = "  " * entry["depth"]
        if old is None:
            note, old_text, delta = "added", "-", "-"
        else:
            old_text = ms_text(old)
            if bad:
                note, delta = "lane changed", "-"
            else:
                delta = f"{entry['ms'] - old:+.1f}"
                note = ""
        lines.append(pad(indent + path, 34) + pad(old_text, 9, True)
                     + pad(ms_text(entry["ms"]), 9, True) + pad(delta, 9, True) + "  " + note)
    now_paths = {entry["path"] for entry in now_stages}
    for path in was_stages:
        if path not in now_paths:
            lines.append(pad(path, 34) + pad(ms_text(was_stages[path]), 9, True)
                         + pad("-", 9, True) + pad("-", 9, True) + "  gone")
    return lines


# ---- main ----

def read(path):
    with open(path) as f:
        return json.load(f)


def latest_path(task, numpy=False):
    directory = "latest-numpy" if numpy else "latest"
    path = os.path.join(ROOT, "results", directory, task + ".json")
    if not os.path.isfile(path):
        raise SystemExit(f"no record for {task}: {path} does not exist (run make experiment TASK={task})")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", nargs="?", help="a record path; with --task, the latest of that task")
    parser.add_argument("--task", help="print the latest record of this task")
    parser.add_argument("--numpy", action="store_true", help="read results/latest-numpy/ with --task")
    parser.add_argument("--against", help="a saved record to hold it against")
    args = parser.parse_args()
    path = args.record or (latest_path(args.task, args.numpy) if args.task else None)
    if not path:
        parser.error("give a record or --task")
    record = read(path)
    against = read(args.against) if args.against else None
    print(page(record, against))
    return 0


if __name__ == "__main__":
    sys.exit(main())
