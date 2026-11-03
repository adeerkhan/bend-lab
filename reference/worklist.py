"""The worklist: what a run could give back, each millisecond once (B4).

  reference/worklist.py <record>        # the section alone, for a quick look

A composite's gain contains its children's, so only leaves are ranked: a leaf
by what a faster kernel or a cheaper host path would recover, a host leaf by
its own time. The lever names the work, and says whose it is where the answer
is upstream (`bend-ml`, `BendTorch`). The set is closed: `less fetch`,
`less preprocess`, `less conversion`, `fewer launches`, `a faster kernel
(upstream)`, `less host work`. `fewer launches` is never invented: the section
says `unavailable`, because a count alone does not say how long the device
waited between launches, and those milliseconds are already listed under the
stages that paid them.
"""
import json
import sys

try:
    import bounds as bounds_mod
except ImportError:
    bounds_mod = None


class Entry:
    def __init__(self, ms, path, lever, why):
        self.ms = ms
        self.path = path
        self.lever = lever
        self.why = why


def leaf(stages, entry):
    return not any(other["path"].startswith(entry["path"] + "/") for other in stages)


def floor_ms(record, path):
    if bounds_mod is None:
        return None
    value = bounds_mod.floor_us(record, path)
    return None if value is None else value / 1000.0


def floor_and_reason(record, path):
    """(floor in ms, why it cannot be scored) — a missing floor is never a lever.

    A stage with no floor is an upper bound on what a faster kernel could give
    back, not a measurement of it: the worklist says `unavailable` and names the
    reason instead of pointing upstream at a number nobody measured.
    """
    floor = floor_ms(record, path)
    if floor is not None:
        return floor, None
    if bounds_mod is None:
        return None, "bounds.py is not importable"
    if not (bounds_mod.bounds_for(record).get(path)):
        return None, "the task declares no bound for this stage"
    _, why = bounds_mod.peaks_and_reason(record)
    return None, why or "the declared bound has no peak to score it"


def counts_of(record, path):
    return (record.get("counts") or {}).get(path) or {}


def host_lever(path):
    if any(word in path for word in ("input", "preprocess", "rows", "stage")):
        return "less preprocess"
    return "less host work"


def entries(record, stages):
    out = []
    run = record.get("run") or {}
    fetch_ms = (run.get("fetch_seconds") or 0.0) * 1000.0
    if fetch_ms >= 50.0:
        out.append(Entry(fetch_ms, "fetch", "less fetch",
                         f"{fetch_ms:.0f} ms filling or hashing the cache; the bytes are pinned"))
    for entry in stages:
        if entry["depth"] == 0 or not leaf(stages, entry):
            continue
        path = entry["path"]
        ms = entry["ms"]
        floor, no_floor = floor_and_reason(record, path)
        counts = counts_of(record, path)
        if entry.get("kind") == "once":
            out.append(Entry(ms, path, "less host work", "once per process (parameter initialisation)"))
            continue
        if entry.get("kind") == "host":
            conversion = counts.get("from_list_ms") or 0.0
            if conversion >= 1.0:
                out.append(Entry(conversion, path + " (from_list)", "less conversion",
                                 "the list-to-matrix build; build the matrix directly"))
            rest = ms - conversion
            if rest >= 1.0:
                rows = counts.get("rows_ms")
                if "target" in path:
                    why = "the target build"
                elif "input" in path:
                    why = "the row build" if rows is None else f"rows {rows:.0f} ms of it"
                else:
                    why = "host work inside the step"
                if conversion:
                    why += "; the conversion is host work too"
                out.append(Entry(rest, path, host_lever(path), why))
            continue
        if floor is None:
            out.append(Entry(ms, path, "unavailable", f"no floor: {no_floor}"))
        else:
            gain = max(0.0, ms - floor)
            pct = 100.0 * floor / ms if ms else 0.0
            basis = (bounds_mod.bounds_for(record).get(path) or {}).get("counts")
            counts_note = f"; the floor counts {basis}" if basis else ""
            out.append(Entry(gain, path, "a faster kernel (upstream)",
                             f"{pct:.1f}% of its floor{counts_note}; bend-ml/BendTorch"))
    out.sort(key=lambda e: e.ms, reverse=True)
    return out


def has_launch_count(record):
    for value in (record.get("counts") or {}).values():
        if isinstance(value, dict) and "launches" in value:
            return True
    return False


def section(record, stages):
    found = entries(record, stages)
    lines = ["", "worklist (what the run could give back, each millisecond once)"]
    if not found:
        lines.append("  unavailable: no stage could be scored")
    for item in found[:5]:
        lines.append(f"  {item.ms:>7.1f} ms  {item.path:<28}  {item.lever:<24}  {item.why}")
    if has_launch_count(record):
        lines.append("  fewer launches            unavailable: launches are counted,"
                     " but not the wait between them")
    else:
        lines.append("  fewer launches            unavailable: the record has no launch count")
    return lines


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: worklist.py <record.json>")
    with open(sys.argv[1]) as f:
        record = json.load(f)
    for line in section(record, record.get("stages") or []):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
