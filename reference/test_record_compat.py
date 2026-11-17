"""Gate: old run records still render (record v3, lesson 4).

  reference/.venv/bin/python reference/test_record_compat.py

The record carries `schema: mlbend.experiment/3`, but readers key on fields:
a `/1` record has no `stages`, `variant`, `peaks`, `bounds` or `notes`. Both
generations must render the reader's order — task, run, metric table, PASS —
and an old record must say `none declared` for stages rather than print a
fabricated zero. This gate reads committed records; it writes nothing.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import report_run  # noqa: E402


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return bool(ok)


def read(path):
    with open(path) as f:
        return json.load(f)


def main():
    ok = True
    old = sorted(glob.glob(os.path.join(ROOT, "results/raw/*.json")))
    old = [p for p in old if read(p).get("schema") == "mlbend.experiment/1"]
    new = os.path.join(ROOT, "results/latest/mamba3-tiny-lm.json")

    ok &= check("a schema /1 record is committed", bool(old),
                f"{len(old)} old records")
    ok &= check("a schema /3 record is committed", os.path.isfile(new),
                os.path.relpath(new, ROOT))
    if not old or not os.path.isfile(new):
        print("FAIL: no record to render")
        print("FAIL: record compatibility")
        sys.exit(1)

    for label, path in (("old /1", old[0]), ("new /3", new)):
        record = read(path)
        try:
            page = report_run.page(record)
        except Exception as exc:
            ok &= check(f"{label} renders", False, f"{type(exc).__name__}: {exc}")
            continue
        lines = page.splitlines()
        headline = next((l for l in lines if l.startswith("task ")), "")
        verdict = next((l for l in lines if l.strip() == "PASS"), "")
        ok &= check(f"{label} renders the reader's order",
                    bool(headline) and bool(verdict), headline[:90])
        # the real shape of the metric header is checked separately so a
        # layout edit cannot silently drop it
        ok &= check(f"{label} prints the metric table",
                    any(l.lstrip().startswith("metric ") for l in lines),
                    next((l.strip() for l in lines if l.lstrip().startswith("metric ")), "missing"))

    old_page = report_run.page(read(old[0]))
    ok &= check("the old record says stages are not declared",
                "none declared: this task writes no stage split" in old_page,
                "no fabricated stage, no zero")

    if not ok:
        print("FAIL: record compatibility")
        sys.exit(1)
    print("PASS: old and new run records render")


if __name__ == "__main__":
    main()
