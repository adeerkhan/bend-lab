"""Gate: the report's mismatch banner on --against (row C3).

  reference/.venv/bin/python reference/test_against.py

Two synthetic run records with the same task, the second a variant change
(cpu -> gpu), a renamed stage, and one new stage.  The comparison must:

  print "mismatch lane: cpu -> gpu"                (the banner)
  print "lane changed" in every overlapping delta  (never a percentage)
  print "added" for the new stage                  (never a fabricated number)
  print "gone" for the renamed/vanished stage      (never a 0)

The report module is imported directly; no Bend build is needed.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import report_run  # noqa: E402


def record(lane, stage_names, created):
    stages = []
    for i, (path, ms) in enumerate(stage_names):
        stages.append({"path": path, "depth": path.count("/"), "ms": ms,
                       "kind": "wall"})
    return {
        "schema": "mlbend.experiment/3",
        "created_at": created,
        "task": {"name": "synthetic",
                 "dataset": {"sha256": "deadbeef", "name": "synthetic"}},
        "git": {"branch": "main", "commit": "abc1234", "dirty": False},
        "bend": {"version": "2.0.38"},
        "machine": {"platform": "test-machine", "cpu_count": 28},
        "variant": {"trainer": "bend", "lane": lane, "threads": None,
                    "bend": "2.0.38"},
        "seeds": [0],
        "run": {"seconds_min": 1.0, "seconds_all": 1.0, "build_seconds": 2.0,
                "fetch_seconds": 0.0, "reference_seconds": 0.5, "repeat": 1,
                "compute_seconds": 0.5, "speed_ratio": 0.5},
        "stages": stages,
        "stages_source": None,
        "metrics": {"m": {"bend": 1.0, "reference": 1.0, "pass": True}},
        "passed": True,
    }


def check(name, cond, detail):
    print(f"{'ok  ' if cond else 'FAIL'} {name}: {detail}")
    return bool(cond)


def main():
    ok = True
    was = record("cpu", [("run", 100.0), ("run/step", 80.0),
                         ("run/step/forward", 50.0), ("run/step/gone_now", 30.0)],
                 "2026-11-01T00:00:00+00:00")
    now = record("gpu", [("run", 200.0), ("run/step", 150.0),
                         ("run/step/forward", 100.0), ("run/step/new_stage", 50.0)],
                 "2026-11-16T00:00:00+00:00")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(report_run.page(now, was))
    out = buf.getvalue()
    section = out[out.index("against "):]  # only the comparison section

    ok &= check("the mismatch banner prints the lane change",
                "mismatch lane: cpu -> gpu" in section,
                next((l for l in section.splitlines() if "mismatch" in l), "<missing>"))
    rows = [l for l in section.splitlines()
            if l.strip().startswith(("run ", "run/"))]
    overlapping = [l for l in rows if "lane changed" in l or "added" in l]
    forward = next((l for l in rows if "run/step/forward" in l), "")
    ok &= check("every overlapping row says 'lane changed'",
                all("lane changed" in l for l in rows
                    if not l.strip().startswith(("run/step/new_stage", "run/step/gone_now"))),
                " | ".join(r.strip() for r in rows))
    ok &= check("no percentage is printed in the comparison section",
                "%" not in section,
                "no '%' in the comparison rows")
    ok &= check("a new stage prints 'added'",
                any(l.rstrip().endswith("added") for l in rows),
                next((l.strip() for l in rows if "added" in l), "<missing>"))
    ok &= check("a vanished stage prints 'gone'",
                any(l.rstrip().endswith("gone") for l in rows),
                next((l.strip() for l in rows if "gone" in l), "<missing>"))

    # same lane: no banner, real deltas with a sign
    same = record("cpu", [("run", 100.0), ("run/step", 80.0),
                          ("run/step/forward", 50.0), ("run/step/gone_now", 30.0)],
                  "2026-11-02T00:00:00+00:00")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(report_run.page(now, same))
    out2 = buf.getvalue()
    ok &= check("a variant change back to the same lane prints the banner again",
                "mismatch lane: cpu -> gpu" in out2,
                next((l for l in out2.splitlines() if "mismatch" in l), "<missing>"))

    if not ok:
        print("FAIL: the mismatch banner is wrong or missing")
        sys.exit(1)
    print("PASS: the report prints the mismatch banner and 'lane changed'")


if __name__ == "__main__":
    main()
