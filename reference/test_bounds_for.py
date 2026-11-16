"""Gate: bounds_for refuses a moved floor (row C3).

  reference/.venv/bin/python reference/test_bounds_for.py

bounds_for's contract (reference/bounds.py):

  no sidecar        -> the record's own bounds copy is returned
  sidecar overlap   -> per path, the sidecar wins over the record's copy
  sidecar extends   -> paths the sidecar does not declare still fall back
                       to the record's own copy
  stale sidecar     -> a path the sidecar declares that is not a stage of
                       this run is a moved floor: it is REFUSED (dropped,
                       named in a printed line), not scored and not an error

The sidecar path is computed from ROOT; the test monkey-patches
`bounds.sidecar` into a tmp directory, so results/ is never touched.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import bounds  # noqa: E402

REC = {
    "task": {"name": "synthetic"},
    "stages": [{"path": "run"}, {"path": "run/step"}, {"path": "run/step/forward"}],
    "bounds": {
        "run/step/forward": {"read": 1000, "write": 500, "flops": 2000},
        "run/step": {"read": 100, "write": 50, "flops": 200},
    },
}


def run(record, sidecar_path, sidecar_content):
    bounds.sidecar = lambda task: sidecar_path
    os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)
    with open(sidecar_path, "w") as f:
        f.write(sidecar_content)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        got = bounds.bounds_for(record)
    return got, out.getvalue()


def check(name, cond, detail):
    print(f"{'ok  ' if cond else 'FAIL'} {name}: {detail}")
    return bool(cond)


def main():
    ok = True
    tmp = tempfile.mkdtemp()
    sidecar_path = os.path.join(tmp, "bounds-synthetic.json")

    # no sidecar: the record's own copy
    if os.path.exists(sidecar_path):
        os.remove(sidecar_path)
    bounds.sidecar = lambda task: sidecar_path
    got = bounds.bounds_for(REC)
    ok &= check("no sidecar returns the record's copy",
                got == REC["bounds"], f"{sorted(got)}")

    # overlap: the sidecar wins for its paths
    got, warn = run(REC, sidecar_path,
                    json.dumps({"run/step/forward": {"read": 1, "write": 2, "flops": 3}}))
    ok &= check("sidecar wins per path",
                got["run/step/forward"] == {"read": 1, "write": 2, "flops": 3},
                f"{got.get('run/step/forward')}")

    # extends: paths the sidecar does not declare still fall back
    ok &= check("undeclared paths fall back to the record",
                got.get("run/step") == REC["bounds"]["run/step"],
                f"{got.get('run/step')}")

    # a moved floor: the stale path is refused, named, and dropped;
    # the rest of the declaration still lands
    got, warn = run(REC, sidecar_path,
                    json.dumps({"run/step/forward": {"read": 1, "write": 2, "flops": 3},
                                "run/step/old_name": {"read": 9, "write": 9, "flops": 9}}))
    ok &= check("a moved floor is refused (dropped)",
                "run/step/old_name" not in got and got["run/step/forward"]["flops"] == 3,
                f"{sorted(got)}")
    ok &= check("the refusal names the sidecar and the moved path",
                "bounds-synthetic" in warn and "run/step/old_name" in warn,
                warn.strip())

    # an all-stale sidecar: refuses everything, keeps the record's copy untouched
    got, warn = run(REC, sidecar_path,
                    json.dumps({"totally/moved": {"read": 1, "write": 1, "flops": 1}}))
    ok &= check("an all-moved sidecar is refused wholesale",
                got == REC["bounds"] and "totally/moved" in warn,
                f"{sorted(got)} / {warn.strip()}")

    if not ok:
        print("FAIL: the moved floor is not refused")
        sys.exit(1)
    print("PASS: bounds_for refuses a moved floor")


if __name__ == "__main__":
    main()
