#!/usr/bin/env python3
"""Register the trained UNet as a run record, with a manifest.

  reference/.venv/bin/python reference/unet_record.py

This is a migration, not a fresh run: the model was trained on 2026-11-08
(3,600 steps, documented in examples/diffusion/unet/TRAINED.md) and its 64
draws are committed in results/diffusion-unet/trained/run.json. The gate
re-scores those draws on the pinned flowers split; the record's numbers come
from that gate, and `run.kind` says migration so nobody reads the timings as
a new measurement.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from test_unet_flowers_metric import HOLD, RUN, TRAIN, metrics, passes  # noqa: E402
from unet_flowers_eval import FLOWERS, read_gray  # noqa: E402

GATE = "reference/.venv/bin/python reference/test_unet_flowers_metric.py"
OUT = os.path.join(ROOT, "results/latest/unet-flowers.json")
RECENT = os.path.join(ROOT, "results/latest/mamba3-tiny-lm.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else "unavailable"


def report(value, reference=None, delta=None):
    return {"bend": value, "reference": reference, "delta": delta,
            "tolerance": None, "mode": "report", "pass": True,
            "bend_sd": None, "reference_sd": None, "delta_sd": None}


def main():
    gate = subprocess.run(GATE.split(), cwd=ROOT, capture_output=True, text=True)
    if gate.returncode != 0:
        sys.stderr.write(gate.stdout + gate.stderr)
        raise SystemExit("the unet gate failed; refusing to register it")

    import numpy as np

    run = json.load(open(RUN))
    draws = np.array(run["draws_u8"], dtype=np.uint8).reshape(64, 64, 64)
    train = read_gray(TRAIN)
    hold = read_gray(HOLD)
    m = metrics(draws, train, hold)
    checks, ok = passes(m)
    if not ok:
        raise SystemExit(f"evaluator disagrees with the gate: {checks}")

    bend_version = subprocess.run([os.path.expanduser("~/.bend/bin/bend"), "version"],
                                  capture_output=True, text=True).stdout.split()[-1]
    recent = json.load(open(RECENT))
    files = {}
    for path in (TRAIN, HOLD):
        name = os.path.relpath(path, FLOWERS)
        files[name] = sha256(path)

    metrics_dict = {
        "params": report(run["params"]),
        "final_loss": report(run["final_loss"]),
        "draws": report(run["draws"]),
        "legibility_held": report(m["held"], m["ceiling"], m["held"] - m["ceiling"]),
        "legibility_ceiling": report(m["ceiling"]),
        "legibility_floor": report(m["floor"]),
        "span": report(m["span"]),
        "spectrum_high": report(m["bands"][3]),
        "spectrum_low": report(m["bands"][0]),
    }

    record = {
        "schema": "mlbend.experiment/3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": {
            "name": "unet-flowers",
            "description": ("the trained 64x64 UNet denoiser on the pinned flowers "
                            "split: 298,439 parameters, 3,600 steps, 64 draws, "
                            "held-out legibility scored against a real held-out "
                            "flower (the ceiling) and noise (the floor)"),
            "program": "examples/diffusion/unet/train.bend",
            "dataset": {"name": "flowers64", "rows": 7066, "features": 4096,
                        "sha256": files["flowers64-trainfull-gray.u8"],
                        "files": files},
            "seed": 0,
        },
        "seeds": [0],
        "trainer": "bend",
        "variant": {"trainer": "bend", "lane": "cpu", "threads": None,
                    "bend": bend_version, "images": run["images"],
                    "channels": run["channels"], "steps": run["steps"],
                    "euler": run["euler"]},
        "git": {"commit": git("rev-parse", "--short", "HEAD"),
                "branch": git("branch", "--show-current"),
                "dirty": bool(git("status", "--porcelain"))},
        "bend": {"version": bend_version},
        "packages": recent.get("packages", []),
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "processor": platform.processor(), "cpu_count": os.cpu_count(),
                    "python": platform.python_version()},
        "run": {"kind": "migration", "source": os.path.relpath(RUN, ROOT),
                "gate": GATE, "gate_stdout": gate.stdout.strip().splitlines(),
                "build_seconds": None, "seconds_min": None, "seconds_all": [],
                "repeat": 0, "fetch_seconds": 0.0, "compute_seconds": None,
                "speed_ratio": None},
        "stages": [],
        "stages_source": {"kind": "migration",
                          "note": "the original run predates the stage runner; "
                                  "no stage split is available"},
        "counts": {}, "peaks": {}, "bounds": {},
        "notes": [
            "migration of a prior run, not a fresh training: the model was "
            "trained 2026-11-08 and the 64 draws are committed in run.json",
            "the numbers come from make unet-metric re-scoring those draws on "
            "the pinned flowers split; the gate's can-fail arms (white noise "
            "and the data mean) must fail",
            "see examples/diffusion/unet/TRAINED.md for the six measured levers "
            "and the layer/twin gates; no checkpoint is committed, so sampling "
            "from the trained model is unavailable",
            "legibility on the training images is 0.633, above the 0.574 "
            "ceiling: the draws memorise; only the held-out number is a result",
            "unavailable timings are null, not zero",
        ],
        "seed_runs": [{"seed": 0, "metrics": metrics_dict}],
        "metrics": metrics_dict,
        "passed": bool(gate.returncode == 0 and ok),
    }
    with open(OUT, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT}; passed={record['passed']}")
    print(f"  held {m['held']:.4f}  ceiling {m['ceiling']:.4f}  "
          f"floor {m['floor']:.4f}  span {m['span']:.4f}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
