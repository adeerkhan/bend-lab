"""The matrix: the declared cells, one record each, and a summary (B5b).

  make matrix                 # every declared cell, against the frozen references
  make matrix LIVE=1          # the same cells, recomputing the references in NumPy

The cells are declared here and nowhere else. A cell is `(task, lane, seed)`:
the trainer is `bend` (the NumPy trainer is the reference arm, so it is the
frozen file, not a cell). A lane is a **build fact**, never a label:

- `gpu` builds with `CUDA_HOME` set and the binary must link `libnvrtc`;
- `cpu` builds without it and the binary must not link `libnvrtc`.

A cpu record whose binary links the CUDA runtime is a bug, and the cell fails
with that sentence rather than a number. A lane a task cannot run says why in
`ABSENT`: the matrix never shrinks silently.

A failure in the summary names `task/lane/seed/metric`.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1",
           PATH=os.path.expanduser("~/.bend/bin") + ":" + os.environ["PATH"])
OUT = os.path.join(ROOT, "results", "matrix")
CUDA = os.path.expanduser("~/.local/cuda-nvrtc")

# The declared cells. Seeds must exist in the task's frozen reference.
CELLS = (
    {"task": "vit-step", "lane": "cpu", "seeds": [0]},
    {"task": "vit-step", "lane": "gpu", "seeds": [0]},
    {"task": "iris-crossval", "lane": "cpu", "seeds": [42, 43, 44]},
    {"task": "attention-parity", "lane": "cpu", "seeds": [0]},
    {"task": "wine-pca", "lane": "cpu", "seeds": [0]},
    {"task": "cifar-train", "lane": "cpu", "seeds": [0]},
    {"task": "vit-first", "lane": "cpu", "seeds": [0]},
    {"task": "transformer-periodic", "lane": "cpu", "seeds": [0]},
    {"task": "diffusion-2d", "lane": "cpu", "seeds": [0]},
    {"task": "iris-mlp", "lane": "cpu", "seeds": [0]},
    {"task": "wine-kmeans", "lane": "cpu", "seeds": [0]},
    {"task": "cancer-knn", "lane": "cpu", "seeds": [0]},
    {"task": "cancer-naive-bayes", "lane": "cpu", "seeds": [0]},
    {"task": "heart-naive-bayes", "lane": "cpu", "seeds": [0]},
    {"task": "cifar-10", "lane": "cpu", "seeds": [0]},
)

# A task with no cell at all, and why, and where its gate is instead. The
# matrix never shrinks silently (appendix D): a task is a cell row or it is
# here with its measured cost and the test that covers it. The cost is
# `build_seconds + compute_seconds` from its last record on this machine.
NO_CELL = (
    ("diffusion-digits", 226,
     "226 s a cell (build 77 + run 148); gated by reference/test_diffusion_digits.py"),
    ("diffusion-vit", 724,
     "724 s a cell (build 100 + run 625, 3000 steps); gated by "
     "reference/test_diffusion_vit.py (first loss, forward, one step, DDIM, eta, legibility)"),
)

# The cells a lane does not have, and why. A program in plain Bend has no
# device path; the GPU lane is BendTorch's.
ABSENT = (
    ("iris-crossval", "gpu", "plain Bend, no device path: the GPU lane is BendTorch's"),
    ("wine-pca", "gpu", "plain Bend, no device path: the GPU lane is BendTorch's"),
    ("attention-parity", "gpu", "plain Bend, no device path: the GPU lane is BendTorch's"),
    ("cifar-train", "gpu", "the device lane's exp differs from the host's; "
                           "no measured tolerance for a 600-step curve yet"),
    ("cifar-10", "gpu", "the device lane's exp differs from the host's; the task has no "
                        "measured device tolerance yet"),
    ("vit-first", "gpu", "the twin is one host window and the device lane's exp is not "
                         "the host's; no measured tolerance for out_sq there yet"),
    ("diffusion-2d", "gpu", "the device lane's exp differs from the host's; no measured "
                            "tolerance for this task's gate yet"),
)


def lane_env(lane):
    """The build environment that makes a lane a fact rather than a label."""
    env = dict(ENV)
    if lane == "gpu":
        env["CUDA_HOME"] = CUDA
        env["LD_LIBRARY_PATH"] = (os.path.join(CUDA, "lib64") + ":"
                                  + os.environ.get("LD_LIBRARY_PATH", "")).rstrip(":")
    else:
        env.pop("CUDA_HOME", None)
        env.pop("LD_LIBRARY_PATH", None)
    return env


def linked_nvrtc(binary):
    """What the built binary links, as the lane's evidence."""
    try:
        out = subprocess.run(["ldd", binary], capture_output=True, text=True, timeout=60)
        text = out.stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [line.strip().split(" => ")[0].strip() for line in text.splitlines() if "nvrtc" in line]


def run_cell(cell, seed, live):
    """One cell: one subprocess, one record."""
    task, lane = cell["task"], cell["lane"]
    out_dir = os.path.join(OUT, "cells")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [PY, os.path.join(ROOT, "reference/experiment.py"),
           "--task", task, "--lane", lane, "--seeds", str(seed),
           "--references", "live" if live else "frozen",
           "--out", out_dir, "--no-latest"]
    started = datetime.now(timezone.utc)
    run = subprocess.run(cmd, capture_output=True, text=True, env=lane_env(lane), cwd=ROOT)
    record, record_path = None, None
    for line in run.stdout.splitlines():
        if line.startswith("report: "):
            record_path = line[len("report: "):].strip()
    if record_path and os.path.isfile(record_path):
        with open(record_path) as f:
            record = json.load(f)
    binary = os.path.join("/tmp", "mlbend-exp-" + task)
    evidence = {
        "cuda_home": lane_env(lane).get("CUDA_HOME"),
        "linked_nvrtc": linked_nvrtc(binary),
        "exit": run.returncode,
        "started_at": started.isoformat(),
    }
    failures = []
    # A nonzero exit with a record means the task ran and failed its gate: the
    # metric rows say which one. Only a missing record is a `run` failure, so a
    # failure names `task/lane/seed/metric` as the docstring promises.
    if record is None:
        detail = (run.stdout + run.stderr).strip().splitlines()
        failures.append({"metric": "run", "reason": detail[-1] if detail else "no record"})
    else:
        wrong = (lane == "gpu" and not evidence["linked_nvrtc"]) or \
                (lane == "cpu" and evidence["linked_nvrtc"])
        if wrong:
            failures.append({"metric": "lane", "reason":
                             f"the {lane} lane is a label: the binary links "
                             f"{', '.join(evidence['linked_nvrtc'] or []) or 'no nvrtc'}"})
        for name, row in (record.get("metrics") or {}).items():
            if not row.get("pass"):
                failures.append({"metric": name, "reason":
                                 f"bend {row.get('bend')} against reference {row.get('reference')}"
                                 f" (delta {row.get('delta')}, tolerance {row.get('tolerance')})"})
    return {"task": task, "lane": lane, "seed": seed, "record": record_path,
            "plan": " ".join(["CUDA_HOME=" + (evidence["cuda_home"] or "")
                              if lane == "gpu" else "env -u CUDA_HOME"] + cmd[1:]),
            "lane_evidence": evidence, "failures": failures}


def summary(cells, absent, no_cell, path_json, path_md, live):
    ran = [c for c in cells if c["record"]]
    failed = [c for c in cells if c["failures"]]
    lines = [
        "# The matrix",
        "",
        f"`make matrix` over the declared cells: {len(cells)} run, {len(ran)} wrote a record,"
        f" {len(failed)} failed. References: {'live (NumPy)' if live else 'frozen'}.",
        "",
        "A lane is a build fact: `gpu` links `libnvrtc`, `cpu` must not.",
        "",
        "| task | lane | seed | linked | record | verdict |",
        "|---|---|---:|---|---|---|",
    ]
    for cell in cells:
        linked = ", ".join(cell["lane_evidence"]["linked_nvrtc"] or []) or "none"
        verdict = "ok" if not cell["failures"] else "FAIL"
        record = os.path.relpath(cell["record"], ROOT) if cell["record"] else "-"
        lines.append(f"| `{cell['task']}` | {cell['lane']} | {cell['seed']} | {linked}"
                     f" | `{record}` | {verdict} |")
    if failed:
        lines += ["", "## Failures", ""]
        for cell in failed:
            for item in cell["failures"]:
                lines.append(f"- `{cell['task']}/{cell['lane']}/{cell['seed']}/{item['metric']}`"
                             f" — {item['reason']}")
    lines += ["", "## Tasks with no cell, and why", ""]
    for task, seconds, why in no_cell:
        lines.append(f"- `{task}` — {seconds} s: {why}")
    lines += ["", "## Cells a lane does not have", ""]
    for task, lane, why in absent:
        lines.append(f"- `{task}` / {lane}: {why}")
    lines.append("")
    with open(path_md, "w") as f:
        f.write("\n".join(lines))
    with open(path_json, "w") as f:
        json.dump({"schema": "mlbend.matrix/1",
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "references": "live" if live else "frozen",
                   "cells": cells,
                   "absent": [{"task": t, "lane": l, "why": w} for t, l, w in absent],
                   "no_cell": [{"task": t, "seconds": s, "why": w} for t, s, w in no_cell]},
                  f, indent=2)
    return path_md, len(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="recompute the references (needs NumPy)")
    parser.add_argument("--task", help="only this task's cells")
    parser.add_argument("--summary-only", action="store_true",
                        help="redo SUMMARY.md and matrix.json from the last cells, no run")
    args = parser.parse_args()
    if args.summary_only:
        # the page from the artifact: a wording change in ABSENT or NO_CELL does
        # not need a 25-minute re-measure, and the two cannot drift
        data = json.load(open(os.path.join(OUT, "matrix.json")))
        path_md, failed = summary(data["cells"], ABSENT, NO_CELL,
                                  os.path.join(OUT, "matrix.json"),
                                  os.path.join(OUT, "SUMMARY.md"), data.get("references") == "live")
        print(open(path_md).read())
        print(f"summary: {os.path.relpath(path_md, ROOT)}")
        return 1 if failed else 0
    cells = []
    for cell in CELLS:
        if args.task and cell["task"] != args.task:
            continue
        for seed in cell["seeds"]:
            print(f"{cell['task']}/{cell['lane']}/{seed}: building and running", flush=True)
            cells.append(run_cell(cell, seed, args.live))
    if not cells:
        raise SystemExit("no cells: --task names a task the matrix does not declare")
    os.makedirs(OUT, exist_ok=True)
    path_md, failed = summary(cells, ABSENT, NO_CELL, os.path.join(OUT, "matrix.json"),
                              os.path.join(OUT, "SUMMARY.md"), args.live)
    print(open(path_md).read())
    print(f"summary: {os.path.relpath(path_md, ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
