"""Run the 400-step dense pretrain and write its record.

  reference/.venv/bin/python reference/glm53_dense_report.py

It builds `train_a.bend` under `scripts/bend-guard.sh` (BEND_MEM 4G), times
the build, runs the 400 steps with `/usr/bin/time -v`, writes the loss table,
the curve and `run.json`, then runs the 20-step twin comparison for the record.
The evaluation is `reference/glm53_dense_eval.py`, run separately, because it
needs the saved final weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import artifacts  # noqa: E402
import glm53_dense_data as D  # noqa: E402

BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))


def parse_losses(text):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", text, re.M)]
    final = re.search(r"^final_loss: ([0-9.eE+-]+)$", text, re.M)
    if final is None:
        raise SystemExit("the run did not print final_loss")
    losses.append(float(final.group(1)))
    return losses


def build(out):
    src = os.path.join(ROOT, "examples/language/glm53_flash/train_a.bend")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    proc = subprocess.run(["scripts/bend-guard.sh", BEND, "-o", out, src],
                          capture_output=True, text=True, env=dict(ENV, BEND_MEM="4G"), cwd=ROOT)
    seconds = time.perf_counter() - started
    if proc.returncode != 0:
        raise SystemExit(f"build failed:\n{proc.stdout}{proc.stderr}")
    return seconds


def timed_run(exe, args):
    timed = shutil.which("time") or "/usr/bin/time"
    proc = subprocess.run([timed, "-v", exe] + args, capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"run failed:\n{proc.stdout}{proc.stderr}")
    elapsed = re.search(r"Elapsed \(wall clock\) time .*: ([0-9:.]+)$", proc.stderr, re.M)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)$", proc.stderr, re.M)
    seconds = 0.0
    if elapsed:
        parts = [float(v) for v in elapsed.group(1).split(":")]
        for part in parts:
            seconds = seconds * 60.0 + part
    return seconds, int(rss.group(1)) if rss else None, proc.stdout


def svg(losses, path):
    width, height = 720, 360
    left, right, top, bottom = 60, 20, 30, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    top_loss = max(losses) * 1.05

    def point(i, n, value):
        x = left + plot_w * (i / max(1, n - 1))
        y = top + plot_h * (1.0 - value / top_loss)
        return f"{x:.1f},{y:.1f}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="20" font-family="sans-serif" font-size="14">'
        'Dense Stage A pretraining loss, fixed synthetic batch (lower is better)</text>',
    ]
    for tick in range(5):
        value = top_loss * tick / 4
        y = top + plot_h * (1.0 - value / top_loss)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2f}</text>')
    points = " ".join(point(i, len(losses), v) for i, v in enumerate(losses))
    parts.append(f'<polyline fill="none" stroke="#c0392b" stroke-width="2" points="{points}"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="12">step</text>')
    parts.append("</svg>")
    return artifacts.save_text(path, "\n".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("results/glm53-dense"))
    ap.add_argument("--binary", type=Path, default=Path("/tmp/opencode/glm53-record/train_a"))
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    D.prepare(D.DATA)
    build_seconds = None if args.skip_build else build(str(args.binary))
    args.out.mkdir(parents=True, exist_ok=True)
    seconds, rss_kib, text = timed_run(str(args.binary),
        ["--steps", str(args.steps), "--data", D.DATA, "--out", str(args.out)])
    losses = parse_losses(text)
    if len(losses) != args.steps:
        raise SystemExit(f"{len(losses)} losses, not {args.steps}")
    grad = re.search(r"^grad_loss: ([0-9.eE+-]+)$", text, re.M)
    rows = ["step\tloss"] + [f"{i}\t{v:.9f}" for i, v in enumerate(losses, 1)]
    artifacts.save_text(str(args.out / "loss.tsv"), "\n".join(rows) + "\n")
    svg(losses, str(args.out / "loss.svg"))
    # the dev/confirm evaluation, then the 20-step record against the twin
    eval_proc = subprocess.run([sys.executable, os.path.join(HERE, "glm53_dense_eval.py"),
                                "--weights", str(args.out / "final_weights.bin"),
                                "--out", str(args.out)],
                               capture_output=True, text=True, env=ENV, cwd=ROOT)
    if eval_proc.returncode != 0:
        print(eval_proc.stdout[-2000:])
        print(eval_proc.stderr[-2000:])
    code = subprocess.run([sys.executable, os.path.join(HERE, "test_glm53_dense.py")],
                          capture_output=True, text=True, env=ENV, cwd=ROOT)
    gate = {}
    m = re.search(r"final weights: .*whole-tensor relative ([0-9.eE+-]+)", code.stdout)
    if m:
        gate["weights_relative"] = float(m.group(1))
    m = re.search(r"final gradients: .*whole-tensor relative ([0-9.eE+-]+)", code.stdout)
    if m:
        gate["gradients_relative"] = float(m.group(1))
    m = re.search(r"^FAIL step (\d+):", code.stdout, re.M)
    gate["loss_gate_passed"] = "FAIL step" not in code.stdout and code.returncode == 0
    gate["first_diverging_step"] = int(m.group(1)) if m else None
    if code.returncode != 0:
        print(code.stdout[-2000:])
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "dense-pretrain",
        "model": "dense Stage A: Embedding, 4 x (RMSNorm, RoPE, causal Attention, residual; RMSNorm, SwiGLU 128, residual), RMSNorm, tied head",
        "params": 74144,
        "dataset": {
            "name": "reference synthetic Python corpus, first 129 byte-tokenizer tokens",
            "inputs": "examples/language/glm53_flash/data/train_batch.bin",
            "labels": "examples/language/glm53_flash/data/train_labels.bin",
            "rows": 2, "positions": 64, "tokens_per_step": 128,
            "inputs_sha256": hashlib.sha256(open(os.path.join(D.DATA, "train_batch.bin"), "rb").read()).hexdigest(),
            "labels_sha256": hashlib.sha256(open(os.path.join(D.DATA, "train_labels.bin"), "rb").read()).hexdigest(),
            "masked_labels": int((D.train_batch()[1] == D.IGNORE).sum()),
        },
        "optimizer": {"name": "AdamW", "lr": 3e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1},
        "steps": args.steps,
        "tokens_seen": args.steps * 128,
        "build_seconds": round(build_seconds, 3) if build_seconds is not None else "unavailable",
        "seconds": round(seconds, 3),
        "peak_rss_kib": rss_kib if rss_kib is not None else "unavailable",
        "loss": {
            "first": losses[0],
            "final": losses[-1],
            "last_100_mean": float(np.mean(losses[-100:])) if len(losses) >= 100 else "unavailable",
            "final_weights_forward": float(grad.group(1)) if grad else "unavailable",
        },
        "twin_20_step": gate,
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "numpy": np.__version__},
    }
    record["passed"] = bool(gate.get("loss_gate_passed") and gate.get("weights_relative", 1) <= 1e-3
                            and gate.get("gradients_relative", 1) <= 1e-3)
    artifacts.save_json(str(args.out / "run.json"), record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
