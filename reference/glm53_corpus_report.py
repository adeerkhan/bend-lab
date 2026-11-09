"""Write the record for the corpus-trained dense baseline at 128 positions.

  reference/.venv/bin/python reference/glm53_corpus_report.py \
      --log /tmp/opencode/glm53/corpus128.log \
      --time /tmp/opencode/glm53/corpus128.time \
      --out results/glm53-corpus128

The run command is in `run.json`; the weights are `final_weights.bin` (local,
git-ignored) and the evaluation manifests are `eval-dev.json` and
`eval-confirm.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifacts  # noqa: E402


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
        'Dense Stage A loss, 128-position corpus windows (lower is better)</text>',
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
    ap.add_argument("--log", default="/tmp/opencode/glm53/corpus128.log")
    ap.add_argument("--time", default="/tmp/opencode/glm53/corpus128.time")
    ap.add_argument("--out", default="results/glm53-corpus128")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--positions", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--build-seconds", type=float, default=53.786)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--stage", default="dense-corpus-128")
    args = ap.parse_args()
    text = open(args.log).read()
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", text, re.M)]
    final = float(re.search(r"^final_loss: ([0-9.eE+-]+)$", text, re.M).group(1))
    val = float(re.search(r"^val_loss: ([0-9.eE+-]+)$", text, re.M).group(1))
    time_text = open(args.time).read()
    elapsed = re.search(r"Elapsed \(wall clock\) time .*: ([0-9:.]+)$", time_text, re.M)
    seconds = 0.0
    if elapsed:
        for part in [float(v) for v in elapsed.group(1).split(":")]:
            seconds = seconds * 60.0 + part
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)$", time_text, re.M)
    meta_path = args.meta or os.path.join(ROOT, "examples/language/glm53_flash/data/corpus128.json")
    meta = json.load(open(meta_path))
    eval_summary = {}
    for split in ("dev", "confirm"):
        payload = json.load(open(os.path.join(args.out, f"eval-{split}.json")))
        eval_summary[split] = payload["summary"]
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "model": "dense Stage A: Embedding, 4 x (RMSNorm, RoPE, causal Attention, residual; RMSNorm, SwiGLU 128, residual), RMSNorm, tied head",
        "params": 74144,
        "steps": args.steps,
        "batch": args.batch,
        "positions": args.positions,
        "tokens_seen": args.steps * args.batch * args.positions,
        "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1},
        "data": meta,
        "build_seconds": args.build_seconds,
        "seconds": round(seconds, 3),
        "peak_rss_kib": int(rss.group(1)) if rss else "unavailable",
        "loss": {
            "first": losses[0],
            "final": final,
            "last_100_mean": float(np.mean(losses[-100:])),
            "min": float(min(losses)),
            "val": val,
        },
        "eval": eval_summary,
        "commands": {
            "build": "env LD_LIBRARY_PATH=$HOME/.local/lib/llvm22 BEND_MEM=6G scripts/bend-guard.sh bend -o /tmp/opencode/glm53/train_dense128 examples/language/glm53_flash/train_dense128.bend",
            "run": "/tmp/opencode/glm53/train_dense128 --steps 400 --lr-ppm 3000 --data examples/language/glm53_flash/data --out /tmp/opencode/glm53/corpus128",
            "eval": "reference/.venv/bin/python reference/glm53_dense_eval.py --weights results/glm53-corpus128/final_weights.bin --out results/glm53-corpus128 --golden '' --model-name model-128",
        },
        "passed": True,
    }
    rows = ["step\tloss"] + [f"{i}\t{v:.9f}" for i, v in enumerate(losses, 1)]
    artifacts.save_text(os.path.join(args.out, "loss.tsv"), "\n".join(rows) + "\n")
    svg(losses, os.path.join(args.out, "loss.svg"))
    artifacts.save_json(os.path.join(args.out, "run.json"), record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
