#!/usr/bin/env python3
"""Aggregate one `train_eval_all` log into per-hop accuracy.

The Bend binary prints one `accuracy` line per eight-puzzle evaluation step.
`eval-all.bin` is hop-major, with the same number of steps per hop, so the first
16 lines are hop 1, the next 16 are hop 2, and so on.

  python3 reference/summarize_eval.py results/<task>/runs/<run>.log
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--steps-per-hop", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hops", type=int, default=12)
    ap.add_argument("--seen", type=int, default=8)
    args = ap.parse_args()

    lines = args.log.read_text().splitlines()
    values = [float(line.split()[1]) for line in lines if line.startswith("accuracy ")]
    expected = args.steps_per_hop * args.hops
    if len(values) != expected:
        raise SystemExit(f"{args.log}: {len(values)} accuracy lines, expected {expected}")

    losses = [line for line in lines if line.startswith(("loss ", "final_loss "))]
    print(f"final loss: {losses[-1].split()[1] if losses else 'unavailable'}")
    means = []
    for hop in range(args.hops):
        start = hop * args.steps_per_hop
        block = values[start:start + args.steps_per_hop]
        mean = sum(block) / len(block)
        hits = round(sum(block) * args.batch)
        means.append(mean)
        print(f"hop {hop + 1:2d}: {mean:6.1%} ({hits}/{len(block) * args.batch})")

    seen = sum(means[:args.seen]) / len(means[:args.seen])
    unseen_means = means[args.seen:]
    if unseen_means:
        unseen = sum(unseen_means) / len(unseen_means)
        print(f"seen 1-{args.seen}: {seen:.1%}; unseen {args.seen + 1}-{args.hops}: {unseen:.1%}")
    else:
        print(f"seen 1-{args.seen}: {seen:.1%}; unseen: no unseen hops")


if __name__ == "__main__":
    main()
