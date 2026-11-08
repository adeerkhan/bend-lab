#!/usr/bin/env python3
"""Train and sample the lucidrains rectified-flow reference on the flowers split.

  # train (the Oxford recipe: 64x64, dim 64, batch 16, 70k steps, EMA)
  reference/.venv-rf/bin/python reference/rectified_flow_compare.py \
      --steps 70000 --batch-size 16

  # draw 64 images from the final EMA checkpoint and write an eval log
  reference/.venv-rf/bin/python reference/rectified_flow_compare.py \
      --sample --checkpoint checkpoint.70000.pt --draws 64 \
      --out .scratch/logs/rf_draws.log

The model is the reference's own `Unet(dim=64, channels=1)` and
`RectifiedFlow(predict='clean', auto_normalize=True)`, the trainer is the
reference's `Trainer` (Adam 3e-4, EMA, grad clip 0.5, batch 16). The data is the
pinned 7,067/102 flowers split at 64x64, one grayscale channel, so it is the
same split the Bend models are scored on.

Two deliberate choices, both recorded in the comparison:

- **one channel, not three.** The flowers export is grayscale; the reference's
  `channels=1` sees exactly the data the Bend model sees. The U-Net shape is
  otherwise the reference's.
- **Euler 32 for the scored draws.** The reference's default sampler is
  `odeint` midpoint with `steps=16`; the Bend models sample Euler 32. The
  trainer's monitoring samples use the reference default, and the final 64-draw
  log uses Euler 32 (`--euler 32`) so the two models are scored through the same
  sampler. The flag can be changed to compare samplers separately.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "reference", "lucidrains", "rectified-flow-pytorch")
sys.path.insert(0, VENDOR)

from rectified_flow_pytorch import RectifiedFlow, Trainer, Unet  # noqa: E402

CACHE = os.path.expanduser("~/.cache/bend-lab/flowers")
TRAIN = os.path.join(CACHE, "flowers64-trainfull-gray.u8")
HOLD = os.path.join(CACHE, "flowers64-hold-gray.u8")
EDGE = 64


class Flowers(Dataset):
    """The pinned grayscale split, one channel in [0, 1]."""

    def __init__(self, path):
        self.x = np.fromfile(path, dtype=np.uint8)
        if self.x.size % (EDGE * EDGE):
            raise SystemExit(f"{path} is not a whole number of {EDGE}x{EDGE} images")
        self.x = self.x.reshape(-1, EDGE, EDGE)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]).float().div(255.0).unsqueeze(0)


def build(args):
    ds = Flowers(TRAIN)
    model = Unet(dim=64, channels=1, mean_variance_net=False)
    flow = RectifiedFlow(
        model,
        predict="clean",
        data_shape=(1, EDGE, EDGE),
        auto_normalize=True,
        odeint_kwargs=dict(method="euler") if args.euler else dict(),
    )
    trainer = Trainer(
        flow,
        dataset=ds,
        batch_size=args.batch_size,
        grad_accum_every=args.grad_accum,
        num_train_steps=args.steps,
        learning_rate=3e-4,
        sample_temperature=1.5,
        freeze_normalizer_after_steps=1000,
        sample_kwargs=dict(steps=args.euler) if args.euler else dict(),
        save_results_every=args.save_every,
        checkpoint_every=args.checkpoint_every,
        checkpoints_folder=args.checkpoints,
        results_folder=args.results,
        clear_results_folder=not args.sample and not args.resume,
        num_samples=16,
    )
    return ds, flow, trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=70_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step; batch 8 x 2 is the reference's effective 16")
    ap.add_argument("--euler", type=int, default=32,
                    help="Euler steps for the scored draws; 0 uses the reference default (midpoint 16)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--checkpoint-every", type=int, default=35_000)
    ap.add_argument("--results", default="results/lucidrains-rectified-flow")
    ap.add_argument("--checkpoints", default=".scratch/rf-checkpoints")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--checkpoint", default="checkpoint.70000.pt")
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--out", default=".scratch/logs/rf_draws.log")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)

    ds, flow, trainer = build(args)
    params = sum(p.numel() for p in flow.parameters())
    print(f"params: {params}", flush=True)
    print(f"train images: {len(ds)}  batch: {args.batch_size}  steps: {args.steps}", flush=True)

    if args.sample:
        trainer.load(args.checkpoint)
        print(f"loaded {args.checkpoint}; drawing {args.draws} images", flush=True)
        t0 = time.time()
        chunks = []
        with torch.no_grad():
            for i in range(0, args.draws, 16):
                chunks.append(trainer.sample(
                    os.path.join(args.results, f"results.final.{i // 16}.png")))
        sampled = torch.cat(chunks, 0)[:args.draws].clamp(0.0, 1.0).cpu().numpy()[:, 0]
        lines = [f"params: {params}", f"euler: {args.euler or 16}", f"steps: {args.steps}"]
        for img in sampled:
            lines.append("sample: " + " ".join(f"{v:.6f}" for v in img.reshape(-1)))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write("\n".join(lines) + "\n")
        print(f"wrote {args.out} ({len(sampled)} draws, {time.time() - t0:.1f} s)", flush=True)
        return

    if args.resume:
        trainer.load("checkpoint.latest.pt")

    t0 = time.time()
    trainer()
    print(f"training wall: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
