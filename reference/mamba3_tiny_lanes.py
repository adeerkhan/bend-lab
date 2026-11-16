"""The tiny-LM training curve through two lanes (row A2).

  reference/.venv-rf/bin/python reference/mamba3_tiny_lanes.py --lane f32
  reference/.venv-rf/bin/python reference/mamba3_tiny_lanes.py --lane bf16
  reference/.venv-rf/bin/python reference/mamba3_tiny_lanes.py --lane both --compare

Same data, same seeds, same schedule, 20 steps, the first 20 train
examples of the byte corpus, batch 1, targets the next byte within the
example (127 targets per example, the last position dropped).

  f32   the twin path on CUDA: W.twin_forward (the explicit t-by-t loop,
        F32 everywhere).  This is the same lane as the CPU F32 record in
        `.scratch/mamba3_tiny/twin.json`; the device differs (CUDA vs
        CPU), which is a banner, never a delta.
  bf16  the public Mamba3 module on CUDA (internally bf16: the wrapper
        casts Q/K/V/trap/angles before the kernel).  Everything else
        (embedding, Wh, nw, cross-entropy) stays F32.

Both lanes load the bit-identical inits from `.scratch/mamba3_tiny/w_init/`
(written by `reference/mamba3_tiny_twin.py --mode dump`, seeds 101/102/SEEDX
20261115).  Writes `.scratch/mamba3_tiny/lane_{f32,bf16}.json`.

A lane measurement, not a verdict: the bf16 numbers are what the public
module costs in bf16; the F32 twin is the oracle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
import mamba3_twin as W  # noqa: E402
import mamba3_tiny_twin as TT  # noqa: E402

OUT = os.path.join(ROOT, ".scratch/mamba3_tiny")
STEPS = 20


def train_f32(steps):
    manifest = TT.load_manifest()
    m1, m2, E, Wh, nw = TT.build(manifest, steps)
    m1, m2 = m1.cuda(), m2.cuda()
    E = E.detach().cuda().requires_grad_(True)
    Wh = Wh.detach().cuda().requires_grad_(True)
    nw = nw.detach().cuda().requires_grad_(True)
    tokens = TT.read_tokens("train")
    return _train(m1, m2, E, Wh, nw, tokens, steps, twin=True)


def train_bf16(steps):
    manifest = TT.load_manifest()
    m1, m2, E, Wh, nw = TT.build(manifest, steps)
    m1, m2 = m1.cuda(), m2.cuda()
    E = E.detach().cuda().requires_grad_(True)
    Wh = Wh.detach().cuda().requires_grad_(True)
    nw = nw.detach().cuda().requires_grad_(True)
    tokens = TT.read_tokens("train")
    return _train(m1, m2, E, Wh, nw, tokens, steps, twin=False)


def _forward_bf16(m1, m2, E, Wh, nw, tokens):
    u = E[torch.tensor(tokens, dtype=torch.long, device=E.device)].unsqueeze(0)
    h1 = u + m1(u)
    h2 = h1 + m2(h1)
    hn = W.rms(h2.float(), nw, eps=1e-5)
    return hn @ Wh.T


def _train(m1, m2, E, Wh, nw, tokens, steps, twin):
    def lm_forward(toks):
        if twin:
            out = TT.lm_forward(m1, m2, E, Wh, nw, toks)
            return out["logits"]
        return _forward_bf16(m1, m2, E, Wh, nw, toks)

    params = (list(m1.parameters()) + list(m2.parameters()) + [E, Wh, nw])
    opt = torch.optim.AdamW(params, lr=TT.PEAK, weight_decay=TT.WD,
                            betas=(TT.BETA1, TT.BETA2), eps=TT.EPS)

    def set_lr(s):
        for g in opt.param_groups:
            g["lr"] = TT.lr_of(s, steps)

    losses, norms = [], []
    hits, total = 0, 0
    for s in range(steps):
        set_lr(s)
        tgt = torch.tensor(tokens[s, 1:], dtype=torch.long, device=E.device)
        logits = lm_forward(tokens[s])
        loss = F.cross_entropy(logits[0, :TT.TARGETS], tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, TT.CLIP)
        opt.step()
        with torch.no_grad():
            hits += int((logits[0, :TT.TARGETS].argmax(-1) == tgt).sum())
            total += TT.TARGETS
        losses.append(float(loss.detach()))
        norms.append(float(gn))
        print(f"step {s} loss {losses[-1]:.7f} gn {float(gn):.4f}", flush=True)

    rec = {
        "lane": "f32-twin-cuda" if twin else "bf16-module-cuda",
        "seed": {"l1": TT.SEED1, "l2": TT.SEED2, "extra": TT.SEEDX},
        "device": "cuda", "steps": steps,
        "losses": losses, "grad_norms": norms,
        "train_hits": int(hits), "train_total": int(total),
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["f32", "bf16", "both"], default="both")
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    lanes = ["f32", "bf16"] if args.lane == "both" else [args.lane]
    for lane in lanes:
        rec = train_f32(args.steps) if lane == "f32" else train_bf16(args.steps)
        path = os.path.join(OUT, f"lane_{lane}.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
