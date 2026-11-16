"""Synthetic angle-wrap case (row B3).

  reference/.venv-rf/bin/python reference/mamba3_wrap.py --dump OUT.npz

Builds the SINGLE-layer golden block from reference/mamba3_data/*.bin and
the TWO-layer weight-tied LM from the same files (train_*), with
`dt_bias` filled to 10.0 in both.  The raw angle cumsum then crosses 2pi
about seven times per head at seqlen 8, so the wrap code paths actually
run.  Writes OUT.npz with:

  angle   the twin's wrapped angle tags, (1, 8, 4, 4)
  y       the twin's y for the u8 input
  tokens  generate_twin's greedy tokens for the two-layer LM, 16 tokens
  raw     the unwrapped cumsum, for the wrap-variant cross-check

Also prints the three-way wrap-variant agreement (per-step fmod wrap vs
cumsum floor-mod vs torch remainder), each pair at most 1e-5.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
import mamba3_twin as W  # noqa: E402

DATA = os.path.join(ROOT, "reference/mamba3_data")
DT_FILL = 10.0


def load(name, shape, like=None):
    a = np.fromfile(os.path.join(DATA, name), dtype="<f4")
    return torch.from_numpy(a.reshape(shape))


def build_block():
    torch.manual_seed(71)
    tz = torch.get_num_threads()
    torch.set_num_threads(1)
    m = W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32,
                 ngroups=1, is_mimo=False).cuda().float()
    sd = {
        "in_proj.weight": load("in_proj.bin", (304, 64)),
        "dt_bias": torch.full((4,), DT_FILL),
        "B_bias": load("B_bias.bin", (4, 16)).unsqueeze(1),
        "C_bias": load("C_bias.bin", (4, 16)).unsqueeze(1),
        "B_norm.weight": load("B_norm_w.bin", (16,)),
        "C_norm.weight": load("C_norm_w.bin", (16,)),
        "D": load("D.bin", (4,)),
        "out_proj.weight": load("out_proj.bin", (64, 128)),
    }
    m.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=True)
    torch.set_num_threads(tz)
    return m


def build_lm():
    torch.set_num_threads(1)
    m = W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32,
                 ngroups=1, is_mimo=False).cuda().float()
    sd = {
        "in_proj.weight": load("train_w_in.bin", (304, 64)),
        "dt_bias": torch.full((4,), DT_FILL),
        "B_bias": load("train_B_bias.bin", (4, 16)).unsqueeze(1),
        "C_bias": load("train_C_bias.bin", (4, 16)).unsqueeze(1),
        "B_norm.weight": load("train_B_norm_w.bin", (16,)),
        "C_norm.weight": load("train_C_norm_w.bin", (16,)),
        "D": load("train_D.bin", (4,)),
        "out_proj.weight": load("train_w_out.bin", (64, 128)),
    }
    m.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=True)
    E = load("train_E.bin", (8, 64)).cuda()
    Wh = load("train_WhT.bin", (8, 64)).cuda()
    return m, E, Wh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()

    m = build_block()
    u = torch.from_numpy(
        np.fromfile(os.path.join(DATA, "u.bin"), dtype="<f4").reshape(1, 8, 64)).cuda()
    f = W.twin_forward(m, u)
    angle = f["angle"].detach().cpu().numpy()
    y = f["y"].detach().cpu().numpy()

    vals = (torch.tanh(f["ang_raw"].float()) * math.pi).unsqueeze(-2) * f["DT"].unsqueeze(-1)
    raw_incr = vals  # per-step increments (pre-cumsum)
    raw = raw_incr.cumsum(dim=1)

    # per-step wrap (Bend's wrap()) vs wrap-after-cumsum (the twin's %)
    TWO_PI = 2 * math.pi
    incr = raw_incr
    per = torch.zeros_like(raw)
    for h in range(4):
        acc = torch.zeros(1, incr.shape[3], device=incr.device)
        for t in range(incr.shape[1]):
            acc = acc + incr[0, t, h]
            mm = torch.fmod(acc, TWO_PI)
            mm = torch.where(mm < 0, mm + TWO_PI, mm)
            per[0, t, h] = mm
    d2 = (per - raw % TWO_PI).abs().max().item()

    m2, E, Wh = build_lm()
    toks = W.generate_twin(m2, E, Wh, steps=12)[0]

    crossings = [int(raw[0, -1, h].abs().max().item() / TWO_PI) for h in range(4)]
    print(f"dt_bias fill {DT_FILL}; DT mean {f['DT'].mean().item():.3f}")
    print(f"wrap variants agree: per-step fmod+fix vs cumsum-mod {d2:.3e}")
    print(f"2pi crossings per head: {crossings}")
    np.savez(args.dump, angle=angle.astype(np.float32), y=y.astype(np.float32),
             raw=raw.detach().cpu().numpy().astype(np.float32),
             tokens=np.asarray(toks, dtype=np.float32))
    print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
