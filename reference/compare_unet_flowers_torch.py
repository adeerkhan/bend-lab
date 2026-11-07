"""The trained UNet's whole forward against PyTorch.

  ../bendtorch/reference/.venv/bin/python reference/compare_unet_flowers_torch.py

This repository's venv has no torch; bendtorch's does. The check reads the flat
weights and the whole forward output from `train.bend --probe --steps 0` (the
same 8,192 cells `reference/test_unet_flowers.py` checks in NumPy), rebuilds
the model in torch, and compares every cell.

Two comparisons are printed: torch against Bend, and torch against the NumPy
twin, so a disagreement says whether the model or one reference is wrong.
Writes `results/diffusion-unet/trained/compare-torch.json`.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

import test_unet_flowers as twin

F = np.float32
HW, NIMG, C = 4096, 2, 64
# measured first, then set (see the run in TRAINED.md): torch and Bend agree to
# 1.18e-05 over 8,192 cells after seven convolutions (576-term sums at the
# widest), so the allowance is 2e-5 plus 1e-6 of the value. It was 1e-5 at three
# levels (3.93e-06); the growth is float32 accumulation down the convolution
# chain, the same reason the NumPy twin's gate moved.
ATOL, RTOL = 2e-5, 1e-6


def forward(flat, inp):
    t = torch.from_numpy
    x = t(np.asarray(inp, dtype=F)).reshape(NIMG, 64, 64, 5).permute(0, 3, 1, 2)  # (2, 5, 64, 64)
    w = np.asarray(flat, dtype=F)
    off, parts = 0, []
    for shape in twin.PARTS:
        size = int(np.prod(shape))
        parts.append(t(w[off:off + size].reshape(shape)))
        off += size
    w1, b1, w3, b3, w5, b5, w7, b7, w8, b8, w6, b6, w4, b4, wh, bh, wl, bl = parts

    def conv(h, wt, b, stride, k=3):
        return Fn.conv2d(h, wt.reshape(wt.shape[0], -1, k, k), b, stride=stride, padding=1 if k > 1 else 0)

    def up(h):
        return Fn.interpolate(h, scale_factor=2, mode="nearest")

    with torch.no_grad():
        y = Fn.silu(conv(x, w1, b1, 2))
        low = Fn.silu(conv(y, w3, b3, 2))
        bot = Fn.silu(conv(low, w5, b5, 2))
        b2 = Fn.silu(conv(bot, w7, b7, 2))
        u8 = Fn.silu(conv(up(b2), w8, b8, 1))
        u16 = Fn.silu(conv(up(u8), w6, b6, 1))
        z = Fn.silu(conv(up(u16 + low), w4, b4, 1)) + y
        o = conv(up(z), wh, bh, 1, k=1) + conv(x, wl, bl, 1, k=1)
    return o.permute(0, 2, 3, 1).reshape(-1).numpy().astype(F)


def main():
    w, bend = twin.probe()
    inp = np.empty((2 * HW, 5), dtype=F)
    from test_diffusion_digits import noise_of
    inp[:, 0] = np.asarray(noise_of(7, 2 * HW), dtype=F).reshape(-1)[:2 * HW]
    tau = F(2.0 * np.pi)
    inp[:, 1] = np.sin(F(0.5) * tau)
    inp[:, 2] = np.cos(F(0.5) * tau)
    inp[:, 3] = np.sin(F(1.0) * tau)
    inp[:, 4] = np.cos(F(1.0) * tau)
    tor = forward(w, inp)
    num = twin.forward(w, inp)

    err = np.abs(bend.astype(np.float64) - tor.astype(np.float64))
    allow = ATOL + RTOL * np.abs(tor.astype(np.float64))
    worst = float(np.max(err / allow))
    d_bn = float(np.max(np.abs(bend.astype(np.float64) - num.astype(np.float64))))
    d_tn = float(np.max(np.abs(tor.astype(np.float64) - num.astype(np.float64))))
    ok = worst <= 1.0
    print(f"{'ok  ' if ok else 'FAIL'} forward: {bend.size} cells  bend-torch {err.max():.3e}  "
          f"bend-numpy {d_bn:.3e}  torch-numpy {d_tn:.3e}  worst {worst:.2f}x the allowance")
    out = os.path.join(ROOT, "results/diffusion-unet/trained/compare-torch.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"task": "unet-flowers", "cells": int(bend.size), "atol": ATOL, "rtol": RTOL,
               "bend_vs_torch": float(err.max()), "worst_ratio": worst,
               "bend_vs_numpy": d_bn, "torch_vs_numpy": d_tn, "torch": torch.__version__,
               "ok": bool(ok)}, open(out, "w"), indent=2)
    print(f"wrote {os.path.relpath(out, ROOT)}")
    print("PASS: torch agrees with Bend on the whole forward" if ok else "FAIL: torch against Bend")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
