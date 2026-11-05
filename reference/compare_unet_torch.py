"""The UNet's two-level forward against PyTorch, whole planes.

  ../bendtorch/reference/.venv/bin/python reference/compare_unet_torch.py

This repository's `reference/.venv` has no torch; `bendtorch`'s does. The check is run with the
`--dump` flag of `examples/diffusion/unet/main.bend`, which prints the whole `y` (32x32x2), `z`
(32x32x2) and `out` (64x64x2) planes instead of eight cells.

The weights are the example's own deterministic ones: cell i of every 36-cell kernel is
`0.01 * i` with alternating sign (row-major `(co, c*k*k)`), the biases are 0.05, 0.06, 0.07, 0.08
for `conv1`, `conv3`, `conv4`, `conv2`, and every `GroupNorm` has scale 0.5 and bias 0.5 with
`eps = 1.0`. The forward is the two-level U:

  y = silu(gn(conv(x,  w1, s=2)))                     32x32
  l = silu(gn(conv(y,  w3, s=2)))                     16x16
  z = silu(gn(conv(up(l), w4, s=1))) + y              32x32
  o = silu(gn(conv(up(z), w2, s=1))) + x              64x64

It also re-runs the NumPy twin (`test_unet.py`) and checks torch against it, so a failure says
whether the disagreement is in the reference or between Bend and both references.

Exit 0 = torch agrees with Bend within tolerance. Writes
`results/diffusion-unet/compare-torch.json`.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

import test_unet as twin   # the NumPy twin: gen, conv, groupnorm, silu, up2

F = np.float32
H = W = 64
C = 2
# the gate is a combined one, so a large value is not held to an absolute floor:
# `|bend - torch| <= ATOL + RTOL * |torch|` for every cell. The worst cell lands at
# 7.63e-06 for `out`, which is exactly one float32 ulp at |65| -- the skip adds the
# input ramp, whose cells reach +-82 -- and 1.19e-06 for `y`/`z` near 1.3.
ATOL, RTOL = 2e-6, 1e-6


def gen(n):
    v = F(0.01) * np.arange(0, n, dtype=F)
    sign = np.where(np.arange(n) % 2 == 0, F(1.0), F(-1.0)).astype(F)
    return F(v * sign)


def torch_forward():
    t = torch.from_numpy
    x = t(gen(H * W * C).reshape(H, W, C)).permute(2, 0, 1).unsqueeze(0)     # (1, 2, 64, 64)
    w = t(gen(C * 9 * C).reshape(C, C * 9)).reshape(C, C, 3, 3)
    b1, b3, b4, b2 = (torch.full((C,), v, dtype=torch.float32) for v in (0.05, 0.06, 0.07, 0.08))
    g = torch.full((C,), 0.5, dtype=torch.float32)
    beta = torch.full((C,), 0.5, dtype=torch.float32)

    def conv(h, b, stride):
        return Fn.conv2d(h, w, b, stride=stride, padding=1)

    def gn(h):
        return Fn.group_norm(h, 1, g, beta, eps=1.0)

    def up(h):
        return Fn.interpolate(h, scale_factor=2, mode="nearest")

    with torch.no_grad():
        y = Fn.silu(gn(conv(x, b1, 2)))
        low = Fn.silu(gn(conv(y, b3, 2)))
        z = Fn.silu(gn(conv(up(low), b4, 1))) + y
        out = Fn.silu(gn(conv(up(z), b2, 1))) + x
    return {k: v.squeeze(0).permute(1, 2, 0).numpy().reshape(-1).astype(F) for k, v in
            {"y": y, "z": z, "out": out}.items()}


def numpy_forward():
    """the whole planes, from the twin's own functions (its `twin()` truncates to eight)."""
    x = gen(H * W * C).reshape(H, W, C)
    w = gen(C * 9 * C).reshape(C, C * 9)
    g = np.full(C, F(0.5), dtype=F)
    beta = np.full(C, F(0.5), dtype=F)
    y = twin.silu(twin.groupnorm(twin.conv(x, w, np.full(C, F(0.05), dtype=F), s=2), g, beta))
    low = twin.silu(twin.groupnorm(twin.conv(y, w, np.full(C, F(0.06), dtype=F), s=2), g, beta))
    z = F(twin.silu(twin.groupnorm(twin.conv(twin.up2(low), w, np.full(C, F(0.07), dtype=F)), g, beta)) + y)
    out = F(twin.silu(twin.groupnorm(twin.conv(twin.up2(z), w, np.full(C, F(0.08), dtype=F)), g, beta)) + x)
    return {"y": y.reshape(-1), "z": z.reshape(-1), "out": out.reshape(-1)}


def bend_dump():
    exe = twin.build()
    out = subprocess.run([exe, "--dump"], capture_output=True, text=True, env=twin.ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the unet probe did not run:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    got = {}
    for key in ("y_all", "z_all", "out_all"):
        m = re.search(rf"^{key}: (.*)$", out.stdout, re.M)
        if not m:
            raise SystemExit(f"no {key}: line in the --dump output")
        got[key[:-4]] = np.array([float(v) for v in m.group(1).split()], dtype=F)
    return got


def main():
    bend = bend_dump()
    tor = torch_forward()
    num = numpy_forward()
    rows, ok = [], True
    for key, size in (("y", 2048), ("z", 2048), ("out", 8192)):
        if bend[key].size != size:
            raise SystemExit(f"bend {key}: {bend[key].size} cells, expected {size}")
        err = np.abs(bend[key].astype(np.float64) - tor[key].astype(np.float64))
        allow = ATOL + RTOL * np.abs(tor[key].astype(np.float64))
        d_bt = float(np.max(err))
        worst = float(np.max(err / allow))
        d_bn = float(np.max(np.abs(bend[key].astype(np.float64) - num[key].astype(np.float64))))
        d_tn = float(np.max(np.abs(tor[key].astype(np.float64) - num[key].astype(np.float64))))
        good = worst <= 1.0
        ok &= good
        rows.append({"plane": key, "cells": int(bend[key].size), "bend_vs_torch": d_bt,
                     "worst_ratio": worst, "bend_vs_numpy": d_bn, "torch_vs_numpy": d_tn,
                     "ok": good})
        print(f"{'ok  ' if good else 'FAIL'} {key:3s} {size:5d} cells  "
              f"bend-torch {d_bt:.2e}  bend-numpy {d_bn:.2e}  torch-numpy {d_tn:.2e}  "
              f"worst {worst:.2f}x the allowance")

    os.makedirs(os.path.join(ROOT, "results/diffusion-unet"), exist_ok=True)
    path = os.path.join(ROOT, "results/diffusion-unet/compare-torch.json")
    json.dump({"task": "unet-two-level", "atol": ATOL, "rtol": RTOL, "planes": rows,
               "torch": torch.__version__}, open(path, "w"), indent=2)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    print("PASS: torch agrees with Bend on the whole two-level forward" if ok
          else "FAIL: torch against Bend")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
