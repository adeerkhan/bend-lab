#!/usr/bin/env python3
"""Forward/backward parity for the local four-token depthwise convolution.

  reference/.venv/bin/python reference/test_short_conv.py

The Bend probe is `examples/looped_transformer/short_conv_probe.bend`: b=2,
t=5, d=3. It reads x, the four kernel rows and dy from
/tmp/opencode/short_conv/ and prints y, dx and dw0..dw3. The reference is
PyTorch `conv1d(groups=d, padding=3, bias=False)` with autograd.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle import close, parse  # noqa: E402

ROOT = os.path.dirname(HERE)
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
GUARD = os.path.join(ROOT, "scripts", "bend-guard.sh")
SRC = os.path.join(ROOT, "examples/looped_transformer/short_conv_probe.bend")
BIN = os.path.join(ROOT, ".scratch", "short_conv_probe")
OUT = "/tmp/opencode/short_conv"
B, T, D = 2, 5, 3


def env() -> dict:
    e = dict(os.environ, BEND_NO_TELEMETRY="1",
             BEND_LIB=os.path.join(ROOT, ".bend", "env"),
             BEND_MEM=os.environ.get("BEND_MEM", "1500M"))
    llvm = os.path.expanduser("~/.local/lib/llvm22")
    if os.path.isdir(llvm):
        e["LD_LIBRARY_PATH"] = llvm + (":" + e["LD_LIBRARY_PATH"] if e.get("LD_LIBRARY_PATH") else "")
    return e


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(BIN), exist_ok=True)
    rng = np.random.RandomState(17)
    x = (rng.randn(B * T, D) * 0.5).astype("<f4")
    w = (rng.randn(D, 4) * 0.5).astype("<f4")
    dy = (rng.randn(B * T, D) * 0.5).astype("<f4")
    x.tofile(os.path.join(OUT, "x.bin"))
    dy.tofile(os.path.join(OUT, "dy.bin"))
    for j in range(4):
        w[:, j].astype("<f4").tofile(os.path.join(OUT, f"w{j}.bin"))

    build = subprocess.run([GUARD, BEND, "-o", BIN, SRC], env=env(),
                           capture_output=True, text=True)
    if build.returncode != 0 or not os.path.exists(BIN):
        raise SystemExit(f"short_conv_probe did not build:\n{build.stdout[-1200:]}{build.stderr[-1200:]}")
    run = subprocess.run([BIN], env=env(), capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(f"short_conv_probe did not run:\n{run.stdout}{run.stderr}")
    got = parse(run.stdout)

    xt = torch.tensor(x.reshape(B, T, D)).permute(0, 2, 1)
    wt = torch.tensor(w.reshape(D, 1, 4)).requires_grad_(True)
    xt_pad = F.pad(xt, (3, 0)).requires_grad_(True)
    y = F.conv1d(xt_pad, wt, groups=D)
    loss = (y.permute(0, 2, 1).reshape(B * T, D) * torch.tensor(dy)).sum()
    loss.backward()

    failures: list[str] = []
    worst = 0.0
    worst = max(worst, close("y", got["y"], y.permute(0, 2, 1).reshape(B * T, D), failures))
    worst = max(worst, close("dx", got["dx"],
                             xt_pad.grad[..., 3:].permute(0, 2, 1).reshape(B * T, D), failures))
    for j in range(4):
        worst = max(worst, close(f"dw{j}", got[f"dw{j}"], wt.grad[:, 0, j], failures))

    print(f"short-conv: {len(got)} tensors, worst error {worst:.3f} of the tolerance")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        raise SystemExit("SHORT CONV FAIL")
    print("SHORT CONV PASS")


if __name__ == "__main__":
    main()
