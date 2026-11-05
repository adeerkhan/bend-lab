"""The UNet's two-level forward against NumPy.

  reference/.venv/bin/python reference/test_unet.py

`examples/diffusion/unet/main.bend` composes a small U with two levels:

  x     (64x64, 2)
  y     conv1 stride 2 -> GroupNorm -> SiLU        (32x32, 2)
  low   conv3 stride 2 -> GroupNorm -> SiLU        (16x16, 2)
  mid   Up2(low) -> conv4 stride 1 -> GroupNorm -> SiLU, + y   (32x32, 2)
  out   Up2(mid) -> conv2 stride 1 -> GroupNorm -> SiLU, + x   (64x64, 2)

This twin rebuilds that forward in float32 and compares `out`, `z` (the 32x32
stage after its skip) and `y` against the first eight cells the program prints.
The second level was added on 2026-11-05 (ISSUE-061); the one-level U this file
used to check is the same arithmetic with the middle stage removed.

The layout the twin has to get right, and the reason this test exists:

  * a Mat row is one pixel and the channels are last (`Conv2d`'s layout), so the
    image is `(4096, 2)` for 64x64 and two channels;
  * `im2col` gathers a patch as `(c, kh, kw)` -- channel outer, kernel inner --
    which is PyTorch's `unfold` order;
  * the conv's stride is `1 + s1`, so `s1 = 1` is stride 2 down and `s1 = 0` is
    stride 1;
  * `GroupNorm(n, hw, ng, cg)` normalises inside a group, here one group of both
    channels, over the group's channels and all its pixels of one image;
  * the skips add the stage's own input, and `Up2` repeats each pixel 2x2.

Measured on 2026-11-05: `out`, `z` and `y` each **5.96e-08**, float32 exact to a
few ulps; the gate is 1e-6, and a cell moved by 0.5 must be rejected.

Exit code 0 = the forward matched. Nothing here writes.
"""
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

PROGRAM = os.path.join(ROOT, "examples/diffusion/unet/main.bend")
BIN = "/tmp/mlbend-unet"
F = np.float32
H = W = 64
C = 2
EPS = F(1.0)   # the example passes 1.0: at 1e-5 the two channels are
               # nearly proportional and the variance is dominated by rounding


def gen(n):
    """the example's `gen`: cell i is 0.01 * i, alternating sign from cell 0."""
    v = F(0.01) * np.arange(0, n, dtype=F)
    sign = np.where(np.arange(n) % 2 == 0, F(1.0), F(-1.0)).astype(F)
    return F(v * sign)


def im2col(x, k=3, p=1, s=1):
    """x: (h, w, c) -> rows are output pixels, columns are (c, kh, kw)."""
    h, w, c = x.shape
    xp = np.pad(x, ((p, p), (p, p), (0, 0)))
    oh = (h + 2 * p - k) // s + 1
    ow = (w + 2 * p - k) // s + 1
    rows = np.empty((oh * ow, c * k * k), dtype=F)
    for oy in range(oh):
        for ox in range(ow):
            win = xp[oy * s:oy * s + k, ox * s:ox * s + k, :]     # (k, k, c)
            rows[oy * ow + ox] = win.transpose(2, 0, 1).reshape(-1)  # (c, kh, kw)
    return rows, oh, ow


def conv(x, w, b, k=3, p=1, s=1):
    """w: (co, c*k*k), b: (co,)."""
    rows, oh, ow = im2col(x, k, p, s)
    y = rows @ w.T + b
    return y.reshape(oh, ow, -1).astype(F)


def groupnorm(x, g, b):
    """one group of both channels, statistics over the group AND all its pixels.

    This is torch's `GroupNorm: (N, C, H, W)` with one group: the mean and the
    variance run over the group's channels and the whole image per sample, not
    per pixel. Getting this wrong was the 1.86e-4 (and the first story about
    collinearity): with per-pixel statistics the normalised values came out
    ~20x too large.
    """
    m = x.mean(axis=(0, 1, 2), keepdims=True)
    v = x.var(axis=(0, 1, 2), keepdims=True)
    r = F(1.0) / np.sqrt(v + EPS)
    return F(F(F(x - m) * r) * g + b)


def silu(x):
    return F(x * (F(1.0) / (F(1.0) + np.exp(-x))))


def up2(x):
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)


def twin():
    x = gen(H * W * C).reshape(H, W, C)
    # every conv is 2 -> 2 channels, 3x3: the same 36-cell pattern, row-major
    # (co, c*k*k), and the biases are the example's 0.05/0.06/0.07/0.08
    w1 = gen(C * 9 * C).reshape(C, C * 9)
    w3 = gen(C * 9 * C).reshape(C, C * 9)
    w4 = gen(C * 9 * C).reshape(C, C * 9)
    w2 = gen(C * 9 * C).reshape(C, C * 9)
    b1 = np.full(C, F(0.05), dtype=F)
    b3 = np.full(C, F(0.06), dtype=F)
    b4 = np.full(C, F(0.07), dtype=F)
    b2 = np.full(C, F(0.08), dtype=F)
    g = np.full(C, F(0.5), dtype=F)               # lnp: both scales 0.5, both informative
    beta = np.full(C, F(0.5), dtype=F)

    y = silu(groupnorm(conv(x, w1, b1, k=3, p=1, s=2), g, beta))          # 32x32x2
    low = silu(groupnorm(conv(y, w3, b3, k=3, p=1, s=2), g, beta))        # 16x16x2
    z = F(silu(groupnorm(conv(up2(low), w4, b4, k=3, p=1, s=1), g, beta)) + y)
    out = F(silu(groupnorm(conv(up2(z), w2, b2, k=3, p=1, s=1), g, beta)) + x)
    return {"y": y.reshape(-1)[:8], "z": z.reshape(-1)[:8], "out": out.reshape(-1)[:8]}


def build():
    if os.path.exists(BIN) and os.path.getmtime(BIN) >= os.path.getmtime(PROGRAM):
        return BIN
    out = subprocess.run([BEND, "-o", BIN, PROGRAM], capture_output=True, text=True,
                         env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the unet probe did not build:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    return BIN


def run():
    out = subprocess.run([build()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the unet probe did not run:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    return out.stdout


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main():
    text = run()
    ok = True
    want = twin()
    TOL = 1e-6   # measured 5.96e-08, so the gate sits ~17x above it
    got_of = {}
    for key in ("out", "z", "y"):
        line = re.search(rf"^{key}: (.*)$", text, re.M)
        got = np.array([float(v) for v in line.group(1).split()], dtype=F) if line else np.zeros(0)
        got_of[key] = got
        d = (float(np.max(np.abs(got.astype(np.float64) - want[key].astype(np.float64))))
             if got.size else float("inf"))
        ok &= check(key, got.size == want[key].size and d <= TOL,
                    f"first eight cells, maxabs {d:.3g}  tol {TOL:g}"
                    + (f"  bend[{got[0]:.6f}]  twin[{want[key][0]:.6f}]" if got.size else "  (no line)"))

    broken = want["out"].copy()
    broken[0] = F(broken[0] + 0.5)
    got = got_of["out"]
    d2 = float(np.max(np.abs(got.astype(np.float64) - broken.astype(np.float64)))) if got.size else float("inf")
    ok &= check("can-fail", d2 > TOL, f"a shifted expectation is rejected (maxabs {d2:.3g})")

    adjoint = re.search(r"^adjoint: (PASS|FAIL)$", text, re.M)
    ok &= check("up2 adjoint", bool(adjoint) and adjoint.group(1) == "PASS",
                adjoint.group(0) if adjoint else "no adjoint line")

    print(f"\n{'PASS' if ok else 'FAIL'}: the UNet's two-level forward")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
