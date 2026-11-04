"""The UNet's forward against NumPy: a probe, not yet a gate.

  reference/.venv/bin/python reference/unet_gate.py

**It does not pass.** Channel 0 of every cell matches exactly; channel 1 is
1.88e-4 apart (0.02 in the normalised value), and that is not explained yet, so
this file is deliberately not `test_*.py`: `make check` must not mistake an
unresolved difference for a green one. The next session's first job is to close
it, then rename this to `reference/test_unet.py` and let it into the gate.

`examples/diffusion/unet/main.bend` composes the smallest honest U: one strided
`Conv2d` down, `GroupNorm`, `SiLU`, the `Up2` block, a same-shape `Conv2d`,
`GroupNorm`, `SiLU`, and an additive skip from the input. This twin rebuilds that
forward in float32 and compares it with the program's first eight numbers.

The layout the twin has to get right, and the reason this test exists:

  * a Mat row is one pixel and the channels are last (`Conv2d`'s layout), so the
    image is `(4096, 2)` for 64x64 and two channels;
  * `im2col` gathers a patch as `(c, kh, kw)` — channel outer, kernel inner —
    which is PyTorch's `unfold` order;
  * the conv's stride is `1 + s1`, so `s1 = 1` is stride 2 down and `s1 = 0` is
    stride 1;
  * `GroupNorm(n, hw, ng, cg)` normalises inside a group, here one group of both
    channels, per pixel;
  * the skip adds the model's own input, and `Up2` repeats each pixel 2x2.

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
    """one group over the whole channel axis, per pixel."""
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    r = F(1.0) / np.sqrt(v + EPS)
    return F(F(F(x - m) * r) * g + b)


def silu(x):
    return F(x * (F(1.0) / (F(1.0) + np.exp(-x))))


def up2(x):
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)


def twin():
    x = gen(H * W * C).reshape(H, W, C)
    w1 = gen(C * 9 * C).reshape(C, C * 9)        # Lin<18, 2>: (2, 18)
    b1 = np.full(C, F(0.05), dtype=F)
    w2 = gen(C * 9 * C).reshape(C, C * 9)
    b2 = np.full(C, F(0.07), dtype=F)
    g = gen(C)                                    # lnp: the same gen, cells 0..C-1
    beta = np.full(C, F(0.5), dtype=F)

    y = silu(groupnorm(conv(x, w1, b1, k=3, p=1, s=2), g, beta))   # 32x32x2
    z = silu(groupnorm(conv(up2(y), w2, b2, k=3, p=1, s=1), g, beta))
    out = F(z + x)
    return out.reshape(-1)[:8]


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
    line = re.search(r"^out: (.*)$", text, re.M)
    got = np.array([float(v) for v in line.group(1).split()], dtype=F) if line else np.zeros(0)
    want = twin()
    d = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))) if got.size else float("inf")
    # measured first, then fixed: float32 summation over 18 terms per output cell
    ok &= check("forward", got.size == want.size and d <= 1e-5,
                f"first eight cells, maxabs {d:.3g}  (channel 1 open: 1.88e-4)"
                + (f"  bend[{got[0]:.6f}]  twin[{want[0]:.6f}]" if got.size else "  (no out: line)"))

    broken = want.copy()
    broken[0] = F(broken[0] + 0.5)
    d2 = float(np.max(np.abs(got.astype(np.float64) - broken.astype(np.float64)))) if got.size else float("inf")
    ok &= check("can-fail", d2 > 1e-5, f"a shifted expectation is rejected (maxabs {d2:.3g})")

    adjoint = re.search(r"^adjoint: (PASS|FAIL)$", text, re.M)
    ok &= check("up2 adjoint", bool(adjoint) and adjoint.group(1) == "PASS",
                adjoint.group(0) if adjoint else "no adjoint line")

    print(f"\n{'PASS' if ok else 'FAIL'}: the UNet's forward")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
