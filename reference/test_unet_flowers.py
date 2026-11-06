"""The trained UNet's forward against NumPy.

  reference/.venv/bin/python reference/test_unet_flowers.py

`examples/diffusion/unet/train.bend --probe --steps 0` prints the model's flat
weights and one whole forward pass at a fixed input: `noise_of(7, 8192)` as the
image channel and `t = 0.5` as the level channel, two images of 64x64. This twin
rebuilds that forward in float32 from the same weights and compares all 8,192
cells (two images, one channel each).

The shapes, in the order the flat weights are laid out (the model's forward
order, 76,871 values):

  w1 (64, 45)  b1 (64)                      conv1 5 -> 64, stride 2
  w3 (64, 576) b3 (64)                      conv3 64 -> 64, stride 2
  w4 (64, 576) b4 (64)                      conv4 64 -> 64, stride 1
  wh (1, 64)   bh (1)                       the head, a 1x1 conv, no activation
  wl (1, 5)    bl (1)                       the linear bypass from the input

  y = silu(conv(x))                        32x32
  l = silu(conv(y))                        16x16
  z = silu(conv(up(l)))                    32x32
  o = conv1x1(up(z + y)) + conv1x1(x)      64x64, one channel

The output is the model's prediction of the clean image `x0` (the training
target since 2026-11-06; the sampler turns it into a velocity with
`v = (x0hat - x)/(1-t)`). There are no normalisations: a group norm over all of
an image's pixels removes the image's amplitude, which is what the model needs
to know to remove noise, and the same model with the norms drew speckle.

The layout facts are `reference/test_unet.py`'s: a Mat row is one pixel and the
channels are last; im2col gathers `(c, kh, kw)`; the conv weight is row-major
`(co, c*k*k)`. The batch is two images, and the residual adds the 32x32 level to
the decoder's 32x32 output.

Exit 0 = the forward matched to 1e-5. Nothing here writes.
"""
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRAM = os.path.join(ROOT, "examples/diffusion/unet/train.bend")
SOURCES = [PROGRAM,
           os.path.join(ROOT, "examples/diffusion/unet/model.bend"),
           os.path.join(ROOT, "examples/diffusion/unet/config.bend")]
BIN = "/tmp/mlbend-unet-flowers"
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

F = np.float32
EPS = F(1e-5)
HW, C = 4096, 64
NIMG = 2
# measured first, then set: 2.15e-06 on 40,960 cells with two group norms at
# eps = 1e-5 (a norm divides small variances up), so the gate sits ~5x above it
TOL = 1e-5

# the flat weights, in the order the model's forward pass dumps them
PARTS = [(64, 45), (64,),
         (64, 576), (64,),
         (64, 576), (64,),
         (1, 64), (1,),
         (1, 5), (1,)]


def im2col(x, k=3, p=1, s=1):
    """x: (h, w, c) -> rows are output pixels, columns are (c, kh, kw)."""
    h, w, c = x.shape
    xp = np.pad(x, ((p, p), (p, p), (0, 0)))
    oh = (h + 2 * p - k) // s + 1
    ow = (w + 2 * p - k) // s + 1
    rows = np.empty((oh * ow, c * k * k), dtype=F)
    for oy in range(oh):
        for ox in range(ow):
            win = xp[oy * s:oy * s + k, ox * s:ox * s + k, :]
            rows[oy * ow + ox] = win.transpose(2, 0, 1).reshape(-1)
    return rows, oh, ow


def conv(x, w, b, k=3, p=1, s=1):
    rows, oh, ow = im2col(x, k, p, s)
    return (rows @ w.T + b).reshape(oh, ow, -1).astype(F)




def silu(x):
    return F(x * (F(1.0) / (F(1.0) + np.exp(-x))))


def up2(x):
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)


def forward(flat, inp):
    f = np.asarray(flat, dtype=F)
    off, parts = 0, []
    for shape in PARTS:
        size = int(np.prod(shape))
        parts.append(f[off:off + size].reshape(shape))
        off += size
    if off != f.size:
        raise SystemExit(f"the probe's weights are {f.size} values, the model takes {off}")
    w1, b1, w3, b3, w4, b4, wh, bh, wl, bl = parts

    outs = []
    for img in range(NIMG):
        x = inp[img * HW:(img + 1) * HW].reshape(64, 64, 5)
        y = silu(conv(x, w1, b1, s=2))
        low = silu(conv(y, w3, b3, s=2))
        z = F(silu(conv(up2(low), w4, b4, s=1)) + y)
        o = conv(up2(z), wh, bh, k=1, p=0, s=1)          # the head: a 1x1 conv, no activation
        o = F(o + conv(x, wl, bl, k=1, p=0, s=1))        # plus the learned linear bypass
        outs.append(o.reshape(HW, 1))
    return np.concatenate(outs).reshape(-1)


def build():
    """the binary, built under the repository's memory guard (the model is a 9 GB build)."""
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(src)
                                   for src in SOURCES):
        return BIN
    guard = os.path.join(ROOT, "scripts", "bend-guard.sh")
    cmd = ([guard, BEND, "-o", BIN, PROGRAM] if os.path.exists(guard) else [BEND, "-o", BIN, PROGRAM])
    env = dict(ENV, BEND_MEM=os.environ.get("BEND_MEM", "9G"))
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the unet-flowers probe did not build:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    return BIN


def probe():
    out = subprocess.run([build(), "--probe", "--steps", "0"], capture_output=True, text=True,
                         env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the unet-flowers probe did not run:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    text = out.stdout
    w = np.array([float(v) for v in re.search(r"^w: (.*)$", text, re.M).group(1).split()], dtype=F)
    o = np.array([float(v) for v in re.search(r"^out: (.*)$", text, re.M).group(1).split()], dtype=F)
    return w, o


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_diffusion_digits import noise_of

    w, got = probe()
    x = np.asarray(noise_of(7, 2 * HW), dtype=F).reshape(-1)[:2 * HW]
    inp = np.empty((2 * HW, 5), dtype=F)
    inp[:, 0] = x
    for k, ch in enumerate((np.sin, np.cos)):
        pass
    tau = F(2.0 * np.pi)
    inp[:, 1] = np.sin(F(0.5) * tau)
    inp[:, 2] = np.cos(F(0.5) * tau)
    inp[:, 3] = np.sin(F(1.0) * tau)
    inp[:, 4] = np.cos(F(1.0) * tau)
    want = forward(w, inp)

    ok = True
    ok &= check("weights", w.size == 76871, f"{w.size} values, the model takes 76871")
    ok &= check("output shape", got.size == want.size, f"{got.size} cells, twin {want.size}")
    d = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))) if got.size == want.size else float("inf")
    ok &= check("forward", d <= TOL, f"two images, maxabs {d:.3e}  tol {TOL:g}")

    broken = want.copy()
    broken[0] = F(broken[0] + 0.5)
    d2 = float(np.max(np.abs(got.astype(np.float64) - broken.astype(np.float64)))) if got.size == want.size else float("inf")
    ok &= check("can-fail", d2 > TOL, f"a shifted expectation is rejected (maxabs {d2:.3g})")

    print(f"\n{'PASS' if ok else 'FAIL'}: the trained UNet's forward")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
