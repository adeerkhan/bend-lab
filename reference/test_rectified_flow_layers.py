"""The rectified-flow port's layers: whole arrays against a NumPy twin.

  reference/.venv/bin/python reference/test_rectified_flow_layers.py

`reference/probe_rectified_flow_layers.bend` prints each layer's whole output
on one tagged line, on a flat ramp (and multiples) where a wrong index is
visible. This script rebuilds the probe when one of its sources is newer and
mirrors the layers with NumPy:

  down       pixel-unshuffle: out[b, y2, x2, c*4+p] = x[b, 2*y2+p//2, 2*x2+p%2, c]
  down_bwd   its exact inverse (the same permutation read backwards)
  pool       2x2 max, with the model's tie rule (a tie keeps the later cell)
  pool_bwd   the whole output gradient at the saved arg-max, zero elsewhere
  xattn      0.5 * (pool(history) + current)
  mix        learned query: softmax([kh.q, kx.q] / sqrt(8)) over (vh, xc)
  mix2       the same with kx = 2 * kh, so the row weights differ
  mixw       the saved softmax weights of the kx = 2 * kh case
  mb_*       its backward: dvh, dxc, dkh, dkx and the query gradient
  softmax    Softmax(2,2) row-wise

The permutation, max, scatter and zero-query mix are exact; the learned mix and
its backward are compared at 5e-3 (float32 exp/softmax at values around 3,000)
and the 2x2 softmax at 1e-6. Exit code 0 = every gate passed. Nothing here
writes results.
"""
import hashlib
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
# this machine's clang wrapper loads libLLVM from the local tree; CI needs none
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))

PROBE = "reference/probe_rectified_flow_layers.bend"
SOURCES = [
    PROBE,
    "examples/diffusion/rectified_flow/model.bend",
    "examples/diffusion/rectified_flow/config.bend",
]
# the root is in the name: two worktrees must not share one cached binary
BIN = "/tmp/mlbend-rectified-flow-layers-" + hashlib.sha1(ROOT.encode()).hexdigest()[:8]

H = 16          # the toy spatial edge
CH = 8          # channels
H2 = H // 2     # the down/pool edge
CH4 = CH * 4    # the unshuffled width


def probe_binary():
    """The probe's binary, rebuilt only when one of its sources is newer."""
    srcs = [os.path.join(ROOT, p) for p in SOURCES]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(p) for p in srcs):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(ROOT, PROBE)],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the layer probe did not build:\n" + build.stdout + build.stderr)
    return BIN


def parse_output(text):
    """The tagged lines, each as one float32 array."""
    got = {}
    for line in text.splitlines():
        match = re.match(r"^([a-z0-9_]+):\s*(.*)$", line)
        if match and match.group(2).strip():
            got[match.group(1)] = np.array(
                [np.float32(v) for v in match.group(2).split()], dtype=np.float32)
    return got


# ---- the twin ---------------------------------------------------------------

def max4(a, b, c, d):
    """max2(max2(a, b), max2(c, d)) with m2i's tie rule: a <= b keeps b."""
    va, ia = (b, 1) if a <= b else (a, 0)
    vq, iq = (d, 3) if c <= d else (c, 2)
    return (vq, iq) if va <= vq else (va, ia)


def down_twin(x):
    out = np.empty((2 * H2 * H2, CH4), dtype=np.float32)
    for r in range(2 * H2 * H2):
        b, rem = divmod(r, H2 * H2)
        y2, x2 = divmod(rem, H2)
        for c in range(CH):
            for p in range(4):
                y = 2 * y2 + p // 2
                xx = 2 * x2 + p % 2
                out[r, c * 4 + p] = x[b * H * H + y * H + xx, c]
    return out


def down_bwd_twin(dy):
    out = np.empty((H * H * 2, CH), dtype=np.float32)
    for r in range(2 * H2 * H2):
        b, rem = divmod(r, H2 * H2)
        y2, x2 = divmod(rem, H2)
        for c in range(CH):
            for p in range(4):
                y = 2 * y2 + p // 2
                xx = 2 * x2 + p % 2
                out[b * H * H + y * H + xx, c] = dy[r, c * 4 + p]
    return out


def pool_twin(x):
    out = np.empty((2 * H2 * H2, CH), dtype=np.float32)
    idx = np.empty((2 * H2 * H2, CH), dtype=np.int64)
    for r in range(2 * H2 * H2):
        b, rem = divmod(r, H2 * H2)
        y2, x2 = divmod(rem, H2)
        base = b * H * H + 2 * y2 * H + 2 * x2
        for c in range(CH):
            vals = [x[base, c], x[base + 1, c], x[base + H, c], x[base + H + 1, c]]
            v, i = max4(*vals)
            out[r, c] = v
            idx[r, c] = i
    return out, idx


def pool_bwd_twin(idx, dy):
    out = np.zeros((H * H * 2, CH), dtype=np.float32)
    for r in range(2 * H2 * H2):
        b, rem = divmod(r, H2 * H2)
        y2, x2 = divmod(rem, H2)
        for c in range(CH):
            i = int(idx[r, c])
            y = 2 * y2 + i // 2
            xx = 2 * x2 + i % 2
            out[b * H * H + y * H + xx, c] = dy[r, c]
    return out


def dot8(xs, ys):
    """kh.q in the model's order: a0*b0, then one rounded add per term."""
    s = np.float32(xs[0] * ys[0])
    for k in range(1, 8):
        s = np.float32(s + np.float32(xs[k] * ys[k]))
    return s


def weights_twin(kh, kx, q):
    scale = np.float32(0.35355338)
    logits = np.empty((len(kh), 2), dtype=np.float32)
    for r in range(len(kh)):
        logits[r, 0] = np.float32(dot8(kh[r], q) * scale)
        logits[r, 1] = np.float32(dot8(kx[r], q) * scale)
    m = np.maximum(logits[:, 0], logits[:, 1])
    e = np.empty_like(logits)
    e[:, 0] = np.exp(np.float32(logits[:, 0] - m), dtype=np.float32)
    e[:, 1] = np.exp(np.float32(logits[:, 1] - m), dtype=np.float32)
    s = np.float32(e[:, 0] + e[:, 1])
    return e / s[:, None]


def mix_twin(vh, xc, kh, kx, q):
    w = weights_twin(kh, kx, q)
    out = np.empty_like(vh)
    for r in range(len(vh)):
        for c in range(vh.shape[1]):
            out[r, c] = np.float32(w[r, 0] * vh[r, c] + w[r, 1] * xc[r, c])
    return out


def dot8row(off, xs, ys):
    """A row dot in the model's order: x[off]*y[off], then one add per term."""
    s = np.float32(xs[off] * ys[off])
    for k in range(1, 8):
        s = np.float32(s + np.float32(xs[off + k] * ys[off + k]))
    return s


# ---- the gates --------------------------------------------------------------

results = []


def gate(name, got, want, atol):
    if got is None:
        results.append(False)
        print(f"FAIL {name}: no line in the probe's output")
        return
    if got.shape != want.shape:
        results.append(False)
        print(f"FAIL {name}: {got.shape} values, the twin has {want.shape}")
        return
    diff = float(np.abs(got - want).max()) if got.size else 0.0
    ok = diff <= atol
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got.size} values, max |diff| {diff:.3e} (atol {atol:g})")


def main():
    out = subprocess.run([probe_binary()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        print(f"FAIL the probe did not run:\n{out.stdout}{out.stderr}")
        return 1
    got = parse_output(out.stdout)

    x = np.arange(2 * H * H * CH, dtype=np.float32).reshape(2 * H * H, CH)
    vh = np.arange(2 * H2 * H2 * CH, dtype=np.float32).reshape(2 * H2 * H2, CH)
    q = np.full(CH, 0.1, dtype=np.float32)

    pool, idx = pool_twin(x)
    gate("down", got.get("down"), down_twin(x).reshape(-1), 0.0)
    gate("down_bwd", got.get("down_bwd"),
         down_bwd_twin(np.arange(2 * H2 * H2 * CH4, dtype=np.float32).reshape(2 * H2 * H2, CH4)).reshape(-1), 0.0)
    gate("pool", got.get("pool"), pool.reshape(-1), 0.0)
    gate("pool_bwd", got.get("pool_bwd"), pool_bwd_twin(idx, vh).reshape(-1), 0.0)
    gate("xattn", got.get("xattn"),
         (np.float32(0.5) * (pool + vh)).astype(np.float32).reshape(-1), 0.0)
    gate("mix", got.get("mix"), mix_twin(vh, 3 * vh, vh, vh, q).reshape(-1), 5e-3)
    gate("mix2", got.get("mix2"), mix_twin(vh, 3 * vh, vh, 2 * vh, q).reshape(-1), 5e-3)
    w = weights_twin(vh, 2 * vh, q)
    gate("mixw", got.get("mixw"), w.reshape(-1), 1e-6)
    dout = vh
    dout_f, vh_f, xc_f = dout.reshape(-1), vh.reshape(-1), (3 * vh).reshape(-1)
    dwh = np.array([dot8row(8 * r, dout_f, vh_f) for r in range(len(vh))], dtype=np.float32)
    dwx = np.array([dot8row(8 * r, dout_f, xc_f) for r in range(len(vh))], dtype=np.float32)
    sft = np.float32(w[:, 0] * dwh + w[:, 1] * dwx)
    da = (w[:, 0] * (dwh - sft)).astype(np.float32)
    db = (w[:, 1] * (dwx - sft)).astype(np.float32)
    dda = (da * np.float32(0.35355338)).astype(np.float32)
    ddb = (db * np.float32(0.35355338)).astype(np.float32)
    gate("mb_dvh", got.get("mb_dvh"), (w[:, 0:1] * dout).astype(np.float32).reshape(-1), 5e-3)
    gate("mb_dxc", got.get("mb_dxc"), (w[:, 1:2] * (3 * vh)).astype(np.float32).reshape(-1), 5e-3)
    gate("mb_dkh", got.get("mb_dkh"), (dda[:, None] * q[None, :]).astype(np.float32).reshape(-1), 5e-3)
    gate("mb_dkx", got.get("mb_dkx"), (ddb[:, None] * q[None, :]).astype(np.float32).reshape(-1), 5e-3)
    dq = (dda[:, None] * vh + ddb[:, None] * (2 * vh)).sum(axis=0, dtype=np.float32)
    gate("mb_dq", got.get("mb_dq"), dq.reshape(-1), 5e-3)

    sm = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    e = np.exp(sm - sm.max(axis=1, keepdims=True), dtype=np.float32)
    gate("softmax", got.get("softmax"), (e / e.sum(axis=1, keepdims=True, dtype=np.float32)).reshape(-1), 1e-6)

    ok = sum(results)
    print(f"\n{ok}/{len(results)} gates passed")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
