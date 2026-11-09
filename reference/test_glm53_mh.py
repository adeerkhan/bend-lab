"""The four-head attention layers at N=8, against the reference modules.

  reference/.venv/bin/python reference/test_glm53_mh.py

A small forward check, not a training step.  x, qkv and out are generated
with the same seed `reference/glm53_attn_ref.py` uses; the NumPy twin is the
reference's `LinearAttention` and `SparseAttention` arithmetic (verified
against the vendored torch modules at 2e-08).  The Bend probe loads the same
files and runs the four-head wrapper at N=8.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

import numpy as np
import glm53_h_data as HD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
BIN = f"/tmp/glm53-mh-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")
DATA = os.path.join(EX, "data/ref8")
N, D, H = 8, 32, 4
HID = 128


def inputs():
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(71)
    x = (rng.standard_normal((1, N, D)) * 0.5).astype(np.float32)
    qkv = (rng.standard_normal((3 * D, D)) * 0.05).astype(np.float32)
    out = (rng.standard_normal((D, D)) * 0.05).astype(np.float32)
    for name, value in (("x", x), ("qkv", qkv), ("out", out)):
        value.astype("<f4").tofile(os.path.join(DATA, name + ".bin"))
    return x.reshape(N, D), qkv, out


def rope(x):
    # x: (N, H, HD)
    hd = x.shape[-1]
    positions = np.arange(N, dtype=np.float64)[:, None, None]
    freqs = 10000.0 ** (-2.0 * np.arange(hd // 2)[None, None, :] / hd)
    angles = positions * freqs
    cos, sin = np.cos(angles), np.sin(angles)
    even, odd = x[..., 0::2], x[..., 1::2]
    return np.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1).reshape(N, H, hd)


def heads(x, qkv, out):
    z = x @ qkv.T
    q = z[:, :D].reshape(N, H, D // H)
    k = z[:, D:2 * D].reshape(N, H, D // H)
    v = z[:, 2 * D:].reshape(N, H, D // H)
    return rope(q), rope(k), v, out


def linear(x, qkv, out):
    q, k, v, out = heads(x, qkv, out)
    qe = np.where(q > 0, q, np.exp(q) - 1.0) + 1.0
    ke = np.where(k > 0, k, np.exp(k) - 1.0) + 1.0
    y = np.zeros((N, H, D // H))
    for h in range(H):
        K = np.zeros(D // H)
        KV = np.zeros((D // H, D // H))
        for t in range(N):
            K = K + ke[t, h]
            KV = KV + np.outer(ke[t, h], v[t, h])
            y[t, h] = (KV.T @ qe[t, h]) / max(qe[t, h] @ K, 1e-6)
    return (y.reshape(N, D) @ out.T).reshape(-1), y.reshape(N, D)


def sparse(x, qkv, out):
    q, k, v, out = heads(x, qkv, out)
    y = np.zeros((N, H, D // H))
    for t in range(N):
        idx = [s for s in range(t + 1) if s % 16 == 0 or t - s < 16]
        for h in range(H):
            scores = np.array([q[t, h] @ k[s, h] for s in idx]) * (D // H) ** -0.5
            p = np.exp(scores - scores.max())
            p = p / p.sum()
            y[t, h] = sum(p[j] * v[s, h] for j, s in enumerate(idx))
    return (y.reshape(N, D) @ out.T).reshape(-1), y.reshape(N, D)


def rms(x, g):
    return g * (x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6))


def silu(x):
    return x / (1.0 + np.exp(-x))


def hybrid_forward(ids, w):
    x = w["emb"][ids].astype(np.float64)
    for i in range(4):
        h = rms(x, w[f"b{i}/ln1"].astype(np.float64))
        if i == 3:
            a, _ = sparse(h, w[f"b{i}/qkv"].astype(np.float64), w[f"b{i}/out"].astype(np.float64))
        else:
            a, _ = linear(h, w[f"b{i}/qkv"].astype(np.float64), w[f"b{i}/out"].astype(np.float64))
        x = x + a.reshape(N, D)
        h = rms(x, w[f"b{i}/ln2"].astype(np.float64))
        gate = h @ w[f"b{i}/gate"].astype(np.float64).T
        up = h @ w[f"b{i}/up"].astype(np.float64).T
        x = x + (silu(gate) * up) @ w[f"b{i}/down"].astype(np.float64).T
    x = rms(x, w["lnf"].astype(np.float64))
    return x @ w["emb"].astype(np.float64).T


def binary():
    src = os.path.join(EX, "probe_attn_ref.bend")
    deps = [os.path.join(EX, name) for name in
            ("probe_attn_ref.bend", "attn_mh8.bend", "attn_mh.bend", "lin_fast.bend",
             "lin_bwd.bend", "attn_c.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, src], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"probe_attn_ref.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def compare(name, got, want, limit=1e-5):
    rel = float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))
    ok = got.shape == want.shape and rel <= limit
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got.size} values, max |diff| "
          f"{np.abs(got - want).max():.3e}, relative {rel:.3e}")
    return ok


def main():
    x, qkv, out = inputs()
    proc = subprocess.run([binary()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_attn_ref.bend did not run:\n{proc.stdout}{proc.stderr}")
    got = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^(lin_y|sp_y):\s*(.*)$", line)
        if m:
            got[m.group(1)] = np.array([np.float32(v) for v in m.group(2).split()], dtype=np.float64)
    lin, _ = linear(x, qkv, out)
    sp, _ = sparse(x, qkv, out)
    ok = [compare("linear four-head forward", got["lin_y"], lin),
          compare("sparse four-head forward", got["sp_y"], sp)]
    rejected = not compare("can-fail: linear against sparse", got["lin_y"], sp)
    print(f"{'ok  ' if rejected else 'FAIL'} can-fail: linear output is not the sparse output")
    ok.append(rejected)
    # the full four-block model forward at N=8
    HD.write_weights()
    ids = np.array([1, 39, 36, 71, 115, 113, 116, 112], dtype=np.uint8)
    ids.tofile(os.path.join(DATA, "ids.bin"))
    hp = os.path.join(EX, "probe_h_fwd.bend")
    hbin = BIN + "-h"
    deps = [os.path.join(EX, name) for name in
            ("probe_h_fwd.bend", "model_h8.bend", "model_h.bend", "weights_h.bend",
             "attn_mh8.bend", "attn_mh.bend", "lin_fast.bend", "lin_bwd.bend", "attn_c.bend")]
    if not (os.path.exists(hbin) and all(os.path.getmtime(hbin) >= os.path.getmtime(s) for s in deps)):
        build = subprocess.run([BEND, "-o", hbin, hp], capture_output=True, text=True, env=ENV, cwd=ROOT)
        if build.returncode != 0:
            raise SystemExit(f"probe_h_fwd.bend did not build:\n{build.stdout}{build.stderr}")
    proc = subprocess.run([hbin], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_h_fwd.bend did not run:\n{proc.stdout}{proc.stderr}")
    m = re.search(r"^logits:\s*(.*)$", proc.stdout, re.M)
    got_logits = np.array([np.float32(v) for v in m.group(1).split()], dtype=np.float64)
    want_logits = hybrid_forward(ids, HD.weights()).reshape(-1)
    ok.append(compare("four-block model forward", got_logits, want_logits))

    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
