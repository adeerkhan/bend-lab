"""The Stage B linear-attention backward gate.

  reference/.venv/bin/python reference/test_glm53_bwd.py

Two scales, no PyTorch in the gate:

  t = 2, one head, hd = 8   the probe's y, dq, dk, dv against a NumPy twin,
                            plus the one-hot dout at t = 0: dq, dk and dv
                            must be exactly zero for every t > 0.
  t = 64, one head, hd = 8  the whole 512-value tensors against the frozen
                            PyTorch-autograd golden (`glm53_linear_bwd_golden.npz`).

The first is the can-fail the old 64-step implementation lacked: a past-row
contribution in the suffix makes the t > 0 rows nonzero immediately.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glm53_linear_bwd as G  # noqa: E402

BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
EX = os.path.join(ROOT, "examples/language/glm53_flash")
GOLDEN = os.path.join(ROOT, "reference/glm53_linear_bwd_golden.npz")

T2_Q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
                 0.2, 0.4, -0.1, 0.6, -0.3, 0.8, -0.5, 0.9], dtype=np.float64)
T2_K = np.array([0.3, -0.1, 0.2, 0.5, -0.4, 0.1, -0.2, 0.6,
                 -0.2, 0.7, 0.1, -0.3, 0.4, -0.6, 0.2, 0.5], dtype=np.float64)
T2_V = np.array([1.0, -2.0, 0.5, 0.25, -0.75, 1.5, -0.5, 2.0,
                 0.5, 0.25, -1.0, 2.0, -0.25, 0.75, 1.25, -1.5], dtype=np.float64)
T2_DY = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.4, -0.3, 0.5,
                  -0.1, 0.3, 0.2, -0.4, 0.5, -0.2, 0.1, -0.3], dtype=np.float64)


def binary(source, tag):
    out = f"/tmp/glm53-bwd-{TAG}-{tag}"
    deps = [os.path.join(EX, name) for name in (source, "lin_bwd.bend")]
    if os.path.exists(out) and all(os.path.getmtime(out) >= os.path.getmtime(s) for s in deps):
        return out
    build = subprocess.run([BEND, "-o", out, os.path.join(EX, source)],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"{source} did not build:\n{build.stdout}{build.stderr}")
    return out


def run(source, tag):
    proc = subprocess.run([binary(source, tag)], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"{source} did not run:\n{proc.stdout}{proc.stderr}")
    out = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^([a-z_]+):\s*(.*)$", line)
        if m:
            out[m.group(1)] = np.array([np.float32(v) for v in m.group(2).split()], dtype=np.float64)
    return out


def twin(n, hd, q, k, v, dy):
    q, k, v, dy = (a.reshape(n, hd) for a in (q, k, v, dy))
    qe = np.where(q > 0, q, np.exp(q) - 1.0) + 1.0
    ke = np.where(k > 0, k, np.exp(k) - 1.0) + 1.0
    K = np.zeros((n, hd)); KV = np.zeros((n, hd, hd))
    den = np.zeros(n); num = np.zeros((n, hd)); y = np.zeros((n, hd))
    for t in range(n):
        K[t] = (K[t - 1] if t else 0.0) + ke[t]
        KV[t] = (KV[t - 1] if t else 0.0) + np.outer(ke[t], v[t])
        den[t] = qe[t] @ K[t]
        num[t] = KV[t].T @ qe[t]
        y[t] = num[t] / max(den[t], 1e-6)
    c = np.maximum(den, 1e-6)
    dn = dy / c[:, None]
    dd = np.where(den >= 1e-6, -(dy * num).sum(1) / c**2, 0.0)
    dq = np.zeros_like(q); dk = np.zeros_like(k); dv = np.zeros_like(v)
    N = np.zeros((hd, hd)); Q = np.zeros(hd)
    for s in range(n - 1, -1, -1):
        N = N + np.outer(dn[s], qe[s]); Q = Q + dd[s] * qe[s]
        dv[s] = N @ ke[s]
        dk[s] = N.T @ v[s] + Q
        dq[s] = KV[s] @ dn[s] + dd[s] * K[s]
    dq = dq * np.where(q > 0, 1.0, np.exp(q))
    dk = dk * np.where(k > 0, 1.0, np.exp(k))
    return y.reshape(-1), dq.reshape(-1), dk.reshape(-1), dv.reshape(-1)


def compare(name, got, want, limit=1e-4):
    scale = max(float(np.abs(want).max()), 1e-30)
    rel = float(np.abs(got - want).max() / scale)
    ok = got.shape == want.shape and rel <= limit and np.isfinite(got).all()
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got.size} values, max |diff| "
          f"{np.abs(got - want).max():.3e}, relative {rel:.3e}")
    return ok


def main():
    ok = []
    out = run("probe_lin_bwd_t2.bend", "t2")
    want = twin(2, 8, T2_Q, T2_K, T2_V, T2_DY)
    for tag, name in (("y_a", "y"), ("a_dq", "dq"), ("a_dk", "dk"), ("a_dv", "dv")):
        ok.append(compare(f"t2/{name}", out[tag], want[["y", "dq", "dk", "dv"].index(name)]))
    # the one-hot dout at t = 0: every t > 0 row must be exactly zero
    for tag, name in (("b_dq", "dq"), ("b_dk", "dk"), ("b_dv", "dv")):
        tail = out[tag][8:]
        zero = float(np.abs(tail).max()) == 0.0
        print(f"{'ok  ' if zero else 'FAIL'} t2/one-hot/{name}: max |t>0| {float(np.abs(tail).max()):.3e}")
        ok.append(zero)

    G.write_inputs()
    got64 = run("probe_lin_bwd_64.bend", "t64")
    golden = np.load(GOLDEN)
    for name in ("y", "dq", "dk", "dv"):
        ok.append(compare(f"t64/{name}", got64[name], golden[name].astype(np.float64)))

    # can-fail: a changed dout must be rejected by the same t2 comparison
    bad = T2_DY.copy()
    bad[8] += 0.5
    bad_want = twin(2, 8, T2_Q, T2_K, T2_V, bad)
    got_dq = out["a_dq"]
    rejected = not compare("t2/corrupt-dout", got_dq, bad_want[1], 1e-4)
    print(f"{'ok  ' if rejected else 'FAIL'} t2/corrupt-dout: rejected")
    ok.append(rejected)
    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
