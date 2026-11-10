"""The sparse mask at 64 positions: 16/16 and 32/32 against _indices.

  reference/.venv/bin/python reference/test_glm53_sparse_mask.py

The Bend probe prints three 64 x 64 masks as 0 / -1e30.  The reference recipe
is the vendored SparseAttention._indices: s <= t and (s % stride == 0 or
t - s < window).  The 8/8 mask is the can-fail: it must differ from 16/16.
"""
from __future__ import annotations

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
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
BIN = f"/tmp/glm53-mask-ws-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")
N = 64


def binary():
    deps = [os.path.join(EX, name) for name in ("probe_mask_ws.bend", "sp_dense.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(EX, "probe_mask_ws.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"probe_mask_ws.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def run():
    proc = subprocess.run([binary()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_mask_ws.bend did not run:\n{proc.stdout}{proc.stderr}")
    out = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^(m[0-9]+):\s*(.*)$", line)
        if m:
            values = np.array([np.float32(v) for v in m.group(2).split()], dtype=np.float64)
            out[m.group(1)] = (values.reshape(N, N) > -1e20).astype(np.int8)
    return out


def reference(window, stride):
    out = np.zeros((N, N), dtype=np.int8)
    for t in range(N):
        for s in range(t + 1):
            if s % stride == 0 or t - s < window:
                out[t, s] = 1
    return out


def main():
    got = run()
    ok = []
    for key, window, stride in (("m1616", 16, 16), ("m3232", 32, 32)):
        want = reference(window, stride)
        got_m = got.get(key)
        same = got_m is not None and np.array_equal(got_m, want)
        diff = "unavailable" if got_m is None else str(int(np.abs(got_m - want).sum()))
        print(f"{'ok  ' if same else 'FAIL'} {key}: {diff} cells differ from _indices")
        ok.append(same)
    a, b = got.get("m88"), got.get("m1616")
    differ = a is not None and b is not None and not np.array_equal(a, b)
    cells = "unavailable" if a is None or b is None else str(int(np.abs(a - b).sum()))
    print(f"{'ok  ' if differ else 'FAIL'} can-fail m88 vs m1616: {cells} cells differ")
    ok.append(differ)
    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
