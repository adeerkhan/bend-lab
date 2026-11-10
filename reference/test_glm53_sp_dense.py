"""The vectorized sparse core, against the gathered list core.

  reference/.venv/bin/python reference/test_glm53_sp_dense.py

`probe_sp_dense.bend` runs both cores at 32 and 128 positions on the same
deterministic q, k, v and dy, and prints the largest absolute difference for
y, dq, dk and dv.  The gate is 1e-6; the measured values are 9e-08 to 6e-07,
which is the float32 order of the two summation orders.

The mask is the reference's: s <= t and (s % 16 == 0 or t - s < 16).  A version
with the s == t column missing was the can-fail during development: it moved
y by 0.30 at 4 positions and 0.48 at 128, and every gradient was still right,
so the forward comparison is what catches a bad mask.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

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
BIN = f"/tmp/glm53-sp-dense-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")


def binary():
    deps = [os.path.join(EX, name) for name in
            ("probe_sp_dense.bend", "sp_dense.bend", "attn_c.bend", "lin_bwd.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(EX, "probe_sp_dense.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"probe_sp_dense.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def main():
    proc = subprocess.run([binary()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_sp_dense.bend did not run:\n{proc.stdout}{proc.stderr}")
    got = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^(n(?:32|128)_[a-z]+_diff):\s*([0-9.eE+-]+)$", line)
        if m:
            got[m.group(1)] = float(m.group(2))
    ok = []
    for key in ("n32_y_diff", "n32_dq_diff", "n32_dk_diff", "n32_dv_diff",
                "n128_y_diff", "n128_dq_diff", "n128_dk_diff", "n128_dv_diff"):
        value = got.get(key)
        good = value is not None and value <= 1e-6
        text = "unavailable" if value is None else f"{value:.3e}"
        print(f"{'ok  ' if good else 'FAIL'} {key}: {text} (gate 1e-6)")
        ok.append(good)
    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
