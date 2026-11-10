"""The dense linear attention core, against lin_fast/lin_bwd at 128.

  reference/.venv/bin/python reference/test_glm53_lin_dense.py

`probe_lin_dense.bend` runs the dense core and the per-position prefix core on
the same q, k, v and dy, and prints the largest absolute difference for y, dq,
dk and dv.  The gate is 1e-6.  The last gate removes the causal mask: the same
comparison must reject it, so a core that ignored the mask would not pass.
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
BIN = f"/tmp/glm53-lin-dense-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")


def binary():
    deps = [os.path.join(EX, name) for name in
            ("probe_lin_dense.bend", "lin_dense.bend", "lin_fast.bend", "lin_bwd.bend",
             "sp_dense.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(EX, "probe_lin_dense.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"probe_lin_dense.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def main():
    proc = subprocess.run([binary()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_lin_dense.bend did not run:\n{proc.stdout}{proc.stderr}")
    got = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^(n128_[a-z]+_diff):\s*([0-9.eE+-]+)$", line)
        if m:
            got[m.group(1)] = float(m.group(2))
    ok = []
    for key in ("n128_y_diff", "n128_dq_diff", "n128_dk_diff", "n128_dv_diff"):
        value = got.get(key)
        good = value is not None and value <= 1e-6
        text = "unavailable" if value is None else f"{value:.3e}"
        print(f"{'ok  ' if good else 'FAIL'} {key}: {text} (gate 1e-6)")
        ok.append(good)
    bad = got.get("n128_uncased_diff")
    good = bad is not None and bad > 1e-4
    text = "unavailable" if bad is None else f"{bad:.3e}"
    print(f"{'ok  ' if good else 'FAIL'} n128_uncased_diff: {text} (must be > 1e-4)")
    ok.append(good)
    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
