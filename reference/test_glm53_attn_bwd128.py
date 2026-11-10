"""The four-head attention backward at N=128, against PyTorch autograd.

  reference/.venv/bin/python reference/test_glm53_attn_bwd.py

The forward and the backward through the whole layer (qkv projection, split,
RoPE, core, out) for the linear and the sparse core.  The golden is
`reference/glm53_attn_bwd.py`'s autograd on the vendored modules; the probe
prints dx, the qkv weight gradient and the out weight gradient.  The last
three gates are the can-fail: the gradients of a deliberately moved dout must
NOT match the correct golden, while matching the bad-dout golden.
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
import glm53_attn_bwd128_data as D  # noqa: E402

BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
BIN = f"/tmp/glm53-attn-bwd-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")
GOLDEN = os.path.join(ROOT, "reference/glm53_attn_bwd128_golden.npz")


def binary():
    deps = [os.path.join(EX, name) for name in
            ("probe_attn_bwd128.bend", "attn_mh8.bend", "attn_mh.bend", "lin_fast.bend",
             "lin_bwd.bend", "attn_c.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(EX, "probe_attn_bwd128.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"probe_attn_bwd128.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def run():
    proc = subprocess.run([binary()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_attn_bwd128.bend did not run:\n{proc.stdout}{proc.stderr}")
    out = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^([a-z_]+):\s*(.*)$", line)
        if m:
            out[m.group(1)] = np.array([np.float32(v) for v in m.group(2).split()], dtype=np.float64)
    return out


def compare(name, got, want, limit=1e-5):
    rel = float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))
    ok = got.shape == want.shape and rel <= limit and np.isfinite(got).all()
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got.size} values, max |diff| "
          f"{np.abs(got - want).max():.3e}, relative {rel:.3e}")
    return ok


def reject(name, got, want, limit=1e-5):
    rel = float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))
    ok = got.shape == want.shape and rel > limit
    print(f"{'ok  ' if ok else 'FAIL'} {name}: relative {rel:.3e} "
          f"(a match would be <= {limit:.0e})")
    return ok


def main():
    D.write()
    out = run()
    golden = np.load(GOLDEN)
    ok = []
    for prefix in ("lin", "sp"):
        for name in ("y", "dx", "dqkv", "dout"):
            key = f"{prefix}_{name}"
            ok.append(compare(f"{prefix}/{name}", out[key], golden[key].astype(np.float64)))
    # can-fail: the bad-dout gradients must not match the correct golden
    for name in ("dx", "dqkv", "dout"):
        rejected = reject(f"can-fail: bad dout/{name} is rejected",
                          out[f"lin_bad_{name}"], golden[f"lin_{name}"].astype(np.float64))
        ok.append(rejected)
    # and they must match their own golden, so the probe is not simply broken
    ok.append(compare("can-fail: bad dout/dx matches its own golden",
                      out["lin_bad_dx"], golden["lin_bad_dx"].astype(np.float64)))
    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
