"""Two 128-token sequences in one 256-row model: isolation of sequence 1.

  reference/.venv/bin/python reference/test_glm53_iso2.py

`probe_h2_iso.bend` runs the 2 x 128 model twice, the second time with one
token of sequence 0 changed.  Sequence 1's 128 x 260 logits must be
bit-identical; sequence 0's must move, so the comparison is not blind.
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
BIN = f"/tmp/glm53-iso2-{TAG}"
BIN2 = f"/tmp/glm53-iso2-bwd-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")


def binary(src, out):
    deps = [os.path.join(EX, name) for name in
            (src, "model_h2.bend", "weights_h2.bend", "attn_mh2.bend",
             "lin_dense.bend", "sp_dense.bend", "model_h.bend")]
    if os.path.exists(out) and all(os.path.getmtime(out) >= os.path.getmtime(s) for s in deps):
        return out
    build = subprocess.run([BEND, "-o", out, os.path.join(EX, src)],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"{src} did not build:\n{build.stdout}{build.stderr}")
    return out


def main():
    proc = subprocess.run([binary("probe_h2_iso.bend", BIN)], capture_output=True,
                          text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_h2_iso.bend did not run:\n{proc.stdout}{proc.stderr}")
    got = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^(seq[01]_diff):\s*([0-9.eE+-]+)$", line)
        if m:
            got[m.group(1)] = float(m.group(2))
    proc = subprocess.run([binary("probe_h2_bwd.bend", BIN2)], capture_output=True,
                          text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_h2_bwd.bend did not run:\n{proc.stdout}{proc.stderr}")
    for line in proc.stdout.splitlines():
        m = re.match(r"^(grad_diff):\s*([0-9.eE+-]+)$", line)
        if m:
            got[m.group(1)] = float(m.group(2))
    ok = []
    a = got.get("seq1_diff")
    good = a == 0.0
    print(f"{'ok  ' if good else 'FAIL'} sequence 1 logits unchanged: {a}")
    ok.append(good)
    b = got.get("seq0_diff")
    moved = b is not None and b > 1e-4
    print(f"{'ok  ' if moved else 'FAIL'} sequence 0 logits moved: {b} (must be > 1e-4)")
    ok.append(moved)
    g = got.get("grad_diff")
    grad_same = g == 0.0
    print(f"{'ok  ' if grad_same else 'FAIL'} sequence 1 gradient unchanged: {g}")
    ok.append(grad_same)
    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
