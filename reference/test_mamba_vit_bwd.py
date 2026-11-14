"""MambaViT A2 gate: the hand-written backward against PyTorch autograd.

  reference/.venv/bin/python reference/test_mamba_vit_bwd.py

Requires the reference dump (`reference/mamba_vit_bins.py`, 251 files in
`.scratch/mamba_vit`). Builds `probe_mamba_vit_bwd.bend`, runs it and compares
every gradient the dump carries at 1e-4 relative:

  g_emb, g_tokens        the patch tokens, the 50 tokens with cls + pos
  g_block0..7            each Mamba-1 block's mixer output
  g_final, g_pooled      the final LayerNorm, the mean over the 50 tokens
  g_<param>              every parameter of the model (100 tensors)

The loss is `sum(logits * dy)` from a fixed dy, so the gradients are the same
on every lane. `MAMBA-VIT FWD PASS` is printed by `test_mamba_vit.py`; this
file prints only the backward's verdict.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(ROOT, ".scratch/mamba_vit")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mv_bwd.bin")
SRC = os.path.join(ROOT, "examples/language/mamba_vit/probe_mamba_vit_bwd.bend")
GATE = 1e-4


def run_bend():
    env = dict(os.environ, BEND_NO_TELEMETRY="1")
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")
    subprocess.run([BEND, "-o", BIN, SRC], cwd=ROOT, env=env, check=True,
                   capture_output=True, timeout=1800)
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=3600)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("mamba-vit-bwd binary failed")
    tags = {}
    for line in p.stdout.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        v = np.fromstring(rest, sep=" ", dtype=np.float32)
        if v.size:
            tags[tag] = v
    return tags


def main():
    if not os.path.exists(os.path.join(DUMP, "dy.bin")):
        raise SystemExit(f"missing A2 dump in {DUMP}; run reference/mamba_vit_bins.py")
    keys = sorted(f[:-4] for f in os.listdir(DUMP)
                  if f.startswith("g_") and f.endswith(".bin"))
    if not keys:
        raise SystemExit(f"no g_*.bin in {DUMP}")
    tags = run_bend()
    rows = []
    ok = True
    for key in keys:
        tag = "mv_" + key
        want = np.fromfile(os.path.join(DUMP, key + ".bin"), dtype="<f4").astype(np.float64)
        got = tags.get(tag)
        if got is None:
            print(f"FAIL {tag}: not printed")
            ok = False
            continue
        got = got.astype(np.float64)
        if got.size != want.size:
            print(f"FAIL {tag}: size {got.size} != {want.size}")
            ok = False
            continue
        d = np.abs(got - want)
        scale = float(np.abs(want).max())
        rel = float(d.max()) / max(1e-30, scale)
        one = rel <= GATE
        ok = ok and one
        rows.append((rel, tag, float(d.max()), scale, one))
    rows.sort(reverse=True)
    print(f"{'tag':38s} {'max|d|':>10s} {'max|want|':>10s} {'relative':>10s}")
    for rel, tag, dm, scale, one in rows[:20]:
        print(f"{tag:38s} {dm:10.3e} {scale:10.3e} {rel:10.3e}"
              f"{'' if one else '  FAIL'}")
    worst = rows[0]
    print(f"worst: {worst[1]} rel {worst[0]:.3e} "
          f"(max|d| {worst[2]:.3e}, max|want| {worst[3]:.3e}), gate {GATE:.0e}, "
          f"{len(keys)} gradients")
    print("MAMBA-VIT BWD", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
