"""MambaViT gate, first slice (A1a): the patch embedding and the tokens.

  reference/.venv/bin/python reference/test_mamba_vit.py

Requires the reference dump (`reference/mamba_vit_bins.py`, 130 files in
`.scratch/mamba_vit`). Builds `probe_mamba_vit_emb.bend`, runs it and
compares at 1e-5 relative:

  mv_emb     (49,32)  the patch tokens after LN -> Linear -> LN
  mv_tokens  (50,32)  cls token prepended, pos embedding added

The eight Mamba-1 blocks, the final norm, the pool and the head are not
built yet (A1b); this gate covers the ViT half only, and the row stays
`not run` in the batch until the full forward is compared.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(ROOT, ".scratch/mamba_vit")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mv_emb.bin")
GATE = 1e-5
PAIRS = [("mv_emb", "emb"), ("mv_tokens", "tokens")]


def main():
    if not os.path.exists(os.path.join(DUMP, "emb.bin")):
        raise SystemExit(f"missing dump in {DUMP}; run reference/mamba_vit_bins.py")
    env = dict(os.environ, BEND_NO_TELEMETRY="1")
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")
    subprocess.run([BEND, "-o", BIN,
                    os.path.join(ROOT, "examples/language/mamba_vit/probe_mamba_vit_emb.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True, timeout=1800)
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("mamba-vit-emb binary failed")
    tags = {}
    for line in p.stdout.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        v = np.fromstring(rest, sep=" ", dtype=np.float32)
        if v.size:
            tags[tag] = v
    ok = True
    for tag, key in PAIRS:
        got = tags.get(tag)
        want = np.fromfile(os.path.join(DUMP, key + ".bin"), dtype="<f4").astype(np.float64)
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
        rel = float(d.max()) / max(1e-30, float(np.abs(want).max()))
        good = rel <= GATE
        print(f"{'ok  ' if good else 'FAIL'} {tag}: max|d| {d.max():.6e} "
              f"rel {rel:.6e} gate {GATE:.0e}")
        ok = ok and good
    print("unavailable blocks/final/pooled/logits: A1b not built")
    print("MAMBA-VIT EMB", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
