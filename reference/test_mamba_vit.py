"""MambaViT gate: the whole forward (A1a + A1b).

  reference/.venv/bin/python reference/test_mamba_vit.py

Requires the reference dump (`reference/mamba_vit_bins.py`, 130 files in
`.scratch/mamba_vit`). Builds `probe_mamba_vit_emb.bend`, runs it and
compares every intermediate at 1e-5 relative:

  mv_emb          (49,32)   the patch tokens after LN -> Linear -> LN  (A1a)
  mv_tokens       (50,32)   cls token prepended, pos embedding added   (A1a)
  mv_block0..7    (50,32)   each Mamba-1 block's mixer output          (A1b)
  mv_block0..7_res (50,32)  each block's accumulated residual          (A1b)
  mv_final        (50,32)   the final LayerNorm                        (A1b)
  mv_pooled       (32)      the mean over the 50 tokens                (A1b)
  mv_logits       (10)      the head                                   (A1b)

`MAMBA-VIT EMB PASS` covers the two embedding tags, `MAMBA-VIT FWD PASS`
everything; both lines are printed, so a reader sees which half failed.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(ROOT, ".scratch/mamba_vit")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mv_full.bin")
GATE = 1e-5
EMB = [("mv_emb", "emb"), ("mv_tokens", "tokens")]
FWD = ([("mv_block%d" % i, "block%d" % i) for i in range(8)]
       + [("mv_block%d_res" % i, "block%d_res" % i) for i in range(8)]
       + [("mv_final", "final"), ("mv_pooled", "pooled"), ("mv_logits", "logits")])


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
        raise SystemExit("mamba-vit-fwd binary failed")
    tags = {}
    for line in p.stdout.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        v = np.fromstring(rest, sep=" ", dtype=np.float32)
        if v.size:
            tags[tag] = v
    ok = True
    half = {}

    def compare(pairs, name):
        good = True
        for tag, key in pairs:
            got = tags.get(tag)
            want = np.fromfile(os.path.join(DUMP, key + ".bin"), dtype="<f4").astype(np.float64)
            if got is None:
                print(f"FAIL {tag}: not printed")
                good = False
                continue
            got = got.astype(np.float64)
            if got.size != want.size:
                print(f"FAIL {tag}: size {got.size} != {want.size}")
                good = False
                continue
            d = np.abs(got - want)
            rel = float(d.max()) / max(1e-30, float(np.abs(want).max()))
            one = rel <= GATE
            print(f"{'ok  ' if one else 'FAIL'} {tag}: max|d| {d.max():.6e} "
                  f"rel {rel:.6e} gate {GATE:.0e}")
            good = good and one
        half[name] = good
        return good

    ok = compare(EMB, "emb")
    ok = compare(FWD, "fwd") and ok
    print("MAMBA-VIT EMB", "PASS" if half["emb"] else "FAIL")
    print("MAMBA-VIT FWD", "PASS" if (half["emb"] and half["fwd"]) else "FAIL")
    return 0 if (ok and half["emb"] and half["fwd"]) else 1


if __name__ == "__main__":
    sys.exit(main())
