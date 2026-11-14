"""Gate: a depth-4 stack (four independent blocks, a residual around each).

  reference/.venv/bin/python reference/test_mamba3_deep.py

Dumps the twin's inputs from `reference/mamba3_deep_golden.npz`, builds
`probe_mamba3_deep.bend`, runs it and compares the forward at 1e-5:

  deep_logits, deep_loss, deep_hn

and the ten-step training run and the 16-token greedy generation against
the twin's own run:

  deep4_losses (10 losses)  1e-4 relative
  deep4_gen    (16 tokens)  exactly equal

Depths 8 and 16 are `unavailable`: the port has the 4-layer stack only, and
a deeper record has not been built (the lane is not the limit; the code is).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "reference/mamba3_deep_golden.npz")
DATA = os.path.join(ROOT, ".scratch/deep_data")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mamba3_deep.bin")
GATE = 1e-5
PAIRS = [("deep_logits", "logits"), ("deep_loss", "loss"), ("deep_hn", "hn")]
LOSS_GATE = 1e-4


def main():
    if not os.path.exists(GOLDEN):
        raise SystemExit(f"missing {GOLDEN}; run reference/mamba3_deep_twin.py")
    os.makedirs(DATA, exist_ok=True)
    g = np.load(GOLDEN)
    for key in g.files:
        if key in ("h", "hn", "logits", "loss", "layers"):
            continue
        np.ascontiguousarray(g[key], dtype="<f4").tofile(
            os.path.join(DATA, key + ".bin"))
    env = dict(os.environ)
    env["BEND_NO_TELEMETRY"] = "1"
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")
    t0 = time.monotonic()
    subprocess.run([BEND, "-o", BIN,
                    os.path.join(ROOT, "examples/language/mamba3/probe_mamba3_deep.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True,
                   timeout=1800)
    build = time.monotonic() - t0
    t0 = time.monotonic()
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    seconds = time.monotonic() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("deep binary failed")
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
        want = g[key].astype(np.float64).reshape(-1)
        if got is None:
            print(f"FAIL {tag}: not printed")
            ok = False
            continue
        got = got.astype(np.float64)
        d = np.abs(got - want)
        rel = float(d.max()) / max(1e-30, float(np.abs(want).max()))
        good = rel <= GATE
        print(f"{'ok  ' if good else 'FAIL'} {tag}: max|d| {d.max():.6e} "
              f"rel {rel:.6e} gate {GATE:.0e}")
        ok = ok and good
    got = tags.get("deep4_losses")
    want = g["train_loss"].astype(np.float64).reshape(-1)
    if got is None:
        print("FAIL deep4_losses: not printed")
        ok = False
    else:
        got = got.astype(np.float64)
        d = np.abs(got - want)
        rel = float(d.max()) / max(1e-30, float(np.abs(want).max()))
        good = rel <= LOSS_GATE
        print(f"{'ok  ' if good else 'FAIL'} deep4_losses: max|d| {d.max():.6e} "
              f"rel {rel:.6e} gate {LOSS_GATE:.0e}")
        ok = ok and good

    got = tags.get("deep4_gen")
    want = g["gen_tokens"].astype(int).reshape(-1)
    if got is None:
        print("FAIL deep4_gen: not printed")
        ok = False
    else:
        got = got.astype(int)
        eq = got.size == want.size and np.array_equal(got, want)
        print(f"{'ok  ' if eq else 'FAIL'} deep4_gen: {got.tolist()}")
        if not eq:
            print(f"     twin: {want.tolist()}")
        ok = ok and eq

    print("unavailable depths 8/16: not built (the port has the 4-layer stack)")
    print(f"     build {build:.1f} s, run {seconds:.2f} s wall")
    print("MAMBA3 DEEP", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
