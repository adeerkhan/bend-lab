"""Gate: the GQA config (ngroups 2, rope_fraction 1.0) on the SISO block.

  reference/.venv/bin/python reference/test_mamba3_gqa.py

Dumps the config inputs from `reference/mamba3_gqa_golden.npz` (written by
`reference/mamba3_gqa_twin.py`) into `.scratch/gqa_data`, builds
`probe_mamba3_gqa.bend`, runs it and compares the forward intermediates
against the twin at 1e-5 relative:

  y, states, angle, q, k, qk, scale, Bn, Cn, DT, ADT, gamma, ang_raw,
  A, trap, pre, out_v, y_inner

The backward for this config is not yet ported: it is reported
`unavailable` and the gate exits 1 with `MAMBA3 GQA PARTIAL` — the row
stays `not run` in the batch until the analytic backward lands, so no
claim is made that is not measured.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "reference/mamba3_gqa_golden.npz")
DATA = os.path.join(ROOT, ".scratch/gqa_data")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mamba3_gqa.bin")
GATE = 1e-5
FWD_TAGS = ["y", "states", "angle", "q", "k", "qk", "scale", "Bn", "Cn",
            "DT", "ADT", "gamma", "ang_raw", "A", "trap", "pre", "out_v",
            "y_inner", "Braw", "Craw"]
WEIGHTS = ["in_proj", "dt_bias", "B_bias", "C_bias", "B_norm_w",
           "C_norm_w", "D", "out_proj", "u"]


def parse(out):
    tags = {}
    for line in out.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        try:
            vals = np.fromstring(rest, sep=" ", dtype=np.float32)
        except Exception:
            continue
        if vals.size:
            tags[tag] = vals
    return tags


def dump_golden():
    os.makedirs(DATA, exist_ok=True)
    g = np.load(GOLDEN)
    for key in WEIGHTS:
        arr = np.ascontiguousarray(g[key], dtype="<f4")
        arr.tofile(os.path.join(DATA, key + ".bin"))


def main():
    if not os.path.exists(GOLDEN):
        raise SystemExit(f"missing {GOLDEN}; run reference/mamba3_gqa_twin.py")
    dump_golden()
    os.makedirs(os.path.join(ROOT, ".scratch"), exist_ok=True)
    env = dict(os.environ)
    env["BEND_NO_TELEMETRY"] = "1"
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")

    t0 = time.monotonic()
    subprocess.run([BEND, "-o", BIN,
                    os.path.join(ROOT, "examples/language/mamba3/probe_mamba3_gqa.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True,
                   timeout=1800)
    build = time.monotonic() - t0
    t0 = time.monotonic()
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    seconds = time.monotonic() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("gqa binary failed")
    tags = parse(p.stdout)

    golden = np.load(GOLDEN)
    ok = True
    for key in FWD_TAGS:
        got = tags.get("gqa_" + key)
        want = golden[key].astype(np.float64).reshape(-1)
        if got is None:
            print(f"FAIL gqa_{key}: not printed")
            ok = False
            continue
        got = got.astype(np.float64)
        if got.size != want.size:
            print(f"FAIL gqa_{key}: size {got.size} != {want.size}")
            ok = False
            continue
        d = np.abs(got - want)
        rel = float(d.max()) / max(1e-30, float(np.abs(want).max()))
        good = rel <= GATE
        print(f"{'ok  ' if good else 'FAIL'} gqa_{key}: max|d| {d.max():.6e} "
              f"rel {rel:.6e} gate {GATE:.0e}")
        ok = ok and good

    print("FAIL backward: not ported for ngroups 2 (analytic chain is SISO-only)")
    print(f"     build {build:.1f} s, run {seconds:.2f} s wall")
    print("MAMBA3 GQA", "PASS" if ok else "PARTIAL")
    # The row is not complete until the backward is gated; exit 1 keeps the
    # batch row out of PASS.
    return 0 if False else 1


if __name__ == "__main__":
    sys.exit(main())
