"""Gate: the GQA config (ngroups 2, rope_fraction 1.0) on the SISO block.

  reference/.venv/bin/python reference/test_mamba3_gqa.py

Dumps the config inputs from `reference/mamba3_gqa_golden.npz` (written by
`reference/mamba3_gqa_twin.py`) into `.scratch/gqa_data`, builds
`probe_mamba3_gqa.bend`, runs it and compares the forward intermediates
against the twin at 1e-5 relative:

  y, states, angle, q, k, qk, scale, Bn, Cn, DT, ADT, gamma, ang_raw,
  A, trap, pre, out_v, y_inner

and the analytic backward against the twin's autograd gradients at 1e-4
per tensor: du, dW_in, ddt_bias, dB_bias, dC_bias, dB_norm_w, dC_norm_w,
dD, dW_out, and the g_* intermediates (the angle gradient summed over
tokens, the SISO gate's convention).
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
           "C_norm_w", "D", "out_proj", "u", "dy"]
GRAD_TAGS = [("grad_du", "du"), ("grad_dW_in", "dW_in"),
             ("grad_ddt_bias", "ddt_bias"), ("grad_dB_bias", "dB_bias"),
             ("grad_dC_bias", "dC_bias"), ("grad_dB_norm_w", "dB_norm_w"),
             ("grad_dC_norm_w", "dC_norm_w"), ("grad_dD", "dD"),
             ("grad_dW_out", "dW_out"), ("grad_do_v", "g_out_v"),
             ("grad_dq", "g_q"), ("grad_dk", "g_k"), ("grad_dz", "g_z"),
             ("grad_dx", "g_x"), ("grad_dqk", "g_qk"),
             ("grad_dscale", "g_scale"), ("grad_dADT", "g_ADT"),
             ("grad_dBn", "g_Bn"), ("grad_dCn", "g_Cn"),
             ("grad_dDT", "g_DT"), ("grad_dtrap", "g_trap_raw"),
             ("grad_ddd_dt", "g_dd_dt"), ("grad_ddd_A", "g_dd_A"),
             ("grad_dBraw", "g_Braw"), ("grad_dCraw", "g_Craw")]
GRAD_GATE = 1e-4


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

    for tag, key in GRAD_TAGS:
        got = tags.get(tag)
        want = golden[key].astype(np.float64).reshape(-1)
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
        good = rel <= GRAD_GATE
        print(f"{'ok  ' if good else 'FAIL'} {tag}: max|d| {d.max():.6e} "
              f"rel {rel:.6e} gate {GRAD_GATE:.0e}")
        ok = ok and good

    # The port's dang_raw is the per-token raw-angle gradient summed over
    # tokens, the same convention the SISO gate uses.
    got = tags.get("grad_dang_raw")
    want = golden["g_ang_raw"].astype(np.float64).reshape(8, 8).sum(0)
    if got is None:
        print("FAIL grad_dang_raw: not printed")
        ok = False
    else:
        got = got.astype(np.float64)
        if got.size != want.size:
            print(f"FAIL grad_dang_raw: size {got.size} != {want.size}")
            ok = False
        else:
            d = np.abs(got - want)
            rel = float(d.max()) / max(1e-30, float(np.abs(want).max()))
            good = rel <= GRAD_GATE
            print(f"{'ok  ' if good else 'FAIL'} grad_dang_raw: max|d| "
                  f"{d.max():.6e} rel {rel:.6e} gate {GRAD_GATE:.0e}")
            ok = ok and good

    print(f"     build {build:.1f} s, run {seconds:.2f} s wall")
    print("MAMBA3 GQA", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
