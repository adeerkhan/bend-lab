"""Gate B5: the can-fails — the gate must go red on each injected bug.

  reference/.venv/bin/python reference/test_mamba3_canfail.py

Three corruptions of examples/language/mamba3/mamba3_siso.bend, written
to .scratch/canfail/ at run time (never committed).  Each must build,
and each must produce a `y` that differs from the golden beyond the
FORWARD gate (1e-5) — i.e., the mutation would FAIL `make mamba3`,
not pass silently:

  noshift   the trapezoid's shifted term is dropped
            (`scale = gamma + shift` -> `scale = gamma`)
  ropeneg   the rope rotation's sign is wrong on the cos half
            (`y0 = x0*cos - x1*sin` -> `y0 = x0*cos + x1*sin`)
  zerod     the D skip path is zeroed at the load site
            (`ml1x4(to_list(dd))` -> `ml1x4(zline(4n))`)

The good program must still match the golden (it does; `make mamba3`
gates it).  A corruption that goes unnoticed is a FAIL.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "reference/mamba3_golden.npz")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
SCRATCH = os.path.join(ROOT, ".scratch/canfail")
SRC = os.path.join(ROOT, "examples/language/mamba3/mamba3_siso.bend")
FORWARD_GATE = 1e-5

CORRUPTIONS = {
    "noshift": [("      +scale_t = add_lists(gamma_t, shift_t)",
                 "      +scale_t = gamma_t", 2)],
    "ropeneg": [("      +y0 = ((x0 * c : F32) - (x1 * s : F32) : F32)",
                 "      +y0 = ((x0 * c : F32) + (x1 * s : F32) : F32)", 1)],
    "zerod":   [("ml1x4(TA.Mat.to_list(1n, 4n, dd)),",
                 "zline(4n),", 1)],
}


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


def rel(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.abs(a - b).max()) / max(1e-30, float(np.abs(b).max()))


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    env = dict(os.environ)
    env["BEND_NO_TELEMETRY"] = "1"
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")

    good = open(SRC).read()
    golden_y = np.load(GOLDEN)["y"].astype(np.float32).reshape(-1)
    ok = True

    for name, subs in CORRUPTIONS.items():
        bad = good
        for old, new, count in subs:
            found = bad.count(old)
            if found != count:
                print(f"FAIL {name}: source drifted, expected {count} of "
                      f"{old!r}, found {found}")
                ok = False
                continue
            bad = bad.replace(old, new)

        dst = os.path.join(SCRATCH, f"{name}.bend")
        with open(dst, "w") as f:
            f.write(bad)
        bin_path = os.path.join(SCRATCH, f"{name}.bin")
        b = subprocess.run([BEND, "-o", bin_path, dst], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        if b.returncode != 0:
            print((b.stdout or "") + (b.stderr or ""))
            print(f"FAIL {name}: the corrupted source does not build")
            ok = False
            continue
        r = subprocess.run([bin_path], cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=3600)
        tags = parse(r.stdout)
        if "y" not in tags:
            print(f"FAIL {name}: y not printed")
            ok = False
            continue
        d = rel(tags["y"], golden_y)
        detected = d > FORWARD_GATE
        print(f"{'ok  ' if detected else 'FAIL'} can-fail {name:8s} "
              f"y rel vs golden {d:.3e} (must exceed {FORWARD_GATE:.0e})")
        ok = ok and detected

    # the good program still matches the golden (sanity, cheap)
    dst = os.path.join(SCRATCH, "good.bend")
    with open(dst, "w") as f:
        f.write(good)
    bin_path = os.path.join(SCRATCH, "good.bin")
    if subprocess.run([BEND, "-o", bin_path, dst], cwd=ROOT, env=env,
                      capture_output=True, text=True).returncode == 0:
        r = subprocess.run([bin_path], cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=3600)
        d = rel(parse(r.stdout)["y"], golden_y)
        good_ok = d <= FORWARD_GATE
        print(f"{'ok  ' if good_ok else 'FAIL'} good            "
              f"y rel vs golden {d:.3e} (bound {FORWARD_GATE:.0e})")
        ok = ok and good_ok
    else:
        print("FAIL good: the unchanged source does not build")
        ok = False

    if not ok:
        print("MAMBA3 CANFAIL FAIL")
        sys.exit(1)
    print("MAMBA3 CANFAIL PASS")


if __name__ == "__main__":
    main()
