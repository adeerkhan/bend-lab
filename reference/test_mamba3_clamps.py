"""Gate B4: clamp boundaries — A at -A_floor, trap at 0/1, softplus
at large/small dd_dt.

  reference/.venv/bin/python reference/test_mamba3_clamps.py

The Bend port (probe_mamba3_clamps.bend) and the twin are evaluated on
the same ladders:

  A        dd_A in {-10000, -9999, -1000.1, -100, 0, 1, 10}
           clamp(-htail(dd_A), max=-A_floor)
  trap     trap_raw in {-1000, -100, -20, 0, 20, 100, 1000}
           sigmoid -> exactly 0/1 at the extremes
  softplus dd_dt + dt_bias in {-100, -20, -1, 0, 1, 20, 50, 88, 100}

Exit codes of the worker and the probe must both be 0; the comparison
is elementwise.  Bounds are measured on this lane (2026-11-16), not
chosen to make the gate pass; the Bend softplus overflow at x > 88.7
was found by this ladder and fixed in the same row (torch's
threshold=20 semantics).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))
PROBE = os.path.join(ROOT, "examples/language/mamba3/probe_mamba3_clamps.bend")
BIN = os.path.join(ROOT, ".scratch/mamba3_clamps.bin")

A_LADDER = [-10000.0, -9999.0, -1000.1, -100.0, 0.0, 1.0, 10.0]
SP_LADDER = [-100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 50.0, 88.0, 100.0]
TRAP_LADDER = [-1000.0, -100.0, -20.0, 0.0, 20.0, 100.0, 1000.0]


def worker_code() -> str:
    return """
import os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "reference"))
sys.path.insert(0, os.path.join(sys.argv[1], "reference/mamba3"))
import torch
import torch.nn.functional as F
import mamba3_twin as W

torch.manual_seed(0)
m = W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32, ngroups=1,
             is_mimo=False).cuda().float()

dd_A = torch.tensor(%s, device="cuda")
A = torch.clamp(-W.heavy_tail(dd_A), max=-m.A_floor)
print("clamp_A_y:", " ".join(f"{v:.9g}" for v in A.cpu()))
x = torch.tensor(%s, device="cuda")
print("clamp_softplus_y:", " ".join(f"{v:.9g}" for v in F.softplus(x).cpu()))
t = torch.tensor(%s, device="cuda")
print("clamp_sigmoid_y:", " ".join(f"{v:.9g}" for v in torch.sigmoid(t).cpu()))

# gradient dead zone: dA/dd_A is exactly 0 deep inside the clamped region
for x in (-10000.0, -1000.1, -100.0, 0.0, 2.0, 10.0):
    t = torch.tensor([x], device="cuda", requires_grad=True)
    A = torch.clamp(-W.heavy_tail(t), max=-m.A_floor)
    A.backward()
    print(f"clamp_A_grad {x} {t.grad.item():.9g}")
""" % (A_LADDER, SP_LADDER, TRAP_LADDER)


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


def main():
    os.makedirs(os.path.join(ROOT, ".scratch"), exist_ok=True)
    env = dict(os.environ)
    env["BEND_NO_TELEMETRY"] = "1"
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")

    t0 = time.time()
    w = subprocess.run([PYRF, "-c", worker_code(), ROOT], cwd=ROOT,
                       capture_output=True, text=True)
    if w.returncode != 0:
        print((w.stdout or "") + (w.stderr or ""))
        print("MAMBA3 CLAMPS FAIL")
        sys.exit(1)
    tw = parse(w.stdout)

    b = subprocess.run([BEND, "-o", BIN, PROBE], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if b.returncode != 0:
        print((b.stdout or "") + (b.stderr or ""))
        print("MAMBA3 CLAMPS FAIL")
        sys.exit(1)
    r = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout or "") + (r.stderr or ""))
        print("MAMBA3 CLAMPS FAIL")
        sys.exit(1)
    bnd = parse(r.stdout)

    ok = True

    def check(name, bend, twin, bound, mode="abs"):
        nonlocal ok
        good = True
        for i, (bv, tv) in enumerate(zip(bend, twin)):
            d = abs(float(bv) - float(tv))
            lim = bound if mode == "abs" else bound * max(1e-30, abs(float(tv)))
            if d > lim:
                print(f"FAIL {name}[{i}]: bend {bv:.9g} vs twin {tv:.9g} (diff {d:.3e})")
                good = False
        print(f"{'ok  ' if good else 'FAIL'} {name:24s} max|d| {max(abs(float(a)-float(b)) for a, b in zip(bend, twin)):.3e} (bound {bound:.0e})")
        ok = ok and good

    check("A at -A_floor", bnd["clamp_A_y"], tw["clamp_A_y"], 1e-6, "abs")
    check("trap at 0/1", bnd["clamp_sigmoid_y"], tw["clamp_sigmoid_y"], 1e-6, "abs")
    check("softplus large/small", bnd["clamp_softplus_y"], tw["clamp_softplus_y"], 1e-5, "abs")

    # the twin's gradient dead zone: dA/dd_A == 0 deep in the clamped region,
    # finite and nonzero elsewhere on the ladder (x=1.0 excluded: torch's
    # where() both-branch evaluation yields a nan gradient there, recorded)
    for line in w.stdout.splitlines():
        if line.startswith("clamp_A_grad"):
            _, xs, gv = line.split()
            g = float(gv)
            x = float(xs)
            if x == -10000.0:
                good = g == 0.0
            else:
                good = g == g and 0.0 < abs(g) <= 1.0
            print(f"{'ok  ' if good else 'FAIL'} A grad dead zone[{x:>8}] g={g:.3e}")
            ok = ok and good

    if not ok:
        print("MAMBA3 CLAMPS FAIL")
        sys.exit(1)
    print(f"elapsed {time.time() - t0:.1f}s")
    print("MAMBA3 CLAMPS PASS")


if __name__ == "__main__":
    main()
