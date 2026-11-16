"""Gate A1: the Mamba-3 numerical lanes.

  reference/.venv/bin/python reference/test_mamba3_lanes.py

Dumps the SISO block (T = 1, 2, 4, 8, 16, 64) through four lanes and
asserts the per-lane bounds recorded in `docs/mamba3-lanes.md`:

  IEEE   1e-05   measured worst 2.6e-06 (states, T=64)
  TF32   5e-03   measured worst 2.4e-03 (dAngles, T=16)
  bf16   5e-02   measured worst 2.4e-02 (dB_norm_w, T=8)

The bounds are per-lane tripwires, not an oracle: the F32 twin is the
oracle, the bf16 lane is a lane.  The ieee/tf32 split needs two
processes (TRITON_F32_DEFAULT is read at Triton import), and the torch
build with CUDA lives in `.venv-rf`, which the default gate runner
(`.venv`) does not have, so everything here is a subprocess.

TRITON_F32_DEFAULT is set explicitly for both kernel lanes; the twin
imports cleanly with it unset because mamba3_twin.py only setdefaults it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, ".scratch", "lanes")
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))
LANES = os.path.join(ROOT, "reference", "mamba3_lanes.py")


def run(label, args, env_extra=None):
    env = dict(os.environ)
    env.pop("TRITON_F32_DEFAULT", None)
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    p = subprocess.run([PYRF, LANES] + args, cwd=ROOT, env=env,
                       capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(f"[{label}] exit {p.returncode} in {time.time() - t0:.1f}s", flush=True)
    if p.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-12:])
        print(tail, flush=True)
        print("MAMBA3 LANES FAIL", flush=True)
        sys.exit(1)
    return out


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    for f in os.listdir(SCRATCH):
        if f.endswith(".npz"):
            os.remove(os.path.join(SCRATCH, f))
    run("twin", ["twin"])
    run("kernel ieee", ["kernel"], {"TRITON_F32_DEFAULT": "ieee"})
    run("kernel tf32", ["kernel"], {"TRITON_F32_DEFAULT": "tf32"})
    run("module bf16", ["bf16"])
    out = run("gate", ["gate"])
    for line in out.splitlines():
        if line.startswith(("ok", "FAIL", "MAMBA3")):
            print(line, flush=True)


if __name__ == "__main__":
    main()
