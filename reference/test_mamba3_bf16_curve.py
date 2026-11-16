"""Gate A2: the tiny-LM training curve in two lanes.

  reference/.venv/bin/python reference/test_mamba3_bf16_curve.py

Regenerates the 20-step curve twice (same data, seeds, schedule):

  f32   the twin path on CUDA (explicit t-by-t loop, F32 everywhere)
  bf16  the public Mamba3 module on CUDA (internally bf16)

and asserts:

  f32 matches the recorded CUDA F32 curve (2026-11-16, the run this
  gate was measured from) at 1e-3 — pure torch F32, deterministic;
  measured agreement with the CPU F32 record is 4.8e-07.
  bf16 tracks f32: max |dLoss| <= 5e-04 (measured 5.7e-05, step 19),
  and both lanes land the same train hits (512/2540, measured).

The bf16 lane is a lane, not the oracle, and this gate does not promote
it; it is a tripwire plus a reproducibility check.  A lane, device or
version change is a banner, never a delta.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, ".scratch/mamba3_tiny")
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))
TOOL = os.path.join(ROOT, "reference", "mamba3_tiny_lanes.py")

GOLDEN_F32 = [4.838168144226074, 4.849788188934326, 4.797366142272949,
              4.769842624664307, 4.775183200836182, 4.7134108543396,
              4.679995536804199, 4.607712745666504, 4.581757068634033,
              4.567843437194824, 4.46253776550293, 4.4447922706604,
              4.45286226272583, 4.464232444763184, 4.469738960266113,
              4.318073749542236, 4.344018936157227, 4.3558878898620605,
              4.370820999145508, 4.333054065704346]
F32_GATE = 1e-3      # measured 4.8e-07 (vs CPU) / deterministic per build
BF16_GATE = 5e-04    # measured 5.7e-05 (max |dLoss|, step 19)


def main():
    if not os.path.isdir(os.path.join(OUT, "w_init")):
        print("FAIL w_init missing: run reference/mamba3_tiny_twin.py --mode dump first")
        print("MAMBA3 BF16 CURVE FAIL")
        sys.exit(1)
    t0 = time.time()
    p = subprocess.run([PYRF, TOOL, "--lane", "both"], cwd=ROOT,
                       capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print("\n".join(out.strip().splitlines()[-3:]), flush=True)
    print(f"[lanes] exit {p.returncode} in {time.time() - t0:.1f}s", flush=True)
    if p.returncode != 0:
        print("\n".join(out.strip().splitlines()[-12:]), flush=True)
        print("MAMBA3 BF16 CURVE FAIL")
        sys.exit(1)

    f32 = json.load(open(os.path.join(OUT, "lane_f32.json")))
    bf16 = json.load(open(os.path.join(OUT, "lane_bf16.json")))
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + name)
        ok = ok and bool(cond)

    dl_gold = max(abs(a - b) for a, b in zip(f32["losses"], GOLDEN_F32))
    check(f"f32 curve vs recorded  max|d|={dl_gold:.2e} (bound {F32_GATE:.0e})",
          dl_gold <= F32_GATE and len(f32["losses"]) == 20)
    dl_lane = max(abs(a - b) for a, b in zip(f32["losses"], bf16["losses"]))
    check(f"bf16 vs f32            max|dLoss|={dl_lane:.2e} (bound {BF16_GATE:.0e})",
          dl_lane <= BF16_GATE)
    check("both lanes finite",
          all(map(lambda x: x == x and abs(x) < 1e30, f32["losses"] + bf16["losses"]))
          and all(map(lambda x: x == x and abs(x) < 1e30, f32["grad_norms"] + bf16["grad_norms"])))
    check(f"train hits equal       f32 {f32['train_hits']}  bf16 {bf16['train_hits']}",
          f32["train_hits"] == bf16["train_hits"])

    if not ok:
        print("MAMBA3 BF16 CURVE FAIL")
        sys.exit(1)
    print("MAMBA3 BF16 CURVE PASS")


if __name__ == "__main__":
    main()
