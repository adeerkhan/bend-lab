"""Gate B2: the exact-step decoder's state round-trip.

  reference/.venv/bin/python reference/test_mamba3_state.py

Builds `examples/language/mamba3/probe_mamba3_state.bend`, which decodes
the copy-task LM twice:

  full:  30 steps from a fresh state
  split: 10 steps from a fresh state, then 20 steps resumed from the
         state the first leg returned (t continues at 10, not 0)

Checks, exactly:

  tokens_full == tokens_split                (token for token)
  tokens_full[:16] == reference/mamba3_golden.npz["gen_tokens"]

The first 16 tokens are the existing 16-token golden of `step_mamba3.bend`;
the probe's driver is a sibling of it (shared `S.layer_row`, its own
uncapped 30-token append). 30 steps, not 16, is the point: the golden
knows nothing about the resumed leg.
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
BIN = os.path.join(ROOT, ".scratch/mamba3_state.bin")
SRC = os.path.join(ROOT, "examples/language/mamba3/probe_mamba3_state.bend")
STEPS = 30


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

    t0 = time.monotonic()
    subprocess.run([BEND, "-o", BIN, SRC], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=3600)
    print(p.stdout, end="")
    tags = parse(p.stdout)
    full = tags.get("tokens_full")
    split = tags.get("tokens_split")
    ok = True
    if full is None or split is None:
        print("FAIL tokens_full or tokens_split not printed")
        print("MAMBA3 STATE FAIL")
        sys.exit(1)

    if full.shape != split.shape:
        print(f"FAIL lengths differ: full {full.shape}, split {split.shape}")
        ok = False
    elif not np.array_equal(full, split):
        d = np.abs(full - split)
        print(f"FAIL trajectories differ at {np.nonzero(d > 0)[0].tolist()}")
        ok = False
    else:
        print(f"tokens_full == tokens_split ({full.shape[0]} tokens)")

    golden = np.load(GOLDEN)["gen_tokens"].astype(np.float32).reshape(-1)
    head = full[:golden.shape[0]]
    if not np.array_equal(head, golden):
        print(f"FAIL first {golden.shape[0]} tokens {head.astype(int).tolist()} "
              f"!= gen_tokens golden {golden.astype(int).tolist()}")
        ok = False
    else:
        print(f"first {golden.shape[0]} tokens == gen_tokens golden")

    if not ok:
        print("MAMBA3 STATE FAIL")
        sys.exit(1)
    print(f"elapsed {time.monotonic() - t0:.1f}s")
    print("MAMBA3 STATE PASS")


if __name__ == "__main__":
    main()
