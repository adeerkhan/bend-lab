"""Gate: the SISO copy block wired into BendTorch `T.Layer`/`T.Train`
reproduces the custom ten-step trainer.

  reference/.venv/bin/python reference/test_mamba3_layer.py

Builds examples/language/mamba3/mamba3_layer.bend, runs it once and checks,
against the custom ten-step trainer looped inside the same binary (the two
sides see the same weights):

  layer_tokens == train_tokens (exactly)
  |layer_loss  - train_loss  | <= 1e-3 relative at the final step
  weight heads (w_in, E, Wh, w_out): relative distance <= 1e-5

The loss lists are compared at the final step only: Bend satisfies an
`A & B` tuple via a reachability-based schedule, so after training the two
paths can diverge by F32 rounding noise (measured: final loss 2.1401443 vs
2.1404066, weights at 1e-6 relative). The tokens are compared exactly.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mamba3_layer.bin")
LOSS_GATE = 1e-3
WEIGHT_GATE = 1e-5


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

    subprocess.run([BEND, "-o", BIN,
                    os.path.join(ROOT, "examples/language/mamba3/mamba3_layer.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True,
                   timeout=1800)
    t0 = time.monotonic()
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    seconds = time.monotonic() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("layer binary failed")
    tags = parse(p.stdout)

    ok = True
    train_tokens = tags.get("train_tokens")
    layer_tokens = tags.get("layer_tokens")
    if train_tokens is None or layer_tokens is None:
        raise SystemExit("missing train_tokens or layer_tokens in output")
    same_tokens = np.array_equal(train_tokens.astype(int),
                                 layer_tokens.astype(int))
    print(f"{'ok  ' if same_tokens else 'FAIL'} layer_tokens: "
          f"{layer_tokens.astype(int).tolist()}")
    if not same_tokens:
        print(f"     train_tokens: {train_tokens.astype(int).tolist()}")
    ok = ok and same_tokens

    tl = tags["train_loss"].astype(np.float64)
    ll = tags["layer_loss"].astype(np.float64)
    last = float(tl[-1])
    l_last = float(ll[-1])
    rel = abs(l_last - last) / max(1e-30, abs(l_last))
    tok = rel <= LOSS_GATE
    print(f"{'ok  ' if tok else 'FAIL'} layer final loss: bend-layer {l_last:.7f} "
          f"vs custom {last:.7f} rel {rel:.2e} gate {LOSS_GATE:.0e}")
    ok = ok and tok

    for key in ["w_in", "E", "Wh", "w_out"]:
        c = tags.get(f"custom_{key}")
        l = tags.get(f"layer_{key}")
        if c is None or l is None:
            print(f"FAIL {key}: not printed")
            ok = False
            continue
        c = c.astype(np.float64)
        l = l.astype(np.float64)
        rel = float(np.abs(c - l).max() / max(1e-30, np.abs(c).max()))
        good = rel <= WEIGHT_GATE
        print(f"{'ok  ' if good else 'FAIL'} {key} head: max rel {rel:.2e} "
              f"gate {WEIGHT_GATE:.0e}")
        ok = ok and good

    print(f"     train probe: {seconds:.2f} s wall")
    print("MAMBA3 LAYER", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
