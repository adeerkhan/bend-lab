"""The deterministic inputs for the four-head attention backward gate.

  reference/.venv/bin/python reference/glm53_attn_bwd_data.py --write

x, qkv and out are the same values `glm53_attn_ref.py` and `test_glm53_mh.py`
use (seed 71); dout (seed 72) is the gradient of the layer output.  dout_bad
is dout with one element moved by +0.5: the can-fail case must be rejected by
the same comparison the correct dout passes.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "examples/language/glm53_flash/data/ref8")
N, D, H = 8, 32, 4


def inputs():
    rng = np.random.default_rng(71)
    x = (rng.standard_normal((1, N, D)) * 0.5).astype(np.float32)
    qkv = (rng.standard_normal((3 * D, D)) * 0.05).astype(np.float32)
    out = (rng.standard_normal((D, D)) * 0.05).astype(np.float32)
    dout = (np.random.default_rng(72).standard_normal((1, N, D)) * 0.3).astype(np.float32)
    bad = dout.copy()
    bad[0, 0, 0] += 0.5
    return {"x": x, "qkv": qkv, "out": out, "dout": dout, "dout_bad": bad}


def write(directory=DATA):
    os.makedirs(directory, exist_ok=True)
    w = inputs()
    for name, value in w.items():
        value.astype("<f4").tofile(os.path.join(directory, name + ".bin"))
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if not args.write:
        ap.error("--write is required")
    w = write()
    print(f"wrote {len(w)} files in {DATA}")


if __name__ == "__main__":
    main()
