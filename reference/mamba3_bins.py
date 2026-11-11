"""Write the golden npz as flat little-endian F32 bins for the Bend probes.

  python reference/mamba3_bins.py reference/mamba3_golden.npz reference/mamba3_data

NumPy only: the committed npz is enough, no torch is needed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    golden, out_dir = sys.argv[1], sys.argv[2]
    z = np.load(golden)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for key in z.files:
        a = z[key].astype(np.float32)
        a.tofile(os.path.join(out_dir, key + ".bin"))
        manifest[key] = list(a.shape)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print(f"wrote {len(manifest)} bins to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
