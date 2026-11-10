"""The hybrid's flat weight layout, the one `T.Params.save` writes.

  reference/glm53_hybrid_data.py          # the shapes, for a reader
  reference/glm53_hybrid_data.py --check results/glm53-hybrid128/run/final_weights.bin

`T.Params.save` writes the model's parameters in dump order; for the four-block
hybrid that is the embedding, then each block's RMSNorm, qkv, out, second
RMSNorm, gate, up and down, then the final RMSNorm. The dense Stage A layout in
`glm53_dense_data.py` is different (separate wq/wk/wv/wo), so the evaluator
needs this one to split a hybrid checkpoint into the per-tensor files
`weights_h.bend` reads.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

V, DIM, HID, LAYERS = 260, 32, 128, 4


def shapes():
    out = [("emb", (V, DIM))]
    for i in range(LAYERS):
        out += [
            (f"b{i}/ln1", (DIM,)),
            (f"b{i}/qkv", (3 * DIM, DIM)),
            (f"b{i}/out", (DIM, DIM)),
            (f"b{i}/ln2", (DIM,)),
            (f"b{i}/gate", (HID, DIM)),
            (f"b{i}/up", (HID, DIM)),
            (f"b{i}/down", (DIM, HID)),
        ]
    return out + [("lnf", (DIM,))]


def count():
    return sum(int(np.prod(shape)) for _, shape in shapes())


def unflatten(flat):
    out, at = {}, 0
    for name, shape in shapes():
        n = int(np.prod(shape))
        if at + n > len(flat):
            raise SystemExit(f"the flat file has {len(flat)} values, {name} needs {at + n}")
        out[name] = np.asarray(flat[at:at + n], dtype=np.float32).reshape(shape).copy()
        at += n
    if at != len(flat):
        raise SystemExit(f"the flat file has {len(flat)} values, the layout {at}")
    return out


def write_flat_weights(flat, directory):
    """Split a `T.Params.save` file into the per-tensor files the loader reads."""
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    weights = unflatten(flat)
    for name, _ in shapes():
        path = os.path.join(directory, name + ".bin")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(weights[name].astype("<f4").tobytes())
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flat", nargs="?", help="a T.Params.save file to check")
    parser.add_argument("--out", help="write the per-tensor files here")
    args = parser.parse_args()
    print(f"{len(shapes())} tensors, {count()} parameters")
    if args.flat:
        flat = np.fromfile(args.flat, dtype="<f4")
        weights = unflatten(flat)
        print(f"{args.flat}: {len(flat)} values, the layout matches")
        if args.out:
            write_flat_weights(flat, args.out)
            print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
