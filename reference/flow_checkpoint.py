"""The flow example's checkpoint: its sidecar, and the check a resume runs first.

  reference/.venv/bin/python reference/flow_checkpoint.py --write /tmp/flow.bin --step 1000
  reference/.venv/bin/python reference/flow_checkpoint.py --check /tmp/flow.json

`examples/diffusion/flow_matching/main.bend --save PATH` writes the training
state with `Train.save`: the parameters, Adam's first and second moments, then
`1 - b1^t` and `1 - b2^t`, little-endian F32. Those are the bytes `Train.dump`
lays out and `Train.restore` reads. This module writes the sidecar that travels
with the file — the model, the config, the packages, the weights' sha256 and the
step the run stopped at — and `--check` is what a resume runs before the first
step: a sidecar whose model, config, package or bend version differs is refused
by field name, so a resume never starts on the wrong shape (ISSUE-034, B9).

The example restores bytes; it does not read the sidecar. The caller does, and
refuses before launching the program.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checkpoints
import flow_config
from experiment import bend_version, git_state, packages

# what flow_matching/main.bend declares: three linear layers with SiLU, and the
# config its `cfg()` and `do_step` use
MODEL = {"layers": [3, flow_config.hidden(), flow_config.hidden(), 2], "act": "silu"}
CONFIG = {"optimizer": "adamw", "lr": 0.003, "b1": 0.9, "b2": 0.999,
          "eps": 1e-8, "wd": 0.0, "c1": 0.1, "c2": 0.001, "batch": 128}


def parameters(layers):
    """The weights and biases of a dense stack, the way T.Layer.count counts."""
    return sum(a * b + b for a, b in zip(layers, layers[1:]))


def floats():
    """The numbers Train.save writes for AdamW: p, m, v, and two powers."""
    return 3 * parameters(MODEL["layers"]) + 2


def sidecar_path(weights_path):
    root, ext = os.path.splitext(weights_path)
    return (root if ext == ".bin" else weights_path) + ".json"


def current(config=None, model=None):
    """This tree's identity, without weights: what a resume must match."""
    return checkpoints.identity_only(model or MODEL, config or CONFIG, packages(),
                                     bend_version(), git_state().get("commit"))


def write(weights_path, step):
    """Write the sidecar beside a weights file and return its path."""
    saved = checkpoints.identity(weights_path, MODEL, CONFIG, packages(),
                                 bend_version(), git_state().get("commit"))
    saved["tensors"] = [{"name": "train", "offset": 0,
                         "bytes": saved["weights"]["bytes"],
                         "sha256": saved["weights"]["sha256"]}]
    saved["step"] = step
    path = sidecar_path(weights_path)
    with open(path, "w") as f:
        json.dump(saved, f, indent=2)
        f.write("\n")
    return path


def refusal(saved, config=None, model=None):
    """The sentence a resume prints, or None when the checkpoint may be read."""
    return checkpoints.refuse(saved, current(config=config, model=model),
                              "the flow checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", metavar="BIN", help="a weights file Train.save wrote")
    parser.add_argument("--step", type=int, help="the step the run stopped at")
    parser.add_argument("--check", metavar="JSON", help="a sidecar to verify")
    args = parser.parse_args()
    if args.check:
        with open(args.check) as f:
            saved = json.load(f)
        sentence = refusal(saved)
        print(sentence if sentence else "the same model, config and packages")
        return 0 if sentence is None else 1
    if args.write:
        if args.step is None:
            raise SystemExit("--write needs --step, the step the run stopped at")
        path = write(args.write, args.step)
        print(f"wrote {path} for {os.path.basename(args.write)} at step {args.step}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
