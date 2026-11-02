"""A checkpoint resumes the curve, and a moved config is refused before the step.

  reference/.venv/bin/python reference/test_flow_checkpoint.py

Builds `examples/diffusion/flow_matching/main.bend` once and runs it three
times with `--no-sample`:

  A  200 steps from the initial weights
  B  100 steps, then `--save`
  C  `--resume` B's file at step 100, 100 more steps

B + C must print exactly A's 200 numbers: BendTorch's `Train.save` writes the
parameters, both Adam moments and `1 - b1^t`, `1 - b2^t`, and `Train.restore`
reads them back, so a run stopped and resumed computes the same bits as one
that never stopped. Then the sidecar `reference/flow_checkpoint.py` wrote is
checked against the file, and a resume whose lr or model moved is refused by
field name — the refusal happens before the program is launched.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checkpoints
import flow_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

PROGRAM = os.path.join(ROOT, "examples/diffusion/flow_matching/main.bend")
STEPS = 200
PART = 100


def build():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-flow-checkpoint")
    out = subprocess.run([BEND, "-o", binary, PROGRAM], capture_output=True, text=True,
                         env=ENV, cwd=ROOT, timeout=1800)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + out.stdout + out.stderr)
    return binary


def run(binary, *args):
    out = subprocess.run([binary, *args], capture_output=True, text=True, env=ENV,
                         cwd=ROOT, timeout=1800)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    return out.stdout


def curve(text):
    """The printed losses, as text: loss after each step, final_loss last."""
    return re.findall(r"^(?:loss|final_loss): ([0-9.eE+-]+)$", text, re.M)


def main():
    binary = build()
    tmp = tempfile.mkdtemp()
    weights = os.path.join(tmp, "flow.bin")

    single = curve(run(binary, "--steps", str(STEPS), "--no-sample"))
    first = curve(run(binary, "--steps", str(PART), "--no-sample", "--save", weights))
    sidecar = flow_checkpoint.write(weights, PART)
    rest = curve(run(binary, "--steps", str(STEPS - PART), "--start", str(PART),
                     "--resume", weights, "--no-sample"))

    fail = 0
    if len(single) != STEPS or len(first) + len(rest) != STEPS:
        print(f"ERROR the runs printed {len(single)} and {len(first)}+{len(rest)} losses, "
              f"not {STEPS}")
        fail += 1
    elif first + rest == single:
        print(f"ok   the resumed run continues the curve: {PART} + {STEPS - PART} = {STEPS} "
              "values, bit-identical")
    else:
        at = next(i for i, (a, b) in enumerate(zip(first + rest, single)) if a != b)
        print(f"ERROR the resumed curve differs at step {at}: "
              f"{(first + rest)[at]} vs {single[at]}")
        fail += 1
    if rest:
        print(f"     step {PART}: the stopped run ended {first[-1]}, the resumed one read "
              f"{rest[0]}; the single run printed {single[PART]}")

    size = os.path.getsize(weights)
    expected = 4 * flow_checkpoint.floats()
    if size == expected:
        print(f"ok   the weights are {size} bytes: 3 x {flow_checkpoint.parameters(flow_checkpoint.MODEL['layers'])} "
              "parameters + 2, float32")
    else:
        print(f"ERROR the weights are {size} bytes, not {expected}")
        fail += 1

    saved = json.load(open(sidecar))
    if (saved.get("step") == PART
            and saved["weights"]["sha256"] == checkpoints.sha256_file(weights)
            and saved["weights"]["bytes"] == size
            and saved["tensors"] == [{"name": "train", "offset": 0, "bytes": size,
                                      "sha256": saved["weights"]["sha256"]}]):
        print(f"ok   the sidecar names the file: step {PART}, "
              f"sha256 {saved['weights']['sha256'][:12]}..., one train block")
    else:
        print(f"ERROR the sidecar does not describe the weights: {saved}")
        fail += 1

    if flow_checkpoint.refusal(saved) is None:
        print("ok   the same model and config pass the resume check")
    else:
        print(f"ERROR the same model was refused: {flow_checkpoint.refusal(saved)}")
        fail += 1

    moved = flow_checkpoint.refusal(saved, config=dict(flow_checkpoint.CONFIG, lr=0.004))
    if moved and "config.lr" in moved and "0.003" in moved and "0.004" in moved:
        print(f"ok   a moved lr is refused by name: {moved.splitlines()[1].strip()}")
    else:
        print(f"ERROR a moved lr was not refused by name: {moved}")
        fail += 1

    other = flow_checkpoint.refusal(
        saved, model={"layers": [3, 64, 128, 2], "act": "silu"})
    if other and "model.declared" in other:
        print("ok   a moved model is refused by name: model.declared")
    else:
        print(f"ERROR a moved model was not refused by name: {other}")
        fail += 1

    print(f"{fail} checks differ" if fail else "the checkpoint resumes and the identity refuses")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
