"""The ViT checkpoint resumes the curve, and a zero-step draw says unavailable.

  reference/.venv/bin/python reference/test_vit_checkpoint.py

Builds `examples/diffusion/vit/main.bend` once and runs it with `--no-samples`:

  A  three steps from the initial weights
  B  two steps, then `--save`
  C  `--resume` B's file at step 2, one more step

C must print A's `final_loss` exactly: `Train.save` writes the parameters, both
AdamW moments and `1 - b1^t`, `1 - b2^t`, and `Train.restore` reads them back.
A `--steps 0` draw from the same file must print `final_loss: unavailable`, not
a zero. The file is `3 * 100736 + 2` floats.

The dataset is the example's own cache (`mstz/optdigits`, pinned by hash), so
the first run fetches it and later runs are warm.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

PROGRAM = os.path.join(ROOT, "examples/diffusion/vit/main.bend")
STEPS = 3
PART = 2
PARAMETERS = 100736
FLOATS = 3 * PARAMETERS + 2


def build():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-vit-checkpoint")
    out = subprocess.run([BEND, "-o", binary, PROGRAM], capture_output=True,
                         text=True, env=ENV, cwd=ROOT, timeout=1800)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + out.stdout + out.stderr)
    return binary


def run(binary, *args):
    out = subprocess.run([binary, *args], capture_output=True, text=True,
                         env=ENV, cwd=ROOT, timeout=1800)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    return out.stdout


def final(text):
    found = re.findall(r"^final_loss: ([0-9.eE+-]+|unavailable.*)$", text, re.M)
    if not found:
        raise SystemExit("no final_loss line in:\n" + text)
    return found[-1]


def main():
    binary = build()
    tmp = tempfile.mkdtemp()
    weights = os.path.join(tmp, "vit.bin")

    single = run(binary, "--steps", str(STEPS), "--no-samples")
    saved = run(binary, "--steps", str(PART), "--save", weights, "--no-samples")
    resumed = run(binary, "--steps", str(STEPS - PART), "--start", str(PART),
                  "--resume", weights, "--no-samples")
    drawn = run(binary, "--steps", "0", "--resume", weights, "--no-samples")

    expected, got = final(single), final(resumed)
    size = os.path.getsize(weights)
    ok = True
    if expected == got:
        print(f"ok   the resumed final loss is {got}, as the single run")
    else:
        print(f"FAIL the resumed final loss is {got}, the single run {expected}")
        ok = False
    if size == 4 * FLOATS:
        print(f"ok   the checkpoint is {size} bytes: 3 x {PARAMETERS} + 2 floats")
    else:
        print(f"FAIL the checkpoint is {size} bytes, expected {4 * FLOATS}")
        ok = False
    if final(drawn).startswith("unavailable"):
        print("ok   a zero-step draw prints final_loss: unavailable")
    else:
        print(f"FAIL a zero-step draw printed {final(drawn)}")
        ok = False
    if final(saved) == "0.0":
        print("FAIL the training run printed a zero final loss")
        ok = False
    print("VIT CHECKPOINT PASS" if ok else "VIT CHECKPOINT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
