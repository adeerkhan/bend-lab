"""cifar-train: the same model trained so its loss can fall, against its twin.

  reference/.venv-lucidrains/bin/python reference/test_cifar_train.py

`examples/cifar_train/main.bend` is `examples/cifar/main.bend`'s denoiser over
the same 256 CIFAR-10 images, with a level drawn from {16, 32, 48} each step,
lr 1e-3 and 600 steps: the two things ISSUE-036 says the 40-step plumbing run
lacks (a level is visited once there, so its loss cannot move).

The twin is `reference/test_cifar.py`'s: the same weights, the same counter
noise, the same schedule and the same AdamW, with `train(levels=..., lr=...)`.
It compares one gated number, `loss_drop` = the first loss over the last, and
reports `final_loss`: 600 steps is a training run at plumbing scale, not a
converged model, so the absolute loss is `report` mode.

Writes results/cifar-train/{loss.tsv, run.json}.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts
import test_cifar as twin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

PROGRAM = os.path.join(ROOT, "examples/cifar_train/main.bend")
OUT = os.path.join(ROOT, "results/cifar-train")


def run_bend():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-cifar-train")
    build = subprocess.run([BEND, "-o", binary, PROGRAM],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    t0 = time.time()
    env = dict(ENV, MLBEND_DATA=os.path.join(ROOT, "reference/.cache/datasets"))
    out = subprocess.run([binary], capture_output=True, text=True, env=env, cwd=ROOT,
                         timeout=3600)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    return out.stdout, time.time() - t0


def number(out, name):
    match = re.search(rf"^{name}: ([0-9.eE+-]+)$", out, re.M)
    if not match:
        raise SystemExit(f"the program printed no `{name}:` line")
    return float(match.group(1))


def main():
    out, seconds = run_bend()
    print(out.strip())
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M)]
    final = number(out, "final_loss")
    drop = number(out, "loss_drop")

    answer = twin.train_answer()

    fail = 0
    if losses and abs(losses[0] - answer["first_loss"]) <= 1e-4 * max(1.0, abs(answer["first_loss"])):
        print(f"ok   the first loss agrees: {losses[0]} vs {answer['first_loss']}")
    else:
        print(f"ERROR the first loss differs: {losses[0]} vs {answer['first_loss']}")
        fail += 1

    delta = abs(drop - answer["loss_drop"])
    if delta <= 0.002 * abs(answer["loss_drop"]):
        print(f"ok   the drop agrees: {drop} vs {answer['loss_drop']:.7f} "
              f"(delta {delta:.2e})")
    else:
        print(f"ERROR the drop differs: {drop} vs {answer['loss_drop']:.7f} "
              f"(delta {delta:.2e})")
        fail += 1

    if drop > 1.0:
        print(f"ok   the loss fell: {drop:.6f} (first over last)")
    else:
        print(f"ERROR the loss did not fall: {drop:.6f}")
        fail += 1
    print(f"     final_loss (report): {final} vs {answer['final_loss']:.7f}")

    os.makedirs(OUT, exist_ok=True)
    rows = ["step\tloss"]
    for i, value in enumerate(losses):
        rows.append(f"{1 if i == 0 else i * 50}\t{value:.8g}")
    wrote = artifacts.save_text(os.path.join(OUT, "loss.tsv"), "\n".join(rows) + "\n")
    wrote |= artifacts.save_json(os.path.join(OUT, "run.json"), {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "example": "examples/cifar_train/main.bend",
        "seconds": seconds,
        "steps": twin.TRAIN_STEPS,
        "levels": list(twin.TRAIN_LEVELS),
        "lr": twin.TRAIN_LR,
        "bend": {"first_loss": losses[0], "final_loss": final, "loss_drop": drop},
        "numpy": answer,
    })
    print(f"{'wrote' if wrote else 'unchanged'} {OUT}/loss.tsv, run.json")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
