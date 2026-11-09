"""The integrated Stage C gate: 20 AdamW steps against the PyTorch twin.

  reference/.venv/bin/python reference/test_glm53_c.py

The model is `model_c.bend`: one batch, 64 positions, one head of width 8,
linear attention then sparse attention (window 16, stride 16), each with a
SwiGLU hidden 32, a final RMSNorm and the tied head: **4,168 parameters**.
The twin is `reference/glm53_c_twin.py`.  Every step's loss is compared at
1e-4 relative and the final weights and gradients at 1e-3 whole-tensor
relative (20 AdamW steps amplify last-bit rounding).  Two one-step can-fail
runs (a changed weight and a changed input byte) must diverge at step 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glm53_c_data as D  # noqa: E402

BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
BIN = f"/tmp/glm53-c-train-{TAG}"
GOLDEN = os.path.join(ROOT, "reference/glm53_c_golden.npz")
EX = os.path.join(ROOT, "examples/language/glm53_flash")


def binary():
    deps = [os.path.join(EX, name) for name in
            ("train_c.bend", "model_c.bend", "attn_c.bend", "lin_bwd.bend",
             "weights_c.bend", "loss_masked.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(EX, "train_c.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"train_c.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def run_train(data, out, steps):
    os.makedirs(out, exist_ok=True)
    proc = subprocess.run([binary(), "--steps", str(steps), "--data", data, "--out", out],
                          capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"train_c.bend did not run:\n{proc.stdout}{proc.stderr}")
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", proc.stdout, re.M)]
    final = re.search(r"^final_loss: ([0-9.eE+-]+)$", proc.stdout, re.M)
    params = re.search(r"^params: (\d+)$", proc.stdout, re.M)
    if final is None or params is None:
        raise SystemExit(f"the run did not print final_loss/params:\n{proc.stdout}")
    losses.append(float(final.group(1)))
    return losses, int(params.group(1)), proc.stdout


def relative(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return float("inf")
    return float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))


def gate_losses(got, want, limit=1e-4):
    ok = True
    rels = []
    for i, (a, b) in enumerate(zip(got, want), 1):
        a, b = float(a), float(b)
        rel = abs(a - b) / max(abs(b), 1e-30)
        rels.append(rel)
        if rel > limit:
            print(f"FAIL step {i}: {a:.9g} vs {b:.9g}, relative {rel:.3e}")
            ok = False
            break
    if len(got) != len(want):
        print(f"FAIL loss count: {len(got)} vs {len(want)}")
        ok = False
    if ok:
        print(f"ok   loss: {len(got)} steps, max relative {max(rels):.3e} "
              f"(step 1 {got[0]:.9g} vs {want[0]:.9g})")
    return ok


def gate_tensor(name, got, want, limit=1e-3):
    rel = relative(got, want)
    ok = got.shape == want.shape and rel <= limit
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got.size} values, max |diff| "
          f"{np.abs(got - want).max():.3e}, whole-tensor relative {rel:.3e}")
    return ok


def copy_data(source):
    target = tempfile.mkdtemp(prefix="glm53-c-corrupt-")
    for name in os.listdir(source):
        src = os.path.join(source, name)
        dst = os.path.join(target, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return target


def corrupt_weight(source):
    target = copy_data(source)
    path = os.path.join(target, "emb.bin")
    x = np.fromfile(path, dtype="<f4")
    x[0] += 2.0
    x.astype("<f4").tofile(path)
    return target


def corrupt_batch(source):
    target = copy_data(source)
    path = os.path.join(target, "c_batch.bin")
    x = bytearray(open(path, "rb").read())
    x[0] = 42
    open(path, "wb").write(bytes(x))
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", type=os.path.abspath, default=None)
    args = ap.parse_args()
    D.prepare()
    golden = np.load(GOLDEN)
    out = tempfile.mkdtemp(prefix="glm53-c-20-")
    losses, params, _ = run_train(D.DATA, out, 20)
    ok = [params == 4168]
    print(f"params: {params}")
    ok.append(gate_losses(losses, golden["losses"]))
    weights = np.fromfile(os.path.join(out, "final_weights.bin"), dtype="<f4")
    grads = np.fromfile(os.path.join(out, "final_grads.bin"), dtype="<f4")
    ok.append(gate_tensor("final weights", weights, golden["weights"]))
    ok.append(gate_tensor("final gradients", grads, golden["grads"]))

    corrupt_rel = {}
    for name, corrupt in (("weight", corrupt_weight), ("batch", corrupt_batch)):
        bad_dir = corrupt(D.DATA)
        bad_out = tempfile.mkdtemp(prefix=f"glm53-c-{name}-")
        bad, _, _ = run_train(bad_dir, bad_out, 1)
        rel = abs(bad[0] - float(golden["losses"][0])) / max(abs(float(golden["losses"][0])), 1e-30)
        corrupt_rel[name] = rel
        rejected = rel > 1e-4
        print(f"{'ok  ' if rejected else 'FAIL'} corrupt {name}: step-1 relative {rel:.3e} "
              f"({'rejected' if rejected else 'not rejected'})")
        ok.append(rejected)
        shutil.rmtree(bad_dir, ignore_errors=True)

    if args.record:
        record = {
            "stage": "integrated-linear-and-sparse-attention",
            "model": "1 batch, 64 positions, 1 head of width 8, linear then sparse (window 16, stride 16), SwiGLU 32, tied head",
            "params": params,
            "steps": len(losses),
            "tokens_seen": 64 * len(losses),
            "optimizer": {"name": "AdamW", "lr": 3e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1},
            "losses": losses,
            "twin": {
                "loss_max_relative": float(np.max(np.abs(np.asarray(losses) - golden["losses"])
                                                 / np.abs(golden["losses"]))),
                "weights_relative": relative(weights, golden["weights"]),
                "gradients_relative": relative(grads, golden["grads"]),
                "corrupt_weight_step1_relative": corrupt_rel["weight"],
                "corrupt_batch_step1_relative": corrupt_rel["batch"],
            },
            "passed": bool(all(ok)),
        }
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        open(args.record, "w").write(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.record}")

    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
