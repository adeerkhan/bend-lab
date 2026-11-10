"""Twenty AdamW steps of the 128 hybrid against the PyTorch twin.

  reference/.venv/bin/python reference/test_glm53_train20.py

`train_h128 --steps 20 --no-val` against
`reference/glm53_h_train128.py --golden reference/glm53_h_train128_golden.npz`.
The 20 losses and the final 74,144 weights are compared per tensor; a run at
lr 0.004 is the can-fail against the lr 0.003 golden.
"""
from __future__ import annotations

import hashlib
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
import glm53_h_data as D  # noqa: E402

BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
BIN = f"/tmp/glm53-train20-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")
GOLDEN = os.path.join(ROOT, "reference/glm53_h_train128_golden.npz")
BAD = os.path.join(ROOT, "reference/glm53_h_train128_bad.npz")


def binary():
    deps = [os.path.join(EX, name) for name in
            ("train_h128.bend", "model_h.bend", "weights_h.bend", "loss_masked.bend",
             "attn_mh.bend", "lin_dense.bend", "sp_dense.bend", "lin_bwd.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(EX, "train_h128.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"train_h128.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def run(out, lr):
    os.makedirs(out, exist_ok=True)
    proc = subprocess.run([binary(), "--steps", "20", "--no-val", "--lr-ppm", str(lr),
                           "--out", out], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"train_h128 did not run:\n{proc.stdout}{proc.stderr}")
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", proc.stdout, re.M)]
    final = re.search(r"^final_loss: ([0-9.eE+-]+)$", proc.stdout, re.M)
    if final is None:
        raise SystemExit(f"no final_loss:\n{proc.stdout}")
    losses.append(float(final.group(1)))
    weights = np.fromfile(os.path.join(out, "final_weights.bin"), dtype="<f4").astype(np.float64)
    return losses, weights


def main():
    golden = np.load(GOLDEN)
    want_losses = golden["losses"].astype(np.float64)
    want = golden["weights"].astype(np.float64)
    tmp = tempfile.mkdtemp(prefix="glm53-train20-", dir="/tmp/opencode")
    try:
        got_losses, got = run(os.path.join(tmp, "good"), 3000)
        bad = np.load(BAD)["weights"].astype(np.float64)
        ok = []
        same_len = len(got_losses) >= len(want_losses)
        print(f"{'ok  ' if same_len else 'FAIL'} loss count: {len(got_losses)} printed, {len(want_losses)} golden")
        ok.append(same_len)
        loss_rel = max(abs(a - b) / max(abs(b), 1e-30)
                       for a, b in zip(got_losses[:len(want_losses)], want_losses))
        loss_ok = same_len and loss_rel <= 1e-5
        print(f"{'ok  ' if loss_ok else 'FAIL'} 20 losses: worst relative {loss_rel:.3e} (gate 1e-5)")
        ok.append(loss_ok)
        at = 0
        for name, shape in D.shapes():
            n = int(np.prod(shape))
            w = want[at:at + n]
            rel = float(np.abs(got[at:at + n] - w).max() / max(float(np.abs(w).max()), 1e-30))
            good = rel <= 1e-3
            print(f"{'ok  ' if good else 'FAIL'} weights/{name}: relative {rel:.3e} (gate 1e-3)")
            ok.append(good)
            at += n
        if at != got.size:
            raise SystemExit(f"the flat weights are {got.size}, the shapes total {at}")
        bad_rel = float(np.abs(bad - want).max() / max(float(np.abs(want).max()), 1e-30))
        rejected = bad_rel > 1e-3
        print(f"{'ok  ' if rejected else 'FAIL'} can-fail lr 0.004: relative {bad_rel:.3e} (must be > 1e-3)")
        ok.append(rejected)
        print(f"{sum(ok)}/{len(ok)} gates passed")
        return 0 if all(ok) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
