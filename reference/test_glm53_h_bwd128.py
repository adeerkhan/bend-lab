"""The four-block hybrid's full gradient at N=8, against PyTorch autograd.

  reference/.venv/bin/python reference/test_glm53_h_bwd128.py

One backward through the whole model from the masked cross-entropy: every one
of the 74,144 parameter gradients is compared per tensor against the golden
`reference/glm53_h_bwd128_golden.npz` (`glm53_h_bwd.py` on the vendored modules).
The last two gates are the can-fail: the gradient of one moved label must NOT
match the correct golden, while matching its own bad-label golden.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glm53_h_data as D  # noqa: E402
import glm53_h_bwd_data as B  # noqa: E402

BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
BIN = f"/tmp/glm53-h-bwd-{TAG}"
EX = os.path.join(ROOT, "examples/language/glm53_flash")
DATA = os.path.join(EX, "data/ref8")
GOLDEN = os.path.join(ROOT, "reference/glm53_h_bwd128_golden.npz")


def binary():
    deps = [os.path.join(EX, name) for name in
            ("probe_model_bwd128.bend", "model_h8.bend", "model_h.bend", "weights_h.bend",
             "loss_masked.bend", "attn_mh8.bend", "attn_mh.bend", "lin_fast.bend",
             "lin_bwd.bend", "attn_c.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(EX, "probe_model_bwd128.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"probe_model_bwd128.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def run():
    proc = subprocess.run([binary()], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"probe_model_bwd128.bend did not run:\n{proc.stdout}{proc.stderr}")
    out = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^([a-z_]+):\s*([0-9.eE+-]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def compare(name, got, want, limit=1e-5):
    rel = float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))
    ok = got.shape == want.shape and rel <= limit and np.isfinite(got).all()
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got.size} values, max |diff| "
          f"{np.abs(got - want).max():.3e}, relative {rel:.3e}")
    return ok


def reject(name, got, want, limit=1e-5):
    rel = float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))
    ok = got.shape == want.shape and rel > limit
    print(f"{'ok  ' if ok else 'FAIL'} {name}: relative {rel:.3e} "
          f"(a match would be <= {limit:.0e})")
    return ok


def main():
    B.write128()
    out = run()
    h1 = hashlib.sha256(open(os.path.join(DATA, "model128_good_grad.bin"), "rb").read()).hexdigest()
    out2 = run()
    h2 = hashlib.sha256(open(os.path.join(DATA, "model128_good_grad.bin"), "rb").read()).hexdigest()
    golden = np.load(GOLDEN)
    bend = np.fromfile(os.path.join(DATA, "model128_good_grad.bin"), dtype="<f4").astype(np.float64)
    bad = np.fromfile(os.path.join(DATA, "model128_bad_grad.bin"), dtype="<f4").astype(np.float64)
    ok = []
    same = h1 == h2 and out["good_loss"] == out2["good_loss"]
    print(f"{'ok  ' if same else 'FAIL'} determinism: two probe runs, same loss and gradient sha256 {h1}")
    ok.append(same)
    loss = out["good_loss"]
    want_loss = float(golden["loss"])
    rel = abs(loss - want_loss) / max(abs(want_loss), 1e-30)
    good_loss = rel <= 1e-5
    print(f"{'ok  ' if good_loss else 'FAIL'} loss: {loss:.7f} against {want_loss:.7f}, "
          f"relative {rel:.3e}")
    ok.append(good_loss)
    want = golden["grad"].astype(np.float64)
    at = 0
    for name, shape in D.shapes():
        n = int(np.prod(shape))
        ok.append(compare(f"grad/{name}", bend[at:at + n], want[at:at + n]))
        at += n
    if at != bend.size:
        raise SystemExit(f"the flat gradient is {bend.size} values, the shapes total {at}")
    # can-fail: the moved label's gradient must not match the correct golden
    ok.append(reject("can-fail: moved label is rejected", bad, want))
    ok.append(compare("can-fail: moved label matches its own golden",
                      bad, golden["bad_grad"].astype(np.float64)))
    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
