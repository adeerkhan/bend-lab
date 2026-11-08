"""The GLM Stage A harness and Stage B linear-attention forward, against a
frozen NumPy/PyTorch golden.

  reference/.venv/bin/python reference/test_glm53_layers.py

The Stage A model is 74,144 parameters: width 32, four dense-causal blocks,
four heads of 8, 64 positions, SwiGLU 128, tied embedding/head. The golden in
reference/glm53_golden_a.npz was produced by reference/glm53_twin.py (PyTorch
autograd) on the same NumPy-generated tensors; the Gate A probe prints whole
logits and all 74,144 parameter gradients. The Stage B probe prints the
parameterless linear-attention output on ramps; this test recomputes it with
NumPy. No PyTorch is needed in the gate.
"""
import hashlib
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))

V, DIM, LAYERS, H, HD, FF, T, B = 260, 32, 4, 4, 8, 128, 64, 2
N = B * T
P = [0, 4, 1, 5, 2, 6, 3, 7]
PI = [P.index(i) for i in range(8)]
DATA = os.path.join(ROOT, "examples/language/glm53_flash/data")
EX = "examples/language/glm53_flash"
BIN = "/tmp/glm53-gate-" + hashlib.sha1(ROOT.encode()).hexdigest()[:8]


def shapes():
    out = [("emb", (V, DIM))]
    for i in range(LAYERS):
        out += [(f"b{i}/ln1", (DIM,)), (f"b{i}/wq", (DIM, DIM)), (f"b{i}/wk", (DIM, DIM)),
                (f"b{i}/wv", (DIM, DIM)), (f"b{i}/wo", (DIM, DIM)), (f"b{i}/ln2", (DIM,)),
                (f"b{i}/wg", (FF, DIM)), (f"b{i}/wu", (FF, DIM)), (f"b{i}/wd", (DIM, FF))]
    return out + [("lnf", (DIM,))]


def prep():
    rng = np.random.default_rng(7)
    os.makedirs(DATA, exist_ok=True)
    for name, shape in shapes():
        x = np.ones(shape, dtype=np.float32) if name.endswith(("ln1", "ln2", "lnf")) else \
            (rng.standard_normal(shape) * 0.02).astype(np.float32)
        y = x.copy()
        if name.endswith(("wq", "wk")):
            for h in range(H):
                for j in range(HD):
                    y[h * HD + j] = x[h * HD + PI[j]]
        path = os.path.join(DATA, name + ".bin")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(y.astype("<f4").tobytes())
    ids = rng.integers(4, V, size=N).astype(np.uint8)
    labels = rng.integers(4, V, size=N).astype(np.uint8)
    open(os.path.join(DATA, "batch.bin"), "wb").write(ids.tobytes())
    open(os.path.join(DATA, "labels.bin"), "wb").write(labels.tobytes())
    alt = ids.copy()
    alt[N // 2:] = np.roll(alt[N // 2:], 1)
    open(os.path.join(DATA, "batch_alt.bin"), "wb").write(alt.tobytes())


def binary(src):
    srcs = [PROBE_A, PROBE_LIN, PROBE_ROPE, os.path.join(ROOT, EX, "model_a.bend"),
            os.path.join(ROOT, EX, "lin_core.bend")]
    out = BIN + "-" + os.path.basename(src).replace(".bend", "")
    if os.path.exists(out) and all(os.path.getmtime(out) >= os.path.getmtime(s) for s in srcs):
        return out
    build = subprocess.run([BEND, "-o", out, os.path.join(ROOT, src)], capture_output=True,
                           text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"{src} did not build:\n{build.stdout}{build.stderr}")
    return out


def run(src):
    out = subprocess.run([binary(src)], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit(f"{src} did not run:\n{out.stdout}{out.stderr}")
    return out.stdout


def tagged(text):
    got = {}
    for line in text.splitlines():
        m = re.match(r"^([a-z0-9_]+):\s*(.*)$", line)
        if m and m.group(2).strip():
            got[m.group(1)] = np.array([np.float32(v) for v in m.group(2).split()], dtype=np.float32)
    return got


def compare(name, got, want, atol, rtol):
    d = np.abs(got - want)
    bound = atol + rtol * np.abs(want)
    ok = got.shape == want.shape and bool((d <= bound).all())
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got.size} values, max |diff| {d.max():.3e}")
    return ok


PROBE_A = os.path.join(EX, "probe_a.bend")
PROBE_LIN = os.path.join(EX, "probe_lin_fwd.bend")
PROBE_ROPE = os.path.join(EX, "probe_rope.bend")


def main():
    prep()
    golden = np.load(os.path.join(ROOT, "reference/glm53_golden_a.npz"))
    out = tagged(run(PROBE_A))
    ok = []
    ok.append(compare("stage_a/logits", out["logits"], golden["logits"], 1e-5, 1e-5))
    ok.append(compare("stage_a/grad", out["grad"], golden["grad"], 1e-4, 1e-4))
    ok.append(abs(float(out["loss"][0]) - float(golden["loss"])) < 1e-5)
    print(f"{'ok  ' if ok[-1] else 'FAIL'} stage_a/loss: {float(out['loss'][0]):.7f}")

    # causality: the probe reads batch.bin; run it on the alt batch and restore
    batch = open(os.path.join(DATA, "batch.bin"), "rb").read()
    try:
        open(os.path.join(DATA, "batch.bin"), "wb").write(open(os.path.join(DATA, "batch_alt.bin"), "rb").read())
        alt = tagged(run(PROBE_A))["logits"].reshape(128, 260)
    finally:
        open(os.path.join(DATA, "batch.bin"), "wb").write(batch)
    orig = out["logits"].reshape(128, 260)
    same = np.allclose(alt[:64], orig[:64], atol=0.0)
    moved = not np.allclose(alt[64:], orig[64:], atol=0.0)
    ok.append(bool(same and moved))
    print(f"{'ok  ' if same and moved else 'FAIL'} stage_a/causality: first half exact, future changed")

    # Stage B: the parameterless linear attention on ramps, NumPy twin
    ramps = np.arange(4096, dtype=np.float32).reshape(128, 32)
    want = np.zeros((128, 32), dtype=np.float32)
    for b in range(2):
        for h in range(4):
            kp = np.zeros(8, dtype=np.float32); kvp = np.zeros((8, 8), dtype=np.float32)
            for t in range(64):
                r = b * 64 + t
                with np.errstate(over='ignore'):
                    kk = np.where(ramps[r, h * 8:h * 8 + 8] > 0, ramps[r, h * 8:h * 8 + 8],
                                  np.exp(ramps[r, h * 8:h * 8 + 8]) - 1).astype(np.float32) + 1
                qq = kk.copy(); vv = ramps[r, h * 8:h * 8 + 8]
                kp = (kp + kk).astype(np.float32); kvp = (kvp + np.outer(kk, vv)).astype(np.float32)
                den = np.float32(max(float(np.dot(qq, kp)), 1e-6))
                want[r, h * 8:h * 8 + 8] = ((kvp.T @ qq) / den).astype(np.float32)
    lin = tagged(run(PROBE_LIN))["y"]
    ok.append(compare("stage_b/linear_forward", lin, want.reshape(-1), 1e-5, 1e-5))

    # RoPE: the reference interleaved layout through the same permutation
    x = np.arange(2048, dtype=np.float32).reshape(64, 32)
    cols = [h * 8 + p for h in range(4) for p in P]
    cols_inv = [h * 8 + p for h in range(4) for p in PI]
    xr = x[:, cols].copy()
    yr = np.empty_like(xr)
    for h in range(4):
        for j in range(4):
            a = xr[:, h * 8 + 2 * j]; b = xr[:, h * 8 + 2 * j + 1]
            ang = np.arange(64, dtype=np.float32) * np.float32(10000.0 ** (-(2 * j) / 8))
            c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
            yr[:, h * 8 + 2 * j] = a * c - b * s
            yr[:, h * 8 + 2 * j + 1] = a * s + b * c
    yb = yr[:, cols_inv]
    got_rope = tagged(run(PROBE_ROPE))["y"].reshape(64, 32)
    ok.append(compare("stage_a/rope_forward", got_rope.reshape(-1), yb.reshape(-1), 1e-5, 1e-5))

    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
