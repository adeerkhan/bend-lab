"""The trained transformer: BendTorch against a NumPy twin.

  reference/.venv/bin/python reference/test_transformer.py

`examples/transformers/periodic/main.bend` trains an embedding, one causal single-head
attention block (RMSNorm, attention, output projection), a second RMSNorm and
a linear head, with AdamW on next-token prediction over the periodic sequence
0..7. This script builds the same architecture with the same deterministic
initial weights in NumPy, trains it with the same optimizer and steps, and
writes both loss curves to results/transformer/:

  loss.tsv    step, bend loss, numpy loss
  loss.svg    the two curves
  run.json    the configuration, the versions and the final numbers

The two implementations are not bit-identical (the attention kernel and the
optimizer round differently), so the comparison is of the curve and the final
loss, not of every weight. Both must learn the cycle; the Bend program must
fall from about log(8) to a small number.
"""
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

import artifacts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

STEPS = 200
VOCAB = 8
WIDTH = 16
POSITIONS = 16
SEQUENCES = 2
ROWS = SEQUENCES * POSITIONS
LR = 0.01
B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 0.0


def init_values(rows, cols, seed):
    return np.array(
        [[((3 * i + 5 * j + seed) % 7) * 0.1 - 0.2 for j in range(cols)] for i in range(rows)],
        dtype=np.float32,
    )


def rms_forward(x, gamma):
    inv = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5)
    xhat = x * inv
    return xhat * gamma, (xhat, inv)


def rms_backward(dy, gamma, cache):
    xhat, inv = cache
    dxhat = dy * gamma
    return inv * (dxhat - (dxhat * xhat).mean(axis=-1, keepdims=True) * xhat)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits, targets):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    n = logits.shape[0]
    return float(-logp[np.arange(n), targets].mean()), (softmax(logits) - np.eye(VOCAB, dtype=np.float32)[targets]) / n


def forward(params, ids):
    x = params["emb"][ids]
    h, c1 = rms_forward(x, params["gamma1"])
    q = (h @ params["wq"].T).reshape(SEQUENCES, POSITIONS, WIDTH)
    k = (h @ params["wk"].T).reshape(SEQUENCES, POSITIONS, WIDTH)
    v = (h @ params["wv"].T).reshape(SEQUENCES, POSITIONS, WIDTH)
    scores = (q @ k.transpose(0, 2, 1)) * 0.25
    mask = np.triu(np.ones((POSITIONS, POSITIONS), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    p = softmax(scores)
    att = (p @ v).reshape(ROWS, WIDTH)
    out = att @ params["wo"].T
    h2, c2 = rms_forward(out, params["gamma2"])
    logits = h2 @ params["head"].T
    cache = {"h": h, "q": q, "k": k, "v": v, "p": p, "att": att, "h2": h2, "c1": c1, "c2": c2}
    return logits, cache


def backward(params, cache, dlogits, ids):
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    grads["head"] = dlogits.T @ cache["h2"]
    dh2 = dlogits @ params["head"]
    dout = rms_backward(dh2, params["gamma2"], cache["c2"])
    grads["wo"] = cache["att"].T @ dout
    datt = (dout @ params["wo"]).reshape(SEQUENCES, POSITIONS, WIDTH)
    dp = datt @ cache["v"].transpose(0, 2, 1)
    dv = cache["p"].transpose(0, 2, 1) @ datt
    dscores = cache["p"] * (dp - (dp * cache["p"]).sum(axis=-1, keepdims=True))
    dscores *= 0.25
    dq = dscores @ cache["k"]
    dk = dscores.transpose(0, 2, 1) @ cache["q"]
    dq = dq.reshape(ROWS, WIDTH)
    dk = dk.reshape(ROWS, WIDTH)
    dv = dv.reshape(ROWS, WIDTH)
    grads["wq"] = cache["h"].T @ dq
    grads["wk"] = cache["h"].T @ dk
    grads["wv"] = cache["h"].T @ dv
    dh = dq @ params["wq"] + dk @ params["wk"] + dv @ params["wv"]
    dx = rms_backward(dh, params["gamma1"], cache["c1"])
    np.add.at(grads["emb"], ids, dx)
    grads["gamma1"] = (dh * cache["c1"][0]).sum(axis=0)
    grads["gamma2"] = (dh2 * cache["c2"][0]).sum(axis=0)
    return grads


def numpy_curve():
    ids = np.array([i % VOCAB for i in range(ROWS)], dtype=np.int64)
    targets = (ids + 1) % VOCAB
    params = {
        "emb": init_values(VOCAB, WIDTH, 1),
        "wq": init_values(WIDTH, WIDTH, 2),
        "wk": init_values(WIDTH, WIDTH, 3),
        "wv": init_values(WIDTH, WIDTH, 4),
        "wo": init_values(WIDTH, WIDTH, 5),
        "gamma1": np.ones(WIDTH, dtype=np.float32),
        "gamma2": np.ones(WIDTH, dtype=np.float32),
        "head": init_values(VOCAB, WIDTH, 6),
    }
    moment = {k: np.zeros_like(v) for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    curve = []
    for step in range(1, STEPS + 1):
        logits, cache = forward(params, ids)
        loss, dlogits = cross_entropy(logits, targets)
        curve.append(loss)
        grads = backward(params, cache, dlogits, ids)
        for k in params:
            g = grads[k]
            moment[k] = B1 * moment[k] + (1 - B1) * g
            velocity[k] = B2 * velocity[k] + (1 - B2) * (g * g)
            mhat = moment[k] / (1 - B1 ** step)
            vhat = velocity[k] / (1 - B2 ** step)
            params[k] -= LR * (mhat / (np.sqrt(vhat) + EPS) + WD * params[k])
    logits, _ = forward(params, ids)
    final, _ = cross_entropy(logits, targets)
    predicted = logits.argmax(axis=-1)
    accuracy = float((predicted == targets).mean())
    return curve, float(final), accuracy


def bend_curve():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-transformer")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/transformers/periodic/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    start = time.perf_counter()
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    seconds = time.perf_counter() - start
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", out.stdout, re.M)]
    accuracy = float(re.search(r"^accuracy: ([0-9.eE+-]+)$", out.stdout, re.M).group(1))
    return [v for v in losses if v > 0.0], accuracy, seconds


def svg(curves, path):
    width, height = 720, 360
    left, right, top, bottom = 60, 20, 30, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    top_loss = max(max(c) for c in curves) * 1.1

    def point(i, n, value):
        x = left + plot_w * (i / max(1, n - 1))
        y = top + plot_h * (1.0 - value / top_loss)
        return f"{x:.1f},{y:.1f}"

    colors, names = ["#c0392b", "#2c3e50"], ["BendTorch", "NumPy"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="20" font-family="sans-serif" font-size="14">'
        'Next-token loss on the periodic sequence (lower is better)</text>',
    ]
    for tick in range(5):
        value = top_loss * tick / 4
        y = top + plot_h * (1.0 - value / top_loss)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2f}</text>')
    for n, (curve, color, name) in enumerate(zip(curves, colors, names)):
        points = " ".join(point(i, len(curve), v) for i, v in enumerate(curve))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        parts.append(f'<text x="{left + 10 + n * 100}" y="{top + 15}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="12">step</text>')
    parts.append("</svg>")
    return artifacts.save_text(path, "\n".join(parts))


def main():
    bend, bend_accuracy, seconds = bend_curve()
    curve, final, numpy_accuracy = numpy_curve()
    ok = True

    print(f"bend loss:  {bend[0]:.4f} -> {bend[-1]:.4f} in {seconds:.2f} s, accuracy {bend_accuracy}")
    print(f"numpy loss: {curve[0]:.4f} -> {final:.4f}, accuracy {numpy_accuracy}")
    if len(bend) != STEPS:
        print(f"ERROR the Bend run printed {len(bend)} losses, not {STEPS}")
        ok = False
    if bend[0] < math.log(VOCAB) * 0.8:
        print(f"ERROR the first Bend loss is too low to be untrained: {bend[0]}")
        ok = False
    if bend[-1] > 0.05:
        print(f"ERROR the final Bend loss is above 0.05: {bend[-1]}")
        ok = False
    if final > 0.2:
        print(f"ERROR the final NumPy loss is above 0.2: {final}")
        ok = False
    if bend_accuracy < 0.99:
        print(f"ERROR the Bend accuracy is below 0.99: {bend_accuracy}")
        ok = False
    if numpy_accuracy < 0.9:
        print(f"ERROR the NumPy accuracy is below 0.9: {numpy_accuracy}")
        ok = False
    if abs(bend[0] - curve[0]) > 0.25:
        print(f"ERROR the initial losses differ too much: {bend[0]} vs {curve[0]}")
        ok = False

    out_dir = os.path.join(ROOT, "results/transformer")
    os.makedirs(out_dir, exist_ok=True)
    n = min(len(bend), len(curve))
    rows = ["step\tbend\tnumpy"]
    for i in range(n):
        rows.append(f"{i+1}\t{bend[i]:.8f}\t{curve[i]:.8f}")
    wrote = artifacts.save_text(os.path.join(out_dir, "loss.tsv"), "\n".join(rows) + "\n")
    wrote |= svg([bend[:n], curve[:n]], os.path.join(out_dir, "loss.svg"))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "example": "examples/transformers/periodic/main.bend",
        "architecture": "embedding, RMSNorm, causal single-head attention, output projection, RMSNorm, linear head",
        "dataset": {"name": "periodic-0-7", "rows": ROWS, "sequences": SEQUENCES,
                    "positions": POSITIONS, "vocab": VOCAB},
        "optimizer": {"name": "AdamW", "lr": LR, "steps": STEPS},
        "bend": {"first_loss": bend[0], "final_loss": bend[-1], "accuracy": bend_accuracy, "seconds": seconds},
        "numpy": {"first_loss": curve[0], "final_loss": final, "accuracy": numpy_accuracy},
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                    "numpy": np.__version__},
        "passed": ok,
    }
    wrote |= artifacts.save_json(os.path.join(out_dir, "run.json"), manifest)
    print(f"{'wrote' if wrote else 'unchanged'} {out_dir}/loss.tsv, loss.svg, run.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
