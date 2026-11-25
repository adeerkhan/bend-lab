"""Gate + record: the ten-step tiny train, custom loop vs T.Train.

  reference/.venv/bin/python reference/test_mamba3_train_curve.py

Extends reference/test_mamba3_layer.py with the recording the layer gate
cannot do: the custom trainer's ten-loss curve is printed verbatim, the
final layer loss is recorded next to it, and the artifacts are written to
`results/mamba3-train-curve.md`. The gate itself is unchanged:

  layer_tokens == train_tokens (exactly)
  |layer_loss  - train_loss[-1] | / |layer_loss| <= 1e-3 (relative)
  weight heads (w_in, E, Wh, w_out): max relative distance <= 1e-5

The per-step loss is NOT readable from outside T.Train (bendtorch#86), so
the per-step curve cannot be compared; only the final step is checked. The
custom curve is recorded here because it is the reference curve the layer
gate would want to match if per-step access were added upstream.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
BIN = os.path.join(ROOT, ".scratch/mamba3_layer.bin")
RESULT = os.path.join(ROOT, "results/mamba3-train-curve.md")
LOSS_GATE = 1e-3
WEIGHT_GATE = 1e-5


def parse(out):
    tags = {}
    for line in out.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        try:
            vals = np.fromstring(rest, sep=" ", dtype=np.float32)
        except Exception:
            continue
        if vals.size:
            tags[tag] = vals
    return tags


def main():
    os.makedirs(os.path.join(ROOT, ".scratch"), exist_ok=True)
    env = dict(os.environ)
    env["BEND_NO_TELEMETRY"] = "1"
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")

    version = subprocess.run([BEND, "version"], cwd=ROOT, env=env,
                             check=True, capture_output=True, text=True).stdout.strip()
    version = version.removeprefix("bend ")

    t0 = time.monotonic()
    subprocess.run([BEND, "-o", BIN,
                    os.path.join(ROOT, "examples/language/mamba3/mamba3_layer.bend")],
                   cwd=ROOT, env=env, check=True, capture_output=True,
                   timeout=1800)
    build = time.monotonic() - t0
    t0 = time.monotonic()
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=1800)
    seconds = time.monotonic() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("layer binary failed")
    tags = parse(p.stdout)

    train_tokens = tags.get("train_tokens")
    layer_tokens = tags.get("layer_tokens")
    if train_tokens is None or layer_tokens is None:
        raise SystemExit("missing train_tokens or layer_tokens in output")
    same_tokens = np.array_equal(train_tokens.astype(int),
                                 layer_tokens.astype(int))

    tl = tags["train_loss"].astype(np.float64)
    ll = tags["layer_loss"].astype(np.float64)
    final_custom = float(tl[-1])
    final_layer = float(ll[-1])
    rel = abs(final_layer - final_custom) / max(1e-30, abs(final_layer))
    loss_ok = rel <= LOSS_GATE

    head_results = []
    heads_ok = True
    for key in ["w_in", "E", "Wh", "w_out"]:
        c = tags.get(f"custom_{key}")
        l = tags.get(f"layer_{key}")
        if c is None or l is None:
            print(f"FAIL {key}: not printed")
            heads_ok = False
            head_results.append((key, None))
            continue
        c = c.astype(np.float64)
        l = l.astype(np.float64)
        r = float(np.abs(c - l).max() / max(1e-30, np.abs(c).max()))
        ok = r <= WEIGHT_GATE
        heads_ok = heads_ok and ok
        head_results.append((key, r))
        print(f"{'ok  ' if ok else 'FAIL'} {key} head: max rel {r:.2e} "
              f"gate {WEIGHT_GATE:.0e}")

    print(f"{'ok  ' if same_tokens else 'FAIL'} tokens: "
          f"{layer_tokens.astype(int).tolist()}")
    print(f"{'ok  ' if loss_ok else 'FAIL'} final loss: T.Train {final_layer:.7f} "
          f"vs custom {final_custom:.7f} rel {rel:.2e} gate {LOSS_GATE:.0e}")
    print(f"     custom curve: {' '.join(f'{v:.7f}' for v in tl)}")
    print(f"     build {build:.1f} s, run {seconds:.2f} s wall")
    ok = same_tokens and loss_ok and heads_ok
    print("MAMBA3 TRAIN CURVE", "PASS" if ok else "FAIL")

    lines = [
        "# mamba3 ten-step train — recorded curve",
        "",
        "Recorded by `reference/test_mamba3_train_curve.py` (regenerated on every run).",
        "",
        "Command: `reference/.venv/bin/python reference/test_mamba3_train_curve.py`",
        f"Build {build:.1f} s, run {seconds:.2f} s wall. Bend {version}, reference/.venv.",
        "",
        "Custom trainer (ten steps, lr 1e-3), ten losses:",
        "",
        "```",
        " ".join(f"{v:.7f}" for v in tl),
        "```",
        "",
        f"- `train_loss[0]` {tl[0]:.7f} → `train_loss[9]` {tl[-1]:.7f}",
        f"- `train_tokens` {train_tokens.astype(int).tolist()}",
        f"- layer final loss (`layer_loss`): {final_layer:.7f}",
        f"- relative gap |layer-custom|/|layer|: {rel:.2e} (gate {LOSS_GATE:.0e})"
        + (" — PASS" if loss_ok else " — FAIL"),
        f"- tokens equal: {'yes' if same_tokens else 'no'}",
    ]
    for key, r in head_results:
        if r is None:
            lines.append(f"- {key} head: NOT PRINTED")
        else:
            lines.append(f"- {key} head: max rel {r:.2e} (gate {WEIGHT_GATE:.0e})")
    lines += [
        "",
        "Per-step losses are not readable from outside `T.Train` (bendtorch#86 —",
        "`Train` carries a single `loss: F32`, the final step only), so no",
        "per-step curve can be recorded from the layer path; this file records",
        "the custom curve and the final-step comparison only.",
        "",
    ]
    with open(RESULT, "w") as f:
        f.write("\n".join(lines))
    print(f"     wrote {os.path.relpath(RESULT, ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
