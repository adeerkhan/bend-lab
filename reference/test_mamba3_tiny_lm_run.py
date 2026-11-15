"""Gate B3: the tiny-LM run on the held-out split, against the twin and the baselines.

  reference/.venv-rf/bin/python reference/test_mamba3_tiny_lm_run.py

Runs the same twin dump/train and the same Bend trainer as the B2 gate
(`reference/test_mamba3_tiny_opt.py`), then:

  checks every B2 invariant again (loss curve 1e-3, grad norms 1e-3,
  train hits within 2, final per-tensor sums of squares 1e-3);
  compares the two models' held-out bits/byte and token accuracy
  (within 0.02 absolute on the bits/byte, within 2 points on the
  accuracy), and
  writes the report card `results/mamba3-tiny-lm.md`, which records
  bits/byte, token accuracy, step count, schedule and seed, the
  baselines (unigram 4.8093, bigram 3.3569 bits/byte) and the exact
  commands.

The full corpus does not fit this lane's budget at Bend's speed, so the
run is the first 20 train examples (one pass, 20 steps) and the corpus's
own 4,096-byte held-out split, 4,064 next-byte targets. The baselines
are corpus statistics (`reference/mamba3_tiny_data.py`'s manifest), not
model results; the card says what the two numbers are.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, ".scratch/mamba3_tiny")
RESULT = os.path.join(ROOT, "results/mamba3-tiny-lm.md")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))
SRC = os.path.join(ROOT, "examples/language/mamba3/train_mamba3_tiny.bend")
BIN = os.path.join(ROOT, ".scratch/tiny_lm.bin")
LOSS_GATE = 1e-3
GN_GATE = 1e-3
HIT_SLACK = 2
W_GATE = 1e-3
BITS_ABS = 0.02
ACC_ABS = 0.02
STEPS = 20
LN2 = np.log(2)


def env():
    e = dict(os.environ, BEND_NO_TELEMETRY="1")
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        e["LD_LIBRARY_PATH"] = llvm + (":" + e["LD_LIBRARY_PATH"]
                                       if e.get("LD_LIBRARY_PATH") else "")
    return e


def parse(out):
    tags = {}
    for line in out.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        try:
            v = np.fromstring(rest, sep=" ", dtype=np.float32)
        except Exception:
            continue
        if v.size:
            tags[tag] = v
    return tags


def main():
    ok = True
    t0 = time.monotonic()
    subprocess.run([PYRF, os.path.join(ROOT, "reference/mamba3_tiny_twin.py"),
                    "--mode", "dump"], check=True, cwd=ROOT, env=env(),
                   capture_output=True)
    subprocess.run([PYRF, os.path.join(ROOT, "reference/mamba3_tiny_twin.py"),
                    "--mode", "train", "--steps", str(STEPS)],
                   check=True, cwd=ROOT, env=env(), capture_output=True,
                   timeout=3600)
    twin = json.load(open(os.path.join(OUT, "twin.json")))

    bend_built = False
    if os.path.exists(BIN):
        bend_built = os.path.getmtime(BIN) > os.path.getmtime(SRC)
    if not bend_built:
        subprocess.run([BEND, "-o", BIN, SRC], cwd=ROOT, env=env(), check=True,
                       capture_output=True, timeout=3600)
    t0 = time.monotonic()
    p = subprocess.run([BIN], cwd=ROOT, env=env(), capture_output=True,
                       text=True, timeout=10800)
    wall = time.monotonic() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("tiny-lm trainer binary failed")
    tags = parse(p.stdout)

    want = np.asarray(twin["losses"], dtype=np.float64)
    got = tags.get("tiny_train_losses").astype(np.float64)
    rel = np.abs(got - want) / np.abs(want)
    loss_ok = rel.max() <= LOSS_GATE
    ok = ok and loss_ok

    gn_t = np.asarray(twin["grad_norms"], dtype=np.float64)
    rel_gn = np.abs(tags["tiny_grad_norms"].astype(np.float64) - gn_t) / gn_t
    gn_ok = rel_gn.max() <= GN_GATE
    ok = ok and gn_ok

    hit_b = int(round(float(tags["tiny_train_hits"][0])))
    hit_t = int(twin["train_hits"])
    hit_ok = abs(hit_b - hit_t) <= HIT_SLACK
    ok = ok and hit_ok

    man = json.load(open(os.path.join(OUT, "w_init", "manifest.json")))
    flat = np.fromfile(os.path.join(OUT, "twin_params_final.bin"), dtype="<f4")
    names = [name for name, _ in man["order"]]
    sizes = [int(np.fromfile(os.path.join(OUT, f), dtype="<f4").size)
             for _, f in man["order"]]
    offs = np.cumsum([0] + sizes[:-1])
    gotw = tags["tiny_w_sumsq"].astype(np.float64)
    wantw = np.asarray([float((flat[o:o + n].astype(np.float64) ** 2).sum())
                        for o, n in zip(offs, sizes)])
    rel_w = np.abs(gotw - wantw) / np.maximum(np.abs(wantw), 1e-12)
    w_ok = rel_w.max() <= W_GATE
    ok = ok and w_ok

    held_t = twin["heldout"]
    bits_b = float(tags["tiny_bits_per_byte"][0])
    acc_b = float(tags["tiny_heldout_acc"][0])
    hits_b = int(round(float(tags["tiny_heldout_hits"][0])))
    nats_b = float(tags["tiny_heldout_nats"][0])
    bits_t = float(held_t["bits_per_byte"])
    acc_t = float(held_t["accuracy"])
    hits_t = int(held_t["hits"])
    bits_ok = abs(bits_b - bits_t) <= BITS_ABS
    acc_ok = abs(acc_b - acc_t) <= ACC_ABS
    ok = ok and bits_ok and acc_ok

    print(f"{'ok  ' if loss_ok else 'FAIL'} loss curve: worst rel {rel.max():.3e} (gate {LOSS_GATE:.0e})")
    print(f"{'ok  ' if gn_ok else 'FAIL'} grad norms: worst rel {rel_gn.max():.3e} (gate {GN_GATE:.0e})")
    print(f"{'ok  ' if hit_ok else 'FAIL'} train hits: bend {hit_b}/2540, twin {hit_t}/2540 (slack {HIT_SLACK})")
    print(f"{'ok  ' if w_ok else 'FAIL'} final weights: worst rel {rel_w.max():.3e} ({names[int(np.argsort(-rel_w)[0])]}), gate {W_GATE:.0e}")
    print(f"{'ok  ' if bits_ok else 'FAIL'} held-out bits/byte: bend {bits_b:.6f}, twin {bits_t:.6f} (abs {abs(bits_b - bits_t):.3e}, tolerance {BITS_ABS})")
    print(f"{'ok  ' if acc_ok else 'FAIL'} held-out accuracy: bend {acc_b:.4f}, twin {acc_t:.4f} (abs {abs(acc_b - acc_t):.3e}, tolerance {ACC_ABS})")
    print(f"bend: {wall:.1f} s wall for {STEPS} steps ({wall / STEPS:.1f} s per step)")

    manifest_tiny = json.load(open(os.path.join(OUT, "manifest.json")))
    lines = [
        "# Mamba-3 tiny-LM — the held-out run",
        "",
        "Recorded by `reference/test_mamba3_tiny_lm_run.py` (regenerated on every run).",
        "",
        "Command: `make mamba3-tiny-lm` (`reference/test_mamba3_tiny_lm_run.py`)",
        "",
        "## Configuration",
        "",
        "- Model: two-layer SISO Mamba-3 LM, d_model 64, d_state 16, expand 2, headdim 32.",
        "- Vocabulary 127 (the corpus's observed maximum + 1); positions 128; targets 127 per example.",
        "- Data: `.scratch/mamba3_tiny/` (built by `reference/mamba3_tiny_data.py`):",
        "  the first 64 corpus examples (8,192 tokens) for training, the corpus's own",
        "  held-out split (32 examples, 4,096 tokens, 4,064 next-byte targets).",
        "- Schedule: batch 1 in file order (no shuffling), one example per step, the",
        "  first 20 train examples, 20 steps. AdamW lr = warmup (5 steps linear 0 -> 1e-3)",
        "  then cosine from 1e-3 to 1e-4 over steps 5..19; weight_decay 0.01,",
        "  betas (0.9, 0.999), eps 1e-8, gradient clipping by global norm at 1.0,",
        "  bias correction, decoupled decay, torch's step order.",
        "- Seed: l1 101, l2 102, E/Wh/nw 20261115 (`reference/mamba3_tiny_twin.py`).",
        "- Bend: float32; the twin: float32 on the CPU.",
        "",
        "## Numbers",
        "",
        "| | Bend | twin (float32, CPU) |",
        "|---|---|---|",
        f"| train hits (pre-update, of 2540) | {hit_b} | {hit_t} |",
        f"| train accuracy (pre-update) | {hit_b / 2540:.4f} | {hit_t / 2540:.4f} |",
        f"| held-out CE, nats (of 4,064 targets) | {nats_b:.1f} | {held_t['ce_nats']:.1f} |",
        f"| held-out bits/byte | **{bits_b:.4f}** | **{bits_t:.4f}** |",
        f"| held-out token accuracy | **{acc_b:.4f}** ({hits_b}/{int(held_t['total'])}) | **{acc_t:.4f}** ({hits_t}/{int(held_t['total'])}) |",
        f"| final weights (19 per-tensor sums of squares), worst rel | {rel_w.max():.3e} | reference |",
        "",
        "## Against the baselines (the same 4,096 held-out bytes)",
        "",
        "- unigram 4.8093 bits/byte; bigram **3.3569** bits/byte (the corpus statistics, unchanged by this run);",
        f"- the twin after {STEPS} steps: {bits_t:.4f} bits/byte; the Bend trainer: {bits_b:.4f}.",
        "",
        "The run is 20 steps of 127-target cross entropy on 20 train examples; the",
        "held-out split was not seen in training. Both numbers above are what those",
        "20 steps reach. **The full corpus pass and a convergence claim are not made",
        "by this card**: the Bend 128-position fwd+bwd measures seconds per image on",
        "this lane, so the full 8,192-token pass is not this session's budget, and the",
        "comparison to the bigram baseline is recorded, not met.",
        "",
        "## Gates on this artifact",
        "",
        "- `reference/test_mamba3_tiny_data.py` (`make mamba3-tiny-data`): the data.",
        "- `reference/test_mamba3_tiny_opt.py` (`make mamba3-tiny-opt`): optimizer parity, same run.",
        "- `reference/test_mamba3_tiny_lm_run.py` (`make mamba3-tiny-lm`): this card.",
        "",
        "MAMBA3 TINY-LM: " + ("PASS" if ok else "FAIL"),
    ]
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(RESULT, ROOT)}")
    print("MAMBA3 TINY-LM", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
