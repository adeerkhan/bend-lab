"""Gate B2: the tiny-LM optimizer and schedule against the twin.

  reference/.venv-rf/bin/python reference/test_mamba3_tiny_opt.py

Builds the twin (`reference/mamba3_tiny_twin.py --mode dump --mode train`)
and the Bend trainer on the same first 20 train examples with the same
schedule (warmup 5, cosine to 1e-4, AdamW lr 1e-3, weight_decay 0.01,
betas (0.9, 0.999), eps 1e-8, clipping by global norm at 1.0, batch 1 in
file order, one example per step, 127 next-byte targets per example) and
compares:

  the loss at every step, at 1e-3 relative;
  the global gradient norm at every step, at 1e-3 relative;
  the train hits (pre-update forwards), within 2 examples;
  the final per-tensor sums of squares (19 tensors), at 1e-3 relative.

The twin runs in float32 on the CPU; the Bend trainer is float32. The
full corpus (64 examples, one pass) and a full convergence claim are not
attempted: a 128-position fwd+bwd is what it costs, and the run uses the
first 20 train examples. The twin's 94.61 %-style headline number does
not exist for this lane at this budget.

The report card (`results/mamba3-tiny-lm.md`) is written by
`reference/test_mamba3_tiny_lm_run.py` (B3), which reads the same twin
and Bend outputs.
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
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))
SRC = os.path.join(ROOT, "examples/language/mamba3/train_mamba3_tiny.bend")
BIN = os.path.join(ROOT, ".scratch/tiny_lm.bin")
LOSS_GATE = 1e-3
GN_GATE = 1e-3
HIT_SLACK = 2          # at most two of the 20 examples may be decided differently
W_GATE = 1e-3
STEPS = 20


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
    got = tags.get("tiny_train_losses")
    if got is None or got.size != want.size:
        print(f"FAIL tiny_train_losses: {None if got is None else got.size} values, "
              f"the twin has {want.size}")
        ok = False
    else:
        got = got.astype(np.float64)
        rel = np.abs(got - want) / np.abs(want)
        i = int(rel.argmax())
        print(f"{'ok  ' if rel.max() <= LOSS_GATE else 'FAIL'} loss curve: "
              f"{got.size} steps, worst step {i} rel {rel[i]:.3e} "
              f"(bend {got[i]:.7f}, twin {want[i]:.7f}), gate {LOSS_GATE:.0e}")
        ok = ok and rel.max() <= LOSS_GATE

    gn_t = np.asarray(twin["grad_norms"], dtype=np.float64)
    gn_b = tags.get("tiny_grad_norms")
    if gn_b is not None and gn_b.size == gn_t.size:
        rel = np.abs(gn_b.astype(np.float64) - gn_t) / gn_t
        print(f"{'ok  ' if rel.max() <= GN_GATE else 'FAIL'} grad norms: "
              f"worst rel {rel.max():.3e}, clipped at 1.0 every step")
        ok = ok and rel.max() <= GN_GATE
    else:
        print("FAIL tiny_grad_norms: not printed")
        ok = False

    th_b = tags.get("tiny_train_hits")
    if th_b is not None:
        hit_b = int(round(float(th_b[0])))
        hit_t = int(twin["train_hits"])
        good = abs(hit_b - hit_t) <= HIT_SLACK
        print(f"{'ok  ' if good else 'FAIL'} train hits: bend {hit_b}/2540, "
              f"twin {hit_t}/2540, {abs(hit_b - hit_t)} apart, "
              f"slack {HIT_SLACK} examples")
        ok = ok and good
    else:
        print("FAIL tiny_train_hits: not printed")
        ok = False

    man = json.load(open(os.path.join(OUT, "w_init", "manifest.json")))
    flat = np.fromfile(os.path.join(OUT, "twin_params_final.bin"), dtype="<f4")
    if flat.size != man["param_count"]:
        print(f"FAIL twin flat: {flat.size} values, manifest has {man['param_count']}")
        ok = False
    else:
        names = [name for name, _ in man["order"]]
        sizes = [int(np.fromfile(os.path.join(OUT, f), dtype="<f4").size)
                 for _, f in man["order"]]
        offs = np.cumsum([0] + sizes[:-1])
        gotw = tags.get("tiny_w_sumsq")
        if gotw is None or gotw.size != len(names):
            print(f"FAIL tiny_w_sumsq: {None if gotw is None else gotw.size} values, "
                  f"the manifest has {len(names)} tensors")
            ok = False
        else:
            gotw = gotw.astype(np.float64)
            wantw = np.asarray([float((flat[o:o + n].astype(np.float64) ** 2).sum())
                                for o, n in zip(offs, sizes)])
            rel = np.abs(gotw - wantw) / np.maximum(np.abs(wantw), 1e-12)
            i = int(np.argsort(-rel)[0])
            print(f"{'ok  ' if rel.max() <= W_GATE else 'FAIL'} final weights: "
                  f"{gotw.size} tensors, worst {names[i]} rel {rel[i]:.3e}, "
                  f"gate {W_GATE:.0e}")
            ok = ok and rel.max() <= W_GATE

    print(f"bend: {wall:.1f} s wall for {STEPS} steps "
          f"({wall / STEPS:.1f} s per step, 128 positions)")
    print("MAMBA3 TINY OPT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
