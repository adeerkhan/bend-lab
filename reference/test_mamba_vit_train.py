"""MambaViT A3 gate: training parity on a fixed MNIST subset, Bend vs the twin.

  reference/.venv/bin/python reference/test_mamba_vit_train.py

The run:
  reference/mamba_vit_train_data.py   the subset, the seed-0 init, the manifest
  reference/mamba_vit_twin_train.py   the twin (PyTorch, float64) on it
  examples/language/mamba_vit/train_mamba_vit.bend
                                      the A1b forward + the A2 backward + AdamW

and the checks:

  the manifest's sha256 hashes, and `w_init_flat.bin` == the 108 per-tensor
  files concatenated in `named_parameters()` order;
  the loss curve, every step, at 1e-3 relative;
  the train and held-out accuracy, within 2 points (<= 2 of 96, <= 2 of 64), the
  train one being the forward that consumed each batch, before its update;
  the final per-tensor sums of squares (108), at 1e-3 relative.

THE FULL-EPOCH NUMBER IS UNAVAILABLE ON BEND AT THIS BUDGET. The reference
README's "~90 % validation in a couple of epochs" and the twin's measured
**94.61 % after one 60,000-image epoch** need 1,875 steps of batch 32. The
measured Bend run on this lane (2026-11-15, `make mamba-vit-train`) is
1869.8 s wall for the 160 training + 256 held-out images (4.49 s per
image), so the fixed 2,048-image subset is on the order of hours and one
60,000-image epoch is on the order of days. This gate is parity on a 160-image
subset (3 steps of 32) with the same schedule on both sides, not a claim on
the full epoch: the accuracy below is what 5 steps at lr 1e-3 reach, and it
is quoted as such.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, ".scratch/mv_mnist")
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))
BIN = os.path.join(ROOT, ".scratch/mv_train.bin")
SRC = os.path.join(ROOT, "examples/language/mamba_vit/train_mamba_vit.bend")
LOSS_GATE = 1e-3
ACC_GATE = 0.02
W_GATE = 1e-3
BUDGET = ("full epoch: unavailable on Bend at this budget. The measured run on this "
          "lane (2026-11-15) is 1869.8 s wall for the old 160 train + 256 held-out config "
          "(4.49 s per image); the config came down to 96 train + 64 held-out to fit the 1800 s gate. The subset is on the order of "
          "hours and a 60,000-image epoch on the order of days. "
          "This gate is parity on the subset at the same schedule, not a claim on the "
          "full epoch: the twin's 94.61 % after one 60,000-image epoch has no Bend "
          "counterpart here.")


def env():
    e = dict(os.environ, BEND_NO_TELEMETRY="1")
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        e["LD_LIBRARY_PATH"] = llvm + (":" + e["LD_LIBRARY_PATH"]
                                       if e.get("LD_LIBRARY_PATH") else "")
    return e


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    subprocess.run([PYRF, os.path.join(ROOT, "reference/mamba_vit_train_data.py")],
                   check=True, cwd=ROOT, env=env())
    manifest = json.load(open(os.path.join(OUT, "manifest.json")))
    budget = manifest["budget"]

    for name in ("train_images", "train_labels", "test_images", "test_labels",
                 "w_init_flat"):
        want = manifest["files"][name]["sha256"]
        got = sha256(os.path.join(OUT, manifest["files"][name]["file"]))
        good = want == got
        print(f"{'ok  ' if good else 'FAIL'} {name}: sha256 {got[:16]}…")
        ok = ok and good
    flat = np.fromfile(os.path.join(OUT, "w_init_flat.bin"), dtype="<f4")
    cat = np.concatenate([np.fromfile(os.path.join(OUT, p["file"]), dtype="<f4")
                          for p in manifest["params"]])
    good = flat.size == manifest["param_count"] == cat.size and np.array_equal(flat, cat)
    print(f"{'ok  ' if good else 'FAIL'} w_init_flat: {flat.size} parameters == "
          f"the 108 per-tensor files, in named_parameters() order")
    ok = ok and good

    t0 = time.monotonic()
    subprocess.run([PYRF, os.path.join(ROOT, "reference/mamba_vit_twin_train.py")],
                   check=True, cwd=ROOT, env=env(), capture_output=True)
    twin = json.load(open(os.path.join(OUT, "twin.json")))
    print(f"twin: torch {twin['config']['torch']} {twin['config']['dtype']} "
          f"{twin['config']['device']}, {twin['config']['steps']} steps x "
          f"{twin['config']['batch']}, {time.monotonic() - t0:.1f} s")
    print(f"budget: {budget['train_used']} train images ({budget['steps']} steps of "
          f"{budget['batch']}), {budget['test_used']} held-out images")

    subprocess.run([BEND, "-o", BIN, SRC], cwd=ROOT, env=env(), check=True,
                   capture_output=True, timeout=3600)
    t0 = time.monotonic()
    p = subprocess.run([BIN], cwd=ROOT, env=env(), capture_output=True, text=True,
                       timeout=10800)
    wall = time.monotonic() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("mamba-vit-train binary failed")
    tags = parse(p.stdout)
    per = wall / (budget["train_used"] + budget["test_used"])
    print(f"bend: {wall:.1f} s wall for {budget['train_used']} train + "
          f"{budget['test_used']} held-out images ({per:.2f} s per image)")

    # the loss curve, every step
    want = np.asarray(twin["losses"], dtype=np.float64)
    got = tags.get("mv_train_loss")
    if got is None or got.size != want.size:
        print(f"FAIL mv_train_loss: {None if got is None else got.size} values, "
              f"the twin has {want.size}")
        ok = False
    else:
        got = got.astype(np.float64)
        rel = np.abs(got - want) / np.abs(want)
        worst = int(rel.argmax())
        print(f"{'ok  ' if rel.max() <= LOSS_GATE else 'FAIL'} loss curve: "
              f"{got.size} steps, worst step {worst} rel {rel[worst]:.3e} "
              f"(bend {got[worst]:.7f}, twin {want[worst]:.7f}), gate {LOSS_GATE:.0e}")
        for i in range(got.size):
            print(f"     step {i}: bend {got[i]:.7f}  twin {want[i]:.7f}  "
                  f"rel {rel[i]:.3e}")
        ok = ok and rel.max() <= LOSS_GATE
    gn = tags.get("mv_grad_norm")
    if gn is not None and "grad_norms" in twin:
        d = np.abs(gn.astype(np.float64) - np.asarray(twin["grad_norms"]))
        rel = d / np.abs(np.asarray(twin["grad_norms"]))
        print(f"{'ok  ' if rel.max() <= LOSS_GATE else 'FAIL'} grad norms: worst rel "
              f"{rel.max():.3e}, clipped at 1.0 every step "
              f"({int((gn > 1.0).sum())} of {gn.size})")
        ok = ok and rel.max() <= LOSS_GATE

    # the accuracies: the same decisions, or 2 points apart at most.  The train
    # one is the forward that consumed each batch, before its update -- the
    # twin's `train_hits_preupdate`, because Bend does not re-run the train set
    print(f"     (twin, for the record: post-update train acc {twin['train_acc']:.4f} "
          f"{int(round(twin['train_acc'] * budget['train_used']))}/{budget['train_used']}, "
          f"which Bend does not re-run)")
    for tag, key, total in (("mv_train_hits", "train_hits_preupdate", budget["train_used"]),
                            ("mv_test_hits", "test_hits", budget["test_used"])):
        g = tags.get(tag)
        if g is None:
            print(f"FAIL {tag}: not printed")
            ok = False
            continue
        got_h = int(round(float(g[0])))
        want_h = int(twin[key])
        frac = abs(got_h - want_h) / total
        good = frac <= ACC_GATE
        print(f"{'ok  ' if good else 'FAIL'} {key}: bend {got_h}/{total} "
              f"({got_h / total:.4f}), twin {want_h}/{total} "
              f"({want_h / total:.4f}), {abs(got_h - want_h)} apart, "
              f"tolerance {ACC_GATE:.0%}")
        ok = ok and good

    # the final weights: per-tensor sums of squares
    w = tags.get("mv_w_sumsq")
    names = [p["name"] for p in manifest["params"]]
    if w is None or w.size != len(names):
        print(f"FAIL mv_w_sumsq: {None if w is None else w.size} values, "
              f"{len(names)} parameters")
        ok = False
    else:
        wantw = np.asarray([twin["final_sumsq"][n] for n in names], dtype=np.float64)
        gotw = w.astype(np.float64)
        rel = np.abs(gotw - wantw) / np.maximum(np.abs(wantw), 1e-12)
        order = np.argsort(-rel)
        worst = int(order[0])
        print(f"{'ok  ' if rel.max() <= W_GATE else 'FAIL'} final weights: "
              f"{gotw.size} tensors, worst {names[worst]} rel {rel[worst]:.3e}, "
              f"gate {W_GATE:.0e}")
        for j in order[:5]:
            print(f"     {names[int(j)]:34s} bend {gotw[int(j)]:.6e}  "
                  f"twin {wantw[int(j)]:.6e}  rel {rel[int(j)]:.3e}")
        print(f"     ({int((rel > W_GATE).sum())} of {rel.size} above the gate)")
        ok = ok and rel.max() <= W_GATE

    print(BUDGET)
    print("MAMBA-VIT TRAIN", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
