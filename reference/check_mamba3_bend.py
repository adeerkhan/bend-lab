"""Run every Bend Mamba-3 gate against the twin's goldens.

  reference/.venv-rf/bin/python reference/check_mamba3_bend.py

Forward: 1e-5 relative at seqlen 8, and at seqlen 4 and 16.
Backward: 1e-4 relative, every parameter and retained intermediate.
Can-fail: one moved dy must change du and match the moved-dy golden.
Training: ten copy-task losses at 1e-4 and the same final tokens.
Generation: two weight-tied layers, sixteen greedy tokens, exact.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "reference/mamba3_golden.npz")
DATA = os.path.join(ROOT, "reference/mamba3_data")
LM_GOLDEN = os.path.join(ROOT, "reference/mamba3_lm_golden.npz")
LM_DATA = os.path.join(ROOT, "reference/mamba3_lm_data")
FORWARD_GATE = 1e-5
BACKWARD_GATE = 1e-4
TRAIN_GATE = 1e-4

FWD_TAGS = ["pre", "A", "DT", "ADT", "trap", "gamma", "shift", "scale",
            "Bn", "Cn", "angle", "q", "k", "qk", "states", "out_v",
            "y_inner", "y"]

GRAD_TAGS = [("du", "du"), ("dW_in", "dW_in"), ("ddt_bias", "ddt_bias"),
             ("dB_bias", "dB_bias"), ("dC_bias", "dC_bias"),
             ("dW_norm_w", "dB_norm_w"), ("dC_norm_w", "dC_norm_w"),
             ("dD", "dD"), ("dW_out", "dW_out"), ("do_v", "g_out_v"),
             ("dz", "g_z"), ("dx", "g_x"), ("dqk", "g_qk"),
             ("dscale", "g_scale"), ("dADT", "g_ADT"), ("dBn", "g_Bn"),
             ("dCn", "g_Cn"), ("dDT", "g_DT"), ("dtrap", "g_trap_raw"),
             ("ddd_dt", "g_dd_dt"), ("ddd_A", "g_dd_A"),
             ("dBraw", "g_Braw"), ("dCraw", "g_Craw")]


def run_bend(program):
    p = subprocess.run(["bend", program], cwd=ROOT, capture_output=True,
                       text=True, timeout=3600)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit(f"bend failed: {program}")
    return p.stdout


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


def compare(name, got, want, gate):
    want = np.asarray(want, dtype=np.float32).reshape(-1)
    if got.size != want.size:
        print(f"FAIL {name}: size {got.size} != {want.size}")
        return False
    d = np.abs(got - want)
    rel = float(d.max()) / max(1e-30, float(np.abs(want).max()))
    ok = rel <= gate
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max|d| {d.max():.6e} "
          f"rel {rel:.6e} gate {gate:.0e}")
    return ok


def exact(name, got, want):
    want = np.asarray(want).reshape(-1)
    got = np.asarray(got).reshape(-1)
    ok = got.size == want.size and np.array_equal(got, want)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got.astype(int).tolist()}")
    if not ok:
        print(f"     golden: {want.astype(int).tolist()}")
    return ok


def main():
    if not os.path.exists(GOLDEN):
        raise SystemExit(f"missing {GOLDEN}; run reference/mamba3_twin.py")
    golden = np.load(GOLDEN)
    subprocess.run([sys.executable, os.path.join(ROOT, "reference/mamba3_bins.py"),
                    GOLDEN, DATA], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "reference/mamba3_bins.py"),
                    LM_GOLDEN, LM_DATA], check=True)

    ok = True
    tags = parse(run_bend("examples/language/mamba3/mamba3_siso.bend"))
    for tag in FWD_TAGS:
        if tag not in tags:
            print(f"FAIL {tag}: not printed")
            ok = False
            continue
        ok = compare(tag, tags[tag], golden[tag], FORWARD_GATE) and ok
    print("FORWARD-8", "PASS" if ok else "FAIL")

    ntags = parse(run_bend("examples/language/mamba3/probe_mamba3_n.bend"))
    nok = compare("y4", ntags["y4"], golden["y4"], FORWARD_GATE)
    nok = compare("y16", ntags["y16"], golden["y16"], FORWARD_GATE) and nok
    print("FORWARD-N", "PASS" if nok else "FAIL")
    ok = ok and nok

    gtags = parse(run_bend("examples/language/mamba3/probe_mamba3_grad.bend"))
    gok = True
    for tag, key in GRAD_TAGS:
        if tag not in gtags:
            print(f"FAIL {tag}: not printed")
            gok = False
            continue
        gok = compare(tag, gtags[tag], golden[key], BACKWARD_GATE) and gok
    gok = compare("dang_raw", gtags.get("dang_raw"),
                  golden["g_ang_raw"].reshape(8, 4).sum(0), BACKWARD_GATE) and gok
    print("BACKWARD", "PASS" if gok else "FAIL")
    ok = ok and gok

    btags = parse(run_bend("examples/language/mamba3/probe_mamba3_bad.bend"))
    cok = compare("du_bad", btags["du"], golden["du_bad"], BACKWARD_GATE)
    good = gtags["du"]
    bad = golden["du_bad"].reshape(-1)
    differs = (np.abs(good - bad).max()
               / max(1e-30, float(np.abs(bad).max())) > BACKWARD_GATE)
    print(f"{'ok  ' if differs else 'FAIL'} can-fail: the moved dy changes du")
    cok = cok and differs
    print("CAN-FAIL", "PASS" if cok else "FAIL")
    ok = ok and cok

    t0 = time.monotonic()
    ttags = parse(run_bend("examples/language/mamba3/probe_mamba3_train.bend"))
    seconds = time.monotonic() - t0
    tok = compare("train_loss", ttags["train_loss"], golden["train_loss"],
                  TRAIN_GATE)
    tok = exact("train_tokens", ttags["train_tokens"],
                golden["train_tokens"]) and tok
    print(f"     train probe: {seconds:.2f} s wall")
    print("TRAIN", "PASS" if tok else "FAIL")
    ok = ok and tok

    gntags = parse(run_bend("examples/language/mamba3/probe_mamba3_gen.bend"))
    gnok = exact("gen_tokens", gntags["gen_tokens"], golden["gen_tokens"])
    if not gnok:
        a = gntags["gen_tokens"].astype(int)
        w = golden["gen_tokens"].astype(int).reshape(-1)
        i = int(np.argmax(a != w))
        print(f"     first differing position {i}: bend {a[i]} golden {w[i]}")
    print("GENERATION", "PASS" if gnok else "FAIL")
    ok = ok and gnok

    lmg = np.load(LM_GOLDEN)
    lmt = parse(run_bend("examples/language/mamba3/probe_mamba3_lm.bend"))
    lmok = compare("lm_hn", lmt["lm_hn"], lmg["lm_hn"], FORWARD_GATE)
    lmok = compare("lm_logits", lmt["lm_logits"], lmg["lm_logits"],
                   FORWARD_GATE) and lmok
    lmok = compare("lm_loss", lmt["lm_loss"], lmg["lm_loss"],
                   FORWARD_GATE) and lmok
    lmok = compare("lm_train_loss", lmt["lm_train_loss"],
                   lmg["lm_train_loss"], TRAIN_GATE) and lmok
    lmok = exact("lm_train_tokens", lmt["lm_train_tokens"],
                 lmg["lm_train_tokens"]) and lmok
    lmok = exact("lm_gen_tokens", lmt["lm_gen_tokens"],
                 lmg["lm_gen_tokens"]) and lmok
    print("LM", "PASS" if lmok else "FAIL")
    ok = ok and lmok

    print("MAMBA3", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
