"""Gate C1: the decode step == the full-prefix pass, against the config twin.

  reference/.venv-rf/bin/python reference/test_mamba3_step_varlen.py

Generates the two-layer siso LM dump and the twin's golden greedy decode
(`reference/mamba3_cfg_lm.py --config siso --mode {{dump,gen}} --pack varlen`),
builds `examples/language/mamba3/step_mamba3_varlen.bend` once, runs it once,
and checks that the step's greedy tokens equal the full-prefix generator's
tokens, exactly, at 16 and 64 tokens, and that both equal the twin's golden:

  varlen_step16 == varlen_ref16 == the twin's golden at 16
  varlen_step64 == varlen_ref64 == the twin's golden at 64

The two Bend paths are independent implementations (carried state vs
recompute-from-scratch), so exact token equality is the check that can fail;
the twin golden guards against both failing the same way. A missing tag or a
mismatch is FAIL, never a zero or an estimate.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))
GOLDEN = os.path.join(ROOT, ".scratch", "siso_lm_data", "gen{len}.json")
SRC = os.path.join(ROOT, "examples/language/mamba3/step_mamba3_varlen.bend")
BIN = os.path.join(ROOT, ".scratch", "mamba3_step_varlen.bin")
CFG = "siso"
PACK = 'varlen'


def main():
    ok = True
    env = dict(os.environ, BEND_NO_TELEMETRY="1")
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")

    cfg = [PYRF, os.path.join(ROOT, "reference/mamba3_cfg_lm.py"),
           "--config", CFG]
    gen = cfg + ["--mode", "gen"]
    subprocess.run(cfg + ["--mode", "dump"] + (["--pack", PACK] if PACK else []),
                   check=True, cwd=ROOT, env=env, capture_output=True)
    for length in (16, 64):
        subprocess.run(gen + ["--length", str(length)] + (["--pack", PACK] if PACK else []),
                       check=True, cwd=ROOT, env=env, capture_output=True)

    subprocess.run([BEND, "-o", BIN, SRC], cwd=ROOT, env=env, check=True,
                   capture_output=True, timeout=1800)
    p = subprocess.run([BIN], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=3600)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("varlen step binary failed")
    tags = {}
    for line in p.stdout.splitlines():
        if ": " not in line:
            continue
        tag, rest = line.split(": ", 1)
        vals = rest.split()
        if vals and all(v.isdigit() for v in vals):
            tags[tag] = [int(v) for v in vals]

    for length in (16, 64):
        golden = json.load(open(GOLDEN.format(len=length)))["tokens"]
        for tag in (f"varlen_step{length}", f"varlen_ref{length}"):
            got = tags.get(tag)
            if got is None:
                print(f"FAIL {tag}: not printed")
                ok = False
                continue
            same = got == golden
            print(("ok  " if same else "FAIL") + f" {tag}: "
                  + (f"matches the twin's greedy decode at {length} tokens"
                     if same else f"{got} != twin {golden}"))
            ok = ok and same
    print("MAMBA3 STEP VARLEN", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
