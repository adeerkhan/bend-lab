"""Gate B3: the angle wrap, on a synthetic sequence that crosses 2pi.

  reference/.venv/bin/python reference/test_mamba3_wrap.py

The seeded golden input never crosses 2pi (its raw angle cumsum stays
within +/-0.22), so the wrap paths would otherwise be untested.  The
synthetic case fills dt_bias with 10.0 (DT mean 10.0): the raw cumsum
reaches |cumsum| ~ 44 rad, about seven 2pi crossings per head at
seqlen 8.

Three Bend paths run on the synthetic weights (all bins from
reference/mamba3_data copied into .scratch/mamba3_wrap_data with
dt_bias.bin and train_dt_bias.bin filled to 10.0):

  wrap_siso.bend  the exact loop:        angle and y tags vs the twin at 1e-5
  wrap_gen.bend   the full-prefix pass:  gen_tokens (two-layer LM, 16 tokens)
  wrap_step.bend  the exact step:        step_tokens (two-layer LM, 16 tokens)

and both token paths must equal the twin's greedy generate exactly.
The two wrap variants (per-step fmod + fix, as Bend implements; vs
wrap-after-cumsum, as the twin's `%` implements) agree at F32 by
construction in this case; the interesting check is Bend-vs-twin.

A skipped check would read like a pass; a mismatch is a FAIL with the
numbers, never a tolerance bump.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND", os.path.join(os.path.expanduser("~"), ".bend/bin/bend"))
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))
SCRATCH = os.path.join(ROOT, ".scratch")
DATA = os.path.join(ROOT, ".scratch", "mamba3_wrap_data")
PROBES = [("wrap_siso.bend", "mamba3_siso.bend"),
          ("wrap_gen.bend", "probe_mamba3_gen.bend"),
          ("wrap_step.bend", "step_mamba3.bend")]
NEED_CROSSINGS = 2


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


def rel(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.abs(a - b).max()) / max(1e-30, float(np.abs(b).max()))


def main():
    os.makedirs(DATA, exist_ok=True)
    for name in os.listdir(os.path.join(ROOT, "reference/mamba3_data")):
        src = os.path.join(ROOT, "reference/mamba3_data", name)
        if os.path.isfile(src):
            dst = os.path.join(DATA, name)
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
    for filled in ("dt_bias.bin", "train_dt_bias.bin"):
        dst = os.path.join(DATA, filled)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        np.full(4, 10.0, dtype="<f4").tofile(dst)

    # the probes import ./mamba3_siso.bend relatively, so the copies are
    # written next to their sources and removed at the end
    examples = os.path.join(ROOT, "examples/language/mamba3")
    probes = {}
    for dst_name, src_name in PROBES:
        src = open(os.path.join(examples, src_name)).read()
        assert '"reference/mamba3_data"' in src, src_name
        dst = os.path.join(examples, dst_name)
        with open(dst, "w") as f:
            f.write(src.replace('"reference/mamba3_data"',
                                '".scratch/mamba3_wrap_data"'))
        probes[src_name] = dst

    env = dict(os.environ)
    env["BEND_NO_TELEMETRY"] = "1"
    llvm = os.path.join(os.path.expanduser("~"), ".local/lib/llvm22")
    if os.path.isdir(llvm):
        env["LD_LIBRARY_PATH"] = llvm + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")

    twin_npz = os.path.join(SCRATCH, "mamba3_wrap_twin.npz")
    t0 = time.time()
    p = subprocess.run([PYRF, os.path.join(ROOT, "reference/mamba3_wrap.py"),
                        "--dump", twin_npz], cwd=ROOT, capture_output=True,
                       text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print("\n".join(l for l in out.splitlines()
                    if l.startswith(("dt_bias", "wrap", "2pi", "wrote"))), flush=True)
    if p.returncode != 0:
        print("\n".join(out.strip().splitlines()[-12:]))
        print("MAMBA3 WRAP FAIL")
        sys.exit(1)
    tw = np.load(twin_npz)
    crossings = [int(float(np.abs(tw["raw"][0, -1, h]).max()) / (2 * np.pi))
                 for h in range(4)]
    if min(crossings) < NEED_CROSSINGS:
        print(f"FAIL synthetic case crosses 2pi only {crossings} times per head; "
              f"need >= {NEED_CROSSINGS}")
        print("MAMBA3 WRAP FAIL")
        sys.exit(1)

    ok = True

    def cleanup():
        for path in probes.values():
            for victim in (path, os.path.join(SCRATCH, os.path.basename(path)[:-5] + ".bin")):
                try:
                    os.remove(victim)
                except OSError:
                    pass

    def run(label, path):
        bin_path = os.path.join(SCRATCH, os.path.basename(path)[:-5] + ".bin")
        b = subprocess.run([BEND, "-o", bin_path, path], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        if b.returncode != 0:
            print((b.stdout or "") + (b.stderr or ""))
            print(f"FAIL {label}: build failed")
            print("MAMBA3 WRAP FAIL")
            sys.exit(1)
        r = subprocess.run([bin_path], cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=3600)
        return parse(r.stdout)

    tags = run("exact loop", probes["mamba3_siso.bend"])
    r_angle = rel(tags["angle"], tw["angle"].reshape(-1))
    r_y = rel(tags["y"], tw["y"].reshape(-1))
    print(f"exact loop  angle vs twin max rel {r_angle:.3e}")
    print(f"exact loop  y      vs twin max rel {r_y:.3e}")
    ok = ok and r_angle < 1e-5 and r_y < 1e-5

    gen = run("full-prefix", probes["probe_mamba3_gen.bend"])
    step = run("step", probes["step_mamba3.bend"])
    tok_gen = gen.get("gen_tokens")
    tok_step = step.get("step_tokens")
    want = tw["tokens"].astype(np.float32)
    got_gen = tok_gen.astype(int).tolist() if tok_gen is not None else None
    got_step = tok_step.astype(int).tolist() if tok_step is not None else None
    want_list = want.astype(int).tolist()
    print(f"full-prefix tokens {got_gen}")
    print(f"exact step  tokens {got_step}")
    print(f"twin        tokens {want_list}")
    ok = ok and got_gen == want_list and got_step == want_list

    cleanup()
    if not ok:
        print("MAMBA3 WRAP FAIL")
        sys.exit(1)
    print(f"elapsed {time.time() - t0:.1f}s")
    print("MAMBA3 WRAP PASS")


if __name__ == "__main__":
    main()
