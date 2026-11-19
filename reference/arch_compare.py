#!/usr/bin/env python3
"""arch_compare.py — the 72k two-arm pilot: Mamba-3 and the GLM hybrid.

  reference/.venv/bin/python reference/arch_compare.py [--steps 20] [--matched]

Both arms read the same byte stream (corpus128-b32); the Mamba trainer's
`train_tokens.bin` and the corpus's `corpus128-train_batch.bin` are asserted
byte-identical over the first `steps * 128` bytes. The Mamba LM is 71,952
parameters, the GLM-5.3-Flash hybrid 74,144, both at 128 positions, batch 1.

This is a pilot, and the report says so: the arms differ in vocabulary (127
against 260, 133 classes the corpus never uses), optimizer and schedule
(Mamba AdamW 0.9/0.999, wd 0.01, clip 1.0 with warmup + cosine; GLM
CAdamW 0.9/0.95, wd 0.1, clip 0.05 at a constant LR), and held-out slice
(the Mamba's 4,096-byte split against the GLM's val batch). It validates the
wiring, the data alignment and the parsers, and it prints the changes a
matched 3,200-step run needs.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
DATA = os.path.join(ROOT, "examples/language/glm53_flash/data/corpus128-b32")
MAMBA_PROG = "examples/language/mamba3/train_mamba3_tiny.bend"
GLM_PROG = "examples/language/glm53_flash/train_h128.bend"
MAMBA_TOKENS = os.path.join(ROOT, ".scratch/mamba3_tiny/train_tokens.bin")
OUT = os.path.join(ROOT, "results/arch-72k-pilot.md")
OUT_JSON = os.path.join(ROOT, "results/arch-72k-pilot.json")
MATCHED_MD = os.path.join(ROOT, "results/arch-matched.md")
MATCHED_JSON = os.path.join(ROOT, "results/arch-matched.json")

ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV["PATH"] = os.pathsep.join([os.path.expanduser("~/.bend/bin"),
                               os.path.expanduser("~/.local/bin"),
                               ENV.get("PATH", "")])
llvm = os.path.expanduser("~/.local/lib/llvm22")
ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [llvm, ENV.get("LD_LIBRARY_PATH")]))
ENV["BEND_LIB"] = os.path.join(ROOT, ".bend/env")
ENV["MLBEND_DATA"] = os.path.join(ROOT, "reference/.cache/datasets")


def steps_arg():
    n = 20
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--steps" and i + 1 < len(argv):
            n = int(argv[i + 1])
    return n


def build(program, out):
    p = subprocess.run([BEND, program, "-o", out], cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit(f"build failed: {program}")


def tag_floats(out, tag):
    for line in out.splitlines():
        if line.startswith(tag + ":"):
            return [float(x) for x in line.split(":", 1)[1].split()]
    return []


def run_mamba(binary, steps):
    t0 = time.time()
    p = subprocess.run([binary, "--steps", str(steps), "--lr-ppm", "300"],
                       cwd=ROOT, env=ENV, capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit("mamba run failed")
    losses = tag_floats(p.stdout, "tiny_train_losses")
    bits = tag_floats(p.stdout, "tiny_bits_per_byte")
    acc = tag_floats(p.stdout, "tiny_heldout_acc")
    return {"losses": losses, "bits_per_byte": bits[0] if bits else None,
            "heldout_acc": acc[0] if acc else None, "wall": wall}


def run_glm(binary, steps, tmp, matched=False):
    t0 = time.time()
    cmd = [binary, "--steps", str(steps), "--lr-ppm", "300", "--no-val", "--no-save",
           "--trace", "--out", tmp]
    if matched:
        cmd.append("--matched")
    p = subprocess.run(cmd, cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit("glm run failed")
    losses = []
    for line in p.stdout.splitlines():
        m = re.search(r'"loss": ([0-9.eE+-]+)', line)
        if m:
            losses.append(float(m.group(1)))
    return {"losses": losses, "wall": wall}


def matched_arg():
    return "--matched" in sys.argv[1:]


def main():
    steps = steps_arg()
    matched = matched_arg()
    need = steps * 128
    corpus = open(os.path.join(DATA, "corpus128-train_batch.bin"), "rb").read(need)
    if not os.path.exists(MAMBA_TOKENS):
        raise SystemExit(f"no {MAMBA_TOKENS}: run make mamba3-tiny-opt once")
    mamba_tokens = open(MAMBA_TOKENS, "rb").read(need)
    if len(corpus) < need or len(mamba_tokens) < need:
        raise SystemExit(f"not enough tokens for {steps} steps "
                         f"(corpus {len(corpus)}, mamba {len(mamba_tokens)}, need {need})")
    aligned = corpus == mamba_tokens
    if not aligned:
        raise SystemExit("the two arms do not read the same bytes; refusing")

    tmp = tempfile.mkdtemp()
    mbin = os.path.join(tmp, "mamba")
    gbin = os.path.join(tmp, "glm")
    print(f"building two arms, {steps} steps each")
    build(MAMBA_PROG, mbin)
    build(GLM_PROG, gbin)
    print("running Mamba-3")
    load0 = os.getloadavg()[0]
    m = run_mamba(mbin, steps)
    m["load"] = [load0, os.getloadavg()[0]]
    print("running the GLM hybrid")
    load0 = os.getloadavg()[0]
    g = run_glm(gbin, steps, tmp, matched)
    g["load"] = [load0, os.getloadavg()[0]]

    banners = [
        "vocabulary: Mamba 127, GLM 260 (133 classes never used), so the "
        "two cross-entropies are not one measurement",
        "held-out: Mamba's 4,096-byte split, GLM's val batch, different slices",
    ]
    blockers = [
        "the GLM head is 260 wide; a 127-wide head or a restricted eval is "
        "still needed for a like-for-like cross-entropy",
    ]
    if matched:
        banners.insert(1, "gradient clipping: Mamba clips the global norm at "
                       "1.0; BendTorch's CAdamW has no clipping field, so the "
                       "GLM arm is unclipped")
    else:
        banners.insert(1, "optimizer: Mamba AdamW 0.9/0.999 wd 0.01 clip 1.0; "
                       "GLM CAdamW 0.9/0.95 wd 0.1 clip 0.05")
        banners.insert(2, "schedule: for --steps N both peak at 3e-4, but Mamba "
                       "warms up 5 steps and cosines to 3e-5 over N while the "
                       "GLM is constant")
        blockers.insert(0, "the GLM arm needs the Mamba optimizer (0.9/0.999, "
                        "wd 0.01) and the same warmup/cosine over the shared "
                        "budget; pass --matched to enable both")
    report = {
        "matched": matched,
        "steps": steps,
        "tokens_seen": steps * 128,
        "same_bytes": aligned,
        "mamba": {"params": 71952, **m},
        "glm": {"params": 74144, **g},
        "banners": banners,
        "blockers": blockers,
    }
    out_json, out_md = (MATCHED_JSON, MATCHED_MD) if matched else (OUT_JSON, OUT)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)
    with open(out_md, "w") as f:
        f.write("# The 72k two-arm pilot" if not matched
                else "# The 72k matched two-arm run")
        f.write("\n\n")
        cmd = f"reference/arch_compare.py --steps {steps}"
        if matched:
            cmd += " --matched"
        f.write(f"`{cmd}`; {steps * 128} "
                f"tokens per arm, the first {steps} windows of corpus128-b32, "
                f"byte-identical for both (checked).\n\n")
        f.write("| arm | params | steps | first loss | last loss | wall | load |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, a in (("Mamba-3", report["mamba"]), ("GLM hybrid", report["glm"])):
            ls = a["losses"]
            f.write(f"| {name} | {a['params']} | {len(ls)} | "
                    f"{ls[0]:.6f} | {ls[-1]:.6f} | {a['wall']:.1f} s | "
                    f"{a['load'][0]:.2f} to {a['load'][1]:.2f} |\n")
        f.write("\n## Banners\n\n")
        for b in report["banners"]:
            f.write(f"- {b}\n")
        f.write("\n## Remaining match gaps\n\n")
        for b in report["blockers"]:
            f.write(f"- {b}\n")
        if matched:
            f.write("\nOptimizer and schedule parity are on: the GLM arm uses "
                    "CAdamW 0.9/0.999, wd 0.01, c1 0.1, c2 0.001 and the Mamba "
                    "warmup/cosine over the same budget. Vocabulary and clipping "
                    "remain bannered; this is not yet a quality verdict.\n")
        else:
            f.write("\nThe pilot validates the wiring and the data alignment "
                    "only; no quality verdict is read from it.\n")
    print(f"wrote {out_md} and {out_json}")
    for name, a in (("Mamba-3", report["mamba"]), ("GLM hybrid", report["glm"])):
        ls = a["losses"]
        print(f"  {name:10s} params {a['params']}  steps {len(ls)}  "
              f"loss {ls[0]:.6f} -> {ls[-1]:.6f}  wall {a['wall']:.1f} s")


if __name__ == "__main__":
    main()
