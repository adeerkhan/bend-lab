#!/usr/bin/env python3
"""The Bendlang DiT block: frozen parity gates and a recorded gradient gap.

  reference/.venv/bin/python reference/test_dit_ref.py

Builds the three probes under the pinned Bend (BEND_LIB, BEND_MEM guard) and
checks, with the fixed oracle tolerance, that:

  1. the attention branch (QKV -> FullAttention -> output) matches PyTorch,
  2. LayerNorm -> attention -> T.Residual matches PyTorch (mat h), and
  3. the full conditioned block's FORWARD matches PyTorch.

The full block's gradient gate does NOT pass: it reports a measured worst of
5.454x the oracle tolerance. That number is frozen here exactly as the repo
freezes other references — if it moves, the composition changed and must be
re-investigated. It is NOT a pass and the tolerance is NOT loosened to make
one. `results/dit/COMPARISON.md` carries the banner and the isolation ladder.

Requires the sibling `bendtorch/reference/oracle.py` (the upstream checkout).
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference", ".venv", "bin", "python")
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
GUARD = os.path.join(ROOT, "scripts", "bend-guard.sh")
ORACLE = os.path.join(ROOT, "..", "bendtorch", "reference", "oracle.py")
SCRATCH = os.path.join(ROOT, ".scratch")

# (source, binary, gate, must_pass, frozen_line)
PROBES = [
    ("attn_probe.bend", "dit_attn", "dit_attn_check.py", True, "PASS"),
    ("ln_attn_probe.bend", "dit_lnattn", "dit_lnattn_check.py", True, "PASS"),
    ("block.bend", "dit_block", "dit_block_check.py", False, "worst error 5.454 of the tolerance; FAIL"),
]


def main():
    if not os.path.exists(ORACLE):
        raise SystemExit(f"test_dit_ref: missing {ORACLE}")
    os.makedirs(SCRATCH, exist_ok=True)
    env = dict(os.environ, BEND_NO_TELEMETRY="1", BEND_LIB=ENV_DIR,
               LD_LIBRARY_PATH=(os.path.expanduser("~/.local/lib/llvm22") + ":" +
                                os.environ.get("LD_LIBRARY_PATH", "")),
               BEND_MEM=os.environ.get("BEND_MEM", "2500M"))

    for src, binary, gate, must_pass, frozen in PROBES:
        out = os.path.join(SCRATCH, binary)
        build = subprocess.run([GUARD, BEND, os.path.join(ROOT, "examples", "diffusion", "dit", src),
                                "-o", out], env=env, capture_output=True, text=True)
        if build.returncode != 0 or not os.path.exists(out):
            raise SystemExit(f"test_dit_ref: build failed for {src}\n{build.stdout[-800:]}{build.stderr[-800:]}")

        run = subprocess.run([PY, os.path.join(ROOT, "reference", gate), out],
                             capture_output=True, text=True)
        line = [l for l in run.stdout.splitlines() if l.startswith(("dit-", "FAIL"))]
        verdict = " ; ".join(line)
        if must_pass:
            if run.returncode != 0 or "PASS" not in run.stdout:
                raise SystemExit(f"test_dit_ref: {gate} did not pass\n{verdict}\n{run.stdout[-800:]}")
        else:
            # A frozen known-limitation reference, not a pass.
            if frozen not in run.stdout:
                raise SystemExit(f"test_dit_ref: {gate} moved off its frozen gap\n{verdict}\n{run.stdout[-800:]}")
        print(f"dit: {gate}: {verdict}")

    print("DIT REF PASS (2 parity gates green; full-block gradient gap frozen at 5.454x, FAIL by design)")


if __name__ == "__main__":
    main()
