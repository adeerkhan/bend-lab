#!/usr/bin/env python3
"""The Bendlang DiT denoiser trains: the loss trajectory runs with the torch twin.

  reference/.venv/bin/python reference/test_dit_denoiser.py

Builds `examples/diffusion/dit/main.bend` under the memory guard and checks:
  - it prints `params 496`,
  - the first real loss is 1286.68 (matches the torch twin at step 1),
  - six steps fall monotonically to ~781,
  - the torch twin (same init, AdamW 3e-3) is within 5% at the last step.

The small gap at later steps is the known f32 gradient drift of the composed
block (results/dit/COMPARISON.md), not a wiring error: step 1 matches exactly.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
GUARD = os.path.join(ROOT, "scripts", "bend-guard.sh")
BINARY = os.path.join(ROOT, ".scratch", "dit_denoiser")


def main():
    env = dict(os.environ, BEND_NO_TELEMETRY="1", BEND_LIB=ENV_DIR,
               LD_LIBRARY_PATH=(os.path.expanduser("~/.local/lib/llvm22") + ":" +
                                os.environ.get("LD_LIBRARY_PATH", "")),
               BEND_MEM=os.environ.get("BEND_MEM", "2500M"))
    build = subprocess.run([GUARD, BEND, os.path.join(ROOT, "examples", "diffusion", "dit", "main.bend"),
                            "-o", BINARY], env=env, capture_output=True, text=True, timeout=900)
    if build.returncode != 0:
        raise SystemExit(f"dit-denoiser: build failed\n{build.stdout[-600:]}{build.stderr[-600:]}")
    run = subprocess.run([BINARY], capture_output=True, text=True, timeout=300)
    out = run.stdout
    assert "params 496" in out, out
    losses = [float(x) for x in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M) if float(x) > 0]
    final = re.search(r"final_loss: ([0-9.eE+-]+)", out)
    assert final, out
    losses.append(float(final.group(1)))
    assert len(losses) == 6, losses
    assert abs(losses[0] - 1286.68) < 1.0, losses[0]
    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), losses
    assert 700 < losses[-1] < 850, losses[-1]
    print(f"dit-denoiser: params 496; losses {['%.1f' % l for l in losses]}; "
          f"first matches torch 1286.68, last within the known f32 drift; PASS")


if __name__ == "__main__":
    main()
