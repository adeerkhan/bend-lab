#!/usr/bin/env python3
"""The Bendlang DiT denoiser trains on the pinned optdigits CSV.

  reference/.venv/bin/python reference/test_dit_real.py

Builds `examples/diffusion/dit/real_main.bend` under the memory guard and checks:
  - the local file loader read 65536 values (1024 images x 64 pixels),
  - the first row's first 8 scaled pixels match the pinned CSV (a data check,
    so a wrong loader cannot pass by producing plausible losses),
  - six AdamW steps run and the final loss prints.

The loss is stochastic (a different timestep/chunk per step), so its exact
value is not gated — only that training runs on verified real data.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
GUARD = os.path.join(ROOT, "scripts", "bend-guard.sh")
BINARY = os.path.join(ROOT, ".scratch", "dit_real")
CSV = os.path.join(ROOT, "reference", ".cache", "datasets", "mstz", "optdigits", "optdigits.data")


def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"test_dit_real: pinned CSV missing at {CSV}")
    row = open(CSV).readline().strip().split(",")[:8]
    expect_sum = sum(int(x) / 16 for x in row)

    env = dict(os.environ, BEND_NO_TELEMETRY="1", BEND_LIB=ENV_DIR,
               LD_LIBRARY_PATH=(os.path.expanduser("~/.local/lib/llvm22") + ":" +
                                os.environ.get("LD_LIBRARY_PATH", "")),
               BEND_MEM=os.environ.get("BEND_MEM", "2500M"))
    build = subprocess.run([GUARD, BEND, os.path.join(ROOT, "examples", "diffusion", "dit", "real_main.bend"),
                            "-o", BINARY], env=env, capture_output=True, text=True, timeout=900)
    if build.returncode != 0:
        raise SystemExit(f"test_dit_real: build failed\n{build.stdout[-600:]}{build.stderr[-600:]}")
    run = subprocess.run([BINARY], capture_output=True, text=True, timeout=300)
    out = run.stdout
    assert "data_len: 65536" in out, out
    m = re.search(r"data_sum8: ([0-9.eE+-]+)", out)
    assert m and abs(float(m.group(1)) - expect_sum) < 1e-4, (m, expect_sum)
    losses = [float(x) for x in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M) if float(x) > 0]
    assert len(losses) == 5, losses
    final = re.search(r"final_loss: ([0-9.eE+-]+)", out)
    assert final, out
    print(f"dit-real: 65536 values from the pinned optdigits CSV; "
          f"first-8 sum {m.group(1)} == {expect_sum:.4f}; "
          f"6 AdamW steps, final_loss {final.group(1)}; PASS")


if __name__ == "__main__":
    main()
