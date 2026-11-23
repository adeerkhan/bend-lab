#!/usr/bin/env python3
"""The full-width Bendlang DiT denoiser: builds and trains on real digits.

  reference/.venv/bin/python reference/test_dit_full.py

`examples/diffusion/dit/full_main.bend` scales the DiT to the ViT's config:
64 images x 16 patches = 1024 tokens, width 64, 4 heads of 16, condition dim 3.
It reads the pinned optdigits CSV directly and trains with the repo's standard
0.1-scaled init. Checks: data integrity (65536 values, first-8 sum matches the
CSV), a sensible initial loss (~1, not the 2e8 of unscaled init), and 6 steps.

This is a wiring-and-scale gate. It does NOT gate a legibility number: sampling
and the held-out metric are a separate stage.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
GUARD = os.path.join(ROOT, "scripts", "bend-guard.sh")
BINARY = os.path.join(ROOT, ".scratch", "dit_full")
CSV = os.path.join(ROOT, "reference", ".cache", "datasets", "mstz", "optdigits", "optdigits.data")


def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"test_dit_full: pinned CSV missing at {CSV}")
    expect_sum = sum(int(x) / 16 for x in open(CSV).readline().strip().split(",")[:8])

    env = dict(os.environ, BEND_NO_TELEMETRY="1", BEND_LIB=ENV_DIR,
               LD_LIBRARY_PATH=(os.path.expanduser("~/.local/lib/llvm22") + ":" +
                                os.environ.get("LD_LIBRARY_PATH", "")),
               BEND_MEM=os.environ.get("BEND_MEM", "2500M"))
    build = subprocess.run([GUARD, BEND, os.path.join(ROOT, "examples", "diffusion", "dit", "full_main.bend"),
                            "-o", BINARY], env=env, capture_output=True, text=True, timeout=900)
    if build.returncode != 0:
        raise SystemExit(f"test_dit_full: build failed\n{build.stdout[-600:]}{build.stderr[-600:]}")
    run = subprocess.run([BINARY], capture_output=True, text=True, timeout=300)
    out = run.stdout
    assert "data_len: 65536" in out, out
    m = re.search(r"data_sum8: ([0-9.eE+-]+)", out)
    assert m and abs(float(m.group(1)) - expect_sum) < 1e-4, (m, expect_sum)
    losses = [float(x) for x in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M) if float(x) > 0]
    assert len(losses) == 5, losses
    # the 0.1-scaled init gives a first real loss near 1; the unscaled init gave 2e8
    assert 0.05 < losses[0] < 10, losses
    final = re.search(r"final_loss: ([0-9.eE+-]+)", out)
    assert final and 0.05 < float(final.group(1)) < 10, out
    print(f"dit-full: 64x16 tokens, width 64; pinned CSV first-8 sum {m.group(1)} == {expect_sum:.4f}; "
          f"0.1-scaled first loss {losses[0]:.3f}; 6 steps, final {final.group(1)}; PASS "
          f"(legibility not gated)")


if __name__ == "__main__":
    main()
