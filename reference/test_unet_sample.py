"""The sampling-only path is a real, gated inference task.

  reference/.venv/bin/python reference/test_unet_sample.py

A checkpoint resume with --steps 0 has no training loss lines. The results
parser must still accept its draws, and the Make target must run inference
only, never a training step.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import unet_flowers_results as results  # noqa: E402


def main():
    text = ("images: 2\nsteps: 0\nparams: 298439\nfinal_loss: 0\n"
            + "sample: " + " ".join(["0.0"] * 4096) + "\n")
    conf, losses, final, draws = results.parse(text)
    assert conf["steps"] == 0, conf
    assert losses == [], losses
    assert final == 0.0, final
    assert draws.shape == (1, 64, 64), draws.shape

    script = open(os.path.join(ROOT, "scripts", "unet_sample.sh")).read()
    assert "--steps 0" in script
    assert "--resume" in script
    assert "test_unet_flowers_metric.py" in script
    makefile = open(os.path.join(ROOT, "Makefile")).read()
    assert "unet-sample:" in makefile
    assert "unet-sample" in makefile.split(".PHONY:")[1].split("\n")[0]
    print("UNET SAMPLE PASS")


if __name__ == "__main__":
    main()
