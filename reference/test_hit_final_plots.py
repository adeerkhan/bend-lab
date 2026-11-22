"""Final v4 plots exist, use common noise, and carry no ceiling/spectrum."""
from __future__ import annotations
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results", "hit-v4")
def main():
    for f in ("samples_3x3.png", "overview.png", "convergence.png", "draws_common.npy"):
        assert os.path.getsize(os.path.join(HERE, f)) > 1000, f
    src = open(os.path.join(ROOT, "reference", "hit_plot_final.py")).read()
    assert "draws_common.npy" in src and "manual_seed(20261122)" in src
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#") and "No ceiling" not in l)
    assert "ceiling" not in code and "spectrum" not in code, "ceiling/spectrum still plotted"
    print("HIT FINAL PLOTS PASS")
if __name__ == "__main__":
    main()
