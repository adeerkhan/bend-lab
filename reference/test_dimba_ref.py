"""Dimba hybrid record, comparison and report are consistent and bannered."""
from __future__ import annotations
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results", "dimba")
def main():
    d = json.load(open(os.path.join(HERE, "hybrid/run.json")))
    j = json.load(open(os.path.join(ROOT, "results/hit-v4/jit/run.json")))
    assert d["arm"] == "dimba-hybrid" and d["steps"] == 1200
    assert d["mamba_blocks"] == 18 and d["gap"] == 1
    assert len(d["losses"]) == 1200
    assert d["legibility_held"] > j["legibility_held"]
    for f in ("comparison.png", "hybrid/model.pt", "hybrid/draws.npy", "COMPARISON.md"):
        assert os.path.getsize(os.path.join(HERE, f)) > 100, f
    rep = open(os.path.join(HERE, "COMPARISON.md")).read()
    for s in ("not** the released Dimba", "No FID", "no VAE, T5", "ignored by git"):
        assert s in rep, s
    print("DIMBA REF PASS")
if __name__ == "__main__":
    main()
