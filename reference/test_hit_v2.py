"""Three-arm HiT comparison is recorded and its banners are intact."""
from __future__ import annotations
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results", "hit-v2")

def main():
    r = {a: json.load(open(os.path.join(HERE, a, "run.json")))
         for a in ("vit", "jit", "hit")}
    assert all(r[a]["steps"] == 1200 and r[a]["batch"] == 16 for a in r)
    assert all(r[a]["paper"].startswith("arXiv:2611.22962") for a in r)
    assert r["hit"]["params"] > r["vit"]["params"] > 800_000
    assert r["hit"]["legibility_held"] > r["jit"]["legibility_held"] and r["hit"]["legibility_held"] > r["vit"]["legibility_held"]
    assert r["hit"]["noise_floor"] < r["hit"]["legibility_held"] < r["hit"]["ceiling"]
    assert all(len(r[a]["losses"]) == 1200 for a in r)
    rep = open(os.path.join(HERE, "COMPARISON.md")).read()
    for s in ("Not Bendlang", "No EMA", "threshold retune",
              "No FID", "1.5M params"):
        assert s in rep, s
    for f in ("comparison.png", "hit/model.pt", "jit/model.pt", "vit/model.pt"):
        assert os.path.getsize(os.path.join(HERE, f)) > 100
    print("HIT V2 PASS")

if __name__ == "__main__":
    main()
