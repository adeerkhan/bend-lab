"""v4 comparison gates: records, corrected result, plots, group maps."""
from __future__ import annotations
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results", "hit-v4")

def main():
    r = {a: json.load(open(os.path.join(HERE, a, "run.json"))) for a in ("vit", "jit", "hit")}
    assert all(r[a]["steps"] == 1200 and len(r[a]["losses"]) == 1200 for a in r)
    assert r["jit"]["legibility_test"] > r["hit"]["legibility_test"] > r["vit"]["legibility_test"]
    assert all(np_ok(v) for v in [x["final_loss"] for x in r.values()])
    for f in ("overview.png", "convergence.png", "efficiency.png", "samples.png",
              "distributions.png", "hit/group_maps.png", "hit/groups4.json"):
        p = os.path.join(HERE, f)
        assert os.path.getsize(p) > 1000, f
    g = json.load(open(os.path.join(HERE, "hit/groups4.json")))
    assert max(g["K2"]) < min(g["K1"]), f"stage 2 did not compress: {g['K1']} -> {g['K2']}"
    rep = open(os.path.join(HERE, "COMPARISON.md")).read()
    for s in ("no HiT or DiT model in Bendlang", "NaN", "own", "six corrections"):
        assert s in rep, s
    print("HIT V4 PASS")

def np_ok(v):
    return isinstance(v, (int, float)) and v == v and abs(v) < 1000

if __name__ == "__main__":
    main()
