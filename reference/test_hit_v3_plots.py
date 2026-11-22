"""The three v3 comparison figures exist and match committed run records."""
from __future__ import annotations
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results", "hit-v3")

def main():
    for f in ("overview.png", "convergence.png", "efficiency.png", "samples.png", "distributions.png"):
        p = os.path.join(HERE, f)
        assert os.path.getsize(p) > 10_000, f
        with open(p, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n", f
    r = {d: json.load(open(os.path.join(HERE, d, "run.json")))
         for d in ("vit", "jit", "hit", "jit-long", "hit-long")}
    assert r["jit"]["legibility_test"] > r["vit"]["legibility_test"] > r["hit"]["legibility_test"]
    assert r["jit-long"]["legibility_test"] > r["hit-long"]["legibility_test"]
    rep = open(os.path.join(HERE, "COMPARISON.md")).read()
    assert "6.62M" in rep and "6.53M" in rep and "4.74M" in rep
    assert "JiT/DiT > ViT >" in rep and "HiT" in rep
    assert "none of the three arms produces flower-like images" in rep
    print("HIT V3 PLOTS PASS")

if __name__ == "__main__":
    main()
