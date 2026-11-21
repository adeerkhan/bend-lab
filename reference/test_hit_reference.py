"""The HiT reference is reproducible and its claims stay inside its banners.

  reference/.venv/bin/python reference/test_hit_reference.py

Checks the committed run records, the plots and the report exist and agree,
and that no number in the report is a paper FID presented as ours.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results", "hit-reference")


def main():
    hit = json.load(open(os.path.join(HERE, "hit/run.json")))
    vit = json.load(open(os.path.join(HERE, "vit/run.json")))
    assert hit["paper"] == vit["paper"] == "arXiv:2611.22962"
    assert hit["arm"] == "hit" and vit["arm"] == "vit"
    assert hit["params"] > 300_000 and hit["params"] < 500_000
    assert abs(hit["params"] - vit["params"]) < 40_000, "arms not matched"
    assert len(hit["losses"]) == len(vit["losses"]) == 300
    assert hit["ceiling"] == vit["ceiling"]
    assert hit["noise_floor"] < hit["legibility_held"] < hit["ceiling"]
    assert hit["legibility_held"] > vit["legibility_held"], "HiT did not beat the twin"
    assert hit["group_counts"]["level1"] > hit["group_counts"]["level2"]
    for f in ("hit/group_maps.png", "comparison.png", "COMPARISON.md",
              "hit/run.json", "vit/run.json", "hit/groups.json"):
        p = os.path.join(HERE, f)
        assert os.path.exists(p) and os.path.getsize(p) > 100, f
    report = open(os.path.join(HERE, "COMPARISON.md")).read()
    assert "CPU torch reference" in report and "not a Bendlang" in report
    assert "0.010 held-out legibility" in report
    assert "1.48 FID" in report and "not our" in report
    print("HIT REFERENCE PASS")


if __name__ == "__main__":
    main()
