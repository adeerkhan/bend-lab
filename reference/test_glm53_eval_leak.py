"""Evaluation leakage: reverse must be out of the holdout training stream and in the eval.

  reference/.venv/bin/python reference/test_glm53_eval_leak.py

The 0/8 reverse result is only meaningful if the reverse family is absent from
the pretraining stream and present, unchanged, in dev and confirm.  The test
regenerates the holdout stream, searches the decoded text for the family's
prompts, counts the frozen eval tasks, and checks the frozen manifest.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/glm53_flash"))
import glm53_dense_data as D  # noqa: E402
from glm53_flash import ByteTokenizer  # noqa: E402
from glm53_flash.tasks import FAMILIES, frozen_tasks, pretraining_text  # noqa: E402
import random  # noqa: E402

PROMPTS = ("Return text in reverse order.", "Reverse the string.")
MANIFEST = os.path.join(ROOT, "reference/glm53_corpus_holdout_reverse_eval_golden.json")


def main():
    tok = ByteTokenizer()
    def text_of(ids):
        return bytes([t - tok.byte_offset for t in ids if t >= tok.byte_offset]).decode("utf-8", errors="replace")
    hold = text_of(D.corpus_stream_holdout("reverse", examples=2000))
    full = text_of(D.corpus_stream(examples=2000))
    rev = []
    for index in range(2000):
        rng = random.Random(42 * 10_000_019 + index)
        family = FAMILIES[rng.randrange(len(FAMILIES))]
        if family.name == "reverse":
            rev.append(pretraining_text(index))
    ok = []
    absent = not any(text in hold for text in rev)
    print(f"{'ok  ' if absent else 'FAIL'} holdout stream has none of the {len(rev)} reverse tasks")
    ok.append(absent)
    present = any(text in full for text in rev)
    print(f"{'ok  ' if present else 'FAIL'} full stream has at least one reverse task (the search works)")
    ok.append(present)
    for split in ("dev", "confirm"):
        tasks = frozen_tasks(split, per_family=8)
        count = sum(1 for t in tasks if t.family == "reverse")
        good = count == 8
        print(f"{'ok  ' if good else 'FAIL'} {split} has {count}/8 reverse tasks")
        ok.append(good)
    manifest = json.load(open(MANIFEST))
    for split in ("dev", "confirm"):
        value = manifest["splits"][split]["by_family"]["reverse"]
        good = value["solved"] == 0 and value["tasks"] == 8
        print(f"{'ok  ' if good else 'FAIL'} {split} manifest reverse: {value['solved']}/8")
        ok.append(good)
    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
