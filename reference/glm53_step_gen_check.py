"""The four-block step decoder's token gate.

  reference/.venv/bin/python reference/glm53_step_gen_check.py <log>

Reads the `tokens:` line the step decoder printed and compares it with the
recorded full-prefix generator's 48 tokens for episode 0 of the dev split
(`results/glm53-hybrid192-ep4/eval-dev-ep4.json`). The step path must produce
the same tokens, in the same order.
"""
from __future__ import annotations

import json
import re
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: glm53_step_gen_check.py <log>")
    text = open(sys.argv[1]).read()
    m = re.search(r"tokens: (.*)", text)
    if not m:
        raise SystemExit("no `tokens:` line")
    step = [int(v) for v in m.group(1).split()]
    payload = json.load(open("results/glm53-hybrid192-ep4/eval-dev-ep4.json"))
    golden = payload["episodes"][0]["model_tokens"]
    if len(step) != len(golden):
        print(f"step {len(step)} tokens, golden {len(golden)}: GLM DECODE FAIL")
        return 1
    for i, (a, b) in enumerate(zip(step, golden)):
        if a != b:
            print(f"first differing token {i}: {a} != {b}: GLM DECODE FAIL")
            return 1
    print(f"{len(step)} tokens, identical to the full prefix: GLM DECODE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
