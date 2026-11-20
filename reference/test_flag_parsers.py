"""Unknown flags must fail loudly in every trainer's parser.

  reference/.venv/bin/python reference/test_flag_parsers.py

A stale binary used to accept --four silently: the wildcard in run_go just
ignored any unrecognised flag, so a mistyped or unsupported flag looked like
a successful run. Each parser now returns IO(Run) and dies on an unknown
flag. This gate guards the source shape of that fix; the built binaries were
checked by hand with --bogus (exit 1, "unknown flag: --bogus").
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = (
    "examples/diffusion/unet/train.bend",
    "examples/language/glm53_flash/train_h128.bend",
    "examples/language/mamba3/train_mamba3_tiny.bend",
)


def main():
    for rel in FILES:
        src = open(os.path.join(ROOT, rel)).read()
        assert "def run_go(args: List<String>, r: Run) -> IO(Run):" in src, rel
        assert "def run_of(args: List<String>) -> IO(Run):" in src, rel
        assert 'IO.die(Run, 1, String.append("unknown flag: ", a))' in src, rel
        old = "case _:\n          run_go(rest, r)"
        assert old not in src, f"the tolerant wildcard is back in {rel}"
        # main binds the parsed Run from the IO action
        assert "r : Run <- run_of(args)" in src, rel
    print("FLAG PARSERS PASS")


if __name__ == "__main__":
    main()
