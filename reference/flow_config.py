"""The 2D flow's configuration, read from the example's source.

`examples/diffusion/flow_matching/main.bend` declares its width, step count and Euler steps as `Nat` defs:

    def hidden() -> Nat:
      256n

The record, the NumPy twin, the torch comparison and the checkpoint sidecar all need the same numbers, and
hardcoding them is how the record described a 256-wide run as 128 wide (ISSUE-059) and how two gates kept
expecting 128-wide weights after the winner moved (ISSUE-062). One reader, four users.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRAM = os.path.join(ROOT, "examples/diffusion/flow_matching/main.bend")


def _read(name, default):
    """The `Nat` a `def <name>() -> Nat:` returns, or `default` when the source does not say."""
    src = open(PROGRAM).read()
    m = re.search(rf"^def {name}\(\) -> Nat:\n  (\d+)n$", src, re.M)
    return int(m.group(1)) if m else default


def hidden():
    return _read("hidden", 128)


def steps():
    return _read("steps", 6000)


def euler():
    return _read("euler", 32)
