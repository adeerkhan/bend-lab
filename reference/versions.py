"""One lane: the Bend pin, bend.toml, .bend/env and the imports, side by side.

  reference/.venv/bin/python reference/versions.py

The first version of this file exists because the four places disagreed:
`bend.toml` said BendTorch 1.1.1.0, `.bend/env` and the examples said 2.0.0.0,
and upstream was 2.1.0. A review run should start here, not at a training run.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ["bendtorch", "bendtorch-gpu", "bend-ml-tensor-array",
            "bend-ml-tensor-gpu", "bend-ml-tensor-fork"]


def bend_version():
    try:
        return subprocess.run(["bend", "version"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def toml_pins():
    with open(os.path.join(ROOT, "bend.toml"), "rb") as fh:
        data = tomllib.load(fh)
    pins = {}
    for key, value in data.get("dependencies", {}).items():
        if key in PACKAGES:
            pins[key] = value
    for key, value in data.get("workspace", {}).get("dependencies", {}).items():
        if key in PACKAGES:
            pins.setdefault(key, value)
    return pins


def env_pins():
    names = os.path.join(ROOT, ".bend", "env", "names")
    out = {p: [] for p in PACKAGES}
    if not os.path.isdir(names):
        return out
    for name in os.listdir(names):
        for pkg in PACKAGES:
            if name.startswith(pkg + "@"):
                out[pkg].append(name.split("@", 1)[1])
    return out


def import_pins():
    out = {p: set() for p in PACKAGES}
    pat = re.compile(r"^import\s+(" + "|".join(re.escape(p) for p in PACKAGES)
                     + r")@([^\s/]+)", re.M)
    for base, _, files in os.walk(ROOT):
        if "/.git" in base or "/.bend" in base or "/.scratch" in base:
            continue
        for name in files:
            if not name.endswith(".bend"):
                continue
            path = os.path.join(base, name)
            try:
                text = open(path, encoding="utf-8").read()
            except OSError:
                continue
            for m in pat.finditer(text):
                out[m.group(1)].add(m.group(2))
    return {p: sorted(v) for p, v in out.items()}


def main():
    pins = toml_pins()
    env = env_pins()
    imports = import_pins()
    print(f"Bend binary: {bend_version()}")
    print(f"{'package':24s} {'bend.toml':12s} {'.bend/env':30s} imports")
    for pkg in PACKAGES:
        envs = ", ".join(sorted(env.get(pkg, []))) or "-"
        imps = ", ".join(imports.get(pkg, [])) or "-"
        print(f"{pkg:24s} {pins.get(pkg, '-'):12s} {envs:30s} {imps}")
    print()
    print("A review lane is one row: bend.toml, .bend/env and every import "
          "agree, and the Bend binary is the pin in the Makefile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
