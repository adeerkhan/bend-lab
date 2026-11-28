#!/usr/bin/env python3
"""Fill `reference/.cache/datasets` from the task registry's pinned URLs.

The clean runner starts with no `reference/.cache/datasets`, so the CIFAR,
optdigits and tabular gates fail while the Bend side reads zeros. The task
registry in `experiment.py` already carries the exact URL and SHA-256; this
script downloads each unique file and verifies a single-file pin.

  reference/.venv/bin/python reference/materialize_datasets.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import experiment


def main() -> None:
    wanted: dict[str, str | None] = {}
    for task in experiment.TASKS.values():
        dataset = task.get("dataset", {})
        files = dataset.get("files", [])
        expected = dataset.get("expected_sha256")
        for url in files:
            path = experiment.cache_file(url)
            if path is None:
                continue
            wanted[url] = expected if len(files) == 1 else wanted.get(url)

    wrote = reused = 0
    for url, expected in sorted(wanted.items()):
        path = experiment.cache_file(url)
        if path is None:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):
            data = open(path, "rb").read()
            if expected is None or hashlib.sha256(data).hexdigest() == expected:
                reused += 1
                continue
        data = urllib.request.urlopen(url, timeout=300).read()
        got = hashlib.sha256(data).hexdigest()
        if expected is not None and got != expected:
            raise SystemExit(f"{path}: sha256 {got} != pinned {expected}")
        with open(path, "wb") as f:
            f.write(data)
        print(f"wrote {os.path.relpath(path)} ({len(data)} bytes, sha256 {got})")
        wrote += 1

    print(f"datasets: {wrote} written, {reused} already pinned")


if __name__ == "__main__":
    main()
