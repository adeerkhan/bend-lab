#!/usr/bin/env python3
"""Materialize the four MNIST gz files used by the MambaViT training gate.

The gate reads `/tmp/opencode/mnist` through `torchvision.datasets.MNIST`. A
clean runner starts with no cache, so setup fetches the canonical files and
checks their SHA-256 before torchvision extracts them.

  reference/.venv/bin/python reference/materialize_mnist.py
"""
from __future__ import annotations

import hashlib
import os
import urllib.request

OUT = "/tmp/opencode/mnist/MNIST/raw"
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    wrote = reused = 0
    for name, want in FILES.items():
        path = os.path.join(OUT, name)
        if os.path.isfile(path) and digest(open(path, "rb").read()) == want:
            reused += 1
        else:
            data = urllib.request.urlopen(BASE + name, timeout=300).read()
            got = digest(data)
            if got != want:
                raise SystemExit(f"{path}: sha256 {got} != pinned {want}")
            with open(path, "wb") as f:
                f.write(data)
            print(f"wrote {path} ({len(data)} bytes, sha256 {got})")
            wrote += 1
        # Remove any stale extraction; torchvision will extract the verified gz.
        plain = path[:-3]
        if os.path.exists(plain):
            os.remove(plain)
    print(f"mnist: {wrote} written, {reused} already pinned")


if __name__ == "__main__":
    main()
