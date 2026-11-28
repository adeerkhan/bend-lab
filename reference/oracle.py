"""The comparison policy shared by the DiT probes.

The upstream BendTorch checkout carries the same policy. Keeping the two
numbers here means a clean bend-lab checkout does not need the sibling
`../bendtorch` directory to run its own gates.

  |bend - torch| <= ATOL + RTOL * |torch|, element by element, float64.
"""
from __future__ import annotations

import subprocess

import numpy as np

ATOL = 1e-6
RTOL = 1e-5


def parse(out: str) -> dict[str, np.ndarray]:
    """One tag per line, each following word the decimal of 32 F32 bits."""
    got: dict[str, list[np.ndarray]] = {}
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        words = np.array([int(w) for w in parts[1:]], dtype=np.uint32)
        got.setdefault(parts[0], []).append(words.view(np.float32))
    return {k: (v[0] if len(v) == 1 else np.stack(v)) for k, v in got.items()}


def run(binary: str, *args: str) -> dict[str, np.ndarray]:
    out = subprocess.run([binary, *args], check=True, capture_output=True, text=True).stdout
    return parse(out)


def close(name: str, bend: np.ndarray, ref, failures: list[str],
          atol: float = ATOL, rtol: float = RTOL) -> float:
    r = ref.detach().cpu().double().reshape(-1).numpy()
    b = np.asarray(bend, dtype=np.float64).reshape(-1)
    if b.shape != r.shape:
        failures.append(f"{name}: {b.size} numbers, expected {r.size}")
        return float("inf")
    same = (np.isnan(b) & np.isnan(r)) | (b == r)
    with np.errstate(invalid="ignore"):
        err = np.where(same, 0.0, np.abs(b - r) / (atol + rtol * np.abs(r)))
    err = np.where(np.isnan(err), np.inf, err)
    worst = float(err.max()) if err.size else 0.0
    if worst > 1.0:
        i = int(err.argmax())
        failures.append(f"{name}[{i}]: bend {b[i]!r}, torch {r[i]!r} ({worst:.2f}x the tolerance)")
    return worst


def bits_equal(bend: np.ndarray, ref) -> tuple[int, int]:
    r = ref.detach().cpu().float().reshape(-1).numpy().view(np.uint32)
    b = np.asarray(bend, dtype=np.float32).reshape(-1).view(np.uint32)
    return int((r == b).sum()), int(r.size)
