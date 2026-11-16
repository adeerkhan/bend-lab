"""Gate B1: finite differences for the scan's y-gradient at seqlen 4.

  reference/.venv/bin/python reference/test_mamba3_fd.py

L = sum(y * dy) with fixed dy; dL/du by autograd through the twin, and
by central finite differences (delta = 1e-2) on the bumped-input twin.
Catches a sign that happens to cancel in the loss.  rel < 1e-2 or FAIL.

The twin (and its `einops` import) lives in reference/.venv-rf, so the
math runs as a worker of `reference/.venv-rf/bin/python` on CPU F32;
the parent (run by check_all under .venv) only reports.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYRF = os.environ.get("PYRF", os.path.join(ROOT, "reference/.venv-rf/bin/python"))

DELTA = 1e-2
GATE = 1e-2


def worker():
    import torch
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
    import mamba3_twin as W

    torch.manual_seed(71)
    torch.set_num_threads(1)
    m = W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32,
                 ngroups=1, is_mimo=False).float()
    torch.manual_seed(123)
    u = (torch.randn(1, 4, 64) * 0.5)
    dy = torch.randn(1, 4, 64, generator=torch.Generator().manual_seed(9)) * 0.1

    u_ref = u.clone().requires_grad_(True)
    loss = lambda uu: (W.twin_forward(m, uu)["y"] * dy).sum()
    l0 = loss(u_ref)
    l0.backward()
    g_auto = u_ref.grad.detach().clone()

    g_fd = torch.zeros_like(u)
    for b in range(u.shape[0]):
        for t in range(u.shape[1]):
            for c in range(u.shape[2]):
                up = u.clone(); up[b, t, c] += DELTA
                dn = u.clone(); dn[b, t, c] -= DELTA
                g_fd[b, t, c] = (loss(up) - loss(dn)) / (2.0 * DELTA)

    d = (g_fd - g_auto).abs()
    denom = g_auto.abs().max().item()
    rel = d.max().item() / max(1e-30, denom)
    print(f"torch {torch.__version__} (cpu)")
    print(f"autograd max|g| {denom:.4f}")
    print(f"finite differences (delta {DELTA}) worst |d| {d.max().item():.4e}")
    print(f"finite differences worst rel {rel:.3e} (bound {GATE:.0e})")
    return 0 if rel < GATE else 1


def main():
    env = dict(os.environ, B1_WORKER="1")
    p = subprocess.run([PYRF, __file__], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    for line in out.splitlines():
        if line.startswith(("torch", "autograd", "finite")):
            print(line)
    if p.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-8:])
        print(tail)
        print("MAMBA3 FD FAIL")
        sys.exit(1)
    print("MAMBA3 FD PASS")


if __name__ == "__main__":
    if os.environ.get("B1_WORKER") == "1":
        sys.exit(worker())
    main()
