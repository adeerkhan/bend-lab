#!/usr/bin/env python3
"""DiT block — adaLN conditioning (g, m, sh) + FullAttention f, against PyTorch.

  python3 reference/dit_block_check.py [binary]

The Bend side is `examples/diffusion/dit/block.bend`. b=2 images of 3 tokens,
width 4, one kv head, scale 0.5. y = h + Tile(g) * f(LN(h)*Tile(m)+Tile(sh)),
where f is QKV -> FullAttention -> output projection. All tensors f32;
compared with oracle.py's tolerance.
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle import close, parse  # noqa: E402

B, T, W, HD, EPS = 2, 3, 4, 4, 1e-5
N = B * T


def vals(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def vals2(n, start=0):
    return np.array([((i * i * 5 + i * 3) % 17 - 8) / 4 for i in range(start, start + n)], dtype=np.float32)


def t(a, r, c):
    return torch.tensor(a.reshape(r, c)).requires_grad_(True)


def lin(i, o, s):
    return t(vals(o * i, s), o, i), t(vals(o, s + 40), 1, o)


def tile(x):
    return x.repeat_interleave(3, dim=0)


def main():
    binary = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(HERE), ".scratch", "dit_block")
    out = subprocess.run([binary], check=True, capture_output=True, text=True)
    got = parse(out.stdout)

    cond, h = t(vals2(6, 23), 2, 3), t(vals2(24, 33), 6, 4)
    (wg, bg), (wm, bm), (ws, bs) = lin(3, 4, 7), lin(3, 4, 9), lin(3, 4, 11)
    lg, lb = t(vals(4, 61), 1, 4), t(vals(4, 67), 1, 4)
    # DiT attention f: q/k/v + out, each Linear(4,4) with seeds 3/7/11/15
    (wq, bq), (wk, bk), (wv, bv), (wo, bo) = lin(4, 4, 3), lin(4, 4, 7), lin(4, 4, 11), lin(4, 4, 15)

    c = F.silu(cond)
    gg, mm, ss = F.linear(c, wg, bg[0]), F.linear(c, wm, bm[0]), F.linear(c, ws, bs[0])
    mod = F.layer_norm(h, (4,), lg[0], lb[0], EPS) * tile(mm) + tile(ss)
    q, k, v = (mod @ w.T + b[0] for w, b in ((wq, bq), (wk, bk), (wv, bv)))
    q, k, v = (z.reshape(B, T, HD) for z in (q, k, v))
    attn = (F.softmax((q @ k.transpose(-1, -2)) * 0.5, dim=-1) @ v).reshape(N, HD)
    fout = attn @ wo.T + bo[0]
    y = h + tile(gg) * fout
    y.backward(torch.tensor(vals(24, 97).reshape(6, 4)))

    checks = [
        ("block.y", y), ("block.dcond", cond.grad), ("block.dh", h.grad),
        ("block.g.dw", wg.grad), ("block.g.db", bg.grad),
        ("block.m.dw", wm.grad), ("block.m.db", bm.grad),
        ("block.sh.dw", ws.grad), ("block.sh.db", bs.grad),
        ("block.ln.dg", lg.grad), ("block.ln.db", lb.grad),
        ("block.f.q.dw", wq.grad), ("block.f.q.db", bq.grad),
        ("block.f.k.dw", wk.grad), ("block.f.k.db", bk.grad),
        ("block.f.v.dw", wv.grad), ("block.f.v.db", bv.grad),
        ("block.f.o.dw", wo.grad), ("block.f.o.db", bo.grad),
    ]
    failures, worst = [], 0.0
    for name, ref in checks:
        if name not in got:
            failures.append(f"{name}: not printed")
            continue
        worst = max(worst, close(name, got[name], ref, failures))
    for f in failures:
        print("FAIL", f)
    # Do NOT loosen the tolerance to make this pass: the oracle is fixed.
    # Report the measured ratio so the gap is visible rather than hidden.
    print(f"dit-block: {len(checks)} tensors, worst error {worst:.3f} of the tolerance; "
          f"{'PASS' if not failures else 'FAIL'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
