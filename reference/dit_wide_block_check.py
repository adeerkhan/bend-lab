#!/usr/bin/env python3
"""Full conditioned DiT block at width 8, two kv heads, against PyTorch.

  python3 reference/dit_wide_block_check.py [binary] [--grads]

`examples/diffusion/dit/wide_block.bend`:
  y = h + Tile(g) * f(LayerNorm(h) * Tile(m) + Tile(sh))
  (g, m, sh) = Linear(silu(cond)), f = QKV -> FullAttention -> out Linear.
b=2, t=3, w=8, kv=2 heads of hd=4, scale 0.5. Forward is the gate; the
gradient check is reported but not gated (see results/dit/COMPARISON.md).
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "bendtorch", "reference"))
from oracle import close, parse  # noqa: E402

B, T, C, CD, KV, HD, SC, EPS = 2, 3, 8, 3, 2, 4, 0.5, 1e-5
N = B * T


def vals(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def vals2(n, start=0):
    return np.array([((i * i * 5 + i * 3) % 17 - 8) / 4 for i in range(start, start + n)], dtype=np.float32)


def tt(a, r, c, grad=True):
    return torch.tensor(a.reshape(r, c)).requires_grad_(grad)


def lin(i, o, s):
    return tt(vals(o * i, s), o, i), tt(vals(o, s + 40), 1, o)


def tile(x):
    return x.repeat_interleave(T, dim=0)


def forward(cond, h, P):
    wg, bg = P["g"]
    wm, bm = P["m"]
    ws, bs = P["sh"]
    lg, lb = P["ln"]
    wq, bq, wk, bk, wv, bv, wo, bo = P["attn"]
    c = F.silu(cond)
    g, m, sh = F.linear(c, wg, bg), F.linear(c, wm, bm), F.linear(c, ws, bs)
    mod = F.layer_norm(h, (C,), lg.view(C), lb.view(C), EPS) * tile(m) + tile(sh)
    q, k, v = (mod @ w.T + b for w, b in ((wq, bq), (wk, bk), (wv, bv)))
    q, k, v = (z.reshape(B, T, KV, HD).transpose(1, 2) for z in (q, k, v))
    a = (F.softmax((q @ k.transpose(-1, -2)) * SC, dim=-1) @ v).transpose(1, 2).reshape(N, C)
    return h + tile(g) * (a @ wo.T + bo)


def main():
    grads = "--grads" in sys.argv
    binary = next((a for a in sys.argv[1:] if not a.startswith("--")), None) or os.path.join(
        os.path.dirname(HERE), ".scratch", "dit_wide_block")
    got = parse(subprocess.run([binary], check=True, capture_output=True, text=True).stdout)

    cond, h = tt(vals2(B * CD, 23), B, CD), tt(vals2(N * C, 33), N, C)
    P = {}
    for key, seed in (("g", 7), ("m", 9), ("sh", 11)):
        P[key] = lin(CD, C, seed)
    P["ln"] = (tt(vals(C, 61), 1, C), tt(vals(C, 67), 1, C))
    P["attn"] = tuple(x for s in (3, 7, 11, 15) for x in lin(C, C, s))

    y = forward(cond, h, P)
    dy = tt(vals(N * C, 97), N, C, grad=False)
    y.backward(dy)
    failures, worst = [], 0.0
    worst = max(worst, close("wide.y", got["wide.y"], y, failures))
    for f in failures:
        print("FAIL", f)
    if failures:
        raise SystemExit(1)
    print(f"dit-wide-block: forward matches PyTorch ({worst:.3f} of tolerance); PASS")

    if grads:
        checks = {"wide.dcond": cond.grad, "wide.dh": h.grad}
        for pref, key in (("wide.g.dw", "g"), ("wide.m.dw", "m"), ("wide.sh.dw", "sh")):
            checks[pref + ".dw"] = P[key][0].grad
        gfail, gworst = [], 0.0
        for name, ref in checks.items():
            if name in got:
                gworst = max(gworst, close(name, got[name], ref, gfail))
        print(f"dit-wide-block grads: {len(checks)} tensors, worst {gworst:.3f} of tolerance "
              f"({'PASS' if not gfail else 'FAIL'})")
        for f in gfail:
            print("  ", f)


if __name__ == "__main__":
    main()
