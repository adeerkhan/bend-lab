"""The GQA reference: ngroups 2, rope_fraction 1.0, same cube as the SISO twin.

  reference/.venv-rf/bin/python reference/mamba3_gqa_twin.py

Writes `reference/mamba3_gqa_golden.npz`: the forward intermediates and the
analytic (autograd) gradients of the packed block at the GQA config, from
our own explicit t-loop.  The shapes:

  d_model 64, d_state 16, expand 2, headdim 32 -> 4 v-heads, 2 q/k groups,
  rope_fraction 1.0 -> 8 angle pairs (all 16 q/k dims rotated), batch 1,
  seqlen 8.

Layout (module): in_proj out = [z 128, x 128, B 2x16, C 2x16,
dd_dt 4, dd_A 4, trap 4, ang_raw 8] = 340.
B/C are per group (2), the B/C biases are per v-head (4); q/k heads are the
groups, and v-head h reads group h//2.  The angle state is per v-head.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
from mamba_ssm.modules.mamba3 import Mamba3  # noqa: E402

D_MODEL, D_STATE, EXPAND, HEADDIM = 64, 16, 2, 32
NGROUPS, ROPE_FRACTION = 2, 1.0
N, T = 8, 8
PI = 3.141592653589793


def heavy_tail(x):
    return torch.where(x >= 0, 1.0 + x, torch.reciprocal(1.0 - x))


def rms(x, w, eps=1e-5):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, angles):
    """Rotate pairs (2j, 2j+1); dims past 2*angles.shape[-1] are unchanged."""
    parts = []
    for j in range(angles.shape[-1]):
        a = angles[..., j].unsqueeze(-1)
        c, s = torch.cos(a), torch.sin(a)
        e, o = x[..., 2 * j:2 * j + 1], x[..., 2 * j + 1:2 * j + 2]
        parts.append(torch.cat([e * c - o * s, e * s + o * c], dim=-1))
    return torch.cat(parts, dim=-1)


def twin_forward(m, u):
    b, l, _ = u.shape
    zx = m.in_proj(u)
    z, x, B, C, dd_dt, dd_A, trap_raw, ang_raw = torch.split(
        zx, [m.d_inner, m.d_inner,
             m.d_state * m.num_bc_heads, m.d_state * m.num_bc_heads,
             m.nheads, m.nheads, m.nheads, m.num_rope_angles], dim=-1)
    z4 = z.view(b, l, m.nheads, m.headdim)
    x4 = x.view(b, l, m.nheads, m.headdim)
    A = torch.clamp(-heavy_tail(dd_A.float()), max=-m.A_floor)
    DT = F.softplus(dd_dt.float() + m.dt_bias)
    ADT = A * DT
    trap = torch.sigmoid(trap_raw.float())
    ng = m.num_bc_heads
    Bg = B.view(b, l, ng, m.d_state)
    Cg = C.view(b, l, ng, m.d_state)
    Bn = rms(Bg, m.B_norm.weight)
    Cn = rms(Cg, m.C_norm.weight)

    vals = (torch.tanh(ang_raw.float()) * PI).unsqueeze(-2) * DT.unsqueeze(-1)
    angle = torch.cumsum(vals, dim=1) % (2 * PI)  # (b, l, nheads, pairs)
    gamma = DT * trap
    shifted = torch.zeros_like(gamma)
    shifted[:, :-1] = DT[:, 1:] * (1 - trap[:, 1:])
    scale = gamma + shifted

    q_heads, k_heads, qk_heads = [], [], []
    for h in range(m.nheads):
        g = h // (m.nheads // m.num_bc_heads)
        q0 = Cn[..., g, :] + m.C_bias[h, 0]
        k0 = Bn[..., g, :] + m.B_bias[h, 0]
        qk_heads.append((q0 * k0).sum(-1) * gamma[..., h])
        q_heads.append(rope(q0, angle[..., h, :]))
        k_heads.append(rope(k0, angle[..., h, :]))
    q = torch.stack(q_heads, dim=2)
    k = torch.stack(k_heads, dim=2)
    qk = torch.stack(qk_heads, dim=2)

    states = [torch.zeros(b, m.nheads, m.headdim, m.d_state, device=u.device)]
    outs = []
    for t in range(l):
        a = torch.exp(ADT[:, t])
        o = (a[..., None, None] * (q[:, t].unsqueeze(-2)
                                   @ states[-1].transpose(-1, -2))).squeeze(-2)
        o = o + (m.D + qk[:, t])[..., None] * x4[:, t]
        outs.append(o)
        states.append(a[..., None, None] * states[-1]
                      + x4[:, t].unsqueeze(-1)
                      * (scale[:, t].unsqueeze(-1) * k[:, t]).unsqueeze(-2))
    out_v = torch.stack(outs, dim=1)
    y_inner = out_v * F.silu(z4)
    y = y_inner.reshape(b, l, m.d_inner) @ m.out_proj.weight.T
    return {"pre": zx, "z": z4, "x": x4, "Braw": Bg, "Craw": Cg,
            "dd_dt": dd_dt, "dd_A": dd_A, "trap_raw": trap_raw,
            "ang_raw": ang_raw, "A": A, "DT": DT, "ADT": ADT, "trap": trap,
            "gamma": gamma, "shift": shifted, "scale": scale, "Bn": Bn,
            "Cn": Cn, "angle": angle, "q": q, "k": k, "qk": qk,
            "states": torch.stack(states, 1), "out_v": out_v,
            "y_inner": y_inner, "y": y}


INTERMEDIATES = ("z", "x", "Braw", "Craw", "dd_dt", "dd_A", "trap_raw",
                 "ang_raw", "A", "DT", "ADT", "trap", "gamma", "shift",
                 "scale", "Bn", "Cn", "angle", "q", "k", "qk", "out_v",
                 "y_inner")


def grads(m, u, dy):
    for p in m.parameters():
        p.grad = None
    if u.grad is not None:
        u.grad = None
    f = twin_forward(m, u)
    for key in INTERMEDIATES:
        f[key].retain_grad()
    (f["y"] * dy).sum().backward()
    g = {
        "du": u.grad.reshape(-1).cpu().numpy(),
        "dW_in": m.in_proj.weight.grad.reshape(-1).cpu().numpy(),
        "ddt_bias": m.dt_bias.grad.reshape(-1).cpu().numpy(),
        "dB_bias": m.B_bias.grad.reshape(-1).cpu().numpy(),
        "dC_bias": m.C_bias.grad.reshape(-1).cpu().numpy(),
        "dB_norm_w": m.B_norm.weight.grad.reshape(-1).cpu().numpy(),
        "dC_norm_w": m.C_norm.weight.grad.reshape(-1).cpu().numpy(),
        "dD": m.D.grad.reshape(-1).cpu().numpy(),
        "dW_out": m.out_proj.weight.grad.reshape(-1).cpu().numpy(),
    }
    for key in INTERMEDIATES:
        g["g_" + key] = f[key].grad.reshape(-1).cpu().numpy()
    return f, g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "reference/mamba3_gqa_golden.npz"))
    args = ap.parse_args()
    torch.manual_seed(71)
    torch.set_num_threads(1)
    m = Mamba3(d_model=D_MODEL, d_state=D_STATE, expand=EXPAND,
               headdim=HEADDIM, ngroups=NGROUPS,
               rope_fraction=ROPE_FRACTION, is_mimo=False,
               device="cpu", dtype=torch.float32)
    u = (torch.randn(1, T, D_MODEL) * 0.5).requires_grad_(True)
    dy = torch.randn(1, T, D_MODEL) * 0.1
    f, g = grads(m, u, dy)
    arrays = {
        "u": u.detach().reshape(1, T, D_MODEL).numpy(),
        "dy": dy.reshape(1, T, D_MODEL).numpy(),
        "in_proj": m.in_proj.weight.detach().numpy(),
        "dt_bias": m.dt_bias.detach().numpy(),
        "B_bias": m.B_bias.detach().squeeze(1).numpy(),
        "C_bias": m.C_bias.detach().squeeze(1).numpy(),
        "B_norm_w": m.B_norm.weight.detach().numpy(),
        "C_norm_w": m.C_norm.weight.detach().numpy(),
        "D": m.D.detach().numpy(),
        "out_proj": m.out_proj.weight.detach().numpy(),
        "ngroups": np.asarray([NGROUPS], dtype=np.float32),
        "rope_fraction": np.asarray([ROPE_FRACTION], dtype=np.float32),
    }
    for key in ("pre",) + INTERMEDIATES + ("states", "y"):
        arrays[key] = f[key].detach().numpy()
    arrays.update(g)
    np.savez(args.out, **arrays)
    print(f"wrote {args.out}")
    print("y max|.| %.6f  du max|.| %.6f  dW_in max|.| %.6f" %
          (float(f["y"].abs().max()), float(np.abs(g["du"]).max()),
           float(np.abs(g["dW_in"]).max())))


if __name__ == "__main__":
    main()
