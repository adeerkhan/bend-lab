"""The outnorm reference: is_outproj_norm=True, same cube as the SISO twin.

  reference/.venv-rf/bin/python reference/mamba3_outnorm_twin.py

With `is_outproj_norm` the kernel gets `Z=None` (no gate inside the block);
the block output is then `rms(out_v) * norm.weight * silu(z)` per
group_size = headdim (32) group, and only then out_proj.  Writes
`reference/mamba3_outnorm_golden.npz`: forward intermediates and autograd
gradients, including the norm weight's gradient.
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
NGROUPS, ROPE_FRACTION = 1, 0.5
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
        parts.append(e * c - o * s)
        parts.append(e * s + o * c)
    for j in range(2 * angles.shape[-1], x.shape[-1]):
        parts.append(x[..., j:j + 1])
    return torch.cat(parts, dim=-1)


def group_rms(x, w, g, eps=1e-5):
    """x (b, l, d), w (d,), groups of g: rms over each group, then * w."""
    b, l, d = x.shape
    return rms(x.reshape(b, l, d // g, g), w.reshape(1, 1, d // g, g), eps).reshape(b, l, d)


def twin_forward(m, u):
    b, l, _ = u.shape
    zx = m.in_proj(u)
    z, x, B, C, dd_dt, dd_A, trap_raw, ang_raw = torch.split(
        zx, [m.d_inner, m.d_inner, m.d_state, m.d_state, m.nheads,
             m.nheads, m.nheads, m.num_rope_angles], dim=-1)
    z4 = z.view(b, l, m.nheads, m.headdim)
    x4 = x.view(b, l, m.nheads, m.headdim)
    A = torch.clamp(-heavy_tail(dd_A.float()), max=-m.A_floor)
    DT = F.softplus(dd_dt.float() + m.dt_bias)
    ADT = A * DT
    trap = torch.sigmoid(trap_raw.float())
    Bn = rms(B.view(b, l, m.d_state), m.B_norm.weight)
    Cn = rms(C.view(b, l, m.d_state), m.C_norm.weight)

    vals = (torch.tanh(ang_raw.float()) * PI).unsqueeze(-2) * DT.unsqueeze(-1)
    angle = torch.cumsum(vals, dim=1) % (2 * PI)
    gamma = DT * trap
    shifted = torch.zeros_like(gamma)
    shifted[:, :-1] = DT[:, 1:] * (1 - trap[:, 1:])
    scale = gamma + shifted

    q_heads, k_heads, qk_heads = [], [], []
    for h in range(m.nheads):
        q0 = Cn + m.C_bias[h, 0]
        k0 = Bn + m.B_bias[h, 0]
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
    y_norm = group_rms(out_v.reshape(b, l, m.d_inner), m.norm.weight,
                       m.headdim)
    y_inner = y_norm * F.silu(z4.reshape(b, l, m.d_inner))
    y = y_inner @ m.out_proj.weight.T
    return {"pre": zx, "z": z4, "x": x4, "Braw": B, "Craw": C,
            "dd_dt": dd_dt, "dd_A": dd_A, "trap_raw": trap_raw,
            "ang_raw": ang_raw, "A": A, "DT": DT, "ADT": ADT, "trap": trap,
            "gamma": gamma, "shift": shifted, "scale": scale, "Bn": Bn,
            "Cn": Cn, "angle": angle, "q": q, "k": k, "qk": qk,
            "states": torch.stack(states, 1), "out_v": out_v,
            "y_norm": y_norm, "y_inner": y_inner, "y": y}


INTERMEDIATES = ("z", "x", "Braw", "Craw", "dd_dt", "dd_A", "trap_raw",
                 "ang_raw", "A", "DT", "ADT", "trap", "gamma", "shift",
                 "scale", "Bn", "Cn", "angle", "q", "k", "qk", "out_v",
                 "y_norm", "y_inner")


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
        "dW_on": m.norm.weight.grad.reshape(-1).cpu().numpy(),
    }
    for key in INTERMEDIATES:
        if f[key].grad is not None:
            g["g_" + key] = f[key].grad.reshape(-1).cpu().numpy()
        else:
            print(f"  no gradient for {key}")
    return f, g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "reference/mamba3_outnorm_golden.npz"))
    args = ap.parse_args()
    torch.manual_seed(71)
    torch.set_num_threads(1)
    m = Mamba3(d_model=D_MODEL, d_state=D_STATE, expand=EXPAND,
               headdim=HEADDIM, ngroups=NGROUPS,
               rope_fraction=ROPE_FRACTION, is_outproj_norm=True,
               is_mimo=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        m.norm.weight.copy_(torch.linspace(0.85, 1.15, m.d_inner))
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
        "norm_w": m.norm.weight.detach().numpy(),
    }
    for key in ("pre",) + INTERMEDIATES + ("states", "y"):
        arrays[key] = f[key].detach().numpy()
    arrays.update(g)
    np.savez(args.out, **arrays)
    print(f"wrote {args.out}")
    print("y max|.| %.6f  dW_on max|.| %.6f" %
          (float(f["y"].abs().max()), float(np.abs(g["dW_on"]).max())))


if __name__ == "__main__":
    main()
