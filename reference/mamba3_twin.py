"""The Mamba-3 SISO reference: a pure-PyTorch twin, its kernel check, and the goldens.

  reference/.venv-rf/bin/python reference/mamba3_twin.py
  reference/.venv-rf/bin/python reference/mamba3_twin.py --bins reference/mamba3_data

The vendored `mamba_ssm.modules.mamba3.Mamba3` is a wrapper over fused Triton
kernels; the mathematical block is written here as an explicit loop over t.

Cube: d_model 64, d_state 16, expand 2, headdim 32 -> 4 heads, ngroups 1,
rope_fraction 0.5 -> 4 angle pairs, chunk 64, batch 1, seqlen 8.

Two lanes:

  kernel_f32    the fwd kernel itself called with float32 inputs and IEEE dots
                (`TRITON_F32_DEFAULT=ieee`).  This checks the math; it is the
                closest run this GPU (sm_120) gives to the reference recurrence.
  module_bf16   the public module path.  Its wrapper casts Q, K, V, Trap,
                Angles and Z to bfloat16 before the kernel, so it is a bf16
                lane and is reported, not gated.

The goldens are the twin's F32 values, their gradients from PyTorch autograd
(loss = sum(y * dy)), and one moved-dy can-fail.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _RMSNorm(nn.Module):
    """The parameter/epsilon contract of the vendored gated RMSNorm."""

    def __init__(self, hidden_size, eps=1e-5, group_size=None,
                 norm_before_gate=True, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        if self.group_size is None:
            y = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            shape = x.shape
            grouped = x.reshape(*shape[:-1], shape[-1] // self.group_size, self.group_size)
            y = grouped * torch.rsqrt(grouped.float().pow(2).mean(-1, keepdim=True) + self.eps)
            y = y.reshape(shape)
        y = y * self.weight
        return y if z is None or not self.norm_before_gate else y * F.silu(z)


class PureMamba3(nn.Module):
    """The Mamba-3 parameter layout, without importing fused Triton ops.

    This mirrors the vendored constructor's random-draw order and tensor
    shapes for the SISO, GQA and outproj-norm configurations. The recurrences
    live in the twin modules, not this module's forward path.
    """

    def __init__(self, d_model, d_state=128, expand=2, headdim=64, ngroups=1,
                 rope_fraction=0.5, dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4, A_floor=1e-4, is_outproj_norm=False,
                 is_mimo=False, mimo_rank=4, fuse_pregate_headwise_norm=True,
                 chunk_size=64, dropout=0.0, layer_idx=None, n_layer=None,
                 device=None, dtype=None, **kwargs):
        super().__init__()
        if is_mimo:
            raise ValueError("the pure twin does not implement MIMO kernels")
        if rope_fraction not in (0.5, 1.0):
            raise ValueError("rope_fraction must be 0.5 or 1.0")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = 1
        self.fuse_pregate_headwise_norm = False
        self.d_inner = int(expand * d_model)
        if self.d_inner % headdim:
            raise ValueError("d_inner must be divisible by headdim")
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        if self.num_rope_angles <= 0:
            raise ValueError("d_state and rope_fraction leave no RoPE pairs")

        d_in_proj = (2 * self.d_inner + 2 * d_state * ngroups * self.mimo_rank
                     + 3 * self.nheads + self.num_rope_angles)
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)
        _dt = torch.exp(torch.rand(self.nheads, device=device, dtype=torch.float32)
                        * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        _dt = torch.clamp(_dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))
        self.dt_bias._no_weight_decay = True
        self.B_bias = nn.Parameter(1 + torch.zeros(
            (self.nheads, self.mimo_rank, d_state), dtype=torch.float32, device=device))
        self.C_bias = nn.Parameter(1 + torch.zeros(
            (self.nheads, self.mimo_rank, d_state), dtype=torch.float32, device=device))
        self.B_norm = _RMSNorm(d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = _RMSNorm(d_state, eps=1e-5, **factory_kwargs)
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        if is_outproj_norm:
            self.norm = _RMSNorm(self.d_inner, eps=1e-5, group_size=headdim,
                                 norm_before_gate=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)


class _Mamba3Proxy:
    """Import the fused module only when a caller actually constructs it.

    The pure recurrence and the tiny-LM dump need no Triton. On a CPU-only
    reference environment the proxy falls back to the parameter-identical
    pure constructor; the kernel lanes still use the vendored fused module
    whenever it imports.
    """

    def __new__(cls, *args, **kwargs):
        sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
        try:
            from mamba_ssm.modules.mamba3 import Mamba3 as FusedMamba3  # noqa: E402
        except ImportError:
            return PureMamba3(*args, **kwargs)
        return FusedMamba3(*args, **kwargs)


Mamba3 = _Mamba3Proxy

D_MODEL, D_STATE, EXPAND, HEADDIM, NGROUPS = 64, 16, 2, 32, 1
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


def twin_forward(m, u, bf16=False):
    """The whole SISO block, t by t.  Returns every intermediate the port needs."""
    b, l, _ = u.shape
    zx = m.in_proj(u)
    z, x, B, C, dd_dt, dd_A, trap_raw, ang_raw = torch.split(
        zx, [m.d_inner, m.d_inner, m.d_state, m.d_state, m.nheads, m.nheads,
             m.nheads, m.num_rope_angles], dim=-1)
    z4 = z.view(b, l, m.nheads, m.headdim)
    x4 = x.view(b, l, m.nheads, m.headdim)
    A = torch.clamp(-heavy_tail(dd_A.float()), max=-m.A_floor)
    DT = F.softplus(dd_dt.float() + m.dt_bias)
    ADT = A * DT
    trap = torch.sigmoid(trap_raw.float())
    Bn = rms(B.view(b, l, m.d_state), m.B_norm.weight, m.B_norm.eps)
    Cn = rms(C.view(b, l, m.d_state), m.C_norm.weight, m.C_norm.eps)

    if bf16:
        # The public wrapper casts these before it calls the kernel.
        Bn, Cn, x4, trap_raw, ang_raw, z4 = [
            t.bfloat16().float() for t in (Bn, Cn, x4, trap_raw, ang_raw, z4)]
        trap = torch.sigmoid(trap_raw)

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
    y_inner = out_v * F.silu(z4)
    y = y_inner.reshape(b, l, m.d_inner) @ m.out_proj.weight.T
    return {
        "pre": zx, "z": z4, "x": x4, "Braw": B, "Craw": C,
        "dd_dt": dd_dt, "dd_A": dd_A, "trap_raw": trap_raw, "ang_raw": ang_raw,
        "A": A, "DT": DT, "ADT": ADT, "trap": trap, "gamma": gamma,
        "shift": shifted, "scale": scale, "Bn": Bn, "Cn": Cn,
        "angle": angle, "q": q, "k": k, "qk": qk, "states": torch.stack(states, 1),
        "out_v": out_v, "y_inner": y_inner, "y": y,
    }


def kernel_check(m, u):
    """Call the fwd kernel directly with F32 inputs and IEEE dots."""
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd
    f = twin_forward(m, u, bf16=False)
    b, l = u.shape[0], u.shape[1]
    res = mamba3_siso_fwd(
        Q=f["Cn"].view(b, l, 1, m.d_state), K=f["Bn"].view(b, l, 1, m.d_state),
        V=f["x"], ADT=f["ADT"].permute(0, 2, 1).contiguous(),
        DT=f["DT"].permute(0, 2, 1).contiguous(),
        Trap=f["trap_raw"].permute(0, 2, 1).contiguous(),
        Q_bias=m.C_bias.squeeze(1), K_bias=m.B_bias.squeeze(1),
        Angles=f["angle"], D=m.D, Z=None, chunk_size=64,
        store_states_adt_outv=True, return_final_states=True)
    out_v, ssm = res[1], res[10][0]
    d_out = (out_v - f["out_v"]).abs()
    d_state = (ssm - f["states"][:, -1]).abs()
    return {
        "out_abs": d_out.max().item(),
        "out_rel": d_out.max().item() / max(1e-30, f["out_v"].abs().max().item()),
        "state_abs": d_state.max().item(),
        "state_rel": d_state.max().item() / max(1e-30, f["states"][:, -1].abs().max().item()),
    }


def public_check(m, u):
    with torch.no_grad():
        yk = m(u)
        yt = twin_forward(m, u, bf16=True)["y"]
    d = (yk - yt).abs()
    return {"abs": d.max().item(),
            "rel": d.max().item() / max(1e-30, yk.abs().max().item())}


GRAD_INTERMEDIATES = ("z", "x", "Braw", "Craw", "dd_dt", "dd_A", "trap_raw",
                      "ang_raw", "A", "DT", "ADT", "trap", "gamma", "scale",
                      "Bn", "Cn", "angle", "q", "k", "qk", "out_v",
                      "y_inner")


def grads(m, u, dy):
    for p in m.parameters():
        p.grad = None
    if u.grad is not None:
        u.grad = None
    f = twin_forward(m, u, bf16=False)
    for key in GRAD_INTERMEDIATES:
        f[key].retain_grad()
    (f["y"] * dy).sum().backward()
    g = {
        "du": u.grad.detach().reshape(-1).cpu().numpy(),
        "dW_in": m.in_proj.weight.grad.detach().reshape(-1).cpu().numpy(),
        "ddt_bias": m.dt_bias.grad.detach().reshape(-1).cpu().numpy(),
        "dB_bias": m.B_bias.grad.detach().reshape(-1).cpu().numpy(),
        "dC_bias": m.C_bias.grad.detach().reshape(-1).cpu().numpy(),
        "dB_norm_w": m.B_norm.weight.grad.detach().reshape(-1).cpu().numpy(),
        "dC_norm_w": m.C_norm.weight.grad.detach().reshape(-1).cpu().numpy(),
        "dD": m.D.grad.detach().reshape(-1).cpu().numpy(),
        "dW_out": m.out_proj.weight.grad.detach().reshape(-1).cpu().numpy(),
    }
    for key in GRAD_INTERMEDIATES:
        g["g_" + key] = f[key].grad.detach().reshape(-1).cpu().numpy()
    return f, g


def init_extra():
    g = torch.Generator(device="cuda")
    g.manual_seed(123)
    E = torch.randn(8, 64, device="cuda", generator=g) * 0.5
    W = torch.randn(8, 64, device="cuda", generator=g) * 0.5
    return E, W


def train_twin(steps=10, lr=1e-3):
    """The copy task: tokens 0..7 in, the same tokens out, ten SGD steps."""
    torch.manual_seed(71)
    torch.set_num_threads(1)
    m = Mamba3(d_model=D_MODEL, d_state=D_STATE, expand=EXPAND, headdim=HEADDIM,
               ngroups=NGROUPS, is_mimo=False).cuda().float()
    E0, W0 = init_extra()
    E = E0.clone().requires_grad_(True)
    W = W0.clone().requires_grad_(True)
    targets = torch.arange(8, device="cuda")
    losses, tokens = [], None
    for _ in range(steps):
        f = twin_forward(m, E.unsqueeze(0))
        logits = f["y"].reshape(8, 64) @ W.T
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        tokens = logits.argmax(-1).detach()
        with torch.no_grad():
            for p in m.parameters():
                p -= lr * p.grad
                p.grad = None
            E -= lr * E.grad
            W -= lr * W.grad
            E.grad = None
            W.grad = None
        losses.append(float(loss))
    return losses, tokens, E.detach(), W.detach(), m


def generate_twin(m, E, W, steps=12):
    """Two weight-tied layers, greedy, starting from tokens 0..3."""
    tokens = [0, 1, 2, 3]
    first = None
    for i in range(steps):
        u = E[torch.tensor(tokens, device="cuda")].unsqueeze(0)
        y1 = twin_forward(m, u)["y"]
        y2 = twin_forward(m, y1)["y"]
        logits = y2[0, -1] @ W.T
        if i == 0:
            first = logits.detach().cpu().numpy().reshape(-1)
        tokens.append(int(logits.argmax()))
    return tokens, first


def make_goldens(out_path):
    torch.manual_seed(71)
    torch.set_num_threads(1)
    m = Mamba3(d_model=D_MODEL, d_state=D_STATE, expand=EXPAND, headdim=HEADDIM,
               ngroups=NGROUPS, is_mimo=False).cuda().float()
    u = (torch.randn(1, T, D_MODEL, device="cuda") * 0.5).requires_grad_(True)
    dy = torch.randn(1, T, D_MODEL, device="cuda") * 0.1
    dy_bad = dy.clone()
    dy_bad[0, 3, 17] += 1.0

    kcheck = kernel_check(m, u.detach())
    pcheck = public_check(m, u.detach())
    print("kernel_f32_ieee out max|d| %.6e rel %.6e" %
          (kcheck["out_abs"], kcheck["out_rel"]))
    print("kernel_f32_ieee state max|d| %.6e rel %.6e" %
          (kcheck["state_abs"], kcheck["state_rel"]))
    print("module_bf16 out max|d| %.6e rel %.6e" % (pcheck["abs"], pcheck["rel"]))

    f, g = grads(m, u, dy)
    _, gb = grads(m, u, dy_bad)

    arrays = {
        "u": u.detach().reshape(1, T, D_MODEL).cpu().numpy(),
        "dy": dy.reshape(1, T, D_MODEL).cpu().numpy(),
        "dy_bad": dy_bad.reshape(1, T, D_MODEL).cpu().numpy(),
        "in_proj": m.in_proj.weight.detach().cpu().numpy(),
        "dt_bias": m.dt_bias.detach().cpu().numpy(),
        "B_bias": m.B_bias.detach().squeeze(1).cpu().numpy(),
        "C_bias": m.C_bias.detach().squeeze(1).cpu().numpy(),
        "B_norm_w": m.B_norm.weight.detach().cpu().numpy(),
        "C_norm_w": m.C_norm.weight.detach().cpu().numpy(),
        "D": m.D.detach().cpu().numpy(),
        "out_proj": m.out_proj.weight.detach().cpu().numpy(),
    }
    for key in ("pre", "z", "x", "Braw", "Craw", "dd_dt", "dd_A", "trap_raw",
                "ang_raw", "A", "DT", "ADT", "trap", "gamma", "shift", "scale",
                "Bn", "Cn", "angle", "q", "k", "qk", "states", "out_v",
                "y_inner", "y"):
        arrays[key] = f[key].detach().cpu().numpy()
    for key, value in g.items():
        arrays[key] = value
    for key, value in gb.items():
        arrays[key + "_bad"] = value
    E0, W0 = init_extra()
    tl, tt, tE, tW, tm = train_twin()
    arrays["E0"] = E0.cpu().numpy()
    arrays["WhT0"] = W0.cpu().numpy()
    arrays["train_loss"] = np.asarray(tl, dtype=np.float32)
    arrays["train_tokens"] = tt.cpu().numpy().astype(np.float32)
    arrays["train_E"] = tE.cpu().numpy()
    arrays["train_WhT"] = tW.cpu().numpy()
    arrays["train_w_in"] = tm.in_proj.weight.detach().cpu().numpy()
    arrays["train_w_out"] = tm.out_proj.weight.detach().cpu().numpy()
    arrays["train_dt_bias"] = tm.dt_bias.detach().cpu().numpy()
    arrays["train_B_bias"] = tm.B_bias.detach().squeeze(1).cpu().numpy()
    arrays["train_C_bias"] = tm.C_bias.detach().squeeze(1).cpu().numpy()
    arrays["train_B_norm_w"] = tm.B_norm.weight.detach().cpu().numpy()
    arrays["train_C_norm_w"] = tm.C_norm.weight.detach().cpu().numpy()
    arrays["train_D"] = tm.D.detach().cpu().numpy()
    gtoks, gfirst = generate_twin(tm, tE, tW)
    arrays["gen_tokens"] = np.asarray(gtoks, dtype=np.float32)
    arrays["gen_logits0"] = gfirst
    torch.manual_seed(77)
    u4 = torch.randn(1, 4, D_MODEL, device="cuda") * 0.5
    u16 = torch.randn(1, 16, D_MODEL, device="cuda") * 0.5
    arrays["u4"] = u4.cpu().numpy()
    arrays["y4"] = twin_forward(m, u4)["y"].detach().cpu().numpy()
    arrays["u16"] = u16.cpu().numpy()
    arrays["y16"] = twin_forward(m, u16)["y"].detach().cpu().numpy()
    np.savez(out_path, **arrays)
    print(f"wrote {out_path}")
    print(f"y max|value| {float(np.abs(arrays['y']).max()):.6e}")
    print(f"dW_in max|value| {float(np.abs(arrays['dW_in']).max()):.6e}")
    return 0


def write_bins(golden_path, out_dir):
    z = np.load(golden_path)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for key in z.files:
        a = z[key].astype(np.float32)
        a.tofile(os.path.join(out_dir, key + ".bin"))
        manifest[key] = list(a.shape)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print(f"wrote {len(manifest)} bins to {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=os.path.abspath,
                    default=os.path.join(ROOT, "reference/mamba3_golden.npz"))
    ap.add_argument("--bins", type=os.path.abspath, default=None)
    args = ap.parse_args()
    if args.bins:
        if not os.path.exists(args.golden):
            make_goldens(args.golden)
        return write_bins(args.golden, args.bins)
    return make_goldens(args.golden)


if __name__ == "__main__":
    sys.exit(main())
