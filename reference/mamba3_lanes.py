"""Mamba-3 numerical lanes: the same block through four F32/TF32/bf16 lanes.

  reference/.venv-rf/bin/python reference/mamba3_lanes.py twin 8        # dump F32 twin
  reference/.venv-rf/bin/python reference/mamba3_lanes.py kernel 8      # dump kernel lane
  reference/.venv-rf/bin/python reference/mamba3_lanes.py bf16 8        # dump module lane
  reference/.venv-rf/bin/python reference/mamba3_lanes.py compare       # write the table
  reference/.venv-rf/bin/python reference/mamba3_lanes.py             # dump all + compare

The lanes, in order:

  twin   the F32 oracle (reference/mamba3_twin.py, explicit loop, F32).
  ieee   the vendored Triton kernel, TRITON_F32_DEFAULT=ieee (IEEE dots),
         reached two ways: mamba3_siso_fwd for the forward intermediates,
         mamba3_siso_combined (autograd) for the backward.
  tf32   the same kernel with the Triton default (TF32 dots).
  module the public Mamba3 module on CUDA; its wrapper casts x/Q/K/V/trap/
         angles to bfloat16 before the kernel, so it is the bf16 lane.

All four lanes see the same weights and the same input (the first T rows of
one fixed 64-token tensor), so the table isolates the lane, not the config.
The env var for the kernel lanes is set at process start and read once at
Triton import time, hence the dump/compare split.

`unavailable` is printed where the lane has no counterpart for a tensor
(the kernel does not store its angle path; gradients of its raw input
angles are summed over heads below to match the twin's broadcast).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))

SCRATCH = os.path.join(ROOT, ".scratch", "lanes")
OUT_DOC = os.path.join(ROOT, "docs", "mamba3-lanes.md")
TS = [1, 2, 4, 8, 16, 64]
H = 4   # nheads (d_inner 128 / headdim 32)
HEADDIM = 32
QK_DIM = 16  # headdim_qk (nheads_qk = 1, headdim_qk = d_state)

# forward intermediates kept for every lane (flattened, F32 or bf16)
FWD_KEYS = ["y", "out_v", "states", "angle", "Bn", "Cn", "DT", "ADT",
            "trap", "gamma", "scale", "q", "k", "qk", "A"]
# backward pairs, kernel-side (kernel input) vs twin-side (twin intermediate)
BWD_PAIRS = [("g_x", "dV"), ("g_ADT", "dADT"), ("g_DT", "dDT"), ("g_trap", "dTrap"),
             ("g_C_bias", "dQ_bias"), ("g_B_bias", "dK_bias"),
             ("g_angle", "dAngles"), ("g_D", "dD"),
             ("g_Cn", "dQ"), ("g_Bn", "dK")]
# backward pairs for the module lane (whole module backward)
MOD_PAIRS = [("du", None), ("dW_in", None), ("ddt_bias", None),
             ("dB_bias", None), ("dC_bias", None),
             ("dB_norm_w", None), ("dC_norm_w", None),
             ("dD", None), ("dW_out", None)]

import numpy as np
import torch


def rel(d: np.ndarray, ref: np.ndarray) -> float:
    return float(np.abs(d).max()) / max(1e-30, float(np.abs(ref).max()))


def build_block():
    import mamba3_twin as W
    torch.manual_seed(71)
    torch.set_num_threads(1)
    m = W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32,
                 ngroups=1, is_mimo=False).cuda().float()
    return W, m


def build_input():
    torch.manual_seed(123)
    u = (torch.randn(1, 64, 64, device="cuda") * 0.5)
    dy = (torch.randn(1, 64, 64, device="cuda",
                      generator=torch.Generator("cuda").manual_seed(9)) * 0.1)
    return u, dy


def dump_twin(T):
    import mamba3_twin as W
    os.makedirs(SCRATCH, exist_ok=True)
    _, m = build_block()
    u_full, dy_full = build_input()
    u = u_full[:, :T].clone().requires_grad_(True)
    dy = dy_full[:, :T].contiguous()
    f = W.twin_forward(m, u)
    want = ("Cn", "Bn", "x", "ADT", "DT", "trap_raw", "ang_raw")
    for key in want:
        f[key].retain_grad()
    (f["y"] * dy).sum().backward()
    data = {"y": f["y"].detach().cpu().numpy().astype(np.float32).reshape(-1),
            "du": u.grad.detach().cpu().numpy().astype(np.float32).reshape(-1)}
    for key in FWD_KEYS:
        if key == "y":
            continue
        v = f[key]
        if key == "states":
            v = f["states"][:, -1]          # final state only
        data[key] = v.detach().cpu().numpy().astype(np.float32).reshape(-1)
    names = {"dW_in": m.in_proj.weight, "ddt_bias": m.dt_bias,
             "dB_bias": m.B_bias, "dC_bias": m.C_bias,
             "dB_norm_w": m.B_norm.weight, "dC_norm_w": m.C_norm.weight,
             "dD": m.D, "dW_out": m.out_proj.weight}
    for k, p in names.items():
        data[k] = p.grad.detach().cpu().numpy().astype(np.float32).reshape(-1)
    for key in want:
        data["g_" + key] = f[key].grad.detach().cpu().numpy().astype(np.float32).reshape(-1)
    np.savez(os.path.join(SCRATCH, f"twin_{T}.npz"), **data)
    print(f"wrote twin_{T}.npz (T={T})")



def _cap_autotune_stages():
    """Keep autotune configs that fit this GPU's shared memory (101376 B).

    The vendored autotune lists include num_stages=3 configs that can ask
    for 122384 B of shared memory; on a loaded GPU Triton selected one and
    raised OutOfResources under the full check (the gate passed standalone).
    This is a compile-resource cap, not a numeric tolerance: the dots are
    unchanged, and the lane bounds are asserted from fresh dumps below.
    """
    import mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd as fwd
    import mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd as bwd
    import mamba_ssm.ops.triton.mamba3.angle_dt as angle
    capped = 0
    for mod in (fwd, bwd, angle):
        for name in dir(mod):
            obj = getattr(mod, name)
            configs = getattr(obj, "configs", None)
            if isinstance(configs, (list, tuple)):
                keep = [c for c in configs if (getattr(c, "num_stages", 1) or 1) <= 2]
                if len(keep) != len(configs):
                    try:
                        obj.configs = keep
                        capped += 1
                    except (AttributeError, TypeError):
                        pass
    return capped

def dump_kernel(T):
    """The F32 kernel lane: low-level Triton fwd/bwd with F32 tensors.

    mamba3_siso_combined always casts Q/K/V/Trap/Angles to bf16, so it is
    the bf16 lane, not this one.  This replicates _Mamba3Function.backward
    step for step, with F32 inputs and F32 nothing cast, so TRITON_F32_DEFAULT
    is the only difference between the ieee and tf32 lanes.
    """
    lane = os.environ.get("TRITON_F32_DEFAULT", "tf32")  # read before triton loads
    import torch.nn.functional as F
    import mamba3_twin as W
    _cap_autotune_stages()
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import (
        compute_dqkv, compute_dqktheta, compute_ddt_dtrap_dinput_states)
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd, angle_dt_bwd
    os.makedirs(SCRATCH, exist_ok=True)
    _, m = build_block()
    u_full, dy_full = build_input()
    u = u_full[:, :T].clone()
    dy = dy_full[:, :T].contiguous()
    f = W.twin_forward(m, u)
    Tloc = dy.shape[1]
    Q = f["Cn"].view(1, Tloc, 1, 16).detach().clone()
    K = f["Bn"].view(1, Tloc, 1, 16).detach().clone()
    V = f["x"].detach().clone()
    ADT = f["ADT"].permute(0, 2, 1).contiguous()
    DT = f["DT"].permute(0, 2, 1).contiguous()
    Trap = f["trap_raw"].permute(0, 2, 1).contiguous()  # the kernel sigmoids internally
    Qb = m.C_bias.squeeze(1).detach().clone()
    Kb = m.B_bias.squeeze(1).detach().clone()
    Ang = f["ang_raw"].unsqueeze(2).expand(1, Tloc, H, f["ang_raw"].shape[-1]) \
        .contiguous()
    D = m.D.detach().clone()
    # dL/d(out_v) = silu(z) * (dy @ out_proj.weight); y = (out_v * silu(z)) @ W_out^T
    dy_eff = (dy.reshape(Tloc, 64) @ m.out_proj.weight).reshape(1, Tloc, H, HEADDIM) \
        * F.silu(f["z"])

    Angles_Cumsum, _ = angle_dt_fwd(Ang, DT, init_state=None,
                                    chunk_size=64, return_output_state=True)
    res = mamba3_siso_fwd(Q, K, V, ADT, DT, Trap, Qb, Kb, Angles_Cumsum, D,
                          Z=None, chunk_size=64, store_states_adt_outv=True,
                          return_final_states=True)
    Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma, Final_States = res
    # the fwd stores SSM states in bf16 by contract (mamba3_siso_fwd.py:604);
    # cast them back to F32 so the backward dots run F32.  The bf16 rounding
    # is baked into both F32 kernel lanes and is recorded in docs/mamba3-lanes.md.
    SSM_States = SSM_States.float()

    dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, _ = compute_dqkv(
        q=Q_rot, k=K_scaled, v=V, da_cs=DA_CS, da_cs_sum=DA_CS_SUM,
        qk_dot=QK_dot, SSM_States=SSM_States, do=dy_eff, D=D,
        chunk_size=64, has_input_state=False)
    dQ, dK, dQ_bias, dK_bias, dAngles_Cumsum, dScale, dGamma = compute_dqktheta(
        q=Q, k=K, scale=Scale, gamma=Gamma, q_bias=Qb, k_bias=Kb,
        angles=Angles_Cumsum, dq_in=dQ_mid, dk_in=dK_mid, dqk=dQK_dot,
        chunk_size=64)
    dDT, dTrap, _, _, _ = compute_ddt_dtrap_dinput_states(
        dscale=dScale, dgamma=dGamma, dt=DT, trap=Trap.float(),
        d_issm_state=None)
    dAngles, dDT_angle, _ = angle_dt_bwd(
        grad_out=dAngles_Cumsum, angle=Ang, dt=DT,
        has_init_state=False, chunk_size=64)
    dDT = dDT + dDT_angle

    bh2bl = lambda t: t.permute(0, 2, 1).contiguous()  # (b, nheads, seqlen) -> (b, seqlen, nheads)
    data = {
        "out_v": Out_v.cpu().numpy().astype(np.float32).reshape(-1),
        "states": Final_States[0].cpu().numpy().astype(np.float32).reshape(-1),
        "q": Q_rot.cpu().numpy().astype(np.float32).reshape(-1),
        "k": K_scaled.cpu().numpy().astype(np.float32).reshape(-1),
        "qk": bh2bl(QK_dot).cpu().numpy().astype(np.float32).reshape(-1),
        "scale": bh2bl(Scale).cpu().numpy().astype(np.float32).reshape(-1),
        "gamma": bh2bl(Gamma).cpu().numpy().astype(np.float32).reshape(-1),
        "dV": dV.cpu().numpy().astype(np.float32).reshape(-1),
        "dADT": bh2bl(dADT).cpu().numpy().astype(np.float32).reshape(-1),
        "dDT": bh2bl(dDT).cpu().numpy().astype(np.float32).reshape(-1),
        "dTrap": bh2bl(dTrap).cpu().numpy().astype(np.float32).reshape(-1),
        "dQ_bias": dQ_bias.cpu().numpy().astype(np.float32).reshape(-1),
        "dK_bias": dK_bias.cpu().numpy().astype(np.float32).reshape(-1),
        "dAngles": dAngles.sum(2).cpu().numpy().astype(np.float32).reshape(-1),
        "dD": dD.cpu().numpy().astype(np.float32).reshape(-1),
        "dQ": dQ.reshape(1, Tloc, 16).cpu().numpy().astype(np.float32).reshape(-1),
        "dK": dK.reshape(1, Tloc, 16).cpu().numpy().astype(np.float32).reshape(-1),
    }
    np.savez(os.path.join(SCRATCH, f"kernel_{lane}_{T}.npz"), **data)
    print(f"wrote kernel_{lane}_{T}.npz (T={T}, TRITON_F32_DEFAULT={lane})")


def dump_bf16(T):
    os.makedirs(SCRATCH, exist_ok=True)
    import mamba3_twin as W
    _, m = build_block()
    u_full, dy_full = build_input()
    u = u_full[:, :T].clone()
    dy = dy_full[:, :T].contiguous()
    for p in m.parameters():
        p.grad = None
    u3 = u.detach().clone().requires_grad_(True)
    out = m(u3)
    (out * dy).sum().backward()
    data = {"out": out.detach().cpu().numpy().astype(np.float32).reshape(-1),
            "du": u3.grad.detach().cpu().numpy().astype(np.float32).reshape(-1)}
    names = {"dW_in": m.in_proj.weight, "ddt_bias": m.dt_bias,
             "dB_bias": m.B_bias, "dC_bias": m.C_bias,
             "dB_norm_w": m.B_norm.weight, "dC_norm_w": m.C_norm.weight,
             "dD": m.D, "dW_out": m.out_proj.weight}
    for k, p in names.items():
        data[k] = p.grad.detach().cpu().numpy().astype(np.float32).reshape(-1)
    np.savez(os.path.join(SCRATCH, f"bf16_{T}.npz"), **data)
    print(f"wrote bf16_{T}.npz (T={T})")


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def compare():
    rows = []
    for T in TS:
        twin = load(os.path.join(SCRATCH, f"twin_{T}.npz"))
        raw = {"ieee": os.path.join(SCRATCH, f"kernel_ieee_{T}.npz"),
               "tf32": os.path.join(SCRATCH, f"kernel_tf32_{T}.npz"),
               "bf16": os.path.join(SCRATCH, f"bf16_{T}.npz")}
        k_ieee, k_tf32, b16 = (load(raw[k]) for k in ("ieee", "tf32", "bf16"))
        row = [T]

        # lane ieee, forward lanes
        # forward: kernel vs twin
        row.append(rel(k_ieee["out_v"] - twin["out_v"], twin["out_v"]))
        row.append(rel(k_tf32["out_v"] - twin["out_v"], twin["out_v"]))
        row.append(rel(k_ieee["q"] - twin["q"], twin["q"]))
        row.append(rel(k_tf32["q"] - twin["q"], twin["q"]))
        # K_store = rope(K + bias) * scale; the twin stores unscaled k
        t_scale = twin["scale"].reshape(1, T, H)[:, :, :, None]
        t_k_scaled = (twin["k"].reshape(1, T, H, QK_DIM) * t_scale).reshape(-1)
        row.append(rel(k_ieee["k"] - t_k_scaled, t_k_scaled))
        row.append(rel(k_tf32["k"] - t_k_scaled, t_k_scaled))
        row.append(rel(k_ieee["qk"] - twin["qk"], twin["qk"]))
        row.append(rel(k_tf32["qk"] - twin["qk"], twin["qk"]))
        row.append(rel(k_ieee["scale"] - twin["scale"], twin["scale"]))
        row.append(rel(k_tf32["scale"] - twin["scale"], twin["scale"]))
        row.append(rel(k_ieee["gamma"] - twin["gamma"], twin["gamma"]))
        row.append(rel(k_tf32["gamma"] - twin["gamma"], twin["gamma"]))
        # module vs twin final out
        row.append(rel(b16["out"] - twin["y"], twin["y"]))
        # states
        row.append(rel(k_ieee["states"] - twin["states"], twin["states"]))
        row.append(rel(k_tf32["states"] - twin["states"], twin["states"]))
        # kernel backward pairs
        row.append(rel(k_ieee["dQ"] - twin["g_Cn"], twin["g_Cn"]))
        row.append(rel(k_tf32["dQ"] - twin["g_Cn"], twin["g_Cn"]))
        row.append(rel(k_ieee["dK"] - twin["g_Bn"], twin["g_Bn"]))
        row.append(rel(k_tf32["dK"] - twin["g_Bn"], twin["g_Bn"]))
        row.append(rel(k_ieee["dADT"] - twin["g_ADT"], twin["g_ADT"]))
        row.append(rel(k_tf32["dADT"] - twin["g_ADT"], twin["g_ADT"]))
        # the twin's DT grad includes the ADT path (ADT = A*DT); the kernel
        # returns it separately as dADT.  Compare the direct part.
        g_DT_direct = twin["g_DT"] - twin["A"] * twin["g_ADT"]
        row.append(rel(k_ieee["dDT"] - g_DT_direct, g_DT_direct))
        row.append(rel(k_tf32["dDT"] - g_DT_direct, g_DT_direct))
        row.append(rel(k_ieee["dTrap"] - twin["g_trap_raw"], twin["g_trap_raw"]))
        row.append(rel(k_tf32["dTrap"] - twin["g_trap_raw"], twin["g_trap_raw"]))
        row.append(rel(k_ieee["dAngles"] - twin["g_ang_raw"], twin["g_ang_raw"]))
        row.append(rel(k_tf32["dAngles"] - twin["g_ang_raw"], twin["g_ang_raw"]))
        row.append(rel(k_ieee["dD"] - twin["dD"], twin["dD"]))
        row.append(rel(k_tf32["dD"] - twin["dD"], twin["dD"]))
        row.append(rel(k_ieee["dQ_bias"] - twin["dC_bias"], twin["dC_bias"]))
        row.append(rel(k_tf32["dQ_bias"] - twin["dC_bias"], twin["dC_bias"]))
        row.append(rel(k_ieee["dK_bias"] - twin["dB_bias"], twin["dB_bias"]))
        row.append(rel(k_tf32["dK_bias"] - twin["dB_bias"], twin["dB_bias"]))
        # bf16 module backward (whole module)
        row.append(rel(b16["du"] - twin["du"], twin["du"]))
        row.append(rel(b16["dW_in"] - twin["dW_in"], twin["dW_in"]))
        row.append(rel(b16["ddt_bias"] - twin["ddt_bias"], twin["ddt_bias"]))
        row.append(rel(b16["dB_bias"] - twin["dB_bias"], twin["dB_bias"]))
        row.append(rel(b16["dC_bias"] - twin["dC_bias"], twin["dC_bias"]))
        row.append(rel(b16["dB_norm_w"] - twin["dB_norm_w"], twin["dB_norm_w"]))
        row.append(rel(b16["dC_norm_w"] - twin["dC_norm_w"], twin["dC_norm_w"]))
        row.append(rel(b16["dD"] - twin["dD"], twin["dD"]))
        row.append(rel(b16["dW_out"] - twin["dW_out"], twin["dW_out"]))
        rows.append(row)

    cols = ["T", "ieee out_v", "tf32 out_v",
            "ieee q", "tf32 q", "ieee k", "tf32 k",
            "ieee qk", "tf32 qk", "ieee scale", "tf32 scale",
            "ieee gamma", "tf32 gamma",
            "module out",
            "ieee states", "tf32 states",
            "ieee dQ", "tf32 dQ", "ieee dK", "tf32 dK",
            "ieee dADT", "tf32 dADT", "ieee dDT", "tf32 dDT",
            "ieee dTrap", "tf32 dTrap", "ieee dAngles", "tf32 dAngles",
            "ieee dD", "tf32 dD", "ieee dQ_bias", "tf32 dQ_bias",
            "ieee dK_bias", "tf32 dK_bias",
            "module du", "module dW_in", "module ddt_bias",
            "module dB_bias", "module dC_bias",
            "module dB_norm_w", "module dC_norm_w",
            "module dD", "module dW_out"]
    header = "| " + " | ".join(cols) + " |\n"
    header += "|" + "---|" * len(cols) + "\n"
    body = ""
    for row in rows:
        cells = []
        for c, v in zip(cols, row):
            cells.append(f"{v:.3e}" if v is not None else "n/a")
        body += "| " + " | ".join(cells) + " |\n"
    print(header + body)


def collect_max(T_loads):
    """Return {lane: max rel over all tensors and all T} from loaded dumps."""
    maxima = {"ieee": 0.0, "tf32": 0.0, "bf16-module": 0.0}
    for T, twin, k_ieee, k_tf32, b16 in T_loads:
        vals = [
            ("ieee", rel(k_ieee["out_v"] - twin["out_v"], twin["out_v"])),
            ("tf32", rel(k_tf32["out_v"] - twin["out_v"], twin["out_v"])),
            ("ieee", rel(k_ieee["q"] - twin["q"], twin["q"])),
            ("tf32", rel(k_tf32["q"] - twin["q"], twin["q"])),
            ("ieee", rel(k_ieee["k"] - (twin["k"].reshape(1, T, H, QK_DIM) * twin["scale"].reshape(1, T, H)[:, :, :, None]).reshape(-1),
                          twin["k"])),
            ("tf32", rel(k_tf32["k"] - (twin["k"].reshape(1, T, H, QK_DIM) * twin["scale"].reshape(1, T, H)[:, :, :, None]).reshape(-1),
                          twin["k"])),
            ("ieee", rel(k_ieee["qk"] - twin["qk"], twin["qk"])),
            ("tf32", rel(k_tf32["qk"] - twin["qk"], twin["qk"])),
            ("ieee", rel(k_ieee["scale"] - twin["scale"], twin["scale"])),
            ("tf32", rel(k_tf32["scale"] - twin["scale"], twin["scale"])),
            ("ieee", rel(k_ieee["gamma"] - twin["gamma"], twin["gamma"])),
            ("tf32", rel(k_tf32["gamma"] - twin["gamma"], twin["gamma"])),
            ("ieee", rel(k_ieee["states"] - twin["states"], twin["states"])),
            ("tf32", rel(k_tf32["states"] - twin["states"], twin["states"])),
            ("bf16-module", rel(b16["out"] - twin["y"], twin["y"])),
            ("ieee", rel(k_ieee["dQ"] - twin["g_Cn"], twin["g_Cn"])),
            ("tf32", rel(k_tf32["dQ"] - twin["g_Cn"], twin["g_Cn"])),
            ("ieee", rel(k_ieee["dK"] - twin["g_Bn"], twin["g_Bn"])),
            ("tf32", rel(k_tf32["dK"] - twin["g_Bn"], twin["g_Bn"])),
            ("ieee", rel(k_ieee["dADT"] - twin["g_ADT"], twin["g_ADT"])),
            ("tf32", rel(k_tf32["dADT"] - twin["g_ADT"], twin["g_ADT"])),
            ("ieee", rel(k_ieee["dDT"] - (twin["g_DT"] - twin["A"] * twin["g_ADT"]), twin["g_DT"])),
            ("tf32", rel(k_tf32["dDT"] - (twin["g_DT"] - twin["A"] * twin["g_ADT"]), twin["g_DT"])),
            ("ieee", rel(k_ieee["dTrap"] - twin["g_trap_raw"], twin["g_trap_raw"])),
            ("tf32", rel(k_tf32["dTrap"] - twin["g_trap_raw"], twin["g_trap_raw"])),
            ("ieee", rel(k_ieee["dAngles"] - twin["g_ang_raw"], twin["g_ang_raw"])),
            ("tf32", rel(k_tf32["dAngles"] - twin["g_ang_raw"], twin["g_ang_raw"])),
            ("ieee", rel(k_ieee["dD"] - twin["dD"], twin["dD"])),
            ("tf32", rel(k_tf32["dD"] - twin["dD"], twin["dD"])),
            ("ieee", rel(k_ieee["dQ_bias"] - twin["dC_bias"], twin["dC_bias"])),
            ("tf32", rel(k_tf32["dQ_bias"] - twin["dC_bias"], twin["dC_bias"])),
            ("ieee", rel(k_ieee["dK_bias"] - twin["dB_bias"], twin["dB_bias"])),
            ("tf32", rel(k_tf32["dK_bias"] - twin["dB_bias"], twin["dB_bias"])),
            ("bf16-module", rel(b16["du"] - twin["du"], twin["du"])),
            ("bf16-module", rel(b16["dW_in"] - twin["dW_in"], twin["dW_in"])),
            ("bf16-module", rel(b16["ddt_bias"] - twin["ddt_bias"], twin["ddt_bias"])),
            ("bf16-module", rel(b16["dB_bias"] - twin["dB_bias"], twin["dB_bias"])),
            ("bf16-module", rel(b16["dC_bias"] - twin["dC_bias"], twin["dC_bias"])),
            ("bf16-module", rel(b16["dB_norm_w"] - twin["dB_norm_w"], twin["dB_norm_w"])),
            ("bf16-module", rel(b16["dC_norm_w"] - twin["dC_norm_w"], twin["dC_norm_w"])),
            ("bf16-module", rel(b16["dD"] - twin["dD"], twin["dD"])),
            ("bf16-module", rel(b16["dW_out"] - twin["dW_out"], twin["dW_out"])),
        ]
        for lane, r in vals:
            maxima[lane] = max(maxima[lane], r)
    return maxima


BOUNDS = {
    # measured 2026-11-16 on the RTX 5060 lane dumps; reasons:
    #   ieee  worst 2.5e-06 (states, T=64): IEEE dots; the kernel still stores
    #         SSM states in bf16 (mamba3_siso_fwd.py:604), which is the only
    #         rounding the lane carries. 1e-05 = 4x headroom for reruns.
    #   tf32  worst 2.4e-03 (dAngles, T=16): TF32's 10-bit mantissa. 5e-03
    #         = 2x headroom; never widened without a new measurement.
    #   bf16  worst 2.4e-02 (dB_norm_w, T=8): the public module, which casts
    #         Q/K/V/trap/angles to bf16. The bf16 lane is a lane, never the
    #         oracle; 5e-02 is a tripwire 2x above the measured worst.
    "ieee": 1e-05,
    "tf32": 5e-03,
    "bf16-module": 5e-02,
}


def gate():
    T_loads = []
    for T in TS:
        twin = load(os.path.join(SCRATCH, f"twin_{T}.npz"))
        k_ieee = load(os.path.join(SCRATCH, f"kernel_ieee_{T}.npz"))
        k_tf32 = load(os.path.join(SCRATCH, f"kernel_tf32_{T}.npz"))
        b16 = load(os.path.join(SCRATCH, f"bf16_{T}.npz"))
        T_loads.append((T, twin, k_ieee, k_tf32, b16))
    maxima = collect_max(T_loads)
    ok = True
    for lane, bound in BOUNDS.items():
        worst = maxima[lane]
        status = "ok  " if worst <= bound else "FAIL"
        if worst > bound:
            ok = False
        print(f"{status} {lane:12s} max rel {worst:.3e} (bound {bound:.0e})")
    if not ok:
        print("MAMBA3 LANES FAIL")
        sys.exit(1)
    print("MAMBA3 LANES PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["twin", "kernel", "bf16", "compare", "gate", "all"])
    ap.add_argument("--T", type=int, default=0)
    args = ap.parse_args()
    for T in (TS if args.T == 0 else [args.T]):
        if args.mode in ("twin", "all"):
            dump_twin(T)
        if args.mode in ("kernel", "all"):
            dump_kernel(T)
        if args.mode in ("bf16", "all"):
            dump_bf16(T)
    if args.mode in ("compare", "all"):
        compare()
    if args.mode == "gate":
        gate()


if __name__ == "__main__":
    main()
