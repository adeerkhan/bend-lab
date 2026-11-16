"""The kernel's explanatory cases (row A3): T = 1, T = 2, T = 64.

  TRITON_F32_DEFAULT=ieee reference/.venv-rf/bin/python reference/mamba3_cases.py kernel
  reference/.venv-rf/bin/python reference/mamba3_cases.py twin
  reference/.venv-rf/bin/python reference/mamba3_cases.py module

T = 1   the SSM term is zero; out_v = (D + qk * gamma) * x.  The skip.
T = 2   token 0 writes the state; token 1 reads it.  state_1 = a_0 * 0 +
        x_0 (x) (scale_0 * k_0); o_1 = a_1 * q_1 . state_1 + (D + qk_1) * x_1.
T = 64  one full chunk (chunk_size = 64); the in-chunk cumulative decay
        DA_CS must equal cumsum(ADT * log2(e)) over the block.

Same tolerances as the A1 lane table (ieee 1e-05, module 5e-02).
Prints one line per check; the gate (reference/test_mamba3_cases.py)
checks them.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
import mamba3_twin as W  # noqa: E402

TS = (1, 2, 64)
LOG2E = 1.44269504089


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max()) / max(1e-30, float(np.abs(b).max()))


def build():
    torch.manual_seed(71)
    torch.set_num_threads(1)
    m = W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32,
                 ngroups=1, is_mimo=False).cuda().float()
    torch.manual_seed(123)
    u = (torch.randn(1, 64, 64, device="cuda") * 0.5)
    return m, u


def twin_checks():
    m, u = build()
    for T in TS:
        f = W.twin_forward(m, u[:, :T].clone())
        # T >= 1: the skip-only case explains t = 0 for any T
        x4, qk = f["x"], f["qk"]
        out_v = f["out_v"]
        D = m.D.detach()
        cf = (D.view(1, 1, -1, 1) + qk[:, 0:1].unsqueeze(-1)) * x4[:, 0:1]
        r = rel(cf.detach().cpu().numpy(), out_v[:, 0:1].detach().cpu().numpy())
        print(f"twin T={T} skip(0)                max rel {r:.3e}")
        if T >= 2:
            states = f["states"]
            ADT = f["ADT"]
            a1 = torch.exp(ADT[:, 1])
            a0 = torch.exp(ADT[:, 0])
            # token 0 -> state after token 0 (states[:, 1])
            k = f["k"]
            scale = f["scale"]
            st1 = (a0[..., None, None] * torch.zeros(1, 4, 32, 16, device=u.device)
                   + x4[:, 0].unsqueeze(-1)
                   * (scale[:, 0].unsqueeze(-1) * k[:, 0]).unsqueeze(-2))
            r2 = rel(st1.detach().cpu().numpy(), states[:, 1].detach().cpu().numpy())
            o1 = (a1[..., None, None] * (f["q"][:, 1].unsqueeze(-2)
                                        @ st1.transpose(-1, -2))).squeeze(-2)
            cf1 = o1 + (D.view(1, 1, -1, 1) + qk[:, 1:2].unsqueeze(-1)) * x4[:, 1:2]
            r3 = rel(cf1.detach().cpu().numpy(), out_v[:, 1:2].detach().cpu().numpy())
            print(f"twin T={T} state_1 enters        max rel {r2:.3e}")
            print(f"twin T={T} read at t=1           max rel {r3:.3e}")


def kernel_checks():
    lane = os.environ.get("TRITON_F32_DEFAULT", "tf32")
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
    m, u = build()
    for T in TS:
        f = W.twin_forward(m, u[:, :T].clone())
        Tloc = T
        Q = f["Cn"].view(1, Tloc, 1, 16).detach()
        K = f["Bn"].view(1, Tloc, 1, 16).detach()
        V = f["x"].detach()
        ADT = f["ADT"].permute(0, 2, 1).contiguous()
        DT = f["DT"].permute(0, 2, 1).contiguous()
        Trap = f["trap_raw"].permute(0, 2, 1).contiguous()
        Qb = m.C_bias.squeeze(1).detach()
        Kb = m.B_bias.squeeze(1).detach()
        Ang = f["ang_raw"].unsqueeze(2).expand(1, Tloc, 4, f["ang_raw"].shape[-1]).contiguous()
        Angles_Cumsum, _ = angle_dt_fwd(Ang, DT, init_state=None,
                                        chunk_size=64, return_output_state=True)
        res = mamba3_siso_fwd(Q, K, V, ADT, DT, Trap, Qb, Kb, Angles_Cumsum,
                              m.D.detach(), Z=None, chunk_size=64,
                              store_states_adt_outv=True, return_final_states=True)
        Out_v, DA_CS = res[1], res[3]
        r = rel(Out_v.cpu().numpy(), f["out_v"].detach().cpu().numpy())
        print(f"kernel({lane}) T={T} out_v vs twin   max rel {r:.3e}")
        if T == 64:
            ref = torch.cumsum(ADT.detach().float() * LOG2E, dim=-1)
            r2 = rel(DA_CS.cpu().numpy(), ref.cpu().numpy())
            print(f"kernel({lane}) T=64 DA_CS vs cumsum max rel {r2:.3e}")


def module_checks():
    m, u = build()
    for T in TS:
        uT = u[:, :T].clone()
        with torch.no_grad():
            out = m(uT)
        f = W.twin_forward(m, uT)
        r = rel(out.cpu().numpy(), f["y"].detach().cpu().numpy())
        print(f"module T={T} out vs twin y      max rel {r:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["twin", "kernel", "module", "all"])
    args = ap.parse_args()
    if args.mode in ("twin", "all"):
        twin_checks()
    if args.mode in ("kernel", "all"):
        kernel_checks()
    if args.mode in ("module", "all"):
        module_checks()


if __name__ == "__main__":
    main()
