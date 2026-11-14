# GQA: ngroups 2, rope_fraction 1.0. Ablation row of
# `reference/test_mamba3_gqa.py`; weights and input are dumped from
# `reference/mamba3_gqa_golden.npz` into `.scratch/gqa_data`.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/gqa_data"

def ml340(ml: TA.ML<340n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml64(ml: TA.ML<8n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show_gqa(f: M.FWD) -> IO(Unit):
  match f:
    case M.FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, y}:
      do IO<Unit>:
        M.show_tag("gqa_y", y)
        M.show_tag("gqa_states", states)
        M.show_tag("gqa_angle", angle)
        M.show_tag("gqa_q", q)
        M.show_tag("gqa_k", k)
        M.show_tag("gqa_qk", qk)
        M.show_tag("gqa_scale", scale)
        M.show_tag("gqa_Bn", Bn)
        M.show_tag("gqa_Cn", Cn)
        M.show_tag("gqa_DT", DT)
        M.show_tag("gqa_ADT", ADT)
        M.show_tag("gqa_gamma", gamma)
        M.show_tag("gqa_ang_raw", ang_raw)
        M.show_tag("gqa_A", A)
        M.show_tag("gqa_trap", trap)
        M.show_tag("gqa_pre", pre)
        M.show_tag("gqa_out_v", out_v)
        M.show_tag("gqa_y_inner", y_inner)
        M.show_tag("gqa_Braw", Braw)
        M.show_tag("gqa_Craw", Craw)
        return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    wi : TA.Mat<340n, 64n> <- T.Mat.load(340n, 64n, String.append(ddir(), "/in_proj.bin"), 0)
    dtb : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/dt_bias.bin"), 0)
    bb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/B_bias.bin"), 0)
    cb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/C_bias.bin"), 0)
    bw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/B_norm_w.bin"), 0)
    cw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/C_norm_w.bin"), 0)
    dd : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/D.bin"), 0)
    wo : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/out_proj.bin"), 0)
    u8 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/u.bin"), 0)
    show_gqa(M.forward_gqa(ml64(TA.Mat.to_list(8n, 64n, u8)),
      ml340(TA.Mat.to_list(340n, 64n, wi)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo)), 8n))
    return Unit{}
