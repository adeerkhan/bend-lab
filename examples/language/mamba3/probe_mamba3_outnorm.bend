# outproj norm (is_outproj_norm=True): ungated out_v, gated grouped RMSNorm,
# then out_proj. Weights and input dumped from
# `reference/mamba3_outnorm_golden.npz` into `.scratch/on_data`.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/on_data"

def ml64(ml: TA.ML<8n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml128(ml: TA.ML<1n, 128n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show_f(f: M.FWD) -> IO(Unit):
  match f:
    case M.FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, y}:
      do IO<Unit>:
        M.show_tag("on_y", y)
        M.show_tag("on_states", states)
        M.show_tag("on_angle", angle)
        M.show_tag("on_q", q)
        M.show_tag("on_k", k)
        M.show_tag("on_qk", qk)
        M.show_tag("on_scale", scale)
        M.show_tag("on_Bn", Bn)
        M.show_tag("on_Cn", Cn)
        M.show_tag("on_DT", DT)
        M.show_tag("on_ADT", ADT)
        M.show_tag("on_gamma", gamma)
        M.show_tag("on_ang_raw", ang_raw)
        M.show_tag("on_A", A)
        M.show_tag("on_trap", trap)
        M.show_tag("on_pre", pre)
        M.show_tag("on_out_v", out_v)
        M.show_tag("on_y_inner", y_inner)
        M.show_tag("on_Braw", Braw)
        M.show_tag("on_Craw", Craw)
        return Unit{}

def show_gon(g: M.GRADON) -> IO(Unit):
  match g:
    case M.GRADON{gr, won}:
      match gr:
        case M.GRAD{du, dW_in, ddt_bias, dB_bias, dC_bias, dB_norm_w,
            dC_norm_w, dD, dW_out, do_v, dq, dk, dz, dx, dqk, dscale,
            dADT, dBn, dCn, dDT, dtrap, ddd_dt, ddd_A, dBraw, dCraw,
            dang_raw}:
          do IO<Unit>:
            M.show_tag("on_grad_du", du)
            M.show_tag("on_grad_dW_in", dW_in)
            M.show_tag("on_grad_ddt_bias", ddt_bias)
            M.show_tag("on_grad_dB_bias", dB_bias)
            M.show_tag("on_grad_dC_bias", dC_bias)
            M.show_tag("on_grad_dB_norm_w", dB_norm_w)
            M.show_tag("on_grad_dC_norm_w", dC_norm_w)
            M.show_tag("on_grad_dD", dD)
            M.show_tag("on_grad_dW_out", dW_out)
            M.show_tag("on_grad_do_v", do_v)
            M.show_tag("on_grad_dq", dq)
            M.show_tag("on_grad_dk", dk)
            M.show_tag("on_grad_dz", dz)
            M.show_tag("on_grad_dx", dx)
            M.show_tag("on_grad_dqk", dqk)
            M.show_tag("on_grad_dscale", dscale)
            M.show_tag("on_grad_dADT", dADT)
            M.show_tag("on_grad_dBn", dBn)
            M.show_tag("on_grad_dCn", dCn)
            M.show_tag("on_grad_dDT", dDT)
            M.show_tag("on_grad_dtrap", dtrap)
            M.show_tag("on_grad_ddd_dt", ddd_dt)
            M.show_tag("on_grad_ddd_A", ddd_A)
            M.show_tag("on_grad_dBraw", dBraw)
            M.show_tag("on_grad_dCraw", dCraw)
            M.show_tag("on_grad_dang_raw", dang_raw)
            M.show_tag("on_grad_dW_on", won)
            return Unit{}

def run_all(+w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +won: List<&2, F32>,
    +u: List<&2, F32>, +dy: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    show_f(M.forward_on(u, w_in, dt_bias, B_bias, C_bias, Bw, Cw, D,
      w_out, won, 8n))
    show_gon(M.backward_on(M.forward_on(u, w_in, dt_bias, B_bias,
      C_bias, Bw, Cw, D, w_out, won, 8n), u, dy, w_in, w_out, Bw, Cw,
      dt_bias, B_bias, C_bias, D, won, 8n))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    wi : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/in_proj.bin"), 0)
    dtb : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/dt_bias.bin"), 0)
    bb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/B_bias.bin"), 0)
    cb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/C_bias.bin"), 0)
    bw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/B_norm_w.bin"), 0)
    cw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/C_norm_w.bin"), 0)
    dd : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/D.bin"), 0)
    wo : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/out_proj.bin"), 0)
    won : TA.Mat<1n, 128n> <- T.Mat.load(1n, 128n, String.append(ddir(), "/norm_w.bin"), 0)
    u8 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/u.bin"), 0)
    dy8 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/dy.bin"), 0)
    run_all(M.ml304(TA.Mat.to_list(304n, 64n, wi)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo)),
      ml128(TA.Mat.to_list(1n, 128n, won)),
      ml64(TA.Mat.to_list(8n, 64n, u8)),
      ml64(TA.Mat.to_list(8n, 64n, dy8)))
    return Unit{}
