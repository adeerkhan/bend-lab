# The packed [u ; u] input with a state reset at the boundary (varlen).
#
# Each segment must reproduce the twin-gated 8-row golden exactly: the
# stateful passes never see the other segment's rows, and the angle/SSM/
# trapezoid states reset at the boundary.  The gate is
# `reference/test_mamba3_varlen.py`.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  "reference/mamba3_data"

def show_vl(v: M.VL) -> IO(Unit):
  match v:
    case M.VL{ya, yb, sa, sb, aa, ab}:
      do IO<Unit>:
        M.show_tag("varlen_y_a", ya)
        M.show_tag("varlen_y_b", yb)
        M.show_tag("varlen_states_a", sa)
        M.show_tag("varlen_states_b", sb)
        M.show_tag("varlen_angle_a", aa)
        M.show_tag("varlen_angle_b", ab)
        return Unit{}

def run_vl(+w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +u: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    show_vl(M.forward_vl(List.append(&2, F32, u, u), w_in, dt_bias,
      B_bias, C_bias, Bw, Cw, D, w_out, 16n, 8n))
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
    u8 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/u.bin"), 0)
    run_vl(M.ml304(TA.Mat.to_list(304n, 64n, wi)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo)),
      M.ml64(TA.Mat.to_list(8n, 64n, u8)))
    return Unit{}
