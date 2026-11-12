# The same block at seqlen 4 and 16.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  "reference/mamba3_data"

def ml4(ml: TA.ML<4n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml16(ml: TA.ML<16n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def run4(+w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +u4: List<&2, F32>,
    +u16: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    M.show_f(M.forward(u4, w_in, dt_bias, B_bias, C_bias, Bw, Cw, D,
      w_out, 4n), "y4")
    M.show_f(M.forward(u16, w_in, dt_bias, B_bias, C_bias, Bw, Cw, D,
      w_out, 16n), "y16")
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
    u4 : TA.Mat<4n, 64n> <- T.Mat.load(4n, 64n, String.append(ddir(), "/u4.bin"), 0)
    u16 : TA.Mat<16n, 64n> <- T.Mat.load(16n, 64n, String.append(ddir(), "/u16.bin"), 0)
    run4(M.ml304(TA.Mat.to_list(304n, 64n, wi)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo)),
      ml4(TA.Mat.to_list(4n, 64n, u4)),
      ml16(TA.Mat.to_list(16n, 64n, u16)))
    return Unit{}
