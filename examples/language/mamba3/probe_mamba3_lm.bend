# The full two-layer Mamba-3 LM: forward, ten SGD steps, generation.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  "reference/mamba3_lm_data"

def ml1x64(ml: TA.ML<1n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def run(+w1: List<&2, F32>, +dt1: List<&2, F32>,
    +bb1: List<&2, F32>, +cb1: List<&2, F32>,
    +bw1: List<&2, F32>, +cw1: List<&2, F32>,
    +D1: List<&2, F32>, +wo1: List<&2, F32>,
    +w2: List<&2, F32>, +dt2: List<&2, F32>,
    +bb2: List<&2, F32>, +cb2: List<&2, F32>,
    +bw2: List<&2, F32>, +cw2: List<&2, F32>,
    +D2: List<&2, F32>, +wo2: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>,
    +nw: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    M.show_lm(M.lm_forward(M.LM{w1, dt1, bb1, cb1, bw1, cw1, D1,
      wo1, w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, E, Wh, nw},
      M.prompt8(), 8n))
    M.show_lm_train(M.lm_train10(M.LM{w1, dt1, bb1, cb1, bw1, cw1,
      D1, wo1, w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, E, Wh, nw},
      0.001))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    p1w : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/l1_in_proj.bin"), 0)
    p1t : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l1_dt_bias.bin"), 0)
    p1b : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l1_B_bias.bin"), 0)
    p1c : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l1_C_bias.bin"), 0)
    p1w2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l1_B_norm_w.bin"), 0)
    p1c2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l1_C_norm_w.bin"), 0)
    p1d : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l1_D.bin"), 0)
    p1o : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/l1_out_proj.bin"), 0)
    p2w : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/l2_in_proj.bin"), 0)
    p2t : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l2_dt_bias.bin"), 0)
    p2b : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l2_B_bias.bin"), 0)
    p2c : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l2_C_bias.bin"), 0)
    p2w2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l2_B_norm_w.bin"), 0)
    p2c2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l2_C_norm_w.bin"), 0)
    p2d : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l2_D.bin"), 0)
    p2o : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/l2_out_proj.bin"), 0)
    e0 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/E0.bin"), 0)
    wh : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/Wh0.bin"), 0)
    nw : TA.Mat<1n, 64n> <- T.Mat.load(1n, 64n, String.append(ddir(), "/nw0.bin"), 0)
    run(M.ml304(TA.Mat.to_list(304n, 64n, p1w)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, p1t)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, p1b)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, p1c)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, p1w2)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, p1c2)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, p1d)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, p1o)),
      M.ml304(TA.Mat.to_list(304n, 64n, p2w)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, p2t)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, p2b)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, p2c)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, p2w2)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, p2c2)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, p2d)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, p2o)),
      M.ml64(TA.Mat.to_list(8n, 64n, e0)),
      M.ml64(TA.Mat.to_list(8n, 64n, wh)),
      ml1x64(TA.Mat.to_list(1n, 64n, nw)))
    return Unit{}
