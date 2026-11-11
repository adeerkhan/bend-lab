# Forward, twenty times in one process: the slope is the per-call cost.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  "reference/mamba3_data"

def rep(+left: Nat, +acc: F32, +u: List<&2, F32>,
    +wi: List<&2, F32>, +dtb: List<&2, F32>,
    +bb: List<&2, F32>, +cb: List<&2, F32>,
    +bw: List<&2, F32>, +cw: List<&2, F32>,
    +dd: List<&2, F32>, +wo: List<&2, F32>) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      +c = M.sum_fwd(M.forward(u, wi, dtb, bb, cb, bw, cw, dd, wo, 8n))
      rep(p, (acc + c : F32), u, wi, dtb, bb, cb, bw, cw, dd, wo)

def main() -> IO(Unit):
  do IO<Unit>:
    u : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/u.bin"), 0)
    wi : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/in_proj.bin"), 0)
    dtb : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/dt_bias.bin"), 0)
    bb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/B_bias.bin"), 0)
    cb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/C_bias.bin"), 0)
    bw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/B_norm_w.bin"), 0)
    cw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/C_norm_w.bin"), 0)
    dd : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/D.bin"), 0)
    wo : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/out_proj.bin"), 0)
    M.show_tag("checksum", Con{rep(20n, 0.0,
      M.ml64(TA.Mat.to_list(8n, 64n, u)),
      M.ml304(TA.Mat.to_list(304n, 64n, wi)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo))), Nil{}})
    return Unit{}
