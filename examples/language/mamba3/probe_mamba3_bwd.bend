# Pass 1 of the Mamba-3 backward: the output gate and the scan.
#
# Prints do_v, dz, dx, dqk, dscale and dADT for comparison with the twin's
# retained gradients (g_out_v, g_z, g_x, g_qk, g_scale, g_ADT).
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  "reference/mamba3_data"

def main() -> IO(Unit):
  do IO<Unit>:
    u : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/u.bin"), 0)
    dy : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/dy.bin"), 0)
    wi : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/in_proj.bin"), 0)
    dtb : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/dt_bias.bin"), 0)
    bb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/B_bias.bin"), 0)
    cb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/C_bias.bin"), 0)
    bw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/B_norm_w.bin"), 0)
    cw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/C_norm_w.bin"), 0)
    dd : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/D.bin"), 0)
    wo : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/out_proj.bin"), 0)
    wo2 : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/out_proj.bin"), 0)
    wo3 : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/out_proj.bin"), 0)
    dy3 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/dy.bin"), 0)
    dd2 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/D.bin"), 0)
    M.show_tag("dyi0", M.matvec_t(M.ml64x128(TA.Mat.to_list(64n, 128n, wo3)),
      M.take_row(M.ml64(TA.Mat.to_list(8n, 64n, dy3)), 0n, 64n), 64n, 128n))
    M.show_bw(M.bw1(M.forward(M.ml64(TA.Mat.to_list(8n, 64n, u)),
      M.ml304(TA.Mat.to_list(304n, 64n, wi)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo)), 8n),
      M.ml64(TA.Mat.to_list(8n, 64n, dy)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo2)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd2)), 8n))
    return Unit{}
