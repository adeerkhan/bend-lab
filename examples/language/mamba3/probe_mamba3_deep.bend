# Depth 4: four independent SISO blocks, a residual around each, a final
# RMSNorm and the head.  Weights dumped from
# `reference/mamba3_deep_golden.npz` into `.scratch/deep_data`.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/deep_data"

def ml1x64(ml: TA.ML<1n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs


def show_lmo(o: M.LMO) -> IO(Unit):
  match o:
    case M.LMO{hn, logits, loss}:
      do IO<Unit>:
        M.show_tag("deep_logits", logits)
        M.show_tag("deep_loss", Con{loss, Nil{}})
        M.show_tag("deep_hn", hn)
        return Unit{}


def main() -> IO(Unit):
  do IO<Unit>:
    wi : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/l1_in_proj.bin"), 0)
    dtb : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l1_dt_bias.bin"), 0)
    bb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l1_B_bias.bin"), 0)
    cb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l1_C_bias.bin"), 0)
    bw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l1_B_norm_w.bin"), 0)
    cw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l1_C_norm_w.bin"), 0)
    dd : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l1_D.bin"), 0)
    wo : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/l1_out_proj.bin"), 0)
    l1 : M.L4W = M.L4W{M.ml304(TA.Mat.to_list(304n, 64n, wi)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo))}
    wi2 : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/l2_in_proj.bin"), 0)
    dtb2 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l2_dt_bias.bin"), 0)
    bb2 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l2_B_bias.bin"), 0)
    cb2 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l2_C_bias.bin"), 0)
    bw2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l2_B_norm_w.bin"), 0)
    cw2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l2_C_norm_w.bin"), 0)
    dd2 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l2_D.bin"), 0)
    wo2 : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/l2_out_proj.bin"), 0)
    l2 : M.L4W = M.L4W{M.ml304(TA.Mat.to_list(304n, 64n, wi2)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb2)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb2)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb2)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw2)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw2)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd2)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo2))}
    wi3 : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/l3_in_proj.bin"), 0)
    dtb3 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l3_dt_bias.bin"), 0)
    bb3 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l3_B_bias.bin"), 0)
    cb3 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l3_C_bias.bin"), 0)
    bw3 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l3_B_norm_w.bin"), 0)
    cw3 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l3_C_norm_w.bin"), 0)
    dd3 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l3_D.bin"), 0)
    wo3 : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/l3_out_proj.bin"), 0)
    l3 : M.L4W = M.L4W{M.ml304(TA.Mat.to_list(304n, 64n, wi3)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb3)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb3)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb3)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw3)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw3)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd3)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo3))}
    wi4 : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/l4_in_proj.bin"), 0)
    dtb4 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l4_dt_bias.bin"), 0)
    bb4 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l4_B_bias.bin"), 0)
    cb4 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l4_C_bias.bin"), 0)
    bw4 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l4_B_norm_w.bin"), 0)
    cw4 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l4_C_norm_w.bin"), 0)
    dd4 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l4_D.bin"), 0)
    wo4 : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/l4_out_proj.bin"), 0)
    l4 : M.L4W = M.L4W{M.ml304(TA.Mat.to_list(304n, 64n, wi4)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb4)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb4)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb4)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw4)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw4)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd4)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo4))}
    e0 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/E.bin"), 0)
    wh : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/Wh.bin"), 0)
    nw : TA.Mat<1n, 64n> <- T.Mat.load(1n, 64n, String.append(ddir(), "/nw.bin"), 0)
    o : M.LMO = M.lm4_forward(M.LM4{l1, l2, l3, l4,
      M.ml64(TA.Mat.to_list(8n, 64n, e0)),
      M.ml64(TA.Mat.to_list(8n, 64n, wh)),
      ml1x64(TA.Mat.to_list(1n, 64n, nw))}, M.prompt8(), 8n)
    show_lmo(o)
    return Unit{}
