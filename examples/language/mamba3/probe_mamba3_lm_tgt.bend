# Next-token loss with explicit targets: what a byte LM needs.
#
# `loss_go`/`dy_all_go`/`dWh_go` target the row index (the copy task).
# These take the target list from a file instead. The gate is
# `reference/test_mamba3_lm_tgt.py`: same y, Wh and targets on both sides,
# NumPy float64 against the Bend float32 kernels.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/mamba3_tgt"

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def ml64(ml: TA.ML<8n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def main() -> IO(Unit):
  do IO<Unit>:
    yy : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/y.bin"), 0)
    wh : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/Wh.bin"), 0)
    ts : List<&2, U32> <- T.read_at(String.append(ddir(), "/targets.bin"), 0, 8)
    +targets : List<&2, Nat> <- IO.pure(List<&2, Nat>, nats(ts))
    +y : List<&2, F32> <- IO.pure(List<&2, F32>, ml64(TA.Mat.to_list(8n, 64n, yy)))
    +Wh : List<&2, F32> <- IO.pure(List<&2, F32>, ml64(TA.Mat.to_list(8n, 64n, wh)))
    +inv : F32 <- IO.pure(F32, (1.0 / 8.0 : F32))
    M.show_tag("tgt_loss", Con{M.loss_tgt_go(8n, 0n, y, Wh, targets, 8n, inv, 0.0), Nil{}})
    M.show_tag("tgt_dy", M.dy_all_tgt_go(8n, 0n, y, Wh, targets, 8n, inv, Nil{}))
    M.show_tag("tgt_dwh", M.dWh_tgt_go(8n, 0n, y, Wh, targets, 8n, inv, M.zline(512n)))
    return Unit{}
