# The q.S^T product of one Mamba-3 head, with bend-ml Mat.matmul_nt.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ml64(ml: TA.ML<8n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}: xs

def ml9x2048(ml: TA.ML<9n, 2048n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}: xs

def ml1x16(ml: TA.ML<1n, 16n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}: xs

def ml32x16(ml: TA.ML<32n, 16n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}: xs

def ml1x32(ml: TA.ML<1n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}: xs

def mm_c(r: TA.MMulNT<1n, 16n, 32n>) -> List<&2, F32>:
  match r:
    case TA.MMulNT{a, b, c}: ml1x32(TA.Mat.to_list(1n, 32n, c))

def qs(q: TA.Mat<1n, 16n>, s: TA.Mat<32n, 16n>) -> List<&2, F32>:
  mm_c(TA.Mat.matmul_nt(1n, 16n, 32n, 0n, q, s))

def rep(+left: Nat, +acc: F32, +ql: List<&2, F32>, +sl: List<&2, F32>) -> F32:
  match left:
    case 0n: acc
    case 1n+pr:
      q : TA.Mat<1n, 16n> = TA.Mat.from_list(1n, 16n, ql)
      s : TA.Mat<32n, 16n> = TA.Mat.from_list(32n, 16n, sl)
      rep(pr, (acc + M.sum_list(qs(q, s)) : F32), ql, sl)

def run(qm: TA.Mat<8n, 64n>, sm: TA.Mat<9n, 2048n>) -> IO(Unit):
  +ql = M.take_row(ml64(TA.Mat.to_list(8n, 64n, qm)), 192n, 16n)
  +sl = M.take_row(ml9x2048(TA.Mat.to_list(9n, 2048n, sm)), 6144n, 512n)
  do IO<Unit>:
    M.show_tag("qs", qs(TA.Mat.from_list(1n, 16n, ql),
      TA.Mat.from_list(32n, 16n, sl)))
    M.show_tag("sum", Con{rep(20000n, 0.0, ql, sl), Nil{}})
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    qm : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, "reference/mamba3_data/q.bin", 0)
    sm : TA.Mat<9n, 2048n> <- T.Mat.load(9n, 2048n, "reference/mamba3_data/states.bin", 0)
    run(qm, sm)
    return Unit{}
