# The q.S^T product of one Mamba-3 head, with list rows.
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

def qs_go(+left: Nat, +p: Nat, +s: List<&2, F32>, +q: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n: acc
    case 1n+pr:
      +d = M.dot(M.take_row(s, Nat.mul(p, 16n), 16n), q)
      qs_go(pr, (p + 1n : Nat), s, q, M.push_row(acc, Con{d, Nil{}}))

def qs(s: List<&2, F32>, q: List<&2, F32>) -> List<&2, F32>:
  qs_go(32n, 0n, s, q, Nil{})

def rep(+left: Nat, +acc: F32, +s: List<&2, F32>, +q: List<&2, F32>) -> F32:
  match left:
    case 0n: acc
    case 1n+pr:
      rep(pr, (acc + M.sum_list(qs(s, q)) : F32), s, q)

def run(qm: TA.Mat<8n, 64n>, sm: TA.Mat<9n, 2048n>) -> IO(Unit):
  +ql = M.take_row(ml64(TA.Mat.to_list(8n, 64n, qm)), 192n, 16n)
  +sl = M.take_row(ml9x2048(TA.Mat.to_list(9n, 2048n, sm)), 6144n, 512n)
  do IO<Unit>:
    M.show_tag("qs", qs(sl, ql))
    M.show_tag("sum", Con{rep(20000n, 0.0, sl, ql), Nil{}})
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    qm : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, "reference/mamba3_data/q.bin", 0)
    sm : TA.Mat<9n, 2048n> <- T.Mat.load(9n, 2048n, "reference/mamba3_data/states.bin", 0)
    run(qm, sm)
    return Unit{}
