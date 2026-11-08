# Learned-query mix forward: vh, xc, kh, kx are ramps, q = 0.1 everywhere.
# The Python twin computes the same softmax-weighted sum; the probe prints the
# first 8 output cells.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model.bend as M

def mk(+n: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      mk(p, Nat.add(i, 1n), Con{F32.from_nat(i), acc})

def mk2(+n: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      mk2(p, Nat.add(i, 1n), Con{(2.0 * F32.from_nat(i) : F32), acc})

def mk3(+n: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      mk3(p, Nat.add(i, 1n), Con{(3.0 * F32.from_nat(i) : F32), acc})

def mk0(+n: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      mk0(p, Con{0.0, acc})

def mk1(+n: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      mk1(p, Con{1.0, acc})

def mkq(+n: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      mkq(p, Con{0.10000000, acc})

def show8(+n: Nat, +xs: List<&2, F32>, +acc: String) -> String:
  match n xs:
    case 0n _:
      acc
    case 1n+p Con{h, t}:
      show8(p, t, String.append(acc, String.append(F32.show(h), " ")))
    case _ _:
      acc

def show(y: TA.Mat<128n, 8n>) -> IO(Unit):
  ys = M.ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, y))
  do IO<Unit>:
    Unit <- IO.print(String.append("first: ", show8(8n, ys, "")))
    return Unit{}

def main() -> IO(Unit):
  vh = TA.Mat.from_list(128n, 8n, mk(1024n, 0n, Nil{}))
  xc = TA.Mat.from_list(128n, 8n, mk3(1024n, 0n, Nil{}))
  kh = TA.Mat.from_list(128n, 8n, mk(1024n, 0n, Nil{}))
  kx = TA.Mat.from_list(128n, 8n, mk(1024n, 0n, Nil{}))
  q = TA.Mat.from_list(1n, 8n, mkq(8n, Nil{}))
  do IO<Unit>:
    Unit <- show(M.mix_forward(M.MXS{vh, xc, kh, kx, q}))
    return Unit{}
