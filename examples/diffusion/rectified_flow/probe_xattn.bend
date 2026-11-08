# XAttnRes at the toy scale, zero query: the first 8 cells of
# 0.5 * (maxpool2(history) + x). The Python twin computes the same value.
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

def show8(+n: Nat, +xs: List<&2, F32>, +acc: String) -> String:
  match n xs:
    case 0n _:
      acc
    case 1n+p Con{h, t}:
      show8(p, t, String.append(acc, String.append(F32.show(h), " ")))
    case _ _:
      acc

def show(y: M.POOL()) -> IO(Unit):
  ys = M.ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, y))
  do IO<Unit>:
    Unit <- IO.print(String.append("first: ", show8(8n, ys, "")))
    return Unit{}

def main() -> IO(Unit):
  h = TA.Mat.from_list(512n, 8n, mk(4096n, 0n, Nil{}))
  x = TA.Mat.from_list(128n, 8n, mk(1024n, 0n, Nil{}))
  show(M.mix_mat(x, M.pool_mat(h)))
