# Pool2 backward: the output gradient is a ramp, so the input gradient at the
# saved arg-max of cell c is c itself. The first 8 cells of the first block are
# the input positions 136..143, so the probe prints 0 1 2 3 4 5 6 7.
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

def tk(+n: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match n xs:
    case 0n _:
      acc
    case 1n+p Con{h, t}:
      tk(p, t, Con{h, acc})
    case _ _:
      acc

def dr(+n: Nat, +xs: List<&2, F32>) -> List<&2, F32>:
  match n xs:
    case 0n _:
      xs
    case 1n+p Con{h, t}:
      dr(p, t)
    case _ _:
      Nil{}

def show_at(+off: Nat, +xs: List<&2, F32>) -> String:
  show8(8n, List.reverse(&2, F32, tk(8n, dr(off, xs), Nil{})), "")

def un_b(b: T.Bwd<Unit, M.IN()>) -> IO(Unit):
  match b:
    case T.Bwd{_, dx, _}:
      xs = M.ml_list(512n, 8n, TA.Mat.to_list(512n, 8n, dx))
      do IO<Unit>:
        Unit <- IO.print(String.append("dx@136: ", show_at(136n, xs)))
        return Unit{}

def un_f(r: T.Fwd<Unit, M.POOL(), M.PS>) -> IO(Unit):
  match r:
    case T.Fwd{_, y, s}:
      dy = TA.Mat.from_list(128n, 8n, mk(1024n, 0n, Nil{}))
      un_b(T.Layer.bwd(Unit, M.IN(), M.POOL(), M.PS, M.Pool2(), Unit{}, s, dy, Unit{}))

def main() -> IO(Unit):
  h = TA.Mat.from_list(512n, 8n, mk(4096n, 0n, Nil{}))
  un_f(T.Layer.fwd(Unit, M.IN(), M.POOL(), M.PS, M.Pool2(), Unit{}, h))
