import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./lin_core.bend as L

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}: ""
    case Con{h, t}: String.append(F32.show(h), String.append(" ", show_go(t)))

def mk(+n: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n: List.reverse(&2, F32, acc)
    case 1n+p: mk(p, Nat.add(i, 1n), Con{F32.from_nat(i), acc})

def flat(-r: Nat, -c: Nat, ml: TA.ML<r, c>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, l}: l

def show_out(f: T.Fwd<Unit, L.M(), L.LinS>) -> IO(Unit):
  match f:
    case T.Fwd{_, y, _}:
      IO.print(String.append("y: ", show_go(flat(128n, 32n, TA.Mat.to_list(128n, 32n, y)))))

def go(q: L.M(), k: L.M(), v: L.M()) -> IO(Unit):
  show_out(L.lin_fwd(q, k, v))

def main() -> IO(Unit):
  go(TA.Mat.from_list(128n, 32n, mk(4096n, 0n, Nil{})),
     TA.Mat.from_list(128n, 32n, mk(4096n, 0n, Nil{})),
     TA.Mat.from_list(128n, 32n, mk(4096n, 0n, Nil{})))
