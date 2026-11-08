# Softmax(2,2) on [[0,0],[2,0]]: row-wise gives 0.5 0.5 0.88 0.12;
# column-wise gives 0.12 0.5 0.88 0.5. The probe prints which one BendTorch does.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model.bend as M

def show4(+n: Nat, +xs: List<&2, F32>, +acc: String) -> String:
  match n xs:
    case 0n _:
      acc
    case 1n+p Con{h, t}:
      show4(p, t, String.append(acc, String.append(F32.show(h), " ")))
    case _ _:
      acc

def show(y: TA.Mat<2n, 2n>) -> IO(Unit):
  ys = M.ml_list(2n, 2n, TA.Mat.to_list(2n, 2n, y))
  do IO<Unit>:
    Unit <- IO.print(show4(4n, ys, ""))
    return Unit{}

def un_r(r: T.Fwd<Unit, TA.Mat<2n, 2n>, TA.Mat<2n, 2n>>) -> IO(Unit):
  match r:
    case T.Fwd{_, y, _}:
      show(y)

def main() -> IO(Unit):
  x = TA.Mat.from_list(2n, 2n, Con{0.0, Con{0.0, Con{2.0, Con{0.0, Nil{}}}}})
  un_r(T.Layer.fwd(Unit, TA.Mat<2n, 2n>, TA.Mat<2n, 2n>, TA.Mat<2n, 2n>, T.Softmax(2n, 2n), Unit{}, x))
