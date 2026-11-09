# The ignored-label branch of the masked loss, on a 2 x 3 logits matrix.
# Row 0 has label 0; row 1 is ignored (label 5, the mask).  Prints the masked
# loss and the whole gradient matrix.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./loss_masked.bend as L

def M() -> Type:
  TA.Mat<2n, 3n>

def vals() -> List<&2, F32>:
  Con{0.1, Con{0.2, Con{0.3, Con{(0.0 - 0.4 : F32), Con{0.5, Con{0.6, Nil{}}}}}}}

def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

def flat(ml: TA.ML<2n, 3n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show_loss(l: T.LossOut<M()>) -> IO(Unit):
  match l:
    case T.LossOut{loss, dy}:
      IO.print(String.append("masked_loss: ", String.append(F32.show(loss),
        String.append(" masked_dy: ", show(flat(TA.Mat.to_list(2n, 3n, dy)))))))

def run(z: M()) -> IO(Unit):
  show_loss(L.CrossEntropyMasked(2n, 3n, z, Con{0n, Con{5n, Nil{}}}, 5n))

def main() -> IO(Unit):
  do IO<Unit>:
    run(TA.Mat.from_list(2n, 3n, vals()))
