# The four-block hybrid forward at N=8, against the NumPy twin.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_h8.bend as H
import ./model_h.bend as HM
import ./weights_h.bend as W

def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def flat(ml: TA.ML<8n, 260n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show_fw(fw: T.Fwd<HM.P(), H.Y(), H.S()>) -> IO(Unit):
  match fw:
    case T.Fwd{p, y, s}:
      IO.print(String.append("logits: ", show(flat(TA.Mat.to_list(8n, 260n, y)))))

def main() -> IO(Unit):
  do IO<Unit>:
    p : HM.P() <- W.load_params("examples/language/glm53_flash/data/h")
    ids : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/ref8/ids.bin", 0, 8)
    show_fw(T.Layer.fwd(HM.P(), HM.X(), H.Y(), H.S(), H.Model(), p, nats(ids)))
