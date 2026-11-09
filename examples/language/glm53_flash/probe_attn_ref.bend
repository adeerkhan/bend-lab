# The four-head linear and sparse attention layers at N=8, against the
# vendored reference modules' forward outputs.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./attn_mh8.bend as A

def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

def flat(ml: TA.ML<8n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show_fw(+tag: String, fw: T.Fwd<A.AttP(), A.M(), A.LinAttS()>) -> IO(Unit):
  match fw:
    case T.Fwd{p, y, s}:
      IO.print(String.append(tag, show(flat(TA.Mat.to_list(8n, 32n, y)))))

def run_lin(x: A.M(), qkv: TA.Mat<96n, 32n>, out: TA.Mat<32n, 32n>) -> IO(Unit):
  show_fw("lin_y: ", T.Layer.fwd(A.AttP(), A.M(), A.M(), A.LinAttS(), A.LinAttention(),
    T.Two{qkv, T.Two{Unit{}, T.Two{Unit{}, out}}}, x))

def run_sp(x: A.M(), qkv: TA.Mat<96n, 32n>, out: TA.Mat<32n, 32n>) -> IO(Unit):
  show_fw("sp_y: ", T.Layer.fwd(A.AttP(), A.M(), A.M(), A.LinAttS(), A.SpAttention(),
    T.Two{qkv, T.Two{Unit{}, T.Two{Unit{}, out}}}, x))

def main() -> IO(Unit):
  do IO<Unit>:
    x : A.M() <- T.Mat.load(8n, 32n, "examples/language/glm53_flash/data/ref8/x.bin", 0)
    qkv : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, "examples/language/glm53_flash/data/ref8/qkv.bin", 0)
    out : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, "examples/language/glm53_flash/data/ref8/out.bin", 0)
    run_lin(x, qkv, out)
    x2 : A.M() <- T.Mat.load(8n, 32n, "examples/language/glm53_flash/data/ref8/x.bin", 0)
    qkv2 : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, "examples/language/glm53_flash/data/ref8/qkv.bin", 0)
    out2 : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, "examples/language/glm53_flash/data/ref8/out.bin", 0)
    run_sp(x2, qkv2, out2)
