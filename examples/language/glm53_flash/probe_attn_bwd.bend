# The four-head linear and sparse attention layers at N=8: forward and the
# backward through the whole layer (qkv projection, split, RoPE, core, out).
# The inputs are `reference/glm53_attn_bwd_data.py`'s; the golden is
# `reference/glm53_attn_bwd.py`'s PyTorch autograd.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./attn_mh8.bend as A

def ml8(ml: TA.ML<8n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml96(ml: TA.ML<96n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml32(ml: TA.ML<32n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

type BWS is Type:
  BWS{dx: String, dqkv: String, dout: String}

def parts(bw: T.Bwd<A.AttP(), A.M()>) -> BWS:
  match bw:
    case T.Bwd{p, dx, g}:
      match g:
        case T.Two{gqkv, T.Two{_, T.Two{_, gout}}}:
          BWS{show(ml8(TA.Mat.to_list(8n, 32n, dx))),
              show(ml96(TA.Mat.to_list(96n, 32n, gqkv))),
              show(ml32(TA.Mat.to_list(32n, 32n, gout)))}

def print_bw(+tag: String, bws: BWS) -> IO(Unit):
  match bws:
    case BWS{dx, dqkv, dout}:
      do IO<Unit>:
        IO.print(String.append(tag, String.append("_dx: ", dx)))
        IO.print(String.append(tag, String.append("_dqkv: ", dqkv)))
        IO.print(String.append(tag, String.append("_dout: ", dout)))
        return Unit{}

def run_fw(+tag: String, fw: T.Fwd<A.AttP(), A.M(), A.LinAttS()>,
    dout: A.M(), zeros: A.AttP(), layer: T.Layer<A.AttP(), A.M(), A.M(), A.LinAttS()>) -> IO(Unit):
  match fw:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        IO.print(String.append(tag, String.append("_y: ", show(ml8(TA.Mat.to_list(8n, 32n, y))))))
        print_bw(tag, parts(T.Layer.bwd(A.AttP(), A.M(), A.M(), A.LinAttS(), layer, p, s, dout, zeros)))
        return Unit{}

def run_lin(+tag: String, x: A.M(), qkv: TA.Mat<96n, 32n>, out: TA.Mat<32n, 32n>,
    dout: A.M()) -> IO(Unit):
  run_fw(tag, T.Layer.fwd(A.AttP(), A.M(), A.M(), A.LinAttS(), A.LinAttention(),
      T.Two{qkv, T.Two{Unit{}, T.Two{Unit{}, out}}}, x),
    dout, T.Layer.zeros(A.AttP(), A.M(), A.M(), A.LinAttS(), A.LinAttention()), A.LinAttention())

def run_sp(+tag: String, x: A.M(), qkv: TA.Mat<96n, 32n>, out: TA.Mat<32n, 32n>,
    dout: A.M()) -> IO(Unit):
  run_fw(tag, T.Layer.fwd(A.AttP(), A.M(), A.M(), A.LinAttS(), A.SpAttention(),
      T.Two{qkv, T.Two{Unit{}, T.Two{Unit{}, out}}}, x),
    dout, T.Layer.zeros(A.AttP(), A.M(), A.M(), A.LinAttS(), A.SpAttention()), A.SpAttention())

def main() -> IO(Unit):
  do IO<Unit>:
    x : A.M() <- T.Mat.load(8n, 32n, "examples/language/glm53_flash/data/ref8/x.bin", 0)
    qkv : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, "examples/language/glm53_flash/data/ref8/qkv.bin", 0)
    out : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, "examples/language/glm53_flash/data/ref8/out.bin", 0)
    dout : A.M() <- T.Mat.load(8n, 32n, "examples/language/glm53_flash/data/ref8/dout.bin", 0)
    run_lin("lin", x, qkv, out, dout)
    x2 : A.M() <- T.Mat.load(8n, 32n, "examples/language/glm53_flash/data/ref8/x.bin", 0)
    qkv2 : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, "examples/language/glm53_flash/data/ref8/qkv.bin", 0)
    out2 : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, "examples/language/glm53_flash/data/ref8/out.bin", 0)
    dout2 : A.M() <- T.Mat.load(8n, 32n, "examples/language/glm53_flash/data/ref8/dout.bin", 0)
    run_sp("sp", x2, qkv2, out2, dout2)
    x3 : A.M() <- T.Mat.load(8n, 32n, "examples/language/glm53_flash/data/ref8/x.bin", 0)
    qkv3 : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, "examples/language/glm53_flash/data/ref8/qkv.bin", 0)
    out3 : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, "examples/language/glm53_flash/data/ref8/out.bin", 0)
    bad : A.M() <- T.Mat.load(8n, 32n, "examples/language/glm53_flash/data/ref8/dout_bad.bin", 0)
    run_lin("lin_bad", x3, qkv3, out3, bad)
