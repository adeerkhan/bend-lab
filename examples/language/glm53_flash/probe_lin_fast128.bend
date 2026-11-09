# Fast prefix-state core against the direct core, 128 positions, one head.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./lin_bwd.bend as L
import ./lin_fast.bend as F

def ml(ml: TA.ML<128n, 8n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

def show_bw(+tag: String, bw: L.BW) -> IO(Unit):
  match bw:
    case L.BW{dq, dk, dv}:
      do IO<Unit>:
        IO.print(String.append(String.append(tag, "_dq: "), show(dq)))
        IO.print(String.append(String.append(tag, "_dk: "), show(dk)))
        IO.print(String.append(String.append(tag, "_dv: "), show(dv)))
        return Unit{}

def run(+q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +dy: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("direct_y: ", show(L.lin_y(128n, 8n, q, k, v))))
    IO.print(String.append("fast_y: ", show(F.lin_y_fast(128n, 8n, q, k, v))))
    show_bw("direct", L.lin_bwd(128n, 8n, q, k, v, dy))
    show_bw("fast", F.lin_bwd_fast(128n, 8n, q, k, v, dy))

def main() -> IO(Unit):
  do IO<Unit>:
    q : TA.Mat<128n, 8n> <- T.Mat.load(128n, 8n, "examples/language/glm53_flash/data/lin128_q.bin", 0)
    k : TA.Mat<128n, 8n> <- T.Mat.load(128n, 8n, "examples/language/glm53_flash/data/lin128_k.bin", 0)
    v : TA.Mat<128n, 8n> <- T.Mat.load(128n, 8n, "examples/language/glm53_flash/data/lin128_v.bin", 0)
    dy : TA.Mat<128n, 8n> <- T.Mat.load(128n, 8n, "examples/language/glm53_flash/data/lin128_dy.bin", 0)
    run(ml(TA.Mat.to_list(128n, 8n, q)), ml(TA.Mat.to_list(128n, 8n, k)),
        ml(TA.Mat.to_list(128n, 8n, v)), ml(TA.Mat.to_list(128n, 8n, dy)))
