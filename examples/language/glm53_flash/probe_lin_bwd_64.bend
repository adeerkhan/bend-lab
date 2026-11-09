# The Stage B backward at 64 positions, one head, head width 8.
# Reads the four deterministic inputs `reference/glm53_linear_bwd.py` writes
# and prints y, dq, dk and dv for the whole 64 x 8 tensor.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./lin_bwd.bend as L

def flat(ml: TA.ML<64n, 8n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

def show_bw(bw: L.BW) -> IO(Unit):
  match bw:
    case L.BW{dq, dk, dv}:
      do IO<Unit>:
        IO.print(String.append("dq: ", show(dq)))
        IO.print(String.append("dk: ", show(dk)))
        IO.print(String.append("dv: ", show(dv)))
        return Unit{}

def run(+q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +dy: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("y: ", show(L.lin_y(64n, 8n, q, k, v))))
    show_bw(L.lin_bwd(64n, 8n, q, k, v, dy))

def main() -> IO(Unit):
  do IO<Unit>:
    mq : TA.Mat<64n, 8n> <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_q.bin", 0)
    mk : TA.Mat<64n, 8n> <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_k.bin", 0)
    mv : TA.Mat<64n, 8n> <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_v.bin", 0)
    md : TA.Mat<64n, 8n> <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_dy.bin", 0)
    run(flat(TA.Mat.to_list(64n, 8n, mq)), flat(TA.Mat.to_list(64n, 8n, mk)),
        flat(TA.Mat.to_list(64n, 8n, mv)), flat(TA.Mat.to_list(64n, 8n, md)))
