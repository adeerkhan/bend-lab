# The integrated Stage B/C cores at 64 positions, one head, width 8.
# Loads the deterministic pre-RoPE q/k/v/dout files and prints the forward and
# the three gradients for each core; `reference/glm53_attn_c.py` is the
# PyTorch-autograd golden.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./attn_c.bend as A
import ./lin_bwd.bend as L

def ml8(ml: TA.ML<64n, 8n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

def show_qkv(dx: A.QKV()) -> IO(Unit):
  match dx:
    case T.Two{dqm, T.Two{dkm, dvm}}:
      do IO<Unit>:
        IO.print(String.append("dq: ", show(ml8(TA.Mat.to_list(64n, 8n, dqm)))))
        IO.print(String.append("dk: ", show(ml8(TA.Mat.to_list(64n, 8n, dkm)))))
        IO.print(String.append("dv: ", show(ml8(TA.Mat.to_list(64n, 8n, dvm)))))
        return Unit{}

def show_bwd(b: T.Bwd<Unit, A.QKV()>) -> IO(Unit):
  match b:
    case T.Bwd{_, dx, g}:
      show_qkv(dx)

def show_fw_lin(fw: T.Fwd<Unit, A.M(), A.LinS>, dy: A.M()) -> IO(Unit):
  match fw:
    case T.Fwd{pp, y, s}:
      do IO<Unit>:
        IO.print(String.append("lin_y: ", show(ml8(TA.Mat.to_list(64n, 8n, y)))))
        show_bwd(T.Layer.bwd(Unit, A.QKV(), A.M(), A.LinS, A.LinCore(), pp, s, dy, Unit{}))

def show_fw_sp(fw: T.Fwd<Unit, A.M(), A.SpS>, dy: A.M()) -> IO(Unit):
  match fw:
    case T.Fwd{pp, y, s}:
      do IO<Unit>:
        IO.print(String.append("sp_y: ", show(ml8(TA.Mat.to_list(64n, 8n, y)))))
        show_bwd(T.Layer.bwd(Unit, A.QKV(), A.M(), A.SpS, A.SpCore(), pp, s, dy, Unit{}))

def do_lin() -> IO(Unit):
  do IO<Unit>:
    q : A.M() <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_q.bin", 0)
    k : A.M() <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_k.bin", 0)
    v : A.M() <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_v.bin", 0)
    dy : A.M() <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_dy.bin", 0)
    fw : T.Fwd<Unit, A.M(), A.LinS> <- IO.pure(T.Fwd<Unit, A.M(), A.LinS>,
      T.Layer.fwd(Unit, A.QKV(), A.M(), A.LinS, A.LinCore(), Unit{}, T.Two{q, T.Two{k, v}}))
    show_fw_lin(fw, dy)

def do_sp() -> IO(Unit):
  do IO<Unit>:
    q : A.M() <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_q.bin", 0)
    k : A.M() <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_k.bin", 0)
    v : A.M() <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_v.bin", 0)
    dy : A.M() <- T.Mat.load(64n, 8n, "examples/language/glm53_flash/data/lin64_dy.bin", 0)
    fw : T.Fwd<Unit, A.M(), A.SpS> <- IO.pure(T.Fwd<Unit, A.M(), A.SpS>,
      T.Layer.fwd(Unit, A.QKV(), A.M(), A.SpS, A.SpCore(), Unit{}, T.Two{q, T.Two{k, v}}))
    show_fw_sp(fw, dy)

def main() -> IO(Unit):
  do IO<Unit>:
    do_lin()
    do_sp()
