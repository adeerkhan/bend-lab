# Whole-array probe of the rectified-flow port's layers, for the NumPy twin.
#
#   down       pixel-unshuffle Down2 forward on the flat ramp 0..4095
#   down_bwd   its exact-inverse backward on the flat ramp 0..4095
#   pool       2x2 max-pool forward (the XAttnRes resize)
#   pool_bwd   the scatter backward at the saved arg-max
#   xattn      the zero-query mix 0.5 * (pooled history + current)
#   mix        the learned-query mix, equal keys (weights 1/2, 1/2)
#   mix2       the learned-query mix, kx = 2 * kh (per-row weights)
#   softmax    Softmax(2,2) on [[0,0],[2,0]], to pin the axis
#
# `reference/test_rectified_flow_layers.py` builds this once and compares every
# number with its own twin. Each tag gets a fresh input: a `Mat` is a value and
# is consumed by its use.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ../examples/diffusion/rectified_flow/model.bend as M

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

def show(tag: String, xs: List<&2, F32>) -> String:
  String.append(tag, String.append(" ", show_go(xs)))

# ---- fresh inputs: the flat ramp and its multiples --------------------------

def ramp(+n: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      ramp(p, Nat.add(i, 1n), Con{F32.from_nat(i), acc})

def ramp2(+n: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      ramp2(p, Nat.add(i, 1n), Con{(2.0 * F32.from_nat(i) : F32), acc})

def ramp3(+n: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      ramp3(p, Nat.add(i, 1n), Con{(3.0 * F32.from_nat(i) : F32), acc})

def constq(+n: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      constq(p, Con{0.10000000, acc})

def in_ramp() -> M.IN():
  TA.Mat.from_list(512n, 8n, ramp(4096n, 0n, Nil{}))

def out_ramp() -> M.OUT():
  TA.Mat.from_list(128n, 32n, ramp(4096n, 0n, Nil{}))

def pool_ramp() -> M.POOL():
  TA.Mat.from_list(128n, 8n, ramp(1024n, 0n, Nil{}))

def q_const() -> TA.Mat<1n, 8n>:
  TA.Mat.from_list(1n, 8n, constq(8n, Nil{}))

# ---- printers ---------------------------------------------------------------

def show_in(tag: String, y: M.IN()) -> IO(Unit):
  do IO<Unit>:
    IO.print(show(tag, M.ml_list(512n, 8n, TA.Mat.to_list(512n, 8n, y))))
    return Unit{}

def show_out(tag: String, y: M.OUT()) -> IO(Unit):
  do IO<Unit>:
    IO.print(show(tag, M.ml_list(128n, 32n, TA.Mat.to_list(128n, 32n, y))))
    return Unit{}

def show_pool(tag: String, y: M.POOL()) -> IO(Unit):
  do IO<Unit>:
    IO.print(show(tag, M.ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, y))))
    return Unit{}

def show_in_bwd(tag: String, b: T.Bwd<Unit, M.IN()>) -> IO(Unit):
  match b:
    case T.Bwd{_, dx, _}:
      show_in(tag, dx)

# ---- the layers -------------------------------------------------------------

def down_all(f: T.Fwd<Unit, M.OUT(), Unit>) -> IO(Unit):
  match f:
    case T.Fwd{_, y, s}:
      do IO<Unit>:
        Unit <- show_out("down:", y)
        show_in_bwd("down_bwd:", T.Layer.bwd(Unit, M.IN(), M.OUT(), Unit,
          M.Down2(), Unit{}, s, out_ramp(), Unit{}))

def pool_all(f: T.Fwd<Unit, M.POOL(), M.PS>) -> IO(Unit):
  match f:
    case T.Fwd{_, y, s}:
      do IO<Unit>:
        Unit <- show_pool("pool:", y)
        show_in_bwd("pool_bwd:", T.Layer.bwd(Unit, M.IN(), M.POOL(), M.PS,
          M.Pool2(), Unit{}, s, pool_ramp(), Unit{}))

def xattn_show() -> IO(Unit):
  h = in_ramp()
  x = pool_ramp()
  show_pool("xattn:", M.mix_mat(x, M.pool_mat(h)))

def mix_a() -> IO(Unit):
  vh = pool_ramp()
  xc = TA.Mat.from_list(128n, 8n, ramp3(1024n, 0n, Nil{}))
  kh = pool_ramp()
  kx = pool_ramp()
  show_pool("mix:", M.mix_forward(M.MXS{vh, xc, kh, kx, q_const()}))

def mix_b() -> IO(Unit):
  vh = pool_ramp()
  xc = TA.Mat.from_list(128n, 8n, ramp3(1024n, 0n, Nil{}))
  kh = pool_ramp()
  kx = TA.Mat.from_list(128n, 8n, ramp2(1024n, 0n, Nil{}))
  show_pool("mix2:", M.mix_forward(M.MXS{vh, xc, kh, kx, q_const()}))

def show_sm(y: TA.Mat<2n, 2n>) -> IO(Unit):
  do IO<Unit>:
    IO.print(show("softmax:", M.ml_list(2n, 2n, TA.Mat.to_list(2n, 2n, y))))
    return Unit{}

def un_sm(r: T.Fwd<Unit, TA.Mat<2n, 2n>, TA.Mat<2n, 2n>>) -> IO(Unit):
  match r:
    case T.Fwd{_, y, _}:
      show_sm(y)

def softmax_show() -> IO(Unit):
  x = TA.Mat.from_list(2n, 2n, Con{0.0, Con{0.0, Con{2.0, Con{0.0, Nil{}}}}})
  un_sm(T.Layer.fwd(Unit, TA.Mat<2n, 2n>, TA.Mat<2n, 2n>, TA.Mat<2n, 2n>, T.Softmax(2n, 2n), Unit{}, x))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- down_all(T.Layer.fwd(Unit, M.IN(), M.OUT(), Unit, M.Down2(), Unit{}, in_ramp()))
    Unit <- pool_all(T.Layer.fwd(Unit, M.IN(), M.POOL(), M.PS, M.Pool2(), Unit{}, in_ramp()))
    Unit <- xattn_show()
    Unit <- mix_a()
    Unit <- mix_b()
    softmax_show()
