# The frozen-model probe of the 2D rectified flow (ISSUE-058): no training, just `params()` and the fixed cloud, so
# `reference/twin_flow2d.py` can compare the forward, one Euler step and the 32-step sample element by element.
#
#   params   the model's parameter count, so a dump that changed says so
#   w        the flat `Layer.dump` list in the loader's order (17,282 values)
#   inp      the model input at t = 0.5 (128 rows of x_t, y_t, t) — the whole input, so a changed token row fails here
#   fwd      one forward at t = 0.5 on the fixed noise
#   step     one Euler step at t = 0 (dt = 1/32, the example's own 0.03125)
#   euler8   the state after 8 of the 32 steps (t = 0 .. 7/32)
#   euler16  the state after 16 steps
#   euler32  the state after 32 steps, the preset's sampler
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../examples/diffusion/flow_matching/main.bend as M

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

def show(tag: String, xs: List<&2, F32>) -> String:
  String.append(tag, String.append(" ", show_go(xs)))

def flat_w(pl: T.PL<M.B4P()>) -> List<&2, F32>:
  match pl:
    case T.PL{p, xs}:
      xs

def count() -> U32:
  T.Layer.count(M.B4P(), M.X(), M.Y(), M.B4S(), M.Model())

def start() -> List<&2, F32>:
  G.noise_of(2000000n, 256n)

def show_ve(ve: M.ES) -> IO(Unit):
  match ve:
    case M.ES{p, v}:
      IO.print(show("fwd:", v))

def show_es(st: M.ES, tag: String) -> IO(Unit):
  match st:
    case M.ES{p, x}:
      IO.print(show(tag, x))

def step(+n: Nat) -> M.ES:
  M.euler_go(n, 0n, M.ES{M.params(), start()})

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("params: ", U32.show(count())))
    Unit <- IO.print(show("w:", flat_w(T.Layer.dump(M.B4P(), M.X(), M.Y(), M.B4S(), M.Model(), M.params()))))
    Unit <- IO.print(show("inp:", M.sample_input(0.5, start())))
    Unit <- show_ve(M.fwd_v(M.params(), 0.5, start()))
    Unit <- show_es(step(1n), "step:")
    Unit <- show_es(step(8n), "euler8:")
    Unit <- show_es(step(16n), "euler16:")
    show_es(step(32n), "euler32:")
