# The frozen-model probe of the 64x64 flow (ISSUE-043): no training, just the
# starting weights, so the NumPy twin can compare the forward and the sampler
# element by element. Each line is one flat list for the twin.
#
#   params   the model's parameter count, so a dump that changed says so
#   w0..w7   the flat `Layer.dump` list in the loader's order, eight parts
#            (413,440 = 8 x 51,680; the twin asserts it consumed all of them)
#   inp      the model input at t = 0.5: the first two tokens (pixels, then
#            position and time features), so the feature order is gated too
#   fwd      one forward at t = 0.5 on the fixed noise
#   step     one Euler step at t = 0 (dt = 1/32, the example's own `upd`)
#   euler8   the state after 8 of the 32 Euler steps (t = 0 .. 7/32)
#   euler16  the state after 16 steps
#   euler32  the state after 32 steps, the preset's sampler
#
# `init_params` is deterministic (noise_of(1) scaled by 0.1, in the loader's
# order), so each output can consume its own copy of the weights; the twin
# takes the dump instead, so neither side's initialisation is trusted.
import Base
import bendtorch@1.3.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../examples/diffusion/flowers_flow/main.bend as M

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

def show(tag: String, xs: List<&2, F32>) -> String:
  String.append(tag, String.append(" ", show_go(xs)))

# the first `left` values only: `show_go` over a 413k list is one String, and
# a part per line keeps each one near the size the ViT probe already prints
def show_n(+left: Nat, +xs: List<&2, F32>) -> String:
  match left:
    case 0n:
      ""
    case 1n+rest:
      match xs:
        case Nil{}:
          ""
        case Con{h, t}:
          String.append(F32.show(h), String.append(" ", show_n(rest, t)))

def drop_n(+left: Nat, +xs: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      xs
    case 1n+rest:
      match xs:
        case Nil{}:
          Nil{}
        case Con{h, t}:
          drop_n(rest, t)

def part() -> Nat:
  51680n

def flat_w(pl: T.PL<M.ModelP()>) -> List<&2, F32>:
  match pl:
    case T.PL{p, xs}:
      xs

def dump_parts(+left: Nat, +i: Nat, +flat: List<&2, F32>) -> IO(Unit):
  match left:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+rest:
      do IO<Unit>:
        IO.print(String.append(String.append("w", Nat.show(i)),
          String.append(": ", show_n(part(), flat))))
        dump_parts(rest, (1n + i : Nat), drop_n(part(), flat))

def dump(p: M.ModelP()) -> IO(Unit):
  dump_parts(8n, 0n, flat_w(T.Layer.dump(M.ModelP(),
    TA.Mat<M.RW(), M.FE()>(), TA.Mat<M.RW(), M.PX()>(), M.ModelS(), M.Model(), p)))

def start() -> List<&2, F32>:
  G.noise_of(2000000n, M.BATCH())

def count() -> U32:
  T.Layer.count(M.ModelP(), M.X(), M.Y(), M.ModelS(), M.Model())

def show_ve(ve: M.VE) -> IO(Unit):
  match ve:
    case M.VE{p, v}:
      do IO<Unit>:
        IO.print(show("fwd:", v))
        return Unit{}

def show_vs(st: M.VS, tag: String) -> IO(Unit):
  match st:
    case M.VS{p, x}:
      do IO<Unit>:
        IO.print(show(tag, x))
        return Unit{}

def euler(+left: Nat) -> M.VS:
  M.euler_go(left, 0n, M.VS{M.init_params(), start()})

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("params: ", U32.show(count())))
    Unit <- IO.print(String.append("inp:", String.append(" ",
      show_n(160n, M.sample_input(0.5, start())))))
    Unit <- dump(M.init_params())
    Unit <- show_ve(M.fwd_v(M.init_params(), M.sample_input(0.5, start())))
    Unit <- show_vs(M.step_vs(0n, M.VS{M.init_params(), start()}), "step:")
    Unit <- show_vs(euler(8n), "euler8:")
    Unit <- show_vs(euler(16n), "euler16:")
    show_vs(euler(32n), "euler32:")
