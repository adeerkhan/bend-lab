# The 64x64 flow's forward and its sampler in seconds instead of a 6,000-step
# run: the cheap summary of `reference/probe_flow64.bend`, so a scale or an
# order drift is a cell rather than a report a week later.
#
#   scripts/bend-guard.sh bend bench/flow64_first/main.bend
#
# No dataset and no training: the noise is the fixed Weyl cloud, the weights
# are the example's own `init_params`, and the levels are the sampler's. It
# prints
#
#   params      413440    the model's parameter count
#   out_sq      ...       mean(model^2) of the forward at t = 0.5
#   step_sq     ...       mean(x^2) after one Euler step at t = 0
#   euler32_sq  ...       mean(x^2) after the preset's 32 Euler steps
#   out0        64 values the first output row, so a transposed window shows
#
# `reference/twin_flow64.py` computes the same four numbers as the task's
# reference, so `make matrix` compares them; `reference/test_flow64.py` is the
# elementwise gate on the same model, weights and states.
import Base
import bendtorch@1.1.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../../examples/diffusion/flowers_flow/main.bend as M

def start() -> List<&2, F32>:
  G.noise_of(2000000n, M.BATCH())

def count() -> U32:
  T.Layer.count(M.ModelP(), M.X(), M.Y(), M.ModelS(), M.Model())

def sq_sum(+xs: List<&2, F32>, +acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      sq_sum(t, (acc + (h * h : F32) : F32))

def mean_sq(+xs: List<&2, F32>) -> F32:
  (sq_sum(xs, 0.0) / 8192.0 : F32)

def head64(xs: List<&2, F32>, +left: Nat, +acc: String) -> String:
  match xs left:
    case Con{h, t} 1n+rest:
      head64(t, rest, String.append(acc, String.append(" ", F32.show(h))))
    case _ _:
      acc

def v_of(ve: M.VE) -> List<&2, F32>:
  match ve:
    case M.VE{p, v}:
      v

def fwd_v() -> List<&2, F32>:
  v_of(M.fwd_v(M.init_params(), M.sample_input(0.5, start())))

def step_x() -> List<&2, F32>:
  M.sample_x(M.step_vs(0n, M.VS{M.init_params(), start()}))

def euler32_x() -> List<&2, F32>:
  M.sample_x(M.euler_go(32n, 0n, M.VS{M.init_params(), start()}))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("params: ", U32.show(count())))
    Unit <- IO.print(String.append("out_sq: ", F32.show(mean_sq(fwd_v()))))
    Unit <- IO.print(String.append("step_sq: ", F32.show(mean_sq(step_x()))))
    Unit <- IO.print(String.append("euler32_sq: ", F32.show(mean_sq(euler32_x()))))
    IO.print(String.append("out0:", head64(fwd_v(), 64n, "")))
