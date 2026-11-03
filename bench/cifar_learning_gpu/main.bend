# The CIFAR-10 model's training step on BendTorch's device lane: the same
# parameters, the same inputs, `LinearF`/`LinearFirstF`/`SiLU`/`MSE` from
# bendtorch-gpu in place of main.bend's layers. The loss curve is the claim:
# at a fixed noise level it must follow `bench/cifar_learning`'s CPU run.
#
#   CPU:   bendenv run -- bend bench/cifar_learning_gpu/main.bend
#   CUDA:  CUDA_HOME=... bend bench/cifar_learning_gpu/main.bend -o /tmp/cl_gpu
#
# Fixed t = 32, lr 1e-3, 600 steps. `bench/cifar_learning`'s CPU run reads
# 0.8405294 at step 50 and 0.5286835 at 600.
import Base
import bendtorch@1.3.0.0/main.bend as T
import bendtorch-gpu@1.3.0.0/main.bend as G
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import ../../examples/cifar/main.bend as M

def every100(+i: Nat) -> Bool:
  Nat.is_eq(Nat.mod(i, 100n), 0n)

def show_maybe(+fire: Bool, +i: Nat, +l: F32) -> IO(Unit):
  match fire:
    case True{}:
      IO.print(String.append(String.append("  step ", Nat.show(i)), String.append(" loss ", F32.show(l))))
    case False{}:
      IO.pure(Unit, Unit{})

def cfg_lr(+lr: F32) -> T.OptCfg:
  T.CAdamW{lr, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

# the example's model, parameter types unchanged, work on the device lane
def B1(~r: Nat) -> T.Layer<M.B1P, M.M2(r, M.inwidth()), M.M2(r, M.hidden()), M.B1S(r)>:
  T.Seq(~T.Lin<M.inwidth(), M.hidden()>, ~Unit, ~M.M2(r, M.inwidth()), ~M.M2(r, M.hidden()),
    ~M.M2(r, M.hidden()), ~M.M2(r, M.inwidth()), ~M.M2(r, M.hidden()),
    ~G.LinearFirstF(r, M.inwidth(), M.hidden(), 0n), ~G.SiLU(r, M.hidden()))

def B2(~r: Nat) -> T.Layer<M.B2P, M.M2(r, M.inwidth()), M.M2(r, M.hidden()), M.B2S(r)>:
  T.Seq(~M.B1P, ~T.Lin<M.hidden(), M.hidden()>, ~M.M2(r, M.inwidth()), ~M.M2(r, M.hidden()),
    ~M.M2(r, M.hidden()), ~M.B1S(r), ~M.M2(r, M.hidden()), ~B1(~r), ~G.LinearF(r, M.hidden(), M.hidden(), 0n))

def B3(~r: Nat) -> T.Layer<M.B3P, M.M2(r, M.inwidth()), M.M2(r, M.hidden()), M.B3S(r)>:
  T.Seq(~M.B2P, ~Unit, ~M.M2(r, M.inwidth()), ~M.M2(r, M.hidden()), ~M.M2(r, M.hidden()),
    ~M.B2S(r), ~M.M2(r, M.hidden()), ~B2(~r), ~G.SiLU(r, M.hidden()))

def Model(~r: Nat) -> T.Layer<M.B4P, M.M2(r, M.inwidth()), M.M2(r, M.d()), M.B4S(r)>:
  T.Seq(~M.B3P, ~T.Lin<M.hidden(), M.d()>, ~M.M2(r, M.inwidth()), ~M.M2(r, M.hidden()), ~M.M2(r, M.d()),
    ~M.B3S(r), ~M.M2(r, M.hidden()), ~B3(~r), ~G.LinearF(r, M.hidden(), M.d(), 0n))

def mse(y: M.Y(), tgt: M.Y()) -> T.LossOut<M.Y()>:
  G.MSE(M.n(), M.d(), y, tgt)

def step_at(+s: Nat, +lr: F32, tr: T.Train<M.B4P()>, +x0: List<&2, F32>) -> T.Train<M.B4P()>:
  +t = Nat.add(Nat.mod(s, 64n), 1n)
  T.Train.step(~M.B4P(), ~M.X(), ~M.Y(), ~M.B4S(M.n()), ~M.Y(), ~Model(M.n()), ~mse, cfg_lr(lr), tr,
    TA.Mat.from_list(M.n(), M.inwidth(), M.input_of(s, t, M.coef_a(t), M.coef_b(t), x0)),
    M.noise_of(s))

def loop(+lr: F32, +left: Nat, +i: Nat, tr: T.Train<M.B4P()>, +x0: List<&2, F32>) ->
  IO(T.Train<M.B4P()>):
  match left tr:
    case 0n T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<M.B4P()>>:
        Unit <- show_maybe(every100(i), i, l)
        return T.Train{p, g, st, q1, q2, l}
    case 1n+rest T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<M.B4P()>>:
        Unit <- show_maybe(every100(i), i, l)
        loop(lr, rest, (1n + i : Nat),
          step_at(31n, lr, T.Train{p, g, st, q1, q2, l}, x0), x0)

def print_end(tr: T.Train<M.B4P()>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, q1, q2, +l}:
      IO.print(String.append("  end loss ", F32.show(l)))

def main() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.csv_of(
      "arwics/cifar10", "f44619abdb57391606d732724f9fc40a684471bf", "train.csv", False{}, 0n)
    x0 : List<&2, F32> <- M.cifar_of(t)
    tt0 : Nat <- IO.now()
    p0 : M.B4P() <- IO.pure(M.B4P, M.params())
    tr0 : T.Train<M.B4P()> <- IO.pure(T.Train<M.B4P()>,
      T.Train.new(~M.B4P(), ~M.X(), ~M.Y(), ~M.B4S(M.n()), ~Model(M.n()), cfg_lr(0.001), p0))
    tr1 : T.Train<M.B4P()> <- loop(0.001, 600n, 0n, tr0, x0)
    tt1 : Nat <- IO.now()
    Unit <- IO.print(String.append("  wall_ms ", Nat.show(Nat.sub(tt1, tt0))))
    print_end(tr1)
