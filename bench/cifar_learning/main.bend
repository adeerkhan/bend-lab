# Can the CIFAR-10 example's model learn at all? The example trains 40 steps and
# its loss barely moves (1.0014539 -> 1.0163207), so this probe asks the
# question directly: hold the noise level fixed and step many times.
#
#   MLBEND_DATA=reference/.cache/datasets bendenv run -- bend bench/cifar_learning/main.bend
#
# `M.do_step` passes the example's own `cfg()` to `Train.step`, so an lr
# passed to `Train.new` alone does nothing (ISSUE-037): the runs below call
# `Train.step` with the lr they name, the way `M.do_step` does with its own.
#
#   A. the example's schedule (t = s mod 64 + 1, one visit each), 600 steps
#   B. a fixed t = 32 (x_t = 0.7 x0 + 0.7 eps), 600 steps
#   C. the same level at lr 0.001 and 0.1, 150 steps each
#
# If B and C move and A does not, the schedule is the reason: 40 steps visit
# 40 different noise levels once each. If nothing moves, the step is broken.
import Base
import bendtorch@1.1.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import ../../examples/cifar/main.bend as M

def every50(+i: Nat) -> Bool:
  Nat.is_eq(Nat.mod(i, 50n), 0n)

def show_maybe(+fire: Bool, +i: Nat, +l: F32) -> IO(Unit):
  match fire:
    case True{}:
      IO.print(String.append(String.append("  step ", Nat.show(i)), String.append(" loss ", F32.show(l))))
    case False{}:
      IO.pure(Unit, Unit{})

# the example's schedule: t = (s mod 64) + 1, s the step index
def sched_s(+i: Nat) -> Nat:
  i

# a fixed level: s = 31 gives t = 32
def fixed_s(+i: Nat) -> Nat:
  31n

def s_of(+mode: Nat, +i: Nat) -> Nat:
  match mode:
    case 0n:
      sched_s(i)
    case _:
      fixed_s(i)

def cfg_lr(+lr: F32) -> T.OptCfg:
  T.CAdamW{lr, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

# `M.do_step`, with the config as a parameter
def step_at(+s: Nat, +lr: F32, tr: T.Train<M.B4P()>, +x0: List<&2, F32>) -> T.Train<M.B4P()>:
  +t = Nat.add(Nat.mod(s, 64n), 1n)
  T.Train.step(~M.B4P(), ~M.X(), ~M.Y(), ~M.B4S(M.n()), ~M.Y(), ~M.Model(M.n()), ~M.mse, cfg_lr(lr), tr,
    TA.Mat.from_list(M.n(), M.inwidth(), M.input_of(s, t, M.coef_a(t), M.coef_b(t), x0)),
    TA.Mat.from_list(M.n(), M.d(), M.noise_of(s)))

def loop(+mode: Nat, +lr: F32, +left: Nat, +i: Nat, tr: T.Train<M.B4P()>, +x0: List<&2, F32>) ->
  IO(T.Train<M.B4P()>):
  match left tr:
    case 0n T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<M.B4P()>>:
        Unit <- show_maybe(every50(i), i, l)
        return T.Train{p, g, st, q1, q2, l}
    case 1n+rest T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<M.B4P()>>:
        Unit <- show_maybe(every50(i), i, l)
        loop(mode, lr, rest, (1n + i : Nat),
          step_at(s_of(mode, i), lr, T.Train{p, g, st, q1, q2, l}, x0), x0)

def print_end(tr: T.Train<M.B4P()>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, q1, q2, +l}:
      IO.print(String.append("  end loss ", F32.show(l)))

def run(+name: String, +mode: Nat, +lr: F32, +steps: Nat) -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.csv_of(
      "arwics/cifar10", "f44619abdb57391606d732724f9fc40a684471bf", "train.csv", False{}, 0n)
    x0 : List<&2, F32> <- M.cifar_of(t)
    Unit <- IO.print(name)
    tt0 : Nat <- IO.now()
    p0 : M.B4P() <- IO.pure(M.B4P, M.params())
    tr0 : T.Train<M.B4P()> <- IO.pure(T.Train<M.B4P()>,
      T.Train.new(~M.B4P(), ~M.X(), ~M.Y(), ~M.B4S(M.n()), ~M.Model(M.n()), cfg_lr(lr), p0))
    tr1 : T.Train<M.B4P()> <- loop(mode, lr, steps, 0n, tr0, x0)
    tt1 : Nat <- IO.now()
    Unit <- IO.print(String.append("  wall_ms ", Nat.show(Nat.sub(tt1, tt0))))
    print_end(tr1)

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- run("A. schedule t=1..40, lr 0.01, 600 steps", 0n, 0.01, 600n)
    Unit <- run("B. fixed t=32, lr 0.01, 600 steps", 1n, 0.01, 600n)
    Unit <- run("C1. fixed t=32, lr 0.001, 150 steps", 1n, 0.001, 150n)
    Unit <- run("C2. fixed t=32, lr 0.1, 150 steps", 1n, 0.1, 150n)
    Unit <- run("D. fixed t=32, lr 0.001, 600 steps", 1n, 0.001, 600n)
    Unit <- run("E. schedule t=1..40, lr 0.001, 600 steps", 0n, 0.001, 600n)
    return Unit{}
