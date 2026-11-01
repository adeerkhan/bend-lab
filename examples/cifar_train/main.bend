# cifar-train: the same CIFAR-10 denoiser as examples/cifar, trained so that
# its loss can fall.
#
#   bendenv run -- bend examples/cifar_train/main.bend
#
# `examples/cifar` is a plumbing check: 40 steps over 40 different noise levels,
# one visit each, so no level is ever trained twice and the loss cannot move
# (ISSUE-036). This example is the same model — `examples/cifar/main.bend`'s
# three linear layers, its parameters and its input builder — over the same
# 256 images, with the two things that make a step learn:
#
#   a level drawn from {16, 32, 48} each step, so every level is visited 200
#   times in 600 steps rather than once in 40;
#   lr 1e-3, the rate the sweep measured as the one that converges (1e-2
#   overshoots before it settles, ISSUE-036).
#
# The draw is a counter, like every other "random" number here: t = 16 +
# 16 * (s mod 3). The loss is printed at step 1 and every 50 steps after, and
# `loss_drop` is the first loss over the last: the one gated number, against
# the NumPy twin (reference/test_cifar_train.py). 600 steps is a plumbing-scale
# training run, not a trained model: no sampler runs here, and the images it
# would draw are the cifar example's business.
import Base
import bendtorch@1.1.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import ../cifar/main.bend as M

def steps() -> Nat:
  600n

# the levels: the three noise levels a step draws from
def levels() -> Nat:
  3n

def level(+s: Nat) -> Nat:
  Nat.add(16n, Nat.mul(16n, Nat.mod(s, levels())))

# the rate the sweep measured: 1e-3, the same optimizer as the example
def cfg() -> T.OptCfg:
  T.CAdamW{0.001, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

# one step at the level s draws: the example's `do_step` with its schedule
# replaced, and the config named at the call (Train.step reads that one,
# BendTorch#24)
def do_step(+s: Nat, tr: T.Train<M.B4P()>, +x0: List<&2, F32>) -> T.Train<M.B4P()>:
  +t = level(s)
  T.Train.step(~M.B4P(), ~M.X(), ~M.Y(), ~M.B4S(M.n()), ~M.Y(), ~M.Model(M.n()), ~M.mse, cfg(), tr,
    TA.Mat.from_list(M.n(), M.inwidth(), M.input_of(s, t, M.coef_a(t), M.coef_b(t), x0)),
    M.noise_of(s))

# step 1 and every 50th after it (not the Train's initial 0.0 at i = 0)
def should_print(+i: Nat) -> Bool:
  Bool.and(Nat.is_lt(0n, i),
    Bool.or(Nat.is_eq(i, 1n), Nat.is_eq(Nat.mod(i, 50n), 0n)))

# the first loss is step 1's; `first` carries it from there
def first_after(+i: Nat, +l: F32, +first: F32) -> F32:
  Bool.pick(F32, Nat.is_eq(i, 1n), l, first)

def print_or(sp: Bool, +l: F32) -> IO(Unit):
  match sp:
    case True{}:
      do IO<Unit>:
        IO.print(String.append("loss: ", F32.show(l)))
        return Unit{}
    case False{}:
      do IO<Unit>:
        return Unit{}

def loop(rows_left: Nat, +i: Nat, +first: F32, +x0: List<&2, F32>, tr: T.Train<M.B4P()>) ->
  IO(T.Train<M.B4P()>):
  match rows_left tr:
    case 0n T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<M.B4P()>>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        IO.print(String.append("loss_drop: ", F32.show((first / l : F32))))
        return T.Train{p, g, st, q1, q2, l}
    case 1n+rest T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<M.B4P()>>:
        Unit <- print_or(should_print(i), l)
        loop(rest, (1n + i : Nat), first_after(i, l, first), x0,
          do_step(i, T.Train{p, g, st, q1, q2, l}, x0))

def cifar_of(r: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(List<&2, F32>):
  match r:
    case Fail{e}:
      do IO<List<&2, F32>>:
        IO.print("the dataset failed")
        return Nil{}
    case Done{t}:
      do IO<List<&2, F32>>:
        return M.cifar_rows(t)

# the first `loss:` line is step 1's, so the twin compares the curve from there
def train_and_report(+x0: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    tr : T.Train<M.B4P()> <- loop(steps(), 0n, 0.0, x0,
      T.Train.new(~M.B4P(), ~M.X(), ~M.Y(), ~M.B4S(M.n()), ~M.Model(M.n()), cfg(), M.params()))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.csv_of(
      "arwics/cifar10", "f44619abdb57391606d732724f9fc40a684471bf", "train.csv", False{}, 0n)
    x0 : List<&2, F32> <- cifar_of(t)
    train_and_report(x0)
