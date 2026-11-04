# Where one CIFAR-10 DDPM step's milliseconds go: the noise, the input and
# target builds, the step itself and one sampler, on the CIFAR example's own
# functions.
#
#   MLBEND_DATA=reference/.cache/datasets bendenv run -- bend bench/plumbing_cifar/main.bend
#
# `examples/cifar/main.bend`'s `do_step` makes 786,432 counter-Gaussians (one
# `fill_gauss` bang for the target's Mat, one for the input's list), the
# 256 x 3080 input list, then `T.Train.step`. This probe times those pieces
# apart, so the host share of a step is a number instead of an assumption.
# Every value is printed after its timer, so the compiler cannot erase the work
# as unused.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import ../../examples/cifar/main.bend as M

def show_ms(+name: String, +ms: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(ms))))

# the flat values of a Mat of the input's shape (the ML record's list)
def ml_list(ml: TA.ML<M.n(), M.inwidth()>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, l}:
      l

# three cells of the input, read back off the timed matrix
def show_cells(+inp: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("input[0]: ", F32.show(M.at(inp, 0n))))
    Unit <- IO.print(String.append("input[73728]: ", F32.show(M.at(inp, 73728n))))
    IO.print(String.append("input[788479]: ", F32.show(M.at(inp, 788479n))))

# the forward's own loss against the target, so the value is used
def loss_of(lo: T.LossOut<M.Y()>) -> F32:
  match lo:
    case T.LossOut{l, dy}:
      l

def fwd_mse(r: T.Fwd<M.B4P(), M.Y(), M.B4S(M.n())>, tgt: TA.Mat<M.n(), M.d()>) -> T.LossOut<M.Y()>:
  match r:
    case T.Fwd{p2, y, s}:
      M.mse(y, tgt)

def fwd_loss(r: T.Fwd<M.B4P(), M.Y(), M.B4S(M.n())>, tgt: TA.Mat<M.n(), M.d()>) -> F32:
  loss_of(fwd_mse(r, tgt))

def step3(+left: Nat, tr: T.Train<M.B4P()>, +x0: List<&2, F32>) -> T.Train<M.B4P()>:
  match left:
    case 0n: tr
    case 1n+p: step3(p, M.do_step(0n, tr, x0), x0)

def bench_with(+x0: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("params: ",
      U32.show(T.Layer.count(M.B4P(), M.M2(M.n(), M.inwidth()),
        M.M2(M.n(), M.d()), M.B4S(M.n()), M.Model(M.n())))))

    t0 : Nat <- IO.now()
    p0 : M.B4P() <- IO.pure(M.B4P, M.params())
    t1 : Nat <- IO.now()
    Unit <- show_ms("init_params", Nat.sub(t1, t0))

    t4 : Nat <- IO.now()
    m_in : TA.Mat<M.n(), M.inwidth()> <- IO.pure(TA.Mat<M.n(), M.inwidth()>,
      TA.Mat.from_list(M.n(), M.inwidth(),
        M.input_of(1n, 1n, M.coef_a(1n), M.coef_b(1n), x0)))
    t5 : Nat <- IO.now()
    Unit <- show_ms("step_in", Nat.sub(t5, t4))

    # the three cells the fills must not move (0, 73728, 788479), read off a
    # second build of the same input: the timed matrix keeps one owner for the
    # forward below, and this read is outside every timer
    Unit <- show_cells(ml_list(TA.Mat.to_list(M.n(), M.inwidth(),
      TA.Mat.from_list(M.n(), M.inwidth(), M.input_of(1n, 1n, M.coef_a(1n), M.coef_b(1n), x0)))))

    t6 : Nat <- IO.now()
    m_tgt : TA.Mat<M.n(), M.d()> <- IO.pure(TA.Mat<M.n(), M.d()>, M.noise_of(1n))
    t7 : Nat <- IO.now()
    Unit <- show_ms("step_tgt", Nat.sub(t7, t6))

    # the forward alone, so the report can put it beside the loss and backward.
    # Its parameters are a second `params()` (3.1 s, outside every timer): the
    # first copy belongs to Train.new below, and a parameter record is affine.
    pf : M.B4P() <- IO.pure(M.B4P, M.params())
    t7a : Nat <- IO.now()
    fwd : T.Fwd<M.B4P(), M.Y(), M.B4S(M.n())> <- IO.pure(T.Fwd<M.B4P(), M.Y(), M.B4S(M.n())>,
      T.Layer.fwd(M.B4P(), M.X(), M.Y(), M.B4S(M.n()), M.Model(M.n()), pf, m_in))
    t7b : Nat <- IO.now()
    Unit <- show_ms("forward", Nat.sub(t7b, t7a))
    Unit <- IO.print(String.append("loss_at_input: ", F32.show(fwd_loss(fwd, m_tgt))))

    t8 : Nat <- IO.now()
    tr0 : T.Train<M.B4P()> <- IO.pure(T.Train<M.B4P()>,
      T.Train.new(~M.B4P(), ~M.X(), ~M.Y(), ~M.B4S(M.n()), ~M.Model(M.n()), M.cfg(), p0))
    t9 : Nat <- IO.now()
    Unit <- show_ms("train_new", Nat.sub(t9, t8))

    t10 : Nat <- IO.now()
    tr1 : T.Train<M.B4P()> <- IO.pure(T.Train<M.B4P()>, step3(3n, tr0, x0))
    t11 : Nat <- IO.now()
    Unit <- show_ms("full_step_x3", Nat.sub(t11, t10))

    p1 : M.B4P() <- IO.pure(M.B4P, M.params_of(tr1))
    t12 : Nat <- IO.now()
    st : M.PS <- IO.pure(M.PS, M.sample(p1))
    t13 : Nat <- IO.now()
    Unit <- show_ms("sampler_64rungs", Nat.sub(t13, t12))
    Unit <- IO.print(String.append("sampler_values: ",
      Nat.show(List.length(&2, F32, M.sample_x(st)))))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.csv_of(
      "arwics/cifar10", "f44619abdb57391606d732724f9fc40a684471bf", "train.csv", False{}, 0n)
    x0 : List<&2, F32> <- M.cifar_of(t)
    bench_with(x0)
