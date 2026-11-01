# Where one CIFAR-10 DDPM step's milliseconds go: the noise, the input and
# target builds, the step itself and one sampler, on the CIFAR example's own
# functions.
#
#   MLBEND_DATA=reference/.cache/datasets bendenv run -- bend bench/plumbing_cifar/main.bend
#
# `examples/cifar/main.bend`'s `do_step` builds 786,432 counter-Gaussians on the
# host (`G.noise_of`), the 256 x 3080 input list, the 256 x 3072 target, then
# `T.Train.step`. This probe times those pieces apart, so the host share of a
# step is a number instead of an assumption. Every value is printed after its
# timer, so the compiler cannot erase the work as unused.
import Base
import bendtorch@1.1.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import ../../examples/cifar/main.bend as M

def show_ms(+name: String, +ms: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(ms))))

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

    t2 : Nat <- IO.now()
    noise : List<&2, F32> <- IO.pure(List<&2, F32>, M.noise_of(1n))
    t3 : Nat <- IO.now()
    Unit <- show_ms("noise", Nat.sub(t3, t2))

    t4 : Nat <- IO.now()
    m_in : TA.Mat<M.n(), M.inwidth()> <- IO.pure(TA.Mat<M.n(), M.inwidth()>,
      TA.Mat.from_list(M.n(), M.inwidth(), M.input_of(1n, 1n, M.coef_a(1n), M.coef_b(1n), x0)))
    t5 : Nat <- IO.now()
    Unit <- show_ms("input", Nat.sub(t5, t4))

    noise2 : List<&2, F32> <- IO.pure(List<&2, F32>, M.noise_of(1n))
    t6 : Nat <- IO.now()
    m_tgt : TA.Mat<M.n(), M.d()> <- IO.pure(TA.Mat<M.n(), M.d()>,
      TA.Mat.from_list(M.n(), M.d(), noise2))
    t7 : Nat <- IO.now()
    Unit <- show_ms("target", Nat.sub(t7, t6))

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
