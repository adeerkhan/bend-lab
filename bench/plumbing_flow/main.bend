# Where one flow-matching step's milliseconds go: the noise, the input and
# target builds, and the step itself, on the flow example's own functions.
#
#   bendenv run -- bend bench/plumbing_flow/main.bend
#
# The example's own `do_step` builds 256 counter-Gaussians on the host
# (`G.noise_of`), then the input and target lists, then `T.Train.step`. This
# probe times those pieces apart, so the noise's share is a number instead of
# an assumption. Every value is printed after its timer, so the compiler cannot
# erase the work as unused.
import Base
import bendtorch@1.1.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../../examples/diffusion/two_moons/main.bend as D2
import ../../examples/diffusion/flow_matching/main.bend as M

def show_ms(+name: String, +ms: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(ms))))

def loss_of(tr: T.Train<M.B4P()>) -> F32:
  match tr:
    case T.Train{p, g, st, q1, q2, +l}: l

def step3(+left: Nat, tr: T.Train<M.B4P()>, +data: List<&2, F32>) -> T.Train<M.B4P()>:
  match left:
    case 0n: tr
    case 1n+p: step3(p, M.do_step(0n, tr, data), data)

def bench_with(+data: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("params: ",
      U32.show(T.Layer.count(M.B4P(), TA.Mat<M.n(), M.inw()>(),
        TA.Mat<M.n(), M.d()>(), M.B4S(), M.Model()))))

    t0 : Nat <- IO.now()
    p0 : M.B4P() <- IO.pure(M.B4P, M.params())
    t1 : Nat <- IO.now()
    Unit <- show_ms("init_params", Nat.sub(t1, t0))

    t2 : Nat <- IO.now()
    noise : List<&2, F32> <- IO.pure(List<&2, F32>, G.noise_of(1n, 256n))
    t3 : Nat <- IO.now()
    Unit <- show_ms("noise", Nat.sub(t3, t2))

    t4 : Nat <- IO.now()
    m_in : TA.Mat<M.n(), M.inw()> <- IO.pure(TA.Mat<M.n(), M.inw()>, TA.Mat.from_list(M.n(), M.inw(), M.input_of(1n, noise, data)))
    t5 : Nat <- IO.now()
    Unit <- show_ms("input", Nat.sub(t5, t4))

    noise2 : List<&2, F32> <- IO.pure(List<&2, F32>, G.noise_of(1n, 256n))
    t6 : Nat <- IO.now()
    m_tgt : TA.Mat<M.n(), M.d()> <- IO.pure(TA.Mat<M.n(), M.d()>, TA.Mat.from_list(M.n(), M.d(), M.target_of(noise2, data)))
    t7 : Nat <- IO.now()
    Unit <- show_ms("target", Nat.sub(t7, t6))

    t8 : Nat <- IO.now()
    tr0 : T.Train<M.B4P()> <- IO.pure(T.Train<M.B4P()>,
      T.Train.new(~M.B4P(), ~M.X(), ~M.Y(), ~M.B4S(), ~M.Model(), M.cfg(), p0))
    t9 : Nat <- IO.now()
    Unit <- show_ms("train_new", Nat.sub(t9, t8))

    t10 : Nat <- IO.now()
    tr1 : T.Train<M.B4P()> <- IO.pure(T.Train<M.B4P()>, step3(3n, tr0, data))
    t11 : Nat <- IO.now()
    Unit <- show_ms("full_step_x3", Nat.sub(t11, t10))
    Unit <- IO.print(String.append("loss_after_3: ", F32.show(loss_of(tr1))))
    return Unit{}

def main() -> IO(Unit):
  bench_with(M.norm(D2.moons()))
