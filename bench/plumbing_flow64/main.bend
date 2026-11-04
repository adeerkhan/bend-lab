# Where one 64x64 flow step's milliseconds go: the batch read, the pair build
# (noise, mix, tokens), the forward, one full step and one Euler step, on the
# flowers example's own functions.
#
#   MLBEND_DATA=$PWD/reference/.cache/datasets bendenv run -- bend bench/plumbing_flow64/main.bend
#
# `examples/diffusion/flowers_flow` reads a batch (`read_batch`), turns it into
# one input and one target (`pair_of`: the noise, the mix, the token rows), and
# runs `Train.step`. The step at the `corpus` preset is ~470 ms (6,000 steps in
# 47 minutes), so the split is worth having: the token build is Bend list work
# and the forward/backward is BendTorch's. Every value is printed after its
# timer, so the compiler cannot erase the work as unused.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../../examples/diffusion/flowers_flow/main.bend as F

def show_ms(+name: String, +ms: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(ms))))

# the parameter count, the model's shape as one number
def count() -> U32:
  T.Layer.count(F.ModelP(), F.X(), F.Y(), F.ModelS(), F.Model())

# one step at counter s: the example's own
def step3(+left: Nat, tr: T.Train<F.ModelP()>, +data: List<&2, U32>) -> T.Train<F.ModelP()>:
  match left:
    case 0n:
      tr
    case 1n+p:
      step3(p, F.do_step(0n, tr, data), data)

def list_of(ml: TA.ML<F.RW(), F.PX()>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, l}:
      l

def head_of(xs: List<&2, F32>) -> F32:
  match xs:
    case Con{h, t}:
      h
    case Nil{}:
      0.0

# the forward on a built pair, its first output value used so it cannot be erased
def fwd_of(lr: F.LR, p: F.ModelP()) -> T.Fwd<F.ModelP(), F.Y(), F.ModelS()>:
  match lr:
    case F.LR{inp, tgt}:
      T.Layer.fwd(F.ModelP(), F.X(), F.Y(), F.ModelS(), F.Model(), p,
        TA.Mat.from_list(F.RW(), F.FE(), inp))

def head_fwd(f: T.Fwd<F.ModelP(), F.Y(), F.ModelS()>) -> F32:
  match f:
    case T.Fwd{p2, y, s}:
      head_of(list_of(TA.Mat.to_list(F.RW(), F.PX(), y)))

def stages(+data: List<&2, U32>, p0: F.ModelP()) -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("params: ", U32.show(count())))

    t4 : Nat <- IO.now()
    lr : F.LR <- IO.pure(F.LR, F.pair_of(1n, data))
    t5 : Nat <- IO.now()
    Unit <- show_ms("step_in", Nat.sub(t5, t4))

    lr2 : F.LR <- IO.pure(F.LR, F.pair_of(1n, data))
    t6 : Nat <- IO.now()
    Unit <- IO.print(String.append("forward_head: ", F32.show(head_fwd(fwd_of(lr2, p0)))))
    t7 : Nat <- IO.now()
    Unit <- show_ms("forward", Nat.sub(t7, t6))

    t8 : Nat <- IO.now()
    tr0 : T.Train<F.ModelP()> <- IO.pure(T.Train<F.ModelP()>,
      T.Train.new(~F.ModelP(), ~F.X(), ~F.Y(), ~F.ModelS(), ~F.Model(), F.cfg(), F.init_params()))
    t9 : Nat <- IO.now()
    Unit <- show_ms("train_new", Nat.sub(t9, t8))

    t10 : Nat <- IO.now()
    tr1 : T.Train<F.ModelP()> <- IO.pure(T.Train<F.ModelP()>, step3(3n, tr0, data))
    t11 : Nat <- IO.now()
    Unit <- show_ms("full_step_x3", Nat.sub(t11, t10))

    p1 : F.ModelP() <- IO.pure(F.ModelP(), F.train_params(tr1))
    t12 : Nat <- IO.now()
    vs : F.VS <- IO.pure(F.VS, F.euler_go(F.EULER(), 0n, F.VS{p1, G.noise_of(2000000n, F.BATCH())}))
    t13 : Nat <- IO.now()
    Unit <- show_ms("euler_32", Nat.sub(t13, t12))
    Unit <- IO.print(String.append("euler_values: ",
      Nat.show(List.length(&2, F32, F.sample_x(vs)))))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    t0 : Nat <- IO.now()
    p0 : F.ModelP() <- IO.pure(F.ModelP(), F.init_params())
    t1 : Nat <- IO.now()
    Unit <- show_ms("init_params", Nat.sub(t1, t0))

    root : String <- F.export_dir()
    t2 : Nat <- IO.now()
    data : List<&2, U32> <- F.read_batch(root, 1n)
    t3 : Nat <- IO.now()
    Unit <- show_ms("rows", Nat.sub(t3, t2))

    stages(data, p0)
