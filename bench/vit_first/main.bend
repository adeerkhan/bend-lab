# The ViT example's step-0 loss, and its two sides, in a second instead of a
# 3000-step run.
#
#   MLBEND_DATA=reference/.cache/datasets bendenv run -- bend bench/vit_first/main.bend
#
# `examples/diffusion/vit` prints `loss: 1.0091865` at step 1, and the task's
# gate compares it against `reference/twin_vit.py`'s step 0. This probe runs the
# example's own `init_params` and `do_step(0n, …)` once and prints
#
#   loss_at_0   1.0091865      the number the record carries
#   tgt_sq      0.9993086      mean(target²): the noise, in token order
#   out_sq      0.009290358    mean(model²)
#   in0/out0    40 and 16 values: the first rows of the input and the output
#
# so a gap with the twin names its side rather than its symptom: a target whose
# mean square differs is the noise or the patch order; a model whose does is the
# forward or the embedding; and a row whose te differs is the level assignment.
#
# That is how ISSUE-032 was diagnosed and ISSUE-040 was found: the twin's
# `loss_at(0)` read 1.0093014 because it gave image i its own level, and row 0
# here carries `temb(60)` — `t_img(0, 60) = 60`, image 63's level, because the
# example's `ab_img`/`te_img` reverse an already-ascending list. With the twin
# mirroring that (`twin_vit.levels_of`) both read 1.0091865, to 6.3e-09.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import ../../examples/diffusion/vit/main.bend as V
import ../../examples/diffusion/digits/main.bend as D

def list_of(ml: TA.ML<V.RW(), V.FE()>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, l}:
      l

def list_out(ml: TA.ML<V.RW(), V.PX()>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, l}:
      l

def start() -> T.Train<V.ModelP()>:
  T.Train.new(~V.ModelP(), ~TA.Mat<V.RW(), V.FE()>(), ~TA.Mat<V.RW(), V.PX()>(),
    ~V.ModelS(), ~V.Model(), V.cfg(), V.init_params())

def show(tr: T.Train<V.ModelP()>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, c, q1, q2, +l}:
      IO.print(String.append("loss_at_0: ", F32.show(l)))

def mse_at(y: TA.Mat<V.RW(), V.PX()>, t: TA.Mat<V.RW(), V.PX()>) -> T.LossOut<TA.Mat<V.RW(), V.PX()>>:
  T.MSE(V.RW(), V.PX(), y, t)

def loss_of(lo: T.LossOut<TA.Mat<V.RW(), V.PX()>>) -> F32:
  match lo:
    case T.LossOut{l, dy}:
      l

def mse0(y: TA.Mat<V.RW(), V.PX()>, t: TA.Mat<V.RW(), V.PX()>) -> F32:
  loss_of(mse_at(y, t))

def zeros0() -> TA.Mat<V.RW(), V.PX()>:
  TA.Mat.fill(V.RW(), V.PX(), 0.0)

def eps0(+x0: List<&2, F32>) -> T.Fwd<V.ModelP(), TA.Mat<V.RW(), V.PX()>, V.ModelS()>:
  T.Layer.fwd(V.ModelP(), TA.Mat<V.RW(), V.FE()>(), TA.Mat<V.RW(), V.PX()>(),
    V.ModelS(), V.Model(), V.init_params(), V.step_in(0n, x0))

def sides(tgt: TA.Mat<V.RW(), V.PX()>,
  f: T.Fwd<V.ModelP(), TA.Mat<V.RW(), V.PX()>, V.ModelS()>) -> IO(Unit):
  match f:
    case T.Fwd{p2, y, s}:
      do IO<Unit>:
        Unit <- IO.print(String.append("tgt_sq: ", F32.show(mse0(zeros0(), tgt))))
        IO.print(String.append("out_sq: ", F32.show(mse0(zeros0(), y))))

# the first rows of the input and of the model's output, so a difference with
# the twin names the builder or the forward
def show_go(left: Nat, +px: List<&2, F32>, +i: Nat, +acc: String) -> String:
  match left:
    case 0n:
      acc
    case 1n+rest:
      show_go(rest, px, (1n + i : Nat),
        String.append(acc, String.append(" ", F32.show(V.at(px, i)))))

def dump_io(inl: List<&2, F32>, f: T.Fwd<V.ModelP(), TA.Mat<V.RW(), V.PX()>, V.ModelS()>) -> IO(Unit):
  match f:
    case T.Fwd{p2, y, s}:
      do IO<Unit>:
        Unit <- IO.print(String.append("in0:", show_go(40n, inl, 0n, "")))
        yl : List<&2, F32> <- IO.pure(List<&2, F32>, list_out(TA.Mat.to_list(V.RW(), V.PX(), y)))
        IO.print(String.append("out0:", show_go(16n, yl, 0n, "")))

def dump(+x0: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    m : TA.Mat<V.RW(), V.FE()> <- IO.pure(TA.Mat<V.RW(), V.FE()>, V.step_in(0n, x0))
    inl : List<&2, F32> <- IO.pure(List<&2, F32>, list_of(TA.Mat.to_list(V.RW(), V.FE(), m)))
    dump_io(inl, eps0(x0))

def probe(+x0: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    tr : T.Train<V.ModelP()> <- IO.pure(T.Train<V.ModelP()>, start())
    Unit <- show(V.do_step(0n, tr, x0))
    Unit <- sides(V.step_tgt(0n), eps0(x0))
    dump(x0)

def main() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.csv_of(
      "mstz/optdigits", "4f20cc827b763fb69e2a80e18e892e4e11498aec", "optdigits.data",
      False{}, 0n)
    x0 : List<&2, F32> <- D.digits_of(t)
    probe(x0)
