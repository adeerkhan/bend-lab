# The plumbing split of one ViT training step (ISSUE-018).
#
# The GPU lane runs the model with the same wall time as the CPU lane. The
# hypothesis is that the step is plumbing, not arithmetic: parameter
# initialisation, the per-step row build, the list-to-matrix conversion, and
# then one forward. This program times each of those phases on the real ViT,
# with the real functions, so the hypothesis is a number instead of a guess.
#
# Run it from the repository root:
#
#   bendenv run -- bend bench/plumbing/main.bend
#
# Every phase is reported in milliseconds. `step_in` is the input half of
# `do_step` (batch, noise, mix, rows, matrix) and `forward` is the model on
# that same matrix. `step_tgt` is the target half. `rows` is the row list
# before the matrix, so `step_in - rows` is the Mat.from_list conversion.
# What is left of the full step is loss, backward and the optimizer.
#
# Every measured value is printed after its timer, so the compiler cannot drop
# the work as unused: a pure phase that returns Unit is erased and times as 0.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../../examples/diffusion/vit/main.bend as M

def show_ms(name: String, total: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(total))))

# the 1024 rows of one step, exactly as step_in builds them, before the matrix
def rows_of(+s: Nat, x0: List<&2, F32>) -> List<&2, F32>:
  +batch = M.batch_of(s, x0)
  +noise = G.noise_of(s, 4096n)
  M.rows_img(M.patch_all(64n,
      M.mix_img(batch, noise, M.ab_img(s, 64n, Nil{}), Nil{}), Nil{}),
    M.te_img(s, 64n, Nil{}), 0n, 0n, 0n, Nil{}, Nil{})

def take_fwd(f: T.Fwd<M.ModelP, TA.Mat<M.RW(), M.PX()>, M.ModelS>) -> M.ModelP:
  match f:
    case T.Fwd{p, y, s}:
      p

def fwd_one(p: M.ModelP(), x: TA.Mat<M.RW(), M.FE()>) -> M.ModelP:
  take_fwd(T.Layer.fwd(M.ModelP(), TA.Mat<M.RW(), M.FE>(),
    TA.Mat<M.RW(), M.PX()>(), M.ModelS(), M.Model(), p, x))

def step_n(left: Nat, +i: Nat, tr: T.Train<M.ModelP()>,
  +x0: List<&2, F32>) -> T.Train<M.ModelP()>:
  match left:
    case 0n:
      tr
    case 1n+q:
      step_n(q, (1n + i : Nat), M.do_step(i, tr, x0), x0)

def train_loss(tr: T.Train<M.ModelP()>) -> F32:
  match tr:
    case T.Train{p, g, st, c, p1, p2, l}:
      l

def bench_with(+x0: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("params: ",
      U32.show(T.Layer.count(M.ModelP(), TA.Mat<M.RW(), M.FE>(),
        TA.Mat<M.RW(), M.PX()>(), M.ModelS(), M.Model()))))

    t0 : Nat <- IO.now()
    p0 : M.ModelP <- IO.pure(M.ModelP, M.init_params())
    t1 : Nat <- IO.now()
    Unit <- show_ms("init_params", Nat.sub(t1, t0))

    t2 : Nat <- IO.now()
    rows : List<&2, F32> <- IO.pure(List<&2, F32>, rows_of(1n, x0))
    t3 : Nat <- IO.now()
    Unit <- show_ms("rows", Nat.sub(t3, t2))
    Unit <- IO.print(String.append("rows_head: ", F32.show(M.head_or_one(rows))))

    t4 : Nat <- IO.now()
    m_in : TA.Mat<M.RW(), M.FE()> <- IO.pure(TA.Mat<M.RW(), M.FE()>, M.step_in(1n, x0))
    t5 : Nat <- IO.now()
    Unit <- show_ms("step_in", Nat.sub(t5, t4))

    t6 : Nat <- IO.now()
    p1 : M.ModelP <- IO.pure(M.ModelP, fwd_one(p0, m_in))
    t7 : Nat <- IO.now()
    Unit <- show_ms("forward", Nat.sub(t7, t6))

    t8 : Nat <- IO.now()
    m_tgt : TA.Mat<M.RW(), M.PX()> <- IO.pure(TA.Mat<M.RW(), M.PX()>(), M.step_tgt(1n))
    t9 : Nat <- IO.now()
    Unit <- show_ms("step_tgt", Nat.sub(t9, t8))
    Unit <- IO.print(String.append("tgt_head: ",
      F32.show(M.head_or_one(M.flat_m2(TA.Mat.to_list(M.RW(), M.PX(), m_tgt))))))

    t10 : Nat <- IO.now()
    tr0 : T.Train<M.ModelP()> <- IO.pure(T.Train<M.ModelP()>,
      T.Train.new(~M.ModelP(), ~TA.Mat<M.RW(), M.FE>(),
        ~TA.Mat<M.RW(), M.PX()>(), ~M.ModelS(), ~M.Model(), M.cfg(), p1))
    t11 : Nat <- IO.now()
    Unit <- show_ms("train_new", Nat.sub(t11, t10))

    t12 : Nat <- IO.now()
    tr1 : T.Train<M.ModelP()> <- IO.pure(T.Train<M.ModelP()>, step_n(3n, 1n, tr0, x0))
    t13 : Nat <- IO.now()
    Unit <- show_ms("full_step_x3", Nat.sub(t13, t12))
    Unit <- IO.print(String.append("loss_after_3: ", F32.show(train_loss(tr1))))

    return Unit{}

def main() -> IO(Unit):
  bench_with(M.ones_of(65536n, 65536n, Nil{}))
