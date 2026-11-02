# Rectified flow on two moons, written from scratch in Bend on BendTorch.
#
#   bendenv run -- bend examples/diffusion/flow_matching/main.bend
#
# Flow matching replaces the diffusion chain with one vector field and an ODE:
#
#   forward   x_t = noise*(1-t) + data*t      (t = 0 noise, t = 1 data)
#   target    v   = data - noise
#   loss      MSE(v_theta(x_t, t), v)
#   sample    x <- x + v_theta(x, t)*dt,  Euler over linspace(0, 1, steps+1)
#
# There is no beta schedule, no posterior, no sigma and no eta. The level of
# each point comes from a U32 Weyl sequence (ISSUE-017: a Nat product would
# fail-stop past 2^48-1). The data is the two-moons set of the DDPM example,
# imported rather than copied, and the model is three linear layers with SiLU,
# trained by `T.Train.step` with AdamW.
#
# The reference is `reference/flow_probe.py`, the NumPy probe written before
# this port: its measured targets are loss 2.1253 -> 0.8662 (last 100),
# samples mean [-0.111, 0.150] / std [0.546, 0.745] against its data's
# [-0.002, -0.000] / [0.583, 0.741], and 465/512 samples inside the data's
# box. This example runs the same method on the repository's own 128-point
# data, so its numbers are compared with its own data, not the probe's.
#
#   --steps N       steps to run (default 6000)
#   --start N       the global index of the first step (default 0)
#   --save PATH     write the training state to PATH with Train.save
#   --resume PATH   restore it with Train.restore (the identity is checked by
#                   reference/flow_checkpoint.py before this program starts)
#   --no-sample     skip the Euler sample (a checkpoint probe does not need it)
#
# Without flags the program prints the same numbers as before. A run stopped
# after k steps and resumed from its checkpoint computes the same bits as a
# run that never stopped (BendTorch's Train.save/restore contract).
import Base
import bendtorch@1.1.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../two_moons/main.bend as D2

def n() -> Nat:
  128n

def d() -> Nat:
  2n

def inw() -> Nat:
  3n

def hidden() -> Nat:
  128n

def steps() -> Nat:
  6000n

def euler() -> Nat:
  32n

def X() -> Type:
  TA.Mat<n(), inw()>

def Y() -> Type:
  TA.Mat<n(), d()>

def M2(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def at_go(xs: List<&2, F32>, +k: Nat, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      Bool.pick(F32, Nat.is_eq(i, k), h, at_go(t, k, (1n + i : Nat)))

def at(xs: List<&2, F32>, k: Nat) -> F32:
  at_go(xs, k, 0n)

# the probe normalises its moons to [-1, 1] per axis; the DDPM example's data
# runs x in [-1, 2] and y in [-0.5, 1], so map it the same way first.
def norm_go(+left: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +i = Nat.mul(Nat.sub(n(), left), 2n)
      a1 = List.append(&2, F32, acc,
        Con{(((2.0 * at(xs, i) : F32) - 1.0 : F32) / 3.0 : F32), Nil{}})
      norm_go(rest, xs,
        List.append(&2, F32, a1,
          Con{(((2.0 * at(xs, (1n + i : Nat)) : F32) - 0.5 : F32) / 1.5 : F32), Nil{}}))

def norm(+xs: List<&2, F32>) -> List<&2, F32>:
  norm_go(n(), xs, Nil{})

# ---- the model: x_t beside its time, three linear layers with SiLU ----

def B1P() -> Type:
  T.Two<T.Lin<inw(), hidden()>, Unit>

def B1S() -> Type:
  T.Two<M2(n(), inw()), M2(n(), hidden())>

def B2P() -> Type:
  T.Two<B1P, T.Lin<hidden(), hidden()>>

def B2S() -> Type:
  T.Two<B1S, M2(n(), hidden())>

def B3P() -> Type:
  T.Two<B2P, Unit>

def B3S() -> Type:
  T.Two<B2S, M2(n(), hidden())>

def B4P() -> Type:
  T.Two<B3P, T.Lin<hidden(), d()>>

def B4S() -> Type:
  T.Two<B3S, M2(n(), hidden())>

def B1() -> T.Layer<B1P, X(), M2(n(), hidden()), B1S>:
  T.Seq(~T.Lin<inw(), hidden()>, ~Unit, ~X(), ~M2(n(), hidden()),
    ~M2(n(), hidden()), ~M2(n(), inw()), ~M2(n(), hidden()),
    ~T.LinearFirst(n(), inw(), hidden(), 0n), ~T.SiLU(n(), hidden()))

def B2() -> T.Layer<B2P, X(), M2(n(), hidden()), B2S>:
  T.Seq(~B1P, ~T.Lin<hidden(), hidden()>, ~X(), ~M2(n(), hidden()),
    ~M2(n(), hidden()), ~B1S, ~M2(n(), hidden()), ~B1(),
    ~T.Linear(n(), hidden(), hidden(), 0n))

def B3() -> T.Layer<B3P, X(), M2(n(), hidden()), B3S>:
  T.Seq(~B2P, ~Unit, ~X(), ~M2(n(), hidden()), ~M2(n(), hidden()),
    ~B2S, ~M2(n(), hidden()), ~B2(), ~T.SiLU(n(), hidden()))

def Model() -> T.Layer<B4P, X(), Y(), B4S>:
  T.Seq(~B3P, ~T.Lin<hidden(), d()>, ~X(), ~M2(n(), hidden()), ~Y(),
    ~B3S, ~M2(n(), hidden()), ~B3(), ~T.Linear(n(), hidden(), d(), 0n))

# the starting weights: a fixed small pattern, as in the DDPM example
def value(+i: Nat, +j: Nat, +seed: Nat) -> F32:
  F32.sub(F32.mul(F32.from_nat(Nat.mod(Nat.add(Nat.add(Nat.mul(i, 3n), Nat.mul(j, 5n)), seed), 7n)),
    0.1), 0.2)

def row_vals(+c: Nat, +i: Nat, +j: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match c:
    case 0n:
      acc
    case 1n+rest:
      +s = (1.0 / F32.sqrt(F32.from_nat(c)) : F32)
      row_vals(rest, i, (1n + j : Nat), seed,
        List.append(&2, F32, acc, Con{(value(i, j, seed) * s : F32), Nil{}}))

def rows_vals(+r: Nat, +c: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match r:
    case 0n:
      acc
    case 1n+rest:
      rows_vals(rest, c, (1n + i : Nat), seed,
        List.append(&2, F32, acc, row_vals(c, i, 0n, seed, Nil{})))

def mat_of(+r: Nat, +c: Nat, +seed: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, rows_vals(r, c, 0n, seed, Nil{}))

def params() -> B4P():
  T.Two{
    T.Two{
      T.Two{T.Two{T.Lin{mat_of(hidden(), inw(), 1n), TA.Mat.fill(1n, hidden(), 0.0)}, Unit{}},
      T.Lin{mat_of(hidden(), hidden(), 2n), TA.Mat.fill(1n, hidden(), 0.0)}},
    Unit{}},
  T.Lin{mat_of(d(), hidden(), 3n), TA.Mat.fill(1n, d(), 0.0)}}

# ---- the time levels: one per point, U32 Weyl ----

def level(+s: Nat, +i: Nat) -> F32:
  +h = U32.add(U32.mul(U32.from_nat(s), 2654435761),
    U32.mul(U32.from_nat(i), 2246822519))
  (F32.from_nat(U32.to_nat(h)) / 4294967296.0 : F32)

# ---- the training pair: x_t and its target ----

def input_row(+s: Nat, +r: Nat, +x0: List<&2, F32>, +x1: List<&2, F32>,
  +acc: List<&2, F32>) -> List<&2, F32>:
  +i = Nat.mul(r, 2n)
  +t = level(s, r)
  +xtx = (((1.0 - t : F32) * at(x0, i) : F32) + (t * at(x1, i) : F32) : F32)
  +xty = (((1.0 - t : F32) * at(x0, (1n + i : Nat)) : F32)
    + (t * at(x1, (1n + i : Nat)) : F32) : F32)
  a1 = List.append(&2, F32, acc, Con{xtx, Nil{}})
  a2 = List.append(&2, F32, a1, Con{xty, Nil{}})
  List.append(&2, F32, a2, Con{t, Nil{}})

def input_go(+s: Nat, +x0: List<&2, F32>, +x1: List<&2, F32>,
  +left: Nat, +r: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      input_go(s, x0, x1, rest, (1n + r : Nat),
        input_row(s, r, x0, x1, acc))

def input_of(+s: Nat, +x0: List<&2, F32>, +x1: List<&2, F32>) -> List<&2, F32>:
  input_go(s, x0, x1, n(), 0n, Nil{})

def target_go(+x0: List<&2, F32>, +x1: List<&2, F32>,
  +left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +i = Nat.mul(Nat.sub(n(), left), 2n)
      a1 = List.append(&2, F32, acc,
        Con{(at(x1, i) - at(x0, i) : F32), Nil{}})
      target_go(x0, x1, rest,
        List.append(&2, F32, a1,
          Con{(at(x1, (1n + i : Nat)) - at(x0, (1n + i : Nat)) : F32), Nil{}}))

def target_of(+x0: List<&2, F32>, +x1: List<&2, F32>) -> List<&2, F32>:
  target_go(x0, x1, n(), Nil{})

# ---- the step ----

def cfg() -> T.OptCfg:
  T.CAdamW{0.003, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

def mse(y: Y(), tgt: Y()) -> T.LossOut<Y()>:
  T.MSE(n(), d(), y, tgt)

def do_step(+s: Nat, tr: T.Train<B4P()>, +data: List<&2, F32>) -> T.Train<B4P()>:
  +x0 = G.noise_of(s, 256n)
  T.Train.step(~B4P(), ~X(), ~Y(), ~B4S(), ~Y(), ~Model(), ~mse, cfg(), tr,
    TA.Mat.from_list(n(), inw(), input_of(s, x0, data)),
    TA.Mat.from_list(n(), d(), target_of(x0, data)))

def loop(left: Nat, +i: Nat, tr: T.Train<B4P()>,
  +data: List<&2, F32>) -> IO(T.Train<B4P()>):
  match left tr:
    case 0n T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        return T.Train{p, g, st, q1, q2, l}
    case 1n+rest T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        IO.print(String.append("loss: ", F32.show(l)))
        loop(rest, (1n + i : Nat), do_step(i, T.Train{p, g, st, q1, q2, l}, data), data)

def train_params(tr: T.Train<B4P()>) -> B4P():
  match tr:
    case T.Train{p, g, st, q1, q2, l}:
      p

# ---- Euler sampling: x <- x + v(x, t) * dt ----

type ES is Type:
  ES{p: B4P(), x: List<&2, F32>}

def ml_list(m: TA.ML<n(), d()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def flat_of(m: TA.Mat<n(), d()>) -> List<&2, F32>:
  ml_list(TA.Mat.to_list(n(), d(), m))

def euler_x_go(+x: List<&2, F32>, +v: List<&2, F32>,
  +left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +i = Nat.mul(Nat.sub(n(), left), 2n)
      a1 = List.append(&2, F32, acc,
        Con{(at(x, i) + (0.03125 * at(v, i) : F32) : F32), Nil{}})
      euler_x_go(x, v, rest,
        List.append(&2, F32, a1,
          Con{(at(x, (1n + i : Nat)) + (0.03125 * at(v, (1n + i : Nat)) : F32) : F32),
            Nil{}}))

def sample_row(+t: F32, +r: Nat, +x: List<&2, F32>,
  +acc: List<&2, F32>) -> List<&2, F32>:
  +i = Nat.mul(r, 2n)
  a1 = List.append(&2, F32, acc, Con{at(x, i), Nil{}})
  a2 = List.append(&2, F32, a1, Con{at(x, (1n + i : Nat)), Nil{}})
  List.append(&2, F32, a2, Con{t, Nil{}})

def sample_input_go(+t: F32, +x: List<&2, F32>, +left: Nat, +r: Nat,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      sample_input_go(t, x, rest, (1n + r : Nat), sample_row(t, r, x, acc))

def sample_input(+t: F32, +x: List<&2, F32>) -> List<&2, F32>:
  sample_input_go(t, x, n(), 0n, Nil{})

def take_v(f: T.Fwd<B4P, Y(), B4S()>, +x: List<&2, F32>) -> ES:
  match f:
    case T.Fwd{p, y, s}:
      ES{p, euler_x_go(x, flat_of(y), n(), Nil{})}

def fwd_v(p: B4P(), +t: F32, +x: List<&2, F32>) -> ES:
  take_v(T.Layer.fwd(B4P(), X(), Y(), B4S(), Model(), p,
    TA.Mat.from_list(n(), inw(), sample_input(t, x))), x)

def euler_go(+left: Nat, +i: Nat, st: ES) -> ES:
  match left:
    case 0n:
      st
    case 1n+rest:
      match st:
        case ES{p, +x}:
          +t = (F32.from_nat(i) / 32.0 : F32)
          euler_go(rest, (1n + i : Nat), fwd_v(p, t, x))

def sample(p: B4P()) -> ES:
  euler_go(euler(), 0n, ES{p, G.noise_of(2000000n, 256n)})

def sample_x(st: ES) -> List<&2, F32>:
  match st:
    case ES{p, x}:
      x

# ---- printing ----

def show_fs_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_fs_go(t)))

def show_pt(tag: String, p: D2.Pt) -> IO(Unit):
  match p:
    case D2.Pt{x, y}:
      IO.print(String.append(tag,
        String.append(F32.show(x), String.append(" ", F32.show(y)))))

def report_x(+x: List<&2, F32>, +data: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    Unit <- show_pt("sample_mean:", D2.means(x))
    Unit <- show_pt("sample_std:", D2.stds(x))
    IO.print(String.append("samples: ", show_fs_go(x)))

def report_sample(p: B4P(), +data: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    x : List<&2, F32> <- IO.pure(List<&2, F32>, sample_x(sample(p)))
    Unit <- show_pt("data_mean:", D2.means(data))
    Unit <- show_pt("data_std:", D2.stds(data))
    report_x(x, data)

# ---- a checkpoint: the state Train.save writes, and Train.restore reads ----

def floats_at(+path: String) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    bs : List<&2, U32> <- T.read_all(path)
    return T.f32s(bs)

def fill_go(left: Nat, xs: List<&2, F32>, +i: U32, arr: Array<F32>) -> Array<F32>:
  match left xs:
    case 0n _:
      arr
    case 1n+rest Con{h, t}:
      fill_go(rest, t, U32.add(i, U32.from_nat(1n)), Array.set(F32, arr, i, h))
    case _ _:
      arr

# parameters, Adam's first and second moments, then 1 - b1^t and 1 - b2^t
def floats_count() -> Nat:
  Nat.add(Nat.mul(3n, U32.to_nat(T.Layer.count(B4P(), X(), Y(), B4S(), Model()))), 2n)

def new_train() -> T.Train<B4P()>:
  T.Train.new(~B4P(), ~X(), ~Y(), ~B4S(), ~Model(), cfg(), params())

def restore_arr(xs: List<&2, F32>) -> T.Train<B4P()>:
  T.Train.restore(~B4P(), ~X(), ~Y(), ~B4S(), ~Model(), new_train(),
    fill_go(floats_count(), xs, U32.from_nat(0n), Array.new(F32, T.depth(floats_count()), 0.0)))

def resumed(+path: String) -> IO(T.Train<B4P()>):
  do IO<T.Train<B4P()>>:
    xs : List<&2, F32> <- floats_at(path)
    return restore_arr(xs)

def save_tr(+path: String, tr: T.Train<B4P()>) -> IO(T.Train<B4P()>):
  do IO<T.Train<B4P()>>:
    kept : T.Train<B4P()> <- T.Train.save(~B4P(), ~X(), ~Y(), ~B4S(), ~Model(), path, tr)
    Unit <- IO.print(String.append("saved: ", path))
    return kept

# ---- the flags: --steps N, --start N, --save PATH, --resume PATH, --no-sample ----

type Run is Data:
  Run{start: Nat, steps: Nat, save: String, resume: String, sample: Bool}

def run_default() -> Run:
  Run{0n, steps(), "", "", True{}}

def nat_or(m: Maybe<&2, Nat>, d: Nat) -> Nat:
  match m:
    case Some{x}:
      x
    case None{}:
      d

def run_go(args: List<String>, r: Run) -> Run:
  match args:
    case Nil{}:
      r
    case Con{a, rest}:
      match a:
        case "--no-sample":
          match r:
            case Run{start, steps, save, resume, sample}:
              run_go(rest, Run{start, steps, save, resume, False{}})
        case "--save":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{start, steps, save, resume, sample}:
                  run_go(more, Run{start, steps, b, resume, sample})
        case "--resume":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{start, steps, save, resume, sample}:
                  run_go(more, Run{start, steps, save, b, sample})
        case "--start":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{start, steps, save, resume, sample}:
                  run_go(more, Run{nat_or(Nat.read(b), start), steps, save, resume, sample})
        case "--steps":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{start, steps, save, resume, sample}:
                  run_go(more, Run{start, nat_or(Nat.read(b), steps), save, resume, sample})
        case _:
          run_go(rest, r)

def run_of(args: List<String>) -> Run:
  match args:
    case Nil{}:
      run_default()
    case Con{program, rest}:
      run_go(rest, run_default())

# ---- the run ----

def start_tr(resume: String) -> IO(T.Train<B4P()>):
  match resume:
    case SNil{}:
      IO.pure(T.Train<B4P()>, new_train())
    case SCon{h, t}:
      resumed(SCon{h, t})

# the first step at `start`, then the rest: one expression, as `loop` uses its index
def train_steps(+start: Nat, +steps: Nat, tr0: T.Train<B4P()>,
  +data: List<&2, F32>) -> IO(T.Train<B4P()>):
  match steps:
    case 0n:
      IO.pure(T.Train<B4P()>, tr0)
    case 1n+rest:
      loop(rest, (1n + start : Nat), do_step(start, tr0, data), data)

def save_if(save: String, tr: T.Train<B4P()>) -> IO(T.Train<B4P()>):
  match save:
    case SNil{}:
      IO.pure(T.Train<B4P()>, tr)
    case SCon{h, t}:
      save_tr(SCon{h, t}, tr)

def finish(save: String, sample: Bool, tr: T.Train<B4P()>, +data: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    tr1 : T.Train<B4P()> <- save_if(save, tr)
    Bool.pick(IO(Unit), sample, report_sample(train_params(tr1), data),
      IO.pure(Unit, Unit{}))

def run_with(+data: List<&2, F32>, r: Run) -> IO(Unit):
  match r:
    case Run{start, steps, save, resume, sample}:
      do IO<Unit>:
        T.OptCfg.check(cfg())
        tr0 : T.Train<B4P()> <- start_tr(resume)
        tr2 : T.Train<B4P()> <- train_steps(start, steps, tr0, data)
        finish(save, sample, tr2, data)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    run_with(norm(D2.moons()), run_of(args))
