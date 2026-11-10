# The stage split of one hybrid training iteration at 128 positions.
#
# The training loop in `train_h128.bend` reports one number per iteration: the
# batch read and `T.Train.step` under one timer. This program times the same
# work in parts, with the same model, the same weights and the same corpus rows:
#
#   load       the batch and labels, decoded to Nat lists
#   forward    `T.Layer.fwd` on the model
#   loss       the masked cross-entropy and its dy
#   backward   `T.Layer.bwd` with that dy
#   update     the AdamW guard and `T.Layer.step`
#
# `full_step_x3_ms` is three full iterations (read the row, then one
# `T.Train.step`), so it can be held against the training loop's trace and
# `loss_after_3` is the loop's own third loss. The parts are timed on a separate
# `Train` state from the same weights, so their sum is close to, not exactly, the
# step; the report prints the difference as unaccounted in the step.
#
# Run it from the repository root:
#
#   bendenv run -- bend bench/plumbing_glm53/main.bend
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ../../examples/language/glm53_flash/model_h.bend as M
import ../../examples/language/glm53_flash/weights_h.bend as W
import ../../examples/language/glm53_flash/loss_masked.bend as L

def ROWS() -> Nat:
  128n

def IGNORE() -> Nat:
  260n

def LR() -> F32:
  0.003

def WEIGHTS() -> String:
  "examples/language/glm53_flash/data/h"

def DATA() -> String:
  "examples/language/glm53_flash/data/corpus128-b32"

def cfg(+lr: F32) -> T.OptCfg:
  T.CAdamW{lr, 0.9, 0.95, 0.00000001, 0.1, 0.1, 0.05}

def ce(z: M.Y(), labels: List<&2, Nat>) -> T.LossOut<M.Y()>:
  L.CrossEntropyMasked(ROWS(), 260n, z, labels, IGNORE())

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def show_ms(name: String, total: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(total))))

def step(tr: T.Train<M.P()>, +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> T.Train<M.P()>:
  T.Train.step(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~List<&2, Nat>, ~M.Model(),
    ~ce, tr, ids, labels)

def loss_of(tr: T.Train<M.P()>) -> F32:
  match tr:
    case T.Train{p, g, st, c, p1, p2, +l}:
      l

# Three training iterations exactly as the loop does them: read the row, then
# one `T.Train.step`. The rows are 0, 1 and 2, so `loss_after_3` is the loop's
# own third loss.
def loop_n(+left: Nat, +i: Nat, tr: T.Train<M.P()>, +data: String) -> IO(T.Train<M.P()>):
  match left:
    case 0n:
      IO.pure(T.Train<M.P()>, tr)
    case 1n+q:
      do IO<T.Train<M.P()>>:
        ids : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-train_batch.bin"),
          U32.from_nat(Nat.mul(i, ROWS())), U32.from_nat(ROWS()))
        labels : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-train_labels.bin"),
          U32.from_nat(Nat.mul(i, ROWS())), U32.from_nat(ROWS()))
        tr1 : T.Train<M.P()> <- IO.pure(T.Train<M.P()>, step(tr, nats(ids), nats(labels)))
        loop_n(q, (i + 1n : Nat), tr1, data)

# The four parts of one step on a state built from the same weights. Each part
# is forced through `IO.pure` inside its timer, and the chain ends at
# `Layer.step`; `split_loss` names the loss of the step the split covers.
def split_done(stp: T.Step<M.P()>, +sl: F32) -> IO(Unit):
  match stp:
    case T.Step{p4, g4, st3}:
      IO.print(String.append("split_loss: ", F32.show(sl)))

def split_update(gd: T.Guarded<M.P()>, p3: M.P(), g3: M.P(), +sl: F32) -> IO(Unit):
  match gd:
    case T.Guarded{o, st2}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        stp : T.Step<M.P()> <- IO.pure(T.Step<M.P()>,
          T.Layer.step(M.P(), M.X(), M.Y(), M.S(), M.Model(), o, p3, g3, st2))
        t1 : Nat <- IO.now()
        show_ms("update", Nat.sub(t1, t0))
        split_done(stp, sl)

def split_after_bwd(bw: T.Bwd<M.P(), M.X()>, +sl: F32, st: T.OptState<M.P()>,
    +oc: T.OptCfg, +p1: F32, +p2: F32) -> IO(Unit):
  match bw:
    case T.Bwd{p3, dx, g3}:
      do IO<Unit>:
        gd : T.Guarded<M.P()> <- IO.pure(T.Guarded<M.P()>,
          T.Opt.guard(M.P(), oc,
            (p1 + (T.OptCfg.c1(oc) * (1.0 - p1 : F32) : F32) : F32),
            (p2 + (T.OptCfg.c2(oc) * (1.0 - p2 : F32) : F32) : F32), st))
        split_update(gd, p3, g3, sl)

def split_after_loss(lo: T.LossOut<M.Y()>, ss: M.S(), pp: M.P(), g: M.P(),
    st: T.OptState<M.P()>, oc: T.OptCfg, +p1: F32, +p2: F32) -> IO(Unit):
  match lo:
    case T.LossOut{+sl, dy}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        bw : T.Bwd<M.P(), M.X()> <- IO.pure(T.Bwd<M.P(), M.X()>,
          T.Layer.bwd(M.P(), M.X(), M.Y(), M.S(), M.Model(), pp, ss, dy, g))
        t1 : Nat <- IO.now()
        show_ms("backward", Nat.sub(t1, t0))
        split_after_bwd(bw, sl, st, oc, p1, p2)

def split_after_fwd(fwd: T.Fwd<M.P(), M.Y(), M.S()>, +labels: List<&2, Nat>, g: M.P(),
    st: T.OptState<M.P()>, oc: T.OptCfg, +p1: F32, +p2: F32) -> IO(Unit):
  match fwd:
    case T.Fwd{pp, y, ss}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        lo : T.LossOut<M.Y()> <- IO.pure(T.LossOut<M.Y()>, ce(y, labels))
        t1 : Nat <- IO.now()
        show_ms("loss", Nat.sub(t1, t0))
        split_after_loss(lo, ss, pp, g, st, oc, p1, p2)

def bench_split(+ids: List<&2, Nat>, +labels: List<&2, Nat>, tr: T.Train<M.P()>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, oc, p1, p2, l}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        fwd : T.Fwd<M.P(), M.Y(), M.S()> <- IO.pure(T.Fwd<M.P(), M.Y(), M.S()>,
          T.Layer.fwd(M.P(), M.X(), M.Y(), M.S(), M.Model(), p, ids))
        t1 : Nat <- IO.now()
        show_ms("forward", Nat.sub(t1, t0))
        split_after_fwd(fwd, labels, g, st, oc, p1, p2)

def bench() -> IO(Unit):
  do IO<Unit>:
    t0 : Nat <- IO.now()
    p0 : M.P() <- W.load_params(WEIGHTS())
    t1 : Nat <- IO.now()
    show_ms("init_params", Nat.sub(t1, t0))
    IO.print(String.append("params: ",
      U32.show(T.Layer.count(M.P(), M.X(), M.Y(), M.S(), M.Model()))))

    t2 : Nat <- IO.now()
    ids : List<&2, U32> <- T.read_at(String.append(DATA(), "/corpus128-train_batch.bin"), 0,
      U32.from_nat(ROWS()))
    labels : List<&2, U32> <- T.read_at(String.append(DATA(), "/corpus128-train_labels.bin"), 0,
      U32.from_nat(ROWS()))
    +ids2 : List<&2, Nat> <- IO.pure(List<&2, Nat>, nats(ids))
    +labels2 : List<&2, Nat> <- IO.pure(List<&2, Nat>, nats(labels))
    t3 : Nat <- IO.now()
    show_ms("load", Nat.sub(t3, t2))

    t4 : Nat <- IO.now()
    tr0 : T.Train<M.P()> <- IO.pure(T.Train<M.P()>,
      T.Train.new(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~M.Model(), cfg(LR()), p0))
    t5 : Nat <- IO.now()
    show_ms("train_new", Nat.sub(t5, t4))

    t6 : Nat <- IO.now()
    tr1 : T.Train<M.P()> <- loop_n(3n, 0n, tr0, DATA())
    t7 : Nat <- IO.now()
    show_ms("full_step_x3", Nat.sub(t7, t6))
    IO.print(String.append("loss_after_3: ", F32.show(loss_of(tr1))))

    p1 : M.P() <- W.load_params(WEIGHTS())
    tr2 : T.Train<M.P()> <- IO.pure(T.Train<M.P()>,
      T.Train.new(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~M.Model(), cfg(LR()), p1))
    bench_split(ids2, labels2, tr2)

def main() -> IO(Unit):
  bench()
