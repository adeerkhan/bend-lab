# A one-block transformer trained on a periodic sequence, with the real
# BendTorch (`bendtorch@1.3.0.0`), not the old `bendtorch-core` tape.
#
#   bendenv run -- bend examples/transformer/main.bend
#
# The model is an embedding, one causal single-head attention block with a
# residual and a normaliser, and a linear head, trained by `T.Train.step`
# with AdamW on next-token prediction over a periodic sequence 0..7. The
# sequence is deterministic, so the task is reproducible and the loss curve
# is the evidence: it falls from log(8) toward zero as the model learns the
# cycle.
#
# The library is nuxyel/BendTorch (`bendtorch@1.3.0.0`), which has
# `Attention`, `RMSNorm`, `Embedding`, `CrossEntropy`, `AdamW` and
# `Train`; the old `bendtorch-core` line has none of these. This file is
# the first task on the maintained library.
import Base
import bendtorch@1.3.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# The shape: two sequences of sixteen positions, width sixteen, vocabulary
# eight, one attention head.

def positions() -> Nat:
  16n

def sequences() -> Nat:
  2n

def rows() -> Nat:
  Nat.mul(sequences(), positions())

def width() -> Nat:
  16n

def vocab() -> Nat:
  8n

def A(c: Nat) -> Type:
  TA.Mat<rows(), c>

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# ---- the attention block: q, k, v projections, the causal attention, the
# output projection, all wrapped with a normaliser and a residual ----

def KVP() -> Type:
  T.Two<M(width(), width()), M(width(), width())>

def KVY() -> Type:
  T.Two<A(width()), A(width())>

def KVS() -> Type:
  T.Two<A(width()), A(width())>

def KVL() -> T.Layer<KVP, A(width()), KVY, KVS>:
  T.Fork(~M(width(), width()), ~M(width(), width()), ~A(width()), ~A(width()),
    ~A(width()), ~A(width()), ~A(width()),
    ~(m => T.Mat.dup(rows(), width(), m)),
    ~(x => y => T.Mat.sum2(rows(), width(), x, y)),
    ~T.LinearNB(rows(), width(), width(), 0n),
    ~T.LinearNB(rows(), width(), width(), 0n))

def QKVP() -> Type:
  T.Two<M(width(), width()), KVP>

def QKVS() -> Type:
  T.Two<A(width()), KVS>

def QKV() -> T.Layer<QKVP, A(width()), T.Two<A(width()), KVY>, QKVS>:
  T.Fork(~M(width(), width()), ~KVP, ~A(width()), ~A(width()), ~KVY,
    ~A(width()), ~KVS,
    ~(m => T.Mat.dup(rows(), width(), m)),
    ~(x => y => T.Mat.sum2(rows(), width(), x, y)),
    ~T.LinearNB(rows(), width(), width(), 0n), ~KVL())

def AX() -> Type:
  T.Two<A(width()), KVY>

def CoreP() -> Type:
  T.Two<Unit, M(width(), width())>

def CoreS() -> Type:
  T.Two<T.AttnS, A(width())>

def Core() -> T.Layer<CoreP, AX, A(width()), CoreS>:
  T.Seq(~Unit, ~M(width(), width()), ~AX, ~A(width()), ~A(width()),
    ~T.AttnS, ~A(width()),
    ~T.Attention(rows(), width(), width(), sequences(), positions(), 1n, 1n, width(),
      0.25, {==}, {==}, {==}),
    ~T.LinearNB(rows(), width(), width(), 0n))

def AttP() -> Type:
  T.Two<QKVP, CoreP>

def AttS() -> Type:
  T.Two<QKVS, CoreS>

def Att() -> T.Layer<AttP, A(width()), A(width()), AttS>:
  T.Seq(~QKVP, ~CoreP, ~A(width()), ~AX, ~A(width()), ~QKVS, ~CoreS,
    ~QKV(), ~Core())

def NAttP() -> Type:
  T.Two<M(1n, width()), AttP>

def NAttS() -> Type:
  T.Two<A(width()), AttS>

def NAtt() -> T.Layer<NAttP, A(width()), A(width()), NAttS>:
  T.Seq(~M(1n, width()), ~AttP, ~A(width()), ~A(width()), ~A(width()),
    ~A(width()), ~AttS,
    ~T.RMSNorm(rows(), width(), 0.00001), ~Att())

# ---- the head: a normaliser and a linear map to the vocabulary ----

def HeadP() -> Type:
  T.Two<M(1n, width()), M(vocab(), width())>

def HeadS() -> Type:
  T.Two<A(width()), A(width())>

def Head() -> T.Layer<HeadP, A(width()), A(vocab()), HeadS>:
  T.Seq(~M(1n, width()), ~M(vocab(), width()), ~A(width()), ~A(width()),
    ~A(vocab()), ~A(width()), ~A(width()),
    ~T.RMSNorm(rows(), width(), 0.00001),
    ~T.LinearNB(rows(), width(), vocab(), 0n))

def BodyP() -> Type:
  T.Two<NAttP, HeadP>

def BodyS() -> Type:
  T.Two<NAttS, HeadS>

def Body() -> T.Layer<BodyP, A(width()), A(vocab()), BodyS>:
  T.Seq(~NAttP, ~HeadP, ~A(width()), ~A(width()), ~A(vocab()),
    ~NAttS, ~HeadS, ~NAtt(), ~Head())

# ---- the model over token ids ----

def P() -> Type:
  T.Two<M(vocab(), width()), BodyP>

def S() -> Type:
  T.Two<List<&2, Nat>, BodyS>

def X() -> Type:
  List<&2, Nat>

def Y() -> Type:
  A(vocab())

def Model() -> T.Layer<P, X, Y, S>:
  T.Seq(~M(vocab(), width()), ~BodyP, ~X, ~A(width()), ~Y,
    ~List<&2, Nat>, ~BodyS,
    ~T.Embedding(rows(), vocab(), width()), ~Body())

# ---- the starting parameters: a fixed small pattern, not zeros ----

def value(+i: Nat, +j: Nat, +seed: Nat) -> F32:
  F32.sub(F32.mul(F32.from_nat(Nat.mod(Nat.add(Nat.add(Nat.mul(i, 3n), Nat.mul(j, 5n)), seed), 7n)),
    0.1), 0.2)

def row_vals(+c: Nat, +i: Nat, +j: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match c:
    case 0n:
      acc
    case 1n+rest:
      row_vals(rest, i, Nat.add(j, 1n), seed,
        List.append(&2, F32, acc, Con{value(i, j, seed), Nil{}}))

def rows_vals(+r: Nat, +c: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match r:
    case 0n:
      acc
    case 1n+rest:
      rows_vals(rest, c, Nat.add(i, 1n), seed, List.append(&2, F32, acc, row_vals(c, i, 0n, seed, Nil{})))

def values(r: Nat, c: Nat, seed: Nat) -> List<&2, F32>:
  rows_vals(r, c, 0n, seed, Nil{})

def emb_some(m: Maybe<&1, TA.Mat<8n, 16n>>) -> TA.Mat<8n, 16n>:
  match m:
    case Some{x}:
      x
    case None{}:
      TA.Mat.fill(8n, 16n, 0.0)

def emb_init(seed: Nat) -> TA.Mat<8n, 16n>:
  emb_some(TA.Mat.of_list(8n, 16n, values(8n, 16n, seed)))

def w_some(m: Maybe<&1, TA.Mat<16n, 16n>>) -> TA.Mat<16n, 16n>:
  match m:
    case Some{x}:
      x
    case None{}:
      TA.Mat.fill(16n, 16n, 0.0)

def w_init(seed: Nat) -> TA.Mat<16n, 16n>:
  w_some(TA.Mat.of_list(16n, 16n, values(16n, 16n, seed)))

def head_some(m: Maybe<&1, TA.Mat<8n, 16n>>) -> TA.Mat<8n, 16n>:
  match m:
    case Some{x}:
      x
    case None{}:
      TA.Mat.fill(8n, 16n, 0.0)

def head_init(seed: Nat) -> TA.Mat<8n, 16n>:
  head_some(TA.Mat.of_list(8n, 16n, values(8n, 16n, seed)))

def natt_params() -> NAttP:
  T.Two{TA.Mat.fill(1n, width(), 1.0),
    T.Two{T.Two{w_init(2n),
        T.Two{w_init(3n), w_init(4n)}},
      T.Two{Unit{}, w_init(5n)}}}

def head_params() -> HeadP:
  T.Two{TA.Mat.fill(1n, width(), 1.0), head_init(6n)}

def body_params() -> BodyP:
  T.Two{natt_params(), head_params()}

def params() -> P:
  T.Two{emb_init(1n), body_params()}

# ---- the data: the periodic sequence and its next token ----

def ids_go(+n: Nat, +i: Nat, +offset: Nat, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match n:
    case 0n:
      acc
    case 1n+rest:
      ids_go(rest, Nat.add(i, 1n), offset,
        List.append(&2, Nat, acc, Con{Nat.mod(Nat.add(i, offset), vocab()), Nil{}}))

def ids(n: Nat, offset: Nat) -> List<&2, Nat>:
  ids_go(n, 0n, offset, Nil{})

def xs() -> List<&2, Nat>:
  ids(rows(), 0n)

def ys() -> List<&2, Nat>:
  ids(rows(), 1n)

# ---- the step and the loop ----

def ce(z: Y, labels: List<&2, Nat>) -> T.LossOut<Y>:
  T.CrossEntropy(rows(), vocab(), z, labels)

def step(+cfg: T.OptCfg, tr: T.Train<P>) -> T.Train<P>:
  T.Train.step(~P, ~X, ~Y, ~S, ~List<&2, Nat>, ~Model, ~ce, cfg, tr,
    xs(), ys())

def loop(n: Nat, +i: Nat, +cfg: T.OptCfg, tr: T.Train<P>) -> IO(T.Train<P>):
  match n tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        IO.print(String.append("loss: ", F32.show(l)))
        return T.Train{p, g, st, p1, p2, l}
    case 1n+rest T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        IO.print(String.append("loss: ", F32.show(l)))
        loop(rest, Nat.add(i, 1n), cfg, step(cfg, T.Train{p, g, st, p1, p2, l}))

def cfg() -> T.OptCfg:
  T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

# ---- the accuracy of the trained model: the argmax of every row ----

def argmax_go(+xs: List<&2, F32>, +i: Nat, +best: Nat, +best_v: F32) -> Nat:
  match xs:
    case Nil{}:
      best
    case Con{+h, t}:
      Bool.pick(Nat, F32.is_gt(h, best_v),
        argmax_go(t, Nat.add(i, 1n), i, h),
        argmax_go(t, Nat.add(i, 1n), best, best_v))

def argmax_row(xs: List<&2, F32>) -> Nat:
  argmax_go(xs, 0n, 0n, (0.0 - 1.0e30 : F32))

def accuracy_go(labels: List<&2, Nat>, +data: List<&2, F32>, +right: Nat) -> Nat:
  match labels:
    case Nil{}:
      right
    case Con{y, ls}:
      +row = List.take(&2, F32, data, vocab())
      +rest = List.drop(&2, F32, data, vocab())
      accuracy_go(ls, rest,
        Bool.pick(Nat, Nat.is_eq(argmax_row(row), y), Nat.add(right, 1n), right))

def accuracy_list(m: TA.ML<32n, 8n>) -> F32:
  match m:
    case TA.ML{mat, l}:
      (F32.div(F32.from_nat(accuracy_go(ys(), l, 0n)), F32.from_nat(rows())) : F32)

def accuracy_mat(y: Y) -> F32:
  accuracy_list(TA.Mat.to_list(rows(), vocab(), y))

def accuracy_fwd(r: T.Fwd<P, Y, S>) -> F32:
  match r:
    case T.Fwd{p, y, s}:
      accuracy_mat(y)

def accuracy(tr: T.Train<P>) -> F32:
  match tr:
    case T.Train{p, g, st, p1, p2, l}:
      accuracy_fwd(T.Layer.fwd(P, X, Y, S, Model(), p, xs()))

def main() -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    tr : T.Train<P> <- loop(200n, 0n, cfg(),
      T.Train.new(~P, ~X, ~Y, ~S, ~Model, cfg(), params()))
    IO.print(String.append("accuracy: ", F32.show(accuracy(tr))))
