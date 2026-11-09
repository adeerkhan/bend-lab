# The Stage A layer stack at 128 positions, for corpus training.
#
# The trained model (`train_a.bend`) uses `model_a.bend` at 64 positions. A
# dev/confirm prompt is 76-95 tokens and the evaluator allows 48 new tokens,
# so generation needs 192 positions. The weights are position-independent:
# this file has the same layers, dimensions and parameter tree as
# `model_a.bend` (whose parameter types it imports), and loads the same files.
# Only RoPE's maximum position and the attention length change.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_a.bend as A

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def R(b: Nat) -> Nat:
  Nat.mul(b, 128n)

def A(b: Nat, c: Nat) -> Type:
  TA.Mat<R(b), c>

def EPS() -> F32:
  0.000001

# ---- parameters: the same tree as model_a.bend ----

def KVP() -> Type:
  T.Two<M(32n, 32n), M(32n, 32n)>

def QKP() -> Type:
  A.QKP()

def CoreP() -> Type:
  A.CoreP()

def GateP() -> Type:
  T.Two<M(128n, 32n), Unit>

def GUP() -> Type:
  A.GUP()

def GLUP() -> Type:
  T.Two<GUP(), Unit>

def MlpP() -> Type:
  A.MlpP()

def NMlpP() -> Type:
  A.NMlpP()

def AttP() -> Type:
  A.AttP()

def NAttP() -> Type:
  A.NAttP()

def BlockP() -> Type:
  A.BlockP()

def B2P() -> Type:
  A.B2P()

def BBP() -> Type:
  A.BBP()

def BodyP() -> Type:
  A.BodyP()

def P() -> Type:
  A.P()

# ---- state: rows = batch times 192 ----

def QKY(b: Nat) -> Type:
  T.Two<A(b, 32n), T.Two<A(b, 32n), A(b, 32n)>>

def KVY(b: Nat) -> Type:
  T.Two<A(b, 32n), A(b, 32n)>

def GateS(b: Nat) -> Type:
  T.Two<A(b, 32n), A(b, 128n)>

def GUY(b: Nat) -> Type:
  T.Two<A(b, 128n), A(b, 128n)>

def GUS(b: Nat) -> Type:
  T.Two<GateS(b), A(b, 32n)>

def GLUS(b: Nat) -> Type:
  T.Two<GUS(b), GUY(b)>

def MlpS(b: Nat) -> Type:
  T.Two<GLUS(b), A(b, 128n)>

def NMlpS(b: Nat) -> Type:
  T.Two<A(b, 32n), MlpS(b)>

def CoreS(b: Nat) -> Type:
  T.Two<T.AttnS, A(b, 32n)>

def RCS(b: Nat) -> Type:
  T.Two<T.Two<Unit, T.Two<Unit, Unit>>, CoreS(b)>

def AttS(b: Nat) -> Type:
  T.Two<QKY(b), RCS(b)>

def NAttS(b: Nat) -> Type:
  T.Two<A(b, 32n), AttS(b)>

def BlockS(b: Nat) -> Type:
  T.Two<T.ResidualType(NAttS(b)), T.ResidualType(NMlpS(b))>

def B2S(b: Nat) -> Type:
  T.Two<BlockS(b), BlockS(b)>

def BBS(b: Nat) -> Type:
  T.Two<B2S(b), B2S(b)>

def BodyS(b: Nat) -> Type:
  T.Two<BBS(b), A(b, 32n)>

def Y(b: Nat) -> Type:
  TA.Mat<R(b), 260n>

def X() -> Type:
  List<&2, Nat>

def S(b: Nat) -> Type:
  T.Three<List<&2, Nat>, BodyS(b), A(b, 32n)>

# ---- the layers ----

def KV(~b: Nat) -> T.Layer<KVP(), A(b, 32n), KVY(b), KVY(b)>:
  T.Fork(~M(32n, 32n), ~M(32n, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n),
    ~(m => T.Mat.dup(R(b), 32n, m)), ~(x => y => T.Mat.sum2(R(b), 32n, x, y)),
    ~T.LinearNB(R(b), 32n, 32n, 0n), ~T.LinearNB(R(b), 32n, 32n, 0n))

def QKV(~b: Nat) -> T.Layer<QKP(), A(b, 32n), QKY(b), QKY(b)>:
  T.Fork(~M(32n, 32n), ~T.Two<M(32n, 32n), M(32n, 32n)>, ~A(b, 32n), ~A(b, 32n), ~T.Two<A(b, 32n), A(b, 32n)>,
    ~A(b, 32n), ~T.Two<A(b, 32n), A(b, 32n)>,
    ~(m => T.Mat.dup(R(b), 32n, m)), ~(x => y => T.Mat.sum2(R(b), 32n, x, y)),
    ~T.LinearNB(R(b), 32n, 32n, 0n), ~KV(~b))

def RopeK(~b: Nat) -> T.Layer<T.Two<Unit, Unit>, T.Two<A(b, 32n), A(b, 32n)>, T.Two<A(b, 32n), A(b, 32n)>, T.Two<Unit, Unit>>:
  T.Both(~Unit, ~Unit, ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~Unit, ~Unit,
    ~T.RoPE(R(b), 32n, 128n, 4n, 4n, 10000.0, {==}), ~T.Id(~A(b, 32n)))

def Ropes(~b: Nat) -> T.Layer<T.Two<Unit, T.Two<Unit, Unit>>, QKY(b), QKY(b), T.Two<Unit, T.Two<Unit, Unit>>>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~A(b, 32n), ~T.Two<A(b, 32n), A(b, 32n)>,
    ~A(b, 32n), ~T.Two<A(b, 32n), A(b, 32n)>, ~Unit, ~T.Two<Unit, Unit>,
    ~T.RoPE(R(b), 32n, 128n, 4n, 4n, 10000.0, {==}), ~RopeK(~b))

def Core(~b: Nat) -> T.Layer<CoreP(), QKY(b), A(b, 32n), CoreS(b)>:
  T.Seq(~Unit, ~M(32n, 32n), ~QKY(b), ~A(b, 32n), ~A(b, 32n), ~T.AttnS, ~A(b, 32n),
    ~T.Attention(R(b), 32n, 32n, b, 128n, 4n, 1n, 8n, 0.35355339059327373, {==}, {==}, {==}),
    ~T.LinearNB(R(b), 32n, 32n, 0n))

def Att(~b: Nat) -> T.Layer<AttP(), A(b, 32n), A(b, 32n), AttS(b)>:
  T.Seq(~QKP(), ~A.RCP(), ~A(b, 32n), ~QKY(b), ~A(b, 32n), ~QKY(b), ~RCS(b),
    ~QKV(~b), ~T.Seq(~T.Two<Unit, T.Two<Unit, Unit>>, ~CoreP(), ~QKY(b), ~QKY(b), ~A(b, 32n),
      ~T.Two<Unit, T.Two<Unit, Unit>>, ~CoreS(b), ~Ropes(~b), ~Core(~b)))

def NAtt(~b: Nat) -> T.Layer<NAttP(), A(b, 32n), A(b, 32n), NAttS(b)>:
  T.Seq(~M(1n, 32n), ~AttP(), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~AttS(b),
    ~T.RMSNorm(R(b), 32n, EPS()), ~Att(~b))

def Gate(~b: Nat) -> T.Layer<GateP(), A(b, 32n), A(b, 128n), GateS(b)>:
  T.Seq(~M(128n, 32n), ~Unit, ~A(b, 32n), ~A(b, 128n), ~A(b, 128n), ~A(b, 32n), ~A(b, 128n),
    ~T.LinearNB(R(b), 32n, 128n, 0n), ~T.SiLU(R(b), 128n))

def GU(~b: Nat) -> T.Layer<GUP(), A(b, 32n), GUY(b), GUS(b)>:
  T.Fork(~T.Two<M(128n, 32n), Unit>, ~M(128n, 32n), ~A(b, 32n), ~A(b, 128n), ~A(b, 128n),
    ~GateS(b), ~A(b, 32n),
    ~(m => T.Mat.dup(R(b), 32n, m)), ~(x => y => T.Mat.sum2(R(b), 32n, x, y)),
    ~Gate(~b), ~T.LinearNB(R(b), 32n, 128n, 0n))

def GLU(~b: Nat) -> T.Layer<GLUP(), A(b, 32n), A(b, 128n), GLUS(b)>:
  T.Seq(~GUP(), ~Unit, ~A(b, 32n), ~GUY(b), ~A(b, 128n), ~GUS(b), ~GUY(b),
    ~GU(~b), ~T.Mul2(R(b), 128n))

def Mlp(~b: Nat) -> T.Layer<MlpP(), A(b, 32n), A(b, 32n), MlpS(b)>:
  T.Seq(~T.Two<GUP(), Unit>, ~M(32n, 128n), ~A(b, 32n), ~A(b, 128n), ~A(b, 32n),
    ~GLUS(b), ~A(b, 128n), ~GLU(~b), ~T.LinearNB(R(b), 128n, 32n, 0n))

def NMlp(~b: Nat) -> T.Layer<NMlpP(), A(b, 32n), A(b, 32n), NMlpS(b)>:
  T.Seq(~M(1n, 32n), ~MlpP(), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~MlpS(b),
    ~T.RMSNorm(R(b), 32n, EPS()), ~Mlp(~b))

def Block(~b: Nat) -> T.Layer<BlockP(), A(b, 32n), A(b, 32n), BlockS(b)>:
  T.Seq(~T.ResidualType(NAttP()), ~T.ResidualType(NMlpP()), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n),
    ~T.ResidualType(NAttS(b)), ~T.ResidualType(NMlpS(b)),
    ~T.Residual(~R(b), ~32n, ~NAttP(), ~NAttS(b), ~NAtt(~b)),
    ~T.Residual(~R(b), ~32n, ~NMlpP(), ~NMlpS(b), ~NMlp(~b)))

def B2(~b: Nat) -> T.Layer<B2P(), A(b, 32n), A(b, 32n), B2S(b)>:
  T.Seq(~BlockP(), ~BlockP(), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~BlockS(b), ~BlockS(b), ~Block(~b), ~Block(~b))

def Body(~b: Nat) -> T.Layer<BodyP(), A(b, 32n), A(b, 32n), BodyS(b)>:
  T.Seq(~BBP(), ~M(1n, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~BBS(b), ~A(b, 32n),
    ~T.Seq(~B2P(), ~B2P(), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~B2S(b), ~B2S(b), ~B2(~b), ~B2(~b)),
    ~T.RMSNorm(R(b), 32n, EPS()))

def Model(~b: Nat) -> T.Layer<P(), X(), Y(b), S(b)>:
  T.Tie(~M(260n, 32n), ~BodyP(), ~X(), ~A(b, 32n), ~A(b, 32n), ~Y(b),
    ~List<&2, Nat>, ~BodyS(b), ~A(b, 32n),
    ~T.Embedding(R(b), 260n, 32n), ~Body(~b), ~T.LinearNB(R(b), 32n, 260n, 0n))
