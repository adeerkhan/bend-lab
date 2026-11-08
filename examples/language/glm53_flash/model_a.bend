# Stage A: the dense-attention harness of the GLM-5.3-Flash-from-scratch port.
#
# Tiny config: vocab 260, width 32, 4 blocks, 4 heads of 8, 64 positions,
# SwiGLU hidden 128, tied embedding/head. It is built from BendTorch's existing
# layers, so this stage tests the harness itself (Embedding, RMSNorm, RoPE,
# causal Attention, SwiGLU, Tie, CrossEntropy) before any custom attention.
#
# The q/k weights are stored permuted by `P` in the loader (see the twin), so
# BendTorch's half-split RoPE is the reference's interleaved RoPE.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def R(b: Nat) -> Nat:
  Nat.mul(b, 64n)

def A(b: Nat, c: Nat) -> Type:
  TA.Mat<R(b), c>

def EPS() -> F32:
  0.000001

# ---- q, k, v projections: one fork gives the three matrices ----

def QKP() -> Type:
  T.Two<M(32n, 32n), T.Two<M(32n, 32n), M(32n, 32n)>>

def QKY(b: Nat) -> Type:
  T.Two<A(b, 32n), T.Two<A(b, 32n), A(b, 32n)>>

def KV(~b: Nat) -> T.Layer<T.Two<M(32n, 32n), M(32n, 32n)>, A(b, 32n), T.Two<A(b, 32n), A(b, 32n)>, T.Two<A(b, 32n), A(b, 32n)>>:
  T.Fork(~M(32n, 32n), ~M(32n, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n),
    ~(m => T.Mat.dup(R(b), 32n, m)), ~(x => y => T.Mat.sum2(R(b), 32n, x, y)),
    ~T.LinearNB(R(b), 32n, 32n, 0n), ~T.LinearNB(R(b), 32n, 32n, 0n))

def QKV(~b: Nat) -> T.Layer<QKP, A(b, 32n), QKY(b), QKY(b)>:
  T.Fork(~M(32n, 32n), ~T.Two<M(32n, 32n), M(32n, 32n)>, ~A(b, 32n), ~A(b, 32n), ~T.Two<A(b, 32n), A(b, 32n)>,
    ~A(b, 32n), ~T.Two<A(b, 32n), A(b, 32n)>,
    ~(m => T.Mat.dup(R(b), 32n, m)), ~(x => y => T.Mat.sum2(R(b), 32n, x, y)),
    ~T.LinearNB(R(b), 32n, 32n, 0n), ~KV(~b))

# ---- RoPE on q and k, identity on v ----

def RopeK(~b: Nat) -> T.Layer<T.Two<Unit, Unit>, T.Two<A(b, 32n), A(b, 32n)>, T.Two<A(b, 32n), A(b, 32n)>, T.Two<Unit, Unit>>:
  T.Both(~Unit, ~Unit, ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~Unit, ~Unit,
    ~T.RoPE(R(b), 32n, 64n, 4n, 4n, 10000.0, {==}), ~T.Id(~A(b, 32n)))

def Ropes(~b: Nat) -> T.Layer<T.Two<Unit, T.Two<Unit, Unit>>, QKY(b), QKY(b), T.Two<Unit, T.Two<Unit, Unit>>>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~A(b, 32n), ~T.Two<A(b, 32n), A(b, 32n)>,
    ~A(b, 32n), ~T.Two<A(b, 32n), A(b, 32n)>, ~Unit, ~T.Two<Unit, Unit>,
    ~T.RoPE(R(b), 32n, 64n, 4n, 4n, 10000.0, {==}), ~RopeK(~b))

# ---- the causal attention core and its output projection ----

def CoreP() -> Type:
  T.Two<Unit, M(32n, 32n)>

def CoreS(b: Nat) -> Type:
  T.Two<T.AttnS, A(b, 32n)>

def Core(~b: Nat) -> T.Layer<CoreP, QKY(b), A(b, 32n), CoreS(b)>:
  T.Seq(~Unit, ~M(32n, 32n), ~QKY(b), ~A(b, 32n), ~A(b, 32n), ~T.AttnS, ~A(b, 32n),
    ~T.Attention(R(b), 32n, 32n, b, 64n, 4n, 1n, 8n, 0.35355339059327373, {==}, {==}, {==}),
    ~T.LinearNB(R(b), 32n, 32n, 0n))

def RCP() -> Type:
  T.Two<T.Two<Unit, T.Two<Unit, Unit>>, CoreP>

def RCS(b: Nat) -> Type:
  T.Two<T.Two<Unit, T.Two<Unit, Unit>>, CoreS(b)>

def AttP() -> Type:
  T.Two<QKP, RCP>

def AttS(b: Nat) -> Type:
  T.Two<QKY(b), RCS(b)>

def Att(~b: Nat) -> T.Layer<AttP, A(b, 32n), A(b, 32n), AttS(b)>:
  T.Seq(~QKP, ~RCP, ~A(b, 32n), ~QKY(b), ~A(b, 32n), ~QKY(b), ~RCS(b),
    ~QKV(~b), ~T.Seq(~T.Two<Unit, T.Two<Unit, Unit>>, ~CoreP, ~QKY(b), ~QKY(b), ~A(b, 32n),
      ~T.Two<Unit, T.Two<Unit, Unit>>, ~CoreS(b), ~Ropes(~b), ~Core(~b)))

def NAttP() -> Type:
  T.Two<M(1n, 32n), AttP>

def NAttS(b: Nat) -> Type:
  T.Two<A(b, 32n), AttS(b)>

def NAtt(~b: Nat) -> T.Layer<NAttP, A(b, 32n), A(b, 32n), NAttS(b)>:
  T.Seq(~M(1n, 32n), ~AttP, ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~AttS(b),
    ~T.RMSNorm(R(b), 32n, EPS()), ~Att(~b))

# ---- SwiGLU: down(silu(gate(x)) * up(x)), hidden 128 ----

def GateS(b: Nat) -> Type:
  T.Two<A(b, 32n), A(b, 128n)>

def Gate(~b: Nat) -> T.Layer<T.Two<M(128n, 32n), Unit>, A(b, 32n), A(b, 128n), GateS(b)>:
  T.Seq(~M(128n, 32n), ~Unit, ~A(b, 32n), ~A(b, 128n), ~A(b, 128n), ~A(b, 32n), ~A(b, 128n),
    ~T.LinearNB(R(b), 32n, 128n, 0n), ~T.SiLU(R(b), 128n))

def GUP() -> Type:
  T.Two<T.Two<M(128n, 32n), Unit>, M(128n, 32n)>

def GUY(b: Nat) -> Type:
  T.Two<A(b, 128n), A(b, 128n)>

def GUS(b: Nat) -> Type:
  T.Two<GateS(b), A(b, 32n)>

def GU(~b: Nat) -> T.Layer<GUP, A(b, 32n), GUY(b), GUS(b)>:
  T.Fork(~T.Two<M(128n, 32n), Unit>, ~M(128n, 32n), ~A(b, 32n), ~A(b, 128n), ~A(b, 128n),
    ~GateS(b), ~A(b, 32n),
    ~(m => T.Mat.dup(R(b), 32n, m)), ~(x => y => T.Mat.sum2(R(b), 32n, x, y)),
    ~Gate(~b), ~T.LinearNB(R(b), 32n, 128n, 0n))

def GLUS(b: Nat) -> Type:
  T.Two<GUS(b), GUY(b)>

def GLU(~b: Nat) -> T.Layer<T.Two<GUP, Unit>, A(b, 32n), A(b, 128n), GLUS(b)>:
  T.Seq(~GUP, ~Unit, ~A(b, 32n), ~GUY(b), ~A(b, 128n), ~GUS(b), ~GUY(b),
    ~GU(~b), ~T.Mul2(R(b), 128n))

def MlpP() -> Type:
  T.Pair3(GUP, Unit, M(32n, 128n))

def MlpS(b: Nat) -> Type:
  T.Two<GLUS(b), A(b, 128n)>

def Mlp(~b: Nat) -> T.Layer<MlpP, A(b, 32n), A(b, 32n), MlpS(b)>:
  T.Seq(~T.Two<GUP, Unit>, ~M(32n, 128n), ~A(b, 32n), ~A(b, 128n), ~A(b, 32n),
    ~GLUS(b), ~A(b, 128n), ~GLU(~b), ~T.LinearNB(R(b), 128n, 32n, 0n))

def NMlpP() -> Type:
  T.Two<M(1n, 32n), MlpP>

def NMlpS(b: Nat) -> Type:
  T.Two<A(b, 32n), MlpS(b)>

def NMlp(~b: Nat) -> T.Layer<NMlpP, A(b, 32n), A(b, 32n), NMlpS(b)>:
  T.Seq(~M(1n, 32n), ~MlpP, ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~MlpS(b),
    ~T.RMSNorm(R(b), 32n, EPS()), ~Mlp(~b))

# ---- one block, then four blocks, then the tied model ----

def BlockP() -> Type:
  T.Two<T.ResidualType(NAttP), T.ResidualType(NMlpP)>

def BlockS(b: Nat) -> Type:
  T.Two<T.ResidualType(NAttS(b)), T.ResidualType(NMlpS(b))>

def Block(~b: Nat) -> T.Layer<BlockP, A(b, 32n), A(b, 32n), BlockS(b)>:
  T.Seq(~T.ResidualType(NAttP), ~T.ResidualType(NMlpP), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n),
    ~T.ResidualType(NAttS(b)), ~T.ResidualType(NMlpS(b)),
    ~T.Residual(~R(b), ~32n, ~NAttP, ~NAttS(b), ~NAtt(~b)),
    ~T.Residual(~R(b), ~32n, ~NMlpP, ~NMlpS(b), ~NMlp(~b)))

def B2P() -> Type:
  T.Two<BlockP, BlockP>

def B2S(b: Nat) -> Type:
  T.Two<BlockS(b), BlockS(b)>

def B2(~b: Nat) -> T.Layer<B2P, A(b, 32n), A(b, 32n), B2S(b)>:
  T.Seq(~BlockP, ~BlockP, ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~BlockS(b), ~BlockS(b), ~Block(~b), ~Block(~b))

def BBP() -> Type:
  T.Two<B2P, B2P>

def BBS(b: Nat) -> Type:
  T.Two<B2S(b), B2S(b)>

def BodyP() -> Type:
  T.Two<BBP, M(1n, 32n)>

def BodyS(b: Nat) -> Type:
  T.Two<BBS(b), A(b, 32n)>

def Body(~b: Nat) -> T.Layer<BodyP, A(b, 32n), A(b, 32n), BodyS(b)>:
  T.Seq(~BBP, ~M(1n, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~BBS(b), ~A(b, 32n),
    ~T.Seq(~B2P, ~B2P, ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~B2S(b), ~B2S(b), ~B2(~b), ~B2(~b)),
    ~T.RMSNorm(R(b), 32n, EPS()))

def P() -> Type:
  T.Two<M(260n, 32n), BodyP>

def X() -> Type:
  List<&2, Nat>

def Y(b: Nat) -> Type:
  TA.Mat<R(b), 260n>

def S(b: Nat) -> Type:
  T.Three<List<&2, Nat>, BodyS(b), A(b, 32n)>

def Model(~b: Nat) -> T.Layer<P, X, Y(b), S(b)>:
  T.Tie(~M(260n, 32n), ~BodyP, ~X, ~A(b, 32n), ~A(b, 32n), ~Y(b),
    ~List<&2, Nat>, ~BodyS(b), ~A(b, 32n),
    ~T.Embedding(R(b), 260n, 32n), ~Body(~b), ~T.LinearNB(R(b), 32n, 260n, 0n))

def ce(z: Y(2n), labels: List<&2, Nat>) -> T.LossOut<Y(2n)>:
  T.CrossEntropy(R(2n), 260n, z, labels)
