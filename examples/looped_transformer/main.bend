# A looped transformer on the chain puzzle, after burn-lab/looped.
#
#   python3 examples/looped_transformer/reference/make_data.py --out examples/looped_transformer/data
#   bend examples/looped_transformer/main.bend -o .scratch/looped
#   .scratch/looped train 500 1 examples/looped_transformer/data/train.bin
#   .scratch/looped train 500 4 examples/looped_transformer/data/train.bin
#   .scratch/looped eval 4 examples/looped_transformer/data/eval-hop-1.bin
#
# Four stored blocks: one prelude, two middle blocks reused `passes` times, one
# coda. The first pass starts from the prelude; each later pass starts from
# `start + RMSNorm(previous middle output)`. This is the article's residual
# message without its learned per-channel gate, and without the four-token
# depthwise K/V convolution yet. Both are banners in result tables.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./short_conv.bend as SC

def D() -> Nat:
  32n

def H() -> Nat:
  4n

def HD() -> Nat:
  8n

def FF() -> Nat:
  64n

def V() -> Nat:
  39n

def SL() -> Nat:
  66n

def B() -> Nat:
  8n

def R() -> Nat:
  Nat.mul(B(), SL())

def EH() -> Nat:
  12n

def EP() -> Nat:
  128n

def EC() -> Nat:
  Nat.div(EP(), B())

def ES() -> Nat:
  Nat.mul(EH(), EC())

def EPS() -> F32:
  0.00001

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def A(c: Nat) -> Type:
  TA.Mat<R(), c>

# RMSNorm with the library's zero initializer replaced by a gain of one. The
# forward, backward and parameter round trip are the published layer's.
def RMSNormOne(+n: Nat, +d: Nat, +eps: F32) ->
  T.Layer<M(1n, d), TA.Mat<n, d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  T.Layer{
    p => x => T.Layer.fwd(M(1n, d), TA.Mat<n, d>, TA.Mat<n, d>, TA.Mat<n, d>,
      T.RMSNorm(n, d, eps), p, x),
    p => s => dy => g => T.Layer.bwd(M(1n, d), TA.Mat<n, d>, TA.Mat<n, d>, TA.Mat<n, d>,
      T.RMSNorm(n, d, eps), p, s, dy, g),
    op => p => g => st => T.Layer.step(M(1n, d), TA.Mat<n, d>, TA.Mat<n, d>, TA.Mat<n, d>,
      T.RMSNorm(n, d, eps), op, p, g, st),
    u => TA.Mat.fill(1n, d, 1.0),
    p => T.Mat.dump(1n, d, p),
    p => xs => off => T.Mat.restore(1n, d, p, xs, off),
    p => xs => off => T.Mat.store(1n, d, p, xs, off),
    u => U32.from_nat(d)
  }

# ---- one transformer block, built from BendTorch's layers ----

def KVY() -> Type:
  T.Two<A(D()), A(D())>

def KVPairP() -> Type:
  T.Two<M(D(), D()), M(D(), D())>

def KVPairS() -> Type:
  T.Two<A(D()), A(D())>

def KVPair() -> T.Layer<KVPairP, A(D()), KVY(), KVPairS()>:
  T.Fork(~M(D(), D()), ~M(D(), D()), ~A(D()), ~A(D()), ~A(D()),
    ~A(D()), ~A(D()),
    ~(m => T.Mat.dup(R(), D(), m)),
    ~(x => y => T.Mat.sum2(R(), D(), x, y)),
    ~T.LinearNB(R(), D(), D(), 0n),
    ~T.LinearNB(R(), D(), D(), 0n))

def KVConvP() -> Type:
  T.Two<SC.ConvP32(), SC.ConvP32()>

def KVConvS() -> Type:
  T.Two<A(D()), A(D())>

def KVConv() -> T.Layer<KVConvP, KVY(), KVY(), KVConvS()>:
  T.Both(~SC.ConvP32(), ~SC.ConvP32(), ~A(D()), ~A(D()), ~A(D()), ~A(D()),
    ~A(D()), ~A(D()),
    ~SC.ShortConv32(B(), SL()), ~SC.ShortConv32(B(), SL()))

def KVP() -> Type:
  T.Two<KVPairP, KVConvP>

def KVS() -> Type:
  T.Two<KVPairS, KVConvS>

def KVL() -> T.Layer<KVP, A(D()), KVY(), KVS>:
  T.Seq(~KVPairP, ~KVConvP, ~A(D()), ~KVY(), ~KVY(), ~KVPairS, ~KVConvS,
    ~KVPair(), ~KVConv())

def AX() -> Type:
  T.Two<A(D()), KVY()>

def QKVP() -> Type:
  T.Two<M(D(), D()), KVP>

def QKVS() -> Type:
  T.Two<A(D()), KVS()>

def QKV() -> T.Layer<QKVP, A(D()), AX(), QKVS()>:
  T.Fork(~M(D(), D()), ~KVP, ~A(D()), ~A(D()), ~KVY(), ~A(D()),
    ~KVS(), ~(m => T.Mat.dup(R(), D(), m)),
    ~(x => y => T.Mat.sum2(R(), D(), x, y)),
    ~T.LinearNB(R(), D(), D(), 0n), ~KVL())

def RopeK() -> T.Layer<T.Two<Unit, Unit>, KVY(), KVY(), T.Two<Unit, Unit>>:
  T.Both(~Unit, ~Unit, ~A(D()), ~A(D()), ~A(D()), ~A(D()), ~Unit, ~Unit,
    ~T.RoPE(R(), D(), SL(), H(), 4n, 10000.0, {==}),
    ~T.Id(~A(D())))

def RU() -> Type:
  T.Two<Unit, T.Two<Unit, Unit>>

def Ropes() -> T.Layer<RU, AX(), AX(), RU>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~A(D()), ~KVY(), ~A(D()), ~KVY(),
    ~Unit, ~T.Two<Unit, Unit>,
    ~T.RoPE(R(), D(), SL(), H(), 4n, 10000.0, {==}), ~RopeK())

def CoreP() -> Type:
  T.Two<Unit, M(D(), D())>

def CoreS() -> Type:
  T.Two<T.AttnS, A(D())>

def Core() -> T.Layer<CoreP, AX(), A(D()), CoreS()>:
  T.Seq(~Unit, ~M(D(), D()), ~AX(), ~A(D()), ~A(D()), ~T.AttnS,
    ~A(D()), ~T.Attention(R(), D(), D(), B(), SL(), H(), 1n, HD(),
      0.35355339059327373, {==}, {==}, {==}),
    ~T.LinearNB(R(), D(), D(), 0n))

def RCP() -> Type:
  T.Two<RU, CoreP>

def RCS() -> Type:
  T.Two<RU, CoreS()>

def RopeCore() -> T.Layer<RCP, AX(), A(D()), RCS()>:
  T.Seq(~RU, ~CoreP, ~AX(), ~AX(), ~A(D()), ~RU, ~CoreS(), ~Ropes(), ~Core())

def AttP() -> Type:
  T.Two<QKVP, RCP>

def AttS() -> Type:
  T.Two<QKVS, RCS()>

def Att() -> T.Layer<AttP, A(D()), A(D()), AttS()>:
  T.Seq(~QKVP, ~RCP, ~A(D()), ~AX(), ~A(D()), ~QKVS, ~RCS(),
    ~QKV(), ~RopeCore())

def NAttP() -> Type:
  T.Two<M(1n, D()), AttP>

def NAttS() -> Type:
  T.Two<A(D()), AttS()>

def NAtt() -> T.Layer<NAttP, A(D()), A(D()), NAttS()>:
  T.Seq(~M(1n, D()), ~AttP, ~A(D()), ~A(D()), ~A(D()), ~A(D()), ~AttS(),
    ~RMSNormOne(R(), D(), EPS()), ~Att())

def GateS() -> Type:
  T.Two<A(D()), A(FF())>

def Gate() -> T.Layer<T.Two<M(FF(), D()), Unit>, A(D()), A(FF()), GateS()>:
  T.Seq(~M(FF(), D()), ~Unit, ~A(D()), ~A(FF()), ~A(FF()), ~A(D()), ~A(FF()),
    ~T.LinearNB(R(), D(), FF(), 0n), ~T.SiLU(R(), FF()))

def GUP() -> Type:
  T.Two<T.Two<M(FF(), D()), Unit>, M(FF(), D())>

def GUY() -> Type:
  T.Two<A(FF()), A(FF())>

def GUS() -> Type:
  T.Two<GateS(), A(D())>

def GU() -> T.Layer<GUP, A(D()), GUY(), GUS()>:
  T.Fork(~T.Two<M(FF(), D()), Unit>, ~M(FF(), D()), ~A(D()), ~A(FF()),
    ~A(FF()), ~GateS(), ~A(D()),
    ~(m => T.Mat.dup(R(), D(), m)),
    ~(x => y => T.Mat.sum2(R(), D(), x, y)),
    ~Gate(), ~T.LinearNB(R(), D(), FF(), 0n))

def GLUS() -> Type:
  T.Two<GUS(), GUY()>

def GLU() -> T.Layer<T.Two<GUP, Unit>, A(D()), A(FF()), GLUS()>:
  T.Seq(~GUP, ~Unit, ~A(D()), ~GUY(), ~A(FF()), ~GUS(), ~GUY(),
    ~GU(), ~T.Mul2(R(), FF()))

def MlpP() -> Type:
  T.Pair3(GUP, Unit, M(D(), FF()))

def MlpS() -> Type:
  T.Two<GLUS(), A(FF())>

def Mlp() -> T.Layer<MlpP, A(D()), A(D()), MlpS()>:
  T.Seq(~T.Two<GUP, Unit>, ~M(D(), FF()), ~A(D()), ~A(FF()), ~A(D()),
    ~GLUS(), ~A(FF()), ~GLU(), ~T.LinearNB(R(), FF(), D(), 0n))

def NMlpP() -> Type:
  T.Two<M(1n, D()), MlpP>

def NMlpS() -> Type:
  T.Two<A(D()), MlpS()>

def NMlp() -> T.Layer<NMlpP, A(D()), A(D()), NMlpS()>:
  T.Seq(~M(1n, D()), ~MlpP, ~A(D()), ~A(D()), ~A(D()), ~A(D()), ~MlpS(),
    ~RMSNormOne(R(), D(), EPS()), ~Mlp())

def BlockP() -> Type:
  T.Two<T.ResidualType(NAttP), T.ResidualType(NMlpP)>

def BlockS() -> Type:
  T.Two<T.ResidualType(NAttS), T.ResidualType(NMlpS)>

def Block() -> T.Layer<BlockP, A(D()), A(D()), BlockS()>:
  T.Seq(~T.ResidualType(NAttP), ~T.ResidualType(NMlpP), ~A(D()), ~A(D()),
    ~A(D()), ~T.ResidualType(NAttS), ~T.ResidualType(NMlpS),
    ~T.Residual(~R(), ~D(), ~NAttP, ~NAttS(), ~NAtt()),
    ~T.Residual(~R(), ~D(), ~NMlpP, ~NMlpS(), ~NMlp()))

# ---- the repeated middle ----

def rep_bind(-P: Type, -X: Type, -S: Type, r: T.Fwd<P, X, S>,
  k: P -> X -> S -> T.Fwd<P, X, List<&1, S>>) ->
  T.Fwd<P, X, List<&1, S>>:
  match r:
    case T.Fwd{p, y, s}:
      k(p, y, s)

def rep_fwd(~P: Type, ~X: Type, ~S: Type, ~l: T.Layer<P, X, X, S>,
  +k: Nat, p: P, x: X, acc: List<&1, S>) -> T.Fwd<P, X, List<&1, S>>:
  match k:
    case 0n:
      T.Fwd{p, x, List.reverse(&1, S, acc)}
    case 1n+q:
      rep_bind(P, X, S, T.Layer.fwd(P, X, X, S, l, p, x),
        p2 => y => s => rep_fwd(~P, ~X, ~S, ~l, q, p2, y, Con{s, acc}))

def rep_bind_b(-P: Type, -X: Type, r: T.Bwd<P, X>,
  k: P -> X -> P -> T.Bwd<P, X>) -> T.Bwd<P, X>:
  match r:
    case T.Bwd{p, dx, g}:
      k(p, dx, g)

def rep_bwd(~P: Type, ~X: Type, ~S: Type, ~l: T.Layer<P, X, X, S>,
  +k: Nat, p: P, ss: List<&1, S>, dy: X, g: P) -> T.Bwd<P, X>:
  match k:
    case 0n:
      T.Bwd{p, dy, g}
    case 1n+q:
      match ss:
        case Nil{}:
          T.Bwd{p, dy, g}
        case Con{s, st}:
          rep_bind_b(P, X, T.Layer.bwd(P, X, X, S, l, p, s, dy, g),
            p2 => dx => g2 => rep_bwd(~P, ~X, ~S, ~l, q, p2, st, dx, g2))

def Repeat(~P: Type, ~X: Type, ~S: Type, ~l: T.Layer<P, X, X, S>, +k: Nat) ->
  T.Layer<P, X, X, List<&1, S>>:
  T.Layer{
    p => x => rep_fwd(~P, ~X, ~S, ~l, k, p, x, Nil{}),
    p => ss => dy => g => rep_bwd(~P, ~X, ~S, ~l, k, p, ss, dy, g),
    op => p => g => st => T.Layer.step(P, X, X, S, l, op, p, g, st),
    u => T.Layer.zeros(P, X, X, S, l),
    p => T.Layer.dump(P, X, X, S, l, p),
    p => xs => off => T.Layer.load(P, X, X, S, l, p, xs, off),
    p => xs => off => T.Layer.store(P, X, X, S, l, p, xs, off),
    u => T.Layer.count(P, X, X, S, l)
  }

def LoopP() -> Type:
  T.Two<BlockP, BlockP>

def LoopS() -> Type:
  T.Two<BlockS, BlockS>

def Loop2() -> T.Layer<LoopP, A(D()), A(D()), LoopS()>:
  T.Seq(~BlockP, ~BlockP, ~A(D()), ~A(D()), ~A(D()), ~BlockS(), ~BlockS(),
    ~Block(), ~Block())

def PassP() -> Type:
  T.Two<LoopP, M(1n, D())>

def PassS() -> Type:
  T.Two<LoopS, A(D())>

def then_fwd(-P: Type, -Y: Type, -S: Type, -R: Type, r: T.Fwd<P, Y, S>,
  k: P -> Y -> S -> R) -> R:
  match r:
    case T.Fwd{p, y, s}:
      k(p, y, s)

def then_bwd(-P: Type, -X: Type, -R: Type, r: T.Bwd<P, X>,
  k: P -> X -> P -> R) -> R:
  match r:
    case T.Bwd{p, x, g}:
      k(p, x, g)

def split_pair(-R: Type, d: T.Two<A(D()), A(D())>, k: A(D()) -> A(D()) -> R) -> R:
  match d:
    case T.Two{a, b}:
      k(a, b)

def pass_fwd(p: PassP(), x: T.Two<A(D()), A(D())>) ->
  T.Fwd<PassP(), T.Two<A(D()), A(D())>, PassS()>:
  match p x:
    case T.Two{lp, np} T.Two{start, msg}:
      split_pair(T.Fwd<PassP(), T.Two<A(D()), A(D())>, PassS()>,
        T.Mat.dup(R(), D(), start),
        s1 => s2 =>
          then_fwd(M(1n, D()), A(D()), A(D()),
            T.Fwd<PassP(), T.Two<A(D()), A(D())>, PassS()>,
            T.Layer.fwd(M(1n, D()), A(D()), A(D()), A(D()),
              RMSNormOne(R(), D(), EPS()), np, msg),
            np2 => nmsg => ns =>
              then_fwd(Unit, A(D()), Unit,
                T.Fwd<PassP(), T.Two<A(D()), A(D())>, PassS()>,
                T.Layer.fwd(Unit, T.Two<A(D()), A(D())>, A(D()), Unit,
                  T.Add2(R(), D()), Unit{}, T.Two{s1, nmsg}),
                u => z => us =>
                  then_fwd(LoopP(), A(D()), LoopS(),
                    T.Fwd<PassP(), T.Two<A(D()), A(D())>, PassS()>,
                    T.Layer.fwd(LoopP(), A(D()), A(D()), LoopS(), Loop2(), lp, z),
                    lp2 => y => ls =>
                      T.Fwd{T.Two{lp2, np2}, T.Two{s2, y}, T.Two{ls, ns}}))))

def pass_bwd(p: PassP(), s: PassS(), dy: T.Two<A(D()), A(D())>, g: PassP()) ->
  T.Bwd<PassP(), T.Two<A(D()), A(D())>>:
  match p s dy g:
    case T.Two{lp, np} T.Two{ls, ns} T.Two{dstart, dyout} T.Two{gl, gn}:
      then_bwd(LoopP(), A(D()), T.Bwd<PassP(), T.Two<A(D()), A(D())>>,
        T.Layer.bwd(LoopP(), A(D()), A(D()), LoopS(), Loop2(),
          lp, ls, dyout, gl),
        lp2 => dz => gl2 =>
          then_bwd(Unit, T.Two<A(D()), A(D())>, T.Bwd<PassP(), T.Two<A(D()), A(D())>>,
            T.Layer.bwd(Unit, T.Two<A(D()), A(D())>, A(D()), Unit,
              T.Add2(R(), D()), Unit{}, Unit{}, dz, Unit{}),
            u => dadd => u2 =>
              split_pair(T.Bwd<PassP(), T.Two<A(D()), A(D())>>, dadd,
                dz1 => dz2 =>
                  then_bwd(M(1n, D()), A(D()), T.Bwd<PassP(), T.Two<A(D()), A(D())>>,
                    T.Layer.bwd(M(1n, D()), A(D()), A(D()), A(D()),
                      RMSNormOne(R(), D(), EPS()), np, ns, dz2, gn),
                    np2 => dmsg => gn2 =>
                      T.Bwd{T.Two{lp2, np2},
                        T.Two{T.Mat.sum2(R(), D(), dstart, dz1), dmsg},
                        T.Two{gl2, gn2}}))))

def Pass() -> T.Layer<PassP, T.Two<A(D()), A(D())>, T.Two<A(D()), A(D())>, PassS()>:
  T.Layer{
    p => x => pass_fwd(p, x),
    p => s => dy => g => pass_bwd(p, s, dy, g),
    op => p => g => st => T.pair_step(~LoopP(), ~A(D()), ~A(D()), ~LoopS(),
      ~M(1n, D()), ~A(D()), ~A(D()), ~A(D()), ~Loop2(), ~RMSNormOne(R(), D(), EPS()),
      op, p, g, st),
    u => T.pair_zeros(~LoopP(), ~A(D()), ~A(D()), ~LoopS(), ~M(1n, D()),
      ~A(D()), ~A(D()), ~A(D()), ~Loop2(), ~RMSNormOne(R(), D(), EPS())),
    p => T.pair_dump(~LoopP(), ~A(D()), ~A(D()), ~LoopS(), ~M(1n, D()),
      ~A(D()), ~A(D()), ~A(D()), ~Loop2(), ~RMSNormOne(R(), D(), EPS()), p),
    p => xs => off => T.pair_load(~LoopP(), ~A(D()), ~A(D()), ~LoopS(),
      ~M(1n, D()), ~A(D()), ~A(D()), ~A(D()), ~Loop2(),
      ~RMSNormOne(R(), D(), EPS()), p, xs, off),
    p => xs => off => T.pair_store(~LoopP(), ~A(D()), ~A(D()), ~LoopS(),
      ~M(1n, D()), ~A(D()), ~A(D()), ~A(D()), ~Loop2(),
      ~RMSNormOne(R(), D(), EPS()), p, xs, off),
    u => (T.Layer.count(LoopP(), A(D()), A(D()), LoopS(), Loop2()) +
          T.Layer.count(M(1n, D()), A(D()), A(D()), A(D()),
            RMSNormOne(R(), D(), EPS())) : U32)
  }

# ---- the model ----

def DupZeroP() -> Type:
  Unit

def dupzero_bwd(dy: T.Two<A(D()), A(D())>) -> T.Bwd<Unit, A(D())>:
  match dy:
    case T.Two{d1, d2}:
      T.Bwd{Unit{}, d1, Unit{}}

def DupZero() -> T.Layer<Unit, A(D()), T.Two<A(D()), A(D())>, Unit>:
  T.Layer{
    p => x => T.Fwd{Unit{}, T.Two{x, TA.Mat.zeros(R(), D())}, Unit{}},
    p => s => dy => g => dupzero_bwd(dy),
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0
  }

def snd_fwd(x: T.Two<A(D()), A(D())>) -> T.Fwd<Unit, A(D()), Unit>:
  match x:
    case T.Two{start, msg}:
      T.Fwd{Unit{}, msg, Unit{}}

def Snd() -> T.Layer<Unit, T.Two<A(D()), A(D())>, A(D()), Unit>:
  T.Layer{
    p => x => snd_fwd(x),
    p => s => dy => g => T.Bwd{Unit{}, T.Two{TA.Mat.zeros(R(), D()), dy}, Unit{}},
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0
  }

def LoopLayerP() -> Type:
  T.Two<T.Two<Unit, PassP>, Unit>

def LoopLayerS() -> Type:
  T.Two<T.Two<Unit, List<&1, PassS>>, Unit>

def LoopLayer(+passes: Nat) ->
  T.Layer<LoopLayerP(), A(D()), A(D()), LoopLayerS()>:
  T.Seq.at(T.Two<Unit, PassP>, Unit, A(D()), T.Two<A(D()), A(D())>, A(D()),
    T.Two<Unit, List<&1, PassS>>, Unit,
    T.Seq.at(Unit, PassP, A(D()), T.Two<A(D()), A(D())>, T.Two<A(D()), A(D())>,
      Unit, List<&1, PassS>, DupZero(),
      Repeat(~PassP(), ~T.Two<A(D()), A(D())>, ~PassS(), ~Pass(), passes)),
    Snd())

def EmbP() -> Type:
  M(V(), D())

def EmbS() -> Type:
  List<&2, Nat>

def HeadP() -> Type:
  T.Two<M(1n, D()), M(V(), D())>

def HeadS() -> Type:
  T.Two<A(D()), A(D())>

def Head() -> T.Layer<HeadP, A(D()), M(R(), V()), HeadS()>:
  T.Seq(~M(1n, D()), ~M(V(), D()), ~A(D()), ~A(D()), ~M(R(), V()),
    ~A(D()), ~A(D()), ~RMSNormOne(R(), D(), EPS()),
    ~T.LinearNB(R(), D(), V(), 0n))

def BodyP() -> Type:
  T.Two<T.Two<BlockP, LoopLayerP()>, BlockP>

def BodyS() -> Type:
  T.Two<T.Two<BlockS, LoopLayerS()>, BlockS>

def Body(+passes: Nat) -> T.Layer<BodyP(), A(D()), A(D()), BodyS()>:
  T.Seq.at(T.Two<BlockP, LoopLayerP()>, BlockP, A(D()), A(D()), A(D()),
    T.Two<BlockS, LoopLayerS()>, BlockS(),
    T.Seq.at(BlockP, LoopLayerP(), A(D()), A(D()), A(D()), BlockS(),
      LoopLayerS(), Block(), LoopLayer(passes)),
    Block())

def ModelP() -> Type:
  T.Two<T.Two<EmbP, BodyP()>, HeadP>

def ModelS() -> Type:
  T.Two<T.Two<EmbS, BodyS()>, HeadS()>

def X() -> Type:
  List<&2, Nat>

def Y() -> Type:
  M(R(), V())

def Model(+passes: Nat) -> T.Layer<ModelP(), X(), Y(), ModelS()>:
  T.Seq.at(T.Two<EmbP, BodyP()>, HeadP, X(), A(D()), Y(),
    T.Two<EmbS, BodyS()>, HeadS(),
    T.Seq.at(EmbP(), BodyP(), X(), A(D()), A(D()), EmbS(), BodyS(),
      T.Embedding(R(), V(), D()), Body(passes)),
    Head())

def rand_f32(+i: U32, +seed: U32) -> F32:
  (U32.to_f32(T.mix32(U32.xor(seed, T.mix32(i)))) / 4294967296.0 - 0.5 : F32)

def dump_bind(-Z: Type, r: T.PL<ModelP()>, k: List<&2, F32> -> Z) -> Z:
  match r:
    case T.PL{_, xs}:
      k(xs)

def init_fill_go(left: Nat, +i: U32, +seed: U32, xs: List<&2, F32>,
  arr: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      arr
    case 1n+q:
      match xs:
        case Nil{}:
          arr
        case Con{+v, rest}:
          value = Bool.pick(F32, F32.is_eq(v, 0.0), rand_f32(i, seed), v)
          init_fill_go(q, (i + 1 : U32), seed, rest, Array.set(F32, arr, i, value))

def load_bind(r: T.PA<ModelP()>) -> ModelP():
  match r:
    case T.PA{p, xs, off}:
      p

def init_params() -> ModelP():
  +n = T.Layer.count(ModelP(), X(), Y(), ModelS(), Model(1n))
  z = T.Layer.zeros(ModelP(), X(), Y(), ModelS(), Model(1n))
  xs = dump_bind(List<&2, F32>,
    T.Layer.dump(ModelP(), X(), Y(), ModelS(), Model(1n), z), x => x)
  arr = init_fill_go(U32.to_nat(n), 0, 12345, xs,
    Array.new(F32, TA.cap_depth(U32.to_nat(n)), 0.0))
  load_bind(T.Layer.load(ModelP(), X(), Y(), ModelS(), Model(1n),
    T.Layer.zeros(ModelP(), X(), Y(), ModelS(), Model(1n)), arr, 0))

def row_bind_rc(~R0: Nat, ~C0: Nat, ~Z: Type, r: TA.MRow<R0, C0>,
  k: TA.Mat<R0, C0> -> List<&2, F32> -> Z) -> Z:
  match r:
    case TA.MRow{m, row}:
      k(m, row)

def gather_go(left: Nat, +b: U32, +idx: U32, z: Y(),
  acc: TA.Mat<B(), V()>) -> TA.Mat<B(), V()>:
  match left:
    case 0n:
      acc
    case 1n+q:
      row_bind_rc(R(), V(), TA.Mat<B(), V()>,
        TA.Mat.read_row(R(), V(), z, idx),
        m => row =>
          gather_go(q, (b + 1 : U32), (idx + U32.from_nat(SL()) : U32), m,
            TA.Mat.write_row(B(), V(), acc, b, row)))

def maybe_nat(-Z: Type, m: Maybe<&2, Nat>, k: Nat -> Z, d: Unit -> Z) -> Z:
  match m:
    case Some{v}:
      k(v)
    case None{}:
      d(Unit{})

def ans_labels(left: Nat, +idx: U32, +labels: List<&2, Nat>,
  +acc: List<&2, Nat>) -> List<&2, Nat>:
  match left:
    case 0n:
      acc
    case 1n++q:
      maybe_nat(List<&2, Nat>, List.get(&2, Nat, labels, U32.to_nat(idx)),
        v => ans_labels(q, (idx + U32.from_nat(SL()) : U32), labels, Con{v, acc}),
        u => ans_labels(q, (idx + U32.from_nat(SL()) : U32), labels, acc))

def scatter_go(left: Nat, +b: U32, +idx: U32, dz: TA.Mat<B(), V()>,
  dy: Y()) -> Y():
  match left:
    case 0n:
      dy
    case 1n+q:
      row_bind_rc(B(), V(), Y(), TA.Mat.read_row(B(), V(), dz, b),
        m => row =>
          scatter_go(q, (b + 1 : U32), (idx + U32.from_nat(SL()) : U32), m,
            TA.Mat.write_row(R(), V(), dy, idx, row)))

def loss_bind(-Z: Type, r: T.LossOut<TA.Mat<B(), V()>>,
  k: F32 -> TA.Mat<B(), V()> -> Z) -> Z:
  match r:
    case T.LossOut{l, dy}:
      k(l, dy)

def ce(z: Y(), labels: List<&2, Nat>) -> T.LossOut<Y()>:
  za = gather_go(B(), 0, U32.from_nat(Nat.sub(SL(), 1n)), z, TA.Mat.zeros(B(), V()))
  la = List.reverse(&2, Nat,
    ans_labels(B(), U32.from_nat(Nat.sub(SL(), 1n)), labels, Nil{}))
  loss_bind(T.LossOut<Y()>, T.CrossEntropy(B(), V(), za, la),
    l => dz => T.LossOut{l,
      scatter_go(B(), 0, U32.from_nat(Nat.sub(SL(), 1n)), dz,
        TA.Mat.zeros(R(), V()))})

def cfg() -> T.OptCfg:
  T.CAdamW{0.0003, 0.9, 0.95, 0.00000001, 0.1, 0.1, 0.05}

# ---- batches: each sequence is 67 bytes: 66 inputs plus the shifted last target ----

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> nats(t)

def xs_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.take(&2, Nat, ids, R())

def ys_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.take(&2, Nat, List.drop(&2, Nat, ids, R()), R())

def load_batch(+path: String, +step: U32) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at(path, (step * U32.from_nat(Nat.add(R(), R())) : U32),
      U32.from_nat(Nat.add(R(), R())))
    return nats(bs)

# ---- training ----

def step_with(+passes: Nat, +ids: List<&2, Nat>, tr: T.Train<ModelP()>) -> T.Train<ModelP()>:
  T.Train.step_at(ModelP(), X(), Y(), ModelS(), List<&2, Nat>,
    Model(passes), ce, tr, xs_of(ids), ys_of(ids))

def tloop(k: Nat, +step: U32, +passes: Nat, +path: String,
  tr: T.Train<ModelP()>) -> IO(T.Train<ModelP()>):
  match k tr:
    case 0n T.Train{p, g, st, oc, p1, p2, +l}:
      do IO<T.Train<ModelP()>>:
        IO.print(String.append("loss ", F32.show(l)))
        return T.Train{p, g, st, oc, p1, p2, l}
    case 1n+q T.Train{p, g, st, oc, p1, p2, +l}:
      do IO<T.Train<ModelP()>>:
        IO.print(String.append("loss ", F32.show(l)))
        ids : List<&2, Nat> <- load_batch(path, step)
        tloop(q, (step + 1 : U32), passes, path, step_with(passes, ids, T.Train{p, g, st, oc, p1, p2, l}))

def finish_train(tr: T.Train<ModelP()>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, oc, p1, p2, l}:
      IO.print(String.append("final_loss ", F32.show(l)))

def train_run(+k: Nat, +passes: Nat, +path: String) -> IO(T.Train<ModelP()>):
  do IO<T.Train<ModelP()>>:
    T.OptCfg.check(cfg())
    p : ModelP() <- IO.pure(ModelP(), init_params())
    tr : T.Train<ModelP()> <- IO.pure(T.Train<ModelP()>,
      T.Train.new(~ModelP(), ~X(), ~Y(), ~ModelS(), ~Model(1n), cfg(), p))
    tloop(k, 0, passes, path, tr)

def train(+k: Nat, +passes: Nat, +path: String) -> IO(Unit):
  do IO<Unit>:
    tr : T.Train<ModelP()> <- train_run(k, passes, path)
    finish_train(tr)

# ---- evaluation: one answer per sequence, at the last position ----

def row_bind(-Z: Type, r: TA.MRow<R(), V()>,
  k: TA.Mat<R(), V()> -> List<&2, F32> -> Z) -> Z:
  match r:
    case TA.MRow{m, row}:
      k(m, row)

def eval_one(+pred: Nat, got: Maybe<&2, Nat>, +ok: Nat) -> Nat:
  match got:
    case Some{v}:
      Bool.pick(Nat, Nat.is_eq(pred, v), 1n+ok, ok)
    case None{}:
      ok

def eval_go(left: Nat, +idx: U32, y: Y(), +labels: List<&2, Nat>, +ok: Nat) -> Nat:
  match left:
    case 0n:
      ok
    case 1n+q:
      row_bind(Nat, TA.Mat.read_row(R(), V(), y, idx),
        m => row =>
          eval_go(q, (idx + U32.from_nat(SL()) : U32), m, labels,
            eval_one(TA.argmax(row), List.get(&2, Nat, labels, U32.to_nat(idx)), ok)))

def eval_fwd(labels: List<&2, Nat>, r: T.Fwd<ModelP(), Y(), ModelS()>) -> IO(Unit):
  match r:
    case T.Fwd{_, y, _}:
      +ok = eval_go(B(), U32.from_nat(Nat.sub(SL(), 1n)), y, labels, 0n)
      IO.print(String.append("accuracy ", F32.show((F32.from_nat(ok) / F32.from_nat(B()) : F32))))

def eval_with(p: ModelP(), +passes: Nat, +path: String) -> IO(Unit):
  do IO<Unit>:
    +ids : List<&2, Nat> <- load_batch(path, 0)
    eval_fwd(ys_of(ids), T.Layer.fwd(ModelP(), X(), Y(), ModelS(), Model(passes), p, xs_of(ids)))

def eval_batch(+passes: Nat, +path: String) -> IO(Unit):
  do IO<Unit>:
    p : ModelP() <- IO.pure(ModelP(), init_params())
    eval_with(p, passes, path)

def eval_tr(tr: T.Train<ModelP()>, +passes: Nat, +path: String) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, oc, p1, p2, l}:
      eval_with(p, passes, path)

def train_eval(+k: Nat, +passes: Nat, +train_path: String, +eval_path: String) -> IO(Unit):
  do IO<Unit>:
    tr : T.Train<ModelP()> <- train_run(k, passes, train_path)
    eval_tr(tr, passes, eval_path)

# ---- evaluation at scale: every hop's whole sample in one file, one step at a time ----

def params_list(p: ModelP()) -> List<&2, F32>:
  dump_bind(List<&2, F32>,
    T.Layer.dump(ModelP(), X(), Y(), ModelS(), Model(1n), p), x => x)

def list_array_go(left: Nat, +i: U32, +xs: List<&2, F32>, arr: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      arr
    case 1n+q:
      match xs:
        case Nil{}:
          arr
        case Con{v, rest}:
          list_array_go(q, (i + 1 : U32), rest, Array.set(F32, arr, i, v))

def list_to_array(+xs: List<&2, F32>) -> Array<F32>:
  arr = Array.new(F32, TA.cap_depth(List.length(&2, F32, xs)), 0.0)
  list_array_go(List.length(&2, F32, xs), 0, xs, arr)

def params_of_array(xs: Array<F32>) -> ModelP():
  load_bind(T.Layer.load(ModelP(), X(), Y(), ModelS(), Model(1n),
    T.Layer.zeros(ModelP(), X(), Y(), ModelS(), Model(1n)), xs, 0))

def eval_steps(k: Nat, +step: U32, +xs: List<&2, F32>, +passes: Nat,
  +path: String) -> IO(Unit):
  match k:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+q:
      do IO<Unit>:
        +ids : List<&2, Nat> <- load_batch(path, step)
        p : ModelP() <- IO.pure(ModelP(), params_of_array(list_to_array(xs)))
        done : Unit <- eval_fwd(ys_of(ids),
          T.Layer.fwd(ModelP(), X(), Y(), ModelS(), Model(passes), p, xs_of(ids)))
        eval_steps(q, (step + 1 : U32), xs, passes, path)

def eval_all_with(p: ModelP(), +passes: Nat, +path: String) -> IO(Unit):
  xs = params_list(p)
  eval_steps(ES(), 0, xs, passes, path)

def eval_all_m(+passes: Nat, +path: String) -> IO(Unit):
  do IO<Unit>:
    p : ModelP() <- IO.pure(ModelP(), init_params())
    eval_all_with(p, passes, path)

def eval_all_tr(tr: T.Train<ModelP()>, +passes: Nat, +path: String) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, oc, p1, p2, l}:
      eval_all_with(p, passes, path)

def train_eval_all(+k: Nat, +passes: Nat, +train_path: String,
  +eval_path: String) -> IO(Unit):
  do IO<Unit>:
    tr : T.Train<ModelP()> <- train_run(k, passes, train_path)
    eval_all_tr(tr, passes, eval_path)

# ---- the command line ----

def usage() -> IO(Unit):
  IO.print("usage: looped train <steps> <passes> <batch-file> | eval <passes> <batch-file> | eval_all <passes> <eval-all-file> | train_eval_all <steps> <passes> <train-file> <eval-all-file>")

def train_m(pk: Maybe<&2, Nat>, pp: Maybe<&2, Nat>, +path: String) -> IO(Unit):
  match pk pp:
    case Some{k} Some{p}:
      train(k, p, path)
    case _ _:
      usage()

def eval_m(pp: Maybe<&2, Nat>, +path: String) -> IO(Unit):
  match pp:
    case Some{p}:
      eval_batch(p, path)
    case None{}:
      usage()

def eval_all_maybe(pp: Maybe<&2, Nat>, +path: String) -> IO(Unit):
  match pp:
    case Some{p}:
      eval_all_m(p, path)
    case None{}:
      usage()

def dispatch_train(ok: Bool, k: String, passes: String, +path: String) -> IO(Unit):
  match ok:
    case True{}:
      train_m(Nat.read(k), Nat.read(passes), path)
    case False{}:
      usage()

def dispatch_eval(ok: Bool, passes: String, +path: String) -> IO(Unit):
  match ok:
    case True{}:
      eval_m(Nat.read(passes), path)
    case False{}:
      usage()

def dispatch_eval_all_ok(ok: Bool, passes: String, +path: String) -> IO(Unit):
  match ok:
    case True{}:
      eval_all_maybe(Nat.read(passes), path)
    case False{}:
      usage()

def dispatch_eval_ok(ok: Bool, +mode: String, passes: String, +path: String) -> IO(Unit):
  match ok:
    case True{}:
      dispatch_eval(True{}, passes, path)
    case False{}:
      dispatch_eval_all_ok(String.eq(mode, "eval_all"), passes, path)

def dispatch_five(+mode: String, passes: String, +path: String) -> IO(Unit):
  dispatch_eval_ok(String.eq(mode, "eval"), mode, passes, path)

def train_eval_m(pk: Maybe<&2, Nat>, pp: Maybe<&2, Nat>, +train_path: String,
  +eval_path: String) -> IO(Unit):
  match pk pp:
    case Some{k} Some{p}:
      train_eval(k, p, train_path, eval_path)
    case _ _:
      usage()

def train_eval_all_m(pk: Maybe<&2, Nat>, pp: Maybe<&2, Nat>, +train_path: String,
  +eval_path: String) -> IO(Unit):
  match pk pp:
    case Some{k} Some{p}:
      train_eval_all(k, p, train_path, eval_path)
    case _ _:
      usage()

def dispatch_train_eval(ok: Bool, k: String, passes: String, +train_path: String,
  +eval_path: String) -> IO(Unit):
  match ok:
    case True{}:
      train_eval_m(Nat.read(k), Nat.read(passes), train_path, eval_path)
    case False{}:
      usage()

def dispatch_train_eval_all_ok(ok: Bool, k: String, passes: String,
  +train_path: String, +eval_path: String) -> IO(Unit):
  match ok:
    case True{}:
      train_eval_all_m(Nat.read(k), Nat.read(passes), train_path, eval_path)
    case False{}:
      usage()

def dispatch_train_eval_ok(ok: Bool, +mode: String, k: String, passes: String,
  +train_path: String, +eval_path: String) -> IO(Unit):
  match ok:
    case True{}:
      dispatch_train_eval(True{}, k, passes, train_path, eval_path)
    case False{}:
      dispatch_train_eval_all_ok(String.eq(mode, "train_eval_all"), k, passes,
        train_path, eval_path)

def dispatch_seven(+mode: String, k: String, passes: String, +train_path: String,
  +eval_path: String) -> IO(Unit):
  dispatch_train_eval_ok(String.eq(mode, "train_eval"), mode, k, passes,
    train_path, eval_path)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{mode, Con{k, Con{passes, Con{train_path, Con{eval_path, Nil{}}}}}}}:
      dispatch_seven(mode, k, passes, train_path, eval_path)
    case Con{prog, Con{mode, Con{k, Con{passes, Con{path, Nil{}}}}}}:
      dispatch_train(String.eq(mode, "train"), k, passes, path)
    case Con{prog, Con{mode, Con{passes, Con{path, Nil{}}}}}:
      dispatch_five(mode, passes, path)
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
