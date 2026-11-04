# Rectified flow at 64x64 on Oxford Flowers, written in Bend on BendTorch.
#
#   make flow64 PRESET=tiny        (see config.bend for the presets)
#   scripts/bend-guard.sh bend examples/diffusion/flowers_flow/main.bend
#
# One 64-pixel row is one token, so the image is read in raster order and the
# token builder never indexes backwards. config.bend sets the images per step,
# the width and the rows; one pre-LN block and a linear head are fixed here
# because three blocks need about 10 GB to compile.
#
# The objective is rectified flow:
#
#   x_t = noise*(1-t) + data*t,  target v = data - noise,  MSE
#   sample: x <- x + v(x,t)*dt,  Euler over linspace(0,1,steps+1)
#
# Data: 512 flowers at 64x64, one grayscale byte per pixel, made from the pinned
# RGB export by reference/flowers_export.py and pinned itself at
# 6b9c4c66ad5c03939875557d8a77af02845bf924849b4964bcd46b69300a4956.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bendtorch-gpu@2.0.0.0/main.bend as F
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ./config.bend as C

def B() -> Nat:
  C.B()

def TK() -> Nat:
  C.TK()

def RW() -> Nat:
  C.RW()

def FE() -> Nat:
  C.FE()

def PX() -> Nat:
  C.PX()

def W() -> Nat:
  C.W()

def HI() -> Nat:
  C.HI()

def IMG() -> Nat:
  C.PIL()

def BATCH() -> Nat:
  C.BATCH()

def NIMG() -> Nat:
  C.NIMG()

def STEPS() -> Nat:
  C.STEPS()

def EULER() -> Nat:
  C.EULER()

# The export directory: the runner's cache when MLBEND_DATA is set, the
# machine's export cache otherwise. The file name is the config's, so a preset
# picks the export and the environment picks where it lives.
def cache_root() -> String:
  "/home/adeerkhan/.cache/bend-lab"

def root_of(e: Result<&1, &1, U32 & String, String>) -> IO(String):
  match e:
    case Fail{code}:
      IO.pure(String, cache_root())
    case Done{d}:
      IO.pure(String, d)

def export_dir() -> IO(String):
  do IO<String>:
    e : Result<&1, &1, U32 & String, String> <- IO.get_env("MLBEND_DATA")
    root_of(e)

def path(root: String) -> String:
  String.append(root, String.append("/flowers/", C.DATA()))

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def A(c: Nat) -> Type:
  TA.Mat<RW(), c>

def X() -> Type:
  TA.Mat<RW(), FE()>

def Y() -> Type:
  TA.Mat<RW(), PX()>

# ---- the transformer, the ViT example's model at 64 tokens of 64 pixels ----

def KVP() -> Type:
  T.Two<M(W(), W()), M(W(), W())>

def KVY() -> Type:
  T.Two<A(W()), A(W())>

def KVS() -> Type:
  T.Two<A(W()), A(W())>

def KVL() -> T.Layer<KVP, A(W()), KVY, KVS>:
  T.Fork(~M(W(), W()), ~M(W(), W()), ~A(W()), ~A(W()), ~A(W()), ~A(W()), ~A(W()),
    ~(m => T.Mat.dup(RW(), W(), m)),
    ~(x => y => T.Mat.sum2(RW(), W(), x, y)),
    ~F.LinearNBF(RW(), W(), W(), 0n),
    ~F.LinearNBF(RW(), W(), W(), 0n))

def QKVP() -> Type:
  T.Two<M(W(), W()), KVP>

def QKVS() -> Type:
  T.Two<A(W()), KVS>

def QKV() -> T.Layer<QKVP, A(W()), T.Two<A(W()), KVY>, QKVS>:
  T.Fork(~M(W(), W()), ~KVP, ~A(W()), ~A(W()), ~KVY, ~A(W()), ~KVS,
    ~(m => T.Mat.dup(RW(), W(), m)),
    ~(x => y => T.Mat.sum2(RW(), W(), x, y)),
    ~F.LinearNBF(RW(), W(), W(), 0n), ~KVL())

def AX() -> Type:
  T.Two<A(W()), KVY>

def CoreP() -> Type:
  T.Two<Unit, M(W(), W())>

def CoreS() -> Type:
  T.Two<T.AttnS, A(W())>

def Core() -> T.Layer<CoreP, AX, A(W()), CoreS>:
  T.Seq(~Unit, ~M(W(), W()), ~AX, ~A(W()), ~A(W()), ~T.AttnS, ~A(W()),
    ~F.Attention(RW(), W(), W(), B(), TK(), 1n, 1n, W(), 0.125, {==}, {==}, {==}),
    ~F.LinearNBF(RW(), W(), W(), 0n))

def AttP() -> Type:
  T.Two<QKVP, CoreP>

def AttS() -> Type:
  T.Two<QKVS, CoreS>

def Att() -> T.Layer<AttP, A(W()), A(W()), AttS>:
  T.Seq(~QKVP, ~CoreP, ~A(W()), ~AX, ~A(W()), ~QKVS, ~CoreS, ~QKV(), ~Core())

def AttBlkP() -> Type:
  T.Two<T.Two<Unit, T.Two<T.LN<W()>, AttP>>, Unit>

def AttBlkS() -> Type:
  T.Two<T.Two<Unit, T.Two<A(W()), AttS>>, Unit>

def AttIn() -> T.Layer<T.Two<T.LN<W()>, AttP>, A(W()), A(W()), T.Two<A(W()), AttS>>:
  T.Seq(~T.LN<W()>, ~AttP, ~A(W()), ~A(W()), ~A(W()), ~A(W()), ~AttS,
    ~T.LayerNorm(RW(), W(), 0.00001), ~Att())

def AttBlk() -> T.Layer<AttBlkP, A(W()), A(W()), AttBlkS>:
  T.Residual(RW(), W(), T.Two<T.LN<W()>, AttP>, T.Two<A(W()), AttS>, AttIn())

def MIP() -> Type:
  T.Two<T.Two<M(HI(), W()), Unit>, M(W(), HI())>

def MIS() -> Type:
  T.Two<T.Two<A(W()), A(HI())>, A(HI())>

def MlpP() -> Type:
  T.Two<T.Two<Unit, T.Two<T.LN<W()>, MIP>>, Unit>

def MlpS() -> Type:
  T.Two<T.Two<Unit, T.Two<A(W()), MIS>>, Unit>

def MlpHalf() -> T.Layer<T.Two<M(HI(), W()), Unit>, A(W()), A(HI()), T.Two<A(W()), A(HI())>>:
  T.Seq(~M(HI(), W()), ~Unit, ~A(W()), ~A(HI()), ~A(HI()), ~A(W()), ~A(HI()),
    ~F.LinearNBF(RW(), W(), HI(), 0n), ~T.GELU(RW(), HI()))

def MlpInner() -> T.Layer<MIP, A(W()), A(W()), MIS>:
  T.Seq(~T.Two<M(HI(), W()), Unit>, ~M(W(), HI()), ~A(W()), ~A(HI()), ~A(W()),
    ~T.Two<A(W()), A(HI())>, ~A(HI()), ~MlpHalf(), ~F.LinearNBF(RW(), HI(), W(), 0n))

def MlpIn() -> T.Layer<T.Two<T.LN<W()>, MIP>, A(W()), A(W()), T.Two<A(W()), MIS>>:
  T.Seq(~T.LN<W()>, ~MIP, ~A(W()), ~A(W()), ~A(W()), ~A(W()), ~MIS,
    ~T.LayerNorm(RW(), W(), 0.00001), ~MlpInner())

def Mlp() -> T.Layer<MlpP, A(W()), A(W()), MlpS>:
  T.Residual(RW(), W(), T.Two<T.LN<W()>, MIP>, T.Two<A(W()), MIS>, MlpIn())

def B12P() -> Type:
  T.Two<AttBlkP, MlpP>

def B12S() -> Type:
  T.Two<AttBlkS, MlpS>

def B12() -> T.Layer<B12P, A(W()), A(W()), B12S>:
  T.Seq(~AttBlkP, ~MlpP, ~A(W()), ~A(W()), ~A(W()), ~AttBlkS, ~MlpS,
    ~AttBlk(), ~Mlp())


def B3P() -> Type:
  T.Two<T.Two<B12P, B12P>, B12P>

def B3S() -> Type:
  T.Two<T.Two<B12S, B12S>, B12S>

def B2() -> T.Layer<T.Two<B12P, B12P>, A(W()), A(W()), T.Two<B12S, B12S>>:
  T.Seq(~B12P, ~B12P, ~A(W()), ~A(W()), ~A(W()), ~B12S, ~B12S, ~B12(), ~B12())

def Body3() -> T.Layer<B3P, A(W()), A(W()), B3S>:
  T.Seq(~T.Two<B12P, B12P>, ~B12P, ~A(W()), ~A(W()), ~A(W()),
    ~T.Two<B12S, B12S>, ~B12S, ~B2(), ~B12())

def EmbP() -> Type:
  TA.Mat<W(), FE()>

def HP() -> Type:
  T.Two<T.LN<W()>, TA.Mat<PX(), W()>>

def HS() -> Type:
  T.Two<A(W()), A(W())>

def ModelP() -> Type:
  T.Two<T.Two<EmbP, B3P>, HP>

def ModelS() -> Type:
  T.Two<T.Two<TA.Mat<RW(), FE()>, B3S>, HS>

def HeadP() -> Type:
  TA.Mat<PX(), W()>

def HeadS() -> Type:
  TA.Mat<RW(), W()>

def Embed() -> T.Layer<EmbP, TA.Mat<RW(), FE()>, A(W()), TA.Mat<RW(), FE()>>:
  F.LinearNBF(RW(), FE(), W(), 0n)

def Head() -> T.Layer<HeadP, A(W()), TA.Mat<RW(), PX()>, HeadS>:
  F.LinearNBF(RW(), W(), PX(), 0n)

def HeadN() -> T.Layer<HP, A(W()), TA.Mat<RW(), PX()>, HS>:
  T.Seq(~T.LN<W()>, ~TA.Mat<PX(), W()>, ~A(W()), ~A(W()), ~TA.Mat<RW(), PX()>(),
    ~A(W()), ~A(W()), ~T.LayerNorm(RW(), W(), 0.00001), ~Head())

def Model() -> T.Layer<ModelP, TA.Mat<RW(), FE()>, TA.Mat<RW(), PX()>, ModelS>:
  T.Seq(~T.Two<EmbP, B3P>, ~HP, ~TA.Mat<RW(), FE()>, ~A(W()), ~TA.Mat<RW(), PX()>,
    ~T.Two<TA.Mat<RW(), FE()>, B3S>, ~HS,
    ~T.Seq(~EmbP, ~B3P, ~TA.Mat<RW(), FE()>, ~A(W()), ~A(W()),
      ~TA.Mat<RW(), FE()>, ~B3S, ~Embed(), ~Body3()),
    ~HeadN())

# ---- the data: 512 flowers, uint8, one row per token ----

def rd_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  match m:
    case Tuple{f, r}:
      do IO<List<&2, U32>>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
        File.close(f)
        return bs

def read_batch(root: String, +s: Nat) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path(root), "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <-
      File.read_at(f, U32.from_nat(Nat.mul(Nat.mod(Nat.mul(s, B()), NIMG()), IMG())),
        U32.from_nat(BATCH()))
    rd_done(m)

# one level per image, U32 Weyl in [0, 1)
def level(+s: Nat, +i: Nat) -> F32:
  +h = U32.add(U32.mul(U32.from_nat(s), 2654435761),
    U32.mul(U32.from_nat(i), 2246822519))
  (F32.from_nat(U32.to_nat(h)) / 4294967296.0 : F32)

def freqs() -> List<&2, F32>:
  Con{0.5, Con{1.0, Con{2.0, Con{4.0, Nil{}}}}}

def cons2(+a: F32, +b: F32, acc: List<&2, F32>) -> List<&2, F32>:
  Con{b, Con{a, acc}}

def tp_go(fs: List<&2, F32>, +u: F32, +acc: List<&2, F32>) -> List<&2, F32>:
  match fs:
    case Nil{}:
      acc
    case Con{+f, t}:
      tp_go(t, u, cons2(F32.sin((6.2831855 * f * u : F32)),
        F32.cos((6.2831855 * f * u : F32)), acc))

def feat(t: F32, k: Nat) -> List<&2, F32>:
  tp_go(freqs(), t, tp_go(freqs(), (F32.from_nat(k) / F32.from_nat(TK()) : F32), Nil{}))

type RS is Type:
  RS{data: List<&2, U32>, noise: List<&2, F32>, inp: List<&2, F32>, tgt: List<&2, F32>}

def feats(+t: F32, +k: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  tp_go(freqs(), t, tp_go(freqs(), (F32.from_nat(k) / F32.from_nat(TK()) : F32), acc))

# one pass over the batch: image, row, column; one pixel per call
def tok_go(+left: Nat, +img: Nat, +row: Nat, +col: Nat, data: List<&2, U32>,
  noise: List<&2, F32>, +inp: List<&2, F32>, +tgt: List<&2, F32>,
  +t: F32, +s: Nat) -> RS:
  match left:
    case 0n:
      RS{data, noise, inp, tgt}
    case 1n+rest:
      # one value matched, not three: `match img row col` costs 4.8 GB of
      # codegen, `match img col` costs 6 MB for the same work (2026-10-30)
      match img col:
        case 8n _:
          RS{data, noise, inp, tgt}
        case _ 64n:
          tok_go(rest, img, (1n + row : Nat), 0n, data, noise, feats(t, row, inp),
            tgt, t, s)
        case _ _:
          match data noise:
            case Con{b, dt} Con{+x0, nt}:
              +x1 = (((F32.from_nat(U32.to_nat(b)) / 127.5 : F32) - 1.0) : F32)
              +xt = ((x0 * (1.0 - t) : F32) + (x1 * t : F32) : F32)
              tok_go(rest, img, row, (1n + col : Nat), dt, nt,
                Con{xt, inp}, Con{(x1 - x0 : F32), tgt}, t, s)
            case _ _:
              RS{data, noise, inp, tgt}

type LR is Type:
  LR{inp: List<&2, F32>, tgt: List<&2, F32>}

def lr_of(r: RS) -> LR:
  match r:
    case RS{d, n, inp, tgt}:
      LR{List.reverse(&2, F32, inp), List.reverse(&2, F32, tgt)}

def pair_of(+s: Nat, data: List<&2, U32>) -> LR:
  lr_of(tok_go(40000n, 0n, 0n, 0n, data, G.noise_of(s, BATCH()), Nil{}, Nil{},
    level(s, 0n), s))

# ---- the model input while sampling: rows of x, then time and position ----

type XS is Type:
  XS{x: List<&2, F32>, acc: List<&2, F32>}

def stok_go(+left: Nat, +img: Nat, +row: Nat, +col: Nat, x: List<&2, F32>,
  +t: F32, +acc: List<&2, F32>) -> XS:
  match left:
    case 0n:
      XS{x, acc}
    case 1n+rest:
      match img col:
        case 8n _:
          XS{x, acc}
        case _ 64n:
          stok_go(rest, img, (1n + row : Nat), 0n, x, t, feats(t, row, acc))
        case _ _:
          match x:
            case Con{h, xt}:
              stok_go(rest, img, row, (1n + col : Nat), xt, t, Con{h, acc})
            case Nil{}:
              XS{x, acc}

def inp_of(r: XS) -> List<&2, F32>:
  match r:
    case XS{x, a}:
      List.reverse(&2, F32, a)

def sample_input(+t: F32, x: List<&2, F32>) -> List<&2, F32>:
  inp_of(stok_go(40000n, 0n, 0n, 0n, x, t, Nil{}))

# ---- the step ----

def cfg() -> T.OptCfg:
  T.CAdam{0.001, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

def mse(y: Y(), tgt: Y()) -> T.LossOut<Y()>:
  T.MSE(RW(), PX(), y, tgt)

def scale_rev(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      scale_rev(t, Con{(0.1 * h : F32), acc})

def scale_go(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  List.reverse(&2, F32, scale_rev(xs, acc))

def fill_go(left: Nat, xs: List<&2, F32>, +i: U32, arr: Array<F32>) -> Array<F32>:
  match left xs:
    case 0n _:
      arr
    case 1n+rest Con{h, t}:
      fill_go(rest, t, U32.add(i, U32.from_nat(1n)), Array.set(F32, arr, i, h))
    case _ _:
      arr

def un_pa(pa: T.PA<ModelP()>) -> ModelP():
  match pa:
    case T.PA{p, xs, off}:
      p

def load_p(p0: ModelP(), arr: Array<F32>) -> ModelP():
  un_pa(T.Layer.load(ModelP(), X(), Y(), ModelS(), Model(), p0, arr, U32.from_nat(0n)))

def init_params() -> ModelP():
  +n = T.Layer.count(ModelP(), X(), Y(), ModelS(), Model())
  +nn = U32.to_nat(n)
  arr = fill_go(nn, scale_go(G.noise_of(1n, nn), Nil{}), U32.from_nat(0n),
    Array.new(F32, T.depth(nn), 0.0))
  load_p(T.Layer.zeros(ModelP(), X(), Y(), ModelS(), Model()), arr)

def step_lr(lr: LR, tr: T.Train<ModelP()>) -> T.Train<ModelP()>:
  match lr:
    case LR{inp, tgt}:
      T.Train.step(~ModelP(), ~X(), ~Y(), ~ModelS(), ~Y(), ~Model(), ~mse, tr,
        TA.Mat.from_list(RW(), FE(), inp), TA.Mat.from_list(RW(), PX(), tgt))

def do_step(+s: Nat, tr: T.Train<ModelP()>, data: List<&2, U32>) -> T.Train<ModelP()>:
  step_lr(pair_of(s, data), tr)

def should_print(+i: Nat) -> Bool:
  Bool.or(Nat.is_eq(Nat.mod(i, 50n), 1n), Nat.is_eq(i, 999n))

def print_or(sp: Bool, +l: F32) -> IO(Unit):
  match sp:
    case True{}:
      do IO<Unit>:
        IO.print(String.append("loss: ", F32.show(l)))
        return Unit{}
    case False{}:
      do IO<Unit>:
        return Unit{}

def loop(+root: String, left: Nat, +i: Nat, tr: T.Train<ModelP()>) -> IO(T.Train<ModelP()>):
  match left tr:
    case 0n T.Train{p, g, st, c, q1, q2, +l}:
      do IO<T.Train<ModelP()>>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        return T.Train{p, g, st, c, q1, q2, l}
    case 1n+rest T.Train{p, g, st, c, q1, q2, +l}:
      do IO<T.Train<ModelP()>>:
        data : List<&2, U32> <- read_batch(root, i)
        Unit <- print_or(should_print(i), l)
        loop(root, rest, (1n + i : Nat), do_step(i, T.Train{p, g, st, c, q1, q2, l}, data))

# ---- sampling: Euler from noise ----

type VE is Type:
  VE{p: ModelP(), v: List<&2, F32>}

type VS is Type:
  VS{p: ModelP(), x: List<&2, F32>}

def flat_m(m: TA.ML<RW(), PX()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def ve_of(f: T.Fwd<ModelP, Y(), ModelS()>) -> VE:
  match f:
    case T.Fwd{p2, y, s}:
      VE{p2, flat_m(TA.Mat.to_list(RW(), PX(), y))}

def fwd_v(p: ModelP(), +inp: List<&2, F32>) -> VE:
  ve_of(T.Layer.fwd(ModelP(), X(), Y(), ModelS(), Model(),
    p, TA.Mat.from_list(RW(), FE(), inp)))

def upd_go(+left: Nat, x: List<&2, F32>, v: List<&2, F32>,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      match x v:
        case Con{a, xt} Con{b, vt}:
          upd_go(rest, xt, vt, Con{(a + (0.03125 * b : F32) : F32), acc})
        case _ _:
          acc

def upd(x: List<&2, F32>, v: List<&2, F32>) -> List<&2, F32>:
  List.reverse(&2, F32, upd_go(BATCH(), x, v, Nil{}))

def ve_step(ve: VE, x: List<&2, F32>) -> VS:
  match ve:
    case VE{p2, v}:
      VS{p2, upd(x, v)}

def step_vs(+i: Nat, st: VS) -> VS:
  match st:
    case VS{p, +x}:
      +t = (F32.from_nat(i) / F32.from_nat(EULER()) : F32)
      ve_step(fwd_v(p, sample_input(t, x)), x)

def euler_go(+left: Nat, +i: Nat, st: VS) -> VS:
  match left:
    case 0n:
      st
    case 1n+rest:
      euler_go(rest, (1n + i : Nat), step_vs(i, st))

def train_params(tr: T.Train<ModelP()>) -> ModelP():
  match tr:
    case T.Train{p, g, st, c, q1, q2, l}:
      p

def sample_x(st: VS) -> List<&2, F32>:
  match st:
    case VS{p, x}:
      x

def take64(+left: Nat, x: List<&2, F32>, +acc: List<&2, F32>) -> XS:
  match left:
    case 0n:
      XS{x, acc}
    case 1n+rest:
      match x:
        case Con{h, t}:
          take64(rest, t, Con{h, acc})
        case Nil{}:
          XS{x, acc}

def show_row(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_row(t)))

def show_one(r: XS) -> IO(List<&2, F32>):
  match r:
    case XS{x2, a2}:
      do IO<List<&2, F32>>:
        Unit <- IO.print(show_row(List.reverse(&2, F32, a2)))
        return x2

def show_rows(+left: Nat, x: List<&2, F32>) -> IO(Unit):
  match left:
    case 0n:
      IO.print("samples_end")
    case 1n+rest:
      do IO<Unit>:
        x2 : List<&2, F32> <- show_one(take64(64n, x, Nil{}))
        show_rows(rest, x2)

def report(x: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("params: ", U32.show(T.Layer.count(ModelP(), X(), Y(),
      ModelS(), Model()))))
    IO.print("samples:")
    show_rows(RW(), x)

def main() -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    root : String <- export_dir()
    tr : T.Train<ModelP()> <- loop(root, STEPS(), 0n,
      T.Train.new(~ModelP(), ~X(), ~Y(), ~ModelS(), ~Model(), cfg(), init_params()))
    p : ModelP() <- IO.pure(ModelP(), train_params(tr))
    report(sample_x(euler_go(EULER(), 0n, VS{p, G.noise_of(2000000n, BATCH())})))
