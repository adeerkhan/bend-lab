# The trained UNet's shape, as one layer: 64x64 -> 32x32 -> 16x16 -> 32x32 -> 64x64,
# 16 channels at each level, a group norm before every activation, an up2 at each
# level, a skip at 32x32, and a skip from the input to the 5-channel linear head.
#
# Both parts are load-bearing. A SiLU on the raw input crushes the negative half of
# the velocity (it is bounded below), so the norms come first; and the outer skip is
# the only path that carries the image's amplitude, which the norms would remove.
#
#   bend examples/diffusion/unet/train.bend
#
# The shapes are type-level, so this file is the model and nothing else; the
# rows of `Up2` and `Residual` are literals because a `Nat.mul` in a type
# argument to a polymorphic def overflows the checker's stack. The layer types
# say LOW() -> MID() and IN() -> IN(), so a B other than 2 fails to compile.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./config.bend as C

def B() -> Nat:
  C.B()

def CH() -> Nat:
  C.CH()

def HW() -> Nat:
  C.HW()

def HW2() -> Nat:
  C.HW2()

def HW4() -> Nat:
  C.HW4()

# ---- lists to images and back: the gate example's up2, as a layer ----------

def tk(+n: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match n xs:
    case 0n _:
      acc
    case 1n+p Con{h, t}:
      tk(p, t, Con{h, acc})
    case _ _:
      acc

def dr(+n: Nat, +xs: List<&2, F32>) -> List<&2, F32>:
  match n xs:
    case 0n _:
      xs
    case 1n+p Con{h, t}:
      dr(p, t)
    case _ _:
      Nil{}

def dup(+left: Nat, +xs: List<&2, F32>, +chunk: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+p:
      c = List.reverse(&2, F32, tk(chunk, xs, Nil{}))
      d = List.reverse(&2, F32, tk(chunk, xs, Nil{}))
      List.append(&2, F32, c, List.append(&2, F32, d, dup(p, dr(chunk, xs), chunk)))

def add2(+xs: List<&2, F32>, +ys: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Con{a, at} Con{b, bt}:
      add2(at, bt, Con{(a + b : F32), acc})
    case _ _:
      List.reverse(&2, F32, acc)

def sum2c(+left: Nat, +xs: List<&2, F32>, +chunk: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+p:
      a = List.reverse(&2, F32, tk(chunk, xs, Nil{}))
      +r1 = dr(chunk, xs)
      b = List.reverse(&2, F32, tk(chunk, r1, Nil{}))
      List.append(&2, F32, add2(a, b, Nil{}), sum2c(p, dr(chunk, r1), chunk))

def up_list(+rows: Nat, +w: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, F32>:
  dup(Nat.mul(Nat.mul(2n, rows), w), dup(rows, xs, Nat.mul(w, c)), c)

def down_list(+rows: Nat, +w: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, F32>:
  sum2c(rows, sum2c(Nat.mul(Nat.mul(2n, rows), w), xs, c), Nat.mul(w, c))

def ml_list(-r: Nat, -c: Nat, ml: TA.ML<r, c>) -> List<&2, F32>:
  match ml:
    case TA.ML{mat, l}:
      l

def up_mat(+rows: Nat, +w: Nat, +c: Nat, x: TA.Mat<Nat.mul(rows, w), c>) -> TA.Mat<Nat.mul(Nat.mul(2n, rows), Nat.mul(2n, w)), c>:
  TA.Mat.from_list(Nat.mul(Nat.mul(2n, rows), Nat.mul(2n, w)), c, up_list(rows, w, c, ml_list(Nat.mul(rows, w), c, TA.Mat.to_list(Nat.mul(rows, w), c, x))))

def down_mat(+rows: Nat, +w: Nat, +c: Nat, x: TA.Mat<Nat.mul(Nat.mul(2n, rows), Nat.mul(2n, w)), c>) -> TA.Mat<Nat.mul(rows, w), c>:
  TA.Mat.from_list(Nat.mul(rows, w), c, down_list(rows, w, c, ml_list(Nat.mul(Nat.mul(2n, rows), Nat.mul(2n, w)), c, TA.Mat.to_list(Nat.mul(Nat.mul(2n, rows), Nat.mul(2n, w)), c, x))))

# up2: a parameterless layer whose backward is its exact adjoint (the gate's
# own argument: the four cells that came from one input cell are summed)
def Up2(+rows: Nat, +w: Nat, +c: Nat) -> T.Layer<Unit, TA.Mat<Nat.mul(rows, w), c>, TA.Mat<Nat.mul(Nat.mul(2n, rows), Nat.mul(2n, w)), c>, Unit>:
  T.Layer{p => x => T.Fwd{p, up_mat(rows, w, c, x), Unit{}},
    p => s => dy => g => T.Bwd{p, down_mat(rows, w, c, dy), g},
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0}

def INW() -> Nat:
  5n

def IN() -> Type:
  TA.Mat<8192n, 5n>

def MID() -> Type:
  TA.Mat<2048n, 16n>

def LOW() -> Type:
  TA.Mat<512n, 16n>

def UP() -> Type:
  TA.Mat<8192n, 16n>

def OUT() -> Type:
  TA.Mat<8192n, 1n>

def CP1() -> Type:
  T.Lin<45n, 16n>

def S1() -> Type:
  TA.Mat<2048n, 45n>

def CP3() -> Type:
  T.Lin<144n, 16n>

def S3() -> Type:
  TA.Mat<512n, 144n>

def S4() -> Type:
  TA.Mat<2048n, 144n>

def CHP() -> Type:
  T.Lin<16n, 1n>

def SH() -> Type:
  TA.Mat<8192n, 16n>

def conv1() -> T.Layer<CP1(), IN(), MID(), S1()>:
  T.Conv2d(2n, INW(), 16n, 3n, 63n, 63n, 1n, 1n, 0n)

def gn1() -> T.Layer<T.LN<16n>, MID(), MID(), MID()>:
  T.GroupNorm(2n, 1024n, 1n, 16n, 0.00001)

def silu1() -> T.Layer<Unit, MID(), MID(), MID()>:
  T.SiLU(2048n, 16n)

def conv3() -> T.Layer<CP3(), MID(), LOW(), S3()>:
  T.Conv2d(2n, 16n, 16n, 3n, 31n, 31n, 1n, 1n, 0n)

def gn4() -> T.Layer<T.LN<16n>, LOW(), LOW(), LOW()>:
  T.GroupNorm(2n, 256n, 1n, 16n, 0.00001)

def silu4() -> T.Layer<Unit, LOW(), LOW(), LOW()>:
  T.SiLU(512n, 16n)

def conv4() -> T.Layer<CP3(), MID(), MID(), S4()>:
  T.Conv2d(2n, 16n, 16n, 3n, 31n, 31n, 1n, 0n, 0n)

def gn3() -> T.Layer<T.LN<16n>, MID(), MID(), MID()>:
  T.GroupNorm(2n, 1024n, 1n, 16n, 0.00001)

def silu3() -> T.Layer<Unit, MID(), MID(), MID()>:
  T.SiLU(2048n, 16n)

def head() -> T.Layer<CHP(), UP(), OUT(), SH()>:
  T.Conv2d(2n, 16n, 1n, 1n, 63n, 63n, 0n, 0n, 0n)

def P2C() -> Type:
  T.Two<T.Two<CP3(), T.LN<16n>>, Unit>

def S2C() -> Type:
  T.Two<T.Two<S4(), MID()>, MID()>

def DownP() -> Type:
  T.Two<T.Two<CP1(), T.LN<16n>>, Unit>

def DownS() -> Type:
  T.Two<T.Two<S1(), MID()>, MID()>

def Down() -> T.Layer<DownP(), IN(), MID(), DownS()>:
  T.Seq(~T.Two<CP1(), T.LN<16n>>, ~Unit, ~IN(), ~MID(), ~MID(), ~(T.Two<S1(), MID()>), ~MID(),
    T.Seq(~CP1(), ~T.LN<16n>, ~IN(), ~MID(), ~MID(), ~S1(), ~MID(), conv1(), gn1()), silu1())

def LowP() -> Type:
  T.Two<T.Two<CP3(), T.LN<16n>>, Unit>

def LowS() -> Type:
  T.Two<T.Two<S3(), LOW()>, LOW()>

def Low() -> T.Layer<LowP(), MID(), LOW(), LowS()>:
  T.Seq(~T.Two<CP3(), T.LN<16n>>, ~Unit, ~MID(), ~LOW(), ~LOW(), ~(T.Two<S3(), LOW()>), ~LOW(),
    T.Seq(~CP3(), ~T.LN<16n>, ~MID(), ~LOW(), ~LOW(), ~S3(), ~LOW(), conv3(), gn4()), silu4())

def UpP() -> Type:
  T.Two<Unit, P2C()>

def UpS() -> Type:
  T.Two<Unit, S2C()>

def UpMiddle() -> T.Layer<UpP(), LOW(), MID(), UpS()>:
  T.Seq(~Unit, ~P2C(), ~LOW(), ~MID(), ~MID(), ~Unit, ~S2C(), Up2(32n, 16n, 16n),
    T.Seq(~(T.Two<CP3(), T.LN<16n>>), ~Unit, ~MID(), ~MID(), ~MID(), ~(T.Two<S4(), MID()>), ~MID(),
      T.Seq(~CP3(), ~T.LN<16n>, ~MID(), ~MID(), ~MID(), ~S4(), ~MID(), conv4(), gn3()),
      silu3()))

def LevelP() -> Type:
  T.Two<T.Two<Unit, T.Two<LowP(), UpP()>>, Unit>

def LevelS() -> Type:
  T.Two<T.Two<Unit, T.Two<LowS(), UpS()>>, Unit>

def Level() -> T.Layer<LevelP(), MID(), MID(), LevelS()>:
  T.Residual(2048n, 16n, T.Two<LowP(), UpP()>, T.Two<LowS(), UpS()>,
    T.Seq(~LowP(), ~UpP(), ~MID(), ~LOW(), ~MID(), ~LowS(), ~UpS(), Low(), UpMiddle()))

def TopP() -> Type:
  T.Two<Unit, T.Two<CHP(), Unit>>

def TopS() -> Type:
  T.Two<Unit, T.Two<SH(), Unit>>

def Top() -> T.Layer<TopP(), MID(), OUT(), TopS()>:
  T.Seq(~Unit, ~(T.Two<CHP(), Unit>), ~MID(), ~UP(), ~OUT(), ~Unit, ~(T.Two<SH(), Unit>),
    Up2(64n, 32n, 16n),
    T.Seq(~CHP(), ~Unit, ~UP(), ~OUT(), ~OUT(), ~SH(), ~Unit, head(), T.Id(OUT())))

def BodyP() -> Type:
  T.Two<T.Two<DownP(), LevelP()>, TopP>

def BodyS() -> Type:
  T.Two<T.Two<DownS(), LevelS()>, TopS>

def Body() -> T.Layer<BodyP(), IN(), OUT(), BodyS()>:
  T.Seq(~(T.Two<DownP(), LevelP()>), ~TopP(), ~IN(), ~MID(), ~OUT(),
    ~(T.Two<DownS(), LevelS()>), ~TopS(),
    T.Seq(~DownP(), ~LevelP(), ~IN(), ~MID(), ~MID(), ~DownS(), ~LevelS(), Down(), Level()), Top())

# a learned linear bypass beside the U: the marginal conditional field is linear,
# and a one-layer linear map finds it in 300 steps where the stack does not
def LINP() -> Type:
  T.Lin<5n, 1n>

def LINS() -> Type:
  TA.Mat<8192n, 5n>

def bypass() -> T.Layer<LINP(), IN(), OUT(), LINS()>:
  T.Conv2d(2n, INW(), 1n, 1n, 63n, 63n, 0n, 0n, 0n)

def ForkP() -> Type:
  T.Two<BodyP(), LINP()>

def ForkS() -> Type:
  T.Two<BodyS(), LINS()>

def Forked() -> T.Layer<ForkP(), IN(), T.Two<OUT(), OUT()>, ForkS()>:
  T.Fork(~BodyP(), ~LINP(), ~IN(), ~OUT(), ~OUT(), ~BodyS(), ~LINS(),
    ~(m => T.Mat.dup(8192n, 5n, m)), ~(a => b => T.Mat.sum2(8192n, 5n, a, b)),
    Body(), bypass())

def ModelP() -> Type:
  T.Two<ForkP(), Unit>

def ModelS() -> Type:
  T.Two<ForkS(), Unit>

def Model() -> T.Layer<ModelP(), IN(), OUT(), ModelS()>:
  T.Seq(~ForkP(), ~Unit, ~IN(), ~(T.Two<OUT(), OUT()>), ~OUT(), ~ForkS(), ~Unit,
    Forked(), T.Add2(8192n, 1n))
