# The trained UNet's shape, as one layer: 64x64 -> 32x32 -> 16x16 -> 32x32 -> 64x64,
# CH channels at each level, an up2 upsample and additive skips at both levels.
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

# ---- the model's types ------------------------------------------------------

def INW() -> Nat:
  5n

def IN() -> Type:
  TA.Mat<8192n, 5n>

def OUT() -> Type:
  TA.Mat<8192n, 1n>

def MID() -> Type:
  TA.Mat<2048n, 16n>

def LOW() -> Type:
  TA.Mat<512n, 16n>

def CP1() -> Type:
  T.Lin<45n, CH()>

def CP3() -> Type:
  T.Lin<144n, CH()>

def HEAD() -> Type:
  T.Lin<16n, 1n>

def S1() -> Type:
  TA.Mat<2048n, 45n>

def S3() -> Type:
  TA.Mat<512n, 144n>

def S4() -> Type:
  TA.Mat<2048n, 144n>

def SH() -> Type:
  TA.Mat<8192n, 16n>

# ---- the four convolutions and the four norms ------------------------------

def conv1() -> T.Layer<CP1(), IN(), MID(), S1()>:
  T.Conv2d(B(), INW(), CH(), 3n, 63n, 63n, 1n, 1n, 0n)

def gn1() -> T.Layer<T.LN<CH()>, MID(), MID(), MID()>:
  T.GroupNorm(B(), HW2(), 1n, CH(), 0.00001)

def silu1() -> T.Layer<Unit, MID(), MID(), MID()>:
  T.SiLU(Nat.mul(B(), HW2()), CH())

def conv3() -> T.Layer<CP3(), MID(), LOW(), S3()>:
  T.Conv2d(B(), CH(), CH(), 3n, 31n, 31n, 1n, 1n, 0n)

def gn4() -> T.Layer<T.LN<CH()>, LOW(), LOW(), LOW()>:
  T.GroupNorm(B(), HW4(), 1n, CH(), 0.00001)

def silu4() -> T.Layer<Unit, LOW(), LOW(), LOW()>:
  T.SiLU(Nat.mul(B(), HW4()), CH())

# the row counts are literals because a `Nat.mul` in a type argument here blows
# the checker's stack; the layer's type still says LOW() -> MID(), so a B other
# than 2 fails to compile instead of building the wrong shape
def upL() -> T.Layer<Unit, LOW(), MID(), Unit>:
  Up2(32n, 16n, CH())

def conv4() -> T.Layer<CP3(), MID(), MID(), S4()>:
  T.Conv2d(B(), CH(), CH(), 3n, 31n, 31n, 1n, 0n, 0n)

def gn3() -> T.Layer<T.LN<CH()>, MID(), MID(), MID()>:
  T.GroupNorm(B(), HW2(), 1n, CH(), 0.00001)

def silu3() -> T.Layer<Unit, MID(), MID(), MID()>:
  T.SiLU(Nat.mul(B(), HW2()), CH())

def upM() -> T.Layer<Unit, MID(), TA.Mat<8192n, 16n>, Unit>:
  Up2(64n, 32n, CH())

# the head is a 1x1 conv: a linear map from 16 channels to the velocity, with no
# activation, so the field can be negative at t = 0 (a SiLU bottom would clip it)
def head() -> T.Layer<HEAD(), TA.Mat<8192n, 16n>, OUT(), SH()>:
  T.Conv2d(B(), CH(), 1n, 1n, 63n, 63n, 0n, 0n, 0n)



# ---- the two levels, as one layer each -------------------------------------

def L1P() -> Type:
  T.Two<CP3(), T.LN<CH()>>

def L1S() -> Type:
  T.Two<S3(), LOW()>

def L1() -> T.Layer<L1P(), MID(), LOW(), L1S()>:
  T.Seq(~CP3(), ~T.LN<CH()>, ~MID(), ~LOW(), ~LOW(), ~S3(), ~LOW(), conv3(), gn4())

def L2P() -> Type:
  T.Two<L1P(), Unit>

def L2S() -> Type:
  T.Two<L1S(), LOW()>

def L2() -> T.Layer<L2P(), MID(), LOW(), L2S()>:
  T.Seq(~L1P(), ~Unit, ~MID(), ~LOW(), ~LOW(), ~L1S(), ~LOW(), L1(), silu4())

def L3P() -> Type:
  T.Two<L2P(), Unit>

def L3S() -> Type:
  T.Two<L2S(), Unit>

def L3() -> T.Layer<L3P(), MID(), MID(), L3S()>:
  T.Seq(~L2P(), ~Unit, ~MID(), ~LOW(), ~MID(), ~L2S(), ~Unit, L2(), upL())

def L4P() -> Type:
  T.Two<L3P(), CP3()>

def L4S() -> Type:
  T.Two<L3S(), S4()>

def L4() -> T.Layer<L4P(), MID(), MID(), L4S()>:
  T.Seq(~L3P(), ~CP3(), ~MID(), ~MID(), ~MID(), ~L3S(), ~S4(), L3(), conv4())

def L5P() -> Type:
  T.Two<L4P(), T.LN<CH()>>

def L5S() -> Type:
  T.Two<L4S(), MID()>

def L5() -> T.Layer<L5P(), MID(), MID(), L5S()>:
  T.Seq(~L4P(), ~T.LN<CH()>, ~MID(), ~MID(), ~MID(), ~L4S(), ~MID(), L4(), gn3())

def L6P() -> Type:
  T.Two<L5P(), Unit>

def L6S() -> Type:
  T.Two<L5S(), MID()>

def L6() -> T.Layer<L6P(), MID(), MID(), L6S()>:
  T.Seq(~L5P(), ~Unit, ~MID(), ~MID(), ~MID(), ~L5S(), ~MID(), L5(), silu3())

def LevelP() -> Type:
  T.Two<T.Two<Unit, L6P()>, Unit>

def LevelS() -> Type:
  T.Two<T.Two<Unit, L6S()>, Unit>

def Level() -> T.Layer<LevelP(), MID(), MID(), LevelS()>:
  T.Residual(2048n, CH(), L6P(), L6S(), L6())

# 64x64 -> 32x32 -> 16x16 -> 32x32 -> (one conv at 64x64)
def D1P() -> Type:
  T.Two<CP1(), T.LN<CH()>>

def D1S() -> Type:
  T.Two<S1(), MID()>

def D1() -> T.Layer<D1P(), IN(), MID(), D1S()>:
  T.Seq(~CP1(), ~T.LN<CH()>, ~IN(), ~MID(), ~MID(), ~S1(), ~MID(), conv1(), gn1())

def D2P() -> Type:
  T.Two<D1P(), Unit>

def D2S() -> Type:
  T.Two<D1S(), MID()>

def D2() -> T.Layer<D2P(), IN(), MID(), D2S()>:
  T.Seq(~D1P(), ~Unit, ~IN(), ~MID(), ~MID(), ~D1S(), ~MID(), D1(), silu1())

def H1P() -> Type:
  T.Two<Unit, HEAD()>

def H1S() -> Type:
  T.Two<Unit, SH()>

def H3() -> T.Layer<H1P(), MID(), OUT(), H1S()>:
  T.Seq(~Unit, ~HEAD(), ~MID(), ~TA.Mat<8192n, 16n>, ~OUT(), ~Unit, ~SH(), upM(), head())

def H1() -> T.Layer<H1P(), MID(), OUT(), H1S()>:
  H3()

def H2P() -> Type:
  H1P()

def H2S() -> Type:
  H1S()

def H2() -> T.Layer<H2P(), MID(), OUT(), H2S()>:
  H3()

def Body1P() -> Type:
  T.Two<D2P(), LevelP()>

def Body1S() -> Type:
  T.Two<D2S(), LevelS()>

def Body1() -> T.Layer<Body1P(), IN(), MID(), Body1S()>:
  T.Seq(~D2P(), ~LevelP(), ~IN(), ~MID(), ~MID(), ~D2S(), ~LevelS(), D2(), Level())

def BodyP() -> Type:
  T.Two<Body1P(), H1P()>

def BodyS() -> Type:
  T.Two<Body1S(), H1S()>

def Body() -> T.Layer<BodyP(), IN(), OUT(), BodyS()>:
  T.Seq(~Body1P(), ~H1P(), ~IN(), ~MID(), ~OUT(), ~Body1S(), ~H1S(), Body1(), H1())

def ModelP() -> Type:
  BodyP()

def ModelS() -> Type:
  BodyS()

def Model() -> T.Layer<ModelP(), IN(), OUT(), ModelS()>:
  Body()

