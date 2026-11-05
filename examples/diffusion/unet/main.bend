# The UNet's smallest honest U, two levels deep: 64x64 -> 32x32 -> 16x16 ->
# 32x32 -> 64x64, with two additive skips. BendTorch has Conv2d, GroupNorm, SiLU
# and FullAttention, but no transposed convolution and no channel concatenation,
# so the decoder upsamples with a parameterless layer whose forward repeats
# every pixel 2x2 and whose backward is its exact adjoint (sum the four cells
# that came from one input cell).
#
#   bendenv run -- bend examples/diffusion/unet/main.bend
#   /tmp/mlbend-unet --dump        the whole y/z/out planes, for the torch check
#
# The gates: `reference/test_up2.py` compares the upsample against NumPy,
# `reference/test_unet.py` rebuilds the two-level forward (first eight cells of
# y, z and out), and `reference/compare_unet_torch.py` checks the whole planes in
# PyTorch. `reference/unet_results.py` writes the record under
# results/diffusion-unet/; the numbers and pictures are in RESULTS.md.
#
# Adjacent facts a reader needs: the image is flat (h rows of w*c cells, row
# major), the channels are last (Conv2d's layout), and the two passes are
# vertical then horizontal, so one input pixel becomes a 2x2 block whose four
# cells all carry its value.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

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

# every run of `chunk` cells, twice; one step per run
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

# two runs to one, summed; one step per output run
def sum2c(+left: Nat, +xs: List<&2, F32>, +chunk: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+p:
      a = List.reverse(&2, F32, tk(chunk, xs, Nil{}))
      +r1 = dr(chunk, xs)
      b = List.reverse(&2, F32, tk(chunk, r1, Nil{}))
      List.append(&2, F32, add2(a, b, Nil{}), sum2c(p, dr(chunk, r1), chunk))

# rows x pixels of w*c cells, `rows` rows in all
def up_list(+rows: Nat, +w: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, F32>:
  dup(Nat.mul(Nat.mul(2n, rows), w), dup(rows, xs, Nat.mul(w, c)), c)

def down_list(+rows: Nat, +w: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, F32>:
  sum2c(rows, sum2c(Nat.mul(Nat.mul(2n, rows), w), xs, c), Nat.mul(w, c))

def list_of4(m: TA.ML<4n, 2n>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

# a 2x2 image of 2 channels, cells 1..8, so the upsample is a visible pattern
def x0() -> List<&2, F32>:
  Con{1.0, Con{2.0, Con{3.0, Con{4.0, Con{5.0, Con{6.0, Con{7.0, Con{8.0, Nil{}}}}}}}}}


# ---- the model: one down, one up, one additive skip ----------------------
#   x    (64x64, 2)          the image, two channels
#   y    conv1 stride 2, GN, SiLU      (32x32, 2)
#   u    Up2(y)                        (64x64, 2)
#   z    conv2 stride 1, GN, SiLU      (64x64, 2)
#   out  z + x
# Conv2d(n, c, co, k, hs, ws, p, s1, par): the image is the kernel plus a
# surplus, so at 64x64 with k=3, s=2, p=1 (hs = ws = 63) the output is 32x32.
def IN() -> Type:
  TA.Mat<4096n, 2n>

def MID() -> Type:
  TA.Mat<1024n, 2n>

def CP() -> Type:
  T.Lin<18n, 2n>

def CST1() -> Type:
  TA.Mat<1024n, 18n>

def CST2() -> Type:
  TA.Mat<4096n, 18n>

# the second level: 32x32 down to 16x16 and back (ISSUE-061)
def LOW() -> Type:
  TA.Mat<256n, 2n>

def CST3() -> Type:
  TA.Mat<256n, 18n>

type P is Type:
  P{c1: CP(), g1: T.LN<2n>, c3: CP(), g3: T.LN<2n>, c4: CP(), g4: T.LN<2n>, c2: CP(), g2: T.LN<2n>}

def list_mid(m: TA.ML<1024n, 2n>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def list_in(m: TA.ML<4096n, 2n>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def mid_of(f: T.Fwd<CP(), MID(), CST1()>) -> MID():
  match f:
    case T.Fwd{p, y, s}:
      y

def list_low(m: TA.ML<256n, 2n>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def low_of(f: T.Fwd<CP(), LOW(), CST3()>) -> LOW():
  match f:
    case T.Fwd{p, y, s}:
      y

def glmid_of(f: T.Fwd<T.LN<2n>, MID(), MID()>) -> MID():
  match f:
    case T.Fwd{p, y, s}:
      y

def glof(f: T.Fwd<T.LN<2n>, LOW(), LOW()>) -> LOW():
  match f:
    case T.Fwd{p, y, s}:
      y

def gl_conv(f: T.Fwd<CP(), MID(), CST1()>) -> MID():
  match f:
    case T.Fwd{p, y, s}:
      y

def slof(f: T.Fwd<Unit, LOW(), LOW()>) -> LOW():
  match f:
    case T.Fwd{p, y, s}:
      y

def smid2_of(f: T.Fwd<Unit, MID(), MID()>) -> MID():
  match f:
    case T.Fwd{p, y, s}:
      y

def in_of(f: T.Fwd<CP(), IN(), CST2()>) -> IN():
  match f:
    case T.Fwd{p, y, s}:
      y

def gmid_of(f: T.Fwd<T.LN<2n>, MID(), MID()>) -> MID():
  match f:
    case T.Fwd{p, y, s}:
      y

def gin_of(f: T.Fwd<T.LN<2n>, IN(), IN()>) -> IN():
  match f:
    case T.Fwd{p, y, s}:
      y

def smid_of(f: T.Fwd<Unit, MID(), MID()>) -> MID():
  match f:
    case T.Fwd{p, y, s}:
      y

def sin_of(f: T.Fwd<Unit, IN(), IN()>) -> IN():
  match f:
    case T.Fwd{p, y, s}:
      y

def conv1(p: CP(), x: IN()) -> MID():
  mid_of(T.Layer.fwd(CP(), IN(), MID(), CST1(), T.Conv2d(1n, 2n, 2n, 3n, 63n, 63n, 1n, 1n, 0n), p, x))

# stride 1 is s1 = 0 (the layer's stride is 1 + s1), the same 63 surplus
# keeps the 64x64 shape
def conv2(p: CP(), x: IN()) -> IN():
  in_of(T.Layer.fwd(CP(), IN(), IN(), CST2(), T.Conv2d(1n, 2n, 2n, 3n, 63n, 63n, 1n, 0n, 0n), p, x))

def gn1(g: T.LN<2n>, y: MID()) -> MID():
  gmid_of(T.Layer.fwd(T.LN<2n>, MID(), MID(), MID(), T.GroupNorm(1n, 1024n, 1n, 2n, 1.0), g, y))

def gn2(g: T.LN<2n>, y: IN()) -> IN():
  gin_of(T.Layer.fwd(T.LN<2n>, IN(), IN(), IN(), T.GroupNorm(1n, 4096n, 1n, 2n, 1.0), g, y))

def silu_mid(y: MID()) -> MID():
  smid_of(T.Layer.fwd(Unit, MID(), MID(), MID(), T.SiLU(1024n, 2n), Unit{}, y))

def silu_in(y: IN()) -> IN():
  sin_of(T.Layer.fwd(Unit, IN(), IN(), IN(), T.SiLU(4096n, 2n), Unit{}, y))

# ---- the second level (ISSUE-061) ----
# down 2: 32x32 -> 16x16 with k=3, p=1, stride 2 (hs = ws = 31, so cimg = 32)
def conv3(p: CP(), x: MID()) -> LOW():
  low_of(T.Layer.fwd(CP(), MID(), LOW(), CST3(), T.Conv2d(1n, 2n, 2n, 3n, 31n, 31n, 1n, 1n, 0n), p, x))

# up 2: 16x16 -> 32x32, then a same-shape conv (k=3, p=1, stride 1)
def up_l(y: LOW()) -> MID():
  TA.Mat.from_list(1024n, 2n, up_list(16n, 16n, 2n, list_low(TA.Mat.to_list(256n, 2n, y))))

def conv4(p: CP(), x: MID()) -> MID():
  gl_conv(T.Layer.fwd(CP(), MID(), MID(), CST1(), T.Conv2d(1n, 2n, 2n, 3n, 31n, 31n, 1n, 0n, 0n), p, x))

def gn3(g: T.LN<2n>, y: MID()) -> MID():
  glmid_of(T.Layer.fwd(T.LN<2n>, MID(), MID(), MID(), T.GroupNorm(1n, 1024n, 1n, 2n, 1.0), g, y))

def gn4(g: T.LN<2n>, y: LOW()) -> LOW():
  glof(T.Layer.fwd(T.LN<2n>, LOW(), LOW(), LOW(), T.GroupNorm(1n, 256n, 1n, 2n, 1.0), g, y))

def silu_low(y: LOW()) -> LOW():
  slof(T.Layer.fwd(Unit, LOW(), LOW(), LOW(), T.SiLU(256n, 2n), Unit{}, y))

def silu_mid2(y: MID()) -> MID():
  smid2_of(T.Layer.fwd(Unit, MID(), MID(), MID(), T.SiLU(1024n, 2n), Unit{}, y))

def up_m(y: MID()) -> IN():
  TA.Mat.from_list(4096n, 2n, up_list(32n, 32n, 2n, list_mid(TA.Mat.to_list(1024n, 2n, y))))

# the 32x32 stage needs y twice (the second level and its skip), and a Mat is linear: `Mat.dup` hands both
# copies to a def that matches the pair, the same trick as `Fork` and the input's skip
def mid_pair(c3: CP(), g3: T.LN<2n>, c4: CP(), g4: T.LN<2n>, d: T.Two<MID(), MID()>) -> MID():
  match d:
    case T.Two{a, b}:
      T.Mat.sum2(1024n, 2n, silu_mid2(gn3(g3, conv4(c4, up_l(silu_low(gn4(g4, conv3(c3, a))))))), b)

def fwd_pair(p: P(), d: T.Two<IN(), IN()>) -> IN():
  match p d:
    case P{c1, g1, c3, g3, c4, g4, c2, g2} T.Two{a, b}:
      T.Mat.sum2(4096n, 2n, silu_in(gn2(g2, conv2(c2, up_m(mid_pair(c3, g3, c4, g4, T.Mat.dup(1024n, 2n, silu_mid(gn1(g1, conv1(c1, a))))))))), b)

def fwd(p: P(), x: IN()) -> IN():
  fwd_pair(p, T.Mat.dup(4096n, 2n, x))

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

def up0() -> List<&2, F32>:
  up_list(2n, 2n, 2n, x0())

def dot(+xs: List<&2, F32>, +ys: List<&2, F32>, +acc: F32) -> F32:
  match xs ys:
    case Con{a, at} Con{b, bt}:
      dot(at, bt, (acc + (a * b : F32) : F32))
    case _ _:
      acc

def verdict(same: Bool) -> String:
  match same:
    case True{}:
      "PASS"
    case False{}:
      "FAIL"


# ---- deterministic weights, and the forward's first numbers ---------------
# Every cell is 0.01 * (its index + 1), so the kernel varies per position and
# the im2col order is visible in the result. The twin computes the same values.
# alternating signs: a smooth ramp made the two output channels of the first
# conv nearly proportional, so GroupNorm divided by a variance near zero and
# amplified everything. Alternating keeps the two channels independent.
def signed(+i: Nat) -> F32:
  +v = (0.01 * F32.from_nat(i) : F32)
  Bool.pick(F32, Nat.is_eq(Nat.mod(i, 2n), 0n), v, (0.0 - v : F32))

def gen(+left: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      gen(p, (1n + i : Nat), Con{signed(i), acc})

def mat(+r: Nat, +c: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, gen(Nat.mul(r, c), 0n, Nil{}))

# both scales non-zero and moderate: `gen` would give [0, -0.01], and a zero
# scale makes that channel's GroupNorm output independent of the conv, so it
# could not test anything (2026-11-04)
def lnp(s: Nat) -> T.LN<2n>:
  T.LN{TA.Mat.fill(1n, 2n, 0.5), TA.Mat.fill(1n, 2n, (0.5 + 0.0 * F32.from_nat(s) : F32))}

def params() -> P():
  P{T.Lin{mat(2n, 18n), TA.Mat.fill(1n, 2n, 0.05)}, lnp(0n),
    T.Lin{mat(2n, 18n), TA.Mat.fill(1n, 2n, 0.06)}, lnp(1n),
    T.Lin{mat(2n, 18n), TA.Mat.fill(1n, 2n, 0.07)}, lnp(2n),
    T.Lin{mat(2n, 18n), TA.Mat.fill(1n, 2n, 0.08)}, lnp(3n)}

def x_in() -> IN():
  mat(4096n, 2n)

def head8(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Con{a, Con{b, Con{c, Con{d, Con{e, Con{f, Con{g, Con{h, t}}}}}}}}:
      Con{a, Con{b, Con{c, Con{d, Con{e, Con{f, Con{g, Con{h, Nil{}}}}}}}}}
    case _:
      Nil{}

def show_go4(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go4(t)))

def gn_of_params(p: P()) -> String:
  match p:
    case P{c1, g1, c3, g3, c4, g4, c2, g2}:
      show_go4(head8(list_mid(TA.Mat.to_list(1024n, 2n, gn1(g1, conv1(c1, x_in()))))))

def gn8() -> String:
  gn_of_params(params())

def raw_of_params(p: P()) -> String:
  match p:
    case P{c1, g1, c3, g3, c4, g4, c2, g2}:
      show_go4(head8(list_mid(TA.Mat.to_list(1024n, 2n, conv1(c1, x_in())))))

def raw8() -> String:
  raw_of_params(params())

def z_of_params(p: P()) -> String:
  match p:
    case P{c1, g1, c3, g3, c4, g4, c2, g2}:
      show_go4(head8(list_mid(TA.Mat.to_list(1024n, 2n, mid_pair(c3, g3, c4, g4, T.Mat.dup(1024n, 2n,
        silu_mid(gn1(g1, conv1(c1, x_in())))))))))

def z8() -> String:
  z_of_params(params())

def y_of_params(p: P()) -> String:
  match p:
    case P{c1, g1, c3, g3, c4, g4, c2, g2}:
      show_go4(head8(list_mid(TA.Mat.to_list(1024n, 2n, silu_mid(gn1(g1, conv1(c1, x_in())))))))

def y8() -> String:
  y_of_params(params())

def out8() -> String:
  show_go4(head8(list_in(TA.Mat.to_list(4096n, 2n, fwd(params(), x_in())))))

def dump_go(args: List<String>, +dump: Bool) -> Bool:
  match args:
    case Nil{}:
      dump
    case Con{a, rest}:
      match a:
        case "--dump":
          dump_go(rest, True{})
        case _:
          dump_go(rest, dump)

def dump_of(args: List<String>) -> Bool:
  match args:
    case Nil{}:
      False{}
    case Con{program, rest}:
      dump_go(rest, False{})

# `--dump` prints the whole planes, one line each, so the PyTorch comparison covers 8,192 numbers of
# `out` and 2,048 of each intermediate instead of eight; the eight-cell lines above stay the gates
# that localise a mismatch, because a full-plane diff says only "somewhere"
def dump_y(p: P()) -> IO(Unit):
  match p:
    case P{c1, g1, c3, g3, c4, g4, c2, g2}:
      IO.print(String.append("y_all: ",
        show_go(list_mid(TA.Mat.to_list(1024n, 2n, silu_mid(gn1(g1, conv1(c1, x_in()))))))))

def dump_z(p: P()) -> IO(Unit):
  match p:
    case P{c1, g1, c3, g3, c4, g4, c2, g2}:
      IO.print(String.append("z_all: ",
        show_go(list_mid(TA.Mat.to_list(1024n, 2n, mid_pair(c3, g3, c4, g4,
          T.Mat.dup(1024n, 2n, silu_mid(gn1(g1, conv1(c1, x_in()))))))))))

def dump_out(p: P()) -> IO(Unit):
  match p:
    case P{c1, g1, c3, g3, c4, g4, c2, g2}:
      IO.print(String.append("out_all: ",
        show_go(list_in(TA.Mat.to_list(4096n, 2n, fwd(p, x_in()))))))

def dump_all() -> IO(Unit):
  do IO<Unit>:
    Unit <- dump_y(params())
    Unit <- dump_z(params())
    dump_out(params())

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    Unit <- IO.print(String.append("in cells: ", Nat.show(List.length(&2, F32, x0()))))
    Unit <- IO.print(String.append("up cells: ", Nat.show(List.length(&2, F32, up0()))))
    Unit <- IO.print(String.append("down cells: ", Nat.show(List.length(&2, F32, down_list(2n, 2n, 2n, up0())))))
    Unit <- IO.print(String.append("up: ", show_go(up0())))
    Unit <- IO.print(String.append("left:  ", F32.show(dot(up0(), up0(), 0.0))))
    Unit <- IO.print(String.append("right: ", F32.show(dot(x0(), down_list(2n, 2n, 2n, up0()), 0.0))))
    Unit <- IO.print(String.append("conv1: ", raw8()))
    Unit <- IO.print(String.append("gn: ", gn8()))
    Unit <- IO.print(String.append("z: ", z8()))
    Unit <- IO.print(String.append("y: ", y8()))
    Unit <- IO.print(String.append("out: ", out8()))
    Bool.pick(IO(Unit), dump_of(args), dump_all(), IO.pure(Unit, Unit{}))
    IO.print(String.append("adjoint: ",
      verdict(F32.is_eq(dot(up0(), up0(), 0.0), dot(x0(), down_list(2n, 2n, 2n, up0()), 0.0)))))
