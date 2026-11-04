# The UNet's first building block, and the check that it is the layer a UNet
# needs: BendTorch has Conv2d, GroupNorm, SiLU and FullAttention, but no
# transposed convolution and no channel concatenation. So the decoder upsamples
# with a parameterless layer whose forward repeats every pixel 2x2 and whose
# backward is its exact adjoint (sum the four cells that came from one input
# cell). `main` prints the adjoint identity for a 2x2 image of 2 channels, and
# `reference/test_up2.py` compares the forward against NumPy.
#
#   bendenv run -- bend examples/diffusion/unet/main.bend
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

def list_of(m: TA.ML<4n, 2n>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

# a 2x2 image of 2 channels, cells 1..8, so the upsample is a visible pattern
def x0() -> List<&2, F32>:
  Con{1.0, Con{2.0, Con{3.0, Con{4.0, Con{5.0, Con{6.0, Con{7.0, Con{8.0, Nil{}}}}}}}}}

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

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("in cells: ", Nat.show(List.length(&2, F32, x0()))))
    Unit <- IO.print(String.append("up cells: ", Nat.show(List.length(&2, F32, up0()))))
    Unit <- IO.print(String.append("down cells: ", Nat.show(List.length(&2, F32, down_list(2n, 2n, 2n, up0())))))
    Unit <- IO.print(String.append("up: ", show_go(up0())))
    Unit <- IO.print(String.append("left:  ", F32.show(dot(up0(), up0(), 0.0))))
    Unit <- IO.print(String.append("right: ", F32.show(dot(x0(), down_list(2n, 2n, 2n, up0()), 0.0))))
    IO.print(String.append("adjoint: ",
      verdict(F32.is_eq(dot(up0(), up0(), 0.0), dot(x0(), down_list(2n, 2n, 2n, up0()), 0.0)))))
    return Unit{}
