# Feasibility: can tensor-gpu@0.1.5.0's counter fill cross into
# tensor-array@0.1.14.0's Mat?
#
#   bendenv run -- bend bench/fill_cross/main.bend
#
# The clash: `examples/cifar` builds its noise with
# `TA.Mat.from_list(n(), d(), G.noise_of(s))` — 786,432 `G.gauss_at` calls and
# a list — on tensor-array@0.1.14.0, while `bend-ml-tensor-gpu@0.1.5.0` (the
# package carrying bend-ml#23's fills) imports tensor-array@0.1.18.0. A Mat is
# the package's own type, so a Mat of 0.1.18 cannot be passed where 0.1.14 is
# expected. What CAN cross is `fill_gauss`'s return: a Base `Array<F32>`, which
# belongs to no package's stack.
#
# So: eight values from `fill_gauss(1, 8, cap_depth(8))`, the same array
# wrapped as a 0.1.14 `Mat<2, 4>`, the eight values printed beside
# `G.gauss_at(1, i)`, the largest difference, and one matmul on the wrapped
# Mat. `fill_gauss` draws Box-Muller over rng_ln/rng_sc (polynomials, the same
# bits on every lane); `gauss_at` calls the host's F32.log/cos/sin, so the two
# agree to a few ulp, not bit for bit. The task's gate is 1.3e-6.
import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import bend-ml-tensor-gpu@0.1.5.0/main.bend as TG
import mlbend-generative@0.1.1.0/main.bend as G

def cells() -> Nat:
  8n

def cap() -> Nat:
  3n

def counter() -> U32:
  1

# the fill: n Gaussians of counter s, in an array of 2^c cells, made in one bang
def filled() -> Array<F32>:
  TG.fill_gauss(counter(), cells(), cap())

# the same array as this project's Mat: the constructor takes any Array<F32>
def wrapped() -> TA.Mat<2n, 4n>:
  TA.Mat{filled()}

# Array.get hands the array back beside the cell, so the walk carries the pair
def show_go(left: Nat, +i: Nat, r: Array<F32> & F32, +acc: String) -> String:
  match left r:
    case 0n _:
      acc
    case 1n+rest Tuple{a, +x}:
      show_go(rest, (1n + i : Nat), Array.get(F32, a, U32.from_nat((1n + i : Nat))),
        String.append(acc, String.append(" ", F32.show(x))))

def show_arr(arr: Array<F32>) -> String:
  show_go(cells(), 0n, Array.get(F32, arr, 0), "")

# the reference values, one gauss_at a cell
def ref_go(left: Nat, +i: Nat, +acc: String) -> String:
  match left:
    case 0n:
      acc
    case 1n+rest:
      ref_go(rest, (1n + i : Nat), String.append(acc, String.append(" ", F32.show(G.gauss_at(1n, i)))))

# the largest |fill_gauss(1, i) - gauss_at(1, i)|
def diff_go(left: Nat, +i: Nat, r: Array<F32> & F32, +worst: F32) -> F32:
  match left r:
    case 0n _:
      worst
    case 1n+rest Tuple{a, +x}:
      diff_go(rest, (1n + i : Nat), Array.get(F32, a, U32.from_nat((1n + i : Nat))),
        F32.max(worst, F32.abs(F32.sub(x, G.gauss_at(1n, i)))))

def max_diff(arr: Array<F32>) -> F32:
  diff_go(cells(), 0n, Array.get(F32, arr, 0), 0.0)

def list_of(left: Nat, xs: List<&2, F32>, +acc: String) -> String:
  match left xs:
    case 0n Nil{}:
      acc
    case 0n Con{h, t}:
      acc
    case 1n+rest Con{h, t}:
      list_of(rest, t, String.append(acc, String.append(" ", F32.show(h))))
    case 1n+rest Nil{}:
      acc

# a second fill for the right-hand side, 4 x 2
def wrapped2() -> TA.Mat<4n, 2n>:
  TA.Mat{TG.fill_gauss(2, cells(), cap())}

def un_ml(ml: TA.ML<2n, 2n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, l}:
      l

def un_mm(mm: TA.MMul<2n, 4n, 2n>) -> List<&2, F32>:
  match mm:
    case TA.MMul{a, b, c}:
      un_ml(TA.Mat.to_list(2n, 2n, c))

# one product on the wrapped Mats: 2 x 4 times 4 x 2
def product() -> List<&2, F32>:
  un_mm(TA.Mat.matmul(2n, 4n, 2n, 1n, wrapped(), wrapped2()))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("fill_gauss:", show_arr(filled())))
    Unit <- IO.print(String.append("gauss_at:", ref_go(cells(), 0n, "")))
    Unit <- IO.print(String.append("max_diff: ", F32.show(max_diff(filled()))))
    IO.print(String.append("matmul:", list_of(4n, product(), "")))

# the array as a reusable list: the type tells whether to_list yields List<&2, F32>
def lens(xs: List<&2, F32>) -> Nat:
  List.length(&2, F32, xs)

def check_list() -> Nat:
  lens(Array.to_list(F32, filled()))
