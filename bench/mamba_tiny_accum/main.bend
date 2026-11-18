# List vs array accumulation of the per-token in_proj gradient (the hot loop
# found in bench/plumbing_mamba_tiny). 128 tokens of a 304-wide dpre against a
# 64-wide u_t: 19,456 cells. The list version is the trainer's `outer` +
# `add_lists`; the array version is mamba3_siso's `acc_outer`, the same
# arithmetic in place. The last block walks both results cell by cell and
# prints the sum of the squared differences: it must be exactly 0.0, which is
# what says the array layout and the add order are the list's, not merely a
# permutation with the same total.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../../examples/language/mamba3/mamba3_siso.bend as S

def show_ms(name: String, total: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(total))))

def show_f32(tag: String, v: F32) -> IO(Unit):
  IO.print(String.append(tag, F32.show(v)))

# distinct values, so a transposed or shifted layout cannot pass the diff
def ramp(+left: Nat, +v: F32, +step: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ramp(p, (v + step : F32), step, Con{v, acc})

def zeros_row(+left: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      zeros_row(p, Con{0.0, acc})

def lsq(xs: List<F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      lsq(t, (acc + (h * h : F32) : F32))

def list_go(+left: Nat, +dpre: List<&2, F32>, +u: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      list_go(p, dpre, u, S.add_lists(acc, S.outer(dpre, u)))

def arr_go(+left: Nat, +dpre: List<&2, F32>, +u: List<&2, F32>,
    a: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      a
    case 1n+p:
      arr_go(p, dpre, u, S.acc_outer(dpre, u, a))

def list_run(+dpre: List<&2, F32>, +u: List<&2, F32>, +left: Nat) -> F32:
  S.sumsq(list_go(left, dpre, u, zeros_row(19456n, Nil{})))

# accumulation + the one conversion the trainer pays + the check sum
def array_run(+dpre: List<&2, F32>, +u: List<&2, F32>, +left: Nat) -> F32:
  S.sumsq(S.arr_dW(19456n,
    arr_go(left, dpre, u, Array.new(F32, TA.cap_depth(19456n), 0.0))))

# the conversion alone, for its share of the array figure
def convert_run(+left: Nat) -> F32:
  S.sumsq(S.arr_dW(19456n, Array.new(F32, TA.cap_depth(19456n), 0.0)))

def cell_diff(+left: Nat, xs: List<&2, F32>, ys: List<&2, F32>,
    acc: F32) -> F32:
  match left xs ys:
    case 0n _ _:
      acc
    case 1n+p Con{+a, at} Con{+b, bt}:
      +d = (a - b : F32)
      cell_diff(p, at, bt, (acc + (d * d : F32) : F32))
    case _ _ _:
      acc

# the list trainer and the array trainer on the same inputs, cell by cell
def list_vs_array(+dpre: List<&2, F32>, +u: List<&2, F32>,
    +toks: Nat) -> F32:
  cell_diff(19456n, list_go(toks, dpre, u, zeros_row(19456n, Nil{})),
    S.arr_dW(19456n,
      arr_go(toks, dpre, u, Array.new(F32, TA.cap_depth(19456n), 0.0))),
    0.0)

def main() -> IO(Unit):
  do IO<Unit>:
    +dpre : List<&2, F32> <- IO.pure(List<&2, F32>, ramp(304n, 1.0, 1.0, Nil{}))
    +u64 : List<&2, F32> <- IO.pure(List<&2, F32>, ramp(64n, 0.25, 0.25, Nil{}))

    t0 : Nat <- IO.now()
    +lsumv : F32 <- IO.pure(F32, list_run(dpre, u64, 128n))
    t1 : Nat <- IO.now()
    Unit <- show_ms("list_x128", Nat.sub(t1, t0))
    Unit <- show_f32("list_sum: ", lsumv)

    t2 : Nat <- IO.now()
    +asum : F32 <- IO.pure(F32, array_run(dpre, u64, 128n))
    t3 : Nat <- IO.now()
    Unit <- show_ms("array_x128", Nat.sub(t3, t2))
    Unit <- show_f32("array_sum: ", asum)

    t4 : Nat <- IO.now()
    +csum : F32 <- IO.pure(F32, convert_run(19456n))
    t5 : Nat <- IO.now()
    Unit <- show_ms("convert_only", Nat.sub(t5, t4))
    Unit <- show_f32("convert_sum: ", csum)

    +cdiff : F32 <- IO.pure(F32, list_vs_array(dpre, u64, 128n))
    Unit <- show_f32("cells_diff: ", cdiff)
    return Unit{}
