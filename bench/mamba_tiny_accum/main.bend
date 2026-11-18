# List vs array accumulation of the per-token in_proj gradient (the hot loop
# found in bench/plumbing_mamba_tiny). 128 tokens of a 304-wide dpre against a
# 64-wide u_t: 19,456 cells. The list version is the trainer's `outer` +
# `add_lists`; the array version adds each product in place with Array.get/set.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../../examples/language/mamba3/mamba3_siso.bend as S

def show_ms(name: String, total: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(total))))

def show_f32(tag: String, v: F32) -> IO(Unit):
  IO.print(String.append(tag, F32.show(v)))

def ones_row(+left: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ones_row(p, Con{1.0, acc})

def zeros_row(+left: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      zeros_row(p, Con{0.0, acc})

def head_or_one(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      h

def list_go(+left: Nat, +dpre: List<&2, F32>, +u: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      list_go(p, dpre, u, S.add_lists(acc, S.outer(dpre, u)))

def get_v(r: Array<F32> & F32) -> F32:
  match r:
    case Tuple{d, +v}:
      v

def inc_tuple(r: Array<F32> & F32, +idx: U32, +v: F32) -> Array<F32>:
  match r:
    case Tuple{a, +cur}:
      Array.set(F32, a, idx, (cur + v : F32))

def arr_row(+left: Nat, +off: Nat, +dpre: List<&2, F32>, +uj: F32,
    a: Array<F32>) -> Array<F32>:
  match left dpre:
    case 0n _:
      a
    case 1n+p Con{+di, dt}:
      +idx = U32.from_nat(off)
      arr_row(p, (off + 1n : Nat), dt, uj,
        inc_tuple(Array.get(F32, a, idx), idx, (di * uj : F32)))
    case _ _:
      a

def arr_cols(+left: Nat, +off: Nat, +dpre: List<&2, F32>, +u: List<&2, F32>,
    a: Array<F32>) -> Array<F32>:
  match left u:
    case 0n _:
      a
    case 1n+p Con{+uj, ut}:
      arr_cols(p, (off + 304n : Nat), dpre, ut, arr_row(304n, off, dpre, uj, a))
    case _ _:
      a

def arr_go(+left: Nat, +dpre: List<&2, F32>, +u: List<&2, F32>,
    a: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      a
    case 1n+p:
      arr_go(p, dpre, u, arr_cols(64n, 0n, dpre, u, a))

def lsq(xs: List<F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      lsq(t, (acc + (h * h : F32) : F32))

def lsum(xs: List<F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      lsum(t, (acc + h : F32))

def arr_total(a: Array<F32>) -> F32:
  lsq(Array.to_list(F32, a), 0.0)

def array_total(+dpre: List<&2, F32>, +u: List<&2, F32>, +left: Nat) -> F32:
  arr_total(arr_go(left, dpre, u, Array.new(F32, TA.cap_depth(19456n), 0.0)))

def mv_go(+left: Nat, +w: List<&2, F32>, +dpre: List<&2, F32>,
    +acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      mv_go(p, w, dpre, (acc + S.sumsq(S.matvec_t(w, dpre, 304n, 64n)) : F32))

def main() -> IO(Unit):
  do IO<Unit>:
    +dpre : List<&2, F32> <- IO.pure(List<&2, F32>, ones_row(304n, Nil{}))
    +u64 : List<&2, F32> <- IO.pure(List<&2, F32>, ones_row(64n, Nil{}))
    +z : List<&2, F32> <- IO.pure(List<&2, F32>, zeros_row(19456n, Nil{}))

    t0 : Nat <- IO.now()
    +lsumv : F32 <- IO.pure(F32, S.sumsq(list_go(128n, dpre, u64, z)))
    t1 : Nat <- IO.now()
    Unit <- show_ms("list_x128", Nat.sub(t1, t0))
    Unit <- show_f32("list_sum: ", lsumv)

    t2 : Nat <- IO.now()
    +asum : F32 <- IO.pure(F32, array_total(dpre, u64, 128n))
    t3 : Nat <- IO.now()
    Unit <- show_ms("array_x128", Nat.sub(t3, t2))
    Unit <- show_f32("array_sum: ", asum)

    t4 : Nat <- IO.now()
    +mvv : F32 <- IO.pure(F32, mv_go(128n, z, dpre, 0.0))
    t5 : Nat <- IO.now()
    Unit <- show_ms("matvec_x128", Nat.sub(t5, t4))
    Unit <- show_f32("matvec_sum: ", mvv)
    return Unit{}
