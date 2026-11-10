# The sparse attention core, vectorized: a dense T x T score matrix, the
# reference's window-16 / stride-16 causal mask, softmax, and matmuls for the
# forward and every gradient.  The gathered per-query list version stays in
# `attn_c.bend` as the oracle.
#
# `valid(t, s)` is the reference's `_indices`: s <= t, and either an anchor
# (s % 16 == 0) or inside the local window (t - s < 16).
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# ---- result projections: a match must scrutinize a parameter, not a call ----

def mmul_c(-n: Nat, -k: Nat, -m: Nat, r: TA.MMul<n, k, m>) -> TA.Mat<n, m>:
  match r:
    case TA.MMul{a, b, c}:
      c

def mmulnt_c(-n: Nat, -k: Nat, -m: Nat, r: TA.MMulNT<n, k, m>) -> TA.Mat<n, m>:
  match r:
    case TA.MMulNT{a, b, c}:
      c

def mmultn_c(-n: Nat, -k: Nat, -m: Nat, r: TA.MMulTN<n, k, m>) -> TA.Mat<n, m>:
  match r:
    case TA.MMulTN{a, b, c}:
      c

def mpair_a(-r: Nat, -c: Nat, p: TA.MPair<r, c>) -> TA.Mat<r, c>:
  match p:
    case TA.MPair{a, b}:
      a

def fwd_y(-P: Type, -Y: Type, -S: Type, r: T.Fwd<P, Y, S>) -> Y:
  match r:
    case T.Fwd{p, y, s}:
      y

def bwd_dx(-P: Type, -X: Type, r: T.Bwd<P, X>) -> X:
  match r:
    case T.Bwd{p, dx, g}:
      dx

# ---- the mask ----

def valid(+t: Nat, +s: Nat) -> Bool:
  +le = Nat.is_le(s, t)
  +anchor = Nat.is_eq(Nat.mod(s, 16n), 0n)
  +local = Nat.is_lt(Nat.sub(t, s), 16n)
  Bool.and(le, Bool.or(anchor, local))

def mask_row(+left: Nat, +t: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +v = Bool.pick(F32, valid(t, p), 0.0, (0.0 - 1.0e30 : F32))
      mask_row(p, t, Con{v, acc})

def mask_rows(+n: Nat, +left: Nat, +t: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      mask_rows(n, p, (t + 1n : Nat), List.append(&2, F32, acc, mask_row(n, t, Nil{})))

def mask_mat(+n: Nat) -> TA.Mat<n, n>:
  TA.Mat.from_list(n, n, mask_rows(n, n, 0n, Nil{}))

# ---- element-wise scale ----

def scale_one(+i: U32, +c: F32, av: Array<F32> & F32) -> Array<F32>:
  match av:
    case Tuple{a, +v}:
      Array.set(F32, a, i, (v * c : F32))

def scale_arr(+left: Nat, +i: U32, +c: F32, a: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      a
    case 1n+p:
      a2 = scale_one(i, c, Array.get(F32, a, i))
      scale_arr(p, (i + 1 : U32), c, a2)

def scale_mat(+r: Nat, +c: Nat, m: TA.Mat<r, c>, +s: F32) -> TA.Mat<r, c>:
  match m:
    case TA.Mat{a}:
      TA.Mat{scale_arr(Nat.mul(r, c), 0, s, a)}

def sc_of(+hd: Nat) -> F32:
  (1.0 / F32.sqrt(F32.from_nat(hd)) : F32)

# ---- forward and backward ----

def softmax_fwd(+n: Nat, m: TA.Mat<n, n>) -> TA.Mat<n, n>:
  fwd_y(Unit, TA.Mat<n, n>, TA.Mat<n, n>,
    T.Layer.fwd(Unit, TA.Mat<n, n>, TA.Mat<n, n>, TA.Mat<n, n>, T.Softmax(n, n), Unit{}, m))

def softmax_bwd(+n: Nat, p: TA.Mat<n, n>, dp: TA.Mat<n, n>) -> TA.Mat<n, n>:
  bwd_dx(Unit, TA.Mat<n, n>,
    T.Layer.bwd(Unit, TA.Mat<n, n>, TA.Mat<n, n>, TA.Mat<n, n>, T.Softmax(n, n), Unit{}, p, dp, Unit{}))

def ds_of(+n: Nat, -hd: Nat, r: TA.MMulNT<n, hd, n>, p: TA.Mat<n, n>) -> TA.Mat<n, n>:
  match r:
    case TA.MMulNT{a, b, dp}:
      softmax_bwd(n, p, dp)

def p_of(+n: Nat, +hd: Nat, q: TA.Mat<n, hd>, k: TA.Mat<n, hd>,
    mask: TA.Mat<n, n>) -> TA.Mat<n, n>:
  softmax_fwd(n, mpair_a(n, n, TA.Mat.add(n, n,
    mmulnt_c(n, hd, n, TA.Mat.matmul_nt(n, hd, n, 0n, q, k)), mask)))

def y_of(+n: Nat, +hd: Nat, q: TA.Mat<n, hd>, k: TA.Mat<n, hd>,
    v: TA.Mat<n, hd>, mask: TA.Mat<n, n>) -> TA.Mat<n, hd>:
  +sc = sc_of(hd)
  mmul_c(n, n, hd, TA.Mat.matmul(n, n, hd, 0n,
    p_of(n, hd, scale_mat(n, hd, q, sc), k, mask), v))

type DBW<-n: Nat, -hd: Nat> is Type:
  DBW{dq: TA.Mat<n, hd>, dk: TA.Mat<n, hd>, dv: TA.Mat<n, hd>}

def mul_left(+n: Nat, +hd: Nat, d: TA.Mat<n, n>, k: TA.Mat<n, hd>) -> TA.Mat<n, hd>:
  mmul_c(n, n, hd, TA.Mat.matmul(n, n, hd, 0n, d, k))

def mul_tn(+n: Nat, +hd: Nat, d: TA.Mat<n, n>, q: TA.Mat<n, hd>) -> TA.Mat<n, hd>:
  mmultn_c(n, n, hd, TA.Mat.matmul_tn(n, n, hd, 0n, d, q))

def bwd_fin(+n: Nat, +hd: Nat, p2: TA.Mat<n, n>, q2: TA.Mat<n, hd>, k2: TA.Mat<n, hd>,
    dy2: TA.Mat<n, hd>, dd: T.Two<TA.Mat<n, n>, TA.Mat<n, n>>) -> DBW<n, hd>:
  match dd:
    case T.Two{d1, d2}:
      +sc = sc_of(hd)
      DBW{scale_mat(n, hd, mul_left(n, hd, d1, k2), sc),
          mul_tn(n, hd, d2, q2),
          mul_tn(n, hd, p2, dy2)}

def bwd_p(+n: Nat, +hd: Nat, pd: T.Two<TA.Mat<n, n>, TA.Mat<n, n>>,
    q2: TA.Mat<n, hd>, k2: TA.Mat<n, hd>, dy1: TA.Mat<n, hd>, dy2: TA.Mat<n, hd>,
    v: TA.Mat<n, hd>) -> DBW<n, hd>:
  match pd:
    case T.Two{p1, p2}:
      ds = ds_of(n, hd, TA.Mat.matmul_nt(n, hd, n, 0n, dy1, v), p1)
      bwd_fin(n, hd, p2, q2, k2, dy2, T.Mat.dup(n, n, ds))

def bwd_start(+n: Nat, +hd: Nat, qd: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>,
    kd: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>,
    dyd: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>, v: TA.Mat<n, hd>,
    mask: TA.Mat<n, n>) -> DBW<n, hd>:
  match qd kd dyd:
    case T.Two{q1, q2} T.Two{k1, k2} T.Two{dy1, dy2}:
      p = p_of(n, hd, q1, k1, mask)
      bwd_p(n, hd, T.Mat.dup(n, n, p), q2, k2, dy1, dy2, v)

def bwd(+n: Nat, +hd: Nat, dy: TA.Mat<n, hd>, q: TA.Mat<n, hd>, k: TA.Mat<n, hd>,
    v: TA.Mat<n, hd>, mask: TA.Mat<n, n>) -> DBW<n, hd>:
  +sc = sc_of(hd)
  bwd_start(n, hd, T.Mat.dup(n, hd, scale_mat(n, hd, q, sc)), T.Mat.dup(n, hd, k),
    T.Mat.dup(n, hd, dy), v, mask)
