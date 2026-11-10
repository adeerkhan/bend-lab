# The causal linear attention core, dense: q~ = ELU(q)+1, k~ = ELU(k)+1,
# S = q~ k~^T masked to s <= t, den = row sums of the masked S,
# num = S V, y = num / clamp(den, 1e-6).  The per-position prefix in
# `lin_fast`/`lin_bwd` stays as the oracle.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./lin_bwd.bend as L
import ./sp_dense.bend as SD

# ---- elementwise maps ----

def elu1_f(+x: F32) -> F32:
  Bool.pick(F32, F32.is_le(x, 0.0), F32.exp(x), (x + 1.0 : F32))

# the derivative from ELU(x)+1: x if x <= 1 (the exp side), 1 above it
def deriv_elu1_f(+x: F32) -> F32:
  Bool.pick(F32, F32.is_le(x, 1.0), x, 1.0)

def deriv_f(+x: F32) -> F32:
  Bool.pick(F32, F32.is_le(x, 0.0), F32.exp(x), 1.0)

def map_set(~f: @+x: F32 -> F32, +i: U32, av: Array<F32> & F32) -> Array<F32>:
  match av:
    case Tuple{a, +x}:
      Array.set(F32, a, i, f(x))

def map_arr(~f: @+x: F32 -> F32, +left: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      a
    case 1n+p:
      a2 = map_set(~f, i, Array.get(F32, a, i))
      map_arr(~f, p, (i + 1 : U32), a2)

def map_mat(~f: @+x: F32 -> F32, +r: Nat, +c: Nat, m: TA.Mat<r, c>) -> TA.Mat<r, c>:
  match m:
    case TA.Mat{a}:
      TA.Mat{map_arr(~f, Nat.mul(r, c), 0, a)}

# ---- causal 1/0 mask, ones, and the row helpers ----

def causal_row(+left: Nat, +t: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +v = Bool.pick(F32, Nat.is_le(p, t), 1.0, 0.0)
      causal_row(p, t, Con{v, acc})

def causal_rows(+n: Nat, +left: Nat, +t: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      causal_rows(n, p, (t + 1n : Nat), List.append(&2, F32, acc, causal_row(n, t, Nil{})))

def causal_mat(+n: Nat) -> TA.Mat<n, n>:
  TA.Mat.from_list(n, n, causal_rows(n, n, 0n, Nil{}))

def ones_list(+left: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ones_list(p, Con{1.0, acc})

def one_mat(+n: Nat) -> TA.Mat<n, 1n>:
  TA.Mat.from_list(n, 1n, ones_list(n, Nil{}))

def ml1(+n: Nat, ml: TA.ML<n, 1n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def mlhd(+n: Nat, +hd: Nat, ml: TA.ML<n, hd>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def mul2_fwd(+n: Nat, +c: Nat, a: TA.Mat<n, c>, b: TA.Mat<n, c>) -> TA.Mat<n, c>:
  SD.fwd_y(Unit, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>,
    T.Layer.fwd(Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>,
      T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, T.Mul2(n, c), Unit{}, T.Two{a, b}))

def clamp_one(+x: F32) -> F32:
  F32.max(x, 0.000001)

def clamp_list(xs: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      List.reverse(&2, F32, acc)
    case Con{h, t}:
      clamp_list(t, Con{clamp_one(h), acc})

def row_sums(+n: Nat, masked: TA.Mat<n, n>) -> List<&2, F32>:
  denm = SD.mmul_c(n, n, 1n, TA.Mat.matmul(n, n, 1n, 0n, masked, one_mat(n)))
  clamp_list(ml1(n, TA.Mat.to_list(n, 1n, denm)), Nil{})

# ---- divide each row by its denominator ----

def div_set(+hd: Nat, +i: U32, +dens: List<&2, F32>, av: Array<F32> & F32) -> Array<F32>:
  match av:
    case Tuple{a, +x}:
      +r = Nat.div(U32.to_nat(i), hd)
      Array.set(F32, a, i, (x / L.nth(r, dens) : F32))

def div_arr(+hd: Nat, +left: Nat, +i: U32, +dens: List<&2, F32>, a: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      a
    case 1n+p:
      a2 = div_set(hd, i, dens, Array.get(F32, a, i))
      div_arr(hd, p, (i + 1 : U32), dens, a2)

def div_rows(+n: Nat, +hd: Nat, num: TA.Mat<n, hd>, +dens: List<&2, F32>) -> TA.Mat<n, hd>:
  match num:
    case TA.Mat{a}:
      TA.Mat{div_arr(hd, Nat.mul(n, hd), 0, dens, a)}

# ---- backward pieces ----

def dnum_row(+left: Nat, +i: Nat, +hd: Nat, +dy: List<&2, F32>, +num: List<&2, F32>,
    +dens: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +c = L.nth(i, dens)
      +d = L.dot(L.take_row(dy, Nat.mul(i, hd), hd), L.take_row(num, Nat.mul(i, hd), hd))
      +v = Bool.pick(F32, F32.is_le(0.000001, c), (0.0 - (d / (c * c : F32)) : F32), 0.0)
      dnum_row(p, (i + 1n : Nat), hd, dy, num, dens, Con{v, acc})

def dnum_two(+n: Nat, +hd: Nat, dy1: TA.Mat<n, hd>, dy2: TA.Mat<n, hd>,
    num: TA.Mat<n, hd>, +dens: List<&2, F32>) -> T.Two<List<&2, F32>, List<&2, F32>>:
  dl = mlhd(n, hd, TA.Mat.to_list(n, hd, dy2))
  ul = mlhd(n, hd, TA.Mat.to_list(n, hd, num))
  T.Two{mlhd(n, hd, TA.Mat.to_list(n, hd, div_rows(n, hd, dy1, dens))),
    dnum_row(n, 0n, hd, dl, ul, dens, Nil{})}

def dd_row(+left: Nat, +t: Nat, +d: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      dd_row(p, t, d, Con{d, acc})

def dd_rows(+n: Nat, +left: Nat, +t: Nat, +dden: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      dd_rows(n, p, (t + 1n : Nat), dden, List.append(&2, F32, acc, dd_row(n, t, L.nth(t, dden), Nil{})))

def dd_mat(+n: Nat, +dden: List<&2, F32>) -> TA.Mat<n, n>:
  TA.Mat.from_list(n, n, dd_rows(n, n, 0n, dden, Nil{}))

type DBW<-n: Nat, -hd: Nat> is Type:
  DBW{dq: TA.Mat<n, hd>, dk: TA.Mat<n, hd>, dv: TA.Mat<n, hd>}

def dup_a(+n: Nat, +hd: Nat, p: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>) -> TA.Mat<n, hd>:
  match p:
    case T.Two{a, b}:
      a

def dup_b(+n: Nat, +hd: Nat, p: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>) -> TA.Mat<n, hd>:
  match p:
    case T.Two{a, b}:
      b

def dup3_fin(+n: Nat, +hd: Nat, a: TA.Mat<n, hd>,
    p: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>) -> T.Three<TA.Mat<n, hd>, TA.Mat<n, hd>, TA.Mat<n, hd>>:
  match p:
    case T.Two{b, d}:
      T.Three{a, b, d}

def dup3_a(+n: Nat, +hd: Nat, p: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>) -> T.Three<TA.Mat<n, hd>, TA.Mat<n, hd>, TA.Mat<n, hd>>:
  match p:
    case T.Two{a, b}:
      dup3_fin(n, hd, a, T.Mat.dup(n, hd, b))

def dup3(+n: Nat, +hd: Nat, m: TA.Mat<n, hd>) -> T.Three<TA.Mat<n, hd>, TA.Mat<n, hd>, TA.Mat<n, hd>>:
  dup3_a(n, hd, T.Mat.dup(n, hd, m))

# ---- forward ----

def y_two(+n: Nat, +hd: Nat, m1: TA.Mat<n, n>, m2: TA.Mat<n, n>,
    v: TA.Mat<n, hd>) -> TA.Mat<n, hd>:
  dens = row_sums(n, m1)
  div_rows(n, hd, SD.mmul_c(n, n, hd, TA.Mat.matmul(n, n, hd, 0n, m2, v)), dens)

def y_after(+n: Nat, +hd: Nat, md: T.Two<TA.Mat<n, n>, TA.Mat<n, n>>,
    v: TA.Mat<n, hd>) -> TA.Mat<n, hd>:
  match md:
    case T.Two{m1, m2}:
      y_two(n, hd, m1, m2, v)

def lin_y_mask(+n: Nat, +hd: Nat, q: TA.Mat<n, hd>, k: TA.Mat<n, hd>,
    v: TA.Mat<n, hd>, mask: TA.Mat<n, n>) -> TA.Mat<n, hd>:
  qe = map_mat(~elu1_f, n, hd, q)
  ke = map_mat(~elu1_f, n, hd, k)
  s = SD.mmulnt_c(n, hd, n, TA.Mat.matmul_nt(n, hd, n, 0n, qe, ke))
  masked = mul2_fwd(n, n, s, mask)
  y_after(n, hd, T.Mat.dup(n, n, masked), v)

def lin_y(+n: Nat, +hd: Nat, q: TA.Mat<n, hd>, k: TA.Mat<n, hd>,
    v: TA.Mat<n, hd>) -> TA.Mat<n, hd>:
  lin_y_mask(n, hd, q, k, v, causal_mat(n))

# ---- backward ----

def bwd_final(+n: Nat, +hd: Nat, dQe: TA.Mat<n, hd>, dKe: TA.Mat<n, hd>,
    dV: TA.Mat<n, hd>, qdr: TA.Mat<n, hd>, kdr: TA.Mat<n, hd>) -> DBW<n, hd>:
  DBW{mul2_fwd(n, hd, dQe, qdr), mul2_fwd(n, hd, dKe, kdr), dV}

def bwd_ds(+n: Nat, +hd: Nat, dnum2: TA.Mat<n, hd>,
    dsd: T.Two<TA.Mat<n, n>, TA.Mat<n, n>>, m3: TA.Mat<n, n>,
    qe2: TA.Mat<n, hd>, ke2: TA.Mat<n, hd>,
    qdr: TA.Mat<n, hd>, kdr: TA.Mat<n, hd>) -> DBW<n, hd>:
  match dsd:
    case T.Two{sa, sb}:
      dq = SD.mmul_c(n, n, hd, TA.Mat.matmul(n, n, hd, 0n, sa, ke2))
      dk = SD.mmultn_c(n, n, hd, TA.Mat.matmul_tn(n, n, hd, 0n, sb, qe2))
      dv = SD.mmultn_c(n, n, hd, TA.Mat.matmul_tn(n, n, hd, 0n, m3, dnum2))
      bwd_final(n, hd, dq, dk, dv, qdr, kdr)

def bwd_dnum(+n: Nat, +hd: Nat, dnumd: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>,
    v2: TA.Mat<n, hd>, +ddenl: List<&2, F32>, m3: TA.Mat<n, n>,
    qe2: TA.Mat<n, hd>, ke2: TA.Mat<n, hd>,
    qdr: TA.Mat<n, hd>, kdr: TA.Mat<n, hd>, mask: TA.Mat<n, n>) -> DBW<n, hd>:
  match dnumd:
    case T.Two{da, db}:
      dsbase = SD.mmulnt_c(n, hd, n, TA.Mat.matmul_nt(n, hd, n, 0n, da, v2))
      dsum = SD.mpair_a(n, n, TA.Mat.add(n, n, dsbase, dd_mat(n, ddenl)))
      ds = mul2_fwd(n, n, dsum, mask)
      bwd_ds(n, hd, db, T.Mat.dup(n, n, ds), m3, qe2, ke2, qdr, kdr)

def bwd_nd(+n: Nat, +hd: Nat, nd: T.Two<List<&2, F32>, List<&2, F32>>,
    v2: TA.Mat<n, hd>, m3: TA.Mat<n, n>, qe2: TA.Mat<n, hd>, ke2: TA.Mat<n, hd>,
    qdr: TA.Mat<n, hd>, kdr: TA.Mat<n, hd>, mask: TA.Mat<n, n>) -> DBW<n, hd>:
  match nd:
    case T.Two{dnuml, ddenl}:
      dnum = TA.Mat.from_list(n, hd, dnuml)
      bwd_dnum(n, hd, T.Mat.dup(n, hd, dnum), v2, ddenl, m3, qe2, ke2, qdr, kdr, mask)

def bwd_dy(+n: Nat, +hd: Nat, dyd: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>,
    num: TA.Mat<n, hd>, +dens: List<&2, F32>) -> T.Two<List<&2, F32>, List<&2, F32>>:
  match dyd:
    case T.Two{da, db}:
      dnum_two(n, hd, da, db, num, dens)

def bwd_three(+n: Nat, +hd: Nat, m3: T.Three<TA.Mat<n, n>, TA.Mat<n, n>, TA.Mat<n, n>>,
    v1: TA.Mat<n, hd>, v2: TA.Mat<n, hd>, dy: TA.Mat<n, hd>,
    qe2: TA.Mat<n, hd>, ke2: TA.Mat<n, hd>,
    qdr: TA.Mat<n, hd>, kdr: TA.Mat<n, hd>, mask: TA.Mat<n, n>) -> DBW<n, hd>:
  match m3:
    case T.Three{m1, m2, m3}:
      dens = row_sums(n, m1)
      num = SD.mmul_c(n, n, hd, TA.Mat.matmul(n, n, hd, 0n, m2, v1))
      bwd_nd(n, hd, bwd_dy(n, hd, T.Mat.dup(n, hd, dy), num, dens),
        v2, m3, qe2, ke2, qdr, kdr, mask)

def bwd_masked(+n: Nat, +hd: Nat, md: T.Two<TA.Mat<n, n>, TA.Mat<n, n>>,
    s: TA.Mat<n, n>, v1: TA.Mat<n, hd>, v2: TA.Mat<n, hd>, dy: TA.Mat<n, hd>,
    qe2: TA.Mat<n, hd>, ke2: TA.Mat<n, hd>, qdr: TA.Mat<n, hd>,
    kdr: TA.Mat<n, hd>) -> DBW<n, hd>:
  match md:
    case T.Two{ma, mb}:
      masked = mul2_fwd(n, n, s, ma)
      bwd_three(n, hd, dup3(n, n, masked), v1, v2, dy, qe2, ke2, qdr, kdr, mb)

def bwd_qkv(+n: Nat, +hd: Nat, qd: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>,
    kd: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>, vd: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>,
    dy: TA.Mat<n, hd>, qdr: TA.Mat<n, hd>, kdr: TA.Mat<n, hd>,
    mask: TA.Mat<n, n>) -> DBW<n, hd>:
  match qd kd vd:
    case T.Two{q1, q2} T.Two{k1, k2} T.Two{v1, v2}:
      s = SD.mmulnt_c(n, hd, n, TA.Mat.matmul_nt(n, hd, n, 0n, q1, k1))
      bwd_masked(n, hd, T.Mat.dup(n, n, mask), s, v1, v2, dy, q2, k2, qdr, kdr)

type QKR<-n: Nat, -hd: Nat> is Type:
  QKR{qe: TA.Mat<n, hd>, qdr: TA.Mat<n, hd>, ke: TA.Mat<n, hd>, kdr: TA.Mat<n, hd>}

def qk_make(+n: Nat, +hd: Nat, qraw: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>,
    kraw: T.Two<TA.Mat<n, hd>, TA.Mat<n, hd>>) -> QKR<n, hd>:
  match qraw kraw:
    case T.Two{q1, q2} T.Two{k1, k2}:
      QKR{map_mat(~elu1_f, n, hd, q1), map_mat(~deriv_f, n, hd, q2),
          map_mat(~elu1_f, n, hd, k1), map_mat(~deriv_f, n, hd, k2)}

def bwd_qk2(+n: Nat, +hd: Nat, qk: QKR<n, hd>, v: TA.Mat<n, hd>,
    dy: TA.Mat<n, hd>, mask: TA.Mat<n, n>) -> DBW<n, hd>:
  match qk:
    case QKR{qe, qdr, ke, kdr}:
      bwd_qkv(n, hd, T.Mat.dup(n, hd, qe), T.Mat.dup(n, hd, ke), T.Mat.dup(n, hd, v),
        dy, qdr, kdr, mask)

def bwd_qk(+n: Nat, +hd: Nat, q: TA.Mat<n, hd>, k: TA.Mat<n, hd>,
    v: TA.Mat<n, hd>, dy: TA.Mat<n, hd>) -> DBW<n, hd>:
  bwd_qk2(n, hd, qk_make(n, hd, T.Mat.dup(n, hd, q), T.Mat.dup(n, hd, k)), v, dy,
    causal_mat(n))

def lin_bwd_mask(+n: Nat, +hd: Nat, dy: TA.Mat<n, hd>, q: TA.Mat<n, hd>, k: TA.Mat<n, hd>,
    v: TA.Mat<n, hd>, mask: TA.Mat<n, n>) -> DBW<n, hd>:
  bwd_qk2(n, hd, qk_make(n, hd, T.Mat.dup(n, hd, q), T.Mat.dup(n, hd, k)), v, dy, mask)

def lin_bwd(+n: Nat, +hd: Nat, dy: TA.Mat<n, hd>, q: TA.Mat<n, hd>, k: TA.Mat<n, hd>,
    v: TA.Mat<n, hd>) -> DBW<n, hd>:
  bwd_qk(n, hd, q, k, v, dy)
