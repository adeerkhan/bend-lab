# Exact causal linear attention, single head, forward and backward.
#
# The reference (`glm53_flash/model.py`, `LinearAttention`) computes, per
# batch and head, q~ = ELU(q)+1, k~ = ELU(k)+1, the causal prefix sums
#     K_t = sum_{s<=t} k~_s,   KV_t = sum_{s<=t} k~_s v_s^T,
# and y_t = (KV_t^T q~_t) / clamp(q~_t·K_t, 1e-6).
#
# Backward, with c_t = clamp(den_t, 1e-6), dnum_t = dy_t/c_t and
# dden_t = -dy_t·num_t / c_t^2 (zero under the clamp), and
#     a_{t,s} = dnum_t·v_s + dden_t   for s <= t,
#     dq~_t = sum_{s<=t} a_{t,s} k~_s
#     dk~_s = sum_{t>=s} a_{t,s} q~_t
#     dv_s  = sum_{t>=s} dnum_t (q~_t·k~_s)
# and the ELU chain multiplies dq~/dk~ by 1 (q>0) or exp(q) (q<=0).
#
# This is the direct O(T^2 d) form at one head.  It exists to be obviously
# correct at t = 2 and to be the oracle for the prefix-state rewrite.  A
# one-hot dout at t = 0 must give dv = 0 for every t > 0; that is the can-fail
# case the old 64-step implementation lacked.
import Base

def zeros(+left: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      zeros(p, Con{0.0, acc})

def zline(+n: Nat) -> List<&2, F32>:
  zeros(n, Nil{})

def nth(+n: Nat, xs: List<&2, F32>) -> F32:
  match n xs:
    case 0n Con{h, t}:
      h
    case 1n+p Con{h, t}:
      nth(p, t)
    case _ _:
      0.0

def take_row(xs: List<&2, F32>, +off: Nat, +n: Nat) -> List<&2, F32>:
  List.take(&2, F32, List.drop(&2, F32, xs, off), n)

def push_row(acc: List<&2, F32>, row: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, acc, row)

def add_lists(xs: List<&2, F32>, ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Nil{} _:
      Nil{}
    case _ Nil{}:
      Nil{}
    case Con{h, t} Con{g, u}:
      Con{(h + g : F32), add_lists(t, u)}

def mul_lists(xs: List<&2, F32>, ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Nil{} _:
      Nil{}
    case _ Nil{}:
      Nil{}
    case Con{h, t} Con{g, u}:
      Con{(h * g : F32), mul_lists(t, u)}

def dot(xs: List<&2, F32>, ys: List<&2, F32>) -> F32:
  match xs ys:
    case Nil{} _:
      0.0
    case _ Nil{}:
      0.0
    case Con{h, t} Con{g, u}:
      ((h * g : F32) + dot(t, u) : F32)

def scale_list(xs: List<&2, F32>, +c: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(h * c : F32), scale_list(t, c)}

def elu_if(le: Bool, +x: F32) -> F32:
  match le:
    case True{}:
      (F32.exp(x) - 1.0 : F32)
    case False{}:
      x

def elu(+x: F32) -> F32:
  elu_if(F32.is_le(x, 0.0), x)

def elu1(+x: F32) -> F32:
  (elu(x) + 1.0 : F32)

def deriv_if(le: Bool, +x: F32) -> F32:
  match le:
    case True{}:
      F32.exp(x)
    case False{}:
      1.0

def elu_deriv(+x: F32) -> F32:
  deriv_if(F32.is_le(x, 0.0), x)

def elu_row(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{elu1(h), elu_row(t)}

def deriv_row(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{elu_deriv(h), deriv_row(t)}

def add_outer_b(ys: List<&2, F32>, +acc: List<&2, F32>, +x: F32, +hd: Nat, +a: Nat, +b: Nat) -> List<&2, F32>:
  match ys:
    case Nil{}:
      acc
    case Con{h, t}:
      +idx = Nat.add(Nat.mul(a, hd), b)
      +v = (x * h : F32)
      add_outer_b(t, List.set(&2, F32, acc, idx, (nth(idx, acc) + v : F32)), x, hd, a, (b + 1n : Nat))

def add_outer_a(xs: List<&2, F32>, +acc: List<&2, F32>, +ys: List<&2, F32>, +hd: Nat, +a: Nat) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      add_outer_a(t, add_outer_b(ys, acc, h, hd, a, 0n), ys, hd, (a + 1n : Nat))

def add_outer(acc: List<&2, F32>, xs: List<&2, F32>, ys: List<&2, F32>, +hd: Nat) -> List<&2, F32>:
  add_outer_a(xs, acc, ys, hd, 0n)

def matvec_go(+left: Nat, +m: List<&2, F32>, +xs: List<&2, F32>, +hd: Nat, +a: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+p:
      +row = take_row(m, Nat.mul(a, hd), hd)
      Con{dot(row, xs), matvec_go(p, m, xs, hd, (a + 1n : Nat))}

def matvec(m: List<&2, F32>, xs: List<&2, F32>, +hd: Nat) -> List<&2, F32>:
  matvec_go(hd, m, xs, hd, 0n)

def matvec_t_go(+left: Nat, +m: List<&2, F32>, +xs: List<&2, F32>, +hd: Nat, +a: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +row = take_row(m, Nat.mul(a, hd), hd)
      matvec_t_go(p, m, xs, hd, (a + 1n : Nat), add_lists(acc, scale_list(row, nth(a, xs))))

def matvec_t(m: List<&2, F32>, xs: List<&2, F32>, +hd: Nat) -> List<&2, F32>:
  matvec_t_go(hd, m, xs, hd, 0n, zline(hd))

def clamp_if(le: Bool, +x: F32) -> F32:
  match le:
    case True{}:
      0.000001
    case False{}:
      x

def clamp_den(+x: F32) -> F32:
  clamp_if(F32.is_le(x, 0.000001), x)

def active_if(le: Bool) -> Bool:
  match le:
    case True{}:
      False{}
    case False{}:
      True{}

def active_den(+x: F32) -> Bool:
  active_if(F32.is_le(x, 0.000001))

def dden_of(active: Bool, +dn: F32, +c: F32) -> F32:
  match active:
    case True{}:
      ((0.0 - dn : F32) / (c * c : F32) : F32)
    case False{}:
      0.0

# ---- rows and causal prefixes, recomputed on demand ----

def qe_at(+t: Nat, +hd: Nat, +q: List<&2, F32>) -> List<&2, F32>:
  elu_row(take_row(q, Nat.mul(t, hd), hd))

def ke_at(+t: Nat, +hd: Nat, +k: List<&2, F32>) -> List<&2, F32>:
  elu_row(take_row(k, Nat.mul(t, hd), hd))

def v_at(+t: Nat, +hd: Nat, +v: List<&2, F32>) -> List<&2, F32>:
  take_row(v, Nat.mul(t, hd), hd)

def dy_at(+t: Nat, +hd: Nat, +dy: List<&2, F32>) -> List<&2, F32>:
  take_row(dy, Nat.mul(t, hd), hd)

def sum_k(+left: Nat, +s: Nat, +hd: Nat, +k: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      sum_k(p, (s + 1n : Nat), hd, k, add_lists(acc, ke_at(s, hd, k)))

def prefix_k(+t: Nat, +hd: Nat, +k: List<&2, F32>) -> List<&2, F32>:
  sum_k(Nat.add(t, 1n), 0n, hd, k, zline(hd))

def sum_kv(+left: Nat, +s: Nat, +hd: Nat, +k: List<&2, F32>, +v: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      sum_kv(p, (s + 1n : Nat), hd, k, v,
        add_outer(acc, ke_at(s, hd, k), v_at(s, hd, v), hd))

def prefix_kv(+t: Nat, +hd: Nat, +k: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  sum_kv(Nat.add(t, 1n), 0n, hd, k, v, zline(Nat.mul(hd, hd)))

def den_at(+t: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>) -> F32:
  dot(qe_at(t, hd, q), prefix_k(t, hd, k))

def c_at(+t: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>) -> F32:
  clamp_den(den_at(t, hd, q, k))

def num_at(+t: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  matvec_t(prefix_kv(t, hd, k, v), qe_at(t, hd, q), hd)

def dden_at(+t: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +dy: List<&2, F32>) -> F32:
  +den = den_at(t, hd, q, k)
  dden_of(active_den(den), dot(dy_at(t, hd, dy), num_at(t, hd, q, k, v)), clamp_den(den))

def dn_at(+t: Nat, +hd: Nat, +dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>) -> List<&2, F32>:
  scale_list(dy_at(t, hd, dy), (1.0 / c_at(t, hd, q, k) : F32))

# a_{t,s} = dnum_t·v_s + dden_t
def a_ts(+t: Nat, +s: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +dy: List<&2, F32>) -> F32:
  ((dot(dn_at(t, hd, dy, q, k), v_at(s, hd, v)) + dden_at(t, hd, q, k, v, dy) : F32))

# ---- the forward output ----

def y_go(+left: Nat, +t: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      y_go(p, (t + 1n : Nat), hd, q, k, v,
        push_row(acc, scale_list(num_at(t, hd, q, k, v), (1.0 / c_at(t, hd, q, k) : F32))))

def lin_y(+n: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  y_go(n, 0n, hd, q, k, v, Nil{})

# ---- the backward output ----

type BW is Type:
  BW{dq: List<&2, F32>, dk: List<&2, F32>, dv: List<&2, F32>}

def dq_row(+left: Nat, +s: Nat, +t: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, +dy: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      dq_row(p, (s + 1n : Nat), t, hd, q, k, v, dy,
        add_lists(acc, scale_list(ke_at(s, hd, k), a_ts(t, s, hd, q, k, v, dy))))

def dq_rows(+left: Nat, +t: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, +dy: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +row0 = dq_row(Nat.add(t, 1n), 0n, t, hd, q, k, v, dy, zline(hd))
      +row = mul_lists(row0, deriv_row(take_row(q, Nat.mul(t, hd), hd)))
      dq_rows(p, (t + 1n : Nat), hd, q, k, v, dy, push_row(acc, row))

def dk_row(+left: Nat, +t: Nat, +s: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, +dy: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      dk_row(p, (t + 1n : Nat), s, hd, q, k, v, dy,
        add_lists(acc, scale_list(qe_at(t, hd, q), a_ts(t, s, hd, q, k, v, dy))))

def dk_rows(+left: Nat, +s: Nat, +hd: Nat, +n: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, +dy: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +row0 = dk_row(Nat.sub(n, s), s, s, hd, q, k, v, dy, zline(hd))
      +row = mul_lists(row0, deriv_row(take_row(k, Nat.mul(s, hd), hd)))
      dk_rows(p, (s + 1n : Nat), hd, n, q, k, v, dy, push_row(acc, row))

def dv_row(+left: Nat, +t: Nat, +s: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, +dy: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +d = dot(qe_at(t, hd, q), ke_at(s, hd, k))
      dv_row(p, (t + 1n : Nat), s, hd, q, k, v, dy,
        add_lists(acc, scale_list(dn_at(t, hd, dy, q, k), d)))

def dv_rows(+left: Nat, +s: Nat, +hd: Nat, +n: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, +dy: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +row = dv_row(Nat.sub(n, s), s, s, hd, q, k, v, dy, zline(hd))
      dv_rows(p, (s + 1n : Nat), hd, n, q, k, v, dy, push_row(acc, row))

def lin_bwd(+n: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>,
    +dy: List<&2, F32>) -> BW:
  BW{
    dq_rows(n, 0n, hd, q, k, v, dy, Nil{}),
    dk_rows(n, 0n, hd, n, q, k, v, dy, Nil{}),
    dv_rows(n, 0n, hd, n, q, k, v, dy, Nil{})}
