# The Stage B/C attention cores at the integrated scale: one batch, 64
# positions, one head of width 8, pre-RoPE q/k.
#
# The linear core is `lin_bwd.bend`; RoPE is applied here and its transpose is
# applied to dq/dk.  The sparse core is the reference's gathered causal
# attention: anchors every 16 positions plus a local window of 16, sorted and
# deduplicated by the validity test
#     s <= t  and  (s % 16 == 0  or  t - s < 16).
# It has no ELU feature map.  Both cores are parameterless.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./lin_bwd.bend as L

def N() -> Nat:
  64n

def HD() -> Nat:
  8n

def W() -> Nat:
  Nat.mul(HD(), 3n)

def M() -> Type:
  TA.Mat<N(), HD()>

def C() -> Type:
  TA.Mat<N(), W()>

def QKV() -> Type:
  T.Two<M(), T.Two<M(), M()>>

def ml8(ml: TA.ML<N(), HD()>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml24(ml: TA.ML<N(), W()>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

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

def zeros(+left: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      zeros(p, Con{0.0, acc})

def zline(+n: Nat) -> List<&2, F32>:
  zeros(n, Nil{})

def dot(xs: List<&2, F32>, ys: List<&2, F32>) -> F32:
  match xs ys:
    case Nil{} _:
      0.0
    case _ Nil{}:
      0.0
    case Con{h, t} Con{g, u}:
      ((h * g : F32) + dot(t, u) : F32)

def add_lists(xs: List<&2, F32>, ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Nil{} _:
      Nil{}
    case _ Nil{}:
      Nil{}
    case Con{h, t} Con{g, u}:
      Con{(h + g : F32), add_lists(t, u)}

def scale_list(xs: List<&2, F32>, +c: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(h * c : F32), scale_list(t, c)}

def fill_go(row: List<&2, F32>, acc: List<&2, F32>, +off: Nat, +i: Nat) -> List<&2, F32>:
  match row:
    case Nil{}:
      acc
    case Con{h, t}:
      fill_go(t, List.set(&2, F32, acc, Nat.add(off, i), h), off, (i + 1n : Nat))

def fill_row(acc: List<&2, F32>, +off: Nat, row: List<&2, F32>) -> List<&2, F32>:
  fill_go(row, acc, off, 0n)

def add_at_go(row: List<&2, F32>, +acc: List<&2, F32>, +off: Nat, +i: Nat) -> List<&2, F32>:
  match row:
    case Nil{}:
      acc
    case Con{h, t}:
      +idx = Nat.add(off, i)
      add_at_go(t, List.set(&2, F32, acc, idx, (nth(idx, acc) + h : F32)), off, (i + 1n : Nat))

def add_row_at(acc: List<&2, F32>, +off: Nat, row: List<&2, F32>) -> List<&2, F32>:
  add_at_go(row, acc, off, 0n)

# ---- RoPE, one head of width 8, interleaved even/odd pairs ----

def rope_j(+left: Nat, +xs: List<&2, F32>, +t: Nat, +j: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      xs
    case 1n+p:
      +ang = (F32.from_nat(t) * F32.exp(((0.0 - (F32.from_nat(j) * 2.302585092994046 : F32)) : F32)) : F32)
      +c = F32.cos(ang)
      +s = F32.sin(ang)
      +a = nth(Nat.mul(j, 2n), xs)
      +b = nth(Nat.add(Nat.mul(j, 2n), 1n), xs)
      +x2 = List.set(&2, F32, xs, Nat.mul(j, 2n), ((a * c : F32) - (b * s : F32) : F32))
      +x3 = List.set(&2, F32, x2, Nat.add(Nat.mul(j, 2n), 1n), ((a * s : F32) + (b * c : F32) : F32))
      rope_j(p, x3, t, (j + 1n : Nat))

def rope_row(xs: List<&2, F32>, +t: Nat) -> List<&2, F32>:
  rope_j(4n, xs, t, 0n)

def unrope_j(+left: Nat, +xs: List<&2, F32>, +t: Nat, +j: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      xs
    case 1n+p:
      +ang = (F32.from_nat(t) * F32.exp(((0.0 - (F32.from_nat(j) * 2.302585092994046 : F32)) : F32)) : F32)
      +c = F32.cos(ang)
      +s = F32.sin(ang)
      +a = nth(Nat.mul(j, 2n), xs)
      +b = nth(Nat.add(Nat.mul(j, 2n), 1n), xs)
      +x2 = List.set(&2, F32, xs, Nat.mul(j, 2n), (((a * c : F32) + (b * s : F32)) : F32))
      +x3 = List.set(&2, F32, x2, Nat.add(Nat.mul(j, 2n), 1n), (((0.0 - (a * s : F32) : F32) + (b * c : F32)) : F32))
      unrope_j(p, x3, t, (j + 1n : Nat))

def rope_all_go(+left: Nat, +t: Nat, +src: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      rope_all_go(p, (t + 1n : Nat), src, push_row(acc, rope_row(take_row(src, Nat.mul(t, HD()), HD()), t)))

def rope_all(+src: List<&2, F32>) -> List<&2, F32>:
  rope_all_go(N(), 0n, src, Nil{})

def unrope_all_go(+left: Nat, +t: Nat, +src: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      unrope_all_go(p, (t + 1n : Nat), src, push_row(acc, unrope_j(4n, take_row(src, Nat.mul(t, HD()), HD()), t, 0n)))

def unrope_all(+src: List<&2, F32>) -> List<&2, F32>:
  unrope_all_go(N(), 0n, src, Nil{})

def unrope_bw(bw: L.BW) -> L.BW:
  match bw:
    case L.BW{dq, dk, dv}:
      L.BW{unrope_all(dq), unrope_all(dk), dv}

# ---- linear attention: RoPE, then the exact core ----

def lin_y(q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>) -> List<&2, F32>:
  L.lin_y(N(), HD(), rope_all(q), rope_all(k), v)

def lin_bwd(q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>, dy: List<&2, F32>) -> L.BW:
  unrope_bw(L.lin_bwd(N(), HD(), rope_all(q), rope_all(k), v, dy))

# ---- sparse attention: window 16, stride 16, causal ----

def valid(+t: Nat, +s: Nat) -> Bool:
  +le = Nat.is_le(s, t)
  +anchor = Nat.is_eq(Nat.mod(s, 16n), 0n)
  +local = Nat.is_lt(Nat.sub(t, s), 16n)
  Bool.and(le, Bool.or(anchor, local))

def score_at(+t: Nat, +s: Nat, +q: List<&2, F32>, +k: List<&2, F32>) -> F32:
  (dot(take_row(q, Nat.mul(t, HD()), HD()), take_row(k, Nat.mul(s, HD()), HD())) * 0.35355339059327373 : F32)

def smax_go(+left: Nat, +s: Nat, +t: Nat, +q: List<&2, F32>, +k: List<&2, F32>, acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      +v = Bool.pick(F32, valid(t, s), score_at(t, s, q, k), (0.0 - 1.0e30 : F32))
      smax_go(p, (s + 1n : Nat), t, q, k, F32.max(acc, v))

def smax_at(+t: Nat, +q: List<&2, F32>, +k: List<&2, F32>) -> F32:
  smax_go(Nat.add(t, 1n), 0n, t, q, k, (0.0 - 1.0e30 : F32))

def ssum_go(+left: Nat, +s: Nat, +t: Nat, +m: F32, +q: List<&2, F32>, +k: List<&2, F32>, acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      +v = Bool.pick(F32, valid(t, s), F32.exp((score_at(t, s, q, k) - m : F32)), 0.0)
      ssum_go(p, (s + 1n : Nat), t, m, q, k, (acc + v : F32))

def ssum_at(+t: Nat, +m: F32, +q: List<&2, F32>, +k: List<&2, F32>) -> F32:
  ssum_go(Nat.add(t, 1n), 0n, t, m, q, k, 0.0)

def sout_go(+left: Nat, +s: Nat, +t: Nat, +m: F32, +z: F32, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +w = Bool.pick(F32, valid(t, s), (F32.exp((score_at(t, s, q, k) - m : F32)) / z : F32), 0.0)
      sout_go(p, (s + 1n : Nat), t, m, z, q, k, v, add_lists(acc, scale_list(take_row(v, Nat.mul(s, HD()), HD()), w)))

def srow_at(+t: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  +m = smax_at(t, q, k)
  sout_go(Nat.add(t, 1n), 0n, t, m, ssum_at(t, m, q, k), q, k, v, zline(HD()))

def sp_y_go(+left: Nat, +t: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      sp_y_go(p, (t + 1n : Nat), q, k, v, push_row(acc, srow_at(t, q, k, v)))

def sp_y(q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>) -> List<&2, F32>:
  sp_y_go(N(), 0n, rope_all(q), rope_all(k), v, Nil{})

# backward: for each t, p_s, dp_s = dy_t·v_s, sum p*dp, dq_t = scale Σ p(dp-sum)k_s,
# dv_s += p dy_t, dk_s += scale p(dp-sum) q_t.

def sp_dp_go(+left: Nat, +s: Nat, +t: Nat, +m: F32, +z: F32, +dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      +pw = Bool.pick(F32, valid(t, s), (F32.exp((score_at(t, s, q, k) - m : F32)) / z : F32), 0.0)
      +dp = dot(take_row(dy, Nat.mul(t, HD()), HD()), take_row(v, Nat.mul(s, HD()), HD()))
      sp_dp_go(p, (s + 1n : Nat), t, m, z, dy, q, k, v, (acc + (pw * dp : F32) : F32))

def sp_sum_dp(+t: Nat, +m: F32, +z: F32, +dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> F32:
  sp_dp_go(Nat.add(t, 1n), 0n, t, m, z, dy, q, k, v, 0.0)

def sp_dq_go(+left: Nat, +s: Nat, +t: Nat, +m: F32, +z: F32, +sum: F32, +dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +pw = Bool.pick(F32, valid(t, s), (F32.exp((score_at(t, s, q, k) - m : F32)) / z : F32), 0.0)
      +dp = dot(take_row(dy, Nat.mul(t, HD()), HD()), take_row(v, Nat.mul(s, HD()), HD()))
      +ds = (pw * ((dp - sum : F32)) : F32)
      sp_dq_go(p, (s + 1n : Nat), t, m, z, sum, dy, q, k, v,
        add_lists(acc, scale_list(take_row(k, Nat.mul(s, HD()), HD()), (ds * 0.35355339059327373 : F32))))

def sp_dq_row(+t: Nat, +m: F32, +z: F32, +sum: F32, +dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  sp_dq_go(Nat.add(t, 1n), 0n, t, m, z, sum, dy, q, k, v, zline(HD()))

def sp_dv_dk_go(+left: Nat, +s: Nat, +t: Nat, +m: F32, +z: F32, +sum: F32, +dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, dv: List<&2, F32>, dk: List<&2, F32>) -> T.Two<List<&2, F32>, List<&2, F32>>:
  match left:
    case 0n:
      T.Two{dv, dk}
    case 1n+p:
      +pw = Bool.pick(F32, valid(t, s), (F32.exp((score_at(t, s, q, k) - m : F32)) / z : F32), 0.0)
      +dp = dot(take_row(dy, Nat.mul(t, HD()), HD()), take_row(v, Nat.mul(s, HD()), HD()))
      +ds = (pw * ((dp - sum : F32)) : F32)
      +dv2 = add_row_at(dv, Nat.mul(s, HD()), scale_list(take_row(dy, Nat.mul(t, HD()), HD()), pw))
      +dk2 = add_row_at(dk, Nat.mul(s, HD()), scale_list(take_row(q, Nat.mul(t, HD()), HD()), (ds * 0.35355339059327373 : F32)))
      sp_dv_dk_go(p, (s + 1n : Nat), t, m, z, sum, dy, q, k, v, dv2, dk2)

def pair_fst(pair: T.Two<List<&2, F32>, List<&2, F32>>) -> List<&2, F32>:
  match pair:
    case T.Two{a, b}:
      a

def pair_snd(pair: T.Two<List<&2, F32>, List<&2, F32>>) -> List<&2, F32>:
  match pair:
    case T.Two{a, b}:
      b

def sp_t_go(+left: Nat, +t: Nat, +dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, dq: List<&2, F32>, +dv: List<&2, F32>, +dk: List<&2, F32>) -> L.BW:
  match left:
    case 0n:
      L.BW{dq, dk, dv}
    case 1n+p:
      +m = smax_at(t, q, k)
      +z = ssum_at(t, m, q, k)
      +sum = sp_sum_dp(t, m, z, dy, q, k, v)
      +dq2 = fill_row(dq, Nat.mul(t, HD()), sp_dq_row(t, m, z, sum, dy, q, k, v))
      sp_t_go(p, (t + 1n : Nat), dy, q, k, v, dq2,
        pair_fst(sp_dv_dk_go(Nat.add(t, 1n), 0n, t, m, z, sum, dy, q, k, v, dv, dk)),
        pair_snd(sp_dv_dk_go(Nat.add(t, 1n), 0n, t, m, z, sum, dy, q, k, v, dv, dk)))

def sp_bwd(q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>, dy: List<&2, F32>) -> L.BW:
  unrope_bw(sp_t_go(N(), 0n, dy, rope_all(q), rope_all(k), v,
    zeros(Nat.mul(N(), HD()), Nil{}), zeros(Nat.mul(N(), HD()), Nil{}), zeros(Nat.mul(N(), HD()), Nil{})))

# ---- split the 24-wide projection into q, k, v ----

def split_go(+left: Nat, +i: Nat, +xs: List<&2, F32>, qacc: List<&2, F32>, kacc: List<&2, F32>, vacc: List<&2, F32>) -> T.Three<List<&2, F32>, List<&2, F32>, List<&2, F32>>:
  match left:
    case 0n:
      T.Three{qacc, kacc, vacc}
    case 1n+p:
      +off = Nat.mul(i, W())
      +qr = take_row(xs, off, HD())
      +kr = take_row(xs, Nat.add(off, HD()), HD())
      +vr = take_row(xs, Nat.add(off, Nat.mul(HD(), 2n)), HD())
      split_go(p, (i + 1n : Nat), xs,
        List.append(&2, F32, qacc, qr),
        List.append(&2, F32, kacc, kr),
        List.append(&2, F32, vacc, vr))

def split_res(t: T.Three<List<&2, F32>, List<&2, F32>, List<&2, F32>>) -> T.Fwd<Unit, QKV(), Unit>:
  match t:
    case T.Three{q, k, v}:
      T.Fwd{Unit{}, T.Two{TA.Mat.from_list(N(), HD(), q),
        T.Two{TA.Mat.from_list(N(), HD(), k), TA.Mat.from_list(N(), HD(), v)}}, Unit{}}

def split_fwd(x: C()) -> T.Fwd<Unit, QKV(), Unit>:
  match x:
    case TA.Mat{a}:
      split_res(split_go(N(), 0n, ml24(TA.Mat.to_list(N(), W(), TA.Mat{a})), Nil{}, Nil{}, Nil{}))

def join_go(+left: Nat, +i: Nat, +dq: List<&2, F32>, +dk: List<&2, F32>, +dv: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +off = Nat.mul(i, HD())
      +a = List.append(&2, F32, acc, take_row(dq, off, HD()))
      +b = List.append(&2, F32, a, take_row(dk, off, HD()))
      +c = List.append(&2, F32, b, take_row(dv, off, HD()))
      join_go(p, (i + 1n : Nat), dq, dk, dv, c)

def split_bwd3(dqm: M(), dkm: M(), dvm: M()) -> T.Bwd<Unit, C()>:
  T.Bwd{Unit{}, TA.Mat.from_list(N(), W(), join_go(N(), 0n,
    ml8(TA.Mat.to_list(N(), HD(), dqm)),
    ml8(TA.Mat.to_list(N(), HD(), dkm)),
    ml8(TA.Mat.to_list(N(), HD(), dvm)), Nil{})), Unit{}}

def split_bwd(dy: QKV()) -> T.Bwd<Unit, C()>:
  match dy:
    case T.Two{dqm, T.Two{dkm, dvm}}:
      split_bwd3(dqm, dkm, dvm)

def Split3() -> T.Layer<Unit, C(), QKV(), Unit>:
  T.Layer{p => x => split_fwd(x),
    p => s => dy => g => split_bwd(dy),
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0}

# ---- the two core layers ----

type LinS is Type:
  LinS{q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>}

def lin_fwd_state(+v: List<&2, F32>, +qr: List<&2, F32>, +kr: List<&2, F32>) -> T.Fwd<Unit, M(), LinS>:
  T.Fwd{Unit{}, TA.Mat.from_list(N(), HD(), L.lin_y(N(), HD(), qr, kr, v)),
    LinS{qr, kr, v}}

def lin_fwd_of(+q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> T.Fwd<Unit, M(), LinS>:
  lin_fwd_state(v, rope_all(q), rope_all(k))

def lin_fwd3(qm: M(), km: M(), vm: M()) -> T.Fwd<Unit, M(), LinS>:
  lin_fwd_of(ml8(TA.Mat.to_list(N(), HD(), qm)),
    ml8(TA.Mat.to_list(N(), HD(), km)),
    ml8(TA.Mat.to_list(N(), HD(), vm)))

def lin_fwd(qkv: QKV()) -> T.Fwd<Unit, M(), LinS>:
  match qkv:
    case T.Two{qm, T.Two{km, vm}}:
      lin_fwd3(qm, km, vm)

def bwd_mats(bw: L.BW) -> T.Bwd<Unit, QKV()>:
  match bw:
    case L.BW{dq, dk, dv}:
      T.Bwd{Unit{}, T.Two{TA.Mat.from_list(N(), HD(), dq),
        T.Two{TA.Mat.from_list(N(), HD(), dk), TA.Mat.from_list(N(), HD(), dv)}}, Unit{}}

def lin_bwd_lists(d: List<&2, F32>, q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>) -> T.Bwd<Unit, QKV()>:
  bwd_mats(unrope_bw(L.lin_bwd(N(), HD(), q, k, v, d)))

def lin_bwd_of(dy: M(), s: LinS) -> T.Bwd<Unit, QKV()>:
  match dy s:
    case TA.Mat{d} LinS{q, k, v}:
      lin_bwd_lists(ml8(TA.Mat.to_list(N(), HD(), TA.Mat{d})), q, k, v)

def LinCore() -> T.Layer<Unit, QKV(), M(), LinS>:
  T.Layer{p => x => lin_fwd(x),
    p => s => dy => g => lin_bwd_of(dy, s),
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0}

type SpS is Type:
  SpS{q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>}

def sp_y_list(q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>) -> List<&2, F32>:
  sp_y_go(N(), 0n, q, k, v, Nil{})

def sp_fwd_state(+v: List<&2, F32>, +qr: List<&2, F32>, +kr: List<&2, F32>) -> T.Fwd<Unit, M(), SpS>:
  T.Fwd{Unit{}, TA.Mat.from_list(N(), HD(), sp_y_list(qr, kr, v)),
    SpS{qr, kr, v}}

def sp_fwd_of(+q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> T.Fwd<Unit, M(), SpS>:
  sp_fwd_state(v, rope_all(q), rope_all(k))

def sp_fwd3(qm: M(), km: M(), vm: M()) -> T.Fwd<Unit, M(), SpS>:
  sp_fwd_of(ml8(TA.Mat.to_list(N(), HD(), qm)),
    ml8(TA.Mat.to_list(N(), HD(), km)),
    ml8(TA.Mat.to_list(N(), HD(), vm)))

def sp_fwd(qkv: QKV()) -> T.Fwd<Unit, M(), SpS>:
  match qkv:
    case T.Two{qm, T.Two{km, vm}}:
      sp_fwd3(qm, km, vm)

def sp_bwd_list(q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>, dy: List<&2, F32>) -> L.BW:
  sp_bwd(q, k, v, dy)

def sp_bwd_lists(d: List<&2, F32>, q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>) -> T.Bwd<Unit, QKV()>:
  bwd_mats(unrope_bw(sp_t_go(N(), 0n, d, q, k, v,
    zeros(Nat.mul(N(), HD()), Nil{}), zeros(Nat.mul(N(), HD()), Nil{}), zeros(Nat.mul(N(), HD()), Nil{}))))

def sp_bwd_of(dy: M(), s: SpS) -> T.Bwd<Unit, QKV()>:
  match dy s:
    case TA.Mat{d} SpS{q, k, v}:
      sp_bwd_lists(ml8(TA.Mat.to_list(N(), HD(), TA.Mat{d})), q, k, v)

def SpCore() -> T.Layer<Unit, QKV(), M(), SpS>:
  T.Layer{p => x => sp_fwd(x),
    p => s => dy => g => sp_bwd_of(dy, s),
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0}

# ---- the attention block: qkv projection, split, core, output ----

def QKVW() -> Type:
  TA.Mat<24n, 8n>

def OUTW() -> Type:
  TA.Mat<8n, 8n>

def AttP() -> Type:
  T.Two<QKVW(), T.Two<Unit, T.Two<Unit, OUTW()>>>

def LinAttS() -> Type:
  T.Two<M(), T.Two<Unit, T.Two<LinS, M()>>>

def SpAttS() -> Type:
  T.Two<M(), T.Two<Unit, T.Two<SpS, M()>>>

def LinCoreOut() -> T.Layer<T.Two<Unit, OUTW()>, QKV(), M(), T.Two<LinS, M()>>:
  T.Seq(~Unit, ~OUTW(), ~QKV(), ~M(), ~M(), ~LinS, ~M(), ~LinCore(),
    ~T.LinearNB(64n, 8n, 8n, 0n))

def SpCoreOut() -> T.Layer<T.Two<Unit, OUTW()>, QKV(), M(), T.Two<SpS, M()>>:
  T.Seq(~Unit, ~OUTW(), ~QKV(), ~M(), ~M(), ~SpS, ~M(), ~SpCore(),
    ~T.LinearNB(64n, 8n, 8n, 0n))

def LinSplitCore() -> T.Layer<T.Two<Unit, T.Two<Unit, OUTW()>>, C(), M(), T.Two<Unit, T.Two<LinS, M()>>>:
  T.Seq(~Unit, ~T.Two<Unit, OUTW()>, ~C(), ~QKV(), ~M(), ~Unit(), ~T.Two<LinS, M()>,
    ~Split3(), ~LinCoreOut())

def SpSplitCore() -> T.Layer<T.Two<Unit, T.Two<Unit, OUTW()>>, C(), M(), T.Two<Unit, T.Two<SpS, M()>>>:
  T.Seq(~Unit, ~T.Two<Unit, OUTW()>, ~C(), ~QKV(), ~M(), ~Unit(), ~T.Two<SpS, M()>,
    ~Split3(), ~SpCoreOut())

def LinAttention() -> T.Layer<AttP(), M(), M(), LinAttS()>:
  T.Seq(~QKVW(), ~T.Two<Unit, T.Two<Unit, OUTW()>>, ~M(), ~C(), ~M(), ~M(),
    ~T.Two<Unit, T.Two<LinS, M()>>,
    ~T.LinearNB(64n, 8n, 24n, 0n), ~LinSplitCore())

def SpAttention() -> T.Layer<AttP(), M(), M(), SpAttS()>:
  T.Seq(~QKVW(), ~T.Two<Unit, T.Two<Unit, OUTW()>>, ~M(), ~C(), ~M(), ~M(),
    ~T.Two<Unit, T.Two<SpS, M()>>,
    ~T.LinearNB(64n, 8n, 24n, 0n), ~SpSplitCore())
