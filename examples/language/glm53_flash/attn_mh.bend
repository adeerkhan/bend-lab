# Four-head causal attention at 128 positions: the linear and sparse cores
# wrapped per head, with a single qkv projection and RoPE per head.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./lin_bwd.bend as L
import ./lin_fast.bend as LF
import ./sp_dense.bend as SD

def N() -> Nat:
  128n

def D() -> Nat:
  32n

def H() -> Nat:
  4n

def HD() -> Nat:
  8n

def W3() -> Nat:
  Nat.mul(D(), 3n)

def M() -> Type:
  TA.Mat<N(), D()>

def C() -> Type:
  TA.Mat<N(), W3()>

def QKV() -> Type:
  T.Two<M(), T.Two<M(), M()>>

def mlD(ml: TA.ML<N(), D()>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def mlW(ml: TA.ML<N(), W3()>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

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

# ---- RoPE, one head of width 8, N positions ----

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

def rope_row(xs: List<&2, F32>, +t: Nat) -> List<&2, F32>:
  rope_j(4n, xs, t, 0n)

def rope_all_go(+left: Nat, +t: Nat, +src: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      rope_all_go(p, (t + 1n : Nat), src, L.push_row(acc, rope_row(take_row(src, Nat.mul(t, HD()), HD()), t)))

def rope_all(+src: List<&2, F32>) -> List<&2, F32>:
  rope_all_go(N(), 0n, src, Nil{})

def unrope_all_go(+left: Nat, +t: Nat, +src: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      unrope_all_go(p, (t + 1n : Nat), src,
        L.push_row(acc, unrope_j(4n, take_row(src, Nat.mul(t, HD()), HD()), t, 0n)))

def unrope_all(+src: List<&2, F32>) -> List<&2, F32>:
  unrope_all_go(N(), 0n, src, Nil{})

# ---- head gather and scatter ----

def gather_go(+left: Nat, +t: Nat, +h: Nat, +flat: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +off = Nat.add(Nat.mul(t, D()), Nat.mul(h, HD()))
      gather_go(p, (t + 1n : Nat), h, flat, List.append(&2, F32, acc, take_row(flat, off, HD())))

def gather_head(+flat: List<&2, F32>, +h: Nat) -> List<&2, F32>:
  gather_go(N(), 0n, h, flat, Nil{})

def join_heads_go(+left: Nat, +t: Nat, +h0: List<&2, F32>, +h1: List<&2, F32>,
    +h2: List<&2, F32>, +h3: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +o = Nat.mul(t, HD())
      +r0 = take_row(h0, o, HD())
      +r1 = take_row(h1, o, HD())
      +r2 = take_row(h2, o, HD())
      +r3 = take_row(h3, o, HD())
      +row = List.append(&2, F32, r0, List.append(&2, F32, r1,
        List.append(&2, F32, r2, r3)))
      join_heads_go(p, (t + 1n : Nat), h0, h1, h2, h3, List.append(&2, F32, acc, row))

def join_heads(+h0: List<&2, F32>, +h1: List<&2, F32>, +h2: List<&2, F32>,
    +h3: List<&2, F32>) -> List<&2, F32>:
  join_heads_go(N(), 0n, h0, h1, h2, h3, Nil{})

# ---- linear attention, four heads ----

def lin_head(+h: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  LF.lin_y_fast(N(), HD(), rope_all(gather_head(q, h)), rope_all(gather_head(k, h)),
    gather_head(v, h))

def lin_y4(+q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  +y0 = lin_head(0n, q, k, v)
  +y1 = lin_head(1n, q, k, v)
  +y2 = lin_head(2n, q, k, v)
  +y3 = lin_head(3n, q, k, v)
  join_heads(y0, y1, y2, y3)

def unrope_bw(bw: L.BW) -> L.BW:
  match bw:
    case L.BW{dq, dk, dv}:
      L.BW{unrope_all(dq), unrope_all(dk), dv}

def lin_bwd_head(+h: Nat, +dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>) -> L.BW:
  unrope_bw(LF.lin_bwd_fast(N(), HD(), rope_all(gather_head(q, h)), rope_all(gather_head(k, h)),
    gather_head(v, h), gather_head(dy, h)))

type ACC is Type:
  ACC{dq: List<&2, F32>, dk: List<&2, F32>, dv: List<&2, F32>}

def join4_bw(b0: L.BW, b1: L.BW, b2: L.BW, b3: L.BW) -> ACC:
  match b0 b1 b2 b3:
    case L.BW{a0, b0v, c0} L.BW{a1, b1v, c1} L.BW{a2, b2v, c2} L.BW{a3, b3v, c3}:
      ACC{join_heads(a0, a1, a2, a3), join_heads(b0v, b1v, b2v, b3v),
          join_heads(c0, c1, c2, c3)}

def lin_bwd4(+dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>) -> ACC:
  join4_bw(lin_bwd_head(0n, dy, q, k, v), lin_bwd_head(1n, dy, q, k, v),
    lin_bwd_head(2n, dy, q, k, v), lin_bwd_head(3n, dy, q, k, v))

# ---- sparse attention, four heads ----

def mlh(ml: TA.ML<N(), HD()>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def sp_head(+h: Nat, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  mlh(TA.Mat.to_list(N(), HD(), SD.y_of(N(), HD(),
    TA.Mat.from_list(N(), HD(), rope_all(gather_head(q, h))),
    TA.Mat.from_list(N(), HD(), rope_all(gather_head(k, h))),
    TA.Mat.from_list(N(), HD(), gather_head(v, h)), SD.mask_mat(N()))))

def sp_y4(+q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  join_heads(sp_head(0n, q, k, v), sp_head(1n, q, k, v), sp_head(2n, q, k, v), sp_head(3n, q, k, v))

def dbw_parts(dbw: SD.DBW<N(), HD()>) -> L.BW:
  match dbw:
    case SD.DBW{dq, dk, dv}:
      L.BW{unrope_all(mlh(TA.Mat.to_list(N(), HD(), dq))),
        unrope_all(mlh(TA.Mat.to_list(N(), HD(), dk))),
        mlh(TA.Mat.to_list(N(), HD(), dv))}

def sp_bwd_head(+h: Nat, +dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>) -> L.BW:
  dbw_parts(SD.bwd(N(), HD(), TA.Mat.from_list(N(), HD(), gather_head(dy, h)),
    TA.Mat.from_list(N(), HD(), rope_all(gather_head(q, h))),
    TA.Mat.from_list(N(), HD(), rope_all(gather_head(k, h))),
    TA.Mat.from_list(N(), HD(), gather_head(v, h)), SD.mask_mat(N())))

def sp_bwd4(+dy: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>) -> ACC:
  join4_bw(sp_bwd_head(0n, dy, q, k, v), sp_bwd_head(1n, dy, q, k, v),
    sp_bwd_head(2n, dy, q, k, v), sp_bwd_head(3n, dy, q, k, v))

# ---- split 96 columns into q, k, v ----

def split_go(+left: Nat, +i: Nat, +xs: List<&2, F32>, qacc: List<&2, F32>,
    kacc: List<&2, F32>, vacc: List<&2, F32>) -> T.Three<List<&2, F32>, List<&2, F32>, List<&2, F32>>:
  match left:
    case 0n:
      T.Three{qacc, kacc, vacc}
    case 1n+p:
      +off = Nat.mul(i, W3())
      split_go(p, (i + 1n : Nat), xs,
        List.append(&2, F32, qacc, take_row(xs, off, D())),
        List.append(&2, F32, kacc, take_row(xs, Nat.add(off, D()), D())),
        List.append(&2, F32, vacc, take_row(xs, Nat.add(off, Nat.mul(D(), 2n)), D())))

def split_res(t: T.Three<List<&2, F32>, List<&2, F32>, List<&2, F32>>) -> T.Fwd<Unit, QKV(), Unit>:
  match t:
    case T.Three{q, k, v}:
      T.Fwd{Unit{}, T.Two{TA.Mat.from_list(N(), D(), q),
        T.Two{TA.Mat.from_list(N(), D(), k), TA.Mat.from_list(N(), D(), v)}}, Unit{}}

def split_fwd(x: C()) -> T.Fwd<Unit, QKV(), Unit>:
  match x:
    case TA.Mat{a}:
      split_res(split_go(N(), 0n, mlW(TA.Mat.to_list(N(), W3(), TA.Mat{a})), Nil{}, Nil{}, Nil{}))

def join_go(+left: Nat, +i: Nat, +dq: List<&2, F32>, +dk: List<&2, F32>, +dv: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +off = Nat.mul(i, D())
      +a = List.append(&2, F32, acc, take_row(dq, off, D()))
      +b = List.append(&2, F32, a, take_row(dk, off, D()))
      +c = List.append(&2, F32, b, take_row(dv, off, D()))
      join_go(p, (i + 1n : Nat), dq, dk, dv, c)

def split_bwd3(dqm: M(), dkm: M(), dvm: M()) -> T.Bwd<Unit, C()>:
  match dqm dkm dvm:
    case TA.Mat{dq} TA.Mat{dk} TA.Mat{dv}:
      T.Bwd{Unit{}, TA.Mat.from_list(N(), W3(), join_go(N(), 0n,
        mlD(TA.Mat.to_list(N(), D(), TA.Mat{dq})),
        mlD(TA.Mat.to_list(N(), D(), TA.Mat{dk})),
        mlD(TA.Mat.to_list(N(), D(), TA.Mat{dv})), Nil{})), Unit{}}

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

# ---- the core layers ----

type MHS is Type:
  MHS{q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>}

def lin_fwd_of(+q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> T.Fwd<Unit, M(), MHS>:
  T.Fwd{Unit{}, TA.Mat.from_list(N(), D(), lin_y4(q, k, v)), MHS{q, k, v}}

def lin_fwd3(qm: M(), km: M(), vm: M()) -> T.Fwd<Unit, M(), MHS>:
  lin_fwd_of(mlD(TA.Mat.to_list(N(), D(), qm)), mlD(TA.Mat.to_list(N(), D(), km)),
    mlD(TA.Mat.to_list(N(), D(), vm)))

def lin_fwd(qkv: QKV()) -> T.Fwd<Unit, M(), MHS>:
  match qkv:
    case T.Two{qm, T.Two{km, vm}}:
      lin_fwd3(qm, km, vm)

def bwd_mats(acc: ACC) -> T.Bwd<Unit, QKV()>:
  match acc:
    case ACC{dq, dk, dv}:
      T.Bwd{Unit{}, T.Two{TA.Mat.from_list(N(), D(), dq),
        T.Two{TA.Mat.from_list(N(), D(), dk), TA.Mat.from_list(N(), D(), dv)}}, Unit{}}

def lin_bwd_of(dy: M(), s: MHS) -> T.Bwd<Unit, QKV()>:
  match dy s:
    case TA.Mat{d} MHS{q, k, v}:
      bwd_mats(lin_bwd4(mlD(TA.Mat.to_list(N(), D(), TA.Mat{d})), q, k, v))

def LinCore() -> T.Layer<Unit, QKV(), M(), MHS>:
  T.Layer{p => x => lin_fwd(x),
    p => s => dy => g => lin_bwd_of(dy, s),
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0}

def sp_fwd_of(+q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>) -> T.Fwd<Unit, M(), MHS>:
  T.Fwd{Unit{}, TA.Mat.from_list(N(), D(), sp_y4(q, k, v)), MHS{q, k, v}}

def sp_fwd3(qm: M(), km: M(), vm: M()) -> T.Fwd<Unit, M(), MHS>:
  sp_fwd_of(mlD(TA.Mat.to_list(N(), D(), qm)), mlD(TA.Mat.to_list(N(), D(), km)),
    mlD(TA.Mat.to_list(N(), D(), vm)))

def sp_fwd(qkv: QKV()) -> T.Fwd<Unit, M(), MHS>:
  match qkv:
    case T.Two{qm, T.Two{km, vm}}:
      sp_fwd3(qm, km, vm)

def sp_bwd_of(dy: M(), s: MHS) -> T.Bwd<Unit, QKV()>:
  match dy s:
    case TA.Mat{d} MHS{q, k, v}:
      bwd_mats(sp_bwd4(mlD(TA.Mat.to_list(N(), D(), TA.Mat{d})), q, k, v))

def SpCore() -> T.Layer<Unit, QKV(), M(), MHS>:
  T.Layer{p => x => sp_fwd(x),
    p => s => dy => g => sp_bwd_of(dy, s),
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0}

# ---- the attention block: qkv, split, core, out ----

def AttP() -> Type:
  T.Two<TA.Mat<W3(), D()>, T.Two<Unit, T.Two<Unit, TA.Mat<D(), D()>>>>

def LinAttS() -> Type:
  T.Two<M(), T.Two<Unit, T.Two<MHS, M()>>>

def LinCoreOut() -> T.Layer<T.Two<Unit, TA.Mat<D(), D()>>, QKV(), M(), T.Two<MHS, M()>>:
  T.Seq(~Unit, ~TA.Mat<D(), D()>, ~QKV(), ~M(), ~M(), ~MHS, ~M(), ~LinCore(),
    ~T.LinearNB(N(), D(), D(), 0n))

def LinSplitCore() -> T.Layer<T.Two<Unit, T.Two<Unit, TA.Mat<D(), D()>>>, C(), M(),
    T.Two<Unit, T.Two<MHS, M()>>>:
  T.Seq(~Unit, ~T.Two<Unit, TA.Mat<D(), D()>>, ~C(), ~QKV(), ~M(), ~Unit(), ~T.Two<MHS, M()>,
    ~Split3(), ~LinCoreOut())

def LinAttention() -> T.Layer<AttP(), M(), M(), LinAttS()>:
  T.Seq(~TA.Mat<W3(), D()>, ~T.Two<Unit, T.Two<Unit, TA.Mat<D(), D()>>>, ~M(), ~C(), ~M(), ~M(),
    ~T.Two<Unit, T.Two<MHS, M()>>,
    ~T.LinearNB(N(), D(), W3(), 0n), ~LinSplitCore())

def SpCoreOut() -> T.Layer<T.Two<Unit, TA.Mat<D(), D()>>, QKV(), M(), T.Two<MHS, M()>>:
  T.Seq(~Unit, ~TA.Mat<D(), D()>, ~QKV(), ~M(), ~M(), ~MHS, ~M(), ~SpCore(),
    ~T.LinearNB(N(), D(), D(), 0n))

def SpSplitCore() -> T.Layer<T.Two<Unit, T.Two<Unit, TA.Mat<D(), D()>>>, C(), M(),
    T.Two<Unit, T.Two<MHS, M()>>>:
  T.Seq(~Unit, ~T.Two<Unit, TA.Mat<D(), D()>>, ~C(), ~QKV(), ~M(), ~Unit(), ~T.Two<MHS, M()>,
    ~Split3(), ~SpCoreOut())

def SpAttention() -> T.Layer<AttP(), M(), M(), LinAttS()>:
  T.Seq(~TA.Mat<W3(), D()>, ~T.Two<Unit, T.Two<Unit, TA.Mat<D(), D()>>>, ~M(), ~C(), ~M(), ~M(),
    ~T.Two<Unit, T.Two<MHS, M()>>,
    ~T.LinearNB(N(), D(), W3(), 0n), ~SpSplitCore())
