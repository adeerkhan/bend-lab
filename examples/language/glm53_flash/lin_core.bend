# Linear attention core of the reference, list form. q,k,v are 128 x 32
# (2 sequences of 64 positions, 4 heads of 8). The core is parameterless.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def M() -> Type:
  TA.Mat<128n, 32n>

type QKV is Type:
  QKV{q: M(), k: M(), v: M()}

type LinS is Type:
  LinS{q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>}

type PK is Data:
  PK{+kp: List<&2, F32>, +kvp: List<&2, F32>}

def nth(+n: Nat, +xs: List<&2, F32>) -> F32:
  match n xs:
    case 0n Con{h, t}: h
    case 1n+p Con{h, t}: nth(p, t)
    case _ _: 0.0

def list8(+acc: List<&2, F32>) -> List<&2, F32>:
  match acc:
    case Con{h, t}: acc
    case Nil{}: Nil{}

def zeros8() -> List<&2, F32>:
  Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Nil{}}}}}}}}}

def zeros64_go(+left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n: acc
    case 1n+p: zeros64_go(p, Con{0.0, acc})

def zeros64() -> List<&2, F32>:
  zeros64_go(64n, Nil{})

def elu_if(le: Bool, +x: F32) -> F32:
  match le:
    case True{}: (F32.exp(x) - 1.0 : F32)
    case False{}: x

def elu(+x: F32) -> F32:
  elu_if(F32.is_le(x, 0.0), x)

def elu1(+x: F32) -> F32:
  (elu(x) + 1.0 : F32)

def den_if(le: Bool, +x: F32) -> F32:
  match le:
    case True{}: 0.000001
    case False{}: x

def den1(+x: F32) -> F32:
  den_if(F32.is_le(x, 0.000001), x)

def upd_kp(+left: Nat, +d: Nat, +base: Nat, +ka: List<&2, F32>, +kp: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n: kp
    case 1n+p:
      +v = (nth(d, kp) + elu1(nth(Nat.add(base, d), ka)) : F32)
      upd_kp(p, Nat.add(d, 1n), base, ka, List.set(&2, F32, kp, d, v))

def upd_kv_e(+left: Nat, +e: Nat, +d: Nat, +kp: F32, +base: Nat, +va: List<&2, F32>, +kv: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n: kv
    case 1n+p:
      +i = Nat.add(Nat.mul(d, 8n), e)
      +v = (nth(i, kv) + (kp * nth(Nat.add(base, e), va) : F32) : F32)
      upd_kv_e(p, Nat.add(e, 1n), d, kp, base, va, List.set(&2, F32, kv, i, v))

def upd_kv_d(+left: Nat, +d: Nat, +base: Nat, +ka: List<&2, F32>, +va: List<&2, F32>, +kv: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n: kv
    case 1n+p:
      +kp = elu1(nth(Nat.add(base, d), ka))
      upd_kv_d(p, Nat.add(d, 1n), base, ka, va, upd_kv_e(8n, 0n, d, kp, base, va, kv))

def prefix_go(+left: Nat, +s: Nat, +b: Nat, +h: Nat, +ka: List<&2, F32>, +va: List<&2, F32>, pk: PK) -> PK:
  match left pk:
    case 0n _: pk
    case 1n+p PK{kp, kvp}:
      +base = Nat.add(Nat.mul(Nat.add(Nat.mul(b, 64n), s), 32n), Nat.mul(h, 8n))
      prefix_go(p, Nat.add(s, 1n), b, h, ka, va, PK{upd_kp(8n, 0n, base, ka, kp), upd_kv_d(8n, 0n, base, ka, va, kvp)})

def prefix_to(+t: Nat, +b: Nat, +h: Nat, +ka: List<&2, F32>, +va: List<&2, F32>) -> PK:
  prefix_go(Nat.add(t, 1n), 0n, b, h, ka, va, PK{zeros8(), zeros64()})

def kdot(+left: Nat, +d: Nat, +base: Nat, +qa: List<&2, F32>, +kp: List<&2, F32>, acc: F32) -> F32:
  match left:
    case 0n: acc
    case 1n+p:
      kdot(p, Nat.add(d, 1n), base, qa, kp,
        (acc + (elu1(nth(Nat.add(base, d), qa)) * nth(d, kp) : F32) : F32))

def num_e(+left: Nat, +d: Nat, +base: Nat, +qa: List<&2, F32>, +kvp: List<&2, F32>, +e: Nat, acc: F32) -> F32:
  match left:
    case 0n: acc
    case 1n+p:
      num_e(p, Nat.add(d, 1n), base, qa, kvp, e,
        (acc + (elu1(nth(Nat.add(base, d), qa)) * nth(Nat.add(Nat.mul(d, 8n), e), kvp) : F32) : F32))

def out_e(+left: Nat, +e: Nat, +base: Nat, +qa: List<&2, F32>, +kvp: List<&2, F32>, +den: F32, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n: acc
    case 1n+p:
      +v = (num_e(8n, 0n, base, qa, kvp, e, 0.0) / den : F32)
      out_e(p, Nat.add(e, 1n), base, qa, kvp, den, Con{v, acc})

def out_one(pk: PK, +h: Nat, +t: Nat, +b: Nat, +qa: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match pk:
    case PK{kp, kvp}:
      +base = Nat.add(Nat.mul(Nat.add(Nat.mul(b, 64n), t), 32n), Nat.mul(h, 8n))
      +den = den1(kdot(8n, 0n, base, qa, kp, 0.0))
      out_e(8n, 0n, base, qa, kvp, den, acc)

def out_t(+left: Nat, +t: Nat, +b: Nat, +ka: List<&2, F32>, +va: List<&2, F32>, +qa: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n: acc
    case 1n+p:
      +p0 = prefix_to(t, b, 0n, ka, va)
      +p1 = prefix_to(t, b, 1n, ka, va)
      +p2 = prefix_to(t, b, 2n, ka, va)
      +p3 = prefix_to(t, b, 3n, ka, va)
      +a1 = out_one(p0, 0n, t, b, qa, acc)
      +a2 = out_one(p1, 1n, t, b, qa, a1)
      +a3 = out_one(p2, 2n, t, b, qa, a2)
      +a4 = out_one(p3, 3n, t, b, qa, a3)
      out_t(p, Nat.add(t, 1n), b, ka, va, qa, a4)

def out_b(+b: Nat, +qa: List<&2, F32>, +ka: List<&2, F32>, +va: List<&2, F32>) -> List<&2, F32>:
  List.reverse(&2, F32, out_t(64n, 0n, b, ka, va, qa, Nil{}))

def ml_list(-r: Nat, -c: Nat, ml: TA.ML<r, c>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, l}: l

def lin_fwd(q: M(), k: M(), v: M()) -> T.Fwd<Unit, M(), LinS>:
  match q k v:
    case TA.Mat{qa} TA.Mat{ka} TA.Mat{va}:
      +ql = ml_list(128n, 32n, TA.Mat.to_list(128n, 32n, TA.Mat{qa}))
      +kl = ml_list(128n, 32n, TA.Mat.to_list(128n, 32n, TA.Mat{ka}))
      +vl = ml_list(128n, 32n, TA.Mat.to_list(128n, 32n, TA.Mat{va}))
      +out = List.append(&2, F32, out_b(0n, ql, kl, vl), out_b(1n, ql, kl, vl))
      T.Fwd{Unit{}, TA.Mat.from_list(128n, 32n, out), LinS{ql, kl, vl}}



