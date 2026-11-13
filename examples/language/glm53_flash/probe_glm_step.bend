# The 192-position attention cores fed one token at a time.
#
# The full-prefix cores are the oracle: `lin_y_fast` carries the causal linear
# prefix state, and `sp_y_go` computes the sparse causal rows. The step
# versions carry the same state explicitly — `K` and `KV` for the linear core,
# a post-RoPE K/V cache for the sparse core — and must produce the same row at
# every position. No weights are involved, so a failure is the cache protocol,
# not the model. This is the first gate before the 4-block decode path.
import Base
import ./attn_mh192.bend as A
import ./attn_c.bend as AC
import ./lin_fast.bend as LF
import ./lin_bwd.bend as L

def val(+i: Nat, +seed: Nat) -> F32:
  +h = Nat.mod(Nat.add(Nat.mul(i, 37n), Nat.add(Nat.mul(seed, 101n), 11n)), 1009n)
  (((F32.from_nat(h) / 1009.0 : F32) - 0.5 : F32))

def ramp_go(+left: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ramp_go(p, (i + 1n : Nat), seed, Con{val(i, seed), acc})

def ramp(+n: Nat, +seed: Nat) -> List<&2, F32>:
  List.reverse(&2, F32, ramp_go(n, 0n, seed, Nil{}))

# ---- the linear core: prefix K and KV, one row at a time ----

type LS is Type:
  LS{K: List<&2, F32>, KV: List<&2, F32>, ys: List<&2, F32>}

def lin_step(+left: Nat, +t: Nat, +hd: Nat, +q0: List<&2, F32>,
    +k0: List<&2, F32>, +v0: List<&2, F32>, st: LS) -> LS:
  match left st:
    case 0n _:
      st
    case 1n+p LS{K, KV, ys}:
      +off = Nat.mul(t, hd)
      +qe = L.elu_row(A.rope_row(L.take_row(q0, off, hd), t))
      +ke = L.elu_row(A.rope_row(L.take_row(k0, off, hd), t))
      +vr = L.take_row(v0, off, hd)
      +K2 = L.add_lists(K, ke)
      +KV2 = L.add_outer(KV, ke, vr, hd)
      +c = L.clamp_den(L.dot(qe, K2))
      +num = L.matvec_t(KV2, qe, hd)
      +yr = L.scale_list(num, (1.0 / c : F32))
      lin_step(p, (t + 1n : Nat), hd, q0, k0, v0,
        LS{K2, KV2, L.push_row(ys, yr)})

# ---- the sparse core: the post-RoPE K/V cache, one row at a time ----

type SS is Type:
  SS{q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>,
     ys: List<&2, F32>}

def sp_step(+left: Nat, +t: Nat, +hd: Nat, +q0: List<&2, F32>,
    +k0: List<&2, F32>, +v0: List<&2, F32>, st: SS) -> SS:
  match left st:
    case 0n _:
      st
    case 1n+p SS{qc, kc, vc, ys}:
      +off = Nat.mul(t, hd)
      +qr = A.rope_row(L.take_row(q0, off, hd), t)
      +kr = A.rope_row(L.take_row(k0, off, hd), t)
      +vr = L.take_row(v0, off, hd)
      +q2 = L.push_row(qc, qr)
      +k2 = L.push_row(kc, kr)
      +v2 = L.push_row(vc, vr)
      +yr = AC.srow_at(t, q2, k2, v2)
      sp_step(p, (t + 1n : Nat), hd, q0, k0, v0,
        SS{q2, k2, v2, L.push_row(ys, yr)})

def maxdiff(xs: List<&2, F32>, ys: List<&2, F32>, acc: F32) -> F32:
  match xs ys:
    case Nil{} _:
      acc
    case _ Nil{}:
      acc
    case Con{h, t} Con{g, u}:
      maxdiff(t, u, F32.max(acc, F32.abs((h - g : F32))))

def verdict(+dlin: F32, +dsp: F32, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("positions ", String.append(Nat.show(n),
      String.append("  lin max|d| ", F32.show(dlin)))))
    Unit <- IO.print(String.append("           sp  max|d| ", F32.show(dsp)))
    Unit <- IO.print(Bool.pick(String,
      Bool.and(F32.is_le(dlin, 1.0e-06), F32.is_le(dsp, 1.0e-06)),
      "GLM STEP PASS", "GLM STEP FAIL"))
    return Unit{}

def finish(lin_st: LS, sp_st: SS, +ylin_full: List<&2, F32>,
    +ysp_full: List<&2, F32>, +n: Nat) -> IO(Unit):
  match lin_st:
    case LS{K, KV, ys}:
      match sp_st:
        case SS{qc, kc, vc, yss}:
          +dlin = maxdiff(ys, ylin_full, 0.0)
          +dsp = maxdiff(yss, ysp_full, 0.0)
          verdict(dlin, dsp, n)

def run(+n: Nat, +hd: Nat) -> IO(Unit):
  +q = ramp(Nat.mul(n, hd), 1n)
  +k = ramp(Nat.mul(n, hd), 2n)
  +v = ramp(Nat.mul(n, hd), 3n)
  +ylin_full = LF.lin_y_fast(n, hd, A.rope_all(q), A.rope_all(k), v)
  +ysp_full = AC.sp_y_go(n, 0n, A.rope_all(q), A.rope_all(k), v, Nil{})
  lin_st = lin_step(n, 0n, hd, q, k, v,
    LS{L.zline(hd), L.zline(Nat.mul(hd, hd)), Nil{}})
  sp_st = sp_step(n, 0n, hd, q, k, v, SS{Nil{}, Nil{}, Nil{}, Nil{}})
  finish(lin_st, sp_st, ylin_full, ysp_full, n)

def main() -> IO(Unit):
  do IO<Unit>:
    run(192n, 8n)
