# The prefix-state causal linear attention, single head.
#
# Same forward and backward as `lin_bwd.bend`, with the prefix K_t/KV_t and
# the suffix N_s/Q_s carried as state instead of recomputed.  The direct core
# is the oracle; this one removes the direct O(T²·d) sums.
import Base
import ./lin_bwd.bend as L

type FP is Type:
  FP{K: List<&2, F32>, KV: List<&2, F32>, kall: List<&2, F32>,
     kvall: List<&2, F32>, dens: List<&2, F32>, nums: List<&2, F32>,
     qes: List<&2, F32>, kes: List<&2, F32>, vs: List<&2, F32>,
     dn: List<&2, F32>, dd: List<&2, F32>, ys: List<&2, F32>}

# fp_fin builds the extended record.  Every value is a + parameter, so the
# pushes and the next prefix can share it.
def fp_fin(+K2: List<&2, F32>, +KV2: List<&2, F32>, +qe: List<&2, F32>,
    +ke: List<&2, F32>, +vr: List<&2, F32>, +den: F32, +num: List<&2, F32>,
    +dnr: List<&2, F32>, +ddr: F32, +yr: List<&2, F32>,
    +kall: List<&2, F32>, +kvall: List<&2, F32>, +dens: List<&2, F32>,
    +nums: List<&2, F32>, +qes: List<&2, F32>, +kes: List<&2, F32>,
    +vs: List<&2, F32>, +dn: List<&2, F32>, +dd: List<&2, F32>,
    +ys: List<&2, F32>) -> FP:
  FP{K2, KV2, L.push_row(kall, K2), L.push_row(kvall, KV2),
     L.push_row(dens, Con{den, Nil{}}), L.push_row(nums, num),
     L.push_row(qes, qe), L.push_row(kes, ke), L.push_row(vs, vr),
     L.push_row(dn, dnr), L.push_row(dd, Con{ddr, Nil{}}), L.push_row(ys, yr)}

def fp_calc(+K2: List<&2, F32>, +KV2: List<&2, F32>, +qe: List<&2, F32>,
    +ke: List<&2, F32>, +vr: List<&2, F32>, +dyr: List<&2, F32>, +hd: Nat,
    +kall: List<&2, F32>, +kvall: List<&2, F32>, +dens: List<&2, F32>,
    +nums: List<&2, F32>, +qes: List<&2, F32>, +kes: List<&2, F32>,
    +vs: List<&2, F32>, +dn: List<&2, F32>, +dd: List<&2, F32>,
    +ys: List<&2, F32>) -> FP:
  +den = L.dot(qe, K2)
  +c = L.clamp_den(den)
  +num = L.matvec_t(KV2, qe, hd)
  +dnr = L.scale_list(dyr, (1.0 / c : F32))
  +ddr = L.dden_of(L.active_den(den), L.dot(dyr, num), c)
  +yr = L.scale_list(num, (1.0 / c : F32))
  fp_fin(K2, KV2, qe, ke, vr, den, num, dnr, ddr, yr,
    kall, kvall, dens, nums, qes, kes, vs, dn, dd, ys)

def fp_prefix(+K: List<&2, F32>, +KV: List<&2, F32>, +ke: List<&2, F32>,
    +vr: List<&2, F32>, +qe: List<&2, F32>, +dyr: List<&2, F32>, +hd: Nat,
    +kall: List<&2, F32>, +kvall: List<&2, F32>, +dens: List<&2, F32>,
    +nums: List<&2, F32>, +qes: List<&2, F32>, +kes: List<&2, F32>,
    +vs: List<&2, F32>, +dn: List<&2, F32>, +dd: List<&2, F32>,
    +ys: List<&2, F32>) -> FP:
  +K2 = L.add_lists(K, ke)
  +KV2 = L.add_outer(KV, ke, vr, hd)
  fp_calc(K2, KV2, qe, ke, vr, dyr, hd, kall, kvall, dens, nums, qes, kes, vs, dn, dd, ys)

def fp_go(+left: Nat, +t: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, +dy: List<&2, F32>, st: FP) -> FP:
  match left st:
    case 0n _:
      st
    case 1n+p FP{K, KV, kall, kvall, dens, nums, qes, kes, vs, dn, dd, ys}:
      +off = Nat.mul(t, hd)
      +qe = L.elu_row(L.take_row(q, off, hd))
      +ke = L.elu_row(L.take_row(k, off, hd))
      +vr = L.take_row(v, off, hd)
      +dyr = L.take_row(dy, off, hd)
      fp_go(p, (t + 1n : Nat), hd, q, k, v, dy,
        fp_prefix(K, KV, ke, vr, qe, dyr, hd, kall, kvall, dens, nums, qes, kes, vs, dn, dd, ys))

def fp_start(+n: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, +dy: List<&2, F32>) -> FP:
  fp_go(n, 0n, hd, q, k, v, dy,
    FP{L.zline(hd), L.zline(Nat.mul(hd, hd)), Nil{}, Nil{}, Nil{}, Nil{},
       Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}})

def fp_ys(fp: FP) -> List<&2, F32>:
  match fp:
    case FP{K, KV, kall, kvall, dens, nums, qes, kes, vs, dn, dd, ys}:
      ys

def lin_y_fast(+n: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>) -> List<&2, F32>:
  fp_ys(fp_start(n, hd, q, k, v, L.zline(Nat.mul(n, hd))))

def put_go(row: List<&2, F32>, acc: List<&2, F32>, +off: Nat, +i: Nat) -> List<&2, F32>:
  match row:
    case Nil{}:
      acc
    case Con{h, t}:
      put_go(t, List.set(&2, F32, acc, Nat.add(off, i), h), off, (i + 1n : Nat))

def put_row(acc: List<&2, F32>, +off: Nat, row: List<&2, F32>) -> List<&2, F32>:
  put_go(row, acc, off, 0n)

type FWB is Type:
  FWB{dq: List<&2, F32>, dk: List<&2, F32>, dv: List<&2, F32>,
      N: List<&2, F32>, Q: List<&2, F32>}

def bw_fill(+N2: List<&2, F32>, +Q2: List<&2, F32>, +dnr: List<&2, F32>,
    +qer: List<&2, F32>, +ker: List<&2, F32>, +vr: List<&2, F32>,
    +Kr: List<&2, F32>, +KVr: List<&2, F32>, +dden: F32,
    +s: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>, st: FWB) -> FWB:
  match st:
    case FWB{dq, dk, dv, N, Q}:
      +off = Nat.mul(s, hd)
      +dv_s = L.matvec(N2, ker, hd)
      +dk_s0 = L.add_lists(L.matvec_t(N2, vr, hd), Q2)
      +dq_s0 = L.add_lists(L.matvec(KVr, dnr, hd), L.scale_list(Kr, dden))
      +dq_s = L.mul_lists(dq_s0, L.deriv_row(L.take_row(q, off, hd)))
      +dk_s = L.mul_lists(dk_s0, L.deriv_row(L.take_row(k, off, hd)))
      FWB{put_row(dq, off, dq_s), put_row(dk, off, dk_s), put_row(dv, off, dv_s),
          N2, Q2}

def bw_go(+left: Nat, +s: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    fp: FP, st: FWB) -> FWB:
  match left fp st:
    case 0n _ _:
      st
    case 1n+p FP{K, KV, +kall, +kvall, dens, nums, +qes, +kes, +vs, +dn, +dd, ys} FWB{dq, dk, dv, +N, +Q}:
      +off = Nat.mul(s, hd)
      +dnr = L.take_row(dn, off, hd)
      +qer = L.take_row(qes, off, hd)
      +ker = L.take_row(kes, off, hd)
      +vr = L.take_row(vs, off, hd)
      +Kr = L.take_row(kall, off, hd)
      +KVr = L.take_row(kvall, Nat.mul(off, hd), Nat.mul(hd, hd))
      +dden = L.nth(0n, L.take_row(dd, s, 1n))
      +N2 = L.add_outer(N, dnr, qer, hd)
      +Q2 = L.add_lists(Q, L.scale_list(qer, dden))
      bw_go(p, (s - 1n : Nat), hd, q, k,
        FP{K, KV, kall, kvall, dens, nums, qes, kes, vs, dn, dd, ys},
        bw_fill(N2, Q2, dnr, qer, ker, vr, Kr, KVr, dden, s, hd, q, k,
          FWB{dq, dk, dv, N2, Q2}))

def bw_out(fwb: FWB) -> L.BW:
  match fwb:
    case FWB{dq, dk, dv, N, Q}:
      L.BW{dq, dk, dv}

def lin_bwd_fast(+n: Nat, +hd: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, +dy: List<&2, F32>) -> L.BW:
  bw_out(bw_go(n, Nat.sub(n, 1n), hd, q, k, fp_start(n, hd, q, k, v, dy),
    FWB{L.zline(Nat.mul(n, hd)), L.zline(Nat.mul(n, hd)), L.zline(Nat.mul(n, hd)),
        L.zline(Nat.mul(hd, hd)), L.zline(hd)}))
