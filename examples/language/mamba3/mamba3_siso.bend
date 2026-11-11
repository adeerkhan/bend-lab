# Mamba-3 SISO, one layer, exactly as the vendored kernel loop.
#
# Shapes: d_model 64, d_state 16, expand 2, headdim 32 -> 4 heads,
# ngroups 1, rope_fraction 0.5 -> 4 angle pairs, batch 1, seqlen 8.
#
# `reference/mamba3_twin.py` writes the goldens and `check_mamba3_bend.py`
# compares this program against them.  The forward is the exact recurrent
# loop, not a chunked or associative scan:
#
#   A = -heavy_tail(dd_A)              (clamped at -1e-4)
#   DT = softplus(dd_dt + dt_bias), ADT = A*DT
#   trap = sigmoid(trap_raw), gamma = DT*trap
#   shifted_t = DT_{t+1}*(1-trap_{t+1}), scale = gamma + shifted
#   angle_t = mod2pi(angle_{t-1} + tanh(raw)*pi*DT_t)
#   q = rope(C_norm + C_bias), k = rope(B_norm + B_bias)
#   o_t = exp(ADT_t)*(q_t S_t^T) + (D + gamma_t*(q_t.k_t))*x_t
#   S_{t+1} = exp(ADT_t)*S_t + x_t (x) (scale_t*k_t)
#   y = (o * silu(z)) @ out_proj^T
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# ---------------------------------------------------------------------------
# list helpers, flat row-major

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

def push_row(acc: List<&2, F32>, +row: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, acc, row)

def add_lists(+xs: List<&2, F32>, +ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Nil{} _:
      Nil{}
    case _ Nil{}:
      Nil{}
    case Con{h, t} Con{g, u}:
      Con{(h + g : F32), add_lists(t, u)}

def sub_lists(+xs: List<&2, F32>, +ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Nil{} _:
      Nil{}
    case _ Nil{}:
      Nil{}
    case Con{h, t} Con{g, u}:
      Con{(h - g : F32), sub_lists(t, u)}

def mul_lists(+xs: List<&2, F32>, +ys: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Nil{} _:
      Nil{}
    case _ Nil{}:
      Nil{}
    case Con{h, t} Con{g, u}:
      Con{(h * g : F32), mul_lists(t, u)}

def scale_list(+xs: List<&2, F32>, +c: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(h * c : F32), scale_list(t, c)}

def one_minus(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(1.0 - h : F32), one_minus(t)}

def dot(+xs: List<&2, F32>, +ys: List<&2, F32>) -> F32:
  match xs ys:
    case Nil{} _:
      0.0
    case _ Nil{}:
      0.0
    case Con{h, t} Con{g, u}:
      ((h * g : F32) + dot(t, u) : F32)

def sum_list(+xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      (h + sum_list(t) : F32)

def sumsq(+xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      ((h * h : F32) + sumsq(t) : F32)

def matvec_go(+left: Nat, +m: List<&2, F32>, +xs: List<&2, F32>,
    +w: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +row = take_row(m, 0n, w)
      matvec_go(p, List.drop(&2, F32, m, w), xs, w,
        push_row(acc, Con{dot(row, xs), Nil{}}))

def matvec(+m: List<&2, F32>, +xs: List<&2, F32>, +w: Nat) -> List<&2, F32>:
  +rows = Nat.div(List.length(&2, F32, m), w)
  matvec_go(rows, m, xs, w, Nil{})

def matvec_t_go(+left: Nat, +m: List<&2, F32>, +xs: List<&2, F32>,
    +w: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n _:
      acc
    case 1n+p Con{x, xt}:
      +row = take_row(m, 0n, w)
      matvec_t_go(p, List.drop(&2, F32, m, w), xt, w,
        add_lists(acc, scale_list(row, x)))
    case 1n+p _:
      acc

def matvec_t(+m: List<&2, F32>, +xs: List<&2, F32>, +rows: Nat,
    +w: Nat) -> List<&2, F32>:
  matvec_t_go(rows, m, xs, w, zline(w))

def outer_go(+left: Nat, +xs: List<&2, F32>, +ys: List<&2, F32>,
    +acc: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n _:
      acc
    case 1n+p Con{x, xt}:
      outer_go(p, xt, ys, push_row(acc, scale_list(ys, x)))
    case 1n+p _:
      acc

def outer(+xs: List<&2, F32>, +ys: List<&2, F32>) -> List<&2, F32>:
  outer_go(List.length(&2, F32, xs), xs, ys, Nil{})

# ---------------------------------------------------------------------------
# scalar functions

def htail_if(b: Bool, +x: F32) -> F32:
  match b:
    case True{}:
      (1.0 + x : F32)
    case False{}:
      (1.0 / (1.0 - x : F32) : F32)

def htail(+x: F32) -> F32:
  htail_if(F32.is_ge(x, 0.0), x)

def neg_if(b: Bool, +f: F32) -> F32:
  match b:
    case True{}:
      (0.0 - 0.0001 : F32)
    case False{}:
      (0.0 - f : F32)

def neg_htail(+x: F32) -> F32:
  +f = htail(x)
  neg_if(F32.is_lt(f, 0.0001), f)

def softplus(+x: F32) -> F32:
  F32.log((1.0 + F32.exp(x) : F32))

def sigmoid(+x: F32) -> F32:
  +e = F32.exp((0.0 - x : F32))
  (1.0 / (1.0 + e : F32) : F32)

def silu(+x: F32) -> F32:
  (x * sigmoid(x) : F32)

def rms(+xs: List<&2, F32>, +w: List<&2, F32>, +n: Nat) -> List<&2, F32>:
  +nf = F32.from_nat(n)
  +mean = (sumsq(xs) / nf : F32)
  +r = (1.0 / F32.sqrt((mean + 0.00001 : F32)) : F32)
  mul_lists(xs, scale_list(w, r))

def softplus_row(+xs: List<&2, F32>, +bias: List<&2, F32>) -> List<&2, F32>:
  match xs bias:
    case Nil{} _:
      Nil{}
    case _ Nil{}:
      Nil{}
    case Con{h, t} Con{g, u}:
      Con{softplus((h + g : F32)), softplus_row(t, u)}

def htail_row(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{neg_htail(h), htail_row(t)}

def sigmoid_row(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{sigmoid(h), sigmoid_row(t)}

def silu_row(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{silu(h), silu_row(t)}

def wrap_if(b: Bool, +m: F32, +p: F32) -> F32:
  match b:
    case True{}:
      p
    case False{}:
      m

def wrap(+x: F32) -> F32:
  +m = F32.mod(x, 6.283185307179586)
  wrap_if(F32.is_lt(m, 0.0), m, (m + 6.283185307179586 : F32))

def angle_step_go(+left: Nat, +j: Nat, +prev: List<&2, F32>,
    +raw: List<&2, F32>, +dt: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +a = nth(j, raw)
      +v = ((F32.tanh(a) * 3.141592653589793 : F32) * dt : F32)
      +s = wrap((nth(j, prev) + v : F32))
      angle_step_go(p, (j + 1n : Nat), prev, raw, dt,
        push_row(acc, Con{s, Nil{}}))

def angle_step(+prev: List<&2, F32>, +raw: List<&2, F32>, +dt: F32,
    +n: Nat) -> List<&2, F32>:
  angle_step_go(n, 0n, prev, raw, dt, Nil{})

def rope_go(+left: Nat, +j: Nat, +xs: List<&2, F32>,
    +angles: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +a = nth(j, angles)
      +c = F32.cos(a)
      +s = F32.sin(a)
      +i0 = Nat.mul(2n, j)
      +i1 = (Nat.mul(2n, j) + 1n : Nat)
      +x0 = nth(i0, xs)
      +x1 = nth(i1, xs)
      +y0 = ((x0 * c : F32) - (x1 * s : F32) : F32)
      +y1 = ((x0 * s : F32) + (x1 * c : F32) : F32)
      rope_go(p, (j + 1n : Nat), xs, angles,
        push_row(push_row(acc, Con{y0, Nil{}}), Con{y1, Nil{}}))

def rope(+xs: List<&2, F32>, +angles: List<&2, F32>,
    +pairs: Nat) -> List<&2, F32>:
  rope_go(pairs, 0n, xs, angles, Nil{})

# ---------------------------------------------------------------------------
# pass A: in_proj and the elementwise state

type PRE is Type:
  PRE{pre: List<&2, F32>, z: List<&2, F32>, x: List<&2, F32>,
      Braw: List<&2, F32>, Craw: List<&2, F32>,
      dd_dt: List<&2, F32>, dd_A: List<&2, F32>,
      trap_raw: List<&2, F32>, ang_raw: List<&2, F32>,
      A: List<&2, F32>, DT: List<&2, F32>, ADT: List<&2, F32>,
      trap: List<&2, F32>, Bn: List<&2, F32>, Cn: List<&2, F32>}

def pa_step(p: PRE, +u_t: List<&2, F32>, +w_in: List<&2, F32>,
    +dt_bias: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>) -> PRE:
  match p:
    case PRE{pre0, z0, x0, Braw0, Craw0, dd_dt0, dd_A0, trap_raw0,
        ang_raw0, A0, DT0, ADT0, trap0, Bn0, Cn0}:
      +pre = matvec(w_in, u_t, 64n)
      +z = take_row(pre, 0n, 128n)
      +x = take_row(pre, 128n, 128n)
      +Braw = take_row(pre, 256n, 16n)
      +Craw = take_row(pre, 272n, 16n)
      +dd_dt = take_row(pre, 288n, 4n)
      +dd_A = take_row(pre, 292n, 4n)
      +trap_raw = take_row(pre, 296n, 4n)
      +ang_raw = take_row(pre, 300n, 4n)
      +A = htail_row(dd_A)
      +DT = softplus_row(dd_dt, dt_bias)
      +ADT = mul_lists(A, DT)
      +trap = sigmoid_row(trap_raw)
      +Bn = rms(Braw, Bw, 16n)
      +Cn = rms(Craw, Cw, 16n)
      PRE{push_row(pre0, pre), push_row(z0, z), push_row(x0, x),
        push_row(Braw0, Braw), push_row(Craw0, Craw),
        push_row(dd_dt0, dd_dt), push_row(dd_A0, dd_A),
        push_row(trap_raw0, trap_raw), push_row(ang_raw0, ang_raw),
        push_row(A0, A), push_row(DT0, DT), push_row(ADT0, ADT),
        push_row(trap0, trap), push_row(Bn0, Bn), push_row(Cn0, Cn)}

def pa_go(+left: Nat, +t: Nat, +u: List<&2, F32>, +w_in: List<&2, F32>,
    +dt_bias: List<&2, F32>, +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    p: PRE) -> PRE:
  match left:
    case 0n:
      p
    case 1n+q:
      +u_t = take_row(u, Nat.mul(t, 64n), 64n)
      pa_go(q, (t + 1n : Nat), u, w_in, dt_bias, Bw, Cw,
        pa_step(p, u_t, w_in, dt_bias, Bw, Cw))

def pass_a(+u: List<&2, F32>, +w_in: List<&2, F32>,
    +dt_bias: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +n: Nat) -> PRE:
  pa_go(n, 0n, u, w_in, dt_bias, Bw, Cw,
    PRE{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{},
      Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}})

# ---------------------------------------------------------------------------
# pass B: gamma, shift, scale, running angles, q/k/qk

type QB is Type:
  QB{gamma: List<&2, F32>, shift: List<&2, F32>, scale: List<&2, F32>,
     angle: List<&2, F32>, q: List<&2, F32>, k: List<&2, F32>,
     qk: List<&2, F32>}

type PBR is Type:
  PBR{angle_t: List<&2, F32>, qb: QB}

def pb_heads(+left: Nat, +h: Nat, +angle_prev: List<&2, F32>,
    +Bn_t: List<&2, F32>, +Cn_t: List<&2, F32>,
    +ang_raw_t: List<&2, F32>, +DT_t: List<&2, F32>,
    +gamma_t: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +gamma0: List<&2, F32>,
    +shift0: List<&2, F32>, +scale0: List<&2, F32>,
    +angle0: List<&2, F32>, +q0: List<&2, F32>,
    +k0: List<&2, F32>, +qk0: List<&2, F32>,
    +angle_t: List<&2, F32>) -> PBR:
  match left:
    case 0n:
      PBR{angle_t, QB{gamma0, shift0, scale0, angle0, q0, k0, qk0}}
    case 1n+p:
      +cb = take_row(C_bias, Nat.mul(h, 16n), 16n)
      +bb = take_row(B_bias, Nat.mul(h, 16n), 16n)
      +qpre = add_lists(Cn_t, cb)
      +kpre = add_lists(Bn_t, bb)
      +qk_h = (nth(h, gamma_t) * dot(qpre, kpre) : F32)
      +ap = take_row(angle_prev, Nat.mul(h, 4n), 4n)
      +ang_h = angle_step(ap, ang_raw_t, nth(h, DT_t), 4n)
      +qh = rope(qpre, ang_h, 8n)
      +kh = rope(kpre, ang_h, 8n)
      pb_heads(p, (h + 1n : Nat), angle_prev, Bn_t, Cn_t, ang_raw_t,
        DT_t, gamma_t, B_bias, C_bias, gamma0, shift0, scale0,
        push_row(angle0, ang_h), push_row(q0, qh), push_row(k0, kh),
        push_row(qk0, Con{qk_h, Nil{}}), push_row(angle_t, ang_h))

def pb_step(q: QB, +DT: List<&2, F32>, +trap: List<&2, F32>,
    +Bn: List<&2, F32>, +Cn: List<&2, F32>,
    +ang_raw: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +t: Nat,
    +angle_prev: List<&2, F32>) -> PBR:
  match q:
    case QB{gamma0, shift0, scale0, angle0, q0, k0, qk0}:
      +off = Nat.mul(t, 4n)
      +nxt = Nat.mul((t + 1n : Nat), 4n)
      +DT_t = take_row(DT, off, 4n)
      +DT_n = take_row(DT, nxt, 4n)
      +trap_t = take_row(trap, off, 4n)
      +trap_n = take_row(trap, nxt, 4n)
      +gamma_t = mul_lists(DT_t, trap_t)
      +shift_t = mul_lists(DT_n, one_minus(trap_n))
      +scale_t = add_lists(gamma_t, shift_t)
      +Bn_t = take_row(Bn, Nat.mul(t, 16n), 16n)
      +Cn_t = take_row(Cn, Nat.mul(t, 16n), 16n)
      +ang_raw_t = take_row(ang_raw, off, 4n)
      pb_heads(4n, 0n, angle_prev, Bn_t, Cn_t, ang_raw_t, DT_t,
        gamma_t, B_bias, C_bias, push_row(gamma0, gamma_t),
        push_row(shift0, shift_t), push_row(scale0, scale_t),
        angle0, q0, k0, qk0, Nil{})

def pb_go(+left: Nat, +t: Nat, +DT: List<&2, F32>,
    +trap: List<&2, F32>, +Bn: List<&2, F32>, +Cn: List<&2, F32>,
    +ang_raw: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, r: PBR) -> QB:
  match left r:
    case 0n PBR{angle_prev, qb}:
      qb
    case 1n+p PBR{angle_prev, qb}:
      pb_go(p, (t + 1n : Nat), DT, trap, Bn, Cn, ang_raw, B_bias,
        C_bias, pb_step(qb, DT, trap, Bn, Cn, ang_raw, B_bias,
          C_bias, t, angle_prev))

def pass_b(+DT: List<&2, F32>, +trap: List<&2, F32>,
    +Bn: List<&2, F32>, +Cn: List<&2, F32>,
    +ang_raw: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +n: Nat) -> QB:
  pb_go(n, 0n, DT, trap, Bn, Cn, ang_raw, B_bias, C_bias,
    PBR{Nil{}, QB{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}}})

# ---------------------------------------------------------------------------
# pass C: the recurrent scan and the output projection

type SC is Type:
  SC{states: List<&2, F32>, out_v: List<&2, F32>,
     y_inner: List<&2, F32>, y: List<&2, F32>}

type SH is Type:
  SH{state: List<&2, F32>, out: List<&2, F32>}

type ST is Type:
  ST{state: List<&2, F32>, sc: SC}

def scan_head_go(+left: Nat, +p: Nat, +state_h: List<&2, F32>,
    +q_h: List<&2, F32>, +k_h: List<&2, F32>, +x_h: List<&2, F32>,
    +a: F32, +skip: F32, +sc: F32, +acc_s: List<&2, F32>,
    +acc_o: List<&2, F32>) -> SH:
  match left:
    case 0n:
      SH{acc_s, acc_o}
    case 1n+pr:
      +row = take_row(state_h, Nat.mul(p, 16n), 16n)
      +qs = dot(row, q_h)
      +xv = nth(p, x_h)
      +o = ((a * qs : F32) + (skip * xv : F32) : F32)
      +row2 = add_lists(scale_list(row, a),
        scale_list(k_h, (sc * xv : F32)))
      scan_head_go(pr, (p + 1n : Nat), state_h, q_h, k_h, x_h, a,
        skip, sc, push_row(acc_s, row2), push_row(acc_o, Con{o, Nil{}}))

def scan_head(+state_h: List<&2, F32>, +q_h: List<&2, F32>,
    +k_h: List<&2, F32>, +x_h: List<&2, F32>, +a: F32, +skip: F32,
    +sc: F32, +acc_s: List<&2, F32>,
    +acc_o: List<&2, F32>) -> SH:
  scan_head_go(32n, 0n, state_h, q_h, k_h, x_h, a, skip, sc,
    acc_s, acc_o)

def scan_one(sh: SH, +h: Nat, +state: List<&2, F32>,
    +q_t: List<&2, F32>, +k_t: List<&2, F32>,
    +x_t: List<&2, F32>, +qk_t: List<&2, F32>,
    +scale_t: List<&2, F32>, +ADT_t: List<&2, F32>,
    +D: List<&2, F32>) -> SH:
  match sh:
    case SH{acc_s, acc_o}:
      scan_head(take_row(state, Nat.mul(h, 512n), 512n),
        take_row(q_t, Nat.mul(h, 16n), 16n),
        take_row(k_t, Nat.mul(h, 16n), 16n),
        take_row(x_t, Nat.mul(h, 32n), 32n),
        F32.exp(nth(h, ADT_t)), (nth(h, D) + nth(h, qk_t) : F32),
        nth(h, scale_t), acc_s, acc_o)

def scan_all(+state: List<&2, F32>, +q_t: List<&2, F32>,
    +k_t: List<&2, F32>, +x_t: List<&2, F32>,
    +qk_t: List<&2, F32>, +scale_t: List<&2, F32>,
    +ADT_t: List<&2, F32>, +D: List<&2, F32>) -> SH:
  sh0 : SH = scan_head(take_row(state, 0n, 512n),
    take_row(q_t, 0n, 16n), take_row(k_t, 0n, 16n),
    take_row(x_t, 0n, 32n), F32.exp(nth(0n, ADT_t)),
    (nth(0n, D) + nth(0n, qk_t) : F32), nth(0n, scale_t),
    Nil{}, Nil{})
  sh1 : SH = scan_one(sh0, 1n, state, q_t, k_t, x_t, qk_t,
    scale_t, ADT_t, D)
  sh2 : SH = scan_one(sh1, 2n, state, q_t, k_t, x_t, qk_t,
    scale_t, ADT_t, D)
  scan_one(sh2, 3n, state, q_t, k_t, x_t, qk_t, scale_t, ADT_t, D)

def sc_finish(sh: SH, +states: List<&2, F32>,
    +out_v: List<&2, F32>, +y_inner: List<&2, F32>,
    +y0: List<&2, F32>, +z_t: List<&2, F32>,
    +w_out: List<&2, F32>) -> ST:
  match sh:
    case SH{+state, +out}:
      +yi = mul_lists(out, silu_row(z_t))
      +y = matvec(w_out, yi, 128n)
      ST{state, SC{push_row(states, state), push_row(out_v, out),
        push_row(y_inner, yi), push_row(y0, y)}}

def sc_step(s: SC, +state: List<&2, F32>, +x: List<&2, F32>,
    +z: List<&2, F32>, +ADT: List<&2, F32>, +q: List<&2, F32>,
    +k: List<&2, F32>, +qk: List<&2, F32>,
    +scale: List<&2, F32>, +t: Nat, +D: List<&2, F32>,
    +w_out: List<&2, F32>) -> ST:
  match s:
    case SC{states, out_v, y_inner, y0}:
      +q_t = take_row(q, Nat.mul(t, 64n), 64n)
      +k_t = take_row(k, Nat.mul(t, 64n), 64n)
      +x_t = take_row(x, Nat.mul(t, 128n), 128n)
      +qk_t = take_row(qk, Nat.mul(t, 4n), 4n)
      +scale_t = take_row(scale, Nat.mul(t, 4n), 4n)
      +ADT_t = take_row(ADT, Nat.mul(t, 4n), 4n)
      +z_t = take_row(z, Nat.mul(t, 128n), 128n)
      sc_finish(scan_all(state, q_t, k_t, x_t, qk_t, scale_t,
        ADT_t, D), states, out_v, y_inner, y0, z_t, w_out)

def sc_go(+left: Nat, +t: Nat, +x: List<&2, F32>,
    +z: List<&2, F32>, +ADT: List<&2, F32>, +q: List<&2, F32>,
    +k: List<&2, F32>, +qk: List<&2, F32>,
    +scale: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, st: ST) -> SC:
  match left st:
    case 0n ST{state, s}:
      s
    case 1n+p ST{state, s}:
      sc_go(p, (t + 1n : Nat), x, z, ADT, q, k, qk, scale, D,
        w_out, sc_step(s, state, x, z, ADT, q, k, qk, scale, t, D,
          w_out))

def pass_c(+x: List<&2, F32>, +z: List<&2, F32>,
    +ADT: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>,
    +qk: List<&2, F32>, +scale: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>, +n: Nat) -> SC:
  +state0 = zline(2048n)
  sc_go(n, 0n, x, z, ADT, q, k, qk, scale, D, w_out,
    ST{state0, SC{state0, Nil{}, Nil{}, Nil{}}})

# ---------------------------------------------------------------------------
# the whole forward

type FWD is Type:
  FWD{pre: List<&2, F32>, z: List<&2, F32>, x: List<&2, F32>,
      Braw: List<&2, F32>, Craw: List<&2, F32>,
      dd_dt: List<&2, F32>, dd_A: List<&2, F32>,
      trap_raw: List<&2, F32>, ang_raw: List<&2, F32>,
      A: List<&2, F32>, DT: List<&2, F32>, ADT: List<&2, F32>,
      trap: List<&2, F32>, gamma: List<&2, F32>,
      shift: List<&2, F32>, scale: List<&2, F32>,
      Bn: List<&2, F32>, Cn: List<&2, F32>,
      angle: List<&2, F32>, q: List<&2, F32>, k: List<&2, F32>,
      qk: List<&2, F32>, states: List<&2, F32>,
      out_v: List<&2, F32>, y_inner: List<&2, F32>,
      y: List<&2, F32>}

def forward_sc(sc: SC, +pre: List<&2, F32>, +z: List<&2, F32>,
    +x: List<&2, F32>, +Braw: List<&2, F32>,
    +Craw: List<&2, F32>, +dd_dt: List<&2, F32>,
    +dd_A: List<&2, F32>, +trap_raw: List<&2, F32>,
    +ang_raw: List<&2, F32>, +A: List<&2, F32>,
    +DT: List<&2, F32>, +ADT: List<&2, F32>,
    +trap: List<&2, F32>, +gamma: List<&2, F32>,
    +shift: List<&2, F32>, +scale: List<&2, F32>,
    +Bn: List<&2, F32>, +Cn: List<&2, F32>,
    +angle: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>,
    +qk: List<&2, F32>) -> FWD:
  match sc:
    case SC{states, out_v, y_inner, y}:
      FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, y}

def forward_qb(qb: QB, +pre: List<&2, F32>, +z: List<&2, F32>,
    +x: List<&2, F32>, +Braw: List<&2, F32>,
    +Craw: List<&2, F32>, +dd_dt: List<&2, F32>,
    +dd_A: List<&2, F32>, +trap_raw: List<&2, F32>,
    +ang_raw: List<&2, F32>, +A: List<&2, F32>,
    +DT: List<&2, F32>, +ADT: List<&2, F32>,
    +trap: List<&2, F32>, +Bn: List<&2, F32>,
    +Cn: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +n: Nat) -> FWD:
  match qb:
    case QB{gamma, shift, +scale, angle, +q, +k, +qk}:
      sc : SC = pass_c(x, z, ADT, q, k, qk, scale, D, w_out, n)
      +DTu = take_row(DT, 0n, Nat.mul(n, 4n))
      +trapu = take_row(trap, 0n, Nat.mul(n, 4n))
      forward_sc(sc, pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw,
        ang_raw, A, DTu, ADT, trapu, gamma, shift, scale, Bn, Cn,
        angle, q, k, qk)

def forward_pre(pr: PRE, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +n: Nat) -> FWD:
  match pr:
    case PRE{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, +ang_raw,
        A, +DT, ADT, +trap, +Bn, +Cn}:
      +DT2 = push_row(DT, zline(4n))
      +trap2 = push_row(trap, zline(4n))
      qb : QB = pass_b(DT2, trap2, Bn, Cn, ang_raw, B_bias, C_bias, n)
      forward_qb(qb, pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw,
        ang_raw, A, DT2, ADT, trap2, Bn, Cn, D, w_out, n)

def forward(+u: List<&2, F32>, +w_in: List<&2, F32>,
    +dt_bias: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +n: Nat) -> FWD:
  forward_pre(pass_a(u, w_in, dt_bias, Bw, Cw, n), B_bias, C_bias,
    D, w_out, n)


def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

def show_tag(+tag: String, +xs: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append(String.append(tag, ": "), show(xs)))
    return Unit{}

def dir() -> String:
  "reference/mamba3_data"

def run(f: FWD) -> IO(Unit):
  match f:
    case FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, +ADT, trap, gamma, shift, scale, Bn, Cn, angle, +q, +k,
        +qk, states, out_v, y_inner, y}:
      do IO<Unit>:
        show_tag("pre", pre)
        show_tag("A", A)
        show_tag("DT", DT)
        show_tag("ADT", ADT)
        show_tag("trap", trap)
        show_tag("gamma", gamma)
        show_tag("shift", shift)
        show_tag("scale", scale)
        show_tag("Bn", Bn)
        show_tag("Cn", Cn)
        show_tag("angle", angle)
        show_tag("q", q)
        show_tag("k", k)
        show_tag("qk", qk)
        show_tag("states", states)
        show_tag("out_v", out_v)
        show_tag("y_inner", y_inner)
        show_tag("y", y)
        return Unit{}

# ---------------------------------------------------------------------------
# backward, pass 1: the output gate and the scan

def rev_go(+xs: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      rev_go(t, Con{h, acc})

def rev(+xs: List<&2, F32>) -> List<&2, F32>:
  rev_go(xs, Nil{})

def rev_rows(+left: Nat, +xs: List<&2, F32>, +w: Nat,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +row = take_row(xs, 0n, w)
      rev_rows(p, List.drop(&2, F32, xs, w), w,
        List.append(&2, F32, row, acc))

def add_nth(+xs: List<&2, F32>, +i: Nat, +d: F32) -> List<&2, F32>:
  match xs i:
    case Nil{} _:
      Nil{}
    case Con{h, t} 0n:
      Con{(h + d : F32), t}
    case Con{h, t} 1n+p:
      Con{h, add_nth(t, p, d)}

def silu_deriv(+x: F32) -> F32:
  +s = sigmoid(x)
  ((s * (1.0 + (x * (1.0 - s : F32) : F32) : F32) : F32) : F32)

def silu_deriv_row(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{silu_deriv(h), silu_deriv_row(t)}

type HG is Type:
  HG{dq: List<&2, F32>, dk: List<&2, F32>, dx: List<&2, F32>,
     dS: List<&2, F32>, dqk: F32, dscale: F32, dADT: F32, dD: F32}

def head_bw(+left: Nat, +p: Nat, +S: List<&2, F32>,
    +dS_next: List<&2, F32>, +q: List<&2, F32>,
    +k: List<&2, F32>, +x: List<&2, F32>, +dov: List<&2, F32>,
    +a: F32, +skip: F32, +sc: F32, +dq: List<&2, F32>,
    +dk: List<&2, F32>, +dx: List<&2, F32>,
    +dS: List<&2, F32>, +dqk: F32, +dscale: F32, +dADT: F32,
    +dD: F32) -> HG:
  match left:
    case 0n:
      HG{dq, dk, dx, dS, dqk, dscale, dADT, dD}
    case 1n+pr:
      +Srow = take_row(S, Nat.mul(p, 16n), 16n)
      +Nrow = take_row(dS_next, Nat.mul(p, 16n), 16n)
      +qs = dot(Srow, q)
      +dp = nth(p, dov)
      +xv = nth(p, x)
      +dkd = dot(Nrow, k)
      +dq2 = add_lists(dq, scale_list(Srow, (a * dp : F32)))
      +dk2 = add_lists(dk, scale_list(Nrow, (sc * xv : F32)))
      +dx2 = add_nth(dx, p,
        ((skip * dp : F32) + (sc * dkd : F32) : F32))
      +dS2 = push_row(dS, add_lists(scale_list(q, (a * dp : F32)),
        scale_list(Nrow, a)))
      head_bw(pr, (p + 1n : Nat), S, dS_next, q, k, x, dov, a, skip,
        sc, dq2, dk2, dx2, dS2, (dqk + (dp * xv : F32) : F32),
        (dscale + (xv * dkd : F32) : F32),
        ((dADT + ((a * dp : F32) * qs : F32) : F32)
          + (a * dot(Nrow, Srow) : F32) : F32),
        (dD + (dp * xv : F32) : F32))

type SB is Type:
  SB{dq: List<&2, F32>, dk: List<&2, F32>, dx: List<&2, F32>,
     dS: List<&2, F32>, dqk: List<&2, F32>,
     dscale: List<&2, F32>, dADT: List<&2, F32>,
     dD: List<&2, F32>}

def scan_bw_head(+h: Nat, +S_t: List<&2, F32>,
    +dstate: List<&2, F32>, +do_t: List<&2, F32>,
    +q_t: List<&2, F32>, +k_t: List<&2, F32>,
    +x_t: List<&2, F32>, +qk_t: List<&2, F32>,
    +scale_t: List<&2, F32>, +ADT_t: List<&2, F32>,
    +D: List<&2, F32>) -> HG:
  head_bw(32n, 0n, take_row(S_t, Nat.mul(h, 512n), 512n),
    take_row(dstate, Nat.mul(h, 512n), 512n),
    take_row(q_t, Nat.mul(h, 16n), 16n),
    take_row(k_t, Nat.mul(h, 16n), 16n),
    take_row(x_t, Nat.mul(h, 32n), 32n),
    take_row(do_t, Nat.mul(h, 32n), 32n),
    F32.exp(nth(h, ADT_t)), (nth(h, D) + nth(h, qk_t) : F32),
    nth(h, scale_t), zline(16n), zline(16n), zline(32n), Nil{},
    0.0, 0.0, 0.0, 0.0)

def scan_bw_add(hg: HG, sb: SB) -> SB:
  match hg sb:
    case HG{dq, dk, dx, dS, dqk, dscale, dADT, dD}
        SB{dq0, dk0, dx0, dS0, dqk0, dscale0, dADT0, dD0}:
      SB{push_row(dq0, dq), push_row(dk0, dk), push_row(dx0, dx),
        push_row(dS0, dS), push_row(dqk0, Con{dqk, Nil{}}),
        push_row(dscale0, Con{dscale, Nil{}}),
        push_row(dADT0, Con{dADT, Nil{}}),
        push_row(dD0, Con{dD, Nil{}})}

def scan_bw_all(+S_t: List<&2, F32>, +dstate: List<&2, F32>,
    +do_t: List<&2, F32>, +q_t: List<&2, F32>,
    +k_t: List<&2, F32>, +x_t: List<&2, F32>,
    +qk_t: List<&2, F32>, +scale_t: List<&2, F32>,
    +ADT_t: List<&2, F32>, +D: List<&2, F32>) -> SB:
  sb0 : SB = scan_bw_add(scan_bw_head(0n, S_t, dstate, do_t, q_t,
    k_t, x_t, qk_t, scale_t, ADT_t, D),
    SB{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}})
  sb1 : SB = scan_bw_add(scan_bw_head(1n, S_t, dstate, do_t, q_t,
    k_t, x_t, qk_t, scale_t, ADT_t, D), sb0)
  sb2 : SB = scan_bw_add(scan_bw_head(2n, S_t, dstate, do_t, q_t,
    k_t, x_t, qk_t, scale_t, ADT_t, D), sb1)
  scan_bw_add(scan_bw_head(3n, S_t, dstate, do_t, q_t, k_t, x_t,
    qk_t, scale_t, ADT_t, D), sb2)

type BW is Type:
  BW{dq: List<&2, F32>, dk: List<&2, F32>, dx: List<&2, F32>,
     dqk: List<&2, F32>, dscale: List<&2, F32>,
     dADT: List<&2, F32>, dD: List<&2, F32>,
     dstate: List<&2, F32>, do_v: List<&2, F32>,
     dz: List<&2, F32>, dW_out: List<&2, F32>,
     du: List<&2, F32>}

def bw_add(sb: SB, +do_v: List<&2, F32>, +dz: List<&2, F32>,
    +dW_out: List<&2, F32>, +do_v0: List<&2, F32>,
    +dz0: List<&2, F32>, +dq: List<&2, F32>,
    +dk: List<&2, F32>, +dx: List<&2, F32>,
    +dqk: List<&2, F32>, +dscale: List<&2, F32>,
    +dADT: List<&2, F32>, +dD: List<&2, F32>,
    +du: List<&2, F32>) -> BW:
  match sb:
    case SB{dq0, dk0, dx0, dS, dqk0, dscale0, dADT0, dD0}:
      BW{push_row(dq, dq0), push_row(dk, dk0), push_row(dx, dx0),
        push_row(dqk, dqk0), push_row(dscale, dscale0),
        push_row(dADT, dADT0), add_lists(dD, dD0), dS,
        push_row(do_v0, do_v), push_row(dz0, dz), dW_out, du}

def bw_go(+left: Nat, +t: Nat, +z: List<&2, F32>,
    +x: List<&2, F32>, +ADT: List<&2, F32>,
    +q: List<&2, F32>, +k: List<&2, F32>,
    +qk: List<&2, F32>, +scale: List<&2, F32>,
    +states: List<&2, F32>, +out_v: List<&2, F32>,
    +y_inner: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +dy: List<&2, F32>,
    bw: BW) -> BW:
  match left bw:
    case 0n BW{+dq, +dk, +dx, +dqk, +dscale, +dADT, dD, dstate, +do_v,
        +dz, dW_out, du}:
      +ndq = Nat.div(List.length(&2, F32, dq), 64n)
      +ndk = Nat.div(List.length(&2, F32, dk), 64n)
      +ndx = Nat.div(List.length(&2, F32, dx), 128n)
      +nqk = Nat.div(List.length(&2, F32, dqk), 4n)
      +nsc = Nat.div(List.length(&2, F32, dscale), 4n)
      +ndt = Nat.div(List.length(&2, F32, dADT), 4n)
      +ndo = Nat.div(List.length(&2, F32, do_v), 128n)
      +ndz = Nat.div(List.length(&2, F32, dz), 128n)
      BW{rev_rows(ndq, dq, 64n, Nil{}),
        rev_rows(ndk, dk, 64n, Nil{}),
        rev_rows(ndx, dx, 128n, Nil{}),
        rev_rows(nqk, dqk, 4n, Nil{}),
        rev_rows(nsc, dscale, 4n, Nil{}),
        rev_rows(ndt, dADT, 4n, Nil{}),
        dD, dstate,
        rev_rows(ndo, do_v, 128n, Nil{}),
        rev_rows(ndz, dz, 128n, Nil{}),
        dW_out, du}
    case 1n+p BW{dq, dk, dx, dqk, dscale, dADT, dD, dstate, do_v,
        dz, dW_out, du}:
      +dy_t = take_row(dy, Nat.mul(t, 64n), 64n)
      +yi_t = take_row(y_inner, Nat.mul(t, 128n), 128n)
      +z_t = take_row(z, Nat.mul(t, 128n), 128n)
      +ov_t = take_row(out_v, Nat.mul(t, 128n), 128n)
      +x_t = take_row(x, Nat.mul(t, 128n), 128n)
      +q_t = take_row(q, Nat.mul(t, 64n), 64n)
      +k_t = take_row(k, Nat.mul(t, 64n), 64n)
      +qk_t = take_row(qk, Nat.mul(t, 4n), 4n)
      +scale_t = take_row(scale, Nat.mul(t, 4n), 4n)
      +ADT_t = take_row(ADT, Nat.mul(t, 4n), 4n)
      +S_t = take_row(states, Nat.mul(t, 2048n), 2048n)
      +dyi = matvec_t(w_out, dy_t, 64n, 128n)
      +do_v_t = mul_lists(dyi, silu_row(z_t))
      +dz_t = mul_lists(mul_lists(dyi, ov_t), silu_deriv_row(z_t))
      +dW_out2 = add_lists(dW_out, outer(dy_t, yi_t))
      sb : SB = scan_bw_all(S_t, dstate, do_v_t, q_t, k_t, x_t,
        qk_t, scale_t, ADT_t, D)
      bw_go(p, (t - 1n : Nat), z, x, ADT, q, k, qk, scale, states,
        out_v, y_inner, D, w_out, dy,
        bw_add(sb, do_v_t, dz_t, dW_out2, do_v, dz, dq, dk, dx,
          dqk, dscale, dADT, dD, du))

def bw_start(+z: List<&2, F32>, +x: List<&2, F32>,
    +ADT: List<&2, F32>, +q: List<&2, F32>,
    +k: List<&2, F32>, +qk: List<&2, F32>,
    +scale: List<&2, F32>, +states: List<&2, F32>,
    +out_v: List<&2, F32>, +y_inner: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +dy: List<&2, F32>, +n: Nat) -> BW:
  bw_go(n, (n - 1n : Nat), z, x, ADT, q, k, qk, scale, states,
    out_v, y_inner, D, w_out, dy,
    BW{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, zline(4n),
      zline(2048n), Nil{}, Nil{}, zline(8192n), Nil{}})



# ---------------------------------------------------------------------------
# backward, pass 2: the trapezoid chain, the angles and the RMS norms

type RB is Type:
  RB{dpre: List<&2, F32>, dangle: List<&2, F32>}

def rope_bwd_go(+left: Nat, +j: Nat, +drot: List<&2, F32>,
    +rot: List<&2, F32>, +angles: List<&2, F32>,
    +adp: List<&2, F32>, +adang: List<&2, F32>) -> RB:
  match left:
    case 0n:
      RB{List.append(&2, F32, adp, take_row(drot, 8n, 8n)), adang}
    case 1n+p:
      +a = nth(j, angles)
      +c = F32.cos(a)
      +s = F32.sin(a)
      +d0 = nth(Nat.mul(2n, j), drot)
      +d1 = nth((Nat.mul(2n, j) + 1n : Nat), drot)
      +r0 = nth(Nat.mul(2n, j), rot)
      +r1 = nth((Nat.mul(2n, j) + 1n : Nat), rot)
      +y0 = ((d0 * c : F32) + (d1 * s : F32) : F32)
      +y1 = ((0.0 - (d0 * s : F32) : F32) + (d1 * c : F32) : F32)
      +da = ((0.0 - (d0 * r1 : F32) : F32) + (d1 * r0 : F32) : F32)
      rope_bwd_go(p, (j + 1n : Nat), drot, rot, angles,
        push_row(push_row(adp, Con{y0, Nil{}}), Con{y1, Nil{}}),
        push_row(adang, Con{da, Nil{}}))

def rope_bwd(+drot: List<&2, F32>, +rot: List<&2, F32>,
    +angles: List<&2, F32>) -> RB:
  rope_bwd_go(4n, 0n, drot, rot, angles, Nil{}, Nil{})

type HPR is Type:
  HPR{dq_pre: List<&2, F32>, dk_pre: List<&2, F32>,
      dangle: List<&2, F32>}

def both_fin(rbk: RB, +dq_pre: List<&2, F32>,
    +dq_ang: List<&2, F32>) -> HPR:
  match rbk:
    case RB{dk_pre, dk_ang}:
      HPR{dq_pre, dk_pre, add_lists(dq_ang, dk_ang)}

def both_q(rbq: RB, +dk_rot: List<&2, F32>,
    +k_rot: List<&2, F32>, +kang: List<&2, F32>) -> HPR:
  match rbq:
    case RB{dq_pre, dq_ang}:
      both_fin(rope_bwd(dk_rot, k_rot, kang), dq_pre, dq_ang)

def both_rope(+dq_rot: List<&2, F32>, +q_rot: List<&2, F32>,
    +qang: List<&2, F32>, +dk_rot: List<&2, F32>,
    +k_rot: List<&2, F32>, +kang: List<&2, F32>) -> HPR:
  both_q(rope_bwd(dq_rot, q_rot, qang), dk_rot, k_rot, kang)

type HPB is Type:
  HPB{dq_pre: List<&2, F32>, dk_pre: List<&2, F32>,
      dangle: List<&2, F32>, dgamma: F32}

def hp_fin(hpr: HPR, +qk_q: List<&2, F32>, +qk_k: List<&2, F32>,
    +dgamma: F32) -> HPB:
  match hpr:
    case HPR{dq_pre, dk_pre, dangle}:
      HPB{add_lists(dq_pre, qk_k), add_lists(dk_pre, qk_q),
        dangle, dgamma}

def head_pre_bw(+h: Nat, +Cn_t: List<&2, F32>,
    +Bn_t: List<&2, F32>, +C_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +dq_all: List<&2, F32>,
    +dk_all: List<&2, F32>, +q_t: List<&2, F32>,
    +k_t: List<&2, F32>, +qk_t: List<&2, F32>,
    +gamma_t: List<&2, F32>, +dqk_t: List<&2, F32>,
    +dscale_t: List<&2, F32>, +angle_t: List<&2, F32>) -> HPB:
  +cb = take_row(C_bias, Nat.mul(h, 16n), 16n)
  +bb = take_row(B_bias, Nat.mul(h, 16n), 16n)
  +qpre = add_lists(Cn_t, cb)
  +kpre = add_lists(Bn_t, bb)
  +qk_h = nth(h, dqk_t)
  +g_h = nth(h, gamma_t)
  +qk_term = (qk_h * g_h : F32)
  +ang = take_row(angle_t, Nat.mul(h, 4n), 4n)
  hpr : HPR = both_rope(take_row(dq_all, Nat.mul(h, 16n), 16n),
    take_row(q_t, Nat.mul(h, 16n), 16n), ang,
    take_row(dk_all, Nat.mul(h, 16n), 16n),
    take_row(k_t, Nat.mul(h, 16n), 16n), ang)
  hp_fin(hpr, scale_list(qpre, qk_term), scale_list(kpre, qk_term),
    ((qk_h * dot(qpre, kpre) : F32) + nth(h, dscale_t) : F32))

type HP is Type:
  HP{dCn: List<&2, F32>, dBn: List<&2, F32>,
     dCb: List<&2, F32>, dBb: List<&2, F32>,
     dangle: List<&2, F32>, dgamma: List<&2, F32>}

def hp_add(hpb: HPB, hp: HP) -> HP:
  match hpb hp:
    case HPB{+dq_pre, +dk_pre, dangle, dgamma}
        HP{dCn0, dBn0, dCb0, dBb0, dangle0, dgamma0}:
      HP{add_lists(dCn0, dq_pre), add_lists(dBn0, dk_pre),
        push_row(dCb0, dq_pre), push_row(dBb0, dk_pre),
        push_row(dangle0, dangle),
        push_row(dgamma0, Con{dgamma, Nil{}})}

def hp_all(+Cn_t: List<&2, F32>, +Bn_t: List<&2, F32>,
    +C_bias: List<&2, F32>, +B_bias: List<&2, F32>,
    +dq_all: List<&2, F32>, +dk_all: List<&2, F32>,
    +q_t: List<&2, F32>, +k_t: List<&2, F32>,
    +qk_t: List<&2, F32>, +gamma_t: List<&2, F32>,
    +dqk_t: List<&2, F32>, +dscale_t: List<&2, F32>,
    +angle_t: List<&2, F32>) -> HP:
  hp0 : HP = hp_add(head_pre_bw(0n, Cn_t, Bn_t, C_bias, B_bias,
    dq_all, dk_all, q_t, k_t, qk_t, gamma_t, dqk_t, dscale_t, angle_t),
    HP{zline(16n), zline(16n), Nil{}, Nil{}, Nil{}, Nil{}})
  hp1 : HP = hp_add(head_pre_bw(1n, Cn_t, Bn_t, C_bias, B_bias,
    dq_all, dk_all, q_t, k_t, qk_t, gamma_t, dqk_t, dscale_t, angle_t),
    hp0)
  hp2 : HP = hp_add(head_pre_bw(2n, Cn_t, Bn_t, C_bias, B_bias,
    dq_all, dk_all, q_t, k_t, qk_t, gamma_t, dqk_t, dscale_t, angle_t),
    hp1)
  hp_add(head_pre_bw(3n, Cn_t, Bn_t, C_bias, B_bias, dq_all, dk_all,
    q_t, k_t, qk_t, gamma_t, dqk_t, dscale_t, angle_t), hp2)


type RMSG is Type:
  RMSG{dx: List<&2, F32>, dw: List<&2, F32>}

def rms_s(+xs: List<&2, F32>, +ws: List<&2, F32>,
    +dys: List<&2, F32>) -> F32:
  match xs ws dys:
    case Nil{} _ _:
      0.0
    case _ Nil{} _:
      0.0
    case _ _ Nil{}:
      0.0
    case Con{x, xt} Con{w, wt} Con{d, dt}:
      (((w * x : F32) * d : F32) + rms_s(xt, wt, dt) : F32)

def rms_dx_go(+xs: List<&2, F32>, +ws: List<&2, F32>,
    +dys: List<&2, F32>, +r: F32, +c: F32,
    acc: List<&2, F32>) -> List<&2, F32>:
  match xs ws dys:
    case Nil{} _ _:
      acc
    case _ Nil{} _:
      acc
    case _ _ Nil{}:
      acc
    case Con{x, xt} Con{w, wt} Con{d, dt}:
      rms_dx_go(xt, wt, dt, r, c,
        push_row(acc, Con{(((w * r : F32) * d : F32)
          - (c * x : F32) : F32), Nil{}}))

def rms_dw_go(+xs: List<&2, F32>, +ws: List<&2, F32>,
    +dys: List<&2, F32>, +r: F32,
    acc: List<&2, F32>) -> List<&2, F32>:
  match xs ws dys:
    case Nil{} _ _:
      acc
    case _ Nil{} _:
      acc
    case _ _ Nil{}:
      acc
    case Con{x, xt} Con{w, wt} Con{d, dt}:
      rms_dw_go(xt, wt, dt, r,
        push_row(acc, Con{(((d * x : F32) * r : F32)), Nil{}}))

def rms_bwd(+xs: List<&2, F32>, +ws: List<&2, F32>,
    +dys: List<&2, F32>, +n: Nat) -> RMSG:
  +nf = F32.from_nat(n)
  +mean = (sumsq(xs) / nf : F32)
  +r = (1.0 / F32.sqrt((mean + 0.00001 : F32)) : F32)
  +s = rms_s(xs, ws, dys)
  +c = (((r * r : F32) * r : F32) / nf : F32)
  RMSG{rms_dx_go(xs, ws, dys, r, (c * s : F32), Nil{}),
    rms_dw_go(xs, ws, dys, r, Nil{})}

def hderiv_if(b: Bool, +x: F32) -> F32:
  match b:
    case True{}:
      1.0
    case False{}:
      (1.0 / ((1.0 - x : F32) * (1.0 - x : F32)) : F32)

def htail_deriv(+x: F32) -> F32:
  hderiv_if(F32.is_ge(x, 0.0), x)

def clamp_grad_if(b: Bool, +v: F32) -> F32:
  match b:
    case True{}:
      0.0
    case False{}:
      v

def dA_to_ddA(+dA: F32, +x: F32) -> F32:
  +v = (0.0 - (dA * htail_deriv(x) : F32) : F32)
  clamp_grad_if(F32.is_lt(htail(x), 0.0001), v)

def prev_shift(+t: Nat, +h: Nat, +dscale: List<&2, F32>) -> F32:
  match t:
    case 0n:
      0.0
    case 1n+p:
      nth((Nat.mul(p, 4n) + h : Nat), dscale)

def sigmoid_deriv(+x: F32) -> F32:
  +s = sigmoid(x)
  (s * (1.0 - s : F32) : F32)

def dang_j_go(+left: Nat, +j: Nat, +dangle2: List<&2, F32>,
    +ang_raw: List<&2, F32>, +DT_t: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +a = nth(j, ang_raw)
      +th = F32.tanh(a)
      +dt = nth(0n, DT_t)
      +d = ((nth(j, dangle2) * 3.141592653589793 : F32) * dt : F32)
      +v = (d * (1.0 - (th * th : F32) : F32) : F32)
      dang_j_go(p, (j + 1n : Nat), dangle2, ang_raw, DT_t,
        push_row(acc, Con{v, Nil{}}))

def dang_head_go(+left: Nat, +h: Nat,
    +dangle2: List<&2, F32>, +ang_raw: List<&2, F32>,
    +DT_t: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +ah = take_row(dangle2, Nat.mul(h, 4n), 4n)
      +d = dang_j_go(4n, 0n, ah, ang_raw,
        take_row(DT_t, h, 1n), Nil{})
      dang_head_go(p, (h + 1n : Nat), dangle2, ang_raw, DT_t,
        add_lists(acc, d))

def dang_t(+dangle2: List<&2, F32>, +ang_raw: List<&2, F32>,
    +DT_t: List<&2, F32>) -> List<&2, F32>:
  dang_head_go(4n, 0n, dangle2, ang_raw, DT_t, zline(4n))

type PHB is Type:
  PHB{dDT: List<&2, F32>, dtrap: List<&2, F32>,
      ddd_dt: List<&2, F32>, ddd_A: List<&2, F32>,
      dang: List<&2, F32>}

def ang_ddt_go(+left: Nat, +j: Nat, +dangle_h: List<&2, F32>,
    +ang_raw: List<&2, F32>, +acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      ang_ddt_go(p, (j + 1n : Nat), dangle_h, ang_raw,
        (acc + ((nth(j, dangle_h) * 3.141592653589793 : F32)
          * F32.tanh(nth(j, ang_raw)) : F32) : F32))

def pre_head_go(+left: Nat, +h: Nat, +t: Nat,
    +dgamma: List<&2, F32>, +dADT_t: List<&2, F32>,
    +A_t: List<&2, F32>, +DT_t: List<&2, F32>,
    +trap_t: List<&2, F32>, +dd_dt_t: List<&2, F32>,
    +dd_A_t: List<&2, F32>, +dt_bias: List<&2, F32>,
    +ang_raw_t: List<&2, F32>, +dangle2: List<&2, F32>,
    +dscale: List<&2, F32>, +accDT: List<&2, F32>,
    +acctrap: List<&2, F32>, +accddt: List<&2, F32>,
    +accdA: List<&2, F32>, +accdang: List<&2, F32>) -> PHB:
  match left:
    case 0n:
      PHB{accDT, acctrap, accddt, accdA, accdang}
    case 1n+p:
      +dg = nth(h, dgamma)
      +dtv = nth(h, DT_t)
      +tr = nth(h, trap_t)
      +ad = nth(h, dADT_t)
      +av = nth(h, A_t)
      +sh = prev_shift(t, h, dscale)
      +ang_h = take_row(dangle2, Nat.mul(h, 4n), 4n)
      +ang_ddt = ang_ddt_go(4n, 0n, ang_h, ang_raw_t, 0.0)
      +dang_h = dang_j_go(4n, 0n, ang_h, ang_raw_t,
        take_row(DT_t, h, 1n), Nil{})
      +dDT = ((((dg * tr : F32) + (ad * av : F32) : F32)
        + (sh * (1.0 - tr : F32) : F32) : F32) + ang_ddt : F32)
      +dtrap = ((((dg * dtv : F32) - (sh * dtv : F32) : F32)
        * tr : F32) * (1.0 - tr : F32) : F32)
      +ddd_dt = (dDT * sigmoid((nth(h, dd_dt_t)
        + nth(h, dt_bias) : F32)) : F32)
      +dA = (ad * dtv : F32)
      +ddd_A = dA_to_ddA(dA, nth(h, dd_A_t))
      pre_head_go(p, (h + 1n : Nat), t, dgamma, dADT_t, A_t, DT_t,
        trap_t, dd_dt_t, dd_A_t, dt_bias, ang_raw_t, dangle2, dscale,
        push_row(accDT, Con{dDT, Nil{}}),
        push_row(acctrap, Con{dtrap, Nil{}}),
        push_row(accddt, Con{ddd_dt, Nil{}}),
        push_row(accdA, Con{ddd_A, Nil{}}),
        add_lists(accdang, dang_h))

def pre_heads(+t: Nat, +dgamma: List<&2, F32>,
    +dADT_t: List<&2, F32>, +A_t: List<&2, F32>,
    +DT_t: List<&2, F32>, +trap_t: List<&2, F32>,
    +dd_dt_t: List<&2, F32>, +dd_A_t: List<&2, F32>,
    +dt_bias: List<&2, F32>, +ang_raw_t: List<&2, F32>,
    +dangle2: List<&2, F32>, +dscale: List<&2, F32>) -> PHB:
  ph : PHB = pre_head_go(4n, 0n, t, dgamma, dADT_t, A_t, DT_t, trap_t,
    dd_dt_t, dd_A_t, dt_bias, ang_raw_t, dangle2, dscale, Nil{},
    Nil{}, Nil{}, Nil{}, zline(4n))
  ph


type G2 is Type:
  G2{dW_in: List<&2, F32>, du: List<&2, F32>,
     ddt_bias: List<&2, F32>, dB_bias: List<&2, F32>,
     dC_bias: List<&2, F32>, dBw: List<&2, F32>,
     dCw: List<&2, F32>, dang_raw: List<&2, F32>,
     dangle: List<&2, F32>, dBn: List<&2, F32>,
     dCn: List<&2, F32>, dDT: List<&2, F32>,
     dtrap: List<&2, F32>, ddd_dt: List<&2, F32>,
     ddd_A: List<&2, F32>, dBraw: List<&2, F32>,
     dCraw: List<&2, F32>, dD: List<&2, F32>,
     dW_out: List<&2, F32>}

def cat2(+a: List<&2, F32>, +b: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, a, b)

def cat8(+a: List<&2, F32>, +b: List<&2, F32>,
    +c: List<&2, F32>, +d: List<&2, F32>,
    +e: List<&2, F32>, +f: List<&2, F32>,
    +g: List<&2, F32>, +h: List<&2, F32>) -> List<&2, F32>:
  cat2(cat2(cat2(a, b), cat2(c, d)), cat2(cat2(e, f), cat2(g, h)))

def pw_fin2(ph: PHB, rb: RMSG, rc: RMSG,
    +dCn: List<&2, F32>, +dBn: List<&2, F32>,
    +dCb: List<&2, F32>, +dBb: List<&2, F32>,
    +dangle2: List<&2, F32>, +dW_in: List<&2, F32>,
    +du: List<&2, F32>, +ddt_bias: List<&2, F32>,
    +dB_bias: List<&2, F32>, +dC_bias: List<&2, F32>,
    +dBw: List<&2, F32>, +dCw: List<&2, F32>,
    +dang_raw: List<&2, F32>, +dBn_all: List<&2, F32>,
    +dCn_all: List<&2, F32>, +dDT_all: List<&2, F32>,
    +dtrap_all: List<&2, F32>, +ddd_dt_all: List<&2, F32>,
    +ddd_A_all: List<&2, F32>, +dBraw_all: List<&2, F32>,
    +dCraw_all: List<&2, F32>, +dz_t: List<&2, F32>,
    +dx_t: List<&2, F32>, +u_t: List<&2, F32>,
    +w_in: List<&2, F32>, +dD: List<&2, F32>,
    +dW_out: List<&2, F32>) -> G2:
  match ph rb rc:
    case PHB{dDT, +dtrap, +ddd_dt, +ddd_A, +dang}
        RMSG{+dxb, dwb} RMSG{+dxc, dwc}:
      +dpre = cat8(dz_t, dx_t, dxb, dxc, ddd_dt, ddd_A, dtrap, dang)
      G2{add_lists(dW_in, outer(dpre, u_t)),
        push_row(du, matvec_t(w_in, dpre, 304n, 64n)),
        add_lists(ddt_bias, ddd_dt), add_lists(dB_bias, dBb),
        add_lists(dC_bias, dCb), add_lists(dBw, dwb),
        add_lists(dCw, dwc), add_lists(dang_raw, dang), dangle2,
        push_row(dBn_all, dBn), push_row(dCn_all, dCn),
        push_row(dDT_all, dDT), push_row(dtrap_all, dtrap),
        push_row(ddd_dt_all, ddd_dt), push_row(ddd_A_all, ddd_A),
        push_row(dBraw_all, dxb), push_row(dCraw_all, dxc),
        dD, dW_out}

def pw_finish(hp: HP, g: G2, +t: Nat,
    +DT_t: List<&2, F32>, +trap_t: List<&2, F32>,
    +A_t: List<&2, F32>, +dADT_t: List<&2, F32>,
    +dd_dt_t: List<&2, F32>, +dd_A_t: List<&2, F32>,
    +dt_bias: List<&2, F32>, +ang_raw_t: List<&2, F32>,
    +Braw_t: List<&2, F32>, +Craw_t: List<&2, F32>,
    +dscale_all: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +dz_t: List<&2, F32>,
    +dx_t: List<&2, F32>, +u_t: List<&2, F32>,
    +w_in: List<&2, F32>) -> G2:
  match hp g:
    case HP{+dCn, +dBn, dCb, dBb, dangle, dgamma}
        G2{dW_in, du, ddt_bias, dB_bias, dC_bias, dBw, dCw,
           dang_raw, dangle0, dBn_all, dCn_all, dDT_all, dtrap_all,
           ddd_dt_all, ddd_A_all, dBraw_all, dCraw_all, dD, dW_out}:
      +dangle2 = add_lists(dangle0, dangle)
      ph : PHB = pre_heads(t, dgamma, dADT_t, A_t, DT_t, trap_t,
        dd_dt_t, dd_A_t, dt_bias, ang_raw_t, dangle2, dscale_all)
      pw_fin2(ph, rms_bwd(Braw_t, Bw, dBn, 16n),
        rms_bwd(Craw_t, Cw, dCn, 16n), dCn, dBn, dCb, dBb,
        dangle2, dW_in, du, ddt_bias, dB_bias, dC_bias, dBw, dCw,
        dang_raw, dBn_all, dCn_all, dDT_all, dtrap_all, ddd_dt_all,
        ddd_A_all, dBraw_all, dCraw_all, dz_t, dx_t, u_t, w_in,
        dD, dW_out)

def pw_go(+left: Nat, +t: Nat, +DT: List<&2, F32>,
    +trap: List<&2, F32>, +A: List<&2, F32>,
    +dADT: List<&2, F32>, +dd_dt: List<&2, F32>,
    +dd_A: List<&2, F32>, +dt_bias: List<&2, F32>,
    +ang_raw: List<&2, F32>, +Braw: List<&2, F32>,
    +Craw: List<&2, F32>, +Bn: List<&2, F32>,
    +Cn: List<&2, F32>, +angle: List<&2, F32>,
    +q: List<&2, F32>, +k: List<&2, F32>,
    +qk: List<&2, F32>, +gamma: List<&2, F32>,
    +dq_all: List<&2, F32>, +dk_all: List<&2, F32>,
    +dqk_all: List<&2, F32>, +dscale_all: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +w_in: List<&2, F32>, +u: List<&2, F32>,
    +dz_all: List<&2, F32>, +dx_all: List<&2, F32>,
    g: G2) -> G2:
  match left:
    case 0n:
      g
    case 1n+p:
      +DT_t = take_row(DT, Nat.mul(t, 4n), 4n)
      +trap_t = take_row(trap, Nat.mul(t, 4n), 4n)
      +A_t = take_row(A, Nat.mul(t, 4n), 4n)
      +dADT_t = take_row(dADT, Nat.mul(t, 4n), 4n)
      +dd_dt_t = take_row(dd_dt, Nat.mul(t, 4n), 4n)
      +dd_A_t = take_row(dd_A, Nat.mul(t, 4n), 4n)
      +ang_raw_t = take_row(ang_raw, Nat.mul(t, 4n), 4n)
      +Braw_t = take_row(Braw, Nat.mul(t, 16n), 16n)
      +Craw_t = take_row(Craw, Nat.mul(t, 16n), 16n)
      +Bn_t = take_row(Bn, Nat.mul(t, 16n), 16n)
      +Cn_t = take_row(Cn, Nat.mul(t, 16n), 16n)
      +angle_t = take_row(angle, Nat.mul(t, 16n), 16n)
      +q_t = take_row(q, Nat.mul(t, 64n), 64n)
      +k_t = take_row(k, Nat.mul(t, 64n), 64n)
      +qk_t = take_row(qk, Nat.mul(t, 4n), 4n)
      +gamma_t = take_row(gamma, Nat.mul(t, 4n), 4n)
      +dq_t = take_row(dq_all, Nat.mul(t, 64n), 64n)
      +dk_t = take_row(dk_all, Nat.mul(t, 64n), 64n)
      +dqk_t = take_row(dqk_all, Nat.mul(t, 4n), 4n)
      +dsc_t = take_row(dscale_all, Nat.mul(t, 4n), 4n)
      +dz_t = take_row(dz_all, Nat.mul(t, 128n), 128n)
      +dx_t = take_row(dx_all, Nat.mul(t, 128n), 128n)
      +u_t = take_row(u, Nat.mul(t, 64n), 64n)
      hp : HP = hp_all(Cn_t, Bn_t, C_bias, B_bias, dq_t, dk_t,
        q_t, k_t, qk_t, gamma_t, dqk_t, dsc_t, angle_t)
      pw_go(p, (t - 1n : Nat), DT, trap, A, dADT, dd_dt, dd_A,
        dt_bias, ang_raw, Braw, Craw, Bn, Cn, angle, q, k, qk,
        gamma, dq_all, dk_all, dqk_all, dscale_all, Bw, Cw,
        B_bias, C_bias, w_in, u, dz_all, dx_all,
        pw_finish(hp, g, t, DT_t, trap_t, A_t, dADT_t, dd_dt_t,
          dd_A_t, dt_bias, ang_raw_t, Braw_t, Craw_t, dscale_all,
          Bw, Cw, dz_t, dx_t, u_t, w_in))

type GRAD is Type:
  GRAD{du: List<&2, F32>, dW_in: List<&2, F32>,
       ddt_bias: List<&2, F32>, dB_bias: List<&2, F32>,
       dC_bias: List<&2, F32>, dB_norm_w: List<&2, F32>,
       dC_norm_w: List<&2, F32>, dD: List<&2, F32>,
       dW_out: List<&2, F32>, do_v: List<&2, F32>,
       dq: List<&2, F32>, dk: List<&2, F32>,
       dz: List<&2, F32>, dx: List<&2, F32>,
       dqk: List<&2, F32>, dscale: List<&2, F32>,
       dADT: List<&2, F32>, dBn: List<&2, F32>,
       dCn: List<&2, F32>, dDT: List<&2, F32>,
       dtrap: List<&2, F32>, ddd_dt: List<&2, F32>,
       ddd_A: List<&2, F32>, dBraw: List<&2, F32>,
       dCraw: List<&2, F32>, dang_raw: List<&2, F32>}

def to_grad(g: G2, +do_v: List<&2, F32>, +dz: List<&2, F32>,
    +dx: List<&2, F32>, +dqk: List<&2, F32>,
    +dscale: List<&2, F32>, +dADT: List<&2, F32>,
    +dq: List<&2, F32>, +dk: List<&2, F32>) -> GRAD:
  match g:
    case G2{dW_in, +du, ddt_bias, dB_bias, dC_bias, dBw, dCw,
        dang_raw, dangle, +dBn, +dCn, +dDT, +dtrap, +ddd_dt, +ddd_A,
        +dBraw, +dCraw, dD, dW_out}:
      GRAD{rev_rows(Nat.div(List.length(&2, F32, du), 64n), du, 64n, Nil{}),
        dW_in, ddt_bias, dB_bias, dC_bias, dBw, dCw, dD,
        dW_out, do_v, dq, dk, dz, dx, dqk, dscale, dADT,
        rev_rows(Nat.div(List.length(&2, F32, dBn), 16n), dBn, 16n, Nil{}),
        rev_rows(Nat.div(List.length(&2, F32, dCn), 16n), dCn, 16n, Nil{}),
        rev_rows(Nat.div(List.length(&2, F32, dDT), 4n), dDT, 4n, Nil{}),
        rev_rows(Nat.div(List.length(&2, F32, dtrap), 4n), dtrap, 4n, Nil{}),
        rev_rows(Nat.div(List.length(&2, F32, ddd_dt), 4n), ddd_dt, 4n, Nil{}),
        rev_rows(Nat.div(List.length(&2, F32, ddd_A), 4n), ddd_A, 4n, Nil{}),
        rev_rows(Nat.div(List.length(&2, F32, dBraw), 16n), dBraw, 16n, Nil{}),
        rev_rows(Nat.div(List.length(&2, F32, dCraw), 16n), dCraw, 16n, Nil{}),
        dang_raw}

def pw_start(bw: BW, +DT: List<&2, F32>, +trap: List<&2, F32>,
    +A: List<&2, F32>, +dADT: List<&2, F32>,
    +dd_dt: List<&2, F32>, +dd_A: List<&2, F32>,
    +dt_bias: List<&2, F32>, +ang_raw: List<&2, F32>,
    +Braw: List<&2, F32>, +Craw: List<&2, F32>,
    +Bn: List<&2, F32>, +Cn: List<&2, F32>,
    +angle: List<&2, F32>, +q: List<&2, F32>,
    +k: List<&2, F32>, +qk: List<&2, F32>,
    +gamma: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +w_in: List<&2, F32>,
    +u: List<&2, F32>, +n: Nat) -> GRAD:
  match bw:
    case BW{+dq, +dk, +dx, +dqk, +dscale, +dADT, dD, dstate, do_v, +dz,
        dW_out, du}:
      g : G2 = pw_go(n, (n - 1n : Nat), DT, trap, A, dADT, dd_dt, dd_A,
        dt_bias, ang_raw, Braw, Craw, Bn, Cn, angle, q, k, qk,
        gamma, dq, dk, dqk, dscale, Bw, Cw, B_bias, C_bias, w_in,
        u, dz, dx,
        G2{zline(19456n), Nil{}, zline(4n), zline(64n), zline(64n),
          zline(16n), zline(16n), zline(4n), zline(16n), Nil{},
          Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, dD,
          dW_out})
      to_grad(g, do_v, dz, dx, dqk, dscale, dADT, dq, dk)

def backward(f: FWD, +u: List<&2, F32>, +dy: List<&2, F32>,
    +w_in: List<&2, F32>, +w_out: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +dt_bias: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +D: List<&2, F32>,
    +n: Nat) -> GRAD:
  match f:
    case FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, +ADT, trap, gamma, shift, scale, Bn, Cn, angle, +q, +k,
        +qk, states, out_v, y_inner, y}:
      bw : BW = bw_start(z, x, ADT, q, k, qk, scale, states, out_v,
        y_inner, D, w_out, dy, n)
      pw_start(bw, DT, trap, A, ADT, dd_dt, dd_A, dt_bias,
        ang_raw, Braw, Craw, Bn, Cn, angle, q, k, qk, gamma, Bw,
        Cw, B_bias, C_bias, w_in, u, n)


# ---------------------------------------------------------------------------
# the ten-step copy task: tokens 0..7 in, the same tokens out

def sgd(+xs: List<&2, F32>, +gs: List<&2, F32>, +lr: F32) -> List<&2, F32>:
  match xs gs:
    case Nil{} _:
      Nil{}
    case _ Nil{}:
      Nil{}
    case Con{x, xt} Con{g, gt}:
      Con{(x - (lr * g : F32) : F32), sgd(xt, gt, lr)}

def max_list(+xs: List<&2, F32>, +m: F32) -> F32:
  match xs:
    case Nil{}:
      m
    case Con{h, t}:
      max_list(t, F32.max(m, h))

def exp_row(+xs: List<&2, F32>, +m: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{F32.exp((h - m : F32)), exp_row(t, m)}

def set_nth(+xs: List<&2, F32>, +i: Nat, +v: F32) -> List<&2, F32>:
  match xs i:
    case Nil{} _:
      Nil{}
    case Con{h, t} 0n:
      Con{v, t}
    case Con{h, t} 1n+p:
      Con{h, set_nth(t, p, v)}

def argmax_go(+left: Nat, b: Bool, +i: Nat, +best: Nat, +bv: F32,
    +v: F32, +xs: List<&2, F32>) -> Nat:
  match left b:
    case 0n _:
      best
    case 1n+p True{}:
      +ni = (i + 1n : Nat)
      +nv = nth(ni, xs)
      argmax_go(p, F32.is_gt(nv, v), ni, i, v, nv, xs)
    case 1n+p False{}:
      +ni = (i + 1n : Nat)
      +nv = nth(ni, xs)
      argmax_go(p, F32.is_gt(nv, bv), ni, best, bv, nv, xs)

def argmax(+xs: List<&2, F32>, +n: Nat) -> Nat:
  argmax_go(n, False{}, 0n, 0n, nth(0n, xs), nth(0n, xs), xs)

def softmax_list(+logits: List<&2, F32>, +n: Nat) -> List<&2, F32>:
  +m = max_list(logits, nth(0n, logits))
  +ex = exp_row(logits, m)
  scale_list(ex, (1.0 / sum_list(ex) : F32))

def ce_loss(+logits: List<&2, F32>, +target: Nat, +n: Nat,
    +invT: F32) -> F32:
  +p = softmax_list(logits, n)
  ((0.0 - F32.log(nth(target, p)) : F32) * invT : F32)

def ce_dl(+logits: List<&2, F32>, +target: Nat, +n: Nat,
    +invT: F32) -> List<&2, F32>:
  +p = softmax_list(logits, n)
  +pt = nth(target, p)
  scale_list(set_nth(p, target, (pt - 1.0 : F32)), invT)

def tok_go(+left: Nat, +t: Nat, +y: List<&2, F32>,
    +Wh: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +y_t = take_row(y, Nat.mul(t, 64n), 64n)
      +logits = matvec(Wh, y_t, 64n)
      tok_go(p, (t + 1n : Nat), y, Wh,
        push_row(acc, Con{F32.from_nat(argmax(logits, 8n)), Nil{}}))

type TP is Type:
  TP{w_in: List<&2, F32>, dt_bias: List<&2, F32>,
     B_bias: List<&2, F32>, C_bias: List<&2, F32>,
     Bw: List<&2, F32>, Cw: List<&2, F32>,
     D: List<&2, F32>, w_out: List<&2, F32>,
     E: List<&2, F32>, Wh: List<&2, F32>}

type TS is Type:
  TS{loss: F32, p: TP, tokens: List<&2, F32>}

type SL is Type:
  SL{loss: F32, p: TP}

def loss_go(+left: Nat, +t: Nat, +y: List<&2, F32>,
    +Wh: List<&2, F32>, +acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      +y_t = take_row(y, Nat.mul(t, 64n), 64n)
      loss_go(p, (t + 1n : Nat), y, Wh,
        (acc + ce_loss(matvec(Wh, y_t, 64n), t, 8n, 0.125) : F32))

def dy_all_go(+left: Nat, +t: Nat, +y: List<&2, F32>,
    +Wh: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +y_t = take_row(y, Nat.mul(t, 64n), 64n)
      +dl = ce_dl(matvec(Wh, y_t, 64n), t, 8n, 0.125)
      dy_all_go(p, (t + 1n : Nat), y, Wh,
        push_row(acc, matvec_t(Wh, dl, 8n, 64n)))

def dWh_go(+left: Nat, +t: Nat, +y: List<&2, F32>,
    +Wh: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +y_t = take_row(y, Nat.mul(t, 64n), 64n)
      +dl = ce_dl(matvec(Wh, y_t, 64n), t, 8n, 0.125)
      dWh_go(p, (t + 1n : Nat), y, Wh,
        add_lists(acc, outer(dl, y_t)))

def train_update(g: GRAD, +dWh: List<&2, F32>,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>, +lr: F32) -> TP:
  match g:
    case GRAD{du, dW_in, ddt_bias, dB_bias, dC_bias, dB_norm_w,
        dC_norm_w, dD, dW_out, do_v, dq, dk, dz, dx, dqk, dscale,
        dADT, dBn, dCn, dDT, dtrap, ddd_dt, ddd_A, dBraw, dCraw,
        dang_raw}:
      TP{sgd(w_in, dW_in, lr), sgd(dt_bias, ddt_bias, lr),
        sgd(B_bias, dB_bias, lr), sgd(C_bias, dC_bias, lr),
        sgd(Bw, dB_norm_w, lr), sgd(Cw, dC_norm_w, lr),
        sgd(D, dD, lr), sgd(w_out, dW_out, lr),
        sgd(E, du, lr), sgd(Wh, dWh, lr)}

def step_f(f: FWD, +w_in: List<&2, F32>,
    +dt_bias: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>, +lr: F32) -> SL:
  match f:
    case FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, +y}:
      +dy = dy_all_go(8n, 0n, y, Wh, Nil{})
      +dWh = dWh_go(8n, 0n, y, Wh, zline(512n))
      f2 : FWD = forward(E, w_in, dt_bias, B_bias, C_bias, Bw, Cw,
        D, w_out, 8n)
      SL{loss_go(8n, 0n, y, Wh, 0.0),
        train_update(backward(f2, E, dy, w_in, w_out, Bw, Cw,
          dt_bias, B_bias, C_bias, D, 8n), dWh, w_in, dt_bias,
          B_bias, C_bias, Bw, Cw, D, w_out, E, Wh, lr)}

def step(p: TP, +lr: F32) -> SL:
  match p:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}:
      step_f(forward(E, w_in, dt_bias, B_bias, C_bias, Bw, Cw, D,
        w_out, 8n), w_in, dt_bias, B_bias, C_bias, Bw, Cw, D,
        w_out, E, Wh, lr)

def tok_f(f: FWD, +Wh: List<&2, F32>) -> List<&2, F32>:
  match f:
    case FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, y}:
      tok_go(8n, 0n, y, Wh, Nil{})

def step_tokens(p: TP) -> List<&2, F32>:
  match p:
    case TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh}:
      f : FWD = forward(E, w_in, dt_bias, B_bias, C_bias, Bw, Cw, D,
        w_out, 8n)
      tok_f(f, Wh)

def last_f(f: FWD, +Wh: List<&2, F32>,
    +losses: List<&2, F32>) -> TR:
  match f:
    case FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, +y}:
      TR{push_row(losses, Con{loss_go(8n, 0n, y, Wh, 0.0), Nil{}}),
        tok_go(8n, 0n, y, Wh, Nil{})}

def rest10(p: TP, +losses: List<&2, F32>) -> TR:
  match p:
    case TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh}:
      last_f(forward(E, w_in, dt_bias, B_bias, C_bias, Bw, Cw, D,
        w_out, 8n), Wh, losses)

def rest9(sl: SL, +losses: List<&2, F32>) -> TR:
  match sl:
    case SL{loss, p}:
      rest10(p, push_row(losses, Con{loss, Nil{}}))

def rest8(sl: SL, +lr: F32, +losses: List<&2, F32>) -> TR:
  match sl:
    case SL{loss, p}:
      rest9(step(p, lr), push_row(losses, Con{loss, Nil{}}))

def rest7(sl: SL, +lr: F32, +losses: List<&2, F32>) -> TR:
  match sl:
    case SL{loss, p}:
      rest8(step(p, lr), lr, push_row(losses, Con{loss, Nil{}}))

def rest6(sl: SL, +lr: F32, +losses: List<&2, F32>) -> TR:
  match sl:
    case SL{loss, p}:
      rest7(step(p, lr), lr, push_row(losses, Con{loss, Nil{}}))

def rest5(sl: SL, +lr: F32, +losses: List<&2, F32>) -> TR:
  match sl:
    case SL{loss, p}:
      rest6(step(p, lr), lr, push_row(losses, Con{loss, Nil{}}))

def rest4(sl: SL, +lr: F32, +losses: List<&2, F32>) -> TR:
  match sl:
    case SL{loss, p}:
      rest5(step(p, lr), lr, push_row(losses, Con{loss, Nil{}}))

def rest3(sl: SL, +lr: F32, +losses: List<&2, F32>) -> TR:
  match sl:
    case SL{loss, p}:
      rest4(step(p, lr), lr, push_row(losses, Con{loss, Nil{}}))

def rest2(sl: SL, +lr: F32, +losses: List<&2, F32>) -> TR:
  match sl:
    case SL{loss, p}:
      rest3(step(p, lr), lr, push_row(losses, Con{loss, Nil{}}))

def rest1(sl: SL, +lr: F32, +losses: List<&2, F32>) -> TR:
  match sl:
    case SL{loss, p}:
      rest2(step(p, lr), lr, push_row(losses, Con{loss, Nil{}}))

def train10(p: TP, +lr: F32) -> TR:
  rest1(step(p, lr), lr, Nil{})

type TR is Type:
  TR{losses: List<&2, F32>, tokens: List<&2, F32>}

def show_train(tr: TR) -> IO(Unit):
  match tr:
    case TR{losses, tokens}:
      do IO<Unit>:
        show_tag("train_loss", losses)
        show_tag("train_tokens", tokens)
        return Unit{}


# ---------------------------------------------------------------------------
# greedy generation, two weight-tied layers, sixteen tokens

def nth_nat(+n: Nat, xs: List<&2, Nat>) -> Nat:
  match n xs:
    case 0n Con{h, t}:
      h
    case 1n+p Con{h, t}:
      nth_nat(p, t)
    case _ _:
      0n

def embed_go(+left: Nat, +i: Nat, +tokens: List<&2, Nat>,
    +E: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +tok = nth_nat(i, tokens)
      embed_go(p, (i + 1n : Nat), tokens, E,
        push_row(acc, take_row(E, Nat.mul(tok, 64n), 64n)))

def embed(+tokens: List<&2, Nat>, +E: List<&2, F32>,
    +n: Nat) -> List<&2, F32>:
  embed_go(n, 0n, tokens, E, Nil{})

def layer2(f: FWD, +w_in: List<&2, F32>,
    +dt_bias: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +n: Nat) -> FWD:
  match f:
    case FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, y}:
      forward(y, w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, n)

def last_tok(f: FWD, +Wh: List<&2, F32>, +n: Nat) -> Nat:
  match f:
    case FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, y}:
      +yl = take_row(y, Nat.mul((n - 1n : Nat), 64n), 64n)
      argmax(matvec(Wh, yl, 64n), 8n)

def gen_go(+left: Nat, +n: Nat, +tokens: List<&2, Nat>,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>) -> List<&2, Nat>:
  match left:
    case 0n:
      tokens
    case 1n+p:
      +u = embed(tokens, E, n)
      y1 : FWD = layer2(forward(u, w_in, dt_bias, B_bias, C_bias, Bw, Cw,
        D, w_out, n), w_in, dt_bias, B_bias, C_bias, Bw, Cw, D,
        w_out, n)
      +next = last_tok(y1, Wh, n)
      gen_go(p, (n + 1n : Nat),
        List.append(&2, Nat, tokens, Con{next, Nil{}}), w_in,
        dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh)

def gen(p: TP, +steps: Nat) -> List<&2, Nat>:
  match p:
    case TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh}:
      gen_go(steps, 4n, Con{0n, Con{1n, Con{2n, Con{3n, Nil{}}}}},
        w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh)

def nat_row_f32(+left: Nat, +xs: List<&2, Nat>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      match xs:
        case Nil{}:
          acc
        case Con{h, t}:
          nat_row_f32(p, t, push_row(acc, Con{F32.from_nat(h), Nil{}}))

def show_f(f: FWD, +tag: String) -> IO(Unit):
  match f:
    case FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, y}:
      show_tag(tag, y)

def show_gen(+tokens: List<&2, Nat>, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    show_tag("gen_tokens", nat_row_f32(n, tokens, Nil{}))
    return Unit{}

def show_grad(g: GRAD) -> IO(Unit):
  match g:
    case GRAD{du, dW_in, ddt_bias, dB_bias, dC_bias, dB_norm_w,
        dC_norm_w, dD, dW_out, do_v, dq, dk, dz, dx, dqk, dscale,
        dADT, dBn, dCn, dDT, dtrap, ddd_dt, ddd_A, dBraw, dCraw,
        dang_raw}:
      do IO<Unit>:
        show_tag("du", du)
        show_tag("dW_in", dW_in)
        show_tag("ddt_bias", ddt_bias)
        show_tag("dB_bias", dB_bias)
        show_tag("dC_bias", dC_bias)
        show_tag("dW_norm_w", dB_norm_w)
        show_tag("dC_norm_w", dC_norm_w)
        show_tag("dD", dD)
        show_tag("dW_out", dW_out)
        show_tag("do_v", do_v)
        show_tag("dq", dq)
        show_tag("dk", dk)
        show_tag("dz", dz)
        show_tag("dx", dx)
        show_tag("dqk", dqk)
        show_tag("dscale", dscale)
        show_tag("dADT", dADT)
        show_tag("dBn", dBn)
        show_tag("dCn", dCn)
        show_tag("dDT", dDT)
        show_tag("dtrap", dtrap)
        show_tag("ddd_dt", ddd_dt)
        show_tag("ddd_A", ddd_A)
        show_tag("dBraw", dBraw)
        show_tag("dCraw", dCraw)
        show_tag("dang_raw", dang_raw)
        return Unit{}

def bw1(f: FWD, +dy: List<&2, F32>, +w_out: List<&2, F32>,
    +D: List<&2, F32>, +n: Nat) -> BW:
  match f:
    case FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, +ADT, trap, gamma, shift, scale, Bn, Cn, angle, +q, +k,
        +qk, states, out_v, y_inner, y}:
      bw_start(z, x, ADT, q, k, qk, scale, states, out_v, y_inner,
        D, w_out, dy, n)

def show_bw(bw: BW) -> IO(Unit):
  match bw:
    case BW{+dq, +dk, +dx, +dqk, +dscale, +dADT, dD, dstate, do_v, +dz,
        dW_out, du}:
      do IO<Unit>:
        show_tag("do_v", do_v)
        show_tag("dq", dq)
        show_tag("dk", dk)
        show_tag("dz", dz)
        show_tag("dx", dx)
        show_tag("dqk", dqk)
        show_tag("dscale", dscale)
        show_tag("dADT", dADT)
        show_tag("dD", dD)
        return Unit{}

# ---------------------------------------------------------------------------
# probe

def ml64(ml: TA.ML<8n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml304(ml: TA.ML<304n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml4x16(ml: TA.ML<4n, 16n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1x16(ml: TA.ML<1n, 16n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1x4(ml: TA.ML<1n, 4n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml64x128(ml: TA.ML<64n, 128n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def main() -> IO(Unit):
  do IO<Unit>:
    u : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(dir(), "/u.bin"), 0)
    wi : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(dir(), "/in_proj.bin"), 0)
    dtb : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(dir(), "/dt_bias.bin"), 0)
    bb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(dir(), "/B_bias.bin"), 0)
    cb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(dir(), "/C_bias.bin"), 0)
    bw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(dir(), "/B_norm_w.bin"), 0)
    cw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(dir(), "/C_norm_w.bin"), 0)
    dd : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(dir(), "/D.bin"), 0)
    wo : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(dir(), "/out_proj.bin"), 0)
    run(forward(ml64(TA.Mat.to_list(8n, 64n, u)), ml304(TA.Mat.to_list(304n, 64n, wi)),
      ml1x4(TA.Mat.to_list(1n, 4n, dtb)), ml4x16(TA.Mat.to_list(4n, 16n, bb)),
      ml4x16(TA.Mat.to_list(4n, 16n, cb)), ml1x16(TA.Mat.to_list(1n, 16n, bw)),
      ml1x16(TA.Mat.to_list(1n, 16n, cw)), ml1x4(TA.Mat.to_list(1n, 4n, dd)),
      ml64x128(TA.Mat.to_list(64n, 128n, wo)), 8n))
    return Unit{}
