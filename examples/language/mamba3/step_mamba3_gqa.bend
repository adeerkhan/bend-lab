# C2: the two-layer LM at ngroups 2, rope_fraction 1.0 decodes greedily.
#
# Same decode step as the SISO LM (step_mamba3.bend), carried row by row,
# with the GQA block inside each layer (2 B/C groups shared by the 4
# v-heads, per-v-head B/C biases, 8 angle pairs per head, all 16 q/k dims
# rotated). The full-prefix pass (`gen_gqa`) recomputes the whole prefix
# for every token, which is the reference; the online step (`step`)
# carries the state row by row with a rank-one correction for the
# trapezoid, which is the decoder. The gate compares both Bend paths at
# 16 and 64 tokens, exactly, against each other and against the twin's
# greedy decode (`reference/mamba3_cfg_lm.py --config gqa`).
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/gqa_lm_data"


# ---- the full-prefix generator (GQA, recompute the prefix) ----

def layer_gqa(+u: List<&2, F32>, +w_in: List<&2, F32>,
    +dt_bias: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +n: Nat) -> List<&2, F32>:
  M.add_lists(u, M.block_y(M.forward_gqa(u, w_in, dt_bias, B_bias,
    C_bias, Bw, Cw, D, w_out, n)))

def lm_h2_gqa(+u: List<&2, F32>, +w1: List<&2, F32>,
    +dt1: List<&2, F32>, +bb1: List<&2, F32>,
    +cb1: List<&2, F32>, +bw1: List<&2, F32>,
    +cw1: List<&2, F32>, +D1: List<&2, F32>,
    +wo1: List<&2, F32>, +w2: List<&2, F32>,
    +dt2: List<&2, F32>, +bb2: List<&2, F32>,
    +cb2: List<&2, F32>, +bw2: List<&2, F32>,
    +cw2: List<&2, F32>, +D2: List<&2, F32>,
    +wo2: List<&2, F32>, +n: Nat) -> List<&2, F32>:
  +h1 = layer_gqa(u, w1, dt1, bb1, cb1, bw1, cw1, D1, wo1, n)
  layer_gqa(h1, w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, n)

def gen_go(+left: Nat, +n: Nat, +tokens: List<&2, Nat>,
    +w1: List<&2, F32>, +dt1: List<&2, F32>,
    +bb1: List<&2, F32>, +cb1: List<&2, F32>,
    +bw1: List<&2, F32>, +cw1: List<&2, F32>,
    +D1: List<&2, F32>, +wo1: List<&2, F32>,
    +w2: List<&2, F32>, +dt2: List<&2, F32>,
    +bb2: List<&2, F32>, +cb2: List<&2, F32>,
    +bw2: List<&2, F32>, +cw2: List<&2, F32>,
    +D2: List<&2, F32>, +wo2: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>,
    +nw: List<&2, F32>) -> List<&2, Nat>:
  match left:
    case 0n:
      tokens
    case 1n+p:
      +u = M.embed(tokens, E, n)
      +h2 = lm_h2_gqa(u, w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
        w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, n)
      +hn = M.rms_rows(h2, nw, n)
      +next = M.argmax(M.matvec(Wh,
        M.take_row(hn, Nat.mul(Nat.sub(n, 1n), 64n), 64n), 64n), 8n)
      gen_go(p, (n + 1n : Nat),
        List.append(&2, Nat, tokens, Con{next, Nil{}}),
        w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
        w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, E, Wh, nw)

def gen_gqa(+steps: Nat, +w1: List<&2, F32>,
    +dt1: List<&2, F32>, +bb1: List<&2, F32>,
    +cb1: List<&2, F32>, +bw1: List<&2, F32>,
    +cw1: List<&2, F32>, +D1: List<&2, F32>,
    +wo1: List<&2, F32>, +w2: List<&2, F32>,
    +dt2: List<&2, F32>, +bb2: List<&2, F32>,
    +cb2: List<&2, F32>, +bw2: List<&2, F32>,
    +cw2: List<&2, F32>, +D2: List<&2, F32>,
    +wo2: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>, +nw: List<&2, F32>) -> List<&2, Nat>:
  gen_go(steps, 4n,
    Con{0n, Con{1n, Con{2n, Con{3n, Nil{}}}}},
    w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
    w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, E, Wh, nw)

# ---- decode state ----
#
# ssm: 2048 floats, 4 heads x 32 v-rows x 16 qk dims (carried as the same
# in-place Array the full pass uses). angle: 16 floats, 4 heads x 4.
# pend_*: row t-1's x (128), roped k (64), DT (4), trap (4), used to apply
# the correction once row t's DT/trap are known. Nil at t == 0 (skipped).

type ST2 is Type:
  ST2{ssm: Array<F32>, angle: List<&2, F32>,
      pend_x: List<&2, F32>, pend_k: List<&2, F32>,
      pend_dt: List<&2, F32>, pend_trap: List<&2, F32>}

type ST is Type:
  ST{l1: ST2, l2: ST2}

type PB is Type:
  PB{angle: List<&2, F32>, q: List<&2, F32>,
     k: List<&2, F32>, qk: List<&2, F32>}

type LOUT is Type:
  LOUT{st: ST2, y: List<&2, F32>}

type SR is Type:
  SR{st: ST, out: List<&2, Nat>, last_in: Nat}

# ---- the additive rank-one correction of the previous row ----

def arr_of_tup(r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{arr, _}:
      arr

def corr_one(+left: Nat, +idx: U32, +add: F32, +k: List<&2, F32>,
    cur: Array<F32> & F32) -> Array<F32> & F32:
  match left k:
    case 0n _:
      cur
    case 1n+p Con{h, t}:
      match cur:
        case Tuple{arr, v}:
          arr2 = Array.set(F32, arr, idx,
            ((v + (add * h : F32) : F32) : F32))
          corr_one(p, U32.add(idx, U32.from_nat(1n)), add, t,
            Array.get(F32, arr2, U32.add(idx, U32.from_nat(1n))))
    case _ _:
      match cur:
        case Tuple{arr, _}:
          Tuple{arr, 0.0}

def corr_rows(+left: Nat, +p: Nat, +off: U32, +x_h: List<&2, F32>,
    +k_h: List<&2, F32>, +sc: F32, arr: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      arr
    case 1n+pr:
      +base = U32.add(off, U32.from_nat(Nat.mul(p, 16n)))
      +xv = M.nth(p, x_h)
      arr2 = arr_of_tup(corr_one(16n, base, (sc * xv : F32), k_h,
        Array.get(F32, arr, base)))
      corr_rows(pr, (p + 1n : Nat), off, x_h, k_h, sc, arr2)

def corr_head(arr: Array<F32>, +off: U32, +x_h: List<&2, F32>,
    +k_h: List<&2, F32>, +sc: F32) -> Array<F32>:
  corr_rows(32n, 0n, off, x_h, k_h, sc, arr)

def corr_heads(arr: Array<F32>, +pend_x: List<&2, F32>,
    +pend_k: List<&2, F32>, +sc: List<&2, F32>) -> Array<F32>:
  corr_head(
    corr_head(
      corr_head(
        corr_head(arr, U32.from_nat(0n),
          M.take_row(pend_x, 0n, 32n), M.take_row(pend_k, 0n, 16n),
          M.nth(0n, sc)),
        U32.from_nat(512n),
        M.take_row(pend_x, 32n, 32n), M.take_row(pend_k, 16n, 16n),
        M.nth(1n, sc)),
      U32.from_nat(1024n),
      M.take_row(pend_x, 64n, 32n), M.take_row(pend_k, 32n, 16n),
      M.nth(2n, sc)),
    U32.from_nat(1536n),
    M.take_row(pend_x, 96n, 32n), M.take_row(pend_k, 48n, 16n),
    M.nth(3n, sc))

# delta per head: gamma (prev row) + DT_t*(1 - trap_t) (current row).
def corr_sc(+DT_t: List<&2, F32>, +trap_t: List<&2, F32>,
    +pend_dt: List<&2, F32>, +pend_trap: List<&2, F32>) -> List<&2, F32>:
  M.add_lists(M.mul_lists(pend_dt, pend_trap),
    M.mul_lists(DT_t, M.one_minus(trap_t)))

# at t == 0 there is no previous row to correct; apply the correction only
# when a pending row exists.
def corr_sel(+t: Nat, arr: Array<F32>, +pend_x: List<&2, F32>,
    +pend_k: List<&2, F32>, +pend_dt: List<&2, F32>,
    +pend_trap: List<&2, F32>, +DT_t: List<&2, F32>,
    +trap_t: List<&2, F32>) -> Array<F32>:
  match t:
    case 0n:
      arr
    case 1n+p:
      corr_heads(arr, pend_x, pend_k,
        corr_sc(DT_t, trap_t, pend_dt, pend_trap))

# ---- pass B for one row (GQA: 2 B/C groups, 8 angle pairs) ----

def pb_one_gqa(+left: Nat, +h: Nat, +angle_prev: List<&2, F32>,
    +Bn_t: List<&2, F32>, +Cn_t: List<&2, F32>,
    +ang_raw_t: List<&2, F32>, +DT_t: List<&2, F32>,
    +gamma_t: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +acc_angle: List<&2, F32>,
    +acc_q: List<&2, F32>, +acc_k: List<&2, F32>,
    acc_qk: List<&2, F32>) -> PB:
  match left:
    case 0n:
      PB{acc_angle, acc_q, acc_k, acc_qk}
    case 1n+p:
      +g = Nat.div(h, 2n)
      +cb = M.take_row(C_bias, Nat.mul(h, 16n), 16n)
      +bb = M.take_row(B_bias, Nat.mul(h, 16n), 16n)
      +qpre = M.add_lists(M.take_row(Cn_t, Nat.mul(g, 16n), 16n), cb)
      +kpre = M.add_lists(M.take_row(Bn_t, Nat.mul(g, 16n), 16n), bb)
      +qk_h = (M.nth(h, gamma_t) * M.dot(qpre, kpre) : F32)
      +ap = M.take_row(angle_prev, Nat.mul(h, 8n), 8n)
      +ang_h = M.angle_step(ap, ang_raw_t, M.nth(h, DT_t), 8n)
      +qh = M.rope(qpre, ang_h, 8n)
      +kh = M.rope(kpre, ang_h, 8n)
      pb_one_gqa(p, (h + 1n : Nat), angle_prev, Bn_t, Cn_t, ang_raw_t,
        DT_t, gamma_t, B_bias, C_bias,
        M.push_row(acc_angle, ang_h), M.push_row(acc_q, qh),
        M.push_row(acc_k, kh), M.push_row(acc_qk, Con{qk_h, Nil{}}))

def pb_row_gqa(+angle_prev: List<&2, F32>, +Bn: List<&2, F32>,
    +Cn: List<&2, F32>, +ang_raw: List<&2, F32>, +DT: List<&2, F32>,
    +gamma: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>) -> PB:
  pb_one_gqa(4n, 0n, angle_prev, Bn, Cn, ang_raw, DT, gamma,
    B_bias, C_bias, Nil{}, Nil{}, Nil{}, Nil{})

# ---- one layer's row; both the producing and the consuming side ----

def layer_row_c(s: M.SA, +angle2: List<&2, F32>,
    +x: List<&2, F32>, +z: List<&2, F32>, +k: List<&2, F32>,
    +DT: List<&2, F32>, +trap: List<&2, F32>,
    +w_out: List<&2, F32>) -> LOUT:
  match s:
    case M.SA{arr, out}:
      +yi = M.mul_lists(out, M.silu_row(z))
      +y = M.matvec(w_out, yi, 128n)
      LOUT{ST2{arr, angle2, x, k, DT, trap}, y}

def layer_row_pb_gqa(ssm1: Array<F32>, +angle2: List<&2, F32>,
    +x: List<&2, F32>, +z: List<&2, F32>,
    +qk: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>,
    +ADT: List<&2, F32>, +trap: List<&2, F32>, +DT: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>) -> LOUT:
  s : M.SA = M.scan_all_st(ssm1, q, k, x, qk, M.zline(4n), ADT, D)
  layer_row_c(s, angle2, x, z, k, DT, trap, w_out)

def layer_row_pb2_gqa(ssm1: Array<F32>,
    +x: List<&2, F32>, +z: List<&2, F32>, +ADT: List<&2, F32>,
    +trap: List<&2, F32>, +DT: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    r: PB) -> LOUT:
  match r:
    case PB{angle2, q, k, qk}:
      layer_row_pb_gqa(ssm1, angle2,
        x, z, qk, q, k, ADT, trap, DT, D, w_out)

def layer_row_gqa(st: ST2, +t: Nat, +u: List<&2, F32>,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>) -> LOUT:
  match st:
    case ST2{ssm, angle, pend_x, pend_k, pend_dt, pend_trap}:
      +pre = M.matvec(w_in, u, 64n)
      +z = M.take_row(pre, 0n, 128n)
      +x = M.take_row(pre, 128n, 128n)
      +Braw = M.take_row(pre, 256n, 32n)
      +Craw = M.take_row(pre, 288n, 32n)
      +dd_dt = M.take_row(pre, 320n, 4n)
      +dd_A = M.take_row(pre, 324n, 4n)
      +trap_raw = M.take_row(pre, 328n, 4n)
      +ang_raw = M.take_row(pre, 332n, 8n)
      +A = M.htail_row(dd_A)
      +DT = M.softplus_row(dd_dt, dt_bias)
      +ADT = M.mul_lists(A, DT)
      +trap = M.sigmoid_row(trap_raw)
      +Bn = List.append(&2, F32, M.rms(M.take_row(Braw, 0n, 16n), Bw, 16n),
        M.rms(M.take_row(Braw, 16n, 16n), Bw, 16n))
      +Cn = List.append(&2, F32, M.rms(M.take_row(Craw, 0n, 16n), Cw, 16n),
        M.rms(M.take_row(Craw, 16n, 16n), Cw, 16n))
      +gamma = M.mul_lists(DT, trap)
      layer_row_pb2_gqa(corr_sel(t, ssm, pend_x, pend_k, pend_dt, pend_trap,
        DT, trap),
        x, z, ADT, trap, DT, D, w_out,
        pb_row_gqa(angle, Bn, Cn, ang_raw, DT, gamma, B_bias, C_bias))


# ---- the two weight-tied layers for one token row ----

def next_in(+t: Nat, +nxt: Nat) -> Nat:
  match t:
    case 0n:
      1n
    case 1n:
      2n
    case 2n:
      3n
    case _:
      nxt

def append_if_eq(+eq: Cmp, +nxt: Nat,
    out: List<&2, Nat>) -> List<&2, Nat>:
  match eq:
    case EQ{}:
      out
    case _:
      List.append(&2, Nat, out, Con{nxt, Nil{}})

def append_if(+lt: Cmp, +eq: Cmp, +nxt: Nat,
    out: List<&2, Nat>) -> List<&2, Nat>:
  match lt:
    case LT{}:
      out
    case _:
      append_if_eq(eq, nxt, out)

def maybe_append(+t: Nat, +nxt: Nat, +first: Nat, +last: Nat,
    out: List<&2, Nat>) -> List<&2, Nat>:
  append_if(Nat.cmp(t, first), Nat.cmp(t, last), nxt, out)

def head_done(l2o: LOUT, s1: ST2, out: List<&2, Nat>, +t: Nat,
    +h1: List<&2, F32>,
    +Wh: List<&2, F32>, +nw: List<&2, F32>,
    +first: Nat, +last: Nat) -> SR:
  match l2o:
    case LOUT{s2, y2}:
      +h2 = M.add_lists(h1, y2)
      +hn = M.rms(h2, nw, 64n)
      +logits = M.matvec(Wh, hn, 64n)
      +nxt = M.argmax(logits, 8n)
      SR{ST{s1, s2}, maybe_append(t, nxt, first, last, out), next_in(t, nxt)}

def layer1_finish(l1o: LOUT, l2: ST2, out: List<&2, Nat>, +t: Nat,
    +w_in2: List<&2, F32>, +dt_bias2: List<&2, F32>,
    +B_bias2: List<&2, F32>, +C_bias2: List<&2, F32>,
    +Bw2: List<&2, F32>, +Cw2: List<&2, F32>,
    +D2: List<&2, F32>, +w_out2: List<&2, F32>,
    +u: List<&2, F32>,
    +Wh: List<&2, F32>, +nw: List<&2, F32>, +first: Nat, +last: Nat) -> SR:
  match l1o:
    case LOUT{s1, y1}:
      +h1 = M.add_lists(u, y1)
      head_done(layer_row_gqa(l2, t, h1, w_in2, dt_bias2, B_bias2, C_bias2,
        Bw2, Cw2, D2, w_out2), s1, out, t, h1, Wh, nw, first, last)

def run_layers(st: ST, +t: Nat, +tok: Nat, out: List<&2, Nat>,
    +w_in1: List<&2, F32>, +dt_bias1: List<&2, F32>,
    +B_bias1: List<&2, F32>, +C_bias1: List<&2, F32>,
    +Bw1: List<&2, F32>, +Cw1: List<&2, F32>,
    +D1: List<&2, F32>, +w_out1: List<&2, F32>,
    +w_in2: List<&2, F32>, +dt_bias2: List<&2, F32>,
    +B_bias2: List<&2, F32>, +C_bias2: List<&2, F32>,
    +Bw2: List<&2, F32>, +Cw2: List<&2, F32>,
    +D2: List<&2, F32>, +w_out2: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>, +nw: List<&2, F32>,
    +first: Nat, +last: Nat) -> SR:
  match st:
    case ST{l1, l2}:
      +u = M.take_row(E, Nat.mul(tok, 64n), 64n)
      layer1_finish(layer_row_gqa(l1, t, u, w_in1, dt_bias1, B_bias1,
        C_bias1, Bw1, Cw1, D1, w_out1), l2, out, t,
        w_in2, dt_bias2, B_bias2, C_bias2, Bw2, Cw2, D2, w_out2,
        u, Wh, nw, first, last)

def step_go(+left: Nat, +t: Nat,
    +w_in1: List<&2, F32>, +dt_bias1: List<&2, F32>,
    +B_bias1: List<&2, F32>, +C_bias1: List<&2, F32>,
    +Bw1: List<&2, F32>, +Cw1: List<&2, F32>,
    +D1: List<&2, F32>, +w_out1: List<&2, F32>,
    +w_in2: List<&2, F32>, +dt_bias2: List<&2, F32>,
    +B_bias2: List<&2, F32>, +C_bias2: List<&2, F32>,
    +Bw2: List<&2, F32>, +Cw2: List<&2, F32>,
    +D2: List<&2, F32>, +w_out2: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>, +nw: List<&2, F32>,
    +first: Nat, +last: Nat,
    r: SR) -> List<&2, Nat>:
  match left r:
    case 0n SR{st, out, last_in}:
      out
    case 1n+p SR{st, out, last_in}:
      +tok = last_in
      step_go(p, (t + 1n : Nat),
        w_in1, dt_bias1, B_bias1, C_bias1, Bw1, Cw1, D1, w_out1,
        w_in2, dt_bias2, B_bias2, C_bias2, Bw2, Cw2, D2, w_out2,
        E, Wh, nw, first, last,
        run_layers(st, t, tok, out,
          w_in1, dt_bias1, B_bias1, C_bias1, Bw1, Cw1, D1, w_out1,
          w_in2, dt_bias2, B_bias2, C_bias2, Bw2, Cw2, D2, w_out2,
          E, Wh, nw, first, last))

def fresh2() -> ST2:
  ST2{M.arr_of(2048n, M.zline(2048n)), M.zline(32n),
      Nil{}, Nil{}, Nil{}, Nil{}}

def fresh() -> ST:
  ST{fresh2(), fresh2()}

def step(+w1: List<&2, F32>, +dt1: List<&2, F32>,
    +bb1: List<&2, F32>, +cb1: List<&2, F32>,
    +bw1: List<&2, F32>, +cw1: List<&2, F32>,
    +D1: List<&2, F32>, +wo1: List<&2, F32>,
    +w2: List<&2, F32>, +dt2: List<&2, F32>,
    +bb2: List<&2, F32>, +cb2: List<&2, F32>,
    +bw2: List<&2, F32>, +cw2: List<&2, F32>,
    +D2: List<&2, F32>, +wo2: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>,
    +steps: Nat, +first: Nat, +last: Nat,
    +nw: List<&2, F32>) -> List<&2, Nat>:
  step_go(steps, 0n,
    w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
    w2, dt2, bb2, cb2, bw2, cw2, D2, wo2,
    E, Wh, nw, first, last,
    SR{fresh(), Con{0n, Con{1n, Con{2n, Con{3n, Nil{}}}}}, 0n})


def ml340(ml: TA.ML<340n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1x64(ml: TA.ML<1n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1x4(ml: TA.ML<1n, 4n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml4x16(ml: TA.ML<4n, 16n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show_tokens(+tokens: List<&2, Nat>, +n: Nat, +tag: String) -> IO(Unit):
  do IO<Unit>:
    M.show_tag(tag, M.nat_row_f32(n, tokens, Nil{}))
    return Unit{}


def run(+w1: List<&2, F32>,
    +dt1: List<&2, F32>, +bb1: List<&2, F32>,
    +cb1: List<&2, F32>, +bw1: List<&2, F32>,
    +cw1: List<&2, F32>, +D1: List<&2, F32>,
    +wo1: List<&2, F32>, +w2: List<&2, F32>,
    +dt2: List<&2, F32>, +bb2: List<&2, F32>,
    +cb2: List<&2, F32>, +bw2: List<&2, F32>,
    +cw2: List<&2, F32>, +D2: List<&2, F32>,
    +wo2: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>, +nw: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    show_tokens(step(w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
      w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, E, Wh,
      16n, 3n, 15n, nw), 16n, "gqa_step16")
    show_tokens(step(w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
      w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, E, Wh,
      64n, 3n, 63n, nw), 64n, "gqa_step64")
    show_tokens(gen_gqa(12n, w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
      w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, E, Wh, nw),
      16n, "gqa_ref16")
    show_tokens(gen_gqa(60n, w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
      w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, E, Wh, nw),
      64n, "gqa_ref64")
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    p1w : TA.Mat<340n, 64n> <- T.Mat.load(340n, 64n, String.append(ddir(), "/l1_in_proj.bin"), 0)
    p1t : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l1_dt_bias.bin"), 0)
    p1b : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l1_B_bias.bin"), 0)
    p1c : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l1_C_bias.bin"), 0)
    p1w2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l1_B_norm_w.bin"), 0)
    p1c2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l1_C_norm_w.bin"), 0)
    p1d : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l1_D.bin"), 0)
    p1o : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/l1_out_proj.bin"), 0)
    p2w : TA.Mat<340n, 64n> <- T.Mat.load(340n, 64n, String.append(ddir(), "/l2_in_proj.bin"), 0)
    p2t : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l2_dt_bias.bin"), 0)
    p2b : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l2_B_bias.bin"), 0)
    p2c : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/l2_C_bias.bin"), 0)
    p2w2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l2_B_norm_w.bin"), 0)
    p2c2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/l2_C_norm_w.bin"), 0)
    p2d : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/l2_D.bin"), 0)
    p2o : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/l2_out_proj.bin"), 0)
    e0 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/E.bin"), 0)
    wh : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/Wh.bin"), 0)
    nw : TA.Mat<1n, 64n> <- T.Mat.load(1n, 64n, String.append(ddir(), "/nw.bin"), 0)
    run(ml340(TA.Mat.to_list(340n, 64n, p1w)), ml1x4(TA.Mat.to_list(1n, 4n, p1t)),
      ml4x16(TA.Mat.to_list(4n, 16n, p1b)), ml4x16(TA.Mat.to_list(4n, 16n, p1c)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, p1w2)), M.ml1x16(TA.Mat.to_list(1n, 16n, p1c2)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, p1d)), M.ml64x128(TA.Mat.to_list(64n, 128n, p1o)),
      ml340(TA.Mat.to_list(340n, 64n, p2w)), ml1x4(TA.Mat.to_list(1n, 4n, p2t)),
      ml4x16(TA.Mat.to_list(4n, 16n, p2b)), ml4x16(TA.Mat.to_list(4n, 16n, p2c)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, p2w2)), M.ml1x16(TA.Mat.to_list(1n, 16n, p2c2)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, p2d)), M.ml64x128(TA.Mat.to_list(64n, 128n, p2o)),
      M.ml64(TA.Mat.to_list(8n, 64n, e0)), M.ml64(TA.Mat.to_list(8n, 64n, wh)),
      ml1x64(TA.Mat.to_list(1n, 64n, nw)))
    return Unit{}
