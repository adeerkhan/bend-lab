# Sixteen-token greedy generation, decoded one token at a time.
#
# The full-prefix generator (`probe_mamba3_gen.bend`) recomputes the whole
# forward on every step. This one carries the decode state across tokens and
# processes each row exactly once.
#
# Exact parity with the full-prefix pass is possible, but only with one
# wrinkle: the reference pass computes row t's `scale` from row t+1's `DT` and
# `trap` (the trapezoid: scale = DT_t*trap_t + DT_{t+1}*(1 - trap_{t+1})),
# and the final row of a pass gets scale = 0 (its `shift` row is `Nil`).
# An online decoder cannot know DT_{t+1} when it emits row t, so it folds
# every row with scale = 0 (the final-row rule) and, when the next row's
# DT/trap arrive, applies the additive rank-one correction
#   S[v, q] += ((DT_t*trap_t + DT_{t+1}*(1 - trap_{t+1})) * x_v) * k_q
# to the carried 2048-float SSM state. The float op order matches the full
# pass's `row_set` exactly, so the state, and every downstream logit, is the
# same F32 computation. The angle state has no future dependency, so it
# threads directly.
#
# Gate: the sixteen greedy tokens equal the full-prefix generator exactly.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

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

# ---- pass B for one row ----

def pb_one(+left: Nat, +h: Nat, +angle_prev: List<&2, F32>,
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
      +cb = M.take_row(C_bias, Nat.mul(h, 16n), 16n)
      +bb = M.take_row(B_bias, Nat.mul(h, 16n), 16n)
      +qpre = M.add_lists(Cn_t, cb)
      +kpre = M.add_lists(Bn_t, bb)
      +qk_h = (M.nth(h, gamma_t) * M.dot(qpre, kpre) : F32)
      +ap = M.take_row(angle_prev, Nat.mul(h, 4n), 4n)
      +ang_h = M.angle_step(ap, ang_raw_t, M.nth(h, DT_t), 4n)
      +qh = M.rope(qpre, ang_h, 8n)
      +kh = M.rope(kpre, ang_h, 8n)
      pb_one(p, (h + 1n : Nat), angle_prev, Bn_t, Cn_t, ang_raw_t,
        DT_t, gamma_t, B_bias, C_bias,
        M.push_row(acc_angle, ang_h), M.push_row(acc_q, qh),
        M.push_row(acc_k, kh), M.push_row(acc_qk, Con{qk_h, Nil{}}))

def pb_row(+angle_prev: List<&2, F32>, +Bn: List<&2, F32>,
    +Cn: List<&2, F32>, +ang_raw: List<&2, F32>, +DT: List<&2, F32>,
    +gamma: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>) -> PB:
  pb_one(4n, 0n, angle_prev, Bn, Cn, ang_raw, DT, gamma,
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

def layer_row_pb(ssm1: Array<F32>, +angle2: List<&2, F32>,
    +x: List<&2, F32>, +z: List<&2, F32>,
    +qk: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>,
    +ADT: List<&2, F32>, +trap: List<&2, F32>, +DT: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>) -> LOUT:
  s : M.SA = M.scan_all_st(ssm1, q, k, x, qk, M.zline(4n), ADT, D)
  layer_row_c(s, angle2, x, z, k, DT, trap, w_out)

def layer_row_pb2(ssm1: Array<F32>,
    +x: List<&2, F32>, +z: List<&2, F32>, +ADT: List<&2, F32>,
    +trap: List<&2, F32>, +DT: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    r: PB) -> LOUT:
  match r:
    case PB{angle2, q, k, qk}:
      layer_row_pb(ssm1, angle2,
        x, z, qk, q, k, ADT, trap, DT, D, w_out)

def layer_row(st: ST2, +t: Nat, +u: List<&2, F32>,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>) -> LOUT:
  match st:
    case ST2{ssm, angle, pend_x, pend_k, pend_dt, pend_trap}:
      +pre = M.matvec(w_in, u, 64n)
      +z = M.take_row(pre, 0n, 128n)
      +x = M.take_row(pre, 128n, 128n)
      +Braw = M.take_row(pre, 256n, 16n)
      +Craw = M.take_row(pre, 272n, 16n)
      +dd_dt = M.take_row(pre, 288n, 4n)
      +dd_A = M.take_row(pre, 292n, 4n)
      +trap_raw = M.take_row(pre, 296n, 4n)
      +ang_raw = M.take_row(pre, 300n, 4n)
      +A = M.htail_row(dd_A)
      +DT = M.softplus_row(dd_dt, dt_bias)
      +ADT = M.mul_lists(A, DT)
      +trap = M.sigmoid_row(trap_raw)
      +Bn = M.rms(Braw, Bw, 16n)
      +Cn = M.rms(Craw, Cw, 16n)
      +gamma = M.mul_lists(DT, trap)
      layer_row_pb2(corr_sel(t, ssm, pend_x, pend_k, pend_dt, pend_trap,
        DT, trap),
        x, z, ADT, trap, DT, D, w_out,
        pb_row(angle, Bn, Cn, ang_raw, DT, gamma, B_bias, C_bias))

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

def maybe_append(+t: Nat, +nxt: Nat,
    out: List<&2, Nat>) -> List<&2, Nat>:
  match t:
    case 0n:
      out
    case 1n:
      out
    case 2n:
      out
    case 15n:
      out
    case _:
      List.append(&2, Nat, out, Con{nxt, Nil{}})

def head_done(l2o: LOUT, s1: ST2, out: List<&2, Nat>, +t: Nat,
    +Wh: List<&2, F32>) -> SR:
  match l2o:
    case LOUT{s2, y2}:
      +logits = M.matvec(Wh, y2, 64n)
      +nxt = M.argmax(logits, 8n)
      SR{ST{s1, s2}, maybe_append(t, nxt, out), next_in(t, nxt)}

def layer1_finish(l1o: LOUT, l2: ST2, out: List<&2, Nat>, +t: Nat,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +Wh: List<&2, F32>) -> SR:
  match l1o:
    case LOUT{s1, y1}:
      head_done(layer_row(l2, t, y1, w_in, dt_bias, B_bias, C_bias,
        Bw, Cw, D, w_out), s1, out, t, Wh)

def run_layers(st: ST, +t: Nat, +tok: Nat, out: List<&2, Nat>,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>) -> SR:
  match st:
    case ST{l1, l2}:
      +u = M.take_row(E, Nat.mul(tok, 64n), 64n)
      layer1_finish(layer_row(l1, t, u, w_in, dt_bias, B_bias, C_bias,
        Bw, Cw, D, w_out), l2, out, t,
        w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, Wh)

def step_go(+left: Nat, +t: Nat,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>,
    r: SR) -> List<&2, Nat>:
  match left r:
    case 0n SR{st, out, last_in}:
      out
    case 1n+p SR{st, out, last_in}:
      +tok = last_in
      step_go(p, (t + 1n : Nat),
        w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh,
        run_layers(st, t, tok, out,
          w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh))

def fresh2() -> ST2:
  ST2{M.arr_of(2048n, M.zline(2048n)), M.zline(16n),
      Nil{}, Nil{}, Nil{}, Nil{}}

def fresh() -> ST:
  ST{fresh2(), fresh2()}

def step(p: M.TP, +steps: Nat) -> List<&2, Nat>:
  match p:
    case M.TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh}:
      step_go(steps, 0n,
        w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh,
        SR{fresh(), Con{0n, Con{1n, Con{2n, Con{3n, Nil{}}}}}, 0n})

def ddir() -> String:
  "reference/mamba3_data"

def show_step(+tokens: List<&2, Nat>, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    M.show_tag("step_tokens", M.nat_row_f32(n, tokens, Nil{}))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    wi : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(ddir(), "/train_w_in.bin"), 0)
    dtb : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/train_dt_bias.bin"), 0)
    bb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/train_B_bias.bin"), 0)
    cb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(ddir(), "/train_C_bias.bin"), 0)
    bw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/train_B_norm_w.bin"), 0)
    cw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/train_C_norm_w.bin"), 0)
    dd : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(ddir(), "/train_D.bin"), 0)
    wo : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(ddir(), "/train_w_out.bin"), 0)
    e0 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/train_E.bin"), 0)
    wh : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(ddir(), "/train_WhT.bin"), 0)
    show_step(step(M.TP{
      M.ml304(TA.Mat.to_list(304n, 64n, wi)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
      M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
      M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
      M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
      M.ml64x128(TA.Mat.to_list(64n, 128n, wo)),
      M.ml64(TA.Mat.to_list(8n, 64n, e0)),
      M.ml64(TA.Mat.to_list(8n, 64n, wh))}, 16n), 16n)
    return Unit{}
