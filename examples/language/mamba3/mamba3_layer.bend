# The SISO copy block as a BendTorch `T.Layer`, trained ten steps with
# `T.Train`. Gate: the loss list and the generated tokens equal the custom
# ten-step trainer at the same learning rate.
#
# The ten-step trainer's math is already hand-written in `mamba3_siso.bend`
# (`forward`, `backward`, `sgd`, `loss_go`, `dy_all_go`, `dWh_go`). This
# module wires it into `T.Layer` and `T.Train`; the layer output `Y` is the
# flat hidden y (512 floats). The loss must know WH, which lives in the
# parameters, so the target type carries the 8 target tokens and a copy of
# WH. The optimizer update in `step` is plain SGD, matching the custom one.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def POS() -> Nat:
  8n

def V() -> Nat:
  8n

type TP is Type:
  TP{w_in: List<&2, F32>, dt_bias: List<&2, F32>,
     B_bias: List<&2, F32>, C_bias: List<&2, F32>,
     Bw: List<&2, F32>, Cw: List<&2, F32>,
     D: List<&2, F32>, w_out: List<&2, F32>,
     E: List<&2, F32>, Wh: List<&2, F32>}

def bp_of(f: M.FWD) -> M.FWD & List<&2, F32>:
  match f:
    case M.FWD{+pre, +z, +x, +Braw, +Craw, +dd_dt, +dd_A, +trap_raw, +ang_raw,
        +A, +DT, +ADT, +trap, +gamma, +shift, +scale, +Bn, +Cn, +angle, +q, +k,
        +qk, +states, +out_v, +y_inner, +y}:
      Tuple{M.FWD{pre, z, x, Braw, Craw, dd_dt, dd_A, trap_raw, ang_raw,
        A, DT, ADT, trap, gamma, shift, scale, Bn, Cn, angle, q, k,
        qk, states, out_v, y_inner, y}, y}

def tp_flat(+w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, w_in, List.append(&2, F32, dt_bias,
    List.append(&2, F32, B_bias, List.append(&2, F32, C_bias,
      List.append(&2, F32, Bw, List.append(&2, F32, Cw,
        List.append(&2, F32, D, List.append(&2, F32, w_out,
          List.append(&2, F32, E, Wh)))))))))

def flat_tp(p: TP) -> List<&2, F32>:
  match p:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}:
      tp_flat(w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh)

def tp_pl(p: TP) -> T.PL<TP>:
  match p:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}:
      T.PL{TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh},
        tp_flat(w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh)}

def tp_count(u: Unit) -> U32:
  U32.from_nat(28840n)

def tp_zeros_grad() -> TP:
  TP{M.zline(19456n), M.zline(4n), M.zline(64n), M.zline(64n),
    M.zline(16n), M.zline(16n), M.zline(4n), M.zline(8192n),
    M.zline(512n), M.zline(512n)}

def tp_zeros(u: Unit) -> TP:
  tp_zeros_grad()

def tp_add_list(+a: List<&2, F32>, +b: List<&2, F32>) -> List<&2, F32>:
  match a b:
    case Nil{} _:
      b
    case _ Nil{}:
      a
    case Con{x, t1} Con{y, t2}:
      Con{(x + y : F32), tp_add_list(t1, t2)}

def match_tp_add(p: TP, g: TP) -> TP:
  match p g:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}
      TP{+dw_in, +ddt_bias, +dB_bias, +dC_bias,
        +dBw, +dCw, +dD, +dw_out, +dE, +dWh}:
      TP{tp_add_list(w_in, dw_in), tp_add_list(dt_bias, ddt_bias),
        tp_add_list(B_bias, dB_bias), tp_add_list(C_bias, dC_bias),
        tp_add_list(Bw, dBw), tp_add_list(Cw, dCw), tp_add_list(D, dD),
        tp_add_list(w_out, dw_out), tp_add_list(E, dE),
        tp_add_list(Wh, dWh)}

def tp_sgd(p: TP, g: TP, +lr: F32) -> TP:
  match p g:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}
      TP{+dw_in, +ddt_bias, +dB_bias, +dC_bias,
        +dBw, +dCw, +dD, +dw_out, +dE, +dWh}:
      TP{M.sgd(w_in, dw_in, lr), M.sgd(dt_bias, ddt_bias, lr),
        M.sgd(B_bias, dB_bias, lr), M.sgd(C_bias, dC_bias, lr),
        M.sgd(Bw, dBw, lr), M.sgd(Cw, dCw, lr), M.sgd(D, dD, lr),
        M.sgd(w_out, dw_out, lr), M.sgd(E, dE, lr), M.sgd(Wh, dWh, lr)}

type Tgt is Type:
  Tgt{targets: List<&2, Nat>, Wh: List<&2, F32>}

def take_n_s(+n: Nat, xs: List<&2, F32>) -> List<&2, F32>:
  match n xs:
    case 0n _:
      Nil{}
    case 1n+p Con{h, t}:
      Con{h, take_n_s(p, t)}
    case _ Nil{}:
      Nil{}

def take_row_s(+off: Nat, +xs: List<&2, F32>, +n: Nat) -> List<&2, F32>:
  match off:
    case 0n:
      take_n_s(n, xs)
    case 1n+p:
      match xs:
        case Nil{}:
          Nil{}
        case Con{_, t}:
          take_row_s(p, t, n)

def nth_nat_s(+n: Nat, xs: List<&2, Nat>) -> Nat:
  match n xs:
    case 0n Con{h, _}:
      h
    case 1n+p Con{_, t}:
      nth_nat_s(p, t)
    case _ Nil{}:
      0n

def ce_fused(+left: Nat, +t: Nat, +targets: List<&2, Nat>,
    +y: List<&2, F32>, +Wh: List<&2, F32>,
    +acc: F32, dy: List<&2, F32>) -> T.LossOut<List<&2, F32>>:
  match left:
    case 0n:
      T.LossOut{acc, dy}
    case 1n+p:
      +y_t = take_row_s(Nat.mul(t, 64n), y, 64n)
      +logits = M.matvec(Wh, y_t, 64n)
      +tt = nth_nat_s(t, targets)
      +dl = M.ce_dl(logits, tt, V(), 0.125)
      ce_fused(p, (t + 1n : Nat), targets, y, Wh,
        (acc + M.ce_loss(logits, tt, V(), 0.125) : F32),
        M.push_row(dy, M.matvec_t(Wh, dl, V(), 64n)))

def loss(y: List<&2, F32>, t: Tgt) -> T.LossOut<List<&2, F32>>:
  match t:
    case Tgt{targets, Wh}:
      ce_fused(POS(), 0n, targets, y, Wh, 0.0, Nil{})

def fwdfin(r: M.FWD & List<&2, F32>,
    p: TP) -> T.Fwd<TP, List<&2, F32>, M.FWD>:
  match r:
    case Tuple{f2, y}:
      T.Fwd{p, y, f2}

def bwd_fin(gr: M.GRAD, p: TP, +dWh: List<&2, F32>,
    g: TP) -> T.Bwd<TP, List<&2, F32>>:
  match gr:
    case M.GRAD{+du, +dW_in, +ddt_bias, +dB_bias, +dC_bias,
        +dB_norm_w, +dC_norm_w, +dD, +dW_out, +do_v, +dq, +dk, +dz, +dx,
        +dqk, +dscale, +dADT, +dBn, +dCn, +dDT, +dtrap, +ddd_dt, +ddd_A,
        +dBraw, +dCraw, +dang_raw}:
      tp2 : TP = TP{dW_in, ddt_bias, dB_bias, dC_bias,
        dB_norm_w, dC_norm_w, dD, dW_out, du, dWh}
      T.Bwd{p, du, match_tp_add(tp2, g)}

def step(op: T.Opt, p: TP, g: TP,
    st: T.OptState<TP>) -> T.Step<TP>:
  match op p g st:
    case T.SGD{lr} TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw,
        +D, +w_out, +E, +Wh} _ _:
      T.Step{tp_sgd(p, g, lr), tp_zeros_grad(), st}
    case T.Momentum{lr, mu} _ _ _:
      T.no_step(TP, T.Momentum{lr, mu}, p, g, st)
    case T.Adam{lr, b1, b2, eps, wd, dec, stp, bc2, c1, c2} _ _ _:
      T.no_step(TP, T.Adam{lr, b1, b2, eps, wd, dec, stp, bc2, c1, c2}, p, g, st)

def li_fill(+left: Nat, cur: Array<F32> & F32, +off: U32,
    acc: List<&2, F32>) -> Array<F32> & (U32 & List<&2, F32>):
  match left:
    case 0n:
      match cur:
        case Tuple{xs, v}:
          Tuple{xs, Tuple{off, List.reverse(&2, F32, Con{v, acc})}}
    case 1n+p:
      match cur:
        case Tuple{xs, v}:
          li_fill(p, Array.get(F32, xs, off), U32.add(off, 1),
            Con{v, acc})

def store_step(+left: Nat, xs: Array<F32>, +off: U32,
    src: List<&2, F32>) -> Array<F32> & U32:
  match left src:
    case 0n _:
      Tuple{xs, off}
    case 1n+p Nil{}:
      Tuple{xs, off}
    case 1n+p Con{v, t}:
      store_step(p, Array.set(F32, xs, off, v),
        U32.add(off, 1), t)

def tp_load11(r: Array<F32> & (U32 & List<&2, F32>),
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +E: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, Wh}:
          T.PA{TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh}, xs, off}

def tp_storeJ(r: Array<F32> & U32) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      T.PA{tp_zeros_grad(), xs, off}

def fwd(p: TP, +x: List<&2, F32>) -> T.Fwd<TP, List<&2, F32>, M.FWD>:
  match p:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}:
      f : M.FWD = M.forward(x, w_in, dt_bias, B_bias, C_bias, Bw, Cw,
        D, w_out, POS())
      fwdfin(bp_of(f), TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D,
        w_out, E, Wh})

def bwdfin(s2: M.FWD,
    +y: List<&2, F32>, +Wh: List<&2, F32>,
    +w_in: List<&2, F32>, +w_out: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +dt_bias: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +D: List<&2, F32>, +E: List<&2, F32>,
    p: TP, dy: List<&2, F32>, g: TP) -> T.Bwd<TP, List<&2, F32>>:
  +dWh = M.dWh_go(POS(), 0n, y, Wh, M.zline(512n))
  bwd_fin(M.backward(s2, E, dy, w_in, w_out, Bw, Cw,
    dt_bias, B_bias, C_bias, D, POS()),
    p, dWh, g)

def tp_load10(r: Array<F32> & (U32 & List<&2, F32>),
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, E}:
          tp_load11(li_fill(511n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}),
            w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E)

def tp_storeI(r: Array<F32> & U32,
    +Wh: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      tp_storeJ(store_step(512n, xs, off, Wh))

def bwdh(r: M.FWD & List<&2, F32>,
    +w_in: List<&2, F32>, +w_out: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +dt_bias: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +D: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>,
    p: TP, dy: List<&2, F32>, g: TP) -> T.Bwd<TP, List<&2, F32>>:
  match r:
    case Tuple{s2, y}:
      bwdfin(s2, y, Wh, w_in, w_out, Bw, Cw, dt_bias, B_bias, C_bias,
        D, E, p, dy, g)

def tp_load9(r: Array<F32> & (U32 & List<&2, F32>),
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>, +D: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, w_out}:
          tp_load10(li_fill(511n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}),
            w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out)

def tp_storeH(r: Array<F32> & U32,
    +E: List<&2, F32>,
    +Wh: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      tp_storeI(store_step(512n, xs, off, E), Wh)

def bwd(p: TP, s: M.FWD, dy: List<&2, F32>,
    g: TP) -> T.Bwd<TP, List<&2, F32>>:
  match p:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}:
      bwdh(bp_of(s), w_in, w_out, Bw, Cw, dt_bias, B_bias, C_bias, D,
        E, Wh, TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out,
          E, Wh}, dy, g)

def tp_load8(r: Array<F32> & (U32 & List<&2, F32>),
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, D}:
          tp_load9(li_fill(8191n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}),
            w_in, dt_bias, B_bias, C_bias, Bw, Cw, D)

def tp_storeG(r: Array<F32> & U32,
    +w_out: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      tp_storeH(store_step(8192n, xs, off, w_out), E, Wh)

def tp_load7(r: Array<F32> & (U32 & List<&2, F32>),
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, Cw}:
          tp_load8(li_fill(3n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}),
            w_in, dt_bias, B_bias, C_bias, Bw, Cw)

def tp_storeF(r: Array<F32> & U32,
    +D: List<&2, F32>,
    +w_out: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      tp_storeG(store_step(4n, xs, off, D), w_out, E, Wh)

def tp_load6(r: Array<F32> & (U32 & List<&2, F32>),
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, Bw}:
          tp_load7(li_fill(15n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}),
            w_in, dt_bias, B_bias, C_bias, Bw)

def tp_storeE(r: Array<F32> & U32,
    +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      tp_storeF(store_step(16n, xs, off, Cw), D, w_out, E, Wh)

def tp_load5(r: Array<F32> & (U32 & List<&2, F32>),
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, C_bias}:
          tp_load6(li_fill(15n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}),
            w_in, dt_bias, B_bias, C_bias)

def tp_storeD(r: Array<F32> & U32,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      tp_storeE(store_step(16n, xs, off, Bw), Cw, D, w_out, E, Wh)

def tp_load4(r: Array<F32> & (U32 & List<&2, F32>),
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, B_bias}:
          tp_load5(li_fill(63n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}), w_in, dt_bias, B_bias)

def tp_storeC(r: Array<F32> & U32,
    +C_bias: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      tp_storeD(store_step(64n, xs, off, C_bias), Bw,
        Cw, D, w_out, E, Wh)

def tp_load3(r: Array<F32> & (U32 & List<&2, F32>),
    +w_in: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, dt_bias}:
          tp_load4(li_fill(63n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}), w_in, dt_bias)

def tp_storeB(r: Array<F32> & U32,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      tp_storeC(store_step(64n, xs, off, B_bias),
        C_bias, Bw, Cw, D, w_out, E, Wh)

def tp_load2(r: Array<F32> & (U32 & List<&2, F32>)) -> T.PA<TP>:
  match r:
    case Tuple{xs, rNested}:
      match rNested:
        case Tuple{+off, w_in}:
          tp_load3(li_fill(3n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}), w_in)

def tp_storeA(r: Array<F32> & U32,
    +dt_bias: List<&2, F32>, +B_bias: List<&2, F32>,
    +C_bias: List<&2, F32>, +Bw: List<&2, F32>,
    +Cw: List<&2, F32>, +D: List<&2, F32>,
    +w_out: List<&2, F32>, +E: List<&2, F32>,
    +Wh: List<&2, F32>) -> T.PA<TP>:
  match r:
    case Tuple{xs, off}:
      tp_storeB(store_step(4n, xs, off, dt_bias), B_bias,
        C_bias, Bw, Cw, D, w_out, E, Wh)

def tp_load(xs: Array<F32>, +off: U32) -> T.PA<TP>:
  tp_load2(li_fill(19455n, Array.get(F32, xs, off), U32.add(off, 1), Nil{}))

def tp_store(p: TP, xs: Array<F32>, +off: U32) -> T.PA<TP>:
  match p:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}:
      tp_storeA(store_step(19456n, xs, off, w_in), dt_bias, B_bias,
        C_bias, Bw, Cw, D, w_out, E, Wh)

def Lay() -> T.Layer<TP, List<&2, F32>, List<&2, F32>, M.FWD>:
  T.Layer{p => x => fwd(p, x), p => s => dy => g => bwd(p, s, dy, g),
    op => p => g => st => step(op, p, g, st),
    u => tp_zeros(u), p => tp_pl(p),
    p => xs => off => tp_load(xs, off),
    p => xs => off => tp_store(p, xs, off),
    u => tp_count(u)}

def nats_to(+left: Nat, +i: Nat, acc: List<&2, Nat>) -> List<&2, Nat>:
  match left:
    case 0n:
      acc
    case 1n+p:
      nats_to(p, (i + 1n : Nat), List.append(&2, Nat, acc, Con{i, Nil{}}))

def train_full(+left: Nat,
    lay: T.Layer<TP, List<&2, F32>, List<&2, F32>, M.FWD>,
    tr: T.Train<TP>, +x: List<&2, F32>, +wh: List<&2, F32>) -> T.Train<TP>:
  match left:
    case 0n:
      tr
    case 1n+p:
      train_full(p, lay,
        T.Train.step(~TP, ~List<&2, F32>, ~List<&2, F32>, ~M.FWD, ~Tgt,
          ~Lay, ~loss, tr, x, Tgt{nats_to(8n, 0n, Nil{}), wh}),
        x, wh)

def mtp_of(p: TP) -> M.TP:
  match p:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}:
      M.TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh}

def dir() -> String:
  "reference/mamba3_data"

def build_tp(wi: TA.Mat<304n, 64n>, dtb: TA.Mat<1n, 4n>,
    bb: TA.Mat<4n, 16n>, cb: TA.Mat<4n, 16n>,
    bw: TA.Mat<1n, 16n>, cw: TA.Mat<1n, 16n>,
    dd: TA.Mat<1n, 4n>, wo: TA.Mat<64n, 128n>,
    e0: TA.Mat<8n, 64n>, wh: TA.Mat<8n, 64n>) -> TP:
  TP{M.ml304(TA.Mat.to_list(304n, 64n, wi)),
    M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
    M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
    M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
    M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
    M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
    M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
    M.ml64x128(TA.Mat.to_list(64n, 128n, wo)),
    M.ml64(TA.Mat.to_list(8n, 64n, e0)),
    M.ml64(TA.Mat.to_list(8n, 64n, wh))}

def mtp_build(wi: TA.Mat<304n, 64n>, dtb: TA.Mat<1n, 4n>,
    bb: TA.Mat<4n, 16n>, cb: TA.Mat<4n, 16n>,
    bw: TA.Mat<1n, 16n>, cw: TA.Mat<1n, 16n>,
    dd: TA.Mat<1n, 4n>, wo: TA.Mat<64n, 128n>,
    e0: TA.Mat<8n, 64n>, wh: TA.Mat<8n, 64n>) -> M.TP:
  M.TP{M.ml304(TA.Mat.to_list(304n, 64n, wi)),
    M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)),
    M.ml4x16(TA.Mat.to_list(4n, 16n, bb)),
    M.ml4x16(TA.Mat.to_list(4n, 16n, cb)),
    M.ml1x16(TA.Mat.to_list(1n, 16n, bw)),
    M.ml1x16(TA.Mat.to_list(1n, 16n, cw)),
    M.ml1x4(TA.Mat.to_list(1n, 4n, dd)),
    M.ml64x128(TA.Mat.to_list(64n, 128n, wo)),
    M.ml64(TA.Mat.to_list(8n, 64n, e0)),
    M.ml64(TA.Mat.to_list(8n, 64n, wh))}

def do_train(tp: TP, e0: TA.Mat<8n, 64n>, wh: TA.Mat<8n, 64n>) -> T.Train<TP>:
  train_full(10n, Lay(), T.Train.new(~TP, ~List<&2, F32>, ~List<&2, F32>, ~M.FWD, ~Lay, T.CSGD{0.001}, tp),
    M.ml64(TA.Mat.to_list(8n, 64n, e0)),
    M.ml64(TA.Mat.to_list(8n, 64n, wh)))

def show_custom(wi: TA.Mat<304n, 64n>, dtb: TA.Mat<1n, 4n>,
    bb: TA.Mat<4n, 16n>, cb: TA.Mat<4n, 16n>,
    bw: TA.Mat<1n, 16n>, cw: TA.Mat<1n, 16n>,
    dd: TA.Mat<1n, 4n>, wo: TA.Mat<64n, 128n>,
    e0: TA.Mat<8n, 64n>, wh: TA.Mat<8n, 64n>) -> IO(Unit):
  do IO<Unit>:
    M.show_train(M.train10(mtp_build(wi, dtb, bb, cb, bw, cw, dd, wo, e0, wh), 0.001))
    return Unit{}

def mtp2_of(mt: M.TP) -> TP:
  match mt:
    case M.TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}:
      TP{w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh}

def step2_fin(r: M.SL, +lr: F32) -> TP:
  match r:
    case M.SL{loss, p2}:
      mtp2_of(p2)

def step2(p: TP, +lr: F32) -> TP:
  step2_fin(M.step(mtp_of(p), lr), lr)

def step_c(+left: Nat, p: TP, +lr: F32) -> TP:
  match left:
    case 0n:
      p
    case 1n+q:
      step_c(q, step2(p, lr), lr)

def print_head(+tag: String, p: TP) -> IO(Unit):
  match p:
    case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
        +w_out, +E, +Wh}:
      do IO<Unit>:
        M.show_tag(String.append(tag, "_w_in"), take_n_s(8n, w_in))
        M.show_tag(String.append(tag, "_E"), take_n_s(8n, E))
        M.show_tag(String.append(tag, "_Wh"), take_n_s(8n, Wh))
        M.show_tag(String.append(tag, "_w_out"), take_n_s(8n, w_out))
        return Unit{}

def show_final(trf: T.Train<TP>) -> IO(Unit):
  match trf:
    case T.Train{p, g, st, cfg, p1, p2, l}:
      match p:
        case TP{+w_in, +dt_bias, +B_bias, +C_bias, +Bw, +Cw, +D,
            +w_out, +E, +Wh}:
          do IO<Unit>:
            M.show_tag("layer_loss", Con{l, Nil{}})
            M.show_tag("layer_w_in", take_n_s(8n, w_in))
            M.show_tag("layer_E", take_n_s(8n, E))
            M.show_tag("layer_Wh", take_n_s(8n, Wh))
            M.show_tag("layer_w_out", take_n_s(8n, w_out))
            M.show_tag("layer_tokens", M.step_tokens(mtp_of(TP{w_in, dt_bias,
              B_bias, C_bias, Bw, Cw, D, w_out, E, Wh})))
            return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    wi : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(dir(), "/in_proj.bin"), 0)
    dtb : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(dir(), "/dt_bias.bin"), 0)
    bb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(dir(), "/B_bias.bin"), 0)
    cb : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(dir(), "/C_bias.bin"), 0)
    bw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(dir(), "/B_norm_w.bin"), 0)
    cw : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(dir(), "/C_norm_w.bin"), 0)
    dd : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(dir(), "/D.bin"), 0)
    wo : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(dir(), "/out_proj.bin"), 0)
    e0 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(dir(), "/E0.bin"), 0)
    wh : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(dir(), "/WhT0.bin"), 0)
    show_custom(wi, dtb, bb, cb, bw, cw, dd, wo, e0, wh)
    wi3 : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(dir(), "/in_proj.bin"), 0)
    dtb3 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(dir(), "/dt_bias.bin"), 0)
    bb3 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(dir(), "/B_bias.bin"), 0)
    cb3 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(dir(), "/C_bias.bin"), 0)
    bw3 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(dir(), "/B_norm_w.bin"), 0)
    cw3 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(dir(), "/C_norm_w.bin"), 0)
    dd3 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(dir(), "/D.bin"), 0)
    wo3 : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(dir(), "/out_proj.bin"), 0)
    e03 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(dir(), "/E0.bin"), 0)
    wh3 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(dir(), "/WhT0.bin"), 0)
    cp : TP <- IO.pure(TP, step_c(10n, build_tp(wi3, dtb3, bb3, cb3, bw3, cw3, dd3, wo3, e03, wh3), 0.001))
    print_head("custom", cp)
    wi2 : TA.Mat<304n, 64n> <- T.Mat.load(304n, 64n, String.append(dir(), "/in_proj.bin"), 0)
    dtb2 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(dir(), "/dt_bias.bin"), 0)
    bb2 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(dir(), "/B_bias.bin"), 0)
    cb2 : TA.Mat<4n, 16n> <- T.Mat.load(4n, 16n, String.append(dir(), "/C_bias.bin"), 0)
    bw2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(dir(), "/B_norm_w.bin"), 0)
    cw2 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(dir(), "/C_norm_w.bin"), 0)
    dd2 : TA.Mat<1n, 4n> <- T.Mat.load(1n, 4n, String.append(dir(), "/D.bin"), 0)
    wo2 : TA.Mat<64n, 128n> <- T.Mat.load(64n, 128n, String.append(dir(), "/out_proj.bin"), 0)
    e02 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(dir(), "/E0.bin"), 0)
    wh2 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(dir(), "/WhT0.bin"), 0)
    e03 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(dir(), "/E0.bin"), 0)
    wh3 : TA.Mat<8n, 64n> <- T.Mat.load(8n, 64n, String.append(dir(), "/WhT0.bin"), 0)
    tp : TP <- IO.pure(TP, build_tp(wi2, dtb2, bb2, cb2, bw2, cw2, dd2, wo2, e02, wh2))
    trf : T.Train<TP> <- IO.pure(T.Train<TP>, do_train(tp, e03, wh3))
    show_final(trf)
    return Unit{}
