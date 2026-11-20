# Mamba-3 tiny-LM trainer (B2/B3): the byte corpus, AdamW + clipping
# + a warmup/cosine schedule, on the two-layer SISO LM.
#
# The block, the residual, the final RMSNorm and the head are the gated
# `mamba3_siso.bend` ones (`lm_forward`'s shape, n = 128). What is new here
# is the trainer:
#
#   targets are the next byte within an example (the last position is
#   dropped: 127 targets per example, no leakage across examples)
#   AdamW lr = warmup (5 steps, linear 0 -> 1e-3) then cosine from 1e-3 to
#   1e-4 over the remaining steps, betas (0.9, 0.999), eps 1e-8,
#   weight_decay 0.01, decoupled decay, torch's step order
#   gradient clipping by global norm at 1.0 (clip_grad_norm_'s convention:
#   coef = 1/(norm + 1e-6) when norm > 1.0)
#   one example per step, in file order, batch 1, constant across the run
#   the loss is the mean cross entropy over the example's 127 targets
#
# The weights travel as ONE flat 71,952-value list in the manifest order
# (`.scratch/mamba3_tiny/w_init/manifest.json`), and the AdamW moments are
# flat lists too, so the update never has to round-trip a record. The
# twin (`reference/mamba3_tiny_twin.py`) reads the same dumps and runs
# the same schedule; the gate compares per-step losses, the gradient
# norms, the train hits and the final per-tensor sums of squares.
#
# Held-out evaluation is by the STEP count: `steps_n()` steps, one
# held-out evaluation after the last step (the corpus's 32 held-out
# examples), which is "every N steps" with N = steps. A mid-training
# evaluation every k < steps steps would multiply the wall clock by
# steps/k and is unavailable at this budget; the report card says so.
#
# The full corpus run (64 examples, one pass) is not attempted in this
# session: a 128-position fwd+bwd measures ~S s here (filled in by the
# gate run), so the run uses the first `steps_n()` train examples and
# the whole 4,096-byte held-out split is evaluated at the end.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/mamba3_tiny"

def steps_n() -> Nat:
  20n

# The run configuration: the steps (the warmup/cosine stretches to it), the
# peak learning rate in parts per million, and the data directory. The
# defaults are the gated run; --steps/--lr-ppm/--data change one field each.
type Run is Type:
  Run{steps: Nat, lrppm: Nat, data: String}

def run_default() -> Run:
  Run{steps_n(), 1000n, ddir()}

def nat_or(m: Maybe<&2, Nat>, d: Nat) -> Nat:
  match m:
    case Some{x}:
      x
    case None{}:
      d

def run_go(args: List<String>, r: Run) -> IO(Run):
  match args:
    case Nil{}:
      IO.pure(Run, r)
    case Con{a, rest}:
      match a:
        case "--steps":
          match rest:
            case Nil{}:
              IO.pure(Run, r)
            case Con{b, more}:
              match r:
                case Run{+steps, lrppm, +data}:
                  run_go(more, Run{nat_or(Nat.read(b), steps), lrppm, data})
        case "--lr-ppm":
          match rest:
            case Nil{}:
              IO.pure(Run, r)
            case Con{b, more}:
              match r:
                case Run{+steps, lrppm, +data}:
                  run_go(more, Run{steps, nat_or(Nat.read(b), lrppm), data})
        case "--data":
          match rest:
            case Nil{}:
              IO.pure(Run, r)
            case Con{b, more}:
              match r:
                case Run{+steps, lrppm, +data}:
                  run_go(more, Run{steps, lrppm, b})
        case _:
          IO.die(Run, 1, String.append("unknown flag: ", a))

def run_of(args: List<String>) -> IO(Run):
  match args:
    case Nil{}:
      IO.pure(Run, run_default())
    case Con{program, rest}:
      run_go(rest, run_default())

def pos_n() -> Nat:
  128n

def tgt_n() -> Nat:
  127n

def vocab_n() -> Nat:
  127n

def params_n() -> Nat:
  71952n

# the flat offsets in the manifest order
#   l1: in_proj 0 (304x64), dt_bias 19456 (4), B_bias 19460 (64),
#       C_bias 19524 (64), B_norm_w 19588 (16), C_norm_w 19604 (16),
#       D 19620 (4), out_proj 19624 (64x128 = 8192)
#   l2: 27816 .. 55632
#   E 55632 (127x64 = 8128), Wh 63760, nw 71888 (64)
def o_l1_in() -> Nat:
  0n
def o_l1_dt() -> Nat:
  19456n
def o_l1_bb() -> Nat:
  19460n
def o_l1_cb() -> Nat:
  19524n
def o_l1_bw() -> Nat:
  19588n
def o_l1_cw() -> Nat:
  19604n
def o_l1_d() -> Nat:
  19620n
def o_l1_wo() -> Nat:
  19624n
def o_l2_in() -> Nat:
  27816n
def o_l2_dt() -> Nat:
  47272n
def o_l2_bb() -> Nat:
  47276n
def o_l2_cb() -> Nat:
  47340n
def o_l2_bw() -> Nat:
  47404n
def o_l2_cw() -> Nat:
  47420n
def o_l2_d() -> Nat:
  47436n
def o_l2_wo() -> Nat:
  47440n
def o_e() -> Nat:
  55632n
def o_wh() -> Nat:
  63760n
def o_nw() -> Nat:
  71888n
# (kept close to the manifest; the gate checks the count)

def pw_go(+left: Nat, +b: F32, acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      pw_go(p, b, (acc * b : F32))

def pw(+t: Nat, +b: F32) -> F32:
  pw_go(t, b, 1.0)

def clip_coef(b: Bool, +nrm: F32) -> F32:
  match b:
    case True{}:
      (1.0 / (nrm + 0.000001 : F32) : F32)
    case False{}:
      1.0

def m_go(+left: Nat, +g: List<&2, F32>, +m: List<&2, F32>,
    +b1: F32, +ob1: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match left g m:
    case 0n _ _:
      List.reverse(&2, F32, acc)
    case _ Nil{} _:
      List.reverse(&2, F32, acc)
    case _ _ Nil{}:
      List.reverse(&2, F32, acc)
    case 1n+p Con{gh, gt} Con{mh, mt}:
      m_go(p, gt, mt, b1, ob1,
        Con{((b1 * mh : F32) + (ob1 * gh : F32) : F32), acc})

def v_go(+left: Nat, +g: List<&2, F32>, +v: List<&2, F32>,
    +b2: F32, +ob2: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match left g v:
    case 0n _ _:
      List.reverse(&2, F32, acc)
    case _ Nil{} _:
      List.reverse(&2, F32, acc)
    case _ _ Nil{}:
      List.reverse(&2, F32, acc)
    case 1n+p Con{gh, gt} Con{vh, vt}:
      v_go(p, gt, vt, b2, ob2,
        Con{((b2 * vh : F32) + (ob2 * (gh * gh : F32) : F32) : F32), acc})

def p_go(+left: Nat, +p: List<&2, F32>, +m: List<&2, F32>,
    +v: List<&2, F32>, +dec: F32, +sc: F32, +b2r: F32, +ep: F32,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left p m v:
    case 0n _ _ _:
      List.reverse(&2, F32, acc)
    case _ Nil{} _ _:
      List.reverse(&2, F32, acc)
    case _ _ Nil{} _:
      List.reverse(&2, F32, acc)
    case _ _ _ Nil{}:
      List.reverse(&2, F32, acc)
    case 1n+p Con{ph, pt} Con{mh, mt} Con{vh, vt}:
      +den = ((F32.sqrt(vh) / b2r : F32) + ep : F32)
      p_go(p, pt, mt, vt, dec, sc, b2r, ep,
        Con{(((ph * dec : F32) - ((sc * mh : F32) / den : F32)) : F32), acc})

def peak_of(+lrppm: Nat) -> F32:
  (F32.from_nat(lrppm) / 1000000.0 : F32)

def lr_floor_of(+peakv: F32) -> F32:
  (peakv / 10.0 : F32)

def warmup_n() -> Nat:
  5n

def beta1() -> F32:
  0.9

def beta2() -> F32:
  0.999

def wd() -> F32:
  0.01

def eps() -> F32:
  0.00000001

def tiny_lr_go(+c: Cmp, +s: Nat, +steps: Nat, +peakv: F32) -> F32:
  match c:
    case LT{}:
      (peakv * (F32.from_nat(Nat.add(s, 1n))
        / F32.from_nat(warmup_n()) : F32) : F32)
    case EQ{}:
      peakv
    case GT{}:
      +t = (F32.from_nat(Nat.sub(s, warmup_n()))
        / F32.from_nat(Nat.sub(steps, warmup_n())) : F32)
      (lr_floor_of(peakv)
        + ((peakv - lr_floor_of(peakv) : F32)
          * (0.5 + (0.5 * F32.cos((3.141592653589793 * t : F32)) : F32)
            : F32) : F32) : F32)

def tiny_lr(+s: Nat, +steps: Nat, +peakv: F32) -> F32:
  tiny_lr_go(Nat.cmp(s, warmup_n()), s, steps, peakv)

def inv127() -> F32:
  (1.0 / 127.0 : F32)

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def tail(xs: List<&2, Nat>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      t

def hit_if(b: Bool) -> Nat:
  match b:
    case True{}:
      1n
    case False{}:
      0n

def hits_tgt_go(+left: Nat, +t: Nat, +y: List<&2, F32>,
    +Wh: List<&2, F32>, +targets: List<&2, Nat>,
    +vocab: Nat, acc: Nat) -> Nat:
  match left:
    case 0n:
      acc
    case 1n+p:
      +y_t = M.take_row(y, Nat.mul(t, 64n), 64n)
      +am = M.argmax(M.matvec(Wh, y_t, 64n), vocab)
      hits_tgt_go(p, (t + 1n : Nat), y, Wh, targets, vocab,
        Nat.add(acc, hit_if(Nat.is_eq(am, M.nth_nat(t, targets)))))

# ---- the per-example forward/backward + AdamW update ----

type TS is Type:
  TS{ps: List<&2, F32>, ms: List<&2, F32>, vs: List<&2, F32>,
     loss: F32, gn: F32, hits: Nat}

type LP is Type:
  LP{w1: List<&2, F32>, dt1: List<&2, F32>,
     bb1: List<&2, F32>, cb1: List<&2, F32>,
     bw1: List<&2, F32>, cw1: List<&2, F32>,
     D1: List<&2, F32>, wo1: List<&2, F32>,
     w2: List<&2, F32>, dt2: List<&2, F32>,
     bb2: List<&2, F32>, cb2: List<&2, F32>,
     bw2: List<&2, F32>, cw2: List<&2, F32>,
     D2: List<&2, F32>, wo2: List<&2, F32>,
     E: List<&2, F32>, Wh: List<&2, F32>,
     nw: List<&2, F32>,
     u: List<&2, F32>, h1: List<&2, F32>,
     h2: List<&2, F32>, hn: List<&2, F32>,
     loss: F32, dWh: List<&2, F32>, dhn: List<&2, F32>,
     hits: Nat}

def tiny_parts(+toks: List<&2, Nat>, +ps: List<&2, F32>) -> LP:
  +tgts = tail(toks)
  +w1 = M.take_row(ps, o_l1_in(), 19456n)
  +dt1 = M.take_row(ps, o_l1_dt(), 4n)
  +bb1 = M.take_row(ps, o_l1_bb(), 64n)
  +cb1 = M.take_row(ps, o_l1_cb(), 64n)
  +bw1 = M.take_row(ps, o_l1_bw(), 16n)
  +cw1 = M.take_row(ps, o_l1_cw(), 16n)
  +D1 = M.take_row(ps, o_l1_d(), 4n)
  +wo1 = M.take_row(ps, o_l1_wo(), 8192n)
  +w2 = M.take_row(ps, o_l2_in(), 19456n)
  +dt2 = M.take_row(ps, o_l2_dt(), 4n)
  +bb2 = M.take_row(ps, o_l2_bb(), 64n)
  +cb2 = M.take_row(ps, o_l2_cb(), 64n)
  +bw2 = M.take_row(ps, o_l2_bw(), 16n)
  +cw2 = M.take_row(ps, o_l2_cw(), 16n)
  +D2 = M.take_row(ps, o_l2_d(), 4n)
  +wo2 = M.take_row(ps, o_l2_wo(), 8192n)
  +E = M.take_row(ps, o_e(), 8128n)
  +Wh = M.take_row(ps, o_wh(), 8128n)
  +nw = M.take_row(ps, o_nw(), 64n)
  +u = M.embed(toks, E, pos_n())
  +y1 = M.block_y(M.forward(u, w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
    pos_n()))
  +h1 = M.add_lists(u, y1)
  +y2 = M.block_y(M.forward(h1, w2, dt2, bb2, cb2, bw2, cw2, D2, wo2,
    pos_n()))
  +h2 = M.add_lists(h1, y2)
  +hn = M.rms_rows(h2, nw, pos_n())
  +loss = M.loss_tgt_go(tgt_n(), 0n, hn, Wh, tgts, vocab_n(),
    inv127(), 0.0)
  +dWh = M.dWh_tgt_go(tgt_n(), 0n, hn, Wh, tgts, vocab_n(),
    inv127(), M.zline(8128n))
  +dhn0 = M.dy_all_tgt_go(tgt_n(), 0n, hn, Wh, tgts, vocab_n(),
    inv127(), Nil{})
  +dhn = List.append(&2, F32, dhn0, M.zline(64n))
  +hits = hits_tgt_go(tgt_n(), 0n, hn, Wh, tgts, vocab_n(), 0n)
  LP{w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
     w2, dt2, bb2, cb2, bw2, cw2, D2, wo2,
     E, Wh, nw, u, h1, h2, hn, loss, dWh, dhn, hits}

# after_rbr: the RMSNorm backward is done (dh2, dnw); layer 2 backward next

# the embedding gradient: scatter the per-position input gradients into the
# 127x64 table by token id (sum over every position that used the id)
def add_if(b: Bool, +x: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match b:
    case True{}:
      M.add_lists(acc, x)
    case False{}:
      acc

def emb_grad_row(+left: Nat, +t: Nat, +toks: List<&2, Nat>,
    +due: List<&2, F32>, +id: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      emb_grad_row(p, (t + 1n : Nat), toks, due, id,
        add_if(Nat.is_eq(M.nth_nat(t, toks), id),
          M.take_row(due, Nat.mul(t, 64n), 64n), acc))

def emb_grad_go(+left: Nat, +id: Nat, +toks: List<&2, Nat>,
    +due: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      emb_grad_go(p, (id + 1n : Nat), toks, due,
        M.push_row(acc, emb_grad_row(pos_n(), 0n, toks, due, id,
          M.zline(64n))))

def emb_grad(+toks: List<&2, Nat>, +due: List<&2, F32>) -> List<&2, F32>:
  emb_grad_go(vocab_n(), 0n, toks, due, Nil{})

def fl2(+a: List<&2, F32>, +b: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, a, b)

def after_g1(g1: M.GRAD, +dh2: List<&2, F32>, +dnw: List<&2, F32>,
    +dh1: List<&2, F32>,
    +dW_in2: List<&2, F32>, +ddt2: List<&2, F32>,
    +dB2: List<&2, F32>, +dC2: List<&2, F32>,
    +dBn2: List<&2, F32>, +dCn2: List<&2, F32>,
    +dD2: List<&2, F32>, +dWo2: List<&2, F32>,
    +ps: List<&2, F32>, +ms: List<&2, F32>, +vs: List<&2, F32>,
    +loss: F32, +dWh: List<&2, F32>, +hits: Nat,
    +toks: List<&2, Nat>,
    +step: Nat, +lr_t: F32) -> TS:
  match g1:
    case M.GRAD{du1, dW_in1, ddt1, dB1, dC1, dBn1, dCn1, dD1x, dWo1,
        do_v1, dq1, dk1, dz1, dx1, dqk1, dscale1, dADT1,
        dBnX1, dCnX1, dDT1, dtrap1, ddd_dt1, ddd_A1, dBraw1,
        dCraw1, dang_raw1}:
      +due = M.add_lists(dh1, du1)
      +Egrad = emb_grad(toks, due)
      +fl = fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(fl2(
        dW_in1, ddt1), dB1), dC1), dBn1), dCn1), dD1x), dWo1), dW_in2),
        ddt2), dB2), dC2), dBn2), dCn2), dD2), dWo2), Egrad), dWh), dnw)
      +nrm = F32.sqrt(M.sumsq(fl))
      +coef = clip_coef(F32.is_gt(nrm, 1.0), nrm)
      +gsc = M.scale_list(fl, coef)
      +tn = Nat.add(step, 1n)
      +bc1 = (1.0 - pw(tn, beta1()) : F32)
      +bc2 = (1.0 - pw(tn, beta2()) : F32)
      +m2v = m_go(params_n(), gsc, ms, beta1(),
        (1.0 - beta1() : F32), Nil{})
      +v2v = v_go(params_n(), gsc, vs, beta2(),
        (1.0 - beta2() : F32), Nil{})
      +p2 = p_go(params_n(), ps, m2v, v2v,
        (1.0 - (lr_t * wd() : F32) : F32), (lr_t / bc1 : F32),
        F32.sqrt(bc2), eps(), Nil{})
      TS{p2, m2v, v2v, loss, nrm, hits}

def after_g2(g2: M.GRAD, +dh2: List<&2, F32>, +dnw: List<&2, F32>,
    +w1: List<&2, F32>, +dt1: List<&2, F32>,
    +bb1: List<&2, F32>, +cb1: List<&2, F32>,
    +bw1: List<&2, F32>, +cw1: List<&2, F32>,
    +D1: List<&2, F32>, +wo1: List<&2, F32>,
    +u: List<&2, F32>,
    +ps: List<&2, F32>, +ms: List<&2, F32>, +vs: List<&2, F32>,
    +loss: F32, +dWh: List<&2, F32>, +hits: Nat,
    +toks: List<&2, Nat>,
    +step: Nat, +lr_t: F32) -> TS:
  match g2:
    case M.GRAD{du2, dW_in2, ddt2, dB2, dC2, dBn2, dCn2, dD2, dWo2,
        do_v2, dq2, dk2, dz2, dx2, dqk2, dscale2, dADT2,
        dBnX2, dCnX2, dDT2, dtrap2, ddd_dt2, ddd_A2, dBraw2,
        dCraw2, dang_raw2}:
      +dh1 = M.add_lists(dh2, du2)
      g1 : M.GRAD = M.backward(M.forward(u, w1, dt1, bb1, cb1, bw1,
          cw1, D1, wo1, pos_n()), u, dh1, w1, wo1, bw1, cw1, dt1,
        bb1, cb1, D1, pos_n())
      after_g1(g1, dh2, dnw, dh1,
        dW_in2, ddt2, dB2, dC2, dBn2, dCn2, dD2, dWo2,
        ps, ms, vs, loss, dWh, hits, toks, step, lr_t)

def after_rbr(rbr: M.RBR, +w2: List<&2, F32>, +dt2: List<&2, F32>,
    +bb2: List<&2, F32>, +cb2: List<&2, F32>,
    +bw2: List<&2, F32>, +cw2: List<&2, F32>,
    +D2: List<&2, F32>, +wo2: List<&2, F32>,
    +h1: List<&2, F32>,
    +w1: List<&2, F32>, +dt1: List<&2, F32>,
    +bb1: List<&2, F32>, +cb1: List<&2, F32>,
    +bw1: List<&2, F32>, +cw1: List<&2, F32>,
    +D1: List<&2, F32>, +wo1: List<&2, F32>,
    +u: List<&2, F32>,
    +ps: List<&2, F32>, +ms: List<&2, F32>, +vs: List<&2, F32>,
    +loss: F32, +dWh: List<&2, F32>, +hits: Nat,
    +toks: List<&2, Nat>,
    +step: Nat, +lr_t: F32) -> TS:
  match rbr:
    case M.RBR{+dh2, dnw}:
      g2 : M.GRAD = M.backward(M.forward(h1, w2, dt2, bb2, cb2, bw2,
          cw2, D2, wo2, pos_n()), h1, dh2, w2, wo2, bw2, cw2, dt2,
        bb2, cb2, D2, pos_n())
      after_g2(g2, dh2, dnw, w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
        u, ps, ms, vs, loss, dWh, hits, toks, step, lr_t)

def tiny_bwd(lp: LP, +ms: List<&2, F32>, +vs: List<&2, F32>,
    +ps_unused: List<&2, F32>, +toks: List<&2, Nat>,
    +step: Nat, +lr_t: F32) -> TS:
  match lp:
    case LP{+w1, +dt1, +bb1, +cb1, +bw1, +cw1, +D1, +wo1,
        +w2, +dt2, +bb2, +cb2, +bw2, +cw2, +D2, +wo2,
        +E, +Wh, +nw, +u, +h1, +h2, +hn, loss, dWh, dhn, hits}:
      after_rbr(M.rms_rows_bwd(h2, nw, dhn, pos_n()),
        w2, dt2, bb2, cb2, bw2, cw2, D2, wo2,
        h1, w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
        u, ps_unused, ms, vs, loss, dWh, hits, toks, step, lr_t)

def tiny_io_example(+ps: List<&2, F32>, +ms: List<&2, F32>,
    +vs: List<&2, F32>, +toks: List<&2, Nat>, +step: Nat,
    +lr_t: F32) -> TS:
  tiny_bwd(tiny_parts(toks, ps), ms, vs, ps, toks, step, lr_t)


# ---- the held-out evaluation (after the last step) ----

type EV is Type:
  EV{nats: F32, hits: Nat}

def u32_row(+xs: List<&2, U32>, +off: Nat, +n: Nat) -> List<&2, U32>:
  List.take(&2, U32, List.drop(&2, U32, xs, off), n)

def bytes_to_nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), bytes_to_nats(t)}

def eval_one(+ps: List<&2, F32>, +toks: List<&2, Nat>) -> EV:
  +tgts = tail(toks)
  +w1 = M.take_row(ps, o_l1_in(), 19456n)
  +dt1 = M.take_row(ps, o_l1_dt(), 4n)
  +bb1 = M.take_row(ps, o_l1_bb(), 64n)
  +cb1 = M.take_row(ps, o_l1_cb(), 64n)
  +bw1 = M.take_row(ps, o_l1_bw(), 16n)
  +cw1 = M.take_row(ps, o_l1_cw(), 16n)
  +D1 = M.take_row(ps, o_l1_d(), 4n)
  +wo1 = M.take_row(ps, o_l1_wo(), 8192n)
  +w2 = M.take_row(ps, o_l2_in(), 19456n)
  +dt2 = M.take_row(ps, o_l2_dt(), 4n)
  +bb2 = M.take_row(ps, o_l2_bb(), 64n)
  +cb2 = M.take_row(ps, o_l2_cb(), 64n)
  +bw2 = M.take_row(ps, o_l2_bw(), 16n)
  +cw2 = M.take_row(ps, o_l2_cw(), 16n)
  +D2 = M.take_row(ps, o_l2_d(), 4n)
  +wo2 = M.take_row(ps, o_l2_wo(), 8192n)
  +E = M.take_row(ps, o_e(), 8128n)
  +Wh = M.take_row(ps, o_wh(), 8128n)
  +nw = M.take_row(ps, o_nw(), 64n)
  +u = M.embed(toks, E, pos_n())
  +y1 = M.block_y(M.forward(u, w1, dt1, bb1, cb1, bw1, cw1, D1, wo1,
    pos_n()))
  +h1 = M.add_lists(u, y1)
  +y2 = M.block_y(M.forward(h1, w2, dt2, bb2, cb2, bw2, cw2, D2, wo2,
    pos_n()))
  +h2 = M.add_lists(h1, y2)
  +hn = M.rms_rows(h2, nw, pos_n())
  EV{M.loss_tgt_go(tgt_n(), 0n, hn, Wh, tgts, vocab_n(),
      (1.0 : F32), 0.0),
    hits_tgt_go(tgt_n(), 0n, hn, Wh, tgts, vocab_n(), 0n)}

def eval_loop(+left: Nat, +i: Nat, +nats: F32, +hits: Nat,
    +ps: List<&2, F32>, +toks: List<&2, U32>,
    acc: Maybe<EV>) -> EV:
  match left acc:
    case 0n None{}:
      EV{nats, hits}
    case 0n Some{ev}:
      match ev:
        case EV{n1, h1}:
          EV{(nats + n1 : F32), Nat.add(hits, h1)}
    case 1n+p None{}:
      +tk : List<&2, Nat> = bytes_to_nats(
        u32_row(toks, Nat.mul(i, pos_n()), pos_n()))
      eval_loop(p, (i + 1n : Nat), nats, hits, ps, toks,
        Some{eval_one(ps, tk)})
    case 1n+p Some{ev}:
      match ev:
        case EV{n1, h1}:
          +tk : List<&2, Nat> = bytes_to_nats(
            u32_row(toks, Nat.mul(i, pos_n()), pos_n()))
          eval_loop(p, (i + 1n : Nat), (nats + n1 : F32),
            Nat.add(hits, h1), ps, toks,
            Some{eval_one(ps, tk)})

# ---- the train loop (self-recursive; the carried record rides a Maybe) ----

type TRES is Type:
  TRES{ps: List<&2, F32>, ls: List<&2, F32>, ns: List<&2, F32>,
       hits: Nat}

def train_loop(+left: Nat, +t: Nat, +ps: List<&2, F32>,
    +ms: List<&2, F32>, +vs: List<&2, F32>,
    +ls: List<&2, F32>, +ns: List<&2, F32>,
    +hits: Nat, +toks: List<&2, U32>,
    +steps: Nat, +peakv: F32, acc: Maybe<TS>) -> IO(TRES):
  match left acc:
    case 0n None{}:
      do IO<TRES>:
        return TRES{ps, ls, ns, hits}
    case 0n Some{ts}:
      match ts:
        case TS{p2, m2, v2, +loss, gn, +h}:
          do IO<TRES>:
            IO.print(String.append(String.append(String.append(
              String.append("step ", Nat.show((t - 1n : Nat))), " loss "),
              F32.show(loss)), " final"))
            return TRES{p2,
              List.append(&2, F32, ls, Con{loss, Nil{}}),
              List.append(&2, F32, ns, Con{gn, Nil{}}),
              Nat.add(hits, h)}
    case 1n+p None{}:
      do IO<TRES>:
        +tk : List<&2, Nat> <- IO.pure(List<&2, Nat>, bytes_to_nats(
          u32_row(toks, Nat.mul(t, pos_n()), pos_n())))
        +lr_t : F32 <- IO.pure(F32, tiny_lr(t, steps, peakv))
        train_loop(p, (t + 1n : Nat), ps, ms, vs, ls, ns, hits,
          toks, steps, peakv, Some{tiny_io_example(ps, ms, vs, tk, t, lr_t)})
    case 1n+p Some{ts}:
      match ts:
        case TS{+p2, +m2, +v2, +loss, +gn, +h}:
          do IO<TRES>:
            IO.print(String.append(String.append(String.append(
              String.append("step ", Nat.show((t - 1n : Nat))),
              " loss "), F32.show(loss)), String.append(String.append(
              " gn ", F32.show(gn)), String.append(" hit ",
              Nat.show(h)))))
            +tk : List<&2, Nat> <- IO.pure(List<&2, Nat>,
              bytes_to_nats(
                u32_row(toks, Nat.mul(t, pos_n()), pos_n())))
            +lr_t : F32 <- IO.pure(F32, tiny_lr(t, steps, peakv))
            train_loop(p, (t + 1n : Nat), p2, m2, v2,
              List.append(&2, F32, ls, Con{loss, Nil{}}),
              List.append(&2, F32, ns, Con{gn, Nil{}}),
              Nat.add(hits, h), toks, steps, peakv,
              Some{tiny_io_example(p2, m2, v2, tk, t, lr_t)})

# ---- the report ----

def wr_of(+ps: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32,
    List.append(&2, F32,
      List.append(&2, F32,
        List.append(&2, F32,
          List.append(&2, F32,
            List.append(&2, F32,
              List.append(&2, F32,
                List.append(&2, F32,
                  List.append(&2, F32,
                    List.append(&2, F32,
                      List.append(&2, F32,
                        List.append(&2, F32,
                          List.append(&2, F32,
                            List.append(&2, F32,
                              List.append(&2, F32,
                                List.append(&2, F32,
                                  Con{M.sumsq(M.take_row(ps, o_l1_in(), 19456n)), Nil{}},
                                  Con{M.sumsq(M.take_row(ps, o_l1_dt(), 4n)), Nil{}}),
                                Con{M.sumsq(M.take_row(ps, o_l1_bb(), 64n)), Nil{}}),
                              Con{M.sumsq(M.take_row(ps, o_l1_cb(), 64n)), Nil{}}),
                            Con{M.sumsq(M.take_row(ps, o_l1_bw(), 16n)), Nil{}}),
                          Con{M.sumsq(M.take_row(ps, o_l1_cw(), 16n)), Nil{}}),
                        Con{M.sumsq(M.take_row(ps, o_l1_d(), 4n)), Nil{}}),
                      Con{M.sumsq(M.take_row(ps, o_l1_wo(), 8192n)), Nil{}}),
                    Con{M.sumsq(M.take_row(ps, o_l2_in(), 19456n)), Nil{}}),
                  Con{M.sumsq(M.take_row(ps, o_l2_dt(), 4n)), Nil{}}),
                Con{M.sumsq(M.take_row(ps, o_l2_bb(), 64n)), Nil{}}),
              Con{M.sumsq(M.take_row(ps, o_l2_cb(), 64n)), Nil{}}),
            Con{M.sumsq(M.take_row(ps, o_l2_bw(), 16n)), Nil{}}),
          Con{M.sumsq(M.take_row(ps, o_l2_cw(), 16n)), Nil{}}),
        Con{M.sumsq(M.take_row(ps, o_l2_d(), 4n)), Nil{}}),
      Con{M.sumsq(M.take_row(ps, o_l2_wo(), 8192n)), Nil{}}),
    List.append(&2, F32,
      List.append(&2, F32,
        Con{M.sumsq(M.take_row(ps, o_e(), 8128n)), Nil{}},
        Con{M.sumsq(M.take_row(ps, o_wh(), 8128n)), Nil{}}),
      Con{M.sumsq(M.take_row(ps, o_nw(), 64n)), Nil{}}))

def ml_flat(m: TA.ML<71952n, 1n>) -> List<&2, F32>:
  match m:
    case TA.ML{d, xs}:
      xs

def show_train(t: TRES) -> IO(Unit):
  match t:
    case TRES{+ps, +ls, +ns, hits}:
      do IO<Unit>:
        M.show_tag("tiny_train_losses", ls)
        M.show_tag("tiny_grad_norms", ns)
        M.show_tag("tiny_train_hits",
          Con{F32.from_nat(hits), Nil{}})
        M.show_tag("tiny_w_sumsq", wr_of(ps))
        return Unit{}

def show_ev(ev: EV) -> IO(Unit):
  match ev:
    case EV{+nats, +hits2}:
      +ln2 = F32.log(2.0)
      +den = (F32.from_nat(4064n) : F32)
      do IO<Unit>:
        M.show_tag("tiny_heldout_nats", Con{nats, Nil{}})
        M.show_tag("tiny_heldout_hits",
          Con{F32.from_nat(hits2), Nil{}})
        M.show_tag("tiny_bits_per_byte",
          Con{(((nats / den : F32) / ln2 : F32) : F32), Nil{}})
        M.show_tag("tiny_heldout_acc",
          Con{((F32.from_nat(hits2) / den : F32) : F32), Nil{}})
        return Unit{}

def run_eval(res: TRES, +toks: List<&2, U32>) -> IO(Unit):
  match res:
    case TRES{+ps, ls, ns, hits}:
      do IO<Unit>:
        show_train(TRES{ps, ls, ns, hits})
        ev : EV <- IO.pure(EV,
          eval_loop(32n, 0n, 0.0, 0n, ps, toks, None{}))
        show_ev(ev)
        return Unit{}

def run_io(r: Run) -> IO(Unit):
  match r:
    case Run{+steps, lrppm, +data}:
      do IO<Unit>:
        wf : TA.Mat<71952n, 1n> <- T.Mat.load(71952n, 1n,
          String.append(data, "/w_init_flat.bin"), 0)
        tb : List<&2, U32> <- T.read_at(
          String.append(data, "/train_tokens.bin"), 0,
          U32.from_nat(Nat.mul(steps, pos_n())))
        hb : List<&2, U32> <- T.read_at(
          String.append(data, "/heldout_tokens.bin"), 0, 4096)
        +toks_all : List<&2, U32> <- IO.pure(List<&2, U32>, tb)
        +htoks : List<&2, U32> <- IO.pure(List<&2, U32>, hb)
        +ps0 : List<&2, F32> <- IO.pure(List<&2, F32>,
          ml_flat(TA.Mat.to_list(71952n, 1n, wf)))
        res : TRES <- train_loop(steps, 0n, ps0,
          M.zline(params_n()), M.zline(params_n()),
          Nil{}, Nil{}, 0n, toks_all, steps, peak_of(lrppm), None{})
        run_eval(res, htoks)
        return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    r : Run <- run_of(args)
    run_io(r)
