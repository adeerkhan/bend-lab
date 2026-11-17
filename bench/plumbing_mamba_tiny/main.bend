# The plumbing split of one tiny-LM training step.
#
# The two-layer SISO Mamba-3 LM trains 20 steps in about 30 minutes on this
# lane. This program times the same functions the trainer calls, so the run
# page can say where that time goes instead of recording "bounds unavailable".
#
#   init_params  reading w_init_flat.bin and its flat list (once per process)
#   step_in      one example's 128 byte tokens as Nat rows
#   forward      tiny_parts: in_proj/scan/loss on those rows, no backward
#   full_step_x3 three tiny_io_example steps (forward + backward + AdamW)
#                from the same weights — one step is a third
#
# IO.now counts whole milliseconds; every value is printed after its timer so
# the compiler cannot erase an unused pure phase. The stage reader computes
# loss_backward_opt as the rest of the step (three steps minus the measured
# input and forward). There is no Mat conversion in this trainer: weights and
# tokens are flat lists, so that column is honestly zero.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../../examples/language/mamba3/train_mamba3_tiny.bend as M
import ../../examples/language/mamba3/mamba3_siso.bend as S

def show_ms(name: String, total: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(total))))

def show_f32(tag: String, v: F32) -> IO(Unit):
  IO.print(String.append(tag, F32.show(v)))

# tiny_parts returns the whole forward state; the bench only needs its loss
def lp_loss(lp: M.LP) -> F32:
  match lp:
    case M.LP{+w1, +dt1, +bb1, +cb1, +bw1, +cw1, +D1, +wo1,
        +w2, +dt2, +bb2, +cb2, +bw2, +cw2, +D2, +wo2,
        +E, +Wh, +nw, +u, +h1, +h2, +hn, loss, +dWh, +dhn, +hits}:
      loss

def ts_loss(ts: M.TS) -> F32:
  match ts:
    case M.TS{+ps, +ms, +vs, loss, +gn, +hits}:
      loss

# pure wrappers returning F32 (Data): a `do IO` bind cannot hold a Type value
def forward_loss(toks: List<&2, Nat>, ps: List<&2, F32>) -> F32:
  lp_loss(M.tiny_parts(toks, ps))

def one_step_loss(+ps: List<&2, F32>, +toks: List<&2, Nat>, +t: Nat) -> F32:
  ts_loss(M.tiny_io_example(ps, S.zline(M.params_n()),
    S.zline(M.params_n()), toks, t, M.tiny_lr(t)))


def head_or_one(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      h

def ones_row(+left: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ones_row(p, Con{1.0, acc})

# the trainer's three AdamW list walks (m, v, p) over the same 71,952 cells,
# with a constant gradient. The bias correction is left out: this times the
# walks, not the exact optimizer numbers.
def opt_walk(+ps: List<&2, F32>, +grad: List<&2, F32>, +lr: F32) -> F32:
  +m2 = M.m_go(M.params_n(), grad, S.zline(M.params_n()), M.beta1(),
    (1.0 - M.beta1() : F32), Nil{})
  +v2 = M.v_go(M.params_n(), grad, S.zline(M.params_n()), M.beta2(),
    (1.0 - M.beta2() : F32), Nil{})
  +p2 = M.p_go(M.params_n(), ps, m2, v2,
    (1.0 - (lr * M.wd() : F32) : F32), lr, F32.sqrt(M.beta2()), M.eps(), Nil{})
  head_or_one(p2)

def ml_l(ml: TA.ML<71952n, 1n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, l}:
      l

def mpair_a(p: TA.MPair<71952n, 1n>) -> TA.Mat<71952n, 1n>:
  match p:
    case TA.MPair{a, b}:
      a


def grad_w(g: S.GRAD) -> F32:
  match g:
    case S.GRAD{+du, +dW_in, +ddt_bias, +dB_bias, +dC_bias, +dB_norm_w,
        +dC_norm_w, +dD, +dW_out, +do_v, +dq, +dk, +dz, +dx,
        +dqk, +dscale, +dADT, +dBn, +dCn, +dDT, +dtrap, +ddd_dt,
        +ddd_A, +dBraw, +dCraw, +dang_raw}:
      S.sumsq(dW_in)

def layer_back(+w: List<&2, F32>, +dt: List<&2, F32>, +bb: List<&2, F32>,
    +cb: List<&2, F32>, +bw: List<&2, F32>, +cw: List<&2, F32>,
    +D: List<&2, F32>, +wo: List<&2, F32>,
    +u: List<&2, F32>, +dy: List<&2, F32>) -> F32:
  f : S.FWD = S.forward(u, w, dt, bb, cb, bw, cw, D, wo, M.pos_n())
  g : S.GRAD = S.backward(f, u, dy, w, wo, bw, cw, dt, bb, cb, D, M.pos_n())
  grad_w(g)

# one backward pass through each block, with a constant upstream gradient;
# includes one forward per block (subtract the measured forward time below)
def two_backs(+ps: List<&2, F32>, +toks: List<&2, Nat>,
    +dy: List<&2, F32>) -> F32:
  +w1 = S.take_row(ps, M.o_l1_in(), 19456n)
  +dt1 = S.take_row(ps, M.o_l1_dt(), 4n)
  +bb1 = S.take_row(ps, M.o_l1_bb(), 64n)
  +cb1 = S.take_row(ps, M.o_l1_cb(), 64n)
  +bw1 = S.take_row(ps, M.o_l1_bw(), 16n)
  +cw1 = S.take_row(ps, M.o_l1_cw(), 16n)
  +D1 = S.take_row(ps, M.o_l1_d(), 4n)
  +wo1 = S.take_row(ps, M.o_l1_wo(), 8192n)
  +w2 = S.take_row(ps, M.o_l2_in(), 19456n)
  +dt2 = S.take_row(ps, M.o_l2_dt(), 4n)
  +bb2 = S.take_row(ps, M.o_l2_bb(), 64n)
  +cb2 = S.take_row(ps, M.o_l2_cb(), 64n)
  +bw2 = S.take_row(ps, M.o_l2_bw(), 16n)
  +cw2 = S.take_row(ps, M.o_l2_cw(), 16n)
  +D2 = S.take_row(ps, M.o_l2_d(), 4n)
  +wo2 = S.take_row(ps, M.o_l2_wo(), 8192n)
  +E = S.take_row(ps, M.o_e(), 8128n)
  +u = S.embed(toks, E, M.pos_n())
  +g1 = layer_back(w1, dt1, bb1, cb1, bw1, cw1, D1, wo1, u, dy)
  +y1 = S.block_y(S.forward(u, w1, dt1, bb1, cb1, bw1, cw1, D1, wo1, M.pos_n()))
  +h1 = S.add_lists(u, y1)
  +g2 = layer_back(w2, dt2, bb2, cb2, bw2, cw2, D2, wo2, h1, dy)
  (g1 + g2 : F32)

def mat_add_of(+grad: List<&2, F32>) -> F32:
  a : TA.Mat<71952n, 1n> = TA.Mat.from_list(71952n, 1n, grad)
  b : TA.Mat<71952n, 1n> = TA.Mat.from_list(71952n, 1n, grad)
  c : TA.Mat<71952n, 1n> = mpair_a(TA.Mat.add(71952n, 1n, a, b))
  head_or_one(ml_l(TA.Mat.to_list(71952n, 1n, c)))

def main() -> IO(Unit):
  do IO<Unit>:
    wf : TA.Mat<71952n, 1n> <- T.Mat.load(71952n, 1n,
      String.append(M.ddir(), "/w_init_flat.bin"), 0)
    tb : List<&2, U32> <- T.read_at(
      String.append(M.ddir(), "/train_tokens.bin"), 0, 2560)
    +toks_all : List<&2, U32> <- IO.pure(List<&2, U32>, tb)
    Unit <- show_f32("params: ", 71952.0)

    t0 : Nat <- IO.now()
    +ps0 : List<&2, F32> <- IO.pure(List<&2, F32>,
      M.ml_flat(TA.Mat.to_list(71952n, 1n, wf)))
    t1 : Nat <- IO.now()
    Unit <- show_ms("init_params", Nat.sub(t1, t0))

    +t2 : Nat <- IO.now()
    +tk : List<&2, Nat> <- IO.pure(List<&2, Nat>,
      M.bytes_to_nats(M.u32_row(toks_all, 0n, M.pos_n())))
    +t3 : Nat <- IO.now()
    Unit <- show_ms("step_in", Nat.sub(t3, t2))
    Unit <- show_ms("rows", Nat.sub(t3, t2))

    t4 : Nat <- IO.now()
    +floss : F32 <- IO.pure(F32, forward_loss(tk, ps0))
    t5 : Nat <- IO.now()
    Unit <- show_ms("forward", Nat.sub(t5, t4))
    Unit <- show_f32("forward_loss: ", floss)

    t6 : Nat <- IO.now()
    +l1 : F32 <- IO.pure(F32, one_step_loss(ps0, tk, 0n))
    +l2 : F32 <- IO.pure(F32, one_step_loss(ps0, tk, 1n))
    +l3 : F32 <- IO.pure(F32, one_step_loss(ps0, tk, 2n))
    t7 : Nat <- IO.now()
    Unit <- show_ms("full_step_x3", Nat.sub(t7, t6))
    Unit <- show_f32("full_step_loss: ", l3)

    # the optimizer's three list walks alone, three times, on the same weights
    +grad : List<&2, F32> <- IO.pure(List<&2, F32>, ones_row(M.params_n(), Nil{}))
    t8 : Nat <- IO.now()
    +o1 : F32 <- IO.pure(F32, opt_walk(ps0, grad, M.tiny_lr(0n)))
    +o2 : F32 <- IO.pure(F32, opt_walk(ps0, grad, M.tiny_lr(0n)))
    +o3 : F32 <- IO.pure(F32, opt_walk(ps0, grad, M.tiny_lr(0n)))
    t9 : Nat <- IO.now()
    Unit <- show_ms("opt_walk_x3", Nat.sub(t9, t8))
    Unit <- show_f32("opt_walk_head: ", o3)

    # one array-style pass over the same 71,952 cells, for scale
    t10 : Nat <- IO.now()
    +mh : F32 <- IO.pure(F32, mat_add_of(grad))
    t11 : Nat <- IO.now()
    Unit <- show_ms("mat_add", Nat.sub(t11, t10))
    Unit <- show_f32("mat_add_head: ", mh)

    # two block backward passes (each with a forward) on the same weights
    +dy8192 : List<&2, F32> <- IO.pure(List<&2, F32>,
      ones_row(Nat.mul(8192n, 1n), Nil{}))
    t12 : Nat <- IO.now()
    +bk : F32 <- IO.pure(F32, two_backs(ps0, tk, dy8192))
    t13 : Nat <- IO.now()
    Unit <- show_ms("back_x2_with_fwd", Nat.sub(t13, t12))
    Unit <- show_f32("back_head: ", bk)
    return Unit{}
