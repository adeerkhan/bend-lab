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
    return Unit{}
