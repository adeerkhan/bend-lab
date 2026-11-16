# Thirty-step state round-trip (B2).
#
# The same 30-step greedy decode twice:
#   full:   30 steps from a fresh state
#   split:  10 steps from a fresh state, then 20 steps resumed from the
#           state the first leg returned (t continues at 10, not 0)
# The two trajectories must be identical, token for token, and the
# first 16 tokens must equal the existing 16-token golden
# (reference/mamba3_golden.npz, gen_tokens).
#
# Only the driver recursion and the token-append rule live here; the
# decode itself (`S.layer_row`) and the prompt/gen semantics of the
# existing step are shared, not copied.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M
import ./step_mamba3.bend as S

type PS is Type:
  PS{st: S.ST, out: List<&2, Nat>, last_in: Nat}

def probe_append(+t: Nat, +nxt: Nat,
    out: List<&2, Nat>) -> List<&2, Nat>:
  match t:
    case 0n:
      out
    case 1n:
      out
    case 2n:
      out
    case _:
      List.append(&2, Nat, out, Con{nxt, Nil{}})

def probe_finish(l2o: S.LOUT, s1: S.ST2, out: List<&2, Nat>, +t: Nat,
    +Wh: List<&2, F32>) -> PS:
  match l2o:
    case S.LOUT{s2, y2}:
      +logits = M.matvec(Wh, y2, 64n)
      +nxt = M.argmax(logits, 8n)
      PS{S.ST{s1, s2}, probe_append(t, nxt, out), S.next_in(t, nxt)}

def probe_layer1done(l1o: S.LOUT, s2: S.ST2, out: List<&2, Nat>, +t: Nat,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +Wh: List<&2, F32>) -> PS:
  match l1o:
    case S.LOUT{s1, y1}:
      probe_finish(S.layer_row(s2, t, y1, w_in, dt_bias, B_bias, C_bias,
        Bw, Cw, D, w_out), s1, out, t, Wh)

def probe_run(st: S.ST, +t: Nat, +tok: Nat, out: List<&2, Nat>,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>) -> PS:
  match st:
    case S.ST{s1, s2}:
      +u = M.take_row(E, Nat.mul(tok, 64n), 64n)
      probe_layer1done(S.layer_row(s1, t, u, w_in, dt_bias, B_bias, C_bias,
        Bw, Cw, D, w_out), s2, out, t, w_in, dt_bias, B_bias, C_bias,
        Bw, Cw, D, w_out, Wh)

def probe_go(+left: Nat, +t: Nat,
    +w_in: List<&2, F32>, +dt_bias: List<&2, F32>,
    +B_bias: List<&2, F32>, +C_bias: List<&2, F32>,
    +Bw: List<&2, F32>, +Cw: List<&2, F32>,
    +D: List<&2, F32>, +w_out: List<&2, F32>,
    +E: List<&2, F32>, +Wh: List<&2, F32>,
    r: PS) -> PS:
  match left r:
    case 0n PS{st, out, last_in}:
      r
    case 1n+p PS{st, out, last_in}:
      +tok = last_in
      probe_go(p, (t + 1n : Nat),
        w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh,
        probe_run(st, t, tok, out,
          w_in, dt_bias, B_bias, C_bias, Bw, Cw, D, w_out, E, Wh))

def ddir() -> String:
  "reference/mamba3_data"

type PS2 is Type:
  PS2{full: PS, split: PS}

def mk_both(+wi_l: List<&2, F32>, +dtb_l: List<&2, F32>,
    +bb_l: List<&2, F32>, +cb_l: List<&2, F32>,
    +bw_l: List<&2, F32>, +cw_l: List<&2, F32>,
    +dd_l: List<&2, F32>, +wo_l: List<&2, F32>,
    +e0_l: List<&2, F32>, +wh_l: List<&2, F32>) -> PS2:
  +prompt : List<&2, Nat> = Con{0n, Con{1n, Con{2n, Con{3n, Nil{}}}}}
  PS2{probe_go(30n, 0n, wi_l, dtb_l, bb_l, cb_l, bw_l, cw_l, dd_l, wo_l, e0_l, wh_l,
        PS{S.fresh(), prompt, 0n}),
      probe_go(20n, 10n, wi_l, dtb_l, bb_l, cb_l, bw_l, cw_l, dd_l, wo_l, e0_l, wh_l,
        probe_go(10n, 0n, wi_l, dtb_l, bb_l, cb_l, bw_l, cw_l, dd_l, wo_l, e0_l, wh_l,
          PS{S.fresh(), prompt, 0n}))}

def show_pair(both: PS2) -> IO(Unit):
  match both:
    case PS2{f, g}:
      match f:
        case PS{_, fout, _}:
          match g:
            case PS{_, sout, _}:
              do IO<Unit>:
                M.show_tag("tokens_full", M.nat_row_f32(31n, fout, Nil{}))
                M.show_tag("tokens_split", M.nat_row_f32(31n, sout, Nil{}))
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
    +wi_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml304(TA.Mat.to_list(304n, 64n, wi)))
    +dtb_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml1x4(TA.Mat.to_list(1n, 4n, dtb)))
    +bb_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml4x16(TA.Mat.to_list(4n, 16n, bb)))
    +cb_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml4x16(TA.Mat.to_list(4n, 16n, cb)))
    +bw_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml1x16(TA.Mat.to_list(1n, 16n, bw)))
    +cw_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml1x16(TA.Mat.to_list(1n, 16n, cw)))
    +dd_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml1x4(TA.Mat.to_list(1n, 4n, dd)))
    +wo_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml64x128(TA.Mat.to_list(64n, 128n, wo)))
    +e0_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml64(TA.Mat.to_list(8n, 64n, e0)))
    +wh_l : List<&2, F32> <- IO.pure(List<&2, F32>, M.ml64(TA.Mat.to_list(8n, 64n, wh)))
    both : PS2 <- IO.pure(PS2, mk_both(wi_l, dtb_l, bb_l, cb_l, bw_l, cw_l, dd_l, wo_l, e0_l, wh_l))
    show_pair(both)
    return Unit{}
