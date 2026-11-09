# Greedy generation for the dense baseline's dev/confirm evaluation.
#
#   BEND_MEM=4G scripts/bend-guard.sh bend -o .scratch/gen_a examples/language/glm53_flash/gen_a.bend
#   .scratch/gen_a --split dev --weights results/glm53-dense/model-400 --data examples/language/glm53_flash/data
#
# The model is `model_eval.bend`: the same Stage A stack at 192 positions.
# Each prompt is encoded with bos=True and padded to 192 positions with the
# pad id; the causal forward at the last prompt row gives the next token.  The
# reference masks pad, bos and sep before the argmax; this does the same by
# skipping indices 0, 1 and 3.  Exactly 48 tokens are produced; the Python
# driver applies the reference's stop rule (eos or a parseable completion),
# which does not change any earlier token.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_eval.bend as E
import ./weights_a.bend as W

def POS() -> Nat:
  192n

def NEW() -> Nat:
  48n

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def nth(+n: Nat, xs: List<&2, Nat>) -> Nat:
  match n xs:
    case 0n Con{h, t}:
      h
    case 1n+p Con{h, t}:
      nth(p, t)
    case _ _:
      0n

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{+h, t}:
      String.append(Nat.show(h), String.append(" ", show_nats(t)))

def skip(+i: Nat) -> Bool:
  Bool.or(Bool.or(Nat.is_eq(i, 0n), Nat.is_eq(i, 1n)), Nat.is_eq(i, 3n))

def argmax_go(xs: List<&2, F32>, +i: Nat, +best_i: Nat, +best_v: F32) -> Nat:
  match xs:
    case Nil{}:
      best_i
    case Con{+h, t}:
      +score = Bool.pick(F32, skip(i), (0.0 - 1.0e30 : F32), h)
      +gt = F32.is_gt(score, best_v)
      argmax_go(t, (i + 1n : Nat), Bool.pick(Nat, gt, i, best_i), Bool.pick(F32, gt, score, best_v))

def flat_ml(ml: TA.ML<POS(), 260n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def argmax_row(y: E.Y(1n), +row: Nat) -> Nat:
  +xs = flat_ml(TA.Mat.to_list(POS(), 260n, y))
  +rest = List.drop(&2, F32, xs, Nat.mul(row, 260n))
  +one = List.take(&2, F32, rest, 260n)
  argmax_go(one, 0n, 4n, (0.0 - 1.0e30 : F32))

def row_at(+path: String, +i: Nat) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at(path, U32.from_nat(Nat.mul(i, POS())), U32.from_nat(POS()))
    return nats(bs)

# ---- one episode: 48 steps, each forward on the whole 192-position row ----

type Step is Type:
  Step{fw: T.Fwd<E.P(), E.Y(1n), E.S(1n)>, ids: List<&2, Nat>, tok: Nat}

def fw_params(fw: T.Fwd<E.P(), E.Y(1n), E.S(1n)>) -> E.P():
  match fw:
    case T.Fwd{pp, y, s}:
      pp

def next_step(fw: T.Fwd<E.P(), E.Y(1n), E.S(1n)>, +ids: List<&2, Nat>, +pos: Nat) -> Step:
  match fw:
    case T.Fwd{pp, y, s}:
      +tok = argmax_row(y, Nat.sub(pos, 1n))
      Step{T.Layer.fwd(E.P(), E.X(), E.Y(1n), E.S(1n), E.Model(~1n), pp,
        List.set(&2, Nat, ids, pos, tok)), List.set(&2, Nat, ids, pos, tok), tok}

def add_tok(first: Bool, +tok: Nat, acc: List<&2, Nat>) -> List<&2, Nat>:
  match first:
    case True{}:
      acc
    case False{}:
      Con{tok, acc}

def gen_step(+left: Nat, +pos: Nat, +len: Nat, +ep: Nat, +acc: List<&2, Nat>,
    step: Step) -> IO(E.P()):
  match left step:
    case 0n Step{fw, ids, +tok}:
      do IO<E.P()>:
        IO.print(String.append("episode: ", String.append(Nat.show(ep),
          String.append(" ", show_nats(List.reverse(&2, Nat, add_tok(Nat.is_eq(pos, len), tok, acc)))))))
        return fw_params(fw)
    case 1n+rest Step{fw, ids, +tok}:
      do IO<E.P()>:
        st : Step <- IO.pure(Step, next_step(fw, ids, pos))
        gen_step(rest, (pos + 1n : Nat), len, ep, add_tok(Nat.is_eq(pos, len), tok, acc), st)

def gen_ep(p: E.P(), +ids: List<&2, Nat>, +len: Nat, +ep: Nat) -> IO(E.P()):
  do IO<E.P()>:
    fw : T.Fwd<E.P(), E.Y(1n), E.S(1n)> <- IO.pure(T.Fwd<E.P(), E.Y(1n), E.S(1n)>,
      T.Layer.fwd(E.P(), E.X(), E.Y(1n), E.S(1n), E.Model(~1n), p, ids))
    gen_step(NEW(), len, len, ep, Nil{}, Step{fw, ids, 0n})

def episodes(+left: Nat, +ep: Nat, +path: String, +lengths: List<&2, Nat>, p: E.P()) -> IO(Unit):
  match left:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+rest:
      do IO<Unit>:
        ids : List<&2, Nat> <- row_at(path, ep)
        p1 : E.P() <- gen_ep(p, ids, nth(ep, lengths), ep)
        episodes(rest, (ep + 1n : Nat), path, lengths, p1)

# ---- arguments ----

type Run is Type:
  Run{split: String, data: String, weights: String}

def run_default() -> Run:
  Run{"dev", "examples/language/glm53_flash/data", "results/glm53-dense/model-400"}

def run_go(args: List<String>, r: Run) -> Run:
  match args:
    case Nil{}:
      r
    case Con{a, rest}:
      match a:
        case "--split":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{split, data, weights}:
                  run_go(more, Run{b, data, weights})
        case "--data":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{split, data, weights}:
                  run_go(more, Run{split, b, weights})
        case "--weights":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{split, data, weights}:
                  run_go(more, Run{split, data, b})
        case _:
          run_go(rest, r)

def run_of(args: List<String>) -> Run:
  match args:
    case Nil{}:
      run_default()
    case Con{program, rest}:
      run_go(rest, run_default())

def run_io(r: Run) -> IO(Unit):
  match r:
    case Run{+split, +data, weights}:
      do IO<Unit>:
        p : E.P() <- W.load_params(weights)
        lengths : List<&2, U32> <- T.read_at(
          String.append(data, String.append("/eval-", String.append(split, "-len.bin"))), 0, 64)
        episodes(64n, 0n,
          String.append(data, String.append("/eval-", String.append(split, ".bin"))),
          nats(lengths), p)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    run_io(run_of(args))
