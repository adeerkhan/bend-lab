# The dense Stage A baseline: 74,144 parameters trained on the reference's
# synthetic Python corpus.
#
#   BEND_MEM=4G scripts/bend-guard.sh bend -o .scratch/train_a examples/language/glm53_flash/train_a.bend
#   .scratch/train_a --steps 400 --data examples/language/glm53_flash/data --out results/glm53-dense
#
# The model is `model_a.bend`: embedding, four dense-causal blocks (RMSNorm,
# RoPE, four-head attention, residual; RMSNorm, SwiGLU 128, residual), a final
# RMSNorm and the tied head.  AdamW is BendTorch's `CAdamW`; the loss is the
# reference's cross-entropy with the ignored-label mask.  The optimizer and
# the data are the same as `reference/glm53_dense_twin.py`:
#
#   lr 3e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.1
#   two fixed rows of 64 next-byte inputs from the synthetic corpus
#
# `--steps 20` is the gate: every step's loss and the final weights and
# gradients are compared with the PyTorch twin.  `--steps 400` is the dense
# pretrain; it writes `final_weights.bin` and `final_grads.bin` in the same
# file layout as the weights it loaded.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_a.bend as M
import ./weights_a.bend as W
import ./loss_masked.bend as L

def B() -> Nat:
  2n

def POS() -> Nat:
  64n

def ROWS() -> Nat:
  Nat.mul(B(), POS())

def VOCAB() -> Nat:
  260n

def IGNORE() -> Nat:
  260n

def STEPS() -> Nat:
  400n

def cfg() -> T.OptCfg:
  T.CAdamW{0.0003, 0.9, 0.95, 0.00000001, 0.1, 0.1, 0.05}

def ce(z: M.Y(B()), labels: List<&2, Nat>) -> T.LossOut<M.Y(B())>:
  L.CrossEntropyMasked(ROWS(), VOCAB(), z, labels, IGNORE())

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

# ---- arguments ----

type Run is Type:
  Run{steps: Nat, data: String, out: String}

def run_default() -> Run:
  Run{STEPS(), "examples/language/glm53_flash/data", "/tmp/glm53-dense"}

def nat_or(m: Maybe<&2, Nat>, d: Nat) -> Nat:
  match m:
    case Some{x}:
      x
    case None{}:
      d

def run_go(args: List<String>, r: Run) -> Run:
  match args:
    case Nil{}:
      r
    case Con{a, rest}:
      match a:
        case "--steps":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{steps, data, out}:
                  run_go(more, Run{nat_or(Nat.read(b), steps), data, out})
        case "--data":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{steps, data, out}:
                  run_go(more, Run{steps, b, out})
        case "--out":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{steps, data, out}:
                  run_go(more, Run{steps, data, b})
        case _:
          run_go(rest, r)

def run_of(args: List<String>) -> Run:
  match args:
    case Nil{}:
      run_default()
    case Con{program, rest}:
      run_go(rest, run_default())

# ---- the loop: the carried loss is printed after the step that produced it ----

def print_or(show: Bool, +l: F32) -> IO(Unit):
  match show:
    case True{}:
      IO.print(String.append("loss: ", F32.show(l)))
    case False{}:
      IO.pure(Unit, Unit{})

def step(tr: T.Train<M.P()>, +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> T.Train<M.P()>:
  T.Train.step(~M.P(), ~M.X(), ~M.Y(B()), ~M.S(B()), ~List<&2, Nat>, ~M.Model(~B()),
    ~ce, tr, ids, labels)

def train_all(+left: Nat, +i: Nat, +ids: List<&2, Nat>, +labels: List<&2, Nat>,
    tr: T.Train<M.P()>) -> IO(T.Train<M.P()>):
  match left tr:
    case 0n T.Train{p, g, st, c, p1, p2, +l}:
      do IO<T.Train<M.P()>>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        return T.Train{p, g, st, c, p1, p2, l}
    case 1n+rest T.Train{p, g, st, c, p1, p2, +l}:
      do IO<T.Train<M.P()>>:
        Unit <- print_or(Nat.is_gt(i, 0n), l)
        tr1 : T.Train<M.P()> <- IO.pure(T.Train<M.P()>,
          step(T.Train{p, g, st, c, p1, p2, l}, ids, labels))
        train_all(rest, (i + 1n : Nat), ids, labels, tr1)

# ---- the final weights and the gradients at the final weights ----

type FG is Type:
  FG{loss: F32, g: M.P()}

def fg_bwd(bw: T.Bwd<M.P(), M.X()>, +loss: F32) -> FG:
  match bw:
    case T.Bwd{_, dx, gp}:
      FG{loss, gp}

def fg_loss(lo: T.LossOut<M.Y(B())>, pp: M.P(), s: M.S(B())) -> FG:
  match lo:
    case T.LossOut{loss, dy}:
      fg_bwd(T.Layer.bwd(M.P(), M.X(), M.Y(B()), M.S(B()), M.Model(~B()), pp, s, dy,
        T.Layer.zeros(M.P(), M.X(), M.Y(B()), M.S(B()), M.Model(~B()))), loss)

def fg_fwd(fw: T.Fwd<M.P(), M.Y(B()), M.S(B())>, +labels: List<&2, Nat>) -> FG:
  match fw:
    case T.Fwd{pp, y, s}:
      fg_loss(ce(y, labels), pp, s)

def final_fg(p: M.P(), +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> FG:
  fg_fwd(T.Layer.fwd(M.P(), M.X(), M.Y(B()), M.S(B()), M.Model(~B()), p, ids), labels)

def save_grads(fg: FG, +out: String) -> IO(Unit):
  match fg:
    case FG{+loss, g}:
      do IO<Unit>:
        IO.print(String.append("grad_loss: ", F32.show(loss)))
        g1 : M.P() <- T.Params.save(~M.P(), ~M.X(), ~M.Y(B()), ~M.S(B()), ~M.Model(~B()),
          String.append(out, "/final_grads.bin"), g)
        return Unit{}

def save_final(tr: T.Train<M.P()>, +out: String, +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, c, p1, p2, +l}:
      do IO<Unit>:
        p1v : M.P() <- T.Params.save(~M.P(), ~M.X(), ~M.Y(B()), ~M.S(B()), ~M.Model(~B()),
          String.append(out, "/final_weights.bin"), p)
        fg : FG <- IO.pure(FG, final_fg(p1v, ids, labels))
        save_grads(fg, out)

def run_loaded(+steps: Nat, +out: String, p: M.P(), +idsn: List<&2, Nat>,
    +labelsn: List<&2, Nat>) -> IO(Unit):
  do IO<Unit>:
    tr : T.Train<M.P()> <- IO.pure(T.Train<M.P()>,
      T.Train.new(~M.P(), ~M.X(), ~M.Y(B()), ~M.S(B()), ~M.Model(~B()), cfg(), p))
    tr1 : T.Train<M.P()> <- train_all(steps, 0n, idsn, labelsn, tr)
    save_final(tr1, out, idsn, labelsn)

def run_io(+steps: Nat, +data: String, +out: String) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    IO.print(String.append("params: ",
      U32.show(T.Layer.count(M.P(), M.X(), M.Y(B()), M.S(B()), M.Model(~B())))))
    IO.print(String.append("steps: ", Nat.show(steps)))
    p : M.P() <- W.load_params(data)
    ids : List<&2, U32> <- T.read_at(String.append(data, "/train_batch.bin"), 0, 128)
    labels : List<&2, U32> <- T.read_at(String.append(data, "/train_labels.bin"), 0, 128)
    run_loaded(steps, out, p, nats(ids), nats(labels))

def run_main(r: Run) -> IO(Unit):
  match r:
    case Run{steps, data, out}:
      run_io(steps, data, out)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    run_main(run_of(args))
