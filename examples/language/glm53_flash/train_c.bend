# The integrated Stage B/C model: one batch, 64 positions, one head of width
# 8, linear attention, then sparse attention (window 16, stride 16), each with
# a SwiGLU hidden 32 MLP, a final RMSNorm and the tied head.  Twenty AdamW
# steps against the PyTorch twin are the integration gate.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_c.bend as M
import ./weights_c.bend as W
import ./loss_masked.bend as L

def STEPS() -> Nat:
  20n

def IGNORE() -> Nat:
  260n

def cfg() -> T.OptCfg:
  T.CAdamW{0.0003, 0.9, 0.95, 0.00000001, 0.1, 0.1, 0.05}

def ce(z: M.Y(), labels: List<&2, Nat>) -> T.LossOut<M.Y()>:
  L.CrossEntropyMasked(64n, 260n, z, labels, IGNORE())

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def step(tr: T.Train<M.P()>, +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> T.Train<M.P()>:
  T.Train.step(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~List<&2, Nat>, ~M.Model(), ~ce, tr, ids, labels)

def print_or(show: Bool, +l: F32) -> IO(Unit):
  match show:
    case True{}:
      IO.print(String.append("loss: ", F32.show(l)))
    case False{}:
      IO.pure(Unit, Unit{})

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

type FG is Type:
  FG{loss: F32, g: M.P()}

def fg_bwd(bw: T.Bwd<M.P(), M.X()>, +loss: F32) -> FG:
  match bw:
    case T.Bwd{_, dx, gp}:
      FG{loss, gp}

def fg_loss(lo: T.LossOut<M.Y()>, pp: M.P(), s: M.S()) -> FG:
  match lo:
    case T.LossOut{loss, dy}:
      fg_bwd(T.Layer.bwd(M.P(), M.X(), M.Y(), M.S(), M.Model(), pp, s, dy,
        T.Layer.zeros(M.P(), M.X(), M.Y(), M.S(), M.Model())), loss)

def fg_fwd(fw: T.Fwd<M.P(), M.Y(), M.S()>, +labels: List<&2, Nat>) -> FG:
  match fw:
    case T.Fwd{pp, y, s}:
      fg_loss(ce(y, labels), pp, s)

def final_fg(p: M.P(), +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> FG:
  fg_fwd(T.Layer.fwd(M.P(), M.X(), M.Y(), M.S(), M.Model(), p, ids), labels)

def save_grads(fg: FG, +out: String) -> IO(Unit):
  match fg:
    case FG{+loss, g}:
      do IO<Unit>:
        IO.print(String.append("grad_loss: ", F32.show(loss)))
        g1 : M.P() <- T.Params.save(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~M.Model(),
          String.append(out, "/final_grads.bin"), g)
        return Unit{}

def save_final(tr: T.Train<M.P()>, +out: String, +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, c, p1, p2, +l}:
      do IO<Unit>:
        p1v : M.P() <- T.Params.save(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~M.Model(),
          String.append(out, "/final_weights.bin"), p)
        fg : FG <- IO.pure(FG, final_fg(p1v, ids, labels))
        save_grads(fg, out)

def run_loaded(+steps: Nat, p: M.P(), +ids: List<&2, Nat>, +labels: List<&2, Nat>, +out: String) -> IO(Unit):
  do IO<Unit>:
    tr : T.Train<M.P()> <- IO.pure(T.Train<M.P()>,
      T.Train.new(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~M.Model(), cfg(), p))
    tr1 : T.Train<M.P()> <- train_all(steps, 0n, ids, labels, tr)
    save_final(tr1, out, ids, labels)

type Run is Type:
  Run{steps: Nat, data: String, out: String}

def run_default() -> Run:
  Run{STEPS(), "examples/language/glm53_flash/data", "/tmp/opencode/glm53/c"}

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

def run_io(r: Run) -> IO(Unit):
  match r:
    case Run{+steps, +data, out}:
      do IO<Unit>:
        T.OptCfg.check(cfg())
        IO.print(String.append("params: ",
          U32.show(T.Layer.count(M.P(), M.X(), M.Y(), M.S(), M.Model()))))
        IO.print(String.append("steps: ", Nat.show(steps)))
        p : M.P() <- W.load_params(data)
        ids : List<&2, U32> <- T.read_at(String.append(data, "/c_batch.bin"), 0, 64)
        labels : List<&2, U32> <- T.read_at(String.append(data, "/c_labels.bin"), 0, 64)
        run_loaded(steps, p, nats(ids), nats(labels), out)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    run_io(run_of(args))
