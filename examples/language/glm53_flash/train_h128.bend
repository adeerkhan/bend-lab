# The four-block hybrid on a real corpus stream at 128 positions.
#
# Batch 8, 128 positions, 400 steps: the windows are the reference's synthetic
# corpus, cut once by `glm53_dense_data.write_windows` (409,600 train tokens,
# one held-out batch at offset 409,600).  The model is `model_128.bend`, the
# same 74,144-parameter Stage A stack at 128 positions.  The learning rate is
# a parameter: the 20-step parity gate at 3e-4 is a correctness gate, not a
# learning benchmark.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_h.bend as M
import ./weights_h.bend as W
import ./loss_masked.bend as L

def B() -> Nat:
  1n

def POS() -> Nat:
  128n

def ROWS() -> Nat:
  Nat.mul(B(), POS())

def IGNORE() -> Nat:
  260n

def STEPS() -> Nat:
  400n

def cfg(+lr: F32) -> T.OptCfg:
  T.CAdamW{lr, 0.9, 0.95, 0.00000001, 0.1, 0.1, 0.05}

def ce(z: M.Y(), labels: List<&2, Nat>) -> T.LossOut<M.Y()>:
  L.CrossEntropyMasked(ROWS(), 260n, z, labels, IGNORE())

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def step(tr: T.Train<M.P()>, +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> T.Train<M.P()>:
  T.Train.step(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~List<&2, Nat>, ~M.Model(),
    ~ce, tr, ids, labels)

# One trace line per loop iteration: the batch read and `T.Train.step` together,
# so the five-stage probe in bench/plumbing_glm53 sums to the same work.
def trace_line(+i: Nat, +loss: F32, +ms: Nat) -> String:
  +a = String.append("{\"step\": ", Nat.show(i))
  +b = String.append(a, ", \"tokens\": ")
  +c = String.append(b, Nat.show(ROWS()))
  +d = String.append(c, ", \"loss\": ")
  +e = String.append(d, F32.show(loss))
  +f = String.append(e, ", \"ms\": ")
  +g = String.append(f, Nat.show(ms))
  String.append(g, "}")

def show_step(+trace: Bool, +i: Nat, +loss: F32, +ms: Nat) -> IO(Unit):
  match trace:
    case True{}:
      IO.print(trace_line(i, loss, ms))
    case False{}:
      IO.print(String.append("loss: ", F32.show(loss)))

def step_res(+trace: Bool, +i: Nat, +ms: Nat, tr2: T.Train<M.P()>) -> IO(T.Train<M.P()>):
  match tr2:
    case T.Train{p, g, st, c, p1, p2, +l}:
      do IO<T.Train<M.P()>>:
        show_step(trace, i, l, ms)
        return T.Train{p, g, st, c, p1, p2, l}

def step_print(+trace: Bool, +i: Nat, tr: T.Train<M.P()>, +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> IO(T.Train<M.P()>):
  do IO<T.Train<M.P()>>:
    t0 : Nat <- IO.now()
    tr1 : T.Train<M.P()> <- IO.pure(T.Train<M.P()>, step(tr, ids, labels))
    t1 : Nat <- IO.now()
    step_res(trace, i, Nat.sub(t1, t0), tr1)

def show_final(+trace: Bool, +l: F32) -> IO(Unit):
  match trace:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      IO.print(String.append("final_loss: ", F32.show(l)))

def train_all(+trace: Bool, +left: Nat, +i: Nat, +data: String, tr: T.Train<M.P()>) -> IO(T.Train<M.P()>):
  match left:
    case 0n:
      match tr:
        case T.Train{p, g, st, c, p1, p2, +l}:
          do IO<T.Train<M.P()>>:
            show_final(trace, l)
            return T.Train{p, g, st, c, p1, p2, l}
    case 1n+rest:
      do IO<T.Train<M.P()>>:
        ids : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-train_batch.bin"),
          U32.from_nat(Nat.mul(i, ROWS())), U32.from_nat(ROWS()))
        labels : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-train_labels.bin"),
          U32.from_nat(Nat.mul(i, ROWS())), U32.from_nat(ROWS()))
        tr1 : T.Train<M.P()> <- step_print(trace, (i + 1n : Nat), tr, nats(ids), nats(labels))
        train_all(trace, rest, (i + 1n : Nat), data, tr1)

# ---- the held-out loss ----

def loss_of(lo: T.LossOut<M.Y()>) -> F32:
  match lo:
    case T.LossOut{+loss, dy}:
      loss

def val_fwd(fw: T.Fwd<M.P(), M.Y(), M.S()>, +labels: List<&2, Nat>) -> F32:
  match fw:
    case T.Fwd{pp, y, s}:
      loss_of(ce(y, labels))

def val_loss(p: M.P(), +data: String) -> IO(F32):
  do IO<F32>:
    ids : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-val_batch.bin"), 0,
      U32.from_nat(ROWS()))
    labels : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-val_labels.bin"), 0,
      U32.from_nat(ROWS()))
    return val_fwd(T.Layer.fwd(M.P(), M.X(), M.Y(), M.S(), M.Model(), p, nats(ids)),
      nats(labels))

def save_final(tr: T.Train<M.P()>, +out: String, +data: String) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, c, p1, p2, +l}:
      do IO<Unit>:
        p1v : M.P() <- T.Params.save(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~M.Model(),
          String.append(out, "/final_weights.bin"), p)
        vl : F32 <- val_loss(p1v, data)
        IO.print(String.append("val_loss: ", F32.show(vl)))

# ---- arguments ----

type Run is Type:
  Run{steps: Nat, lrppm: Nat, data: String, out: String, weights: String, val: Bool,
      trace: Bool, save: Bool}

def run_default() -> Run:
  Run{STEPS(), 3000n, "examples/language/glm53_flash/data/corpus128-b32", "/tmp/opencode/glm53/hybrid128", "examples/language/glm53_flash/data/h", True{}, False{}, True{}}

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
                case Run{steps, lrppm, data, out, weights, val, trace, save}:
                  run_go(more, Run{nat_or(Nat.read(b), steps), lrppm, data, out, weights, val, trace, save})
        case "--lr-ppm":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{steps, lrppm, data, out, weights, val, trace, save}:
                  run_go(more, Run{steps, nat_or(Nat.read(b), lrppm), data, out, weights, val, trace, save})
        case "--data":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{steps, lrppm, data, out, weights, val, trace, save}:
                  run_go(more, Run{steps, lrppm, b, out, weights, val, trace, save})
        case "--out":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{steps, lrppm, data, out, weights, val, trace, save}:
                  run_go(more, Run{steps, lrppm, data, b, weights, val, trace, save})
        case "--no-val":
          match r:
            case Run{steps, lrppm, data, out, weights, val, trace, save}:
              run_go(rest, Run{steps, lrppm, data, out, weights, False{}, trace, save})
        case "--trace":
          match r:
            case Run{steps, lrppm, data, out, weights, val, trace, save}:
              run_go(rest, Run{steps, lrppm, data, out, weights, val, True{}, save})
        case "--no-save":
          match r:
            case Run{steps, lrppm, data, out, weights, val, trace, save}:
              run_go(rest, Run{steps, lrppm, data, out, weights, val, trace, False{}})
        case _:
          run_go(rest, r)

def run_of(args: List<String>) -> Run:
  match args:
    case Nil{}:
      run_default()
    case Con{program, rest}:
      run_go(rest, run_default())

def save_only(tr: T.Train<M.P()>, +out: String) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, c, p1, p2, +l}:
      do IO<Unit>:
        p1v : M.P() <- T.Params.save(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~M.Model(),
          String.append(out, "/final_weights.bin"), p)
        return Unit{}

def finish(tr: T.Train<M.P()>, +out: String, +data: String, +save: Bool, +do_val: Bool) -> IO(Unit):
  match save:
    case False{}:
      IO.pure(Unit, Unit{})
    case True{}:
      match do_val:
        case True{}:
          save_final(tr, out, data)
        case False{}:
          save_only(tr, out)

def show_header(+trace: Bool, +steps: Nat, +lr: F32) -> IO(Unit):
  match trace:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      do IO<Unit>:
        IO.print(String.append("params: ",
          U32.show(T.Layer.count(M.P(), M.X(), M.Y(), M.S(), M.Model()))))
        IO.print(String.append("steps: ", Nat.show(steps)))
        IO.print(String.append("lr: ", F32.show(lr)))
        return Unit{}

def run_io_lr(+steps: Nat, +lr: F32, +data: String, +out: String, +weights: String,
    +do_val: Bool, +trace: Bool, +save: Bool) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg(lr))
    show_header(trace, steps, lr)
    p : M.P() <- W.load_params(weights)
    tr : T.Train<M.P()> <- IO.pure(T.Train<M.P()>,
      T.Train.new(~M.P(), ~M.X(), ~M.Y(), ~M.S(), ~M.Model(), cfg(lr), p))
    tr1 : T.Train<M.P()> <- train_all(trace, steps, 0n, data, tr)
    finish(tr1, out, data, save, do_val)

def run_io(r: Run) -> IO(Unit):
  match r:
    case Run{+steps, +lrppm, +data, out, +weights, +val, +trace, +save}:
      run_io_lr(steps, (F32.div(F32.from_nat(lrppm), 1000000.0) : F32), data, out, weights, val, trace, save)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    run_io(run_of(args))
