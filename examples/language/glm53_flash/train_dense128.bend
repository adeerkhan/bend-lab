# The dense Stage A baseline on a real corpus stream at 128 positions.
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
import ./model_128.bend as M
import ./weights_a.bend as W
import ./loss_masked.bend as L

def B() -> Nat:
  8n

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

def ce(z: M.Y(B()), labels: List<&2, Nat>) -> T.LossOut<M.Y(B())>:
  L.CrossEntropyMasked(ROWS(), 260n, z, labels, IGNORE())

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def step(tr: T.Train<M.P()>, +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> T.Train<M.P()>:
  T.Train.step(~M.P(), ~M.X(), ~M.Y(B()), ~M.S(B()), ~List<&2, Nat>, ~M.Model(~B()),
    ~ce, tr, ids, labels)

def step_res(tr2: T.Train<M.P()>) -> IO(T.Train<M.P()>):
  match tr2:
    case T.Train{p, g, st, c, p1, p2, +l}:
      do IO<T.Train<M.P()>>:
        IO.print(String.append("loss: ", F32.show(l)))
        return T.Train{p, g, st, c, p1, p2, l}

def step_print(tr: T.Train<M.P()>, +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> IO(T.Train<M.P()>):
  step_res(step(tr, ids, labels))

def train_all(+left: Nat, +i: Nat, +data: String, tr: T.Train<M.P()>) -> IO(T.Train<M.P()>):
  match left:
    case 0n:
      match tr:
        case T.Train{p, g, st, c, p1, p2, +l}:
          do IO<T.Train<M.P()>>:
            IO.print(String.append("final_loss: ", F32.show(l)))
            return T.Train{p, g, st, c, p1, p2, l}
    case 1n+rest:
      do IO<T.Train<M.P()>>:
        ids : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-train_batch.bin"),
          U32.from_nat(Nat.mul(i, ROWS())), U32.from_nat(ROWS()))
        labels : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-train_labels.bin"),
          U32.from_nat(Nat.mul(i, ROWS())), U32.from_nat(ROWS()))
        tr1 : T.Train<M.P()> <- step_print(tr, nats(ids), nats(labels))
        train_all(rest, (i + 1n : Nat), data, tr1)

# ---- the held-out loss ----

def loss_of(lo: T.LossOut<M.Y(B())>) -> F32:
  match lo:
    case T.LossOut{+loss, dy}:
      loss

def val_fwd(fw: T.Fwd<M.P(), M.Y(B()), M.S(B())>, +labels: List<&2, Nat>) -> F32:
  match fw:
    case T.Fwd{pp, y, s}:
      loss_of(ce(y, labels))

def val_loss(p: M.P(), +data: String) -> IO(F32):
  do IO<F32>:
    ids : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-val_batch.bin"), 0,
      U32.from_nat(ROWS()))
    labels : List<&2, U32> <- T.read_at(String.append(data, "/corpus128-val_labels.bin"), 0,
      U32.from_nat(ROWS()))
    return val_fwd(T.Layer.fwd(M.P(), M.X(), M.Y(B()), M.S(B()), M.Model(~B()), p, nats(ids)),
      nats(labels))

def save_final(tr: T.Train<M.P()>, +out: String, +data: String) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, c, p1, p2, +l}:
      do IO<Unit>:
        p1v : M.P() <- T.Params.save(~M.P(), ~M.X(), ~M.Y(B()), ~M.S(B()), ~M.Model(~B()),
          String.append(out, "/final_weights.bin"), p)
        vl : F32 <- val_loss(p1v, data)
        IO.print(String.append("val_loss: ", F32.show(vl)))

# ---- arguments ----

type Run is Type:
  Run{steps: Nat, lrppm: Nat, data: String, out: String, weights: String}

def run_default() -> Run:
  Run{STEPS(), 3000n, "examples/language/glm53_flash/data", "/tmp/opencode/glm53/corpus128", "examples/language/glm53_flash/data"}

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
                case Run{steps, lrppm, data, out, weights}:
                  run_go(more, Run{nat_or(Nat.read(b), steps), lrppm, data, out, weights})
        case "--lr-ppm":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{steps, lrppm, data, out, weights}:
                  run_go(more, Run{steps, nat_or(Nat.read(b), lrppm), data, out, weights})
        case "--data":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{steps, lrppm, data, out, weights}:
                  run_go(more, Run{steps, lrppm, b, out, weights})
        case "--out":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{steps, lrppm, data, out, weights}:
                  run_go(more, Run{steps, lrppm, data, b, weights})
        case _:
          run_go(rest, r)

def run_of(args: List<String>) -> Run:
  match args:
    case Nil{}:
      run_default()
    case Con{program, rest}:
      run_go(rest, run_default())

def run_io_lr(+steps: Nat, +lr: F32, +data: String, +out: String, +weights: String) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg(lr))
    IO.print(String.append("params: ",
      U32.show(T.Layer.count(M.P(), M.X(), M.Y(B()), M.S(B()), M.Model(~B())))))
    IO.print(String.append("steps: ", Nat.show(steps)))
    IO.print(String.append("lr: ", F32.show(lr)))
    p : M.P() <- W.load_params(weights)
    tr : T.Train<M.P()> <- IO.pure(T.Train<M.P()>,
      T.Train.new(~M.P(), ~M.X(), ~M.Y(B()), ~M.S(B()), ~M.Model(~B()), cfg(lr), p))
    tr1 : T.Train<M.P()> <- train_all(steps, 0n, data, tr)
    save_final(tr1, out, data)

def run_io(r: Run) -> IO(Unit):
  match r:
    case Run{+steps, +lrppm, +data, out, +weights}:
      run_io_lr(steps, (F32.div(F32.from_nat(lrppm), 1000000.0) : F32), data, out, weights)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    run_io(run_of(args))
