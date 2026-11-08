# Stage A whole-model probe: load the twin's tensors, print the logits, the
# parameter count and the cross-entropy. `probe_a_alt.bend` prints the same for
# the future-modified batch; the test compares the first half (causality).
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_a.bend as M

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

def show(tag: String, xs: List<&2, F32>) -> String:
  String.append(tag, String.append(" ", show_go(xs)))

def flat(-r: Nat, -c: Nat, ml: TA.ML<r, c>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, l}:
      l

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def P() -> Type:
  M.P()

def BlockP() -> Type:
  M.BlockP()

def load_block(+dir: String) -> IO(BlockP()):
  do IO<BlockP()>:
    ln1 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(dir, "/ln1.bin"), 0)
    wq : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, String.append(dir, "/wq.bin"), 0)
    wk : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, String.append(dir, "/wk.bin"), 0)
    wv : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, String.append(dir, "/wv.bin"), 0)
    wo : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, String.append(dir, "/wo.bin"), 0)
    ln2 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(dir, "/ln2.bin"), 0)
    wg : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, String.append(dir, "/wg.bin"), 0)
    wu : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, String.append(dir, "/wu.bin"), 0)
    wd : TA.Mat<32n, 128n> <- T.Mat.load(32n, 128n, String.append(dir, "/wd.bin"), 0)
    return T.Two{
      T.Two{T.Two{Unit{}, T.Two{ln1, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{T.Two{Unit{}, T.Two{Unit{}, Unit{}}}, T.Two{Unit{}, wo}}}}}, Unit{}},
      T.Two{T.Two{Unit{}, T.Two{ln2, T.Two{T.Two{T.Two{T.Two{wg, Unit{}}, wu}, Unit{}}, wd}}}, Unit{}}}

def load_params(+dir: String) -> IO(P()):
  do IO<P()>:
    emb : TA.Mat<260n, 32n> <- T.Mat.load(260n, 32n, String.append(dir, "/emb.bin"), 0)
    b0 : BlockP() <- load_block(String.append(dir, "/b0"))
    b1 : BlockP() <- load_block(String.append(dir, "/b1"))
    b2 : BlockP() <- load_block(String.append(dir, "/b2"))
    b3 : BlockP() <- load_block(String.append(dir, "/b3"))
    lnf : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(dir, "/lnf.bin"), 0)
    return T.Two{emb, T.Two{T.Two{T.Two{b0, b1}, T.Two{b2, b3}}, lnf}}

def show_loss(l: T.LossOut<M.Y(2n)>) -> IO(Unit):
  match l:
    case T.LossOut{loss, dy}:
      IO.print(String.append("loss: ", F32.show(loss)))

def fl(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> List<&2, F32>:
  flat(r, c, TA.Mat.to_list(r, c, m))

def app(a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, a, b)

def flat_block(b: BlockP()) -> List<&2, F32>:
  match b:
    case T.Two{T.Two{T.Two{Unit{}, T.Two{ln1, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{T.Two{Unit{}, T.Two{Unit{}, Unit{}}}, T.Two{Unit{}, wo}}}}}, Unit{}}, T.Two{T.Two{Unit{}, T.Two{ln2, T.Two{T.Two{T.Two{T.Two{wg, Unit{}}, wu}, Unit{}}, wd}}}, Unit{}}}:
      app(fl(1n, 32n, ln1), app(fl(32n, 32n, wq), app(fl(32n, 32n, wk), app(fl(32n, 32n, wv), app(fl(32n, 32n, wo), app(fl(1n, 32n, ln2), app(fl(128n, 32n, wg), app(fl(128n, 32n, wu), fl(32n, 128n, wd)))))))))

def print_grads(b: T.Bwd<P(), M.X()>) -> IO(Unit):
  match b:
    case T.Bwd{_, dx, gp}:
      match gp:
        case T.Two{emb, T.Two{T.Two{T.Two{b0, b1}, T.Two{b2, b3}}, lnf}}:
          do IO<Unit>:
            IO.print(show("grad:", app(fl(260n, 32n, emb), app(flat_block(b0), app(flat_block(b1), app(flat_block(b2), app(flat_block(b3), fl(1n, 32n, lnf))))))))
            return Unit{}

def loss_grad(l: T.LossOut<M.Y(2n)>, p: P(), s: M.S(2n)) -> IO(Unit):
  match l:
    case T.LossOut{loss, dy}:
      do IO<Unit>:
        IO.print(String.append("loss: ", F32.show(loss)))
        print_grads(T.Layer.bwd(P(), M.X(), M.Y(2n), M.S(2n), M.Model(~2n), p, s, dy,
          T.Layer.zeros(P(), M.X(), M.Y(2n), M.S(2n), M.Model(~2n))))

def split_out(mm: TA.MM2<128n, 260n>, labs: List<&2, Nat>, tag: String, p: P(), s: M.S(2n)) -> IO(Unit):
  match mm:
    case TA.MM2{a, b}:
      do IO<Unit>:
        IO.print(String.append("params: ", U32.show(T.Layer.count(P, M.X(), M.Y(2n), M.S(2n), M.Model(~2n)))))
        IO.print(show(tag, flat(128n, 260n, TA.Mat.to_list(128n, 260n, a))))
        loss_grad(T.CrossEntropy(128n, 260n, b, labs), p, s)

def out_of(r: T.Fwd<P, M.Y(2n), M.S(2n)>, labs: List<&2, Nat>, tag: String) -> IO(Unit):
  match r:
    case T.Fwd{pp, y, s}:
      split_out(TA.Mat.clone(128n, 260n, y), labs, tag, pp, s)

def run_fwd(p: P, ids: List<&2, Nat>, labs: List<&2, Nat>, tag: String) -> IO(Unit):
  out_of(T.Layer.fwd(P, M.X(), M.Y(2n), M.S(2n), M.Model(~2n), p, ids), labs, tag)

def run(dir: String, batch: String, label: String, tag: String) -> IO(Unit):
  do IO<Unit>:
    p : P() <- load_params(dir)
    ids : List<&2, U32> <- T.read_at(batch, 0, 128)
    labs : List<&2, U32> <- T.read_at(label, 0, 128)
    run_fwd(p, nats(ids), nats(labs), tag)

def main() -> IO(Unit):
  run("examples/language/glm53_flash/data", "examples/language/glm53_flash/data/batch.bin", "examples/language/glm53_flash/data/labels.bin", "logits:")
