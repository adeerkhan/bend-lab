# The four-block hybrid's gradients at N=8: masked cross-entropy, the full
# model backward, every parameter gradient dumped in the parameter order.
# The inputs are `reference/glm53_h_bwd_data.py`'s; the golden is
# `reference/glm53_h_bwd.py`'s PyTorch autograd.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_h8.bend as M
import ./weights_h.bend as W
import ./loss_masked.bend as L

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def dump_list(pl: T.PL<M.P()>) -> List<&2, F32>:
  match pl:
    case T.PL{p, xs}:
      xs

def fill_list(+left: Nat, +i: U32, xs: List<&2, F32>, a: Array<F32>) -> Array<F32>:
  match left xs:
    case 0n _:
      a
    case 1n+p Con{h, t}:
      fill_list(p, (i + 1 : U32), t, Array.set(F32, a, i, h))
    case _ _:
      a

def write_arr(+path: String, +n: Nat, xs: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    a : Array<F32> = fill_list(n, 0, xs, Array.new(F32, TA.cap_depth(n), 0.0))
    b : Array<F32> <- T.write_f32(path, a, U32.from_nat(n))
    return Unit{}

type RES is Type:
  RES{loss: F32, grad: List<&2, F32>}

def res_bwd(bw: T.Bwd<M.P(), M.X()>, +loss: F32) -> RES:
  match bw:
    case T.Bwd{p3, dx, g}:
      RES{loss, dump_list(T.Layer.dump(M.P(), M.X(), M.Y(), M.S(), M.Model(), g))}

def res_loss(lo: T.LossOut<M.Y()>, p2: M.P(), s: M.S()) -> RES:
  match lo:
    case T.LossOut{loss, dy}:
      bw : T.Bwd<M.P(), M.X()> = T.Layer.bwd(M.P(), M.X(), M.Y(), M.S(), M.Model(), p2, s, dy,
        T.Layer.zeros(M.P(), M.X(), M.Y(), M.S(), M.Model()))
      res_bwd(bw, loss)

def res_fwd(fw: T.Fwd<M.P(), M.Y(), M.S()>, +labels: List<&2, Nat>) -> RES:
  match fw:
    case T.Fwd{p2, y, s}:
      res_loss(L.CrossEntropyMasked(M.ROWS(), M.VOCAB(), y, labels, M.VOCAB()), p2, s)

def grad_of(p: M.P(), +ids: List<&2, Nat>, +labels: List<&2, Nat>) -> RES:
  fw = T.Layer.fwd(M.P(), M.X(), M.Y(), M.S(), M.Model(), p, ids)
  res_fwd(fw, labels)

def count() -> Nat:
  U32.to_nat(T.Layer.count(M.P(), M.X(), M.Y(), M.S(), M.Model()))

def run_res(res: RES, +tag: String) -> IO(Unit):
  match res:
    case RES{loss, grad}:
      do IO<Unit>:
        IO.print(String.append(tag, String.append("_loss: ", F32.show(loss))))
        write_arr(String.append("examples/language/glm53_flash/data/ref8/model_", String.append(tag, "_grad.bin")),
          count(), grad)
        return Unit{}

def run(p: M.P(), +ids: List<&2, Nat>, +labels: List<&2, Nat>, +tag: String) -> IO(Unit):
  run_res(grad_of(p, ids, labels), tag)

def main() -> IO(Unit):
  do IO<Unit>:
    p : M.P() <- W.load_params("examples/language/glm53_flash/data/h")
    ids : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/ref8/model_ids.bin", 0, 8)
    labels : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/ref8/model_labels.bin", 0, 8)
    run(p, nats(ids), nats(labels), "good")
    p2 : M.P() <- W.load_params("examples/language/glm53_flash/data/h")
    ids2 : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/ref8/model_ids.bin", 0, 8)
    bad : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/ref8/model_labels_bad.bin", 0, 8)
    run(p2, nats(ids2), nats(bad), "bad")
