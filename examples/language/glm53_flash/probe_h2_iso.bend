# Two 128-token sequences in one 256-row model: changing sequence 0 must not
# move sequence 1's logits.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_h2.bend as M
import ./weights_h2.bend as W

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def ml(ml: TA.ML<256n, 260n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def head(xs: List<&2, U32>) -> U32:
  match xs:
    case Con{h, t}:
      h
    case Nil{}:
      0

def y_of(fw: T.Fwd<M.P(), M.Y(), M.S()>) -> List<&2, F32>:
  match fw:
    case T.Fwd{pp, y, s}:
      ml(TA.Mat.to_list(256n, 260n, y))

def run(p: M.P(), +ids: List<&2, Nat>) -> List<&2, F32>:
  y_of(T.Layer.fwd(M.P(), M.X(), M.Y(), M.S(), M.Model(), p, ids))

def maxdiff(xs: List<&2, F32>, ys: List<&2, F32>, acc: F32) -> F32:
  match xs ys:
    case Nil{} _:
      acc
    case _ Nil{}:
      acc
    case Con{h, t} Con{g, u}:
      maxdiff(t, u, F32.max(acc, F32.abs((h - g : F32))))

def first_half(xs: List<&2, F32>, ys: List<&2, F32>) -> F32:
  maxdiff(List.take(&2, F32, xs, 33280n), List.take(&2, F32, ys, 33280n), 0.0)

def second_half(xs: List<&2, F32>, ys: List<&2, F32>) -> F32:
  maxdiff(List.drop(&2, F32, xs, 33280n), List.drop(&2, F32, ys, 33280n), 0.0)

def set_first(ids: List<&2, U32>, +v: U32) -> List<&2, U32>:
  match ids:
    case Con{h, t}:
      Con{v, t}
    case Nil{}:
      Nil{}

def main() -> IO(Unit):
  do IO<Unit>:
    p1 : M.P() <- W.load_params("examples/language/glm53_flash/data/h")
    ids1 : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin", 0, 256)
    a : List<&2, F32> = run(p1, nats(ids1))
    p2 : M.P() <- W.load_params("examples/language/glm53_flash/data/h")
    one : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin", 0, 1)
    idsb : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin", 0, 256)
    bad : List<&2, Nat> = nats(set_first(idsb, U32.add(head(one), 1)))
    b : List<&2, F32> = run(p2, bad)
    IO.print(String.append("seq1_diff: ", F32.show(second_half(a, b))))
    p3 : M.P() <- W.load_params("examples/language/glm53_flash/data/h")
    ids3 : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin", 0, 256)
    c : List<&2, F32> = run(p3, nats(ids3))
    p4 : M.P() <- W.load_params("examples/language/glm53_flash/data/h")
    one2 : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin", 0, 1)
    idsb2 : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin", 0, 256)
    bad2 : List<&2, Nat> = nats(set_first(idsb2, U32.add(head(one2), 1)))
    d : List<&2, F32> = run(p4, bad2)
    IO.print(String.append("seq0_diff: ", F32.show(first_half(c, d))))
