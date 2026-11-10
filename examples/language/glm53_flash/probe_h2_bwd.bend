# Two 128-token sequences: the parameter gradient from sequence 1 alone must
# equal the 128-token model's gradient on sequence 1.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_h.bend as M
import ./model_h2.bend as M2
import ./weights_h.bend as W
import ./weights_h2.bend as W2

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def val(+i: Nat) -> F32:
  +k = Nat.mod(Nat.add(Nat.mul(i, 37n), 11n), 101n)
  (F32.from_nat(k) / 50.0 - 1.0 : F32)

def dy_go(+left: Nat, +i: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +v = Bool.pick(F32, Nat.is_lt(i, 33280n), 0.0, val(i))
      dy_go(p, (i + 1n : Nat), Con{v, acc})

def head(xs: List<&2, U32>) -> U32:
  match xs:
    case Con{h, t}:
      h
    case Nil{}:
      0

def set_first(ids: List<&2, U32>, +v: U32) -> List<&2, U32>:
  match ids:
    case Con{h, t}:
      Con{v, t}
    case Nil{}:
      Nil{}

def make_dy() -> TA.Mat<256n, 260n>:
  TA.Mat.from_list(256n, 260n, dy_go(66560n, 0n, Nil{}))

def make_dy1() -> TA.Mat<128n, 260n>:
  TA.Mat.from_list(128n, 260n, List.drop(&2, F32, dy_go(66560n, 0n, Nil{}), 33280n))

def dump(pl: T.PL<M.P()>) -> List<&2, F32>:
  match pl:
    case T.PL{p, xs}:
      xs

def dump2(pl: T.PL<M2.P()>) -> List<&2, F32>:
  match pl:
    case T.PL{p, xs}:
      xs

type GS is Type:
  GS{a1: F32, a2: F32, d: F32}

def stats(xs: List<&2, F32>, ys: List<&2, F32>, a1: F32, a2: F32, d: F32) -> GS:
  match xs ys:
    case Nil{} _:
      GS{a1, a2, d}
    case _ Nil{}:
      GS{a1, a2, d}
    case Con{+h, t} Con{+g, u}:
      stats(t, u, F32.max(a1, F32.abs(h)), F32.max(a2, F32.abs(g)),
        F32.max(d, F32.abs((h - g : F32))))

def stat_of(r: GS) -> IO(Unit):
  match r:
    case GS{a1, a2, d}:
      do IO<Unit>:
        IO.print(String.append("g1_abs: ", F32.show(a1)))
        IO.print(String.append("g2_abs: ", F32.show(a2)))
        IO.print(String.append("grad_diff: ", F32.show(d)))
        return Unit{}

def g2_of(bw: T.Bwd<M2.P(), M2.X()>) -> List<&2, F32>:
  match bw:
    case T.Bwd{p2, dx, g}:
      dump2(T.Layer.dump(M2.P(), M2.X(), M2.Y(), M2.S(), M2.Model(), g))

def bwd2(fw: T.Fwd<M2.P(), M2.Y(), M2.S()>, dy: TA.Mat<256n, 260n>) -> List<&2, F32>:
  match fw:
    case T.Fwd{pp, y, s}:
      g2_of(T.Layer.bwd(M2.P(), M2.X(), M2.Y(), M2.S(), M2.Model(), pp, s, dy,
        T.Layer.zeros(M2.P(), M2.X(), M2.Y(), M2.S(), M2.Model())))

def g1_of(bw: T.Bwd<M.P(), M.X()>) -> List<&2, F32>:
  match bw:
    case T.Bwd{p2, dx, g}:
      dump(T.Layer.dump(M.P(), M.X(), M.Y(), M.S(), M.Model(), g))

def bwd1(fw: T.Fwd<M.P(), M.Y(), M.S()>, dy: TA.Mat<128n, 260n>) -> List<&2, F32>:
  match fw:
    case T.Fwd{pp, y, s}:
      g1_of(T.Layer.bwd(M.P(), M.X(), M.Y(), M.S(), M.Model(), pp, s, dy,
        T.Layer.zeros(M.P(), M.X(), M.Y(), M.S(), M.Model())))

def main() -> IO(Unit):
  do IO<Unit>:
    p1 : M2.P() <- W2.load_params("examples/language/glm53_flash/data/h")
    ids1 : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin", 0, 256)
    fw1 : T.Fwd<M2.P(), M2.Y(), M2.S()> = T.Layer.fwd(M2.P(), M2.X(), M2.Y(), M2.S(), M2.Model(), p1, nats(ids1))
    g1 : List<&2, F32> = bwd2(fw1, make_dy())
    p2 : M2.P() <- W2.load_params("examples/language/glm53_flash/data/h")
    one : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin", 0, 1)
    ids2 : List<&2, U32> <- T.read_at("examples/language/glm53_flash/data/corpus128-b32/corpus128-train_batch.bin", 0, 256)
    bad : List<&2, Nat> = nats(set_first(ids2, U32.add(head(one), 1)))
    fw2 : T.Fwd<M2.P(), M2.Y(), M2.S()> = T.Layer.fwd(M2.P(), M2.X(), M2.Y(), M2.S(), M2.Model(), p2, bad)
    g2 : List<&2, F32> = bwd2(fw2, make_dy())
    stat_of(stats(g1, g2, 0.0, 0.0, 0.0))
