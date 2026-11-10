# A timing probe for the two four-head cores at N=128: linear and sparse.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./attn_mh.bend as A

def val(+i: Nat) -> F32:
  +k = Nat.mod(Nat.add(Nat.mul(i, 73n), 19n), 211n)
  (F32.from_nat(k) / 105.5 - 1.0 : F32)

def gen(+left: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      gen(p, (i + 1n : Nat), seed, Con{val(Nat.add(i, seed)), acc})

def sum(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      sum(t, (acc + h : F32))

def sum_bw(bw: A.ACC, acc: F32) -> F32:
  match bw:
    case A.ACC{a, b, c}:
      sum(a, sum(b, sum(c, acc)))

def ms(+a: Nat, +b: Nat) -> Nat:
  Nat.sub(b, a)

def main() -> IO(Unit):
  do IO<Unit>:
    +t0 : Nat <- IO.now()
    ly : List<&2, F32> = A.lin_y4(gen(4096n, 0n, 0n, Nil{}), gen(4096n, 0n, 1000n, Nil{}), gen(4096n, 0n, 2000n, Nil{}))
    +t1 : Nat <- IO.now()
    lb : A.ACC = A.lin_bwd4(gen(4096n, 0n, 3000n, Nil{}), gen(4096n, 0n, 0n, Nil{}),
      gen(4096n, 0n, 1000n, Nil{}), gen(4096n, 0n, 2000n, Nil{}))
    +t2 : Nat <- IO.now()
    sy : List<&2, F32> = A.sp_y4(gen(4096n, 0n, 0n, Nil{}), gen(4096n, 0n, 1000n, Nil{}), gen(4096n, 0n, 2000n, Nil{}))
    +t3 : Nat <- IO.now()
    sb : A.ACC = A.sp_bwd4(gen(4096n, 0n, 3000n, Nil{}), gen(4096n, 0n, 0n, Nil{}),
      gen(4096n, 0n, 1000n, Nil{}), gen(4096n, 0n, 2000n, Nil{}))
    +t4 : Nat <- IO.now()
    IO.print(String.append("lin_y_ms: ", Nat.show(ms(t0, t1))))
    IO.print(String.append("lin_bwd_ms: ", Nat.show(ms(t1, t2))))
    IO.print(String.append("sp_y_ms: ", Nat.show(ms(t2, t3))))
    IO.print(String.append("sp_bwd_ms: ", Nat.show(ms(t3, t4))))
    IO.print(String.append("checksum: ", F32.show(sum_bw(lb, sum_bw(sb, sum(ly, sum(sy, 0.0)))))))
