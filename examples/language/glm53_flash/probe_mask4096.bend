# The 4096-row block mask: build it, materialise it, and print its length.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./sp_dense.bend as SD

def ml(x: TA.ML<4096n, 4096n>) -> List<&2, F32>:
  match x:
    case TA.ML{m, xs}:
      xs

def len(xs: List<&2, F32>, acc: Nat) -> Nat:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      len(t, (acc + 1n : Nat))

def main() -> IO(Unit):
  do IO<Unit>:
    m : TA.Mat<4096n, 4096n> = SD.mask_block_ws(4096n, 128n, 16n, 16n, True{})
    IO.print(String.append("values: ", Nat.show(len(ml(TA.Mat.to_list(4096n, 4096n, m)), 0n))))
