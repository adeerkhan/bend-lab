# The flow's token order (ISSUE-044): prints `sample_input(0.5, synthetic)` as raw words, so the row reset can be read
# at rows 64..127 instead of inferred from a forward difference.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../examples/diffusion/flowers_flow/main.bend as M

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+r:
      Con{(F32.from_nat(Nat.mod(i, 251n)) / 255.0 : F32), vals(r, Nat.add(i, 1n))}

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

def main() -> IO(Unit):
  IO.print(show_go(M.sample_input(0.5, vals(8192n, 0n))))
