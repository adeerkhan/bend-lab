# Step-0 token probe (ISSUE-040): prints the example's `step_in(0, synthetic)` as raw words, so the token order can be
# compared row by row with the twin's instead of inferred from a loss. Synthetic pixels: (i mod 251) / 255, so every
# value is distinct and a swapped pair or a permuted image shows up immediately.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../examples/diffusion/vit/main.bend as V

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+r:
      Con{(F32.from_nat(Nat.mod(i, 251n)) / 255.0 : F32), vals(r, Nat.add(i, 1n))}

def main() -> IO(Unit):
  T.Mat.print(V.RW(), V.FE(), "tokens", V.step_in(0n, vals(65536n, 0n)))
