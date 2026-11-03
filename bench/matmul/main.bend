# A matrix-product benchmark: the compute kernel, with no dataset and no IO.
#
#   bend examples/matmul/main.bend
#
# Fifty products of two 128 x 128 matrices, sequential (par 0), over
# `bend-ml-tensor-array`'s flat `Array<F32>`. `reference/bench_matmul.py`
# times this program and the same products in NumPy, and writes the
# comparison to results/bench-matmul.json.
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# The first number of a list, or 0 when it is empty.
def first(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{+h, t}:
      h

# The numbers of a matrix, out of the wrapper the read returns.
def list_of(m: TA.ML<128n, 128n>) -> List<&2, F32>:
  match m:
    case TA.ML{m, l}:
      l

# The corner of the product: the checksum that keeps the work alive.
def checksum(product: TA.MMul<128n, 128n, 128n>) -> F32:
  match product:
    case TA.MMul{a, b, c}:
      first(list_of(TA.Mat.to_list(128n, 128n, c)))

# The rounds of products. The operands come back with every product, so the
# next round multiplies the same matrices again.
def rounds_go(fuel: List<&2, Nat>, product: TA.MMul<128n, 128n, 128n>) -> F32:
  match fuel:
    case Nil{}:
      checksum(product)
    case Con{u, t}:
      match product:
        case TA.MMul{a, b, c}:
          rounds_go(t, TA.Mat.matmul(128n, 128n, 128n, 0n, a, b))

# Fifty products of the same two matrices: one, then forty-nine rounds.
def run() -> F32:
  rounds_go(List.range(49n),
    TA.Mat.matmul(128n, 128n, 128n, 0n,
      TA.Mat.fill(128n, 128n, 1.0), TA.Mat.fill(128n, 128n, 2.0)))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("checksum: ", F32.show(run())))
