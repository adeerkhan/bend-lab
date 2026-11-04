# A single-head self-attention block, the core of a transformer, in plain
# Bend over the `bend-ml` tensor shapes but with lists of F32.
#
#   bendenv run -- bend examples/attention/main.bend
#
# Four tokens of eight features, three learned projections (Q, K, V), the
# scaled dot-product scores, a softmax over every row, and the weighted sum
# of V:
#
#   Q = X·Wq, K = X·Wk, V = X·Wv
#   S = Q·Kᵀ / sqrt(8)
#   O = softmax(S)·V
#
# The weights are a fixed small pattern, generated the same way in the
# PyTorch reference, so the two programs compute the same function and the
# outputs can be compared number by number (reference/test_attention.py).
# This is the forward pass. The maintained BendTorch (`bendtorch@2.0.0.0`)
# has recorded attention and softmax; the trained transformer is
# `examples/transformer`.
import Base

# The shape of the block.

def tokens() -> Nat:
  4n

def features() -> Nat:
  8n

# The starting numbers: a small pattern of the row and the column, the
# same formula in the reference.

def x_cell(+i: Nat, +j: Nat) -> F32:
  F32.sub(F32.mul(F32.from_nat(Nat.mod(Nat.add(Nat.mul(i, 3n), Nat.mul(j, 5n)), 7n)), 0.1), 0.2)

def wq_cell(+i: Nat, +j: Nat) -> F32:
  F32.sub(F32.mul(F32.from_nat(Nat.mod(Nat.add(Nat.mul(i, 2n), Nat.mul(j, 3n)), 5n)), 0.1), 0.2)

def wk_cell(+i: Nat, +j: Nat) -> F32:
  F32.sub(F32.mul(F32.from_nat(Nat.mod(Nat.add(Nat.add(Nat.mul(i, 7n), Nat.mul(j, 11n)), 1n), 6n)), 0.1), 0.2)

def wv_cell(+i: Nat, +j: Nat) -> F32:
  F32.sub(F32.mul(F32.from_nat(Nat.mod(Nat.add(Nat.add(Nat.mul(i, 5n), Nat.mul(j, 2n)), 3n), 6n)), 0.1), 0.2)

# The cell at an index, or 0 when the row is short.
def at_go(xs: List<&2, F32>, +k: Nat, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{+h, t}:
      Bool.pick(F32, Nat.is_eq(i, k), h, at_go(t, k, Nat.add(i, 1n)))

# The cell at an index.
def at(xs: List<&2, F32>, k: Nat) -> F32:
  at_go(xs, k, 0n)

# One cell of one builder: 0 is X, 1 is Wq, 2 is Wk, 3 is Wv.
def cell(+kind: Nat, +i: Nat, +j: Nat) -> F32:
  match kind:
    case 0n:
      x_cell(i, j)
    case 1n:
      wq_cell(i, j)
    case 2n:
      wk_cell(i, j)
    case 3n:
      wv_cell(i, j)
    case _:
      0.0

# A row of one builder, cell by cell.
def row_go(+kind: Nat, +i: Nat, +n: Nat, +j: Nat, +acc: List<&2, F32>) ->
  List<&2, F32>:
  match n:
    case 0n:
      acc
    case 1n+rest:
      row_go(kind, i, rest, Nat.add(j, 1n),
        List.append(&2, F32, acc, Con{cell(kind, i, j), Nil{}}))

# A row of one builder.
def row(kind: Nat, i: Nat, n: Nat) -> List<&2, F32>:
  row_go(kind, i, n, 0n, Nil{})

# The rows of one builder.
def rows_go(+kind: Nat, +n: Nat, +i: Nat, +m: Nat,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match n:
    case 0n:
      acc
    case 1n+rest:
      rows_go(kind, rest, Nat.add(i, 1n), m,
        List.append(&2, List<&2, F32>, acc, Con{row(kind, i, m), Nil{}}))

# The rows of one builder.
def rows(kind: Nat, n: Nat, m: Nat) -> List<&2, List<&2, F32>>:
  rows_go(kind, n, 0n, m, Nil{})

# The dot product of two rows.
def dot_go(a: List<&2, F32>, b: List<&2, F32>, +acc: F32) -> F32:
  match a:
    case Nil{}:
      acc
    case Con{+ha, ta}:
      match b:
        case Con{+hb, tb}:
          dot_go(ta, tb, F32.add(acc, F32.mul(ha, hb)))
        case Nil{}:
          acc

# The dot product of two rows.
def dot(a: List<&2, F32>, b: List<&2, F32>) -> F32:
  dot_go(a, b, 0.0)

# One column of a matrix.
def column_go(rows: List<&2, List<&2, F32>>, +j: Nat, +acc: List<&2, F32>) ->
  List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      column_go(t, j, List.append(&2, F32, acc, Con{at(h, j), Nil{}}))

# One column of a matrix.
def column(rows: List<&2, List<&2, F32>>, j: Nat) -> List<&2, F32>:
  column_go(rows, j, Nil{})

# One row against every column of the second matrix.
def matmul_cols(+ra: List<&2, F32>, +b: List<&2, List<&2, F32>>, +n: Nat, +j: Nat,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      acc
    case 1n+rest:
      matmul_cols(ra, b, rest, Nat.add(j, 1n),
        List.append(&2, F32, acc, Con{dot(ra, column(b, j)), Nil{}}))

# The product of two matrices.
def matmul_rows_go(+a: List<&2, List<&2, F32>>, +b: List<&2, List<&2, F32>>, +n: Nat,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match a:
    case Nil{}:
      acc
    case Con{ra, ta}:
      matmul_rows_go(ta, b, n,
        List.append(&2, List<&2, F32>, acc, Con{matmul_cols(ra, b, n, 0n, Nil{}), Nil{}}))

# The transpose of a matrix.
def transpose_go(+rows: List<&2, List<&2, F32>>, +n: Nat, +j: Nat,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match n:
    case 0n:
      acc
    case 1n+rest:
      transpose_go(rows, rest, Nat.add(j, 1n),
        List.append(&2, List<&2, F32>, acc, Con{column(rows, j), Nil{}}))

# The transpose of a matrix.
def transpose(rows: List<&2, List<&2, F32>>, n: Nat) -> List<&2, List<&2, F32>>:
  transpose_go(rows, n, 0n, Nil{})

# The product of two matrices.
def matmul(+a: List<&2, List<&2, F32>>, b: List<&2, List<&2, F32>>, n: Nat) ->
  List<&2, List<&2, F32>>:
  matmul_rows_go(a, b, n, Nil{})

# The largest number of a row.
def max_go(+xs: List<&2, F32>, +best: F32) -> F32:
  match xs:
    case Nil{}:
      best
    case Con{+h, t}:
      max_go(t, F32.max(h, best))

# The largest number of a row.
def max_of(xs: List<&2, F32>) -> F32:
  max_go(xs, (0.0 - 1.0e30 : F32))

# The sum of a row.
def sum_go(+xs: List<&2, F32>, +acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      sum_go(t, F32.add(acc, h))

# The sum of a row.
def sum_of(xs: List<&2, F32>) -> F32:
  sum_go(xs, 0.0)

# One row and a number, cell by cell.
def scale_go(+a: List<&2, F32>, +k: F32, +acc: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      acc
    case Con{ha, ta}:
      scale_go(ta, k, List.append(&2, F32, acc, Con{F32.mul(ha, k), Nil{}}))

# One row and a number, cell by cell.
def scale(a: List<&2, F32>, k: F32) -> List<&2, F32>:
  scale_go(a, k, Nil{})

# One row and a number, cell by cell, the number subtracted.
def shift_go(+a: List<&2, F32>, +k: F32, +acc: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      acc
    case Con{ha, ta}:
      shift_go(ta, k, List.append(&2, F32, acc, Con{F32.sub(ha, k), Nil{}}))

# One row and a number, the number subtracted.
def shift(a: List<&2, F32>, k: F32) -> List<&2, F32>:
  shift_go(a, k, Nil{})

# The exponentials of a row.
def exp_go(+a: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      acc
    case Con{ha, ta}:
      exp_go(ta, List.append(&2, F32, acc, Con{F32.exp(ha), Nil{}}))

# The softmax of a row: shifted by its maximum so the exponentials stay
# finite, then normalised by their sum.
def softmax(+row: List<&2, F32>) -> List<&2, F32>:
  +top = max_of(row)
  +e = exp_go(shift(row, top), Nil{})
  scale(e, (1.0 / sum_of(e) : F32))

# The softmax of every row.
def softmax_rows_go(+rows: List<&2, List<&2, F32>>,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      softmax_rows_go(t, List.append(&2, List<&2, F32>, acc, Con{softmax(h), Nil{}}))

# The softmax of every row.
def softmax_rows(rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  softmax_rows_go(rows, Nil{})

# Every cell of a matrix, scaled by one number.
def scale_rows_go(+rows: List<&2, List<&2, F32>>, +k: F32,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      scale_rows_go(t, k, List.append(&2, List<&2, F32>, acc, Con{scale(h, k), Nil{}}))

# Every cell of a matrix, scaled by one number.
def scale_rows(rows: List<&2, List<&2, F32>>, k: F32) -> List<&2, List<&2, F32>>:
  scale_rows_go(rows, k, Nil{})

# The attention output: the softmax of the scaled scores times V.
def attention() -> List<&2, List<&2, F32>>:
  +x = rows(0n, tokens(), features())
  +q = matmul(x, rows(1n, features(), features()), features())
  +k = matmul(x, rows(2n, features(), features()), features())
  +v = matmul(x, rows(3n, features(), features()), features())
  +scores = matmul(q, transpose(k, features()), tokens())
  +scaled = scale_rows(scores, (1.0 / F32.sqrt(8.0) : F32))
  matmul(softmax_rows(scaled), v, features())

# A row, as text.
def row_text_go(+xs: List<&2, F32>, +acc: String) -> String:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      row_text_go(t, String.append(acc, String.append(F32.show(h), " ")))

# A row, as text.
def row_text(xs: List<&2, F32>) -> String:
  row_text_go(xs, "")

# The checksum of a matrix.
def total_go(+rows: List<&2, List<&2, F32>>, +acc: F32) -> F32:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      total_go(t, F32.add(acc, sum_of(h)))

# The checksum of a matrix.
def total(rows: List<&2, List<&2, F32>>) -> F32:
  total_go(rows, 0.0)

# The rows, one line each.
def lines_go(+rows: List<&2, List<&2, F32>>, +i: Nat) -> IO(Unit):
  match rows:
    case Nil{}:
      IO.print("")
    case Con{h, t}:
      do IO<Unit>:
        IO.print(String.append(String.append("row ", U32.show(U32.from_nat(i))),
          String.append(": ", row_text(h))))
        lines_go(t, Nat.add(i, 1n))

def main() -> IO(Unit):
  +out = attention()
  do IO<Unit>:
    IO.print(String.append("checksum: ", F32.show(total(out))))
    lines_go(out, 0n)
