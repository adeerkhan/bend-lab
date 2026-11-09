# The reference's masked cross-entropy over a logits matrix.
#
# `T.CrossEntropy` averages every row.  This loss ignores rows whose label is
# `ignore` (the tokenizer's pad is 0, but the reference masks with -100; here
# the caller passes 260, outside the vocabulary).  The forward and backward
# arithmetic is the package's own: the row's softmax is written over its
# logits, and the gradient is `(p - onehot) / valid_rows`.  A masked row keeps
# a zero gradient row and contributes nothing to the mean.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def valid_rows(labels: List<&2, Nat>, +ignore: Nat, +acc: Nat) -> Nat:
  match labels:
    case Nil{}:
      acc
    case Con{+y, t}:
      valid_rows(t, ignore, Bool.pick(Nat, Nat.is_eq(y, ignore), acc, Nat.add(acc, 1n)))

def zero_row(+left: Nat, +idx: U32, z: Array<F32>) -> Array<F32>:
  match left z:
    case 0n _:
      z
    case 1n+p _:
      zero_row(p, (idx + 1 : U32), Array.set(F32, z, idx, 0.0))

def mask_one(masked: Bool, +c: Nat, +i: U32, +y: Nat, +nf: F32, +loss: F32, z: Array<F32>) -> T.AS:
  match masked:
    case True{}:
      T.AS{zero_row(c, (i * U32.from_nat(c) : U32), z), loss}
    case False{}:
      T.ce_row2(c, (i * U32.from_nat(c) : U32), y, nf,
        T.ce_pick(Nat.is_lt(y, c), c, (i * U32.from_nat(c) : U32),
          ((i * U32.from_nat(c) : U32) + U32.from_nat(y) : U32), loss, z))

def mask_rows(labels: List<&2, Nat>, +c: Nat, +i: U32, +nf: F32, +ignore: Nat, s: T.AS) -> T.AS:
  match labels s:
    case Nil{} _:
      s
    case Con{+y, t} T.AS{z, +loss}:
      mask_rows(t, c, (i + 1 : U32), nf, ignore,
        mask_one(Nat.is_eq(y, ignore), c, i, y, nf, loss, z))

def masked_fin(-n: Nat, -c: Nat, +nf: F32, r: T.AS) -> T.LossOut<TA.Mat<n, c>>:
  match r:
    case T.AS{d, +l}:
      T.LossOut{(l / nf : F32), TA.Mat{d}}

def CrossEntropyMasked(+n: Nat, +c: Nat, z: TA.Mat<n, c>, +labels: List<&2, Nat>, +ignore: Nat) -> T.LossOut<TA.Mat<n, c>>:
  match z:
    case TA.Mat{d}:
      masked_fin(n, c, F32.from_nat(valid_rows(labels, ignore, 0n)),
        mask_rows(labels, c, 0, F32.from_nat(valid_rows(labels, ignore, 0n)), ignore, T.AS{d, 0.0}))
