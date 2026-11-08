# The first layer of the lucidrains rectified-flow reference: its Downsample.
#
# The reference's `Downsample` is not a strided convolution:
#
#   Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2)
#   Conv2d(dim * 4, dim_out, 1)
#
# i.e. a 2x2 space-to-depth (pixel unshuffle), then a 1x1 convolution. This file
# implements the parameterless permutation; the 1x1 convolution is a normal
# `T.Conv2d` on top. The permutation's backward is its exact inverse, so the
# layer is a pure data movement with no approximation.
#
# At the toy scale: 2 images x 16 x 16 x 8 channels -> 2 x 8 x 8 x 32 channels.
# The channel order is the reference's: c major, then p1, then p2.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./config.bend as C

def IN() -> Type:
  TA.Mat<512n, 8n>

def OUT() -> Type:
  TA.Mat<128n, 32n>

# ---- list helpers -----------------------------------------------------------

def tk(+n: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match n xs:
    case 0n _:
      acc
    case 1n+p Con{h, t}:
      tk(p, t, Con{h, acc})
    case _ _:
      acc

def dr(+n: Nat, +xs: List<&2, F32>) -> List<&2, F32>:
  match n xs:
    case 0n _:
      xs
    case 1n+p Con{h, t}:
      dr(p, t)
    case _ _:
      Nil{}

def nth(+n: Nat, +xs: List<&2, F32>) -> F32:
  match n xs:
    case 0n Con{h, t}:
      h
    case 1n+p Con{h, t}:
      nth(p, t)
    case _ _:
      0.0

def rev(+n: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match n xs:
    case 0n _:
      acc
    case 1n+p Con{h, t}:
      rev(p, t, Con{h, acc})
    case _ _:
      acc

# ---- forward: one output block of 8 x2-rows, 32 channels each ---------------

# A block is the two input rows (y = 2*y2, 2*y2+1) at 16 x 8 = 256 values, and
# it produces the 8 output rows for x2 = 0..7, 32 channels each. The output
# channel is c*4 + p, and the input value is row p/2, x = 2*x2 + p%2, channel c.
def unsh_block(+left: Nat, +o: Nat, +blk: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      x2 = Nat.div(o, C.CH4())
      c = Nat.div(Nat.mod(o, C.CH4()), 4n)
      +q = Nat.mod(o, 4n)
      i = Nat.add(Nat.mul(Nat.div(q, 2n), 128n), Nat.add(Nat.mul(Nat.add(Nat.mul(x2, 2n), Nat.mod(q, 2n)), C.CH()), c))
      unsh_block(p, Nat.add(o, 1n), blk, Con{nth(i, blk), acc})

def unsh_list(+left: Nat, +row: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      b = Nat.div(row, C.HW2())
      y2 = Nat.mod(row, C.HW2())
      off = Nat.mul(Nat.add(Nat.mul(b, 256n), Nat.mul(y2, 32n)), C.CH())
      blk = List.reverse(&2, F32, tk(256n, dr(off, xs), Nil{}))
      out = unsh_block(256n, 0n, blk, Nil{})
      unsh_list(p, Nat.add(row, 1n), xs, List.append(&2, F32, acc, out))

def ml_list(-r: Nat, -c: Nat, ml: TA.ML<r, c>) -> List<&2, F32>:
  match ml:
    case TA.ML{mat, l}:
      l

def unsh_mat(x: IN()) -> OUT():
  TA.Mat.from_list(128n, 32n, unsh_list(16n, 0n, ml_list(512n, 8n, TA.Mat.to_list(512n, 8n, x)), Nil{}))

# ---- backward: the exact inverse -------------------------------------------

# Input index i = (p/2)*128 + x*8 + c with x = 2*x2 + p%2; the output index is
# x2*32 + c*4 + p. Every output value goes to exactly one input cell, so the
# backward is the inverse permutation, not a sum or an average.
def unsh_inv_block(+left: Nat, +i: Nat, +blk: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      row = Nat.div(i, 128n)
      +rem = Nat.mod(i, 128n)
      +x = Nat.div(rem, C.CH())
      c = Nat.mod(rem, C.CH())
      x2 = Nat.div(x, 2n)
      q = Nat.add(Nat.mul(row, 2n), Nat.mod(x, 2n))
      o = Nat.add(Nat.mul(x2, C.CH4()), Nat.add(Nat.mul(c, 4n), q))
      unsh_inv_block(p, Nat.add(i, 1n), blk, Con{nth(o, blk), acc})

def unsh_inv_list(+left: Nat, +row: Nat, +ys: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      off = Nat.mul(row, 256n)
      blk = List.reverse(&2, F32, tk(256n, dr(off, ys), Nil{}))
      out = unsh_inv_block(256n, 0n, blk, Nil{})
      unsh_inv_list(p, Nat.add(row, 1n), ys, List.append(&2, F32, acc, out))

def unsh_inv_mat(dy: OUT()) -> IN():
  TA.Mat.from_list(512n, 8n, unsh_inv_list(16n, 0n, ml_list(128n, 32n, TA.Mat.to_list(128n, 32n, dy)), Nil{}))

# ---- the layer --------------------------------------------------------------

def Down2() -> T.Layer<Unit, IN(), OUT(), Unit>:
  T.Layer{p => x => T.Fwd{p, unsh_mat(x), Unit{}},
    p => s => dy => g => T.Bwd{p, unsh_inv_mat(dy), g},
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0}

# ---- XAttnRes at the toy scale: one history entry, max-pool resize --------
#
# The reference's XAttnRes resizes every earlier stage to the current spatial
# shape, projects it to the current width, optionally adds a LoRA key, appends
# the current feature as the last value, and mixes the history entries with a
# learned pseudo-query softmax. This first block is the resize and the mix with
# a zero query (two entries, weights 0.5/0.5); the projections and the learned
# query come next.
#
# Down stage 1 of the toy: history 2x16x16x8 -> max pool 2x2 -> 2x8x8x8, and
# the current x is 2x8x8x8.

def POOL() -> Type:
  TA.Mat<128n, 8n>

def max2_if(le: Bool, a: F32, b: F32) -> F32:
  match le:
    case True{}:
      b
    case False{}:
      a

def max2(+a: F32, +b: F32) -> F32:
  max2_if(F32.is_le(a, b), a, b)

def max4(a: F32, b: F32, c: F32, d: F32) -> F32:
  max2(max2(a, b), max2(c, d))

# One output row of 8 x2 positions, 8 channels each, from the 2x16x8 block.
def pool_block(+left: Nat, +o: Nat, +blk: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      x2 = Nat.div(o, C.CH())
      c = Nat.mod(o, C.CH())
      +i0 = Nat.add(Nat.mul(Nat.mul(x2, 2n), C.CH()), c)
      +i1 = Nat.add(i0, C.CH())
      i2 = Nat.add(i0, 128n)
      i3 = Nat.add(i1, 128n)
      v = max4(nth(i0, blk), nth(i1, blk), nth(i2, blk), nth(i3, blk))
      pool_block(p, Nat.add(o, 1n), blk, Con{v, acc})

def pool_list(+left: Nat, +row: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      b = Nat.div(row, C.HW2())
      y2 = Nat.mod(row, C.HW2())
      off = Nat.mul(Nat.add(Nat.mul(b, 256n), Nat.mul(y2, 32n)), C.CH())
      blk = List.reverse(&2, F32, tk(256n, dr(off, xs), Nil{}))
      out = pool_block(64n, 0n, blk, Nil{})
      pool_list(p, Nat.add(row, 1n), xs, List.append(&2, F32, acc, out))

def pool_mat(h: IN()) -> POOL():
  TA.Mat.from_list(128n, 8n, pool_list(16n, 0n, ml_list(512n, 8n, TA.Mat.to_list(512n, 8n, h)), Nil{}))

# The zero-query mix: out = 0.5 * (pooled history + current x), per cell.
def mix_list(+left: Nat, +xs: List<&2, F32>, +ys: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left xs ys:
    case 0n _ _:
      acc
    case 1n+p Con{a, at} Con{b, bt}:
      mix_list(p, at, bt, Con{((a + b : F32) * 0.5 : F32), acc})
    case _ _ _:
      acc

def mix_mat(x: POOL(), h: POOL()) -> POOL():
  xs = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, x))
  ys = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, h))
  TA.Mat.from_list(128n, 8n, List.reverse(&2, F32, mix_list(1024n, xs, ys, Nil{})))
