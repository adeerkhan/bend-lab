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
      +off = Nat.mul(Nat.add(Nat.mul(b, 256n), Nat.mul(y2, 32n)), C.CH())
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
      +i2 = Nat.add(i0, 128n)
      +i3 = Nat.add(i1, 128n)
      v = max4(nth(i0, blk), nth(i1, blk), nth(i2, blk), nth(i3, blk))
      pool_block(p, Nat.add(o, 1n), blk, Con{v, acc})

def pool_list(+left: Nat, +row: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      b = Nat.div(row, C.HW2())
      y2 = Nat.mod(row, C.HW2())
      +off = Nat.mul(Nat.add(Nat.mul(b, 256n), Nat.mul(y2, 32n)), C.CH())
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

# ---- the same 2x2 max pool as a layer: forward saves the arg-max ----------
#
# The reference's XAttnRes uses adaptive_max_pool2d on the down path. Its
# backward is not an average: the whole gradient of an output cell goes to the
# one input cell that was the maximum. The forward saves that choice.

type PS is Type:
  PS{idx: List<&2, U32>}

def nth_u(+n: Nat, +xs: List<&2, U32>) -> U32:
  match n xs:
    case 0n Con{h, t}:
      h
    case 1n+p Con{h, t}:
      nth_u(p, t)
    case _ _:
      U32.from_nat(0n)

def tk_u(+n: Nat, +xs: List<&2, U32>, +acc: List<&2, U32>) -> List<&2, U32>:
  match n xs:
    case 0n _:
      acc
    case 1n+p Con{h, t}:
      tk_u(p, t, Con{h, acc})
    case _ _:
      acc

def dr_u(+n: Nat, +xs: List<&2, U32>) -> List<&2, U32>:
  match n xs:
    case 0n _:
      xs
    case 1n+p Con{h, t}:
      dr_u(p, t)
    case _ _:
      Nil{}

def pick_if(ok: Bool, +a: F32, +b: F32) -> F32:
  match ok:
    case True{}:
      a
    case False{}:
      b

def m2i(le: Bool, +a: F32, +b: F32, +ia: U32, +ib: U32) -> T.Two<F32, U32>:
  match le:
    case True{}:
      T.Two{b, ib}
    case False{}:
      T.Two{a, ia}

def m2i_pairs(p: T.Two<F32, U32>, q: T.Two<F32, U32>) -> T.Two<F32, U32>:
  match p q:
    case T.Two{+pa, pi} T.Two{+qa, qi}:
      m2i(F32.is_le(pa, qa), pa, qa, pi, qi)

def tw_fst(a: T.Two<F32, U32>) -> F32:
  match a:
    case T.Two{x, y}:
      x

def tw_snd(a: T.Two<F32, U32>) -> U32:
  match a:
    case T.Two{x, y}:
      y

def m4i(+a: F32, +b: F32, +c: F32, +d: F32) -> T.Two<F32, U32>:
  p = m2i(F32.is_le(a, b), a, b, U32.from_nat(0n), U32.from_nat(1n))
  q = m2i(F32.is_le(c, d), c, d, U32.from_nat(2n), U32.from_nat(3n))
  m2i_pairs(p, q)

def m4_val(+a: F32, +b: F32, +c: F32, +d: F32) -> F32:
  tw_fst(m4i(a, b, c, d))

def m4_idx(+a: F32, +b: F32, +c: F32, +d: F32) -> U32:
  tw_snd(m4i(a, b, c, d))

def tl_fst(a: T.Two<List<&2, F32>, List<&2, U32>>) -> List<&2, F32>:
  match a:
    case T.Two{x, y}:
      x

def tl_snd(a: T.Two<List<&2, F32>, List<&2, U32>>) -> List<&2, U32>:
  match a:
    case T.Two{x, y}:
      y

def pool_block_idx(+left: Nat, +o: Nat, +blk: List<&2, F32>, +acc: List<&2, F32>, +idx: List<&2, U32>) -> T.Two<List<&2, F32>, List<&2, U32>>:
  match left:
    case 0n:
      T.Two{List.reverse(&2, F32, acc), List.reverse(&2, U32, idx)}
    case 1n+p:
      x2 = Nat.div(o, C.CH())
      c = Nat.mod(o, C.CH())
      +i0 = Nat.add(Nat.mul(Nat.mul(x2, 2n), C.CH()), c)
      +i1 = Nat.add(i0, C.CH())
      +i2 = Nat.add(i0, 128n)
      +i3 = Nat.add(i1, 128n)
      pool_block_idx(p, Nat.add(o, 1n), blk, Con{m4_val(nth(i0, blk), nth(i1, blk), nth(i2, blk), nth(i3, blk)), acc}, Con{m4_idx(nth(i0, blk), nth(i1, blk), nth(i2, blk), nth(i3, blk)), idx})

def pool_list_idx(+left: Nat, +row: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>, +idx: List<&2, U32>) -> T.Two<List<&2, F32>, List<&2, U32>>:
  match left:
    case 0n:
      T.Two{acc, idx}
    case 1n+p:
      b = Nat.div(row, C.HW2())
      y2 = Nat.mod(row, C.HW2())
      +off = Nat.mul(Nat.add(Nat.mul(b, 256n), Nat.mul(y2, 32n)), C.CH())
      blk = List.reverse(&2, F32, tk(256n, dr(off, xs), Nil{}))
      bk2 = List.reverse(&2, F32, tk(256n, dr(off, xs), Nil{}))
      out = tl_fst(pool_block_idx(64n, 0n, blk, Nil{}, Nil{}))
      ii = tl_snd(pool_block_idx(64n, 0n, bk2, Nil{}, Nil{}))
      pool_list_idx(p, Nat.add(row, 1n), xs, List.append(&2, F32, acc, out), List.append(&2, U32, idx, ii))

def pool_mat_of(r: T.Two<List<&2, F32>, List<&2, U32>>) -> T.Two<POOL(), List<&2, U32>>:
  match r:
    case T.Two{ys, idx}:
      T.Two{TA.Mat.from_list(128n, 8n, ys), idx}

def pool_mat_idx(x: IN()) -> T.Two<POOL(), List<&2, U32>>:
  pool_mat_of(pool_list_idx(16n, 0n, ml_list(512n, 8n, TA.Mat.to_list(512n, 8n, x)), Nil{}, Nil{}))

def pool_fwd_of(p: T.Two<POOL(), List<&2, U32>>) -> T.Fwd<Unit, POOL(), PS>:
  match p:
    case T.Two{y, idx}:
      T.Fwd{Unit{}, y, PS{idx}}

def pool_fwd(x: IN()) -> T.Fwd<Unit, POOL(), PS>:
  pool_fwd_of(pool_mat_idx(x))

# ---- backward: scatter each output gradient to the saved arg-max ----------

def pool_inv_block(+left: Nat, +i: Nat, +idx: List<&2, U32>, +dy: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +row = Nat.div(i, 128n)
      +rem = Nat.mod(i, 128n)
      +x = Nat.div(rem, C.CH())
      c = Nat.mod(rem, C.CH())
      x2 = Nat.div(x, 2n)
      q = Nat.add(Nat.mul(row, 2n), Nat.mod(x, 2n))
      +cell = Nat.add(Nat.mul(x2, C.CH()), c)
      k = nth_u(cell, idx)
      v = pick_if(U32.is_eq(k, U32.from_nat(q)), nth(cell, dy), 0.0)
      pool_inv_block(p, Nat.add(i, 1n), idx, dy, Con{v, acc})

def pool_inv_list(+left: Nat, +row: Nat, +idx: List<&2, U32>, +dy: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ioff = Nat.mul(row, 64n)
      dblk = List.reverse(&2, F32, tk(64n, dr(Nat.mul(row, 64n), dy), Nil{}))
      iblk = tk_u(64n, dr_u(ioff, idx), Nil{})
      out = pool_inv_block(256n, 0n, iblk, dblk, Nil{})
      pool_inv_list(p, Nat.add(row, 1n), idx, dy, List.append(&2, F32, acc, out))

def pool_inv_mat(dy: POOL(), idx: List<&2, U32>) -> IN():
  ys = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, dy))
  TA.Mat.from_list(512n, 8n, pool_inv_list(16n, 0n, idx, ys, Nil{}))

def pool_bwd_of(dy: POOL(), idx: List<&2, U32>, g: Unit) -> T.Bwd<Unit, IN()>:
  T.Bwd{Unit{}, pool_inv_mat(dy, idx), g}

def pool_bwd(dy: POOL(), s: PS, g: Unit) -> T.Bwd<Unit, IN()>:
  match s:
    case PS{idx}:
      pool_bwd_of(dy, idx, g)

def Pool2() -> T.Layer<Unit, IN(), POOL(), PS>:
  T.Layer{p => x => pool_fwd(x),
    p => s => dy => g => pool_bwd(dy, s, g),
    op => p => g => st => T.no_step(Unit, op, p, g, st),
    u => Unit{},
    p => T.PL{p, Nil{}},
    p => xs => off => T.PA{p, xs, off},
    p => xs => off => T.PA{p, xs, off},
    u => 0}

# ---- learned-query mix, forward -------------------------------------------
#
# The reference's XAttnRes does not average the history blindly: it scores each
# entry with a learned pseudo-query and a softmax. At the toy scale the entries
# are (pooled history value, current x), and the keys are the normalized values.
# This block is the learned query forward and its backward; the backward
# below uses the saved softmax weights and the same dot order.

type MXS is Type:
  MXS{vh: TA.Mat<128n, 8n>, xc: TA.Mat<128n, 8n>, kh: TA.Mat<128n, 8n>, kx: TA.Mat<128n, 8n>, q: TA.Mat<1n, 8n>}

def scale8() -> F32:
  0.35355338

# q is the 1 x 8 query: it is read at 0..7, whatever the row. Only the key and
# value rows move with `off` (the old version indexed q at off + k, so every
# row past the first read past its end and scored 0).
def dot8(+off: Nat, +q: List<&2, F32>, +xs: List<&2, F32>) -> F32:
  s0 = (nth(0n, q) * nth(off, xs) : F32)
  +s1 = (s0 + nth(1n, q) * nth(Nat.add(off, 1n), xs) : F32)
  +s2 = (s1 + nth(2n, q) * nth(Nat.add(off, 2n), xs) : F32)
  +s3 = (s2 + nth(3n, q) * nth(Nat.add(off, 3n), xs) : F32)
  +s4 = (s3 + nth(4n, q) * nth(Nat.add(off, 4n), xs) : F32)
  +s5 = (s4 + nth(5n, q) * nth(Nat.add(off, 5n), xs) : F32)
  +s6 = (s5 + nth(6n, q) * nth(Nat.add(off, 6n), xs) : F32)
  (s6 + nth(7n, q) * nth(Nat.add(off, 7n), xs) : F32)

def logits_list(+left: Nat, +row: Nat, +kh: List<&2, F32>, +kx: List<&2, F32>, +q: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +off = Nat.mul(row, 8n)
      lh = (dot8(off, q, kh) * scale8() : F32)
      lx = (dot8(off, q, kx) * scale8() : F32)
      logits_list(p, Nat.add(row, 1n), kh, kx, q, Con{lx, Con{lh, acc}})

def mix_out_list(+left: Nat, +row: Nat, +vh: List<&2, F32>, +xc: List<&2, F32>, +w: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +off = Nat.mul(row, 8n)
      +woff = Nat.mul(row, 2n)
      +wh = nth(woff, w)
      +wx = nth(Nat.add(woff, 1n), w)
      v0 = (wh * nth(off, vh) + wx * nth(off, xc) : F32)
      v1 = (wh * nth(Nat.add(off, 1n), vh) + wx * nth(Nat.add(off, 1n), xc) : F32)
      v2 = (wh * nth(Nat.add(off, 2n), vh) + wx * nth(Nat.add(off, 2n), xc) : F32)
      v3 = (wh * nth(Nat.add(off, 3n), vh) + wx * nth(Nat.add(off, 3n), xc) : F32)
      v4 = (wh * nth(Nat.add(off, 4n), vh) + wx * nth(Nat.add(off, 4n), xc) : F32)
      v5 = (wh * nth(Nat.add(off, 5n), vh) + wx * nth(Nat.add(off, 5n), xc) : F32)
      v6 = (wh * nth(Nat.add(off, 6n), vh) + wx * nth(Nat.add(off, 6n), xc) : F32)
      v7 = (wh * nth(Nat.add(off, 7n), vh) + wx * nth(Nat.add(off, 7n), xc) : F32)
      mix_out_list(p, Nat.add(row, 1n), vh, xc, w, Con{v7, Con{v6, Con{v5, Con{v4, Con{v3, Con{v2, Con{v1, Con{v0, acc}}}}}}}})

def mix_sm(r: T.Fwd<Unit, TA.Mat<128n, 2n>, TA.Mat<128n, 2n>>, +vh: List<&2, F32>, +xc: List<&2, F32>) -> TA.Mat<128n, 8n>:
  match r:
    case T.Fwd{_, w, _}:
      +wl = ml_list(128n, 2n, TA.Mat.to_list(128n, 2n, w))
      TA.Mat.from_list(128n, 8n, mix_out_list(128n, 0n, vh, xc, wl, Nil{}))

def mix_forward(x: MXS) -> TA.Mat<128n, 8n>:
  match x:
    case MXS{vh, xc, kh, kx, q}:
      +vhl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, vh))
      +xcl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, xc))
      +khl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, kh))
      +kxl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, kx))
      +ql = ml_list(1n, 8n, TA.Mat.to_list(1n, 8n, q))
      logits = TA.Mat.from_list(128n, 2n, logits_list(128n, 0n, khl, kxl, ql, Nil{}))
      mix_sm(T.Layer.fwd(Unit, TA.Mat<128n, 2n>, TA.Mat<128n, 2n>, TA.Mat<128n, 2n>, T.Softmax(128n, 2n), Unit{}, logits), vhl, xcl)

# ---- learned-query mix, backward -------------------------------------------
#
# The gradient of one row, with a = kh.q, b = kx.q and w = softmax([a, b]):
#
#   dvh  = wh * dout              dxc  = wx * dout
#   dwh  = dout . vh              dwx  = dout . xc
#   s    = wh * dwh + wx * dwx    (the softmax's own sum)
#   da   = wh * (dwh - s)         db   = wx * (dwx - s)
#   dq   = sum_r (da*scale) * kh[r] + (db*scale) * kx[r]
#   dkh  = (da*scale) outer q     dkx  = (db*scale) outer q
#
# The softmax derivative is BendTorch's own `Softmax` backward on the saved
# weights, so the forward and the backward use one definition. `w` is cloned
# through its list: the matrix is linear, the list is read-only.

type MB is Type:
  MB{dvh: TA.Mat<128n, 8n>, dxc: TA.Mat<128n, 8n>, dkh: TA.Mat<128n, 8n>, dkx: TA.Mat<128n, 8n>, dq: TA.Mat<1n, 8n>}

def logits_mat(kh: TA.Mat<128n, 8n>, kx: TA.Mat<128n, 8n>, q: TA.Mat<1n, 8n>) -> TA.Mat<128n, 2n>:
  +khl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, kh))
  +kxl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, kx))
  +ql = ml_list(1n, 8n, TA.Mat.to_list(1n, 8n, q))
  TA.Mat.from_list(128n, 2n, logits_list(128n, 0n, khl, kxl, ql, Nil{}))

def w_of(r: T.Fwd<Unit, TA.Mat<128n, 2n>, TA.Mat<128n, 2n>>) -> TA.Mat<128n, 2n>:
  match r:
    case T.Fwd{_, w, _}:
      w

def mix_weights(kh: TA.Mat<128n, 8n>, kx: TA.Mat<128n, 8n>, q: TA.Mat<1n, 8n>) -> TA.Mat<128n, 2n>:
  w_of(T.Layer.fwd(Unit, TA.Mat<128n, 2n>, TA.Mat<128n, 2n>, TA.Mat<128n, 2n>,
    T.Softmax(128n, 2n), Unit{}, logits_mat(kh, kx, q)))

def dx_of(b: T.Bwd<Unit, TA.Mat<128n, 2n>>) -> TA.Mat<128n, 2n>:
  match b:
    case T.Bwd{_, dx, _}:
      dx

def sm_bwd_mat(s: TA.Mat<128n, 2n>, dy: TA.Mat<128n, 2n>) -> TA.Mat<128n, 2n>:
  dx_of(T.Layer.bwd(Unit, TA.Mat<128n, 2n>, TA.Mat<128n, 2n>, TA.Mat<128n, 2n>,
    T.Softmax(128n, 2n), Unit{}, s, dy, Unit{}))

# both operands at off..off + 7 (a row dot)
def row8(+off: Nat, +xs: List<&2, F32>, +ys: List<&2, F32>) -> F32:
  s0 = (nth(off, xs) * nth(off, ys) : F32)
  +s1 = (s0 + nth(Nat.add(off, 1n), xs) * nth(Nat.add(off, 1n), ys) : F32)
  +s2 = (s1 + nth(Nat.add(off, 2n), xs) * nth(Nat.add(off, 2n), ys) : F32)
  +s3 = (s2 + nth(Nat.add(off, 3n), xs) * nth(Nat.add(off, 3n), ys) : F32)
  +s4 = (s3 + nth(Nat.add(off, 4n), xs) * nth(Nat.add(off, 4n), ys) : F32)
  +s5 = (s4 + nth(Nat.add(off, 5n), xs) * nth(Nat.add(off, 5n), ys) : F32)
  +s6 = (s5 + nth(Nat.add(off, 6n), xs) * nth(Nat.add(off, 6n), ys) : F32)
  (s6 + nth(Nat.add(off, 7n), xs) * nth(Nat.add(off, 7n), ys) : F32)

# one source row scaled by w[2*row + col]: dvh (col 0) and dxc (col 1)
def weighted_rows(+left: Nat, +row: Nat, +src: List<&2, F32>, +w: List<&2, F32>, +col: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +off = Nat.mul(row, 8n)
      +f = nth(Nat.add(Nat.mul(row, 2n), col), w)
      v0 = (f * nth(off, src) : F32)
      v1 = (f * nth(Nat.add(off, 1n), src) : F32)
      v2 = (f * nth(Nat.add(off, 2n), src) : F32)
      v3 = (f * nth(Nat.add(off, 3n), src) : F32)
      v4 = (f * nth(Nat.add(off, 4n), src) : F32)
      v5 = (f * nth(Nat.add(off, 5n), src) : F32)
      v6 = (f * nth(Nat.add(off, 6n), src) : F32)
      v7 = (f * nth(Nat.add(off, 7n), src) : F32)
      weighted_rows(p, Nat.add(row, 1n), src, w, col, Con{v7, Con{v6, Con{v5, Con{v4, Con{v3, Con{v2, Con{v1, Con{v0, acc}}}}}}}})

# w's derivative enters the softmax as (dout . vh, dout . xc) per row
def dw_list(+left: Nat, +row: Nat, +vh: List<&2, F32>, +xc: List<&2, F32>, +dout: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +off = Nat.mul(row, 8n)
      hw = row8(off, dout, vh)
      xw = row8(off, dout, xc)
      dw_list(p, Nat.add(row, 1n), vh, xc, dout, Con{xw, Con{hw, acc}})

# dlogit * scale8, 256 values in row-major order
def scale_all(+left: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n _:
      List.reverse(&2, F32, acc)
    case 1n+p Con{h, t}:
      scale_all(p, t, Con{(h * scale8() : F32), acc})
    case _ _:
      acc

# every second value, starting at off: the two softmax logits split apart
def pick_list(+left: Nat, +i: Nat, +off: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      pick_list(p, Nat.add(i, 1n), off, xs, Con{nth(Nat.add(Nat.mul(i, 2n), off), xs), acc})

# one row of dkh (col 0) or dkx (col 1): d[r] outer q
def outer_list(+left: Nat, +row: Nat, +q: List<&2, F32>, +d: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +dr = nth(row, d)
      v0 = (dr * nth(0n, q) : F32)
      v1 = (dr * nth(1n, q) : F32)
      v2 = (dr * nth(2n, q) : F32)
      v3 = (dr * nth(3n, q) : F32)
      v4 = (dr * nth(4n, q) : F32)
      v5 = (dr * nth(5n, q) : F32)
      v6 = (dr * nth(6n, q) : F32)
      v7 = (dr * nth(7n, q) : F32)
      outer_list(p, Nat.add(row, 1n), q, d, Con{v7, Con{v6, Con{v5, Con{v4, Con{v3, Con{v2, Con{v1, Con{v0, acc}}}}}}}})

def dq_sum(+left: Nat, +row: Nat, +k: Nat, +kh: List<&2, F32>, +kx: List<&2, F32>, +dda: List<&2, F32>, +ddb: List<&2, F32>, acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      +off = Nat.add(Nat.mul(row, 8n), k)
      +a = nth(off, kh)
      +b = nth(off, kx)
      +da = nth(row, dda)
      +db = nth(row, ddb)
      dq_sum(p, Nat.add(row, 1n), k, kh, kx, dda, ddb,
        ((acc + (da * a : F32) : F32) + (db * b : F32) : F32))

def dq_list(+left: Nat, +k: Nat, +kh: List<&2, F32>, +kx: List<&2, F32>, +dda: List<&2, F32>, +ddb: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      dq_list(p, Nat.add(k, 1n), kh, kx, dda, ddb,
        Con{dq_sum(128n, 0n, k, kh, kx, dda, ddb, 0.0), acc})

def mix_backward(vh: TA.Mat<128n, 8n>, xc: TA.Mat<128n, 8n>, kh: TA.Mat<128n, 8n>, kx: TA.Mat<128n, 8n>, q: TA.Mat<1n, 8n>, w: TA.Mat<128n, 2n>, dout: TA.Mat<128n, 8n>) -> MB:
  +vhl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, vh))
  +xcl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, xc))
  +khl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, kh))
  +kxl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, kx))
  +ql = ml_list(1n, 8n, TA.Mat.to_list(1n, 8n, q))
  +wl = ml_list(128n, 2n, TA.Mat.to_list(128n, 2n, w))
  +doutl = ml_list(128n, 8n, TA.Mat.to_list(128n, 8n, dout))
  +dwl = dw_list(128n, 0n, vhl, xcl, doutl, Nil{})
  dw = TA.Mat.from_list(128n, 2n, dwl)
  w2 = TA.Mat.from_list(128n, 2n, wl)
  dlogits = sm_bwd_mat(w2, dw)
  dl = ml_list(128n, 2n, TA.Mat.to_list(128n, 2n, dlogits))
  +ds = scale_all(256n, dl, Nil{})
  +dda = pick_list(128n, 0n, 0n, ds, Nil{})
  +ddb = pick_list(128n, 0n, 1n, ds, Nil{})
  MB{TA.Mat.from_list(128n, 8n, weighted_rows(128n, 0n, doutl, wl, 0n, Nil{})),
    TA.Mat.from_list(128n, 8n, weighted_rows(128n, 0n, xcl, wl, 1n, Nil{})),
    TA.Mat.from_list(128n, 8n, outer_list(128n, 0n, ql, dda, Nil{})),
    TA.Mat.from_list(128n, 8n, outer_list(128n, 0n, ql, ddb, Nil{})),
    TA.Mat.from_list(1n, 8n, dq_list(8n, 0n, khl, kxl, dda, ddb, Nil{}))}
