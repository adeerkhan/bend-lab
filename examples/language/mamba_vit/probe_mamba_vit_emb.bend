# MambaViT A1b: the patch embedding, the eight Mamba-1 blocks, the final
# LayerNorm, the mean pool and the head, from the reference dump.
#
# A1a is `mv_emb` and `mv_tokens` (gated in commit 00c22aa); A1b adds
# `mv_block0..7`, `mv_block0..7_res`, `mv_final`, `mv_pooled` and `mv_logits`.
# Shapes: 50 tokens (1 cls + 49 patches), dim 32, d_inner 64, dt_rank 2,
# d_state 16, d_conv 4, 8 blocks, 10 classes.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../mamba3/mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/mamba_vit"

def ml28(ml: TA.ML<28n, 28n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1x16(ml: TA.ML<1n, 16n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml32x16(ml: TA.ML<32n, 16n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1x32(ml: TA.ML<1n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml50x32(ml: TA.ML<50n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml128x32(ml: TA.ML<128n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml64x4(ml: TA.ML<64n, 4n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1x64(ml: TA.ML<1n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml34x64(ml: TA.ML<34n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml64x2(ml: TA.ML<64n, 2n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml64x16(ml: TA.ML<64n, 16n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml32x64(ml: TA.ML<32n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml10x32(ml: TA.ML<10n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1x10(ml: TA.ML<1n, 10n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def sum_go(+left: Nat, +xs: List<&2, F32>, acc: F32) -> F32:
  match left xs:
    case 0n _:
      acc
    case _ Nil{}:
      acc
    case 1n+p Con{h, t}:
      sum_go(p, t, (acc + h : F32))

def sumsq_go(+left: Nat, +xs: List<&2, F32>, acc: F32) -> F32:
  match left xs:
    case 0n _:
      acc
    case _ Nil{}:
      acc
    case 1n+p Con{h, t}:
      sumsq_go(p, t, (acc + (h * h : F32) : F32))

def fill_go(+left: Nat, +v: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      fill_go(p, v, Con{v, acc})

def ln_row(+xs: List<&2, F32>, +ws: List<&2, F32>,
    +bs: List<&2, F32>) -> List<&2, F32>:
  +n = List.length(&2, F32, xs)
  +nf = F32.from_nat(n)
  +mean = (sum_go(n, xs, 0.0) / nf : F32)
  +cent = M.sub_lists(xs, fill_go(n, mean, Nil{}))
  +var = (sumsq_go(n, cent, 0.0) / nf : F32)
  +sc = (1.0 / F32.sqrt((var + 0.00001 : F32)) : F32)
  M.add_lists(M.mul_lists(M.scale_list(cent, sc), ws), bs)

def row(+t: Nat, +xs: List<&2, F32>, +w: Nat) -> List<&2, F32>:
  M.take_row(xs, Nat.mul(t, w), w)

def ln_rows(+left: Nat, +t: Nat, +xs: List<&2, F32>,
    +ws: List<&2, F32>, +bs: List<&2, F32>, +w: Nat,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ln_rows(p, (t + 1n : Nat), xs, ws, bs, w,
        M.push_row(acc, ln_row(row(t, xs, w), ws, bs)))

def patch(+i: Nat, +j: Nat, +img: List<&2, F32>) -> List<&2, F32>:
  +b = Nat.add(Nat.mul(Nat.mul(i, 4n), 28n), Nat.mul(j, 4n))
  List.append(&2, F32, M.take_row(img, b, 4n),
    List.append(&2, F32, M.take_row(img, (b + 28n : Nat), 4n),
      List.append(&2, F32, M.take_row(img, (b + 56n : Nat), 4n),
        M.take_row(img, (b + 84n : Nat), 4n))))

def patches_go(+left: Nat, +p: Nat, +img: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+q:
      patches_go(q, (p + 1n : Nat), img,
        M.push_row(acc, patch(Nat.div(p, 7n),
          Nat.sub(p, Nat.mul(Nat.div(p, 7n), 7n)), img)))

def emb_go(+left: Nat, +t: Nat, +pa: List<&2, F32>,
    +w0: List<&2, F32>, +b0: List<&2, F32>,
    +w1: List<&2, F32>, +b1: List<&2, F32>,
    +w2: List<&2, F32>, +b2: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +p1 = ln_row(row(t, pa, 16n), w0, b0)
      +p2 = ln_row(M.matvec(w1, p1, 16n), w2, b2)
      emb_go(p, (t + 1n : Nat), pa, w0, b0, w1, b1, w2, b2,
        M.push_row(acc, p2))

def add_pos(+left: Nat, +t: Nat, +xs: List<&2, F32>,
    +pos: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      add_pos(p, (t + 1n : Nat), xs, pos,
        M.push_row(acc, M.add_lists(row(t, xs, 32n), row(t, pos, 32n))))

type EMP is Type:
  EMP{emb: List<&2, F32>, toks: List<&2, F32>}

def emb_all(+img: List<&2, F32>, +w0: List<&2, F32>,
    +b0: List<&2, F32>, +w1: List<&2, F32>,
    +b1: List<&2, F32>, +w2: List<&2, F32>,
    +b2: List<&2, F32>, +cls: List<&2, F32>,
    +pos: List<&2, F32>) -> EMP:
  +pa = patches_go(49n, 0n, img, Nil{})
  +emb = emb_go(49n, 0n, pa, w0, b0, w1, b1, w2, b2, Nil{})
  EMP{emb, add_pos(50n, 0n, List.append(&2, F32, cls, emb), pos,
    Nil{})}

def show_emp(e: EMP) -> IO(List<&2, F32>):
  match e:
    case EMP{emb, +toks}:
      do IO<List<&2, F32>>:
        M.show_tag("mv_emb", emb)
        M.show_tag("mv_tokens", toks)
        return toks

# ---------------------------------------------------------------------------
# the Mamba-1 block, one pass per stage: each pass folds a whole batch of
# rows into one flat list, so no stage ever has to hand back two values
# (a computed record or pair can only be taken apart by a def parameter).

# the block's input: the running sum, `x + res`, or `x` on the first block
def res2_of(+x: List<&2, F32>, +res: List<&2, F32>) -> List<&2, F32>:
  match res:
    case Nil{}:
      x
    case Con{h, t}:
      M.add_lists(x, Con{h, t})

# A = -exp(A_log), elementwise
def negexp_row(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(0.0 - F32.exp(h) : F32), negexp_row(t)}

# acc + a*b, elementwise: one fused pass for a whole row
def mad(+acc: List<&2, F32>, +a: List<&2, F32>,
    +b: List<&2, F32>) -> List<&2, F32>:
  match acc a b:
    case Nil{} _ _:
      Nil{}
    case _ Nil{} _:
      Nil{}
    case _ _ Nil{}:
      Nil{}
    case Con{ch, ct} Con{ah, at} Con{bh, bt}:
      Con{((ch + (ah * bh : F32)) : F32), mad(ct, at, bt)}

# one column of the depthwise conv weight (64x4 row-major, c*4+k)
def col_go(+left: Nat, +j: Nat, +cw: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      col_go(p, j, List.drop(&2, F32, cw, 4n),
        M.push_row(acc, M.take_row(cw, j, 1n)))

# y[t,c] = bias[c] + sum_k w[c,k] * xp[t+k,c], with three zero tokens in
# front: y[t] = bias + c3*xr[t] + c2*xr[t-1] + c1*xr[t-2] + c0*xr[t-3]
def conv_row(+xr: List<&2, F32>, +w1: List<&2, F32>,
    +w2: List<&2, F32>, +w3: List<&2, F32>,
    +c0: List<&2, F32>, +c1: List<&2, F32>,
    +c2: List<&2, F32>, +c3: List<&2, F32>,
    +cb: List<&2, F32>) -> List<&2, F32>:
  mad(mad(mad(mad(cb, c3, xr), c2, w1), c1, w2), c0, w3)

type CVW is Type:
  CVW{w1: List<&2, F32>, w2: List<&2, F32>, w3: List<&2, F32>,
      xs: List<&2, F32>}

# pass A: LayerNorm then in_proj (128 wide: x in [0,64), z in [64,128)), and
# the causal conv over the 50 x-rows, carrying the last three rows
def inproj_go(+left: Nat, +t: Nat, +res2: List<&2, F32>,
    +nw: List<&2, F32>, +nb: List<&2, F32>,
    +ip: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +hz = ln_row(row(t, res2, 32n), nw, nb)
      inproj_go(p, (t + 1n : Nat), res2, nw, nb, ip,
        M.push_row(acc, M.matvec(ip, hz, 32n)))

def conv_go(+left: Nat, +t: Nat, +xz: List<&2, F32>,
    +c0: List<&2, F32>, +c1: List<&2, F32>,
    +c2: List<&2, F32>, +c3: List<&2, F32>,
    +cb: List<&2, F32>, cv: CVW) -> List<&2, F32>:
  match left cv:
    case 0n CVW{_, _, _, xs}:
      xs
    case 1n+p CVW{+w1, +w2, w3, xs}:
      +xr = M.take_row(xz, Nat.mul(t, 128n), 64n)
      conv_go(p, (t + 1n : Nat), xz, c0, c1, c2, c3, cb,
        CVW{xr, w1, w2,
          M.push_row(xs, conv_row(xr, w1, w2, w3, c0, c1, c2, c3, cb))})

# dt[c] = softplus(dot(dt_proj_w[c], dt_raw) + dt_proj_b[c] + dt_bias[c])
def dt_go(+left: Nat, +w: List<&2, F32>, +b: List<&2, F32>,
    +tb: List<&2, F32>, +raw: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left b tb:
    case 0n _ _:
      acc
    case 1n+p Con{bh, bt} Con{th, tt}:
      +v = (M.dot(M.take_row(w, 0n, 2n), raw) + bh : F32)
      dt_go(p, List.drop(&2, F32, w, 2n), bt, tt, raw,
        M.push_row(acc, Con{M.softplus((v + th : F32)), Nil{}}))
    case _ _ _:
      acc

# the 16 state slots of one channel: h' = exp(dt*A) h + dt*B*x
def upd_q(+left: Nat, +arow: List<&2, F32>, +hrow: List<&2, F32>,
    +brow: List<&2, F32>, +dtc: F32, +xc: F32,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left arow hrow brow:
    case 0n _ _ _:
      acc
    case 1n+p Con{ah, at} Con{hh, ht} Con{bh, bt}:
      +d = (dtc * bh : F32)
      +nv = ((F32.exp((dtc * ah : F32)) * hh : F32) + (d * xc : F32) : F32)
      upd_q(p, at, ht, bt, dtc, xc, Con{nv, acc})
    case _ _ _ _:
      acc

# pass B: the 64x16 state, one channel row at a time.
#
# `upd_q` builds one channel's 16 slots backwards, so it is PREPENDED as is
# and the single reverse at the end puts every row in slot order AND the rows
# in channel order. Reversing the row here as well cancels across rows but
# reverses the 16 slots inside a row; that layout error is invisible in the
# forward (the state is ~1e-7 of y) and fails the backward's dC, dA and d(dt).
def upd_c(+left: Nat, +A: List<&2, F32>, +h: List<&2, F32>,
    +dtr: List<&2, F32>, +brow: List<&2, F32>,
    +xr: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left dtr xr:
    case 0n _ _:
      List.reverse(&2, F32, acc)
    case 1n+p Con{dh, dt} Con{xh, xt}:
      +arow = M.take_row(A, 0n, 16n)
      +hrow = M.take_row(h, 0n, 16n)
      upd_c(p, List.drop(&2, F32, A, 16n), List.drop(&2, F32, h, 16n),
        dt, brow, xt,
        List.append(&2, F32,
          upd_q(16n, arow, hrow, brow, dh, xh, Nil{}), acc))
    case _ _ _:
      List.reverse(&2, F32, acc)

# y[c] = sum_q h[c,q]*C[q] + D[c]*x[c]
def dotc(+left: Nat, +h: List<&2, F32>, +C: List<&2, F32>,
    +D: List<&2, F32>, +xr: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left D xr:
    case 0n _ _:
      acc
    case 1n+p Con{dh, dt} Con{xh, xt}:
      +hrow = M.take_row(h, 0n, 16n)
      dotc(p, List.drop(&2, F32, h, 16n), C, dt, xt,
        Con{(M.dot(hrow, C) + (dh * xh : F32) : F32), acc})
    case _ _ _:
      acc

type BSW is Type:
  BSW{h: List<&2, F32>, out: List<&2, F32>}

# pass C, one token: B/C and dt from x_proj, the scan, the SiLU gate, out_proj
def blk_step(+t: Nat, +xz: List<&2, F32>, +cv: List<&2, F32>,
    +xpw: List<&2, F32>, +wdt: List<&2, F32>,
    +bdt: List<&2, F32>, +dtb: List<&2, F32>,
    +A: List<&2, F32>, +D: List<&2, F32>,
    +wop: List<&2, F32>, st: BSW) -> BSW:
  match st:
    case BSW{h, outb}:
      +xr = M.silu_row(M.take_row(cv, Nat.mul(t, 64n), 64n))
      +zr = M.take_row(xz, Nat.add(Nat.mul(t, 128n), 64n), 64n)
      +dbl = M.matvec(xpw, xr, 64n)
      +dtr = dt_go(64n, wdt, bdt, dtb, M.take_row(dbl, 0n, 2n), Nil{})
      +h2 = upd_c(64n, A, h, dtr, M.take_row(dbl, 2n, 16n), xr, Nil{})
      +yr = List.reverse(&2, F32,
        dotc(64n, h2, M.take_row(dbl, 18n, 16n), D, xr, Nil{}))
      +yg = M.mul_lists(yr, M.silu_row(zr))
      BSW{h2, M.push_row(outb, M.matvec(wop, yg, 64n))}

def blk_go(+left: Nat, +t: Nat, +xz: List<&2, F32>,
    +cv: List<&2, F32>, +xpw: List<&2, F32>,
    +wdt: List<&2, F32>, +bdt: List<&2, F32>,
    +dtb: List<&2, F32>, +A: List<&2, F32>,
    +D: List<&2, F32>, +wop: List<&2, F32>,
    st: BSW) -> List<&2, F32>:
  match left st:
    case 0n BSW{_, outb}:
      outb
    case 1n+p BSW{h, outb}:
      blk_go(p, (t + 1n : Nat), xz, cv, xpw, wdt, bdt, dtb, A, D, wop,
        blk_step(t, xz, cv, xpw, wdt, bdt, dtb, A, D, wop, BSW{h, outb}))

def blk_out(+res2: List<&2, F32>, +nw: List<&2, F32>,
    +nb: List<&2, F32>, +ip: List<&2, F32>,
    +c0: List<&2, F32>, +c1: List<&2, F32>,
    +c2: List<&2, F32>, +c3: List<&2, F32>,
    +cb: List<&2, F32>, +xpw: List<&2, F32>,
    +wdt: List<&2, F32>, +bdt: List<&2, F32>,
    +dtb: List<&2, F32>, +al: List<&2, F32>,
    +D: List<&2, F32>, +wop: List<&2, F32>) -> List<&2, F32>:
  +xz = inproj_go(50n, 0n, res2, nw, nb, ip, Nil{})
  +cv = conv_go(50n, 0n, xz, c0, c1, c2, c3, cb,
    CVW{M.zline(64n), M.zline(64n), M.zline(64n), Nil{}})
  blk_go(50n, 0n, xz, cv, xpw, wdt, bdt, dtb, negexp_row(al), D, wop,
    BSW{M.zline(1024n), Nil{}})

# the tensors of one block, as lists, for blk_out
def blk_of(nw: TA.Mat<1n, 32n>, nb: TA.Mat<1n, 32n>,
    ip: TA.Mat<128n, 32n>, +cwl: List<&2, F32>,
    cb: TA.Mat<1n, 64n>, xpw: TA.Mat<34n, 64n>,
    dpw: TA.Mat<64n, 2n>, dpb: TA.Mat<1n, 64n>,
    tib: TA.Mat<1n, 64n>, al: TA.Mat<64n, 16n>,
    dd: TA.Mat<1n, 64n>, op: TA.Mat<32n, 64n>,
    +res2: List<&2, F32>) -> List<&2, F32>:
  blk_out(res2,
    ml1x32(TA.Mat.to_list(1n, 32n, nw)),
    ml1x32(TA.Mat.to_list(1n, 32n, nb)),
    ml128x32(TA.Mat.to_list(128n, 32n, ip)),
    col_go(64n, 0n, cwl, Nil{}), col_go(64n, 1n, cwl, Nil{}),
    col_go(64n, 2n, cwl, Nil{}), col_go(64n, 3n, cwl, Nil{}),
    ml1x64(TA.Mat.to_list(1n, 64n, cb)),
    ml34x64(TA.Mat.to_list(34n, 64n, xpw)),
    ml64x2(TA.Mat.to_list(64n, 2n, dpw)),
    ml1x64(TA.Mat.to_list(1n, 64n, dpb)),
    ml1x64(TA.Mat.to_list(1n, 64n, tib)),
    ml64x16(TA.Mat.to_list(64n, 16n, al)),
    ml1x64(TA.Mat.to_list(1n, 64n, dd)),
    ml32x64(TA.Mat.to_list(32n, 64n, op)))

# ---------------------------------------------------------------------------
# the ViT tail: one more add, the final LayerNorm, the mean pool, the head

def pool_go(+left: Nat, +t: Nat, +xs: List<&2, F32>, +w: Nat,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      pool_go(p, (t + 1n : Nat), xs, w,
        M.add_lists(acc, row(t, xs, w)))

def div_list(+xs: List<&2, F32>, +c: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(h / c : F32), div_list(t, c)}

type FIN is Type:
  FIN{fin: List<&2, F32>, pool: List<&2, F32>, logits: List<&2, F32>}

def fin_all(+x: List<&2, F32>, +res: List<&2, F32>,
    +nfw: List<&2, F32>, +nfb: List<&2, F32>,
    +hw: List<&2, F32>, +hb: List<&2, F32>) -> FIN:
  +fr = M.add_lists(x, res)
  +fn = ln_rows(50n, 0n, fr, nfw, nfb, 32n, Nil{})
  +pl = div_list(pool_go(50n, 0n, fn, 32n, M.zline(32n)), 50.0)
  FIN{fn, pl, M.add_lists(M.matvec(hw, pl, 32n), hb)}

def show_fin(f: FIN) -> IO(Unit):
  match f:
    case FIN{fin, pool, logits}:
      do IO<Unit>:
        M.show_tag("mv_final", fin)
        M.show_tag("mv_pooled", pool)
        M.show_tag("mv_logits", logits)
        return Unit{}

# ---------------------------------------------------------------------------
# the eight blocks: the tensors are loaded inside the recursion (a TA.Mat is
# consumed once) and the new x/res travel on to the next block

def wpath(+i: Nat, +nm: String) -> String:
  +pre = String.append(String.append(ddir(), "/w_layers_"), Nat.show(i))
  String.append(String.append(String.append(pre, "_"), nm), ".bin")

def blocks_io(+left: Nat, +i: Nat, +x: List<&2, F32>,
    +res: List<&2, F32>, +nfw: List<&2, F32>,
    +nfb: List<&2, F32>, +hw: List<&2, F32>,
    +hb: List<&2, F32>) -> IO(Unit):
  match left:
    case 0n:
      do IO<Unit>:
        fv : FIN <- IO.pure(FIN, fin_all(x, res, nfw, nfb, hw, hb))
        show_fin(fv)
        return Unit{}
    case 1n+p:
      do IO<Unit>:
        nw : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wpath(i, "norm_weight"), 0)
        nb : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wpath(i, "norm_bias"), 0)
        ip : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, wpath(i, "mixer_in_proj_weight"), 0)
        cw : TA.Mat<64n, 4n> <- T.Mat.load(64n, 4n, wpath(i, "mixer_conv1d_weight"), 0)
        cb : TA.Mat<1n, 64n> <- T.Mat.load(1n, 64n, wpath(i, "mixer_conv1d_bias"), 0)
        xpw : TA.Mat<34n, 64n> <- T.Mat.load(34n, 64n, wpath(i, "mixer_x_proj_weight"), 0)
        dpw : TA.Mat<64n, 2n> <- T.Mat.load(64n, 2n, wpath(i, "mixer_dt_proj_weight"), 0)
        dpb : TA.Mat<1n, 64n> <- T.Mat.load(1n, 64n, wpath(i, "mixer_dt_proj_bias"), 0)
        tib : TA.Mat<1n, 64n> <- T.Mat.load(1n, 64n, wpath(i, "mixer_dt_bias"), 0)
        al : TA.Mat<64n, 16n> <- T.Mat.load(64n, 16n, wpath(i, "mixer_A_log"), 0)
        dd : TA.Mat<1n, 64n> <- T.Mat.load(1n, 64n, wpath(i, "mixer_D"), 0)
        op : TA.Mat<32n, 64n> <- T.Mat.load(32n, 64n, wpath(i, "mixer_out_proj_weight"), 0)
        +r2 : List<&2, F32> <- IO.pure(List<&2, F32>, res2_of(x, res))
        +cwl : List<&2, F32> <- IO.pure(List<&2, F32>,
          ml64x4(TA.Mat.to_list(64n, 4n, cw)))
        +o : List<&2, F32> <- IO.pure(List<&2, F32>,
          blk_of(nw, nb, ip, cwl, cb, xpw, dpw, dpb, tib, al, dd, op, r2))
        M.show_tag(String.append("mv_block", Nat.show(i)), o)
        M.show_tag(String.append(String.append("mv_block",
          Nat.show(i)), "_res"), r2)
        blocks_io(p, (i + 1n : Nat), o, r2, nfw, nfb, hw, hb)

def run_emb(+img: List<&2, F32>, +w0: List<&2, F32>, +b0: List<&2, F32>,
    +w1: List<&2, F32>, +b1: List<&2, F32>,
    +w2: List<&2, F32>, +b2: List<&2, F32>,
    +cls: List<&2, F32>, +pos: List<&2, F32>,
    +nfw: List<&2, F32>, +nfb: List<&2, F32>,
    +hw: List<&2, F32>, +hb: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    +tk : List<&2, F32> <- show_emp(emb_all(img, w0, b0, w1, b1, w2, b2, cls, pos))
    blocks_io(8n, 0n, tk, Nil{}, nfw, nfb, hw, hb)
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    im : TA.Mat<28n, 28n> <- T.Mat.load(28n, 28n, String.append(ddir(), "/img.bin"), 0)
    w0 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/w_to_patch_embedding_0_weight.bin"), 0)
    b0 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/w_to_patch_embedding_0_bias.bin"), 0)
    w1 : TA.Mat<32n, 16n> <- T.Mat.load(32n, 16n, String.append(ddir(), "/w_to_patch_embedding_1_weight.bin"), 0)
    b1 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_to_patch_embedding_1_bias.bin"), 0)
    w2 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_to_patch_embedding_2_weight.bin"), 0)
    b2 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_to_patch_embedding_2_bias.bin"), 0)
    cls : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_cls_token.bin"), 0)
    pos : TA.Mat<50n, 32n> <- T.Mat.load(50n, 32n, String.append(ddir(), "/w_pos_embedding.bin"), 0)
    nfw : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_norm_f_weight.bin"), 0)
    nfb : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_norm_f_bias.bin"), 0)
    hw : TA.Mat<10n, 32n> <- T.Mat.load(10n, 32n, String.append(ddir(), "/w_mlp_head_weight.bin"), 0)
    hb : TA.Mat<1n, 10n> <- T.Mat.load(1n, 10n, String.append(ddir(), "/w_mlp_head_bias.bin"), 0)
    run_emb(ml28(TA.Mat.to_list(28n, 28n, im)),
      ml1x16(TA.Mat.to_list(1n, 16n, w0)),
      ml1x16(TA.Mat.to_list(1n, 16n, b0)),
      ml32x16(TA.Mat.to_list(32n, 16n, w1)),
      ml1x32(TA.Mat.to_list(1n, 32n, b1)),
      ml1x32(TA.Mat.to_list(1n, 32n, w2)),
      ml1x32(TA.Mat.to_list(1n, 32n, b2)),
      ml1x32(TA.Mat.to_list(1n, 32n, cls)),
      ml50x32(TA.Mat.to_list(50n, 32n, pos)),
      ml1x32(TA.Mat.to_list(1n, 32n, nfw)),
      ml1x32(TA.Mat.to_list(1n, 32n, nfb)),
      ml10x32(TA.Mat.to_list(10n, 32n, hw)),
      ml1x10(TA.Mat.to_list(1n, 10n, hb)))
    return Unit{}
