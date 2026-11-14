# MambaViT A2: the hand-written backward of the A1b forward, gated at 1e-4.
#
# The forward is `probe_mamba_vit_emb.bend` (A1a/A1b); this program repeats it
# (it needs the per-token intermediates the forward gate never printed) and
# then walks back through, in this order:
#
#   the head             dW = dy (x) pooled, db = dy, d_pooled = W^T dy
#   the mean pool        d_final[t] = d_pooled / 50
#   the final LayerNorm  d_fra, d_w, d_b
#   the residual stream  dO_i = dR_{i+1}, dR_i = dR_{i+1} + d(LN)
#   each block (7 -> 0)  out_proj, the SiLU gate, the 64x16 scan (t
#                        backwards), dt = softplus, x_proj, the causal conv's
#                        three padded taps, in_proj, the block LayerNorm
#   the position/cls add
#   the patch embedding  LN2 -> Linear -> LN1
#
# `loss = sum(logits * dy)`; `reference/mamba_vit_bins.py` saves dy, the
# gradient of every parameter (`g_*`) and of the intermediates compared here.
#
# Shapes: 50 tokens (1 cls + 49 patches), dim 32, d_inner 64, dt_rank 2,
# d_state 16, d_conv 4, 8 blocks, 10 classes.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../mamba3/mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/mamba_vit"

# ---------------------------------------------------------------------------
# the tensor-array views, as flat row-major lists

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

# ---------------------------------------------------------------------------
# row helpers

def fill_go(+left: Nat, +v: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      fill_go(p, v, Con{v, acc})

def row(+t: Nat, +xs: List<&2, F32>, +w: Nat) -> List<&2, F32>:
  M.take_row(xs, Nat.mul(t, w), w)

# y = (x - mean)/sqrt(var + eps) * w + b, one row
def ln_row(+xs: List<&2, F32>, +ws: List<&2, F32>,
    +bs: List<&2, F32>) -> List<&2, F32>:
  +n = List.length(&2, F32, xs)
  +nf = F32.from_nat(n)
  +mean = (M.sum_list(xs) / nf : F32)
  +cent = M.sub_lists(xs, fill_go(n, mean, Nil{}))
  +var = (M.sumsq(cent) / nf : F32)
  +sc = (1.0 / F32.sqrt((var + 0.00001 : F32)) : F32)
  M.add_lists(M.mul_lists(M.scale_list(cent, sc), ws), bs)

def ln_rows(+left: Nat, +t: Nat, +xs: List<&2, F32>,
    +ws: List<&2, F32>, +bs: List<&2, F32>, +w: Nat,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ln_rows(p, (t + 1n : Nat), xs, ws, bs, w,
        M.push_row(acc, ln_row(row(t, xs, w), ws, bs)))

# acc + a*b, elementwise
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

# A = -exp(A_log), elementwise
def negexp_row(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(0.0 - F32.exp(h) : F32), negexp_row(t)}

# exp(c*x), elementwise
def expsc_row(+xs: List<&2, F32>, +c: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{F32.exp((c * h : F32)), expsc_row(t, c)}

def div_list(+xs: List<&2, F32>, +c: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{(h / c : F32), div_list(t, c)}

def bcast_go(+left: Nat, +v: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      bcast_go(p, v, List.append(&2, F32, acc, v))

# dt_proj(dt_raw) + dt_proj_bias + dt_bias, the argument of softplus
def dtpre_row(+w: List<&2, F32>, +b: List<&2, F32>,
    +tb: List<&2, F32>, +raw: List<&2, F32>) -> List<&2, F32>:
  M.add_lists(M.add_lists(M.matvec(w, raw, 2n), b), tb)

# ---------------------------------------------------------------------------
# the patch embedding forward, keeping p1 for the backward

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

def add_pos(+left: Nat, +t: Nat, +xs: List<&2, F32>,
    +pos: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      add_pos(p, (t + 1n : Nat), xs, pos,
        M.push_row(acc, M.add_lists(row(t, xs, 32n), row(t, pos, 32n))))

type EP is Type:
  EP{p1: List<&2, F32>, emb: List<&2, F32>}

# LN(16) -> Linear(16->32) -> LN(32): the patch projection's bias is zero in
# the twin's init, so the forward adds nothing, but the backward still needs
# that bias's gradient (the sum of d_pre over the 49 patches)
def emb_step(+pa: List<&2, F32>, +w0: List<&2, F32>,
    +b0: List<&2, F32>, +w1: List<&2, F32>,
    +w2: List<&2, F32>, +b2: List<&2, F32>, e: EP) -> EP:
  match e:
    case EP{p1s, emb0}:
      +p1 = ln_row(pa, w0, b0)
      +p2 = ln_row(M.matvec(w1, p1, 16n), w2, b2)
      EP{M.push_row(p1s, p1), M.push_row(emb0, p2)}

def emb_go(+left: Nat, +t: Nat, +pa: List<&2, F32>,
    +w0: List<&2, F32>, +b0: List<&2, F32>,
    +w1: List<&2, F32>, +w2: List<&2, F32>, +b2: List<&2, F32>,
    e: EP) -> EP:
  match left:
    case 0n:
      e
    case 1n+p:
      emb_go(p, (t + 1n : Nat), pa, w0, b0, w1, w2, b2,
        emb_step(row(t, pa, 16n), w0, b0, w1, w2, b2, e))

type EMP is Type:
  EMP{p1: List<&2, F32>, toks: List<&2, F32>}

def emb_fin(e: EP, +cls: List<&2, F32>,
    +pos: List<&2, F32>) -> EMP:
  match e:
    case EP{p1s, emb}:
      EMP{p1s,
        add_pos(50n, 0n, List.append(&2, F32, cls, emb), pos, Nil{})}

def emb_all(+pa: List<&2, F32>, +w0: List<&2, F32>,
    +b0: List<&2, F32>, +w1: List<&2, F32>,
    +w2: List<&2, F32>, +b2: List<&2, F32>,
    +cls: List<&2, F32>, +pos: List<&2, F32>) -> EMP:
  emb_fin(emb_go(49n, 0n, pa, w0, b0, w1, w2, b2, EP{Nil{}, Nil{}}),
    cls, pos)

# ---------------------------------------------------------------------------
# LayerNorm backward, one row: dx, d_w, d_b

type LNB is Type:
  LNB{dx: List<&2, F32>, dw: List<&2, F32>, db: List<&2, F32>}

def dxb_go(+dyw: List<&2, F32>, +xh: List<&2, F32>, +r: F32,
    +k1: F32, +k2: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match dyw xh:
    case Nil{} _:
      acc
    case _ Nil{}:
      acc
    case Con{d, dt} Con{h, ht}:
      +v = (((r * d : F32) - k1 : F32) - (h * k2 : F32) : F32)
      dxb_go(dt, ht, r, k1, k2, M.push_row(acc, Con{v, Nil{}}))

# y_i = xh_i*w_i + b_i with xh = (x - mean)*r:
#   dx_i = r*w_i*dy_i - (r/n)*sum(dy*w) - xh_i*(r/n)*sum(dy*w*xh)
#   dw_i = dy_i*xh_i, db_i = dy_i
def lnr_bwd(+xs: List<&2, F32>, +ws: List<&2, F32>,
    +dys: List<&2, F32>) -> LNB:
  +n = List.length(&2, F32, xs)
  +nf = F32.from_nat(n)
  +mean = (M.sum_list(xs) / nf : F32)
  +cent = M.sub_lists(xs, fill_go(n, mean, Nil{}))
  +var = (M.sumsq(cent) / nf : F32)
  +r = (1.0 / F32.sqrt((var + 0.00001 : F32)) : F32)
  +xh = M.scale_list(cent, r)
  +dyw = M.mul_lists(dys, ws)
  +k1 = ((r / nf : F32) * M.sum_list(dyw) : F32)
  +k2 = ((r / nf : F32) * M.dot(dyw, xh) : F32)
  LNB{dxb_go(dyw, xh, r, k1, k2, Nil{}), M.mul_lists(dys, xh), dys}

type LNR is Type:
  LNR{dxs: List<&2, F32>, dws: List<&2, F32>, dbs: List<&2, F32>}

def lnx_step(lnb: LNB, a: LNR) -> LNR:
  match lnb a:
    case LNB{dx, dw, db} LNR{dxs, dws, dbs}:
      LNR{M.push_row(dxs, dx), M.add_lists(dws, dw), M.add_lists(dbs, db)}

def lnx_go(+left: Nat, +t: Nat, +xs: List<&2, F32>,
    +ws: List<&2, F32>, +dys: List<&2, F32>, +w: Nat,
    a: LNR) -> LNR:
  match left:
    case 0n:
      a
    case 1n+p:
      lnx_go(p, (t + 1n : Nat), xs, ws, dys, w,
        lnx_step(lnr_bwd(row(t, xs, w), ws, row(t, dys, w)), a))

def ln_rows_bwd(+n: Nat, +xs: List<&2, F32>, +ws: List<&2, F32>,
    +dys: List<&2, F32>, +w: Nat) -> LNR:
  lnx_go(n, 0n, xs, ws, dys, w, LNR{Nil{}, M.zline(w), M.zline(w)})

# ---------------------------------------------------------------------------
# the block forward, keeping everything the backward reads: the LayerNorm
# output h, the in_proj row xz (x in [0,64), z in [64,128)), the conv output
# before its SiLU, the x_proj row dbl, dt, y and the gated yg, the 64x16 state
# of every token (prepended, so the list is already in reverse) and the block
# output

# one column of the depthwise conv weight (64x4 row-major, c*4+k)
def col_go(+left: Nat, +j: Nat, +cw: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      col_go(p, j, List.drop(&2, F32, cw, 4n),
        M.push_row(acc, M.take_row(cw, j, 1n)))

# y[t,c] = bias[c] + sum_k w[c,k] * xp[t+k,c], three zero tokens in front:
# y[t] = bias + c3*x[t] + c2*x[t-1] + c1*x[t-2] + c0*x[t-3]
def conv_row(+xr: List<&2, F32>, +w1: List<&2, F32>,
    +w2: List<&2, F32>, +w3: List<&2, F32>,
    +c0: List<&2, F32>, +c1: List<&2, F32>,
    +c2: List<&2, F32>, +c3: List<&2, F32>,
    +cb: List<&2, F32>) -> List<&2, F32>:
  mad(mad(mad(mad(cb, c3, xr), c2, w1), c1, w2), c0, w3)

type CVW is Type:
  CVW{w1: List<&2, F32>, w2: List<&2, F32>, w3: List<&2, F32>,
      xs: List<&2, F32>}

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

type HX is Type:
  HX{h: List<&2, F32>, xz: List<&2, F32>}

# h = LN(res2) per token, xz = in_proj(h)
def hx_step(+hz: List<&2, F32>, +ip: List<&2, F32>, a: HX) -> HX:
  match a:
    case HX{h0, xz0}:
      HX{M.push_row(h0, hz), M.push_row(xz0, M.matvec(ip, hz, 32n))}

# dt[c] = softplus(dot(dt_proj_w[c], dt_raw) + dt_proj_b[c] + dt_bias[c])
def hx_go(+left: Nat, +t: Nat, +res2: List<&2, F32>,
    +nw: List<&2, F32>, +nb: List<&2, F32>,
    +ip: List<&2, F32>, a: HX) -> HX:
  match left:
    case 0n:
      a
    case 1n+p:
      hx_go(p, (t + 1n : Nat), res2, nw, nb, ip,
        hx_step(ln_row(row(t, res2, 32n), nw, nb), ip, a))

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

type BF is Type:
  BF{h: List<&2, F32>, xz: List<&2, F32>, cvs: List<&2, F32>,
     xs: List<&2, F32>, dbl: List<&2, F32>, dts: List<&2, F32>,
     zs: List<&2, F32>, yv: List<&2, F32>, yg: List<&2, F32>,
     hs: List<&2, F32>, out: List<&2, F32>}

type BST is Type:
  BST{xs: List<&2, F32>, dbl: List<&2, F32>, dts: List<&2, F32>,
      zs: List<&2, F32>, yv: List<&2, F32>, yg: List<&2, F32>,
      hs: List<&2, F32>, cur: List<&2, F32>, out: List<&2, F32>}

# one token: B/C and dt from x_proj, the scan, the SiLU gate, out_proj
def bst_step(+t: Nat, +xz: List<&2, F32>, +cv: List<&2, F32>,
    +xpw: List<&2, F32>, +wdt: List<&2, F32>,
    +bdt: List<&2, F32>, +dtb: List<&2, F32>,
    +A: List<&2, F32>, +D: List<&2, F32>,
    +wop: List<&2, F32>, st: BST) -> BST:
  match st:
    case BST{xs0, dbl0, dts0, zs0, yv0, yg0, hs0, cur, outb}:
      +xr = M.silu_row(M.take_row(cv, Nat.mul(t, 64n), 64n))
      +zr = M.take_row(xz, Nat.add(Nat.mul(t, 128n), 64n), 64n)
      +dbl = M.matvec(xpw, xr, 64n)
      +dtr = dt_go(64n, wdt, bdt, dtb, M.take_row(dbl, 0n, 2n), Nil{})
      +h2 = upd_c(64n, A, cur, dtr, M.take_row(dbl, 2n, 16n), xr, Nil{})
      +yr = List.reverse(&2, F32,
        dotc(64n, h2, M.take_row(dbl, 18n, 16n), D, xr, Nil{}))
      +yg = M.mul_lists(yr, M.silu_row(zr))
      BST{M.push_row(xs0, xr), M.push_row(dbl0, dbl), M.push_row(dts0, dtr),
        M.push_row(zs0, zr), M.push_row(yv0, yr), M.push_row(yg0, yg),
        List.append(&2, F32, h2, hs0), h2,
        M.push_row(outb, M.matvec(wop, yg, 64n))}

def bst_go(+left: Nat, +t: Nat, +xz: List<&2, F32>,
    +cv: List<&2, F32>, +xpw: List<&2, F32>,
    +wdt: List<&2, F32>, +bdt: List<&2, F32>,
    +dtb: List<&2, F32>, +A: List<&2, F32>,
    +D: List<&2, F32>, +wop: List<&2, F32>,
    st: BST) -> BST:
  match left:
    case 0n:
      st
    case 1n+p:
      bst_go(p, (t + 1n : Nat), xz, cv, xpw, wdt, bdt, dtb, A, D, wop,
        bst_step(t, xz, cv, xpw, wdt, bdt, dtb, A, D, wop, st))

def blk_end(st: BST, +h: List<&2, F32>, +xz: List<&2, F32>,
    +cvs: List<&2, F32>) -> BF:
  match st:
    case BST{xs, dbl, dts, zs, yv, yg, hs, cur, out}:
      BF{h, xz, cvs, xs, dbl, dts, zs, yv, yg, hs, out}

def blk_fw_c(+xz: List<&2, F32>, +h: List<&2, F32>,
    +cv: List<&2, F32>, +xpw: List<&2, F32>,
    +wdt: List<&2, F32>, +bdt: List<&2, F32>,
    +dtb: List<&2, F32>, +al: List<&2, F32>,
    +D: List<&2, F32>, +wop: List<&2, F32>) -> BF:
  blk_end(bst_go(50n, 0n, xz, cv, xpw, wdt, bdt, dtb, negexp_row(al),
    D, wop, BST{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{},
      M.zline(1024n), Nil{}}), h, xz, cv)

def blk_fw_b(hx: HX, +c0: List<&2, F32>, +c1: List<&2, F32>,
    +c2: List<&2, F32>, +c3: List<&2, F32>, +cb: List<&2, F32>,
    +xpw: List<&2, F32>, +wdt: List<&2, F32>,
    +bdt: List<&2, F32>, +dtb: List<&2, F32>,
    +al: List<&2, F32>, +D: List<&2, F32>,
    +wop: List<&2, F32>) -> BF:
  match hx:
    case HX{+h, +xz}:
      blk_fw_c(xz, h, conv_go(50n, 0n, xz, c0, c1, c2, c3, cb,
        CVW{M.zline(64n), M.zline(64n), M.zline(64n), Nil{}}),
        xpw, wdt, bdt, dtb, al, D, wop)

def blk_fw(+res2: List<&2, F32>, +nw: List<&2, F32>,
    +nb: List<&2, F32>, +ip: List<&2, F32>,
    +c0: List<&2, F32>, +c1: List<&2, F32>,
    +c2: List<&2, F32>, +c3: List<&2, F32>, +cb: List<&2, F32>,
    +xpw: List<&2, F32>, +wdt: List<&2, F32>,
    +bdt: List<&2, F32>, +dtb: List<&2, F32>,
    +al: List<&2, F32>, +D: List<&2, F32>,
    +wop: List<&2, F32>) -> BF:
  blk_fw_b(hx_go(50n, 0n, res2, nw, nb, ip, HX{Nil{}, Nil{}}),
    c0, c1, c2, c3, cb, xpw, wdt, bdt, dtb, al, D, wop)

def bf_out(bf: BF) -> List<&2, F32>:
  match bf:
    case BF{_, _, _, _, _, _, _, _, _, _, out}:
      out

# the block's input: the running sum, `x + res`, or `x` on the first block
def res2_of(+x: List<&2, F32>, +res: List<&2, F32>) -> List<&2, F32>:
  match res:
    case Nil{}:
      x
    case Con{h, t}:
      M.add_lists(x, Con{h, t})

# ---------------------------------------------------------------------------
# the block backward, one token at a time, t from 49 down to 0.
#
# The scan runs backwards because h_t = a_t h_{t-1} + dt_t B_t xs_t is a
# recurrence: the carry is dL/dh_{t-1} = (dL/dh_t) a_t, and each step adds the
# direct term d_y * C_t before it uses it.
#
#   ddt[c]   = sum_q (G a A hp + G B_q xs_c)      (da/ddt = a A)
#   dA[c,q]  = sum_t dt_c a hp G                  (da/dA = dt a)
#   dB[q]   += sum_c dt_c xs_c G
#   dxs[c]  += sum_q dt_c B_q G  +  d_y D_c
#   dG       = G a                                (the next token's carry)

# per channel: dB[q] and dC[q] accumulate a 16-row in place
def dBc_go(+left: Nat, +off: Nat, +G: List<&2, F32>,
    +dt: List<&2, F32>, +xs: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +g = M.take_row(G, Nat.mul(off, 16n), 16n)
      +sc = (M.nth(off, dt) * M.nth(off, xs) : F32)
      dBc_go(p, (off + 1n : Nat), G, dt, xs,
        M.add_lists(acc, M.scale_list(g, sc)))

# ddt, 64 channels, summed over the 16 slots
def ddt_go(+left: Nat, +off: Nat, +G: List<&2, F32>,
    +A: List<&2, F32>, +dt: List<&2, F32>, +hp: List<&2, F32>,
    +B: List<&2, F32>, +xs: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +g = M.take_row(G, Nat.mul(off, 16n), 16n)
      +ar = M.take_row(A, Nat.mul(off, 16n), 16n)
      +hr = M.take_row(hp, Nat.mul(off, 16n), 16n)
      +av = expsc_row(ar, M.nth(off, dt))
      +v = (M.dot(g, M.mul_lists(M.mul_lists(av, ar), hr))
        + (M.nth(off, xs) * M.dot(g, B) : F32) : F32)
      ddt_go(p, (off + 1n : Nat), G, A, dt, hp, B, xs, Con{v, acc})

# dA, 64 channels of 16 slots, appended in order
def da_go(+left: Nat, +off: Nat, +G: List<&2, F32>,
    +A: List<&2, F32>, +dt: List<&2, F32>, +hp: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      +g = M.take_row(G, Nat.mul(off, 16n), 16n)
      +ar = M.take_row(A, Nat.mul(off, 16n), 16n)
      +hr = M.take_row(hp, Nat.mul(off, 16n), 16n)
      +av = expsc_row(ar, M.nth(off, dt))
      da_go(p, (off + 1n : Nat), G, A, dt, hp,
        M.push_row(acc, M.scale_list(M.mul_lists(M.mul_lists(g, av), hr),
          M.nth(off, dt))))

# the next carry: dG = G * exp(dt*A)
def dg_go(+left: Nat, +off: Nat, +G: List<&2, F32>,
    +A: List<&2, F32>, +dt: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      dg_go(p, (off + 1n : Nat), G, A, dt,
        List.append(&2, F32, acc,
          M.mul_lists(M.take_row(G, Nat.mul(off, 16n), 16n),
            expsc_row(M.take_row(A, Nat.mul(off, 16n), 16n),
              M.nth(off, dt)))))

def dxs_go(+left: Nat, +off: Nat, +G: List<&2, F32>,
    +B: List<&2, F32>, +dt: List<&2, F32>, +dyv: List<&2, F32>,
    +D: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      +v = ((M.nth(off, dyv) * M.nth(off, D) : F32)
        + (M.nth(off, dt) * M.dot(M.take_row(G, Nat.mul(off, 16n), 16n), B)
          : F32) : F32)
      dxs_go(p, (off + 1n : Nat), G, B, dt, dyv, D, Con{v, acc})

def dc_go(+left: Nat, +off: Nat, +hs: List<&2, F32>,
    +dyv: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      dc_go(p, (off + 1n : Nat), hs, dyv,
        M.add_lists(acc,
          M.scale_list(M.take_row(hs, Nat.mul(off, 16n), 16n),
            M.nth(off, dyv))))

# h_{t-1} out of the reversed state list (row j is h_{49-j})
def hp_of(+t: Nat, +hs: List<&2, F32>) -> List<&2, F32>:
  match t:
    case 0n:
      M.zline(1024n)
    case 1n+p:
      M.take_row(hs, Nat.mul(Nat.sub(49n, p), 1024n), 1024n)

def xm_if(b: Bool, +t: Nat, +k: Nat,
    +xz: List<&2, F32>) -> List<&2, F32>:
  match b:
    case True{}:
      M.take_row(xz, Nat.mul(Nat.sub(t, k), 128n), 64n)
    case False{}:
      M.zline(64n)

# x[t-k], zero when the tap is in the padding
def xm_of(+t: Nat, +k: Nat, +xz: List<&2, F32>) -> List<&2, F32>:
  xm_if(Nat.is_ge(t, k), t, k, xz)

type BWK is Type:
  BWK{G: List<&2, F32>, d1: List<&2, F32>, d2: List<&2, F32>,
      d3: List<&2, F32>, wop2: List<&2, F32>, xpw2: List<&2, F32>,
      wdt2: List<&2, F32>, bdt2: List<&2, F32>, dtb2: List<&2, F32>,
      dd: List<&2, F32>, cc0: List<&2, F32>, cc1: List<&2, F32>,
      cc2: List<&2, F32>, cc3: List<&2, F32>, cb2: List<&2, F32>,
      ip2: List<&2, F32>, dad: List<&2, F32>, dhacc: List<&2, F32>}

def bwk_step(+t: Nat, +h: List<&2, F32>, +xz: List<&2, F32>,
    +cvs: List<&2, F32>, +xs: List<&2, F32>, +dbl: List<&2, F32>,
    +dts: List<&2, F32>, +zs: List<&2, F32>, +yv: List<&2, F32>,
    +yg: List<&2, F32>, +hs: List<&2, F32>, +A: List<&2, F32>,
    +ip: List<&2, F32>, +c0f: List<&2, F32>, +c1f: List<&2, F32>,
    +c2f: List<&2, F32>, +c3f: List<&2, F32>, +cb: List<&2, F32>,
    +xpw: List<&2, F32>, +wdt: List<&2, F32>, +bdt: List<&2, F32>,
    +dtb: List<&2, F32>, +D: List<&2, F32>, +wop: List<&2, F32>,
    +dO: List<&2, F32>, k: BWK) -> BWK:
  match k:
    case BWK{+G, +d1, +d2, +d3, +wop2, +xpw2, +wdt2, +bdt2, +dtb2, +dd,
        +cc0, +cc1, +cc2, +cc3, +cb2, +ip2, +dad, +dhacc}:
      +o64 = Nat.mul(t, 64n)
      +dout = M.take_row(dO, Nat.mul(t, 32n), 32n)
      +yg_t = M.take_row(yg, o64, 64n)
      +yv_t = M.take_row(yv, o64, 64n)
      +z_t = M.take_row(zs, o64, 64n)
      +xs_t = M.take_row(xs, o64, 64n)
      +yc_t = M.take_row(cvs, o64, 64n)
      +dt_t = M.take_row(dts, o64, 64n)
      +x_t = M.take_row(xz, Nat.mul(t, 128n), 64n)
      +h_t = M.take_row(h, Nat.mul(t, 32n), 32n)
      +dbl_t = M.take_row(dbl, Nat.mul(t, 34n), 34n)
      +B_t = M.take_row(dbl_t, 2n, 16n)
      +C_t = M.take_row(dbl_t, 18n, 16n)
      +dtraw_t = M.take_row(dbl_t, 0n, 2n)
      +hs_t = M.take_row(hs, Nat.mul(Nat.sub(49n, t), 1024n), 1024n)
      +hp_t = hp_of(t, hs)
      +dyg = M.matvec_t(wop, dout, 32n, 64n)
      +dyv = M.mul_lists(dyg, M.silu_row(z_t))
      +dz = M.mul_lists(M.mul_lists(dyg, yv_t), M.silu_deriv_row(z_t))
      +Gt = M.add_lists(G, M.outer(dyv, C_t))
      +ddt = ddt_go(64n, 0n, Gt, A, dt_t, hp_t, B_t, xs_t, Nil{})
      +dAr = da_go(64n, 0n, Gt, A, dt_t, hp_t, Nil{})
      +dGr = dg_go(64n, 0n, Gt, A, dt_t, Nil{})
      +dBv = dBc_go(64n, 0n, Gt, dt_t, xs_t, M.zline(16n))
      +dxs1 = dxs_go(64n, 0n, Gt, B_t, dt_t, dyv, D, Nil{})
      +dCv = dc_go(64n, 0n, hs_t, dyv, M.zline(16n))
      +dpre = M.mul_lists(ddt,
        M.sigmoid_row(dtpre_row(wdt, bdt, dtb, dtraw_t)))
      +ddbl = List.append(&2, F32, M.matvec_t(wdt, dpre, 64n, 2n),
        List.append(&2, F32, dBv, dCv))
      +dxs2 = M.add_lists(dxs1, M.matvec_t(xpw, ddbl, 34n, 64n))
      +dy2 = M.mul_lists(dxs2, M.silu_deriv_row(yc_t))
      +xm1 = xm_of(t, 1n, xz)
      +xm2 = xm_of(t, 2n, xz)
      +xm3 = xm_of(t, 3n, xz)
      +dx_t = mad(mad(mad(mad(M.zline(64n), c3f, dy2), c2f, d1),
        c1f, d2), c0f, d3)
      +dxz_t = List.append(&2, F32, dx_t, dz)
      +dh_t = M.matvec_t(ip, dxz_t, 128n, 32n)
      BWK{dGr, dy2, d1, d2,
        M.add_lists(wop2, M.outer(dout, yg_t)),
        M.add_lists(xpw2, M.outer(ddbl, xs_t)),
        M.add_lists(wdt2, M.outer(dpre, dtraw_t)),
        M.add_lists(bdt2, dpre),
        M.add_lists(dtb2, dpre),
        M.add_lists(dd, M.mul_lists(dyv, xs_t)),
        M.add_lists(cc0, M.mul_lists(dy2, xm3)),
        M.add_lists(cc1, M.mul_lists(dy2, xm2)),
        M.add_lists(cc2, M.mul_lists(dy2, xm1)),
        M.add_lists(cc3, M.mul_lists(dy2, x_t)),
        M.add_lists(cb2, dy2),
        M.add_lists(ip2, M.outer(dxz_t, h_t)),
        M.add_lists(dad, dAr),
        List.append(&2, F32, dh_t, dhacc)}

def bwk_go(+left: Nat, +t: Nat, +h: List<&2, F32>,
    +xz: List<&2, F32>, +cvs: List<&2, F32>,
    +xs: List<&2, F32>, +dbl: List<&2, F32>,
    +dts: List<&2, F32>, +zs: List<&2, F32>,
    +yv: List<&2, F32>, +yg: List<&2, F32>,
    +hs: List<&2, F32>, +A: List<&2, F32>,
    +ip: List<&2, F32>, +c0f: List<&2, F32>, +c1f: List<&2, F32>,
    +c2f: List<&2, F32>, +c3f: List<&2, F32>, +cb: List<&2, F32>,
    +xpw: List<&2, F32>, +wdt: List<&2, F32>, +bdt: List<&2, F32>,
    +dtb: List<&2, F32>, +D: List<&2, F32>, +wop: List<&2, F32>,
    +dO: List<&2, F32>, k: BWK) -> BWK:
  match left:
    case 0n:
      k
    case 1n+p:
      bwk_go(p, (t - 1n : Nat), h, xz, cvs, xs, dbl, dts, zs, yv, yg, hs,
        A, ip, c0f, c1f, c2f, c3f, cb, xpw, wdt, bdt, dtb, D, wop, dO,
        bwk_step(t, h, xz, cvs, xs, dbl, dts, zs, yv, yg, hs, A, ip,
          c0f, c1f, c2f, c3f, cb, xpw, wdt, bdt, dtb, D, wop, dO, k))

type BG is Type:
  BG{dnw: List<&2, F32>, dnb: List<&2, F32>, dip: List<&2, F32>,
     dc0: List<&2, F32>, dc1: List<&2, F32>, dc2: List<&2, F32>,
     dc3: List<&2, F32>, dcb: List<&2, F32>, dxpw: List<&2, F32>,
     dwdt: List<&2, F32>, dbdt: List<&2, F32>, ddtb: List<&2, F32>,
     dal: List<&2, F32>, dD: List<&2, F32>, dop: List<&2, F32>,
     dres: List<&2, F32>}

def bg_of(lnr: LNR, +dal: List<&2, F32>, +wop2: List<&2, F32>,
    +xpw2: List<&2, F32>, +wdt2: List<&2, F32>,
    +bdt2: List<&2, F32>, +dtb2: List<&2, F32>, +dd: List<&2, F32>,
    +cc0: List<&2, F32>, +cc1: List<&2, F32>, +cc2: List<&2, F32>,
    +cc3: List<&2, F32>, +cb2: List<&2, F32>, +ip2: List<&2, F32>,
    +dR: List<&2, F32>) -> BG:
  match lnr:
    case LNR{dxs, dws, dbs}:
      BG{dws, dbs, ip2, cc0, cc1, cc2, cc3, cb2, xpw2, wdt2, bdt2, dtb2,
        dal, dd, wop2, M.add_lists(dR, dxs)}

def bw_fin(k: BWK, +A: List<&2, F32>, +res2: List<&2, F32>,
    +nw: List<&2, F32>, +dR: List<&2, F32>) -> BG:
  match k:
    case BWK{_, +d1, +d2, +d3, +wop2, +xpw2, +wdt2, +bdt2, +dtb2, +dd,
        +cc0, +cc1, +cc2, +cc3, +cb2, +ip2, +dad, +dhacc}:
      bg_of(ln_rows_bwd(50n, res2, nw, dhacc, 32n), M.mul_lists(dad, A),
        wop2, xpw2, wdt2, bdt2, dtb2, dd, cc0, cc1, cc2, cc3, cb2, ip2, dR)

def blk_bwd(bf: BF, +nw: List<&2, F32>, +nb: List<&2, F32>,
    +ip: List<&2, F32>, +c0f: List<&2, F32>, +c1f: List<&2, F32>,
    +c2f: List<&2, F32>, +c3f: List<&2, F32>, +cb: List<&2, F32>,
    +xpw: List<&2, F32>, +wdt: List<&2, F32>, +bdt: List<&2, F32>,
    +dtb: List<&2, F32>, +al: List<&2, F32>, +D: List<&2, F32>,
    +wop: List<&2, F32>, +res2: List<&2, F32>,
    +dO: List<&2, F32>, +dR: List<&2, F32>) -> BG:
  match bf:
    case BF{h, xz, cvs, xs, dbl, dts, zs, yv, yg, hs, out}:
      bw_fin(bwk_go(50n, 49n, h, xz, cvs, xs, dbl, dts, zs, yv, yg, hs,
        negexp_row(al), ip, c0f, c1f, c2f, c3f, cb, xpw, wdt, bdt, dtb,
        D, wop, dO,
        BWK{M.zline(1024n), M.zline(64n), M.zline(64n), M.zline(64n),
          M.zline(2048n), M.zline(2176n), M.zline(128n), M.zline(64n),
          M.zline(64n), M.zline(64n), M.zline(64n), M.zline(64n),
          M.zline(64n), M.zline(64n), M.zline(64n), M.zline(4096n),
          M.zline(1024n), Nil{}}),
        negexp_row(al), res2, nw, dR)

# ---------------------------------------------------------------------------
# the ViT tail backward: the head, the mean pool and the final LayerNorm

def pool_go(+left: Nat, +t: Nat, +xs: List<&2, F32>, +w: Nat,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      pool_go(p, (t + 1n : Nat), xs, w,
        M.add_lists(acc, row(t, xs, w)))

type TB is Type:
  TB{dwh: List<&2, F32>, dhb: List<&2, F32>, dnfw: List<&2, F32>,
     dnfb: List<&2, F32>, dfin: List<&2, F32>, dpo: List<&2, F32>,
     dfra: List<&2, F32>}

def tb_fin(lnr: LNR, +dwh: List<&2, F32>, +dy: List<&2, F32>,
    +dpo: List<&2, F32>, +dfin: List<&2, F32>) -> TB:
  match lnr:
    case LNR{dxs, dws, dbs}:
      TB{dwh, dy, dws, dbs, dfin, dpo, dxs}

def tail_bwd(+ofin: List<&2, F32>, +r7: List<&2, F32>,
    +dy: List<&2, F32>, +hw: List<&2, F32>,
    +nfw: List<&2, F32>, +nfb: List<&2, F32>) -> TB:
  +fra = M.add_lists(ofin, r7)
  +fn = ln_rows(50n, 0n, fra, nfw, nfb, 32n, Nil{})
  +pl = div_list(pool_go(50n, 0n, fn, 32n, M.zline(32n)), 50.0)
  +dpo = M.matvec_t(hw, dy, 10n, 32n)
  +dfin = div_list(dpo, 50.0)
  tb_fin(ln_rows_bwd(50n, fra, nfw, bcast_go(50n, dfin, Nil{}), 32n),
    M.outer(dy, pl), dy, dpo, bcast_go(50n, dfin, Nil{}))

# ---------------------------------------------------------------------------
# the patch embedding backward: LN2, the Linear (its weight AND its zeroed
# bias), LN1

type EB is Type:
  EB{dw0: List<&2, F32>, db0: List<&2, F32>, dw1: List<&2, F32>,
     db1: List<&2, F32>, dw2: List<&2, F32>, db2: List<&2, F32>}

def eb_fin(lnb1: LNB, +dw2v: List<&2, F32>, +db2v: List<&2, F32>,
    +dp2: List<&2, F32>, +p1_t: List<&2, F32>,
    +dw0: List<&2, F32>, +db0: List<&2, F32>,
    +dw1: List<&2, F32>, +db1: List<&2, F32>,
    +dw2: List<&2, F32>, +db2: List<&2, F32>) -> EB:
  match lnb1:
    case LNB{dp1, dw0v, db0v}:
      EB{M.add_lists(dw0, dw0v), M.add_lists(db0, db0v),
        M.add_lists(dw1, M.outer(dp2, p1_t)), M.add_lists(db1, dp2),
        M.add_lists(dw2, dw2v), M.add_lists(db2, db2v)}

def eb_step(lnb: LNB, +pa_t: List<&2, F32>, +p1_t: List<&2, F32>,
    +w0: List<&2, F32>, +w1: List<&2, F32>, a: EB) -> EB:
  match lnb a:
    case LNB{+dp2, dw2v, db2v}
        EB{dw0, db0, dw1, db1, dw2, db2}:
      eb_fin(lnr_bwd(pa_t, w0, M.matvec_t(w1, dp2, 32n, 16n)),
        dw2v, db2v, dp2, p1_t, dw0, db0, dw1, db1, dw2, db2)

def eb_go(+left: Nat, +t: Nat, +pa: List<&2, F32>,
    +p1: List<&2, F32>, +demb: List<&2, F32>,
    +w0: List<&2, F32>, +w1: List<&2, F32>,
    +w2: List<&2, F32>, a: EB) -> EB:
  match left:
    case 0n:
      a
    case 1n+p:
      eb_go(p, (t + 1n : Nat), pa, p1, demb, w0, w1, w2,
        eb_step(lnr_bwd(M.matvec(w1, row(t, p1, 16n), 16n), w2,
          row(t, demb, 32n)), row(t, pa, 16n), row(t, p1, 16n),
          w0, w1, a))

def emb_bwd(+pa: List<&2, F32>, +p1: List<&2, F32>,
    +demb: List<&2, F32>, +w0: List<&2, F32>,
    +w1: List<&2, F32>, +w2: List<&2, F32>) -> EB:
  eb_go(49n, 0n, pa, p1, demb, w0, w1, w2,
    EB{M.zline(16n), M.zline(16n), M.zline(512n), M.zline(32n),
      M.zline(32n), M.zline(32n)})

# ---------------------------------------------------------------------------
# the four conv-weight columns as the parameter's own layout, c*4+k

# the c-th row of the conv weight: (c,0), (c,1), (c,2), (c,3)
def krow(+j: Nat, +c0: List<&2, F32>, +c1: List<&2, F32>,
    +c2: List<&2, F32>, +c3: List<&2, F32>) -> List<&2, F32>:
  Con{M.nth(j, c0),
    Con{M.nth(j, c1), Con{M.nth(j, c2), Con{M.nth(j, c3), Nil{}}}}}

def kcat_go(+left: Nat, +j: Nat, +c0: List<&2, F32>,
    +c1: List<&2, F32>, +c2: List<&2, F32>, +c3: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      kcat_go(p, (j + 1n : Nat), c0, c1, c2, c3,
        List.append(&2, F32, acc, krow(j, c0, c1, c2, c3)))

# ---------------------------------------------------------------------------
# tags

def wpath(+i: Nat, +nm: String) -> String:
  +pre = String.append(String.append(ddir(), "/w_layers_"), Nat.show(i))
  String.append(String.append(String.append(pre, "_"), nm), ".bin")

def gtag(+i: Nat, +nm: String) -> String:
  String.append(String.append(String.append("mv_g_layers_",
    Nat.show(i)), "_"), nm)

def show_block(bg: BG, +i: Nat) -> IO(List<&2, F32>):
  match bg:
    case BG{dnw, dnb, dip, dc0, dc1, dc2, dc3, dcb, dxpw, dwdt, dbdt, ddtb,
        dal, dD, dop, dres}:
      do IO<List<&2, F32>>:
        M.show_tag(gtag(i, "norm_weight"), dnw)
        M.show_tag(gtag(i, "norm_bias"), dnb)
        M.show_tag(gtag(i, "mixer_in_proj_weight"), dip)
        M.show_tag(gtag(i, "mixer_conv1d_weight"),
          kcat_go(64n, 0n, dc0, dc1, dc2, dc3, Nil{}))
        M.show_tag(gtag(i, "mixer_conv1d_bias"), dcb)
        M.show_tag(gtag(i, "mixer_x_proj_weight"), dxpw)
        M.show_tag(gtag(i, "mixer_dt_proj_weight"), dwdt)
        M.show_tag(gtag(i, "mixer_dt_proj_bias"), dbdt)
        M.show_tag(gtag(i, "mixer_dt_bias"), ddtb)
        M.show_tag(gtag(i, "mixer_A_log"), dal)
        M.show_tag(gtag(i, "mixer_D"), dD)
        M.show_tag(gtag(i, "mixer_out_proj_weight"), dop)
        return dres

def show_tail(tb: TB) -> IO(List<&2, F32>):
  match tb:
    case TB{dwh, dhb, dnfw, dnfb, dfin, dpo, dfra}:
      do IO<List<&2, F32>>:
        M.show_tag("mv_g_mlp_head_weight", dwh)
        M.show_tag("mv_g_mlp_head_bias", dhb)
        M.show_tag("mv_g_norm_f_weight", dnfw)
        M.show_tag("mv_g_norm_f_bias", dnfb)
        M.show_tag("mv_g_final", dfin)
        M.show_tag("mv_g_pooled", dpo)
        return dfra

def show_eb(eb: EB, +dcls: List<&2, F32>, +toks: List<&2, F32>,
    +demb: List<&2, F32>) -> IO(Unit):
  match eb:
    case EB{dw0, db0, dw1, db1, dw2, db2}:
      do IO<Unit>:
        M.show_tag("mv_g_cls_token", dcls)
        M.show_tag("mv_g_pos_embedding", toks)
        M.show_tag("mv_g_tokens", toks)
        M.show_tag("mv_g_emb", demb)
        M.show_tag("mv_g_to_patch_embedding_0_weight", dw0)
        M.show_tag("mv_g_to_patch_embedding_0_bias", db0)
        M.show_tag("mv_g_to_patch_embedding_1_weight", dw1)
        M.show_tag("mv_g_to_patch_embedding_1_bias", db1)
        M.show_tag("mv_g_to_patch_embedding_2_weight", dw2)
        M.show_tag("mv_g_to_patch_embedding_2_bias", db2)
        return Unit{}

# ---------------------------------------------------------------------------
# the eight blocks

type FW is Type:
  FW{rs: List<&2, F32>, out: List<&2, F32>}

# the forward the backward needs: the running sum r2 of every block (the
# LayerNorm's input) and the last block's output
def fwd_io(+left: Nat, +i: Nat, +x: List<&2, F32>,
    +res: List<&2, F32>, +rs: List<&2, F32>) -> IO(FW):
  match left:
    case 0n:
      do IO<FW>:
        return FW{rs, x}
    case 1n+p:
      do IO<FW>:
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
        +nwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x32(TA.Mat.to_list(1n, 32n, nw)))
        +nbl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x32(TA.Mat.to_list(1n, 32n, nb)))
        +ipl : List<&2, F32> <- IO.pure(List<&2, F32>, ml128x32(TA.Mat.to_list(128n, 32n, ip)))
        +cwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml64x4(TA.Mat.to_list(64n, 4n, cw)))
        +cbl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x64(TA.Mat.to_list(1n, 64n, cb)))
        +xpwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml34x64(TA.Mat.to_list(34n, 64n, xpw)))
        +dpwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml64x2(TA.Mat.to_list(64n, 2n, dpw)))
        +dpbl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x64(TA.Mat.to_list(1n, 64n, dpb)))
        +tibl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x64(TA.Mat.to_list(1n, 64n, tib)))
        +alll : List<&2, F32> <- IO.pure(List<&2, F32>, ml64x16(TA.Mat.to_list(64n, 16n, al)))
        +ddl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x64(TA.Mat.to_list(1n, 64n, dd)))
        +opl : List<&2, F32> <- IO.pure(List<&2, F32>, ml32x64(TA.Mat.to_list(32n, 64n, op)))
        +r2 : List<&2, F32> <- IO.pure(List<&2, F32>, res2_of(x, res))
        +o : List<&2, F32> <- IO.pure(List<&2, F32>,
          bf_out(blk_fw(r2, nwl, nbl, ipl,
            col_go(64n, 0n, cwl, Nil{}), col_go(64n, 1n, cwl, Nil{}),
            col_go(64n, 2n, cwl, Nil{}), col_go(64n, 3n, cwl, Nil{}),
            cbl, xpwl, dpwl, dpbl, tibl, alll, ddl, opl)))
        fw2 : FW <- fwd_io(p, (i + 1n : Nat), o, r2,
          List.append(&2, F32, rs, r2))
        return fw2

# one block: load it, run its forward (for the intermediates), walk back,
# print its twelve gradients and hand out d(LN)
def blk_io(+i: Nat, +r2: List<&2, F32>, +dO: List<&2, F32>,
    +dR: List<&2, F32>) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
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
    +nwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x32(TA.Mat.to_list(1n, 32n, nw)))
    +nbl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x32(TA.Mat.to_list(1n, 32n, nb)))
    +ipl : List<&2, F32> <- IO.pure(List<&2, F32>, ml128x32(TA.Mat.to_list(128n, 32n, ip)))
    +cwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml64x4(TA.Mat.to_list(64n, 4n, cw)))
    +cbl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x64(TA.Mat.to_list(1n, 64n, cb)))
    +xpwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml34x64(TA.Mat.to_list(34n, 64n, xpw)))
    +dpwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml64x2(TA.Mat.to_list(64n, 2n, dpw)))
    +dpbl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x64(TA.Mat.to_list(1n, 64n, dpb)))
    +tibl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x64(TA.Mat.to_list(1n, 64n, tib)))
    +alll : List<&2, F32> <- IO.pure(List<&2, F32>, ml64x16(TA.Mat.to_list(64n, 16n, al)))
    +ddl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x64(TA.Mat.to_list(1n, 64n, dd)))
    +opl : List<&2, F32> <- IO.pure(List<&2, F32>, ml32x64(TA.Mat.to_list(32n, 64n, op)))
    +c0l : List<&2, F32> <- IO.pure(List<&2, F32>, col_go(64n, 0n, cwl, Nil{}))
    +c1l : List<&2, F32> <- IO.pure(List<&2, F32>, col_go(64n, 1n, cwl, Nil{}))
    +c2l : List<&2, F32> <- IO.pure(List<&2, F32>, col_go(64n, 2n, cwl, Nil{}))
    +c3l : List<&2, F32> <- IO.pure(List<&2, F32>, col_go(64n, 3n, cwl, Nil{}))
    bf : BF <- IO.pure(BF, blk_fw(r2, nwl, nbl, ipl, c0l, c1l, c2l, c3l,
      cbl, xpwl, dpwl, dpbl, tibl, alll, ddl, opl))
    bg : BG <- IO.pure(BG, blk_bwd(bf, nwl, nbl, ipl, c0l, c1l, c2l, c3l,
      cbl, xpwl, dpwl, dpbl, tibl, alll, ddl, opl, r2, dO, dR))
    M.show_tag(String.append("mv_g_block", Nat.show(i)), dO)
    u : List<&2, F32> <- show_block(bg, i)
    return u

def bwd_io(+i: Nat, +dO: List<&2, F32>, +dR: List<&2, F32>,
    +rs: List<&2, F32>) -> IO(List<&2, F32>):
  match i:
    case 0n:
      blk_io(0n, M.take_row(rs, 0n, 1600n), dO, dR)
    case 1n+p:
      do IO<List<&2, F32>>:
        +dr : List<&2, F32> <- blk_io((p + 1n : Nat),
          M.take_row(rs, Nat.mul((p + 1n : Nat), 1600n), 1600n), dO, dR)
        u : List<&2, F32> <- bwd_io(p, dr, dr, rs)
        return u

# ---------------------------------------------------------------------------
# the run

def bwd_all(fw: FW, +p1: List<&2, F32>, +toks: List<&2, F32>,
    +dy: List<&2, F32>,
    +pa: List<&2, F32>, +w0l: List<&2, F32>, +b0l: List<&2, F32>,
    +w1l: List<&2, F32>, +w2l: List<&2, F32>, +b2l: List<&2, F32>,
    +nfw: List<&2, F32>, +nfb: List<&2, F32>,
    +hw: List<&2, F32>) -> IO(Unit):
  match fw:
    case FW{+rs, +out}:
      do IO<Unit>:
        tb : TB <- IO.pure(TB, tail_bwd(out,
          M.take_row(rs, Nat.mul(7n, 1600n), 1600n), dy, hw, nfw, nfb))
        +dfra : List<&2, F32> <- show_tail(tb)
        +dts : List<&2, F32> <- bwd_io(7n, dfra, dfra, rs)
        +demb : List<&2, F32> <- IO.pure(List<&2, F32>,
          M.take_row(dts, 32n, 1568n))
        ebv : EB <- IO.pure(EB, emb_bwd(pa, p1, demb, w0l, w1l, w2l))
        show_eb(ebv, M.take_row(dts, 0n, 32n), dts, demb)

def run_all(emp: EMP, +pa: List<&2, F32>, +dy: List<&2, F32>,
    +w0l: List<&2, F32>, +b0l: List<&2, F32>, +w1l: List<&2, F32>,
    +w2l: List<&2, F32>, +b2l: List<&2, F32>,
    +nfw: List<&2, F32>, +nfb: List<&2, F32>,
    +hw: List<&2, F32>) -> IO(Unit):
  match emp:
    case EMP{+p1, +toks}:
      do IO<Unit>:
        fw : FW <- fwd_io(8n, 0n, toks, Nil{}, Nil{})
        u : Unit <- bwd_all(fw, p1, toks, dy, pa, w0l, b0l, w1l, w2l, b2l,
          nfw, nfb, hw)
        return u

def main() -> IO(Unit):
  do IO<Unit>:
    im : TA.Mat<28n, 28n> <- T.Mat.load(28n, 28n, String.append(ddir(), "/img.bin"), 0)
    w0 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/w_to_patch_embedding_0_weight.bin"), 0)
    b0 : TA.Mat<1n, 16n> <- T.Mat.load(1n, 16n, String.append(ddir(), "/w_to_patch_embedding_0_bias.bin"), 0)
    w1 : TA.Mat<32n, 16n> <- T.Mat.load(32n, 16n, String.append(ddir(), "/w_to_patch_embedding_1_weight.bin"), 0)
    w2 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_to_patch_embedding_2_weight.bin"), 0)
    b2 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_to_patch_embedding_2_bias.bin"), 0)
    cls : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_cls_token.bin"), 0)
    pos : TA.Mat<50n, 32n> <- T.Mat.load(50n, 32n, String.append(ddir(), "/w_pos_embedding.bin"), 0)
    nfw : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_norm_f_weight.bin"), 0)
    nfb : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(ddir(), "/w_norm_f_bias.bin"), 0)
    hw : TA.Mat<10n, 32n> <- T.Mat.load(10n, 32n, String.append(ddir(), "/w_mlp_head_weight.bin"), 0)
    dy : TA.Mat<1n, 10n> <- T.Mat.load(1n, 10n, String.append(ddir(), "/dy.bin"), 0)
    +dyl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x10(TA.Mat.to_list(1n, 10n, dy)))
    +w0l : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x16(TA.Mat.to_list(1n, 16n, w0)))
    +b0l : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x16(TA.Mat.to_list(1n, 16n, b0)))
    +w1l : List<&2, F32> <- IO.pure(List<&2, F32>, ml32x16(TA.Mat.to_list(32n, 16n, w1)))
    +w2l : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x32(TA.Mat.to_list(1n, 32n, w2)))
    +b2l : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x32(TA.Mat.to_list(1n, 32n, b2)))
    +clsl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x32(TA.Mat.to_list(1n, 32n, cls)))
    +posl : List<&2, F32> <- IO.pure(List<&2, F32>, ml50x32(TA.Mat.to_list(50n, 32n, pos)))
    +nfwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x32(TA.Mat.to_list(1n, 32n, nfw)))
    +nfbl : List<&2, F32> <- IO.pure(List<&2, F32>, ml1x32(TA.Mat.to_list(1n, 32n, nfb)))
    +hwl : List<&2, F32> <- IO.pure(List<&2, F32>, ml10x32(TA.Mat.to_list(10n, 32n, hw)))
    +pa : List<&2, F32> <- IO.pure(List<&2, F32>,
      patches_go(49n, 0n, ml28(TA.Mat.to_list(28n, 28n, im)), Nil{}))
    emp : EMP <- IO.pure(EMP, emb_all(pa, w0l, b0l, w1l, w2l, b2l, clsl, posl))
    run_all(emp, pa, dyl, w0l, b0l, w1l, w2l, b2l, nfwl, nfbl, hwl)
