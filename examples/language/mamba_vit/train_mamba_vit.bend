# MambaViT A3: the subset trainer — A1b forward + A2 backward + AdamW.
#
# The forward and the backward below are `probe_mamba_vit_emb.bend` (A1b) and
# `probe_mamba_vit_bwd.bend` (A2), copied unchanged except that the weights
# arrive as slices of one flat parameter list instead of `T.Mat.load` calls,
# and that the gradient of each block rides a prepend chain instead of being
# printed.  The scan's two-pass shape (the forward the backward needs, then
# the backward) is the gated one: this program trains the model those gates
# proved, not a re-derivation of it.
#
# Shape and data: MNIST, 28x28, patch 4, dim 32, n_layer 8, d_state 16,
# d_conv 4, expand 2, mean pool, 83,050 parameters.  The data, the seed-0
# init and the budget are `.scratch/mv_mnist/` (manifest.json):
#
#   the 2048-image subset is the fixed slice; this run trains on its first
#   160 images (5 steps of 32) and evaluates on the first 256 held-out
#   images. Measured 2026-11-15 (`reference/test_mamba_vit_train.py`):
#   1869.8 s wall for the 160 training + 256 held-out images (4.49 s per
#   image), so the 2048-image subset is on the order of hours and a
#   60,000-image epoch on the order of days
#
# Schedule, identical to `reference/mamba_vit_twin_train.py`: AdamW lr 1e-3,
# weight_decay 0.01, betas (0.9, 0.999), eps 1e-8, gradient clipping by global
# norm at 1.0, constant lr, batch 32 in file order (no shuffling), and the
# step order forward -> loss -> backward -> clip -> update.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../mamba3/mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/mv_mnist"

def steps_n() -> Nat:
  5n

def batch_n() -> Nat:
  32n

def train_n() -> Nat:
  160n

def test_n() -> Nat:
  256n

# one MNIST image is 784 floats, one batch 32 images: 25088 floats, 100352 bytes
def img_batch_n() -> Nat:
  25088n

def img_batch_bytes() -> U32:
  100352

# the flat parameter list: 83,050 = 6 (patch) + 1600 (pos) + 32 (cls)
# + 8 * 10048 (blocks) + 394 (final norm + head)
def params_n() -> Nat:
  83050n

# the flat order, as offsets
#   pos 0, cls 1600, patch embed 1632..2271, layers.i at 2272 + 10048*i
#   (norm.weight 0, norm.bias 32, A_log 64, D 1088, dt_bias 1152,
#    in_proj 1216, conv1d.weight 5312, conv1d.bias 5568, x_proj 5632,
#    dt_proj.weight 7808, dt_proj.bias 7936, out_proj 8000)
#   norm_f.weight 82656, norm_f.bias 82688, mlp_head.weight 82720,
#   mlp_head.bias 83040
def blk_base(+i: Nat) -> Nat:
  Nat.add(2272n, Nat.mul(i, 10048n))

def ml_flat(m: TA.ML<83050n, 1n>) -> List<&2, F32>:
  match m:
    case TA.ML{d, xs}:
      xs

def ml_batch(m: TA.ML<25088n, 1n>) -> List<&2, F32>:
  match m:
    case TA.ML{d, xs}:
      xs

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
      ip2: List<&2, F32>, dad: List<&2, F32>, dhacc: List<&2, F32>,
      hsr: List<&2, F32>, hpn: List<&2, F32>}

# The state rows the backward reads are carried, not indexed: `hsr` is the
# remaining per-token state list (row 0 is h_t) and `hpn` is the row one ahead
# (h_{t-1}), empty at the last token where the probe's `hp_of` returns zeros.
# The two random reads walked up to 50x1024 values per token and dominated
# this loop; the carry makes each token's reads three 1024-row steps.
def hp_fix(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      M.zline(1024n)
    case Con{a, b}:
      Con{a, b}

def bwk_step(+t: Nat, +h: List<&2, F32>, +xz: List<&2, F32>,
    +cvs: List<&2, F32>, +xs: List<&2, F32>, +dbl: List<&2, F32>,
    +dts: List<&2, F32>, +zs: List<&2, F32>, +yv: List<&2, F32>,
    +yg: List<&2, F32>, +A: List<&2, F32>,
    +ip: List<&2, F32>, +c0f: List<&2, F32>, +c1f: List<&2, F32>,
    +c2f: List<&2, F32>, +c3f: List<&2, F32>, +cb: List<&2, F32>,
    +xpw: List<&2, F32>, +wdt: List<&2, F32>, +bdt: List<&2, F32>,
    +dtb: List<&2, F32>, +D: List<&2, F32>, +wop: List<&2, F32>,
    +dO: List<&2, F32>, k: BWK) -> BWK:
  match k:
    case BWK{+G, +d1, +d2, +d3, +wop2, +xpw2, +wdt2, +bdt2, +dtb2, +dd,
        +cc0, +cc1, +cc2, +cc3, +cb2, +ip2, +dad, +dhacc, +hsr, hpn}:
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
      +hs_t = M.take_row(hsr, 0n, 1024n)
      +hp_t = hp_fix(hpn)
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
        List.append(&2, F32, dh_t, dhacc),
        List.drop(&2, F32, hsr, 1024n), hs_t}

def bwk_go(+left: Nat, +t: Nat, +h: List<&2, F32>,
    +xz: List<&2, F32>, +cvs: List<&2, F32>,
    +xs: List<&2, F32>, +dbl: List<&2, F32>,
    +dts: List<&2, F32>, +zs: List<&2, F32>,
    +yv: List<&2, F32>, +yg: List<&2, F32>,
    +A: List<&2, F32>,
    +ip: List<&2, F32>, +c0f: List<&2, F32>, +c1f: List<&2, F32>,
    +c2f: List<&2, F32>, +c3f: List<&2, F32>, +cb: List<&2, F32>,
    +xpw: List<&2, F32>, +wdt: List<&2, F32>, +bdt: List<&2, F32>,
    +dtb: List<&2, F32>, +D: List<&2, F32>, +wop: List<&2, F32>,
    +dO: List<&2, F32>, k: BWK) -> BWK:
  match left:
    case 0n:
      k
    case 1n+p:
      bwk_go(p, (t - 1n : Nat), h, xz, cvs, xs, dbl, dts, zs, yv, yg,
        A, ip, c0f, c1f, c2f, c3f, cb, xpw, wdt, bdt, dtb, D, wop, dO,
        bwk_step(t, h, xz, cvs, xs, dbl, dts, zs, yv, yg, A, ip,
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
        +cc0, +cc1, +cc2, +cc3, +cb2, +ip2, +dad, +dhacc, _, _}:
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
    case BF{h, xz, cvs, xs, dbl, dts, zs, yv, yg, +hs, out}:
      bw_fin(bwk_go(50n, 49n, h, xz, cvs, xs, dbl, dts, zs, yv, yg,
        negexp_row(al), ip, c0f, c1f, c2f, c3f, cb, xpw, wdt, bdt, dtb,
        D, wop, dO,
        BWK{M.zline(1024n), M.zline(64n), M.zline(64n), M.zline(64n),
          M.zline(2048n), M.zline(2176n), M.zline(128n), M.zline(64n),
          M.zline(64n), M.zline(64n), M.zline(64n), M.zline(64n),
          M.zline(64n), M.zline(64n), M.zline(64n), M.zline(4096n),
          M.zline(1024n), Nil{}, hs, M.take_row(hs, 1024n, 1024n)}),
        negexp_row(al), res2, nw, dR)

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
# the flat parameter slices, the loss, the argmax, and the record steps
#

type BP is Type:
  BP{nw: List<&2, F32>, nb: List<&2, F32>, ip: List<&2, F32>,
     c0: List<&2, F32>, c1: List<&2, F32>, c2: List<&2, F32>,
     c3: List<&2, F32>, cb: List<&2, F32>, xpw: List<&2, F32>,
     wdt: List<&2, F32>, bdt: List<&2, F32>, dtb: List<&2, F32>,
     al: List<&2, F32>, dd: List<&2, F32>, wop: List<&2, F32>}

# one image's forward: every block's running sum (the LayerNorm's input, in
# block order, 8 * 1600) and the last block's output
type FW is Type:
  FW{rs: List<&2, F32>, out: List<&2, F32>}

# the block backward's carry out: the flat gradient so far and d(tokens)
type BR is Type:
  BR{acc: List<&2, F32>, dts: List<&2, F32>}

# one image's gradient, its loss and its hit
type IR is Type:
  IR{grads: List<&2, F32>, loss: F32, hit: Nat}

# the step's running sums: the flat gradient, the summed loss, the hits
type AC is Type:
  AC{grads: List<&2, F32>, loss: F32, hits: Nat}

# the trainer's state: the parameters, AdamW's two moments, the per-step
# losses and gradient norms, and the train hits
type TR is Type:
  TR{ps: List<&2, F32>, ms: List<&2, F32>, vs: List<&2, F32>,
     ls: List<&2, F32>, ns: List<&2, F32>, hits: Nat}

# `match` scrutinises a parameter, never a computed value (README idiom 1), so
# every record is handled by a def that takes it as a parameter: `bp_step` for
# a block's weights, `bg_step` for its gradients, `tail_step`/`br_step`/
# `eb_step` for the three halves of the backward, `ac_step` for the running
# accumulator, `emp_step`/`fw_step`/`ce_step` for one image's forward.

# acc with xs in front: the flat gradient is built by prepending each block's
# tensors in the reverse of the canonical order, so one pass of `List.append`
# over a small row never walks the whole 83,050.
# ---------------------------------------------------------------------------
# the patch embedding's forward, with the linear's bias
#
# A1b/A2 ran on the seed-0 dump, where `to_patch_embedding.1.bias` is zero, so
# their forward adds nothing.  A3 trains from that same init, and after the
# first update that bias is no longer zero (`db1` is the sum of the next
# LayerNorm's input gradients), so the forward has to add it.  The backward
# does not change: a constant added to a LayerNorm's input shifts its mean and
# not its variance or its centred values, so `lnr_bwd` on the unbiased `pre`
# gives the same dW, db1 and dp1.

type EP is Type:
  EP{p1: List<&2, F32>, emb: List<&2, F32>}

def emb_step(+pa: List<&2, F32>, +w0: List<&2, F32>,
    +b0: List<&2, F32>, +w1: List<&2, F32>, +b1: List<&2, F32>,
    +w2: List<&2, F32>, +b2: List<&2, F32>, e: EP) -> EP:
  match e:
    case EP{p1s, emb0}:
      +p1 = ln_row(pa, w0, b0)
      +p2 = ln_row(M.add_lists(M.matvec(w1, p1, 16n), b1), w2, b2)
      EP{M.push_row(p1s, p1), M.push_row(emb0, p2)}

def emb_go(+left: Nat, +t: Nat, +pa: List<&2, F32>,
    +w0: List<&2, F32>, +b0: List<&2, F32>,
    +w1: List<&2, F32>, +b1: List<&2, F32>,
    +w2: List<&2, F32>, +b2: List<&2, F32>, e: EP) -> EP:
  match left:
    case 0n:
      e
    case 1n+p:
      emb_go(p, (t + 1n : Nat), pa, w0, b0, w1, b1, w2, b2,
        emb_step(row(t, pa, 16n), w0, b0, w1, b1, w2, b2, e))

type EMP is Type:
  EMP{p1: List<&2, F32>, toks: List<&2, F32>}

def emb_fin(e: EP, +cls: List<&2, F32>,
    +pos: List<&2, F32>) -> EMP:
  match e:
    case EP{p1s, emb}:
      EMP{p1s, add_pos(50n, 0n, List.append(&2, F32, cls, emb), pos, Nil{})}

def emb_all(+pa: List<&2, F32>, +w0: List<&2, F32>,
    +b0: List<&2, F32>, +w1: List<&2, F32>, +b1: List<&2, F32>,
    +w2: List<&2, F32>, +b2: List<&2, F32>,
    +cls: List<&2, F32>, +pos: List<&2, F32>) -> EMP:
  emb_fin(emb_go(49n, 0n, pa, w0, b0, w1, b1, w2, b2, EP{Nil{}, Nil{}}),
    cls, pos)


# the patch embedding's backward, with the linear's bias
#
# The LayerNorm here is over the 16 patch values, so its bias is NOT a
# constant shift of the row: `pre = W1 p1 + b1` and `pre` alone give different
# centred values, hence different dW2, db2 and dp.  The A2 dump's b1 is zero,
# so `emb_bwd` there drops it; A3 trains from the same init and the first
# update moves it, so the biased pre is what this backward takes.
def eb_go(+left: Nat, +t: Nat, +pa: List<&2, F32>,
    +p1: List<&2, F32>, +demb: List<&2, F32>,
    +w0: List<&2, F32>, +w1: List<&2, F32>, +b1: List<&2, F32>,
    +w2: List<&2, F32>, a: EB) -> EB:
  match left:
    case 0n:
      a
    case 1n+p:
      eb_go(p, (t + 1n : Nat), pa, p1, demb, w0, w1, b1, w2,
        eb_step(lnr_bwd(M.add_lists(M.matvec(w1, row(t, p1, 16n), 16n), b1),
          w2, row(t, demb, 32n)), row(t, pa, 16n), row(t, p1, 16n),
          w0, w1, a))

def emb_bwd(+pa: List<&2, F32>, +p1: List<&2, F32>,
    +demb: List<&2, F32>, +w0: List<&2, F32>,
    +w1: List<&2, F32>, +b1: List<&2, F32>,
    +w2: List<&2, F32>) -> EB:
  eb_go(49n, 0n, pa, p1, demb, w0, w1, b1, w2,
    EB{M.zline(16n), M.zline(16n), M.zline(512n), M.zline(32n),
      M.zline(32n), M.zline(32n)})

def gpre(+acc: List<&2, F32>, +xs: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, xs, acc)

# one block's weights, sliced out of `ps` at the block's own base offset
def bp_at(+sub: List<&2, F32>) -> BP:
  +cwl = M.take_row(sub, 5312n, 256n)
  BP{M.take_row(sub, 0n, 32n),
    M.take_row(sub, 32n, 32n),
    M.take_row(sub, 1216n, 4096n),
    col_go(64n, 0n, cwl, Nil{}), col_go(64n, 1n, cwl, Nil{}),
    col_go(64n, 2n, cwl, Nil{}), col_go(64n, 3n, cwl, Nil{}),
    M.take_row(sub, 5568n, 64n),
    M.take_row(sub, 5632n, 2176n),
    M.take_row(sub, 7808n, 128n),
    M.take_row(sub, 7936n, 64n),
    M.take_row(sub, 1152n, 64n),
    M.take_row(sub, 64n, 1024n),
    M.take_row(sub, 1088n, 64n),
    M.take_row(sub, 8000n, 2048n)}

def bp_of(+ps: List<&2, F32>, +i: Nat) -> BP:
  bp_at(List.drop(&2, F32, ps, blk_base(i)))

# the ViT tail's forward: one more add, the final LayerNorm, the mean pool, the
# head (the A1b `fin_all`, without the FIN record the gate needs)
def tail_fw(+out: List<&2, F32>, +r7: List<&2, F32>,
    +nfw: List<&2, F32>, +nfb: List<&2, F32>,
    +hw: List<&2, F32>, +hb: List<&2, F32>) -> List<&2, F32>:
  +fra = M.add_lists(out, r7)
  +fn = ln_rows(50n, 0n, fra, nfw, nfb, 32n, Nil{})
  +pl = div_list(pool_go(50n, 0n, fn, 32n, M.zline(32n)), 50.0)
  M.add_lists(M.matvec(hw, pl, 32n), hb)

# ---------------------------------------------------------------------------
# cross entropy: loss = log(sum exp(l - m)) + m - l[y],
# dlogits = (softmax - onehot) / batch, the mean over the batch the twin's
# F.cross_entropy takes

type CE is Type:
  CE{loss: F32, dl: List<&2, F32>}

def mx_go(+left: Nat, +xs: List<&2, F32>, m: F32) -> F32:
  match left xs:
    case 0n _:
      m
    case _ Nil{}:
      m
    case 1n+p Con{h, t}:
      mx_go(p, t, F32.max(m, h))

def esh_go(+left: Nat, +xs: List<&2, F32>, +m: F32,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n _:
      acc
    case _ Nil{}:
      acc
    case 1n+p Con{h, t}:
      esh_go(p, t, m, M.push_row(acc, Con{F32.exp((h - m : F32)), Nil{}}))

def pdiv_go(+left: Nat, +xs: List<&2, F32>, +s: F32,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n _:
      acc
    case _ Nil{}:
      acc
    case 1n+p Con{h, t}:
      pdiv_go(p, t, s, M.push_row(acc, Con{(h / s : F32), Nil{}}))

def eqf(b: Bool, +v: F32) -> F32:
  match b:
    case True{}:
      v
    case False{}:
      0.0

def dl_go(+left: Nat, +i: Nat, +ps: List<&2, F32>, +lab: Nat, +c: F32,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left ps:
    case 0n _:
      acc
    case _ Nil{}:
      acc
    case 1n+p Con{h, t}:
      dl_go(p, (i + 1n : Nat), t, lab, c,
        M.push_row(acc, Con{((c * h : F32) - eqf(Nat.is_eq(i, lab), c) : F32),
          Nil{}}))

def ce_of(+logits: List<&2, F32>, +lab: Nat) -> CE:
  +m = mx_go(10n, logits, M.nth(0n, logits))
  +ex = esh_go(10n, logits, m, Nil{})
  +s = M.sum_list(ex)
  CE{((F32.log(s) + m : F32) - M.nth(lab, logits) : F32),
    dl_go(10n, 0n, pdiv_go(10n, ex, s, Nil{}), lab,
      (1.0 / F32.from_nat(batch_n()) : F32), Nil{})}

# argmax over the ten logits (the first index wins a tie, as torch's argmax)
def am_i(b: Bool, +i: Nat, bi: Nat) -> Nat:
  match b:
    case True{}:
      i
    case False{}:
      bi

def am_v(b: Bool, +h: F32, bv: F32) -> F32:
  match b:
    case True{}:
      h
    case False{}:
      bv

def am_go(+left: Nat, +i: Nat, +xs: List<&2, F32>, bi: Nat,
    +bv: F32) -> Nat:
  match left xs:
    case 0n _:
      bi
    case _ Nil{}:
      bi
    case 1n+p Con{h, t}:
      am_go(p, (i + 1n : Nat), t, am_i(F32.is_gt(h, bv), i, bi),
        am_v(F32.is_gt(h, bv), h, bv))

def argmax(+xs: List<&2, F32>) -> Nat:
  am_go(10n, 0n, xs, 0n, M.nth(0n, xs))

def hit_if(b: Bool) -> Nat:
  match b:
    case True{}:
      1n
    case False{}:
      0n

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def head_of(+xs: List<&2, Nat>) -> Nat:
  match xs:
    case Con{h, t}:
      h
    case Nil{}:
      0n

def tail_of(+xs: List<&2, Nat>) -> List<&2, Nat>:
  match xs:
    case Con{h, t}:
      t
    case Nil{}:
      Nil{}


# ---------------------------------------------------------------------------
# one image: the A1b forward, the cross entropy, the A2 backward, and the flat
# gradient in the canonical order
#
#   pos, cls, to_patch_embedding (w0 b0 w1 b1 w2 b2),
#   layers.0..7 (norm.weight, norm.bias, A_log, D, dt_bias, in_proj,
#                conv1d.weight, conv1d.bias, x_proj, dt_proj.weight,
#                dt_proj.bias, out_proj),
#   norm_f.weight, norm_f.bias, mlp_head.weight, mlp_head.bias
#
# No def may be used before it is defined, so a block's weight slice is
# matched in a leaf (`bp_fw`, `bp_cat`) that the loop calls, and the block
# backward hands back ONE flat list (its 12 gradient tensors in the canonical
# order, then d(tokens)) so the loop never has to take a record apart.

def bp_fw(bp: BP, +r2: List<&2, F32>) -> BF:
  match bp:
    case BP{nw, nb, ip, c0, c1, c2, c3, cb, xpw, wdt, bdt, dtb, al, dd, wop}:
      blk_fw(r2, nw, nb, ip, c0, c1, c2, c3, cb, xpw, wdt, bdt, dtb, al, dd,
        wop)

def blk_fw_ps(+ps: List<&2, F32>, +i: Nat,
    +r2: List<&2, F32>) -> BF:
  bp_fw(bp_of(ps, i), r2)

# a block's gradients as one list: 12 tensors in the canonical order, then
# d(tokens) so the loop can carry it on
def bg_cat(bg: BG) -> List<&2, F32>:
  match bg:
    case BG{dnw, dnb, dip, dc0, dc1, dc2, dc3, dcb, dxpw, dwdt, dbdt,
        ddtb, dal, dD, dop, dres}:
      List.append(&2, F32,
        M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(
          M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(
            M.push_row(M.push_row(Nil{}, dnw), dnb), dal), dD),
            ddtb), dip),
            kcat_go(64n, 0n, dc0, dc1, dc2, dc3, Nil{})), dcb), dxpw),
            dwdt), dbdt), dop),
        dres)

def bp_cat(bp: BP, +r2: List<&2, F32>, +dO: List<&2, F32>,
    +dR: List<&2, F32>) -> List<&2, F32>:
  match bp:
    case BP{+nw, +nb, +ip, +c0, +c1, +c2, +c3, +cb, +xpw, +wdt, +bdt, +dtb,
        +al, +dd, +wop}:
      bf = blk_fw(r2, nw, nb, ip, c0, c1, c2, c3, cb, xpw, wdt, bdt, dtb,
        al, dd, wop)
      bg_cat(blk_bwd(bf, nw, nb, ip, c0, c1, c2, c3, cb, xpw, wdt, bdt, dtb,
        al, dd, wop, r2, dO, dR))

def blk_bwd_cat(+ps: List<&2, F32>, +i: Nat, +r2: List<&2, F32>,
    +dO: List<&2, F32>, +dR: List<&2, F32>) -> List<&2, F32>:
  bp_cat(bp_of(ps, i), r2, dO, dR)

# the eight blocks' forward: each block's running sum is appended (it is the
# LayerNorm's input the backward needs), the last output is the block's out
def fw_go(+left: Nat, +i: Nat, +ps: List<&2, F32>, +x: List<&2, F32>,
    +res: List<&2, F32>, +rs: List<&2, F32>) -> FW:
  match left:
    case 0n:
      FW{rs, x}
    case 1n+p:
      +r2 = res2_of(x, res)
      +o = bf_out(blk_fw_ps(ps, i, r2))
      fw_go(p, (i + 1n : Nat), ps, o, r2, List.append(&2, F32, rs, r2))

# blocks 7 down to 0: the block's forward (the backward reads its per-token
# intermediates), the backward, and its gradient prepended to the flat list
def bw_go(+left: Nat, +i: Nat, +ps: List<&2, F32>,
    +rs: List<&2, F32>, +dO: List<&2, F32>, +dR: List<&2, F32>,
    acc: List<&2, F32>) -> BR:
  match left:
    case 0n:
      BR{acc, dO}
    case 1n+p:
      +cat = blk_bwd_cat(ps, i, M.take_row(rs, Nat.mul(i, 1600n), 1600n),
        dO, dR)
      +nx = M.take_row(cat, 10048n, 1600n)
      bw_go(p, (i - 1n : Nat), ps, rs, nx, nx,
        gpre(acc, M.take_row(cat, 0n, 10048n)))

# the three halves of the backward, each a leaf that takes its record as a
# parameter: the tail (head, pool, final LayerNorm), the block loop's result,
# and the patch embedding
def eb_gacc(eb: EB, +acc: List<&2, F32>) -> List<&2, F32>:
  match eb:
    case EB{dw0, db0, dw1, db1, dw2, db2}:
      gpre(gpre(gpre(gpre(gpre(gpre(acc, db2), dw2), db1), dw1), db0), dw0)

def br_step(br: BR, +ps: List<&2, F32>, +pa: List<&2, F32>,
    +p1: List<&2, F32>) -> List<&2, F32>:
  match br:
    case BR{acc, +dts}:
      +demb = M.take_row(dts, 32n, 1568n)
      +hd = List.take(&2, F32, ps, 2272n)
      +w0 = M.take_row(hd, 1632n, 16n)
      +w1 = M.take_row(hd, 1664n, 512n)
      +b1 = M.take_row(hd, 2176n, 32n)
      +w2 = M.take_row(hd, 2208n, 32n)
      +a1 = eb_gacc(emb_bwd(pa, p1, demb, w0, w1, b1, w2), acc)
      gpre(gpre(a1, M.take_row(dts, 0n, 32n)), dts)

def tail_step(tb: TB, +ps: List<&2, F32>, +rs: List<&2, F32>,
    +pa: List<&2, F32>, +p1: List<&2, F32>) -> List<&2, F32>:
  match tb:
    case TB{dwh, dhb, dnfw, dnfb, _, _, +dfra}:
      +a0 = gpre(gpre(gpre(gpre(Nil{}, dhb), dwh), dnfb), dnfw)
      br_step(bw_go(8n, 7n, ps, rs, dfra, dfra, a0), ps, pa, p1)

def img_grads(+ps: List<&2, F32>, +rs: List<&2, F32>,
    +out: List<&2, F32>, +dy: List<&2, F32>,
    +pa: List<&2, F32>, +p1: List<&2, F32>) -> List<&2, F32>:
  +tl = List.drop(&2, F32, ps, 82656n)
  +nfw = M.take_row(tl, 0n, 32n)
  +nfb = M.take_row(tl, 32n, 32n)
  +hw = M.take_row(tl, 64n, 320n)
  +hb = M.take_row(tl, 384n, 10n)
  tail_step(tail_bwd(out, M.take_row(rs, 11200n, 1600n), dy, hw, nfw, nfb),
    ps, rs, pa, p1)

# one image end to end: patches, the embedding, the eight blocks, the tail,
# the loss, the backward
def ce_step(ce: CE, +ps: List<&2, F32>, +rs: List<&2, F32>,
    +out: List<&2, F32>, +pa: List<&2, F32>, +p1: List<&2, F32>,
    +lg: List<&2, F32>, +lab: Nat) -> IR:
  match ce:
    case CE{loss, dl}:
      IR{img_grads(ps, rs, out, dl, pa, p1), loss,
        hit_if(Nat.is_eq(argmax(lg), lab))}

def fw_step(fw: FW, +ps: List<&2, F32>, +pa: List<&2, F32>,
    +p1: List<&2, F32>, +lab: Nat, +nfw: List<&2, F32>,
    +nfb: List<&2, F32>, +hw: List<&2, F32>,
    +hb: List<&2, F32>) -> IR:
  match fw:
    case FW{+rs, +out}:
      +lg = tail_fw(out, M.take_row(rs, 11200n, 1600n), nfw, nfb, hw, hb)
      ce_step(ce_of(lg, lab), ps, rs, out, pa, p1, lg, lab)

def emp_step(emp: EMP, +ps: List<&2, F32>, +pa: List<&2, F32>,
    +lab: Nat, +nfw: List<&2, F32>, +nfb: List<&2, F32>,
    +hw: List<&2, F32>, +hb: List<&2, F32>) -> IR:
  match emp:
    case EMP{p1, toks}:
      fw_step(fw_go(8n, 0n, ps, toks, Nil{}, Nil{}), ps, pa, p1, lab,
        nfw, nfb, hw, hb)

def img_run(+ps: List<&2, F32>, +img: List<&2, F32>,
    +lab: Nat) -> IR:
  +hd = List.take(&2, F32, ps, 2272n)
  +pos = M.take_row(hd, 0n, 1600n)
  +cls = M.take_row(hd, 1600n, 32n)
  +w0 = M.take_row(hd, 1632n, 16n)
  +b0 = M.take_row(hd, 1648n, 16n)
  +w1 = M.take_row(hd, 1664n, 512n)
  +b1 = M.take_row(hd, 2176n, 32n)
  +w2 = M.take_row(hd, 2208n, 32n)
  +b2 = M.take_row(hd, 2240n, 32n)
  +tl = List.drop(&2, F32, ps, 82656n)
  +nfw = M.take_row(tl, 0n, 32n)
  +nfb = M.take_row(tl, 32n, 32n)
  +hw = M.take_row(tl, 64n, 320n)
  +hb = M.take_row(tl, 384n, 10n)
  +pa = patches_go(49n, 0n, img, Nil{})
  emp_step(emb_all(pa, w0, b0, w1, b1, w2, b2, cls, pos), ps, pa, lab,
    nfw, nfb, hw, hb)

# the 32 images of one step: the gradient is summed, the loss added, the hits
# counted
def ac_step(ir: IR, a: AC) -> AC:
  match ir a:
    case IR{g, ls, hit} AC{gr, lo, hi}:
      AC{M.add_lists(gr, g), (lo + ls : F32), Nat.add(hi, hit)}

def imgs_go(+left: Nat, +ps: List<&2, F32>, +imgs: List<&2, F32>,
    +labs: List<&2, Nat>, a: AC) -> AC:
  match left:
    case 0n:
      a
    case 1n+p:
      imgs_go(p, ps, List.drop(&2, F32, imgs, 784n), tail_of(labs),
        ac_step(img_run(ps, List.take(&2, F32, imgs, 784n), head_of(labs)), a))


# ---------------------------------------------------------------------------
# AdamW and the clipping, in torch's order:
#
#   coef = 1/(norm + 1e-6) if norm > 1.0        (clip_grad_norm_)
#   p = p * (1 - lr*wd)
#   m = b1*m + (1-b1)*g            v = b2*v + (1-b2)*g*g
#   denom = sqrt(v)/sqrt(1-b2^t) + eps
#   p = p - (lr/(1-b1^t)) * m / denom

def beta1() -> F32:
  0.9

def beta2() -> F32:
  0.999

def lr() -> F32:
  0.001

def wd() -> F32:
  0.01

def eps() -> F32:
  0.00000001

def clip() -> F32:
  1.0

def pw_go(+left: Nat, +b: F32, acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+p:
      pw_go(p, b, (acc * b : F32))

def pw(+t: Nat, +b: F32) -> F32:
  pw_go(t, b, 1.0)

def clip_coef(b: Bool, +nrm: F32) -> F32:
  match b:
    case True{}:
      (1.0 / (nrm + 0.000001 : F32) : F32)
    case False{}:
      1.0

def m_go(+left: Nat, +g: List<&2, F32>, +m: List<&2, F32>,
    +b1: F32, +ob1: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match left g m:
    case 0n _ _:
      List.reverse(&2, F32, acc)
    case _ Nil{} _:
      List.reverse(&2, F32, acc)
    case _ _ Nil{}:
      List.reverse(&2, F32, acc)
    case 1n+p Con{gh, gt} Con{mh, mt}:
      m_go(p, gt, mt, b1, ob1,
        Con{((b1 * mh : F32) + (ob1 * gh : F32) : F32), acc})

def v_go(+left: Nat, +g: List<&2, F32>, +v: List<&2, F32>,
    +b2: F32, +ob2: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match left g v:
    case 0n _ _:
      List.reverse(&2, F32, acc)
    case _ Nil{} _:
      List.reverse(&2, F32, acc)
    case _ _ Nil{}:
      List.reverse(&2, F32, acc)
    case 1n+p Con{gh, gt} Con{vh, vt}:
      v_go(p, gt, vt, b2, ob2,
        Con{((b2 * vh : F32) + (ob2 * (gh * gh : F32) : F32) : F32), acc})

def p_go(+left: Nat, +p: List<&2, F32>, +m: List<&2, F32>,
    +v: List<&2, F32>, +dec: F32, +sc: F32, +b2r: F32, +ep: F32,
    acc: List<&2, F32>) -> List<&2, F32>:
  match left p m v:
    case 0n _ _ _:
      List.reverse(&2, F32, acc)
    case _ Nil{} _ _:
      List.reverse(&2, F32, acc)
    case _ _ Nil{} _:
      List.reverse(&2, F32, acc)
    case _ _ _ Nil{}:
      List.reverse(&2, F32, acc)
    case 1n+p Con{ph, pt} Con{mh, mt} Con{vh, vt}:
      +den = ((F32.sqrt(vh) / b2r : F32) + ep : F32)
      p_go(p, pt, mt, vt, dec, sc, b2r, ep,
        Con{(((ph * dec : F32) - ((sc * mh : F32) / den : F32)) : F32), acc})

# the step accumulator as one flat list: the 83,050 gradients, the summed
# loss, the hits -- so the step loop reads plain lists and never has to take a
# record apart (a def that matched AC would have to call the loop, and no def
# may be used before it is defined)
def ac_pack(a: AC) -> List<&2, F32>:
  match a:
    case AC{grads, loss, hits}:
      List.append(&2, F32,
        List.append(&2, F32, grads, Con{loss, Nil{}}),
        Con{F32.from_nat(hits), Nil{}})

# ---------------------------------------------------------------------------
# the step loop: one batch in file order, its 32 images, the clip, the update

def step_io(+left: Nat, +t: Nat, +ps: List<&2, F32>,
    +ms: List<&2, F32>, +vs: List<&2, F32>, +ls: List<&2, F32>,
    +ns: List<&2, F32>, +hits: Nat) -> IO(TR):
  match left:
    case 0n:
      do IO<TR>:
        return TR{ps, ms, vs, ls, ns, hits}
    case 1n+p:
      do IO<TR>:
        ib : TA.Mat<25088n, 1n> <- T.Mat.load(25088n, 1n,
          String.append(ddir(), "/train_images.bin"),
          (U32.from_nat(Nat.mul(t, img_batch_n())) * 4 : U32))
        +imgs : List<&2, F32> <- IO.pure(List<&2, F32>,
          ml_batch(TA.Mat.to_list(25088n, 1n, ib)))
        +lb : List<&2, U32> <- T.read_at(
          String.append(ddir(), "/train_labels.bin"),
          U32.from_nat(Nat.mul(t, batch_n())), U32.from_nat(batch_n()))
        +labs : List<&2, Nat> <- IO.pure(List<&2, Nat>, nats(lb))
        +pk : List<&2, F32> <- IO.pure(List<&2, F32>,
          ac_pack(imgs_go(batch_n(), ps, imgs, labs,
            AC{M.zline(params_n()), 0.0, 0n})))
        +grads : List<&2, F32> <- IO.pure(List<&2, F32>,
          M.take_row(pk, 0n, params_n()))
        +loss : F32 <- IO.pure(F32, M.nth(params_n(), pk))
        +h : Nat <- IO.pure(Nat,
          F32.to_nat(M.nth((params_n() + 1n : Nat), pk)))
        +nrm : F32 <- IO.pure(F32, F32.sqrt(M.sumsq(grads)))
        +coef : F32 <- IO.pure(F32, clip_coef(F32.is_gt(nrm, clip()), nrm))
        +gsc : List<&2, F32> <- IO.pure(List<&2, F32>,
          M.scale_list(grads, coef))
        +tn : Nat <- IO.pure(Nat, Nat.add(t, 1n))
        +b1 : F32 <- IO.pure(F32, beta1())
        +b2 : F32 <- IO.pure(F32, beta2())
        +bc1 : F32 <- IO.pure(F32, (1.0 - pw(tn, b1) : F32))
        +bc2 : F32 <- IO.pure(F32, (1.0 - pw(tn, b2) : F32))
        +m2 : List<&2, F32> <- IO.pure(List<&2, F32>,
          m_go(params_n(), gsc, ms, b1, (1.0 - b1 : F32), Nil{}))
        +v2 : List<&2, F32> <- IO.pure(List<&2, F32>,
          v_go(params_n(), gsc, vs, b2, (1.0 - b2 : F32), Nil{}))
        +p2 : List<&2, F32> <- IO.pure(List<&2, F32>,
          p_go(params_n(), ps, m2, v2, (1.0 - (lr() * wd() : F32) : F32),
            (lr() / bc1 : F32), F32.sqrt(bc2), eps(), Nil{}))
        +ml : F32 <- IO.pure(F32, (loss / F32.from_nat(batch_n()) : F32))
        +hit2 : Nat <- IO.pure(Nat, Nat.add(hits, h))
        IO.print(String.append(String.append(String.append("step ",
          Nat.show(t)), " loss "), String.append(String.append(
          F32.show(ml), " gn "), String.append(F32.show(nrm),
          String.append(" coef ", F32.show(coef))))))
        step_io(p, tn, p2, m2, v2, List.append(&2, F32, ls, Con{ml, Nil{}}),
          List.append(&2, F32, ns, Con{nrm, Nil{}}), hit2)

# ---------------------------------------------------------------------------
# the held-out evaluation: the forward only, one batch of 32 at a time

def eval_fw(fw: FW, +lab: Nat, +nfw: List<&2, F32>,
    +nfb: List<&2, F32>, +hw: List<&2, F32>,
    +hb: List<&2, F32>) -> Nat:
  match fw:
    case FW{rs, out}:
      +lg = tail_fw(out, M.take_row(rs, 11200n, 1600n), nfw, nfb, hw, hb)
      hit_if(Nat.is_eq(argmax(lg), lab))

def eval_emp(emp: EMP, +ps: List<&2, F32>, +lab: Nat,
    +nfw: List<&2, F32>, +nfb: List<&2, F32>,
    +hw: List<&2, F32>, +hb: List<&2, F32>) -> Nat:
  match emp:
    case EMP{_, toks}:
      eval_fw(fw_go(8n, 0n, ps, toks, Nil{}, Nil{}), lab,
        nfw, nfb, hw, hb)

def eval_run(+ps: List<&2, F32>, +img: List<&2, F32>,
    +lab: Nat) -> Nat:
  +hd = List.take(&2, F32, ps, 2272n)
  +pos = M.take_row(hd, 0n, 1600n)
  +cls = M.take_row(hd, 1600n, 32n)
  +w0 = M.take_row(hd, 1632n, 16n)
  +b0 = M.take_row(hd, 1648n, 16n)
  +w1 = M.take_row(hd, 1664n, 512n)
  +b1 = M.take_row(hd, 2176n, 32n)
  +w2 = M.take_row(hd, 2208n, 32n)
  +b2 = M.take_row(hd, 2240n, 32n)
  +tl = List.drop(&2, F32, ps, 82656n)
  +nfw = M.take_row(tl, 0n, 32n)
  +nfb = M.take_row(tl, 32n, 32n)
  +hw = M.take_row(tl, 64n, 320n)
  +hb = M.take_row(tl, 384n, 10n)
  +pa = patches_go(49n, 0n, img, Nil{})
  eval_emp(emb_all(pa, w0, b0, w1, b1, w2, b2, cls, pos), ps, lab,
    nfw, nfb, hw, hb)

def eval_imgs(+left: Nat, +ps: List<&2, F32>, +imgs: List<&2, F32>,
    +labs: List<&2, Nat>, hits: Nat) -> Nat:
  match left:
    case 0n:
      hits
    case 1n+p:
      eval_imgs(p, ps, List.drop(&2, F32, imgs, 784n), tail_of(labs),
        Nat.add(hits, eval_run(ps, List.take(&2, F32, imgs, 784n),
          head_of(labs))))

def eval_io(+left: Nat, +b: Nat, +ps: List<&2, F32>,
    hits: Nat) -> IO(Nat):
  match left:
    case 0n:
      do IO<Nat>:
        return hits
    case 1n+p:
      do IO<Nat>:
        ib : TA.Mat<25088n, 1n> <- T.Mat.load(25088n, 1n,
          String.append(ddir(), "/test_images.bin"),
          (U32.from_nat(Nat.mul(b, img_batch_n())) * 4 : U32))
        +imgs : List<&2, F32> <- IO.pure(List<&2, F32>,
          ml_batch(TA.Mat.to_list(25088n, 1n, ib)))
        +lb : List<&2, U32> <- T.read_at(
          String.append(ddir(), "/test_labels.bin"),
          U32.from_nat(Nat.mul(b, batch_n())), U32.from_nat(batch_n()))
        +labs : List<&2, Nat> <- IO.pure(List<&2, Nat>, nats(lb))
        +h : Nat <- IO.pure(Nat, eval_imgs(batch_n(), ps, imgs, labs, hits))
        eval_io(p, (b + 1n : Nat), ps, h)

# the checksum of every parameter tensor: one tag, the sum of squares of
# the flat slice, in the flat order the manifest names (108 values)
def wsums(+ps: List<&2, F32>) -> List<&2, F32>:
  M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(M.push_row(Nil{},
    Con{M.sumsq(M.take_row(ps, 0n, 1600n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 1600n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 1632n, 16n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 1648n, 16n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 1664n, 512n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 2176n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 2208n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 2240n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 2272n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 2304n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 2336n, 1024n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 3360n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 3424n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 3488n, 4096n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 7584n, 256n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 7840n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 7904n, 2176n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 10080n, 128n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 10208n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 10272n, 2048n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 12320n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 12352n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 12384n, 1024n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 13408n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 13472n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 13536n, 4096n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 17632n, 256n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 17888n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 17952n, 2176n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 20128n, 128n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 20256n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 20320n, 2048n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 22368n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 22400n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 22432n, 1024n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 23456n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 23520n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 23584n, 4096n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 27680n, 256n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 27936n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 28000n, 2176n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 30176n, 128n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 30304n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 30368n, 2048n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 32416n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 32448n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 32480n, 1024n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 33504n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 33568n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 33632n, 4096n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 37728n, 256n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 37984n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 38048n, 2176n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 40224n, 128n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 40352n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 40416n, 2048n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 42464n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 42496n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 42528n, 1024n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 43552n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 43616n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 43680n, 4096n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 47776n, 256n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 48032n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 48096n, 2176n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 50272n, 128n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 50400n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 50464n, 2048n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 52512n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 52544n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 52576n, 1024n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 53600n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 53664n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 53728n, 4096n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 57824n, 256n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 58080n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 58144n, 2176n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 60320n, 128n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 60448n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 60512n, 2048n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 62560n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 62592n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 62624n, 1024n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 63648n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 63712n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 63776n, 4096n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 67872n, 256n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 68128n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 68192n, 2176n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 70368n, 128n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 70496n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 70560n, 2048n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 72608n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 72640n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 72672n, 1024n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 73696n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 73760n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 73824n, 4096n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 77920n, 256n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 78176n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 78240n, 2176n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 80416n, 128n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 80544n, 64n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 80608n, 2048n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 82656n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 82688n, 32n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 82720n, 320n)), Nil{}}),
    Con{M.sumsq(M.take_row(ps, 83040n, 10n)), Nil{}})


# ---------------------------------------------------------------------------
# the report: the checksums of the 108 parameter tensors in the flat order,
# then the accuracies and the recorded curve

def final_io(tr: TR) -> IO(Unit):
  match tr:
    case TR{+ps, _, _, ls, ns, +hits}:
      do IO<Unit>:
        +th : Nat <- eval_io(8n, 0n, ps, 0n)
        IO.print("MambaViT A3: 160 train images, 5 steps of 32, 256 held-out")
        M.show_tag("mv_train_loss", ls)
        M.show_tag("mv_grad_norm", ns)
        M.show_tag("mv_train_hits", Con{F32.from_nat(hits), Nil{}})
        M.show_tag("mv_train_total", Con{F32.from_nat(train_n()), Nil{}})
        M.show_tag("mv_test_hits", Con{F32.from_nat(th), Nil{}})
        M.show_tag("mv_test_total", Con{F32.from_nat(test_n()), Nil{}})
        M.show_tag("mv_train_acc",
          Con{(F32.from_nat(hits) / F32.from_nat(train_n()) : F32), Nil{}})
        M.show_tag("mv_test_acc",
          Con{(F32.from_nat(th) / F32.from_nat(test_n()) : F32), Nil{}})
        M.show_tag("mv_w_sumsq", wsums(ps))
        return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    pb : TA.Mat<83050n, 1n> <- T.Mat.load(83050n, 1n,
      String.append(ddir(), "/w_init_flat.bin"), 0)
    +ps0 : List<&2, F32> <- IO.pure(List<&2, F32>,
      ml_flat(TA.Mat.to_list(83050n, 1n, pb)))
    tr : TR <- step_io(steps_n(), 0n, ps0, M.zline(params_n()),
      M.zline(params_n()), Nil{}, Nil{}, 0n)
    final_io(tr)
    return Unit{}
