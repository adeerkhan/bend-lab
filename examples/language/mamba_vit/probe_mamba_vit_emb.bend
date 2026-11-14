# MambaViT A1a: patch embedding + tokens, from the reference dump.
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

def show_emp(e: EMP) -> IO(Unit):
  match e:
    case EMP{emb, toks}:
      do IO<Unit>:
        M.show_tag("mv_emb", emb)
        M.show_tag("mv_tokens", toks)
        return Unit{}

def run(+img: List<&2, F32>, +w0: List<&2, F32>, +b0: List<&2, F32>,
    +w1: List<&2, F32>, +b1: List<&2, F32>,
    +w2: List<&2, F32>, +b2: List<&2, F32>,
    +cls: List<&2, F32>, +pos: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    show_emp(emb_all(img, w0, b0, w1, b1, w2, b2, cls, pos))
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
    run(ml28(TA.Mat.to_list(28n, 28n, im)),
      ml1x16(TA.Mat.to_list(1n, 16n, w0)),
      ml1x16(TA.Mat.to_list(1n, 16n, b0)),
      ml32x16(TA.Mat.to_list(32n, 16n, w1)),
      ml1x32(TA.Mat.to_list(1n, 32n, b1)),
      ml1x32(TA.Mat.to_list(1n, 32n, w2)),
      ml1x32(TA.Mat.to_list(1n, 32n, b2)),
      ml1x32(TA.Mat.to_list(1n, 32n, cls)),
      ml50x32(TA.Mat.to_list(50n, 32n, pos)))
    return Unit{}
