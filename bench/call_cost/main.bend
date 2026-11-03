# What a small matmul costs, and what its call costs (the second half of the
# ViT forward's 63 ms).
#
#   bendenv run -- bend bench/call_cost/main.bend
#
# The ViT forward runs 1024 rows of width 64 and its dense floor is 20.6 ms, so
# 42 ms of the 63 are not arithmetic (`make report TASK=diffusion-vit`). This
# probe times, at the model's own shape, three things 20 times each:
#
#   matmul   TA.Mat.matmul(1024, 64, 64), the kernel alone
#   fwd      the same product through `T.Layer.fwd(LinearNBF)`, bias included
#   ln       one `LayerNorm(1024, 64)`
#
# `fwd - matmul` is the per-call toll of the layer wrapper; `matmul` against
# 8.4 MFLOP / the measured GFLOP/s says whether the kernel reaches its floor.
# Every phase prints its value after the timer, so nothing is erased.
import Base
import bendtorch@1.3.0.0/main.bend as T
import bendtorch-gpu@1.3.0.0/main.bend as F
import ../../examples/diffusion/vit/main.bend as M
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def N() -> Nat:
  20n

def rows() -> Nat:
  1024n

def w() -> Nat:
  64n

def X() -> Type:
  TA.Mat<1024n, 64n>

def P() -> Type:
  TA.Mat<64n, 64n>

def first(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{+h, t}:
      h

def list_of(m: TA.ML<1024n, 64n>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def head_of(mat: TA.Mat<1024n, 64n>) -> F32:
  first(list_of(TA.Mat.to_list(1024n, 64n, mat)))

def mm_head(m: TA.MMul<1024n, 64n, 64n>) -> F32:
  match m:
    case TA.MMul{a, b, c}:
      head_of(c)

# the kernel alone, N products; the operands come back with every product
def mm_r(+left: Nat, m: TA.MMul<1024n, 64n, 64n>) -> F32:
  match left:
    case 0n:
      mm_head(m)
    case 1n+rest:
      match m:
        case TA.MMul{a, b, c}:
          mm_r(rest, TA.Mat.matmul(1024n, 64n, 64n, 0n, a, b))

# the same product as the model runs it: LinearNBF through Layer.fwd.
# A match cannot scrutinise a computed value, so each step hands the Fwd to a
# def that matches it as a parameter.
type FW is Type:
  FW{p: P(), y: X()}

def fw_of(f: T.Fwd<P(), X(), X()>) -> FW:
  match f:
    case T.Fwd{p, y, s}:
      FW{p, y}

def fwd_one(f: FW) -> FW:
  match f:
    case FW{p, x}:
      fw_of(T.Layer.fwd(P(), X(), X(), X(), F.LinearNBF(1024n, 64n, 64n, 0n), p, x))

def fwd_r(+left: Nat, f: FW) -> F32:
  match left f:
    case 0n FW{p, y}:
      head_of(y)
    case 1n+rest FW{p, y}:
      fwd_r(rest, fwd_one(FW{p, y}))

type LW is Type:
  LW{ln: T.LN<64n>, y: X()}

def lw_of(f: T.Fwd<T.LN<64n>, X(), X()>) -> LW:
  match f:
    case T.Fwd{p, y, s}:
      LW{p, y}

def ln_one(w: LW) -> LW:
  match w:
    case LW{ln, x}:
      lw_of(T.Layer.fwd(T.LN<64n>, X(), X(), X(), T.LayerNorm(1024n, 64n, 0.00001), ln, x))

def ln_r(+left: Nat, w: LW) -> F32:
  match left w:
    case 0n LW{ln, y}:
      head_of(y)
    case 1n+rest LW{ln, y}:
      ln_r(rest, ln_one(LW{ln, y}))

# ---- one block, the model's own: attention and MLP separately ----
# The nested params are the probe_vit_frozen pattern: Seq loads left then
# right, so body = Two{Two{b1, b2}, b3} and b1 = Two{att, mlp}.
type AW is Type:
  AW{p: M.AttBlkP(), y: M.A(M.W())}

type MW is Type:
  MW{p: M.MlpP(), y: M.A(M.W())}

def aw_of(f: T.Fwd<M.AttBlkP(), M.A(M.W()), M.AttBlkS()>) -> AW:
  match f:
    case T.Fwd{p, y, s}:
      AW{p, y}

def blk_one(a: AW) -> AW:
  match a:
    case AW{p, x}:
      aw_of(T.Layer.fwd(M.AttBlkP(), M.A(M.W()), M.A(M.W()), M.AttBlkS(),
        M.AttBlk(), p, x))

def blk_r(+left: Nat, a: AW) -> F32:
  match left a:
    case 0n AW{p, y}:
      head_of(y)
    case 1n+rest AW{p, y}:
      blk_r(rest, blk_one(AW{p, y}))

def mw_of(f: T.Fwd<M.MlpP(), M.A(M.W()), M.MlpS()>) -> MW:
  match f:
    case T.Fwd{p, y, s}:
      MW{p, y}

def mlp_one(m: MW) -> MW:
  match m:
    case MW{p, x}:
      mw_of(T.Layer.fwd(M.MlpP(), M.A(M.W()), M.A(M.W()), M.MlpS(), M.Mlp(), p, x))

def mlp_r(+left: Nat, m: MW) -> F32:
  match left m:
    case 0n MW{p, y}:
      head_of(y)
    case 1n+rest MW{p, y}:
      mlp_r(rest, mlp_one(MW{p, y}))

def att_of(p: M.ModelP()) -> M.AttBlkP():
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      match body:
        case T.Two{T.Two{b1, b2}, b3}:
          match b1:
            case T.Two{att, mlp}:
              att

def mlp_of_p(p: M.ModelP()) -> M.MlpP():
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      match body:
        case T.Two{T.Two{b1, b2}, b3}:
          match b1:
            case T.Two{att, mlp}:
              mlp

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("rows: ", Nat.show(rows())))
    Unit <- IO.print(String.append("width: ", Nat.show(w())))
    Unit <- IO.print(String.append("passes: ", Nat.show(N())))

    x0 : X() <- IO.pure(X(), TA.Mat.fill(1024n, 64n, 0.25))
    w0 : P() <- IO.pure(P(), TA.Mat.fill(64n, 64n, 0.5))
    t0 : Nat <- IO.now()
    a : F32 <- IO.pure(F32, mm_r(N(), TA.Mat.matmul(1024n, 64n, 64n, 0n, x0, w0)))
    t1 : Nat <- IO.now()
    Unit <- IO.print(String.append("matmul_ms: ", Nat.show(Nat.sub(t1, t0))))
    Unit <- IO.print(String.append("matmul_head: ", F32.show(a)))

    x1 : X() <- IO.pure(X(), TA.Mat.fill(1024n, 64n, 0.25))
    p1 : P() <- IO.pure(P(), TA.Mat.fill(64n, 64n, 0.5))
    t2 : Nat <- IO.now()
    b : F32 <- IO.pure(F32, fwd_r(N(), FW{p1, x1}))
    t3 : Nat <- IO.now()
    Unit <- IO.print(String.append("fwd_ms: ", Nat.show(Nat.sub(t3, t2))))
    Unit <- IO.print(String.append("fwd_head: ", F32.show(b)))

    x2 : X() <- IO.pure(X(), TA.Mat.fill(1024n, 64n, 0.25))
    ln : T.LN<64n> <- IO.pure(T.LN<64n>, T.LN{TA.Mat.fill(1n, 64n, 1.0), TA.Mat.fill(1n, 64n, 0.0)})
    t4 : Nat <- IO.now()
    c : F32 <- IO.pure(F32, ln_r(N(), LW{ln, x2}))
    t5 : Nat <- IO.now()
    Unit <- IO.print(String.append("ln_ms: ", Nat.show(Nat.sub(t5, t4))))
    Unit <- IO.print(String.append("ln_head: ", F32.show(c)))

    t6 : Nat <- IO.now()
    b : F32 <- IO.pure(F32, blk_r(N(), AW{att_of(M.init_params()), TA.Mat.fill(1024n, 64n, 0.25)}))
    t7 : Nat <- IO.now()
    Unit <- IO.print(String.append("blk_ms: ", Nat.show(Nat.sub(t7, t6))))
    Unit <- IO.print(String.append("blk_head: ", F32.show(b)))

    t8 : Nat <- IO.now()
    d : F32 <- IO.pure(F32, mlp_r(N(), MW{mlp_of_p(M.init_params()), TA.Mat.fill(1024n, 64n, 0.25)}))
    t9 : Nat <- IO.now()
    Unit <- IO.print(String.append("mlp_ms: ", Nat.show(Nat.sub(t9, t8))))
    IO.print(String.append("mlp_head: ", F32.show(d)))
    return Unit{}

