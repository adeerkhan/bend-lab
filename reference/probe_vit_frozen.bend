# The frozen-model probe of the ViT (issue #16, milestone M2): no training,
# just the starting weights, so the NumPy twin can compare the forward and
# the sampler element by element. Each line is one flat list for the twin.
#
#   eps    the model's noise at t = 64 on the starting cloud
#   step   one ancestral step from t = 64 (the clamp and the posterior mean)
#   ddim   the 16-step DDIM cloud, eta = 0
#
# `init_params` is deterministic (noise_of(1) scaled by 0.1, in the loader's
# order), so each output can consume its own copy of the weights.
import Base
import bendtorch@1.3.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ../examples/diffusion/vit/main.bend as M

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

def show(tag: String, xs: List<&2, F32>) -> String:
  String.append(tag, String.append(" ", show_go(xs)))

def start() -> List<&2, F32>:
  M.patch_all(64n, G.noise_of(2000000n, 4096n), Nil{})

def show_ve(ve: M.VE) -> IO(Unit):
  match ve:
    case M.VE{p, eps}:
      do IO<Unit>:
        IO.print(show("eps:", eps))
        return Unit{}

def show_vs(st: M.VS, tag: String) -> IO(Unit):
  match st:
    case M.VS{p, x}:
      do IO<Unit>:
        IO.print(show(tag, x))
        return Unit{}

def show_pl(pl: T.PL<M.ModelP()>) -> IO(Unit):
  match pl:
    case T.PL{p, xs}:
      do IO<Unit>:
        IO.print(show("weights:", xs))
        return Unit{}

def dump(p: M.ModelP()) -> IO(Unit):
  show_pl(T.Layer.dump(M.ModelP(), TA.Mat<M.RW(), M.FE()>(), TA.Mat<M.RW(), M.PX()>(),
    M.ModelS(), M.Model(), p))

def head20(xs: List<&2, F32>, left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs left:
    case Con{h, t} 1n+rest:
      head20(t, rest, List.append(&2, F32, acc, Con{h, Nil{}}))
    case _ _:
      acc

def row20() -> List<&2, F32>:
  head20(M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}), 20n, Nil{})

def flat_w(m: TA.ML<M.RW(), M.W()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def show_fwd64(f: T.Fwd<M.EmbP, M.A(M.W()), TA.Mat<M.RW(), M.FE()>>) -> IO(Unit):
  match f:
    case T.Fwd{p, y, s}:
      +ys = flat_w(TA.Mat.to_list(M.RW(), M.W(), y))
      do IO<Unit>:
        IO.print(show("emb0:", head20(ys, 64n, Nil{})))
        return Unit{}

def emb_out(p: M.ModelP()) -> IO(Unit):
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      inp = TA.Mat.from_list(M.RW(), M.FE(),
        M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}))
      show_fwd64(T.Layer.fwd(M.EmbP, TA.Mat<M.RW(), M.FE()>(), M.A(M.W()),
        TA.Mat<M.RW(), M.FE()>(), M.Embed(), emb, inp))

def emb_f(emb: M.EmbP, inp: TA.Mat<M.RW(), M.FE()>) -> T.Fwd<M.EmbP, M.A(M.W()), TA.Mat<M.RW(), M.FE()>>:
  T.Layer.fwd(M.EmbP, TA.Mat<M.RW(), M.FE()>(), M.A(M.W()), TA.Mat<M.RW(), M.FE()>(),
    M.Embed(), emb, inp)

def then_body(body: M.B3P, f: T.Fwd<M.EmbP, M.A(M.W()), TA.Mat<M.RW(), M.FE()>>) -> T.Fwd<M.B3P, M.A(M.W()), M.B3S>:
  match f:
    case T.Fwd{e2, y, s2}:
      T.Layer.fwd(M.B3P, M.A(M.W()), M.A(M.W()), M.B3S(), M.Body3(), body, y)

def show_body(f: T.Fwd<M.B3P, M.A(M.W()), M.B3S>) -> IO(Unit):
  match f:
    case T.Fwd{p, y, s}:
      +ys = flat_w(TA.Mat.to_list(M.RW(), M.W(), y))
      do IO<Unit>:
        IO.print(show("body0:", head20(ys, 64n, Nil{})))
        return Unit{}

def body_out(p: M.ModelP()) -> IO(Unit):
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      inp = TA.Mat.from_list(M.RW(), M.FE(),
        M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}))
      show_body(then_body(body, emb_f(emb, inp)))

def emb_y(f: T.Fwd<M.EmbP, M.A(M.W()), TA.Mat<M.RW(), M.FE()>>) -> TA.Mat<M.RW(), M.W()>:
  match f:
    case T.Fwd{p, y, s}:
      y

def ln_f(ln: T.LN<M.W()>, y: TA.Mat<M.RW(), M.W()>) -> T.Fwd<T.LN<M.W()>, M.A(M.W()), M.A(M.W())>:
  T.Layer.fwd(T.LN<M.W()>, M.A(M.W()), M.A(M.W()), M.A(M.W()),
    T.LayerNorm(M.RW(), M.W(), 0.00001), ln, y)

def show_ln(f: T.Fwd<T.LN<M.W()>, M.A(M.W()), M.A(M.W())>) -> IO(Unit):
  match f:
    case T.Fwd{p, y, s}:
      +ys = flat_w(TA.Mat.to_list(M.RW(), M.W(), y))
      do IO<Unit>:
        IO.print(show("ln0:", head20(ys, 64n, Nil{})))
        return Unit{}

def ln_out(p: M.ModelP()) -> IO(Unit):
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      match body:
        case T.Two{T.Two{b1, b2}, b3}:
          match b1:
            case T.Two{att, mlp}:
              match att:
                case T.Two{r2, u3}:
                  match r2:
                    case T.Two{u, r}:
                      match r:
                        case T.Two{ln, ap}:
                          inp = TA.Mat.from_list(M.RW(), M.FE(),
                            M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}))
                          show_ln(ln_f(ln, emb_y(emb_f(emb, inp))))

def ln_y(f: T.Fwd<T.LN<M.W()>, M.A(M.W()), M.A(M.W())>) -> TA.Mat<M.RW(), M.W()>:
  match f:
    case T.Fwd{p, y, s}:
      y

def att_p(att: M.AttBlkP) -> M.AttP:
  match att:
    case T.Two{r2, u3}:
      match r2:
        case T.Two{u, r}:
          match r:
            case T.Two{ln, ap}:
              ap

def att_f(ap: M.AttP, x: TA.Mat<M.RW(), M.W()>) -> T.Fwd<M.AttP, M.A(M.W()), M.AttS>:
  T.Layer.fwd(M.AttP, M.A(M.W()), M.A(M.W()), M.AttS(), M.Att(), ap, x)

def show_att(f: T.Fwd<M.AttP, M.A(M.W()), M.AttS>) -> IO(Unit):
  match f:
    case T.Fwd{p, y, s}:
      +ys = flat_w(TA.Mat.to_list(M.RW(), M.W(), y))
      do IO<Unit>:
        IO.print(show("att0:", head20(ys, 64n, Nil{})))
        return Unit{}

def blk_f(b12: M.B12P, x: TA.Mat<M.RW(), M.W()>) -> T.Fwd<M.B12P, M.A(M.W()), M.B12S>:
  T.Layer.fwd(M.B12P, M.A(M.W()), M.A(M.W()), M.B12S(), M.B12(), b12, x)

def blk_y(f: T.Fwd<M.B12P, M.A(M.W()), M.B12S>) -> TA.Mat<M.RW(), M.W()>:
  match f:
    case T.Fwd{p, y, s}:
      y

def show_blk(f: T.Fwd<M.B12P, M.A(M.W()), M.B12S>) -> IO(Unit):
  match f:
    case T.Fwd{p, y, s}:
      +ys = flat_w(TA.Mat.to_list(M.RW(), M.W(), y))
      do IO<Unit>:
        IO.print(show("blk0:", head20(ys, 64n, Nil{})))
        return Unit{}

def att_out(p: M.ModelP()) -> IO(Unit):
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      match body:
        case T.Two{T.Two{b1, b2}, b3}:
          match b1:
            case T.Two{att, mlp}:
              match att:
                case T.Two{r2, u3}:
                  match r2:
                    case T.Two{u, r}:
                      match r:
                        case T.Two{ln, ap}:
                          inp = TA.Mat.from_list(M.RW(), M.FE(),
                            M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}))
                          e = emb_y(emb_f(emb, inp))
                          l = ln_f(ln, e)
                          show_att(att_f(ap, ln_y(l)))

def blk_out(p: M.ModelP()) -> IO(Unit):
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      match body:
        case T.Two{T.Two{b1, b2}, b3}:
          inp = TA.Mat.from_list(M.RW(), M.FE(),
            M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}))
          show_blk(blk_f(b1, emb_y(emb_f(emb, inp))))

def attblk_f(att: M.AttBlkP, x: TA.Mat<M.RW(), M.W()>) -> T.Fwd<M.AttBlkP, M.A(M.W()), M.AttBlkS>:
  T.Layer.fwd(M.AttBlkP, M.A(M.W()), M.A(M.W()), M.AttBlkS(), M.AttBlk(), att, x)

def show_mat(f: T.Fwd<M.AttBlkP, M.A(M.W()), M.AttBlkS>, tag: String) -> IO(Unit):
  match f:
    case T.Fwd{p, y, s}:
      +ys = flat_w(TA.Mat.to_list(M.RW(), M.W(), y))
      do IO<Unit>:
        IO.print(show(tag, head20(ys, 64n, Nil{})))
        return Unit{}

def attblk_y(f: T.Fwd<M.AttBlkP, M.A(M.W()), M.AttBlkS>) -> TA.Mat<M.RW(), M.W()>:
  match f:
    case T.Fwd{p, y, s}:
      y

def mlp_f(mlp: M.MlpP, x: TA.Mat<M.RW(), M.W()>) -> T.Fwd<M.MlpP, M.A(M.W()), M.MlpS>:
  T.Layer.fwd(M.MlpP, M.A(M.W()), M.A(M.W()), M.MlpS(), M.Mlp(), mlp, x)

def show_mlp(f: T.Fwd<M.MlpP, M.A(M.W()), M.MlpS>) -> IO(Unit):
  match f:
    case T.Fwd{p, y, s}:
      +ys = flat_w(TA.Mat.to_list(M.RW(), M.W(), y))
      do IO<Unit>:
        IO.print(show("mlp0:", head20(ys, 64n, Nil{})))
        return Unit{}

def res_out(p: M.ModelP()) -> IO(Unit):
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      match body:
        case T.Two{T.Two{b1, b2}, b3}:
          match b1:
            case T.Two{att, mlp}:
              inp = TA.Mat.from_list(M.RW(), M.FE(),
                M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}))
              show_mat(attblk_f(att, emb_y(emb_f(emb, inp))), "res0:")

def mlp_out(p: M.ModelP()) -> IO(Unit):
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      match body:
        case T.Two{T.Two{b1, b2}, b3}:
          match b1:
            case T.Two{att, mlp}:
              inp = TA.Mat.from_list(M.RW(), M.FE(),
                M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}))
              e = emb_y(emb_f(emb, inp))
              show_mlp(mlp_f(mlp, attblk_y(attblk_f(att, e))))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- dump(M.init_params())
    Unit <- emb_out(M.init_params())
    Unit <- body_out(M.init_params())
    Unit <- ln_out(M.init_params())
    Unit <- att_out(M.init_params())
    Unit <- blk_out(M.init_params())
    Unit <- res_out(M.init_params())
    Unit <- mlp_out(M.init_params())
    do IO<Unit>:
      IO.print(show("row:", row20()))
      return Unit{}
    Unit <- show_ve(M.predict(M.init_params(), 64n, start()))
    Unit <- show_vs(M.rev_step(64n, M.VS{M.init_params(), start()}), "step:")
    Unit <- show_vs(M.ddim_sample(M.init_params()), "ddim:")
    return Unit{}
