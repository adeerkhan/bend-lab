# DiT denoiser on the pinned optdigits dataset.
#
# Same 496-param model as examples/diffusion/dit/main.bend (embed -> verified
# conditioned DiT block -> head), trained on REAL digits from the pinned
# `mstz/optdigits` dataset instead of fixed tensors. Tiny dims: 2 images per
# batch, 3 of each image's 16 patches, width 8, two kv heads. The patch
# subsample is a banner: this is a wiring proof, not a quality result.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import bendtorch@2.1.0.0/main.bend as T
import mlbend-generative@0.1.1.0/main.bend as G
import ./full_block.bend as W

def FE() -> Nat:
  20n

def PX() -> Nat:
  4n

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

def mat(+r: Nat, +c: Nat, +s: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, vals(Nat.mul(r, c), s))

def X() -> Type:
  T.Two<M(W.B(), W.CD()), M(W.N(), FE())>

def Y() -> Type:
  M(W.N(), PX())

# ---- model (identical structure to main.bend) ----

def ModelP() -> Type:
  T.Two<T.Two<T.Two<Unit, M(W.C(), FE())>, W.BP>, M(PX(), W.C())>

def ModelS() -> Type:
  T.Two<T.Two<T.Two<Unit, M(W.N(), FE())>, W.BS>, M(W.N(), W.C())>

def Embed() -> T.Layer<M(W.C(), FE()), M(W.N(), FE()), M(W.N(), W.C()), M(W.N(), FE())>:
  T.LinearNB(W.N(), FE(), W.C(), 0n)

def Pre() -> T.Layer<T.Two<Unit, M(W.C(), FE())>, X(), W.BX, T.Two<Unit, M(W.N(), FE())>>:
  T.Both(~Unit, ~M(W.C(), FE()), ~M(W.B(), W.CD()), ~M(W.N(), FE()), ~M(W.B(), W.CD()), ~M(W.N(), W.C()),
    ~Unit, ~M(W.N(), FE()), ~T.Id(~M(W.B(), W.CD())), ~Embed())

def Body() -> T.Layer<T.Two<T.Two<Unit, M(W.C(), FE())>, W.BP>, X(), M(W.N(), W.C()), T.Two<T.Two<Unit, M(W.N(), FE())>, W.BS>>:
  T.Seq(~T.Two<Unit, M(W.C(), FE())>, ~W.BP, ~X(), ~W.BX, ~M(W.N(), W.C()),
    ~T.Two<Unit, M(W.N(), FE())>, ~W.BS, ~Pre(), ~W.Block())

def Head() -> T.Layer<M(PX(), W.C()), M(W.N(), W.C()), Y, M(W.N(), W.C())>:
  T.LinearNB(W.N(), W.C(), PX(), 0n)

def Model() -> T.Layer<ModelP, X(), Y, ModelS>:
  T.Seq(~T.Two<T.Two<Unit, M(W.C(), FE())>, W.BP>, ~M(PX(), W.C()), ~X(), ~M(W.N(), W.C()), ~Y,
    ~T.Two<T.Two<Unit, M(W.N(), FE())>, W.BS>, ~M(W.N(), W.C()), ~Body(), ~Head())

def scale_rev(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      scale_rev(t, Con{(0.1 * h : F32), acc})

def scale_go(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  List.reverse(&2, F32, scale_rev(xs, acc))

def fill_go(left: Nat, xs: List<&2, F32>, +i: U32, arr: Array<F32>) -> Array<F32>:
  match left xs:
    case 0n _:
      arr
    case 1n+rest Con{h, t}:
      fill_go(rest, t, U32.add(i, U32.from_nat(1n)), Array.set(F32, arr, i, h))
    case _ _:
      arr

def un_pa(pa: T.PA<ModelP()>) -> ModelP():
  match pa:
    case T.PA{p, xs, off}:
      p

def load_p(p0: ModelP(), arr: Array<F32>) -> ModelP():
  un_pa(T.Layer.load(ModelP(), X, Y, ModelS(), Model(), p0, arr, U32.from_nat(0n)))

def params() -> ModelP:
  +n = T.Layer.count(ModelP(), X, Y, ModelS(), Model())
  +nn = U32.to_nat(n)
  arr = fill_go(nn, scale_go(G.noise_of(1n, nn), Nil{}), U32.from_nat(0n), Array.new(F32, T.depth(nn), 0.0))
  load_p(T.Layer.zeros(ModelP(), X, Y, ModelS(), Model()), arr)

# ---- real-data input builders ----

def at(xs: List<&2, F32>, +k: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      Bool.pick(F32, Nat.is_eq(k, 0n), h, at(t, Nat.sub(k, 1n)))

def take_go(left: Nat, xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n _:
      acc
    case 1n+q Con{h, t}:
      take_go(q, t, Con{h, acc})
    case _ _:
      acc

def drop_go(left: Nat, xs: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n _:
      xs
    case 1n+q Con{h, t}:
      drop_go(q, t)
    case _ _:
      Nil{}

def t_of(+s: Nat) -> Nat:
  (1n + Nat.mod(Nat.div(Nat.mul(s, 2654435761n), 67108864n), 64n) : Nat)

def chunk_of(+s: Nat) -> Nat:
  Nat.mod(Nat.div(Nat.mul(s, 2246822519n), 1048576n), 16n)

def abar_of(+s: Nat) -> F32:
  at(G.lin_alphas(64n), Nat.sub(t_of(s), 1n))

def mix2(xs: List<&2, F32>, ys: List<&2, F32>, +a: F32, +b: F32, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Con{h, t} Con{e, u}:
      mix2(t, u, a, b, List.append(&2, F32, acc, Con{((a * h : F32) + (b * e : F32) : F32), Nil{}}))
    case _ _:
      acc

def pairs(+left: Nat, +v: F32, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+q:
      pairs(q, v, Con{v, acc})

# ---- full-width input: 64 images x 16 patches = 1024 rows of FE=20 ----

def pos_go(+k: F32, fs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match fs:
    case Nil{}:
      acc
    case Con{+f, t}:
      pos_go(k, t, List.append(&2, F32,
        List.append(&2, F32, acc, Con{F32.sin((k * f : F32)), Nil{}}),
        Con{F32.cos((k * f : F32)), Nil{}}))

def pos_of(+k: Nat) -> List<&2, F32>:
  pos_go(F32.from_nat(k), Con{0.001, Con{0.01, Con{0.1, Con{1.0, Nil{}}}}}, Nil{})

def token_row(patch4: List<&2, F32>, te: List<&2, F32>, pos: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, List.append(&2, F32, patch4, te), pos)

def take_n(left: Nat, xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n _:
      acc
    case 1n+q Con{h, t}:
      take_n(q, t, Con{h, acc})
    case _ _:
      acc

# 1024 tokens = 64 images x 16 patches; build rows from mixed pixels + te + pos
def rows_mix(mixed: List<&2, F32>, +te: List<&2, F32>, +k: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match mixed:
    case Nil{}:
      acc
    case Con{a, Con{b, Con{c, Con{d, t}}}}:
      rows_mix(t, te, (1n + k : Nat),
        List.append(&2, F32, acc, token_row(Con{a, Con{b, Con{c, Con{d, Nil{}}}}}, te, pos_of(k))))
    case _:
      acc

def step_in(+s: Nat, +x0: List<&2, F32>) -> X:
  +batch = take_n(4096n, x0, Nil{})
  +noise = G.noise_of(s, 4096n)
  +a = F32.sqrt(abar_of(s))
  +b = F32.sqrt((1.0 - abar_of(s) : F32))
  +mixed = mix2(batch, noise, a, b, Nil{})
  +te = G.temb_of(t_of(s), 64n, 4n)
  T.Two{TA.Mat.from_list(W.B(), W.CD(), pairs(W.B(), 0.25, Nil{})),
    TA.Mat.from_list(W.N(), FE(), rows_mix(mixed, te, 0n, Nil{}))}

def step_tgt(+s: Nat) -> Y:
  TA.Mat.from_list(W.N(), PX(), take_n(4096n, G.noise_of(s, 4096n), Nil{}))

# ---- training ----

def mse(y: Y, tgt: Y) -> T.LossOut<Y>:
  T.MSE(W.N(), PX(), y, tgt)

def cfg() -> T.OptCfg:
  T.CAdam{0.003, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

def new_train() -> T.Train<ModelP>:
  T.Train.new(~ModelP(), ~X(), ~Y(), ~ModelS(), ~Model(), cfg(), params)

# ---- checkpoint: Train.save / Train.restore ----

def floats_at(+path: String) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    bs : List<&2, U32> <- T.read_all(path)
    return T.f32s(bs)

def floats_count() -> Nat:
  Nat.add(Nat.mul(3n, U32.to_nat(T.Layer.count(ModelP(), X, Y, ModelS(), Model()))), 2n)

def restore_arr(xs: List<&2, F32>) -> T.Train<ModelP>:
  T.Train.restore(~ModelP(), ~X(), ~Y(), ~ModelS(), ~Model(), new_train(),
    fill_go(floats_count(), xs, U32.from_nat(0n),
      Array.new(F32, T.depth(floats_count()), 0.0)))

def restored(+path: String) -> IO(T.Train<ModelP>):
  do IO<T.Train<ModelP>>:
    xs : List<&2, F32> <- floats_at(path)
    return restore_arr(xs)

def train_params(tr: T.Train<ModelP>) -> ModelP:
  match tr:
    case T.Train{p, g, st, c, p1, p2, l}:
      p

def restore_params(xs: List<&2, F32>) -> ModelP:
  train_params(restore_arr(xs))

def save_tr(+path: String, tr: T.Train<ModelP>) -> IO(T.Train<ModelP>):
  do IO<T.Train<ModelP>>:
    kept : T.Train<ModelP> <- T.Train.save(~ModelP(), ~X(), ~Y(), ~ModelS(), ~Model(), path, tr)
    return kept


def loss_of(tr: T.Train<ModelP>) -> F32:
  match tr:
    case T.Train{p, g, st, c, p1, p2, +l}:
      l

def train_go(left: Nat, +i: Nat, tr: T.Train<ModelP>, +x0: List<&2, F32>) -> IO(T.Train<ModelP>):
  match left tr:
    case 0n T.Train{p, g, st, c, p1, p2, l}:
      IO.pure(T.Train<ModelP>, T.Train{p, g, st, c, p1, p2, l})
    case 1n+rest T.Train{p, g, st, c, p1, p2, +l}:
      x : X = step_in(i, x0)
      tgt : Y = step_tgt(i)
      tr2 = T.Train.step(~ModelP(), ~X(), ~Y(), ~ModelS(), ~Y(), ~Model(), ~mse,
        T.Train{p, g, st, c, p1, p2, l}, x, tgt)
      do IO<T.Train<ModelP>>:
        Unit <- IO.print(String.append("loss: ", F32.show(l)))
        train_go(rest, (1n + i : Nat), tr2, x0)

# ---- local CSV load: no dataset/HTTP stack (the binary must fit in RAM) ----

def maybe_cell(m: Maybe<&2, F32>) -> F32:
  match m:
    case Some{v}:
      v
    case None{}:
      0.0

def cell(+text: String) -> F32:
  maybe_cell(F32.read(text))

def cells_rev(+xs: List<&2, String>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      cells_rev(t, Con{(cell(h) / 16.0 : F32), acc})

def cells_of(+xs: List<&2, String>) -> List<&2, F32>:
  List.reverse(&2, F32, cells_rev(xs, Nil{}))

def first_fields(+xs: List<&2, String>) -> List<&2, String>:
  List.take(&2, String, xs, 64n)

def row_of_csv(+line: String) -> List<&2, F32>:
  cells_of(first_fields(String.split(line, Chr{44})))

def rows_of_csv(+left: Nat, lines: List<&2, String>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left lines:
    case 0n _:
      acc
    case 1n+q Con{h, t}:
      rows_of_csv(q, t, List.append(&2, F32, acc, row_of_csv(h)))
    case _ _:
      acc

def rd_done(m: File & Result<&1, &1, U32 & String, String>) -> IO(List<&2, F32>):
  match m:
    case Tuple{f, r}:
      do IO<List<&2, F32>>:
        text : String <- IO.pass(String, r)
        Unit <- File.close(f)
        return rows_of_csv(1024n, String.split(text, Chr{10}), Nil{})

def load_digits() -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    f : File <- IO.try(File, File.open("reference/.cache/datasets/mstz/optdigits/optdigits.data", "r"))
    m : File & Result<&1, &1, U32 & String, String> <- File.read(f, U32.from_nat(600000n))
    rd_done(m)

def sum8(+xs: List<&2, F32>, +left: Nat, +i: Nat, +acc: F32) -> F32:
  match left:
    case 0n:
      acc
    case 1n+q:
      sum8(xs, q, (1n + i : Nat), (acc + at(xs, i) : F32))

# ---- CLI: --steps N --save PATH --resume PATH ----

type Run is Data:
  Run{steps: Nat, save: String, resume: String}

def run_default() -> Run:
  Run{6n, "", ""}

def nat_or(m: Maybe<&2, Nat>, d: Nat) -> Nat:
  match m:
    case Some{x}:
      x
    case None{}:
      d

def run_go(args: List<String>, r: Run) -> Run:
  match args:
    case Nil{}:
      r
    case Con{a, rest}:
      match a:
        case "--steps":
          match rest:
            case Con{b, more}:
              match r:
                case Run{steps, save, resume}:
                  run_go(more, Run{nat_or(Nat.read(b), steps), save, resume})
            case Nil{}:
              r
        case "--save":
          match rest:
            case Con{b, more}:
              match r:
                case Run{steps, save, resume}:
                  run_go(more, Run{steps, b, resume})
            case Nil{}:
              r
        case "--resume":
          match rest:
            case Con{b, more}:
              match r:
                case Run{steps, save, resume}:
                  run_go(more, Run{steps, save, b})
            case Nil{}:
              r
        case _:
          run_go(rest, r)

def start_tr(resume: String) -> IO(T.Train<ModelP>):
  match resume:
    case SNil{}:
      IO.pure(T.Train<ModelP>, new_train())
    case SCon{h, t}:
      restored(resume)

def save_if(save: String, tr: T.Train<ModelP>) -> IO(T.Train<ModelP>):
  match save:
    case SNil{}:
      IO.pure(T.Train<ModelP>, tr)
    case SCon{h, t}:
      save_tr(save, tr)

def run_main(r: Run) -> IO(Unit):
  match r:
    case Run{steps, save, resume}:
      do IO<Unit>:
        tr0 : T.Train<ModelP> <- start_tr(resume)
        data : List<&2, F32> <- load_digits()
        tr1 : T.Train<ModelP> <- train_go(steps, 0n, tr0, data)
        tr2 : T.Train<ModelP> <- save_if(save, tr1)
        IO.print(String.append("final_loss: ", F32.show(loss_of(tr2))))

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<&1, String> <- IO.args()
    r : Run = run_go(args, run_default())
    run_main(r)
