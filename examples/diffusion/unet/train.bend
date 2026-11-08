# A trained 64x64 UNet denoiser on Oxford Flowers, written in Bend on BendTorch.
#
#   BEND_MEM=9G scripts/bend-guard.sh bend -o .scratch/unet_train examples/diffusion/unet/train.bend
#   .scratch/unet_train --save /tmp/unet.bin
#   bend examples/diffusion/unet/train.bend --steps 50 --no-sample
#
# The model is `model.bend`: FIVE levels, 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 4 ->
# 8 -> 16 -> 32 -> 64, 64 channels at each level, an `up2` upsample, a residual
# skip at 32x32 and another at 16x16, and a linear bypass from the input beside
# the U, with no normalisations (the reason is in that file).
#
# The objective is rectified flow with the CLEAN IMAGE as the target:
#
#   x_t = noise*(1-t) + data*t,   target x0 = data,   MSE
#   sample: v = (x0hat - x)/(1-t), then x <- x + v*dt, Euler over linspace(0,1,steps+1)
#
# The target was the velocity `v = data - noise` until 2026-11-06, and that is
# what kept every draw noise: E[(d-n)^2] = 1.053 is 95% the noise's variance, so
# the loss could be halved by learning the noise statistics alone, while
# E[x0^2] = 0.203 is the image. With x0 the draws became flowers, blurred, and
# legibility rose 0.274 -> 0.426 against a held-out ceiling of 0.562.
#
# Data: `flowers64-train-gray.u8`, 3,874 flowers at 64x64, one grayscale byte a
# pixel, sha256 1ff86747f9b070c3ed7b2058dd0c1f88317dfea931ed883379513f121385566e,
# the training split of the pinned Oxford parquet (`reference/flowers_export.py`
# makes it). `flowers64-hold-gray.u8`, 102 images, sha256 f08730fe..., is the
# held-out half and is never trained on.
#
#   --steps N       steps to run (default 3600, one pass over the 7,066)
#   --start N       the global index of the first step (default 0)
#   --save PATH     write the training state to PATH with Train.save
#   --resume PATH   restore it with Train.restore
#   --no-sample     skip the Euler draws
#   --probe         print the flat weights and one whole forward output
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ./config.bend as C
import ./model.bend as M

def B() -> Nat:
  C.B()

def CH() -> Nat:
  C.CH()

def HW() -> Nat:
  C.HW()

def HW2() -> Nat:
  C.HW2()

def HW4() -> Nat:
  C.HW4()

def NIMG() -> Nat:
  C.NIMG()

def STEPS() -> Nat:
  C.STEPS()

def EULER() -> Nat:
  C.EULER()

def IN() -> Type:
  M.IN()

def INW() -> Nat:
  M.INW()

def OUT() -> Type:
  M.OUT()

def ModelP() -> Type:
  M.ModelP()

def ModelS() -> Type:
  M.ModelS()

def Model() -> T.Layer<ModelP(), IN(), OUT(), ModelS()>:
  M.Model()

def ml_list(-r: Nat, -c: Nat, ml: TA.ML<r, c>) -> List<&2, F32>:
  match ml:
    case TA.ML{mat, l}:
      l

# ---- the data: 512 flowers, uint8, one grayscale byte a pixel --------------

def cache_root() -> String:
  "/home/adeerkhan/.cache/bend-lab"

def root_of(e: Result<&1, &1, U32 & String, String>) -> IO(String):
  match e:
    case Fail{code}:
      IO.pure(String, cache_root())
    case Done{d}:
      IO.pure(String, d)

def export_dir() -> IO(String):
  do IO<String>:
    e : Result<&1, &1, U32 & String, String> <- IO.get_env("MLBEND_DATA")
    root_of(e)

def path(root: String) -> String:
  String.append(root, String.append("/flowers/", C.DATA()))

def NN() -> Nat:
  Nat.mul(B(), HW())

def batch_bytes() -> Nat:
  Nat.mul(B(), HW())

def rd_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  match m:
    case Tuple{f, r}:
      do IO<List<&2, U32>>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
        File.close(f)
        return bs

def read_batch(root: String, +s: Nat) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path(root), "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <-
      File.read_at(f, U32.from_nat(Nat.mul(Nat.mod(Nat.mul(s, 7919n),
        Nat.add(Nat.sub(NIMG(), B()), 1n)), HW())),
        U32.from_nat(batch_bytes()))
    rd_done(m)

# one level per image, U32 Weyl in [0, 1)
def level(+s: Nat, +i: Nat) -> F32:
  +h = U32.add(U32.mul(U32.from_nat(s), 2654435761),
    U32.mul(U32.from_nat(i), 2246822519))
  (F32.from_nat(U32.to_nat(h)) / 4294967296.0 : F32)

# the training pair: x_t beside its level, and the velocity
type RS is Type:
  RS{inp: List<&2, F32>, tgt: List<&2, F32>, bytes: List<&2, U32>, noise: List<&2, F32>}

def tau() -> F32:
  6.2831855

# the level as four features, so the convolutions see a nonlinear t
def feat1(+t: F32) -> F32:
  F32.sin((tau() * t : F32))

def feat2(+t: F32) -> F32:
  F32.cos((tau() * t : F32))

def feat3(+t: F32) -> F32:
  F32.sin(((2.0 * tau() : F32) * t : F32))

def feat4(+t: F32) -> F32:
  F32.cos(((2.0 * tau() : F32) * t : F32))

def pix(+p: Nat, +t: F32, bytes: List<&2, U32>, noise: List<&2, F32>,
  +ai: List<&2, F32>, +at: List<&2, F32>) -> RS:
  match p:
    case 0n:
      RS{ai, at, bytes, noise}
    case 1n+rest:
      match bytes noise:
        case Con{b, bt} Con{+n, nt}:
          +d = (F32.from_nat(U32.to_nat(b)) / 255.0 : F32)
          +xt = (((1.0 - t : F32) * n : F32) + (t * d : F32) : F32)
          +f1 = feat1(t)
          +f2 = feat2(t)
          +f3 = feat3(t)
          +f4 = feat4(t)
          pix(rest, t, bt, nt,
            Con{f4, Con{f3, Con{f2, Con{f1, Con{xt, ai}}}}},
            Con{d, at})
        case _ _:
          RS{ai, at, bytes, noise}

def imgs(+left: Nat, +i: Nat, +s: Nat, r: RS) -> RS:
  match left r:
    case 0n r:
      r
    case 1n+rest RS{inp, tgt, bytes, noise}:
      imgs(rest, (1n + i : Nat), s, pix(HW(), level(s, i), bytes, noise, inp, tgt))

def rev_rs(r: RS) -> RS:
  match r:
    case RS{inp, tgt, bytes, noise}:
      RS{List.reverse(&2, F32, inp), List.reverse(&2, F32, tgt), bytes, noise}

def pair_of(+s: Nat, bytes: List<&2, U32>) -> RS:
  rev_rs(imgs(B(), 0n, s, RS{Nil{}, Nil{}, bytes, G.noise_of(s, NN())}))

# ---- the step --------------------------------------------------------------

def cfg() -> T.OptCfg:
  T.CAdamW{0.001, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

def mse(y: OUT(), tgt: OUT()) -> T.LossOut<OUT()>:
  T.MSE(NN(), 1n, y, tgt)

# ---- the parameters: noise on the weights, zeros on the biases, ones on      ----
# ---- the norms' scales (the flat order is the model's forward order)         ----

def set_go(+left: Nat, arr: Array<F32>, +i: U32, +v: F32) -> Array<F32>:
  match left:
    case 0n:
      arr
    case 1n+p:
      set_go(p, Array.set(F32, arr, i, v), U32.add(i, 1), v)

def zeros_at(arr: Array<F32>, +start: Nat, +len: Nat, +v: F32) -> Array<F32>:
  set_go(len, arr, U32.from_nat(start), v)

def fill_go(left: Nat, xs: List<&2, F32>, +i: U32, arr: Array<F32>) -> Array<F32>:
  match left xs:
    case 0n _:
      arr
    case 1n+rest Con{h, t}:
      fill_go(rest, t, U32.add(i, U32.from_nat(1n)), Array.set(F32, arr, i, h))
    case _ _:
      arr

def scale_rev(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      scale_rev(t, Con{(0.1 * h : F32), acc})

def scale_go(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  List.reverse(&2, F32, scale_rev(xs, acc))

def count() -> Nat:
  U32.to_nat(T.Layer.count(ModelP(), IN(), OUT(), ModelS(), Model()))

# The flat order is the model's forward order, five levels, count 298,439:
#   conv1 2880; conv3/conv5/conv7/conv9/conv10/conv8/conv6/conv4 36864 each in that order;
#   head w 64, b 298432; bypass w 5, b 298438
def init_arr() -> Array<F32>:
  +n = count()
  arr0 = fill_go(n, scale_go(G.noise_of(1n, n), Nil{}), U32.from_nat(0n),
    Array.new(F32, T.depth(n), 0.0))
  arr1 = zeros_at(arr0, 2880n, 64n, 0.0)
  arr2 = zeros_at(arr1, 39808n, 64n, 0.0)
  arr3 = zeros_at(arr2, 76736n, 64n, 0.0)
  arr4 = zeros_at(arr3, 113664n, 64n, 0.0)
  arr5 = zeros_at(arr4, 150592n, 64n, 0.0)
  arr6 = zeros_at(arr5, 187520n, 64n, 0.0)
  arr7 = zeros_at(arr6, 224448n, 64n, 0.0)
  arr8 = zeros_at(arr7, 261376n, 64n, 0.0)
  arr9 = zeros_at(arr8, 298304n, 64n, 0.0)
  arr10 = zeros_at(arr9, 298432n, 1n, 0.0)
  zeros_at(arr10, 298438n, 1n, 0.0)

def un_pa(pa: T.PA<ModelP()>) -> ModelP():
  match pa:
    case T.PA{p, xs, off}:
      p

def load_p(p0: ModelP(), arr: Array<F32>) -> ModelP():
  un_pa(T.Layer.load(ModelP(), IN(), OUT(), ModelS(), Model(), p0, arr, U32.from_nat(0n)))

def params() -> ModelP():
  load_p(T.Layer.zeros(ModelP(), IN(), OUT(), ModelS(), Model()), init_arr())

def new_train() -> T.Train<ModelP()>:
  T.Train.new(~ModelP(), ~IN(), ~OUT(), ~ModelS(), ~Model(), cfg(), params())

# ---- a checkpoint: what Train.save writes and Train.restore reads ----------

def floats_count() -> Nat:
  Nat.add(Nat.mul(3n, count()), 2n)

def floats_at(+path: String) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    bs : List<&2, U32> <- T.read_all(path)
    return T.f32s(bs)

def restore_arr(xs: List<&2, F32>) -> T.Train<ModelP()>:
  T.Train.restore(~ModelP(), ~IN(), ~OUT(), ~ModelS(), ~Model(), new_train(),
    fill_go(floats_count(), xs, U32.from_nat(0n), Array.new(F32, T.depth(floats_count()), 0.0)))

def resumed(+path: String) -> IO(T.Train<ModelP()>):
  do IO<T.Train<ModelP()>>:
    xs : List<&2, F32> <- floats_at(path)
    return restore_arr(xs)

def save_tr(+path: String, tr: T.Train<ModelP()>) -> IO(T.Train<ModelP()>):
  do IO<T.Train<ModelP()>>:
    kept : T.Train<ModelP()> <- T.Train.save(~ModelP(), ~IN(), ~OUT(), ~ModelS(), ~Model(), path, tr)
    Unit <- IO.print(String.append("saved: ", path))
    return kept

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

def show_pt(xs: List<&2, F32>) -> IO(Unit):
  IO.print(show_go(xs))

def print_or(sp: Bool, +l: F32) -> IO(Unit):
  match sp:
    case True{}:
      do IO<Unit>:
        IO.print(String.append("loss: ", F32.show(l)))
        return Unit{}
    case False{}:
      IO.pure(Unit, Unit{})

# the carried loss printed at step i is the loss of step i-1, so i % 25 == 1
# prints steps 0, 25, 50 ... (the last step's loss is `final_loss`)
def should_print(+i: Nat) -> Bool:
  Nat.is_eq(Nat.mod(i, 25n), 1n)

def loss_of(tr: T.Train<ModelP()>) -> F32:
  match tr:
    case T.Train{p, g, st, c, q1, q2, +l}:
      l

def step_rs(tr: T.Train<ModelP()>, r: RS) -> T.Train<ModelP()>:
  match r:
    case RS{inp, tgt, b, n}:
      T.Train.step(~ModelP(), ~IN(), ~OUT(), ~ModelS(), ~OUT(), ~Model(), ~mse, tr,
        TA.Mat.from_list(NN(), INW(), inp), TA.Mat.from_list(NN(), 1n, tgt))

def do_step(+s: Nat, tr: T.Train<ModelP()>, data: List<&2, U32>) -> T.Train<ModelP()>:
  step_rs(tr, pair_of(s, data))

def train_all(+steps: Nat, +start: Nat, +root: String, tr: T.Train<ModelP()>) -> IO(T.Train<ModelP()>):
  match steps tr:
    case 0n T.Train{p, g, st, c, q1, q2, +l}:
      do IO<T.Train<ModelP()>>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        return T.Train{p, g, st, c, q1, q2, l}
    case 1n+rest T.Train{p, g, st, c, q1, q2, +l}:
      do IO<T.Train<ModelP()>>:
        data : List<&2, U32> <- read_batch(root, start)
        Unit <- print_or(should_print(start), l)
        tr1 : T.Train<ModelP()> <- IO.pure(T.Train<ModelP()>, do_step(start, T.Train{p, g, st, c, q1, q2, l}, data))
        train_all(rest, (1n + start : Nat), root, tr1)

# ---- sampling: Euler from noise --------------------------------------------

def dt() -> F32:
  (1.0 / F32.from_nat(EULER()) : F32)

def interleave(+left: Nat, +t: F32, x: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+rest:
      match x:
        case Con{a, at}:
          +f1 = feat1(t)
          +f2 = feat2(t)
          +f3 = feat3(t)
          +f4 = feat4(t)
          interleave(rest, t, at, Con{f4, Con{f3, Con{f2, Con{f1, Con{a, acc}}}}})
        case Nil{}:
          List.reverse(&2, F32, acc)

def copy2(+left: Nat, xs: List<&2, F32>, +a: List<&2, F32>, +b: List<&2, F32>) -> T.Two<List<&2, F32>, List<&2, F32>>:
  match left:
    case 0n:
      T.Two{List.reverse(&2, F32, a), List.reverse(&2, F32, b)}
    case 1n+rest:
      match xs:
        case Con{+h, t}:
          copy2(rest, t, Con{h, a}, Con{h, b})
        case Nil{}:
          T.Two{List.reverse(&2, F32, a), List.reverse(&2, F32, b)}

def ve_of(f: T.Fwd<ModelP(), OUT(), ModelS()>) -> T.Two<ModelP(), List<&2, F32>>:
  match f:
    case T.Fwd{p, y, s}:
      T.Two{p, ml_list(NN(), 1n, TA.Mat.to_list(NN(), 1n, y))}

def coef(+t: F32) -> F32:
  (dt() / ((1.0 - t) : F32) : F32)

def upd(+left: Nat, +t: F32, x: List<&2, F32>, v: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+rest:
      match x v:
        case Con{+a, at} Con{b, vt}:
          upd(rest, t, at, vt, Con{(a + (coef(t) * (b - a : F32) : F32) : F32), acc})
        case _ _:
          List.reverse(&2, F32, acc)

def step_f(+t: F32, x2: List<&2, F32>, f: T.Fwd<ModelP(), OUT(), ModelS()>) -> T.Two<ModelP(), List<&2, F32>>:
  match f:
    case T.Fwd{p2, y, s}:
      T.Two{p2, upd(NN(), t, x2, ml_list(NN(), 1n, TA.Mat.to_list(NN(), 1n, y)), Nil{})}

def step_two(p: ModelP(), +t: F32, d: T.Two<List<&2, F32>, List<&2, F32>>) -> T.Two<ModelP(), List<&2, F32>>:
  match d:
    case T.Two{x1, x2}:
      step_f(t, x2, T.Layer.fwd(ModelP(), IN(), OUT(), ModelS(), Model(), p,
        TA.Mat.from_list(NN(), INW(), interleave(NN(), t, x1, Nil{}))))

def step_x(p: ModelP(), +k: Nat, x: List<&2, F32>) -> T.Two<ModelP(), List<&2, F32>>:
  +t = (F32.from_nat(k) / F32.from_nat(EULER()) : F32)
  step_two(p, t, copy2(NN(), x, Nil{}, Nil{}))

def euler_two(+left: Nat, +k: Nat, d: T.Two<ModelP(), List<&2, F32>>) -> T.Two<ModelP(), List<&2, F32>>:
  match left:
    case 0n:
      d
    case 1n+rest:
      match d:
        case T.Two{p, x}:
          euler_two(rest, (1n + k : Nat), step_x(p, k, x))

def euler_go(+left: Nat, p: ModelP(), +k: Nat, x: List<&2, F32>) -> T.Two<ModelP(), List<&2, F32>>:
  euler_two(left, k, T.Two{p, x})

def ch0(+left: Nat, xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+rest:
      match xs:
        case Con{a, Con{b, t}}:
          ch0(rest, t, Con{a, acc})
        case _:
          List.reverse(&2, F32, acc)

def drawn(d: T.Two<ModelP(), List<&2, F32>>) -> T.Two<ModelP(), List<&2, F32>>:
  match d:
    case T.Two{p, x}:
      T.Two{p, x}

def draw(p: ModelP(), +round: Nat) -> T.Two<ModelP(), List<&2, F32>>:
  drawn(euler_go(EULER(), p, 0n, G.noise_of(Nat.add(2000000n, round), NN())))

def report_two(d: T.Two<ModelP(), List<&2, F32>>) -> IO(ModelP()):
  match d:
    case T.Two{p, x}:
      do IO<ModelP()>:
        Unit <- IO.print(String.append("sample: ", show_go(x)))
        return p

def sample_go(+left: Nat, +k: Nat, p: ModelP()) -> IO(Unit):
  match left:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+rest:
      do IO<Unit>:
        p2 : ModelP() <- report_two(draw(p, k))
        sample_go(rest, (1n + k : Nat), p2)

# ---- the frozen probe: the flat weights and one forward's first 16 cells ----

def flat_w(pl: T.PL<ModelP()>) -> List<&2, F32>:
  match pl:
    case T.PL{p, xs}:
      xs

def headn(+left: Nat, xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+rest:
      match xs:
        case Con{h, t}:
          headn(rest, t, Con{h, acc})
        case Nil{}:
          List.reverse(&2, F32, acc)

def probe_head(d: T.Two<ModelP(), List<&2, F32>>) -> List<&2, F32>:
  match d:
    case T.Two{p, v}:
      v

def probe_out(p: ModelP()) -> List<&2, F32>:
  x = G.noise_of(7n, NN())
  probe_head(ve_of(T.Layer.fwd(ModelP(), IN(), OUT(), ModelS(), Model(), p,
    TA.Mat.from_list(NN(), INW(), interleave(NN(), 0.5, x, Nil{})))))

def probe2(p: ModelP(), xs: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("params: ", Nat.show(count())))
    Unit <- IO.print(String.append("w: ", show_go(xs)))
    IO.print(String.append("out: ", show_go(probe_out(p))))

def probe(d: T.PL<ModelP()>) -> IO(Unit):
  match d:
    case T.PL{p, xs}:
      probe2(p, xs)

# ---- the flags -------------------------------------------------------------

type Run is Data:
  Run{start: Nat, steps: Nat, save: String, resume: String, sample: Bool, probe: Bool}

def run_default() -> Run:
  Run{0n, STEPS(), "", "", True{}, False{}}

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
        case "--no-sample":
          match r:
            case Run{start, steps, save, resume, sample, probe}:
              run_go(rest, Run{start, steps, save, resume, False{}, probe})
        case "--probe":
          match r:
            case Run{start, steps, save, resume, sample, probe}:
              run_go(rest, Run{start, steps, save, resume, False{}, True{}})
        case "--save":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{start, steps, save, resume, sample, probe}:
                  run_go(more, Run{start, steps, b, resume, sample, probe})
        case "--resume":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{start, steps, save, resume, sample, probe}:
                  run_go(more, Run{start, steps, save, b, sample, probe})
        case "--start":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{start, steps, save, resume, sample, probe}:
                  run_go(more, Run{nat_or(Nat.read(b), start), steps, save, resume, sample, probe})
        case "--steps":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{start, steps, save, resume, sample, probe}:
                  run_go(more, Run{start, nat_or(Nat.read(b), steps), save, resume, sample, probe})
        case _:
          run_go(rest, r)

def run_of(args: List<String>) -> Run:
  match args:
    case Nil{}:
      run_default()
    case Con{program, rest}:
      run_go(rest, run_default())

def start_tr(resume: String) -> IO(T.Train<ModelP()>):
  match resume:
    case SNil{}:
      IO.pure(T.Train<ModelP()>, new_train())
    case SCon{h, t}:
      resumed(SCon{h, t})

def save_if(save: String, tr: T.Train<ModelP()>) -> IO(T.Train<ModelP()>):
  match save:
    case SNil{}:
      IO.pure(T.Train<ModelP()>, tr)
    case SCon{h, t}:
      save_tr(SCon{h, t}, tr)

def finish(p: ModelP(), +probe_on: Bool, +sample: Bool) -> IO(Unit):
  match probe_on:
    case True{}:
      probe(T.Layer.dump(ModelP(), IN(), OUT(), ModelS(), Model(), p))
    case False{}:
      match sample:
        case True{}:
          sample_go(Nat.div(C.DRAWS(), B()), 0n, p)
        case False{}:
          IO.pure(Unit, Unit{})

def train_params(tr: T.Train<ModelP()>) -> ModelP():
  match tr:
    case T.Train{p, g, st, c, q1, q2, l}:
      p

def run_with(root: String, r: Run) -> IO(Unit):
  match r:
    case Run{start, +steps, +save, resume, sample, probe_on}:
      do IO<Unit>:
        Unit <- IO.print(String.append("images: ", Nat.show(B())))
        Unit <- IO.print(String.append("channels: ", Nat.show(CH())))
        Unit <- IO.print(String.append("steps: ", Nat.show(steps)))
        Unit <- IO.print(String.append("euler: ", Nat.show(EULER())))
        Unit <- IO.print(String.append("params: ", Nat.show(count())))
        T.OptCfg.check(cfg())
        tr0 : T.Train<ModelP()> <- start_tr(resume)
        tr1 : T.Train<ModelP()> <- save_if(save, tr0)
        tr2 : T.Train<ModelP()> <- train_all(steps, start, root, tr1)
        tr3 : T.Train<ModelP()> <- save_if(save, tr2)
        p : ModelP() <- IO.pure(ModelP(), train_params(tr3))
        finish(p, probe_on, sample)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    root : String <- export_dir()
    run_with(root, run_of(args))
