# A DDPM over real data: CIFAR-10 images (32x32x3), trained and sampled in
# Bend with the real BendTorch.
#
#   bendenv run -- bend examples/cifar/main.bend
#
# The dataset is arwics/cifar10 at a pinned revision: 5000 rows of 3072
# pixels (0..255) in channel-height-width order and the label, read
# through mlbend-datasets' generic csv_of; the first 256 rows are the
# training set, the pixels scaled to [0, 1]. The denoiser is the digits
# skeleton at CIFAR shapes: a three-linear-layer MLP with SiLU over the
# noised 3072 pixels beside an 8-number time embedding (inwidth 3080),
# hidden 128, trained full-batch by AdamW to predict the noise of the
# linear schedule of mlbend-generative, then 8 fresh images are sampled by
# the ancestral loop and printed as a grid (reference/test_cifar.py draws
# them beside the data).
#
# 40 steps proves the pipeline, not a trained model: the gated numbers are
# the deterministic ones (the first loss and the data mean), and the final
# loss and the sample statistics are report mode (issue #11).
#
# The same findings as examples/diffusion govern the shape: the parameters
# thread through the sampling loop (Layer.fwd returns them), and every
# "random" number is a counter.
import Base
import bendtorch@1.1.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

def n() -> Nat:
  256n

def d() -> Nat:
  3072n

def inwidth() -> Nat:
  3080n

def hidden() -> Nat:
  128n

def steps() -> Nat:
  40n

def samples() -> Nat:
  8n

def X() -> Type:
  TA.Mat<n(), inwidth()>

def Y() -> Type:
  TA.Mat<n(), d()>

def SY() -> Type:
  TA.Mat<samples(), d()>

def SX() -> Type:
  TA.Mat<samples(), inwidth()>

def M2(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# ---- the denoiser: linear, SiLU, linear, SiLU, linear ----

def B1P() -> Type:
  T.Two<T.Lin<inwidth(), hidden()>, Unit>

def B1S(r: Nat) -> Type:
  T.Two<M2(r, inwidth()), M2(r, hidden())>

def B2P() -> Type:
  T.Two<B1P, T.Lin<hidden(), hidden()>>

def B2S(r: Nat) -> Type:
  T.Two<B1S(r), M2(r, hidden())>

def B3P() -> Type:
  T.Two<B2P, Unit>

def B3S(r: Nat) -> Type:
  T.Two<B2S(r), M2(r, hidden())>

def B4P() -> Type:
  T.Two<B3P, T.Lin<hidden(), d()>>

def B4S(r: Nat) -> Type:
  T.Two<B3S(r), M2(r, hidden())>

def B1(~r: Nat) -> T.Layer<B1P, M2(r, inwidth()), M2(r, hidden()), B1S(r)>:
  T.Seq(~T.Lin<inwidth(), hidden()>, ~Unit, ~M2(r, inwidth()), ~M2(r, hidden()),
    ~M2(r, hidden()), ~M2(r, inwidth()), ~M2(r, hidden()),
    ~T.LinearFirst(r, inwidth(), hidden(), 0n), ~T.SiLU(r, hidden()))

def B2(~r: Nat) -> T.Layer<B2P, M2(r, inwidth()), M2(r, hidden()), B2S(r)>:
  T.Seq(~B1P, ~T.Lin<hidden(), hidden()>, ~M2(r, inwidth()), ~M2(r, hidden()),
    ~M2(r, hidden()), ~B1S(r), ~M2(r, hidden()), ~B1(~r), ~T.Linear(r, hidden(), hidden(), 0n))

def B3(~r: Nat) -> T.Layer<B3P, M2(r, inwidth()), M2(r, hidden()), B3S(r)>:
  T.Seq(~B2P, ~Unit, ~M2(r, inwidth()), ~M2(r, hidden()), ~M2(r, hidden()),
    ~B2S(r), ~M2(r, hidden()), ~B2(~r), ~T.SiLU(r, hidden()))

def Model(~r: Nat) -> T.Layer<B4P, M2(r, inwidth()), M2(r, d()), B4S(r)>:
  T.Seq(~B3P, ~T.Lin<hidden(), d()>, ~M2(r, inwidth()), ~M2(r, hidden()), ~M2(r, d()),
    ~B3S(r), ~M2(r, hidden()), ~B3(~r), ~T.Linear(r, hidden(), d(), 0n))

# ---- the data: the first 256 CIFAR-10 images, the pixels scaled to [0, 1] ----

# the loader runs once, so the rows are built the plain way: appended,
# ascending. (The training input is built per step; that one needs the
# cons-and-reverse walk below.)
def scale_go(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      scale_go(t, List.append(&2, F32, acc, Con{(h / 255.0 : F32), Nil{}}))

# one reverse of a whole consed list
def list_rev(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      list_rev(t, Con{h, acc})

def take_rows(left: Nat, xs: List<&2, List<&2, F32>>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n Nil{}:
      acc
    case 0n Con{row, rows}:
      acc
    case 1n+rest Con{row, rows}:
      take_rows(rest, rows, List.append(&2, F32, acc, scale_go(row, Nil{})))
    case 1n+rest Nil{}:
      acc

def cifar_rows(t: Datasets.Table) -> List<&2, F32>:
  match t:
    case Datasets.Table{rows, labels}:
      take_rows(256n, rows, Nil{})

# ---- the schedule and the coefficients ----

def at_go(xs: List<&2, F32>, +k: Nat, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      Bool.pick(F32, Nat.is_eq(i, k), h, at_go(t, k, (1n + i : Nat)))

def at(xs: List<&2, F32>, k: Nat) -> F32:
  at_go(xs, k, 0n)

def abar(+t: Nat) -> F32:
  at(G.lin_alphas(64n), Nat.sub(t, 1n))

# ᾱ_{t-1}, the rung below: ᾱ_0 = 1 at the bottom (the schedule has no
# entry before the first), the entry before otherwise
def abar_below(+t: Nat) -> F32:
  Bool.pick(F32, Nat.is_eq(t, 1n), 1.0, at(G.lin_alphas(64n), Nat.sub(t, 2n)))

def coef_a(+t: Nat) -> F32:
  F32.sqrt(abar(t))

def coef_b(+t: Nat) -> F32:
  F32.sqrt((1.0 - abar(t) : F32))

# ---- the model input: the noised pixels beside the embedding ----
#
# One row is one image: its `width` pixels, noised, then the embedding's
# eight numbers. The row is built by consing, then reversed once: appending
# to a 3072-cell row instead is quadratic in the row width, and at CIFAR
# scale that is the step's whole cost (bend-ml#23). The rows are consed as
# finished blocks; one flatten-cons walk and one reverse turn the blocks
# into the flat row-major list. The noise index j rides across rows: pixel
# p of row r pairs with counter index r*width + p, the same index the
# target noise list holds.

def row_temb_go(+te: List<&2, F32>, +k: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match k:
    case 0n:
      acc
    case 1n+rest:
      row_temb_go(te, rest, Con{at(te, Nat.sub(8n, k)), acc})

# the pixels of one row, noised and consed (the row comes out reversed,
# `row_of` turns it); j is the row's base noise index, riding +1 a pixel
def row_pixels_go(+s: Nat, +a: F32, +b: F32, +te: List<&2, F32>,
  x0: List<&2, F32>, +left: Nat, +j: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match x0 left:
    case Con{h, tail} 1n+rest:
      row_pixels_go(s, a, b, te, tail, rest, (1n + j : Nat),
        Con{(a * h + (b * G.gauss_at(s, j) : F32) : F32), acc})
    case Con{h, tail} 0n:
      row_temb_go(te, 8n, acc)
    case Nil{} 0n:
      row_temb_go(te, 8n, acc)
    case Nil{} 1n+rest:
      row_temb_go(te, 8n, acc)

# one row consumed: the finished block and the rest of the list

type RR is Data:
  RR{frag: List<&2, F32>, rest: List<&2, F32>}

# `width` pixels consumed: the tail that is left of the row.
def drop_go(+xs: List<&2, F32>, +n: Nat) -> List<&2, F32>:
  match xs n:
    case _ 0n:
      xs
    case Con{h, t} 1n+rest:
      drop_go(t, rest)
    case Nil{} _:
      Nil{}

# one row consumed: the finished block and the rest of the list
def row_of(+s: Nat, +a: F32, +b: F32, +te: List<&2, F32>, x0: List<&2, F32>,
  +j: Nat, +width: Nat) -> RR:
  match x0:
    case Nil{}:
      RR{Nil{}, Nil{}}
    case Con{+h, +tail}:
      RR{List.reverse(&2, F32, row_pixels_go(s, a, b, te, Con{h, tail}, width, j, Nil{})),
        drop_go(Con{h, tail}, width)}

def rr_frag(rr: RR) -> List<&2, F32>:
  match rr:
    case RR{frag, rest}:
      frag

def rr_rest(rr: RR) -> List<&2, F32>:
  match rr:
    case RR{frag, rest}:
      rest

# the rows, consed as blocks: the first row ends up deepest
def rows_go(+s: Nat, +a: F32, +b: F32, +t: Nat, +te: List<&2, F32>,
  +rows_left: Nat, +j: Nat, +width: Nat, x0: List<&2, F32>, +acc: List<&2, List<&2, F32>>) ->
  List<&2, List<&2, F32>>:
  match rows_left:
    case 0n:
      acc
    case 1n+rest:
      +rr = row_of(s, a, b, te, x0, j, width)
      rows_go(s, a, b, t, te, rest, (width + j : Nat), width, rr_rest(rr), Con{rr_frag(rr), acc})

# flatten the blocks by consing (each block lands reversed), then one
# reverse of the whole: the flat row-major list
def cons_all(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      cons_all(t, Con{h, acc})

# the rows list back in ascending order: row 0 first
def rows_rev(xs: List<&2, List<&2, F32>>, +acc: List<&2, List<&2, F32>>) ->
  List<&2, List<&2, F32>>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      rows_rev(t, Con{h, acc})

def flat_go(rows: List<&2, List<&2, F32>>, +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{row, rest}:
      flat_go(rest, cons_all(row, acc))

# the training batch: n() images of d() pixels, each beside its embedding
def input_of(+s: Nat, +t: Nat, +a: F32, +b: F32, +x0: List<&2, F32>) -> List<&2, F32>:
  list_rev(flat_go(rows_rev(rows_go(s, a, b, t, G.temb_of(t, 64n, 4n), n(), 0n, d(), x0, Nil{}),
    Nil{}), Nil{}), Nil{})

# the sampler's input: samples() images, the sample not noised again
def sinput_of(+t: Nat, x: List<&2, F32>) -> List<&2, F32>:
  list_rev(flat_go(rows_rev(rows_go((1000000n + t : Nat), 1.0, 0.0, t, G.temb_of(t, 64n, 4n),
    samples(), 0n, d(), x, Nil{}), Nil{}), Nil{}), Nil{})

# ---- the training loop ----

def cfg() -> T.OptCfg:
  T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

def mse(y: Y(), tgt: Y()) -> T.LossOut<Y()>:
  T.MSE(n(), d(), y, tgt)

# The initial weights: deterministic, zero-mean, and scaled by
# 1/sqrt(fan_in) at the row builder — PyTorch's Linear init, which is what
# the vendored reference uses. The previous (k*0.1 - 0.2) had mean +0.15,
# so summed over 72 inputs every unit started saturated: the first loss
# was 8062.7 against the reference's 0.96 and the model never left the
# "predict nothing" floor at 1.0 (ISSUE-001; the ablation that pinned the
# cause is reference/digits_ablation.py).
def value(+i: Nat, +j: Nat, +seed: Nat) -> F32:
  F32.div(F32.sub(F32.from_nat(Nat.mod(Nat.add(Nat.add(Nat.mul(i, 3n), Nat.mul(j, 5n)),
    Nat.mul(seed, 7n)), 97n)), 48.0), 48.0)

# a row of weights, consed and reversed once: appending a cell at a time is
# quadratic in the fan-in, and 3080 is where that shows (6.4 s of the run)
def row_vals(+c: Nat, +i: Nat, +j: Nat, +seed: Nat, +s: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match c:
    case 0n:
      acc
    case 1n+rest:
      row_vals(rest, i, (1n + j : Nat), seed, s,
        Con{(s * value(i, j, seed) : F32), acc})

def rows_vals(+r: Nat, +c: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match r:
    case 0n:
      acc
    case 1n+rest:
      s = F32.div(1.0, F32.sqrt(F32.from_nat(c)))
      rows_vals(rest, c, (1n + i : Nat), seed,
        List.append(&2, F32, acc, List.reverse(&2, F32, row_vals(c, i, 0n, seed, s, Nil{}))))

def mat_of(+r: Nat, +c: Nat, +seed: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, rows_vals(r, c, 0n, seed, Nil{}))

def params() -> B4P():
  T.Two{
    T.Two{
      T.Two{T.Two{T.Lin{mat_of(hidden(), inwidth(), 1n), TA.Mat.fill(1n, hidden(), 0.0)}, Unit{}},
      T.Lin{mat_of(hidden(), hidden(), 2n), TA.Mat.fill(1n, hidden(), 0.0)}},
    Unit{}},
  T.Lin{mat_of(d(), hidden(), 3n), TA.Mat.fill(1n, d(), 0.0)}}

def noise_of(+s: Nat) -> List<&2, F32>:
  G.noise_of(s, 786432n)

def do_step(+s: Nat, tr: T.Train<B4P()>, +x0: List<&2, F32>) -> T.Train<B4P()>:
  +t = Nat.add(Nat.mod(s, 64n), 1n)
  T.Train.step(~B4P(), ~X(), ~Y(), ~B4S(n()), ~Y(), ~Model(n()), ~mse, cfg(), tr,
    TA.Mat.from_list(n(), inwidth(), input_of(s, t, coef_a(t), coef_b(t), x0)),
    TA.Mat.from_list(n(), d(), noise_of(s)))

def should_print(+i: Nat) -> Bool:
  Bool.or(Bool.or(Nat.is_eq(i, 1n), Nat.is_eq(Nat.mod((1n + i : Nat), 100n), 0n)),
    Nat.is_eq((1n + i : Nat), steps()))

def print_or(sp: Bool, +l: F32) -> IO(Unit):
  match sp:
    case True{}:
      do IO<Unit>:
        IO.print(String.append("loss: ", F32.show(l)))
        return Unit{}
    case False{}:
      do IO<Unit>:
        return Unit{}

def loop(rows_left: Nat, +i: Nat, +x0: List<&2, F32>, tr: T.Train<B4P()>) -> IO(T.Train<B4P()>):
  match rows_left tr:
    case 0n T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        return T.Train{p, g, st, q1, q2, l}
    case 1n+rest T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        Unit <- print_or(should_print(i), l)
        loop(rest, (1n + i : Nat), x0, do_step(i, T.Train{p, g, st, q1, q2, l}, x0))

# ---- the sample: 64 fresh digits by the ancestral loop ----
#
# The parameters thread through a PS record (Layer.fwd returns them). The
# reverse update consumes the sample, the prediction and the noise in
# parallel, consing the new sample in reverse; the update is element-wise,
# so order only tracks position, and 64 steps (an even count) leave the
# pixels where they started.

type PS is Type:
  PS{p: B4P(), x: List<&2, F32>}

type PE is Type:
  PE{p: B4P(), eps: List<&2, F32>}

def flat_of(m: TA.ML<samples(), d()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def eps_list(m: SY()) -> List<&2, F32>:
  flat_of(TA.Mat.to_list(samples(), d(), m))

def fwd_eps_of(r: T.Fwd<B4P(), SY(), B4S(samples())>) -> PE:
  match r:
    case T.Fwd{p2, y, s}:
      PE{p2, eps_list(y)}

def predict(p: B4P(), +t: Nat, +x: List<&2, F32>) -> PE:
  fwd_eps_of(T.Layer.fwd(B4P(), SX(), SY(), B4S(samples()), Model(samples()), p,
    TA.Mat.from_list(samples(), inwidth(), sinput_of(t, x))))

# One element of the new sample. The mean is scaled by 1/sqrt(1-beta_t)
# and the draw is added after it, outside the scale (issue #14): Ho et
# al.'s Algorithm 2 adds sigma_t·z to the already-scaled mean, and this
# is exactly what makes DDIM with eta = 1 reproduce the step.
# The five coefficients of one reverse step, computed once per step:
# the two fractions that form the posterior mean, the two roots that
# recover the start from (x, eps), and the posterior's sigma.
type RC is Data:
  RC{c1: F32, c2: F32, sa: F32, sm: F32, sig: F32}

# One element of the new sample. The model's implied start is
# reconstructed first —
#
#   x0 = (x - sqrt(1 - abar) eps) / sqrt(abar)
#
# — clamped to the data range [0, 1], and only then is the posterior mean
# formed from it. Without the clamp a trained digits model's cloud walks
# to std 35.8 against the data's 0.377; with it, 0.30 (ISSUE-005,
# isolated by reference/digits_sample_probe.py). The vendored reference
# clamps the same way (`p_mean_variance` / `clip_x_start`).
def rev_elem(co: RC, +xv: F32, +eps: F32, +z: F32) -> F32:
  match co:
    case RC{c1, c2, sa, sm, sig}:
      x0 = ((xv - (sm * eps : F32) : F32) / sa : F32)
      xc = F32.clamp(x0, 0.0, 1.0)
      (((c1 * xc : F32) + (c2 * xv : F32) : F32) + (sig * z : F32) : F32)

# One step over the sample. The walk conses (issue #13: append would be
# O(n²) over 4096 elements × 64 steps), so the accumulator comes back
# reversed once at the end — one List.reverse per step, and the array
# is in order again.
def rev_go(+co: RC, x: List<&2, F32>, eps: List<&2, F32>,
  z: List<&2, F32>, left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match x eps z left:
    case Con{xv, xt} Con{ev, et} Con{zv, zt} 1n+rest:
      rev_go(co, xt, et, zt, rest, Con{rev_elem(co, xv, ev, zv), acc})
    case _ _ _ 0n:
      acc
    case _ _ _ _:
      acc

def rev_pe(pe: PE, +co: RC, +z: List<&2, F32>, +x: List<&2, F32>) -> PS:
  match pe:
    case PE{p2, eps}:
      PS{p2, List.reverse(&2, F32, rev_go(co, x, eps, z, 24576n, Nil{}))}

# One reverse step of the ancestral loop. The mean is the posterior's,
# formed from the clamped start; the noise is the exact posterior's
# sigma = sqrt(beta_tilde_t) (issue #14) — the one DDIM with eta = 1
# reproduces. At t = 1 that variance is 0 (1-ᾱ_0 = 0), so the last step
# adds nothing.
def reverse_step(+t: Nat, st: PS) -> PS:
  match st:
    case PS{p, +x}:
      +beta = G.lin_beta_at(t, 64n)
      +abar_t = abar(t)
      +bar_prev = abar_below(t)
      +c1 = (beta * (F32.sqrt(bar_prev) / (1.0 - abar_t) : F32) : F32)
      +c2 = (F32.sqrt((1.0 - beta : F32)) * ((1.0 - bar_prev : F32) / (1.0 - abar_t : F32)) : F32)
      +sa = F32.sqrt(abar_t)
      +sm = F32.sqrt((1.0 - abar_t : F32))
      +sig = F32.sqrt((beta * ((1.0 - bar_prev : F32) / (1.0 - abar_t : F32)) : F32))
      rev_pe(predict(p, t, x), RC{c1, c2, sa, sm, sig}, G.noise_of((1000000n + t : Nat), 24576n), x)

def sample_go(t: Nat, st: PS) -> PS:
  match t:
    case 0n:
      st
    case 1n++rest:
      sample_go(rest, reverse_step(t, st))

def sample(p: B4P()) -> PS:
  sample_go(64n, PS{p, G.noise_of(2000000n, 24576n)})

def sample_x(st: PS) -> List<&2, F32>:
  match st:
    case PS{p, x}:
      x

# ---- the DDIM sampler (Song et al., 2020): the same model, every k-th rung ----
#
# The same walk as `examples/diffusion`: of the 64 rungs every k-th one
# (k = 4 below), so the model runs 16 times instead of 64. The two rung
# lists — the ᾱ of each rung and its index — come from the package's
# strided walk and are walked in lockstep (`strided_length`). eta (η)
# picks the stochasticity: at η = 0 a step reads no noise at all (the z
# list is empty and σ·0.0 = 0), so the sample is a pure function of the
# model and the starting noise; at η = 1 over every rung a step IS the
# ancestral one above, because that σ² is exactly the posterior's β̃_t
# (issue #14, gated in reference/test_ddim.py).

# the rung count the report uses: 16 of 64
def stride() -> Nat:
  4n

# the noise level the report uses: deterministic
def eta() -> F32:
  0.0

# The rungs' ᾱ, top first. skip = (T-1) mod k lands the last emission
# on the last entry, so the top rung is ᾱ_T for every k.
def rung_abars(+k: Nat) -> List<&2, F32>:
  List.reverse(&2, F32, G.strided_from(F32, G.lin_alphas(64n), k, Nat.mod(63n, k)))

# the rungs' index: the same walk over List.range(64), so the two lists
# have the same length (strided_length) and run in lockstep
def rung_ts(+k: Nat) -> List<&2, Nat>:
  List.reverse(&2, Nat, G.strided_from(Nat, List.range(64n), k, Nat.mod(63n, k)))

# the ᾱ below a rung: the next entry down, or ᾱ_0 = 1 when the rungs
# are spent — the step that reaches x_0 falls back to ᾱ_0
def head_or_one(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      1.0
    case Con{h, t}:
      h

# the z of one step: a draw from a counter past the training ones and
# the ancestral loop's own (salt 0 lands on it exactly), or nothing at
# all when the step reads no noise (η = 0)
def ddim_z(+t: Nat, +salt: Nat, draw: Bool) -> List<&2, F32>:
  match draw:
    case True{}:
      G.noise_of((1000000n + (salt + t : Nat) : Nat), 24576n)
    case False{}:
      Nil{}

# The noise the clamped start implies: the reference's
# `rederive_pred_noise`. A start outside [0, 1] is pulled back, and the
# step is taken from the noise that start implies rather than the raw
# prediction — the DDIM counterpart of `rev_elem`'s clamp (ISSUE-005).
def clip_eps_go(+sa: F32, +sm: F32, x: List<&2, F32>, eps: List<&2, F32>,
  left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match x eps left:
    case Con{+xv, xt} Con{ev, et} 1n+rest:
      x0 = ((xv - (sm * ev : F32) : F32) / sa : F32)
      xc = F32.clamp(x0, 0.0, 1.0)
      clip_eps_go(sa, sm, xt, et, rest,
        Con{((xv - (sa * xc : F32) : F32) / sm : F32), acc})
    case _ _ 0n:
      acc
    case _ _ _:
      acc

def clip_eps(+abar_now: F32, x: List<&2, F32>, eps: List<&2, F32>) -> List<&2, F32>:
  +sa = F32.sqrt(abar_now)
  +sm = F32.sqrt((1.0 - abar_now : F32))
  List.reverse(&2, F32, clip_eps_go(sa, sm, x, eps, 24576n, Nil{}))

# one rung: the model's noise there, folded with the rung below into
# the sample (the package's ddim does the folding and keeps the shape)
def ddim_pe(pe: PE, +abar_now: F32, +abar_prev: F32, +eta_now: F32, +z: List<&2, F32>,
  +x: List<&2, F32>) -> PS:
  match pe:
    case PE{p2, eps}:
      PS{p2, G.ddim(G.step_of(abar_now, abar_prev, eta_now), x,
        clip_eps(abar_now, x, eps), z)}

# the walk down the rungs: each rung's ᾱ beside its index, the rung
# below as the head of what is left (or ᾱ_0 when nothing is left). The
# two lists come first because the checker reads a self-call left to
# right and wants the shrinking argument before the state that changes.
def ddim_go(abars: List<&2, F32>, ts: List<&2, Nat>, st: PS, +eta_now: F32,
  +salt: Nat, +draw: Bool) -> PS:
  match abars:
    case Nil{}:
      st
    case Con{a, +ar}:
      match ts:
        case Nil{}:
          st
        case Con{t, tr}:
          match st:
            case PS{p, +x}:
              +now = (1n + t : Nat)
              z = ddim_z(now, salt, draw)
              ddim_go(ar, tr,
                ddim_pe(predict(p, now, x), a, head_or_one(ar), eta_now, z, x),
                eta_now, salt, draw)

# the DDIM sample: pure noise, then down every k-th rung to x_0
def ddim_sample(p: B4P(), +eta_now: F32, +k: Nat, draw: Bool, +salt: Nat) -> PS:
  ddim_go(rung_abars(k), rung_ts(k), PS{p, G.noise_of(2000000n, 24576n)},
    eta_now, salt, draw)

def params_of(tr: T.Train<B4P()>) -> B4P():
  match tr:
    case T.Train{p, g, st, q1, q2, l}:
      p

# ---- the statistics and the grid, printed ----

type Pt is Data:
  Pt{x: F32, y: F32}

def sum_go(x: List<&2, F32>, left: Nat, +s: F32) -> F32:
  match x left:
    case Nil{} 0n:
      s
    case Con{h, t} 0n:
      s
    case Con{h, t} 1n+rest:
      sum_go(t, rest, (s + h : F32))
    case Nil{} 1n+rest:
      s

def mean_of(arr: List<&2, F32>, +count: Nat) -> F32:
  (sum_go(arr, count, 0.0) / F32.from_nat(count) : F32)

# The largest |x - m|: the scale the variance is factored by. The
# ancestral cloud of this example's underfit denoiser grows to 1e20
# over 64 steps, and the plain sum of squares overflows f32 there, so
# the variance divides each deviation by this scale and multiplies it
# back — the same variance, finite at any magnitude.
def dev_go(x: List<&2, F32>, +m: F32, +v: F32) -> F32:
  match x:
    case Nil{}:
      v
    case Con{h, t}:
      dev_go(t, m, F32.max(v, F32.abs((h - m : F32))))

def dev_of(+arr: List<&2, F32>, +m: F32) -> F32:
  dev_go(arr, m, 0.0)

# the sum of squared deviations from m, each divided by s
def sq_go(x: List<&2, F32>, +m: F32, +s: F32, +v: F32) -> F32:
  match x:
    case Nil{}:
      v
    case Con{h, t}:
      +d = ((h - m : F32) / s : F32)
      sq_go(t, m, s, (v + (d * d : F32) : F32))

def std_scaled(+arr: List<&2, F32>, +m: F32, +s: F32, +count: Nat, zero: Bool) -> F32:
  match zero:
    case True{}:
      0.0
    case False{}:
      (s * F32.sqrt((sq_go(arr, m, s, 0.0) / F32.from_nat(count) : F32)) : F32)

def std_of(+arr: List<&2, F32>, +count: Nat) -> F32:
  +m = mean_of(arr, count)
  +s = dev_of(arr, m)
  std_scaled(arr, m, s, count, F32.is_eq(s, 0.0))

def print_points_go(x: List<&2, F32>, left: Nat, +acc: String) -> String:
  match x left:
    case Nil{} 0n:
      acc
    case Con{h, t} 0n:
      acc
    case Con{h, t} 1n+rest:
      rest_txt = print_points_go(t, rest, acc)
      String.append(rest_txt, String.append(" ", F32.show(h)))
    case Nil{} 1n+rest:
      acc

def print_sample_x(+x: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("samples:", print_points_go(x, 24576n, "")))
    return Unit{}

def cifar_of(r: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(List<&2, F32>):
  match r:
    case Fail{e}:
      do IO<List<&2, F32>>:
        IO.print("the dataset failed")
        return Nil{}
    case Done{t}:
      do IO<List<&2, F32>>:
        return cifar_rows(t)

# The ancestral cloud's statistics and the DDIM one's: the same trained
# weights walk down all 64 rungs and then down every 4th rung (issue #12).
# The state hands the parameters to the second sampler the way the 2d
# example's does — they are affine, so one sampler must give them back,
# and `sample` threads them through its state. The statistics are bound
# before the `do` block: an untyped let inside a `do` is refused.
def print_results(st: PS) -> IO(Unit):
  match st:
    case PS{p, +x}:
      +dxs = sample_x(ddim_sample(p, eta(), stride(), False{}, 0n))
      sm = mean_of(x, 24576n)
      ss = std_of(x, 24576n)
      dm = mean_of(dxs, 24576n)
      ds = std_of(dxs, 24576n)
      do IO<Unit>:
        IO.print(String.append("sample_mean: ", F32.show(sm)))
        IO.print(String.append("sample_std: ", F32.show(ss)))
        Unit <- print_sample_x(x)
        IO.print(String.append("ddim_mean: ", F32.show(dm)))
        IO.print(String.append("ddim_std: ", F32.show(ds)))
        IO.print(String.append("ddim_samples:", print_points_go(dxs, 24576n, "")))
        return Unit{}

def print_data_stats(+x0: List<&2, F32>) -> IO(Unit):
  +dm = mean_of(x0, 786432n)
  +ds = std_of(x0, 786432n)
  do IO<Unit>:
    IO.print(String.append("data_mean: ", F32.show(dm)))
    IO.print(String.append("data_std: ", F32.show(ds)))
    return Unit{}

def train_and_report(+x0: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    Unit <- print_data_stats(x0)
    tr : T.Train<B4P()> <- loop(steps(), 0n, x0,
      T.Train.new(~B4P(), ~X(), ~Y(), ~B4S(n()), ~Model(n()), cfg(), params()))
    Unit <- print_results(sample(params_of(tr)))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.csv_of(
      "arwics/cifar10", "f44619abdb57391606d732724f9fc40a684471bf", "train.csv", False{}, 0n)
    x0 : List<&2, F32> <- cifar_of(t)
    Unit <- train_and_report(x0)
    return Unit{}
