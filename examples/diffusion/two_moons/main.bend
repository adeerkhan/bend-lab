# A diffusion model trained and sampled in Bend, with the real BendTorch
# (`bendtorch@2.0.0.0`).
#
#   bendenv run -- bend examples/diffusion/main.bend
#
# The model is a small denoiser — three linear layers with SiLU — trained
# by `T.Train.step` with AdamW to predict the noise ε of a DDPM forward
# process over a two-moons dataset of 128 points, then sampled two ways:
# the ancestral DDPM loop over all 64 rungs, and DDIM over every 4th
# rung (16 model calls instead of 64, deterministic at η = 0). Everything
# that looks random is a pure function of counters: the Gaussian noise
# is Box-Muller over `mix32`-hashed uniforms (murmur3's finaliser, over
# U32 arithmetic that wraps), so a NumPy twin can draw the same numbers
# bit for bit (reference/test_diffusion.py, reference/test_ddim.py).
#
# The forward process is x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε over 32 steps
# of a linear β schedule; the model input concatenates x_t (2 features)
# with an 8-number sinusoidal time embedding. F32.sin, F32.cos, F32.log
# and F32.exp are Base's own. The data, the noise, the embeddings and the
# samples move as List<&2, F32> (duplicable); only the model's matrices
# are bend-ml-tensor-array's, at the edges.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G

# The shape: 128 points of 2 features, a time embedding of 8, a hidden
# width of 32, 32 diffusion steps, 400 training steps.

def n() -> Nat:
  128n

def d() -> Nat:
  2n

def inwidth() -> Nat:
  10n

def hidden() -> Nat:
  64n

def steps() -> Nat:
  4000n

def X() -> Type:
  TA.Mat<n(), inwidth()>

def Y() -> Type:
  TA.Mat<n(), d()>

def M2(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# A pair of numbers, as a record: the affine rule has no field access, so
# a pair is matched through a def, never read.

type Pt is Data:
  Pt{x: F32, y: F32}

def pt_x(p: Pt) -> F32:
  match p:
    case Pt{a, b}:
      a

def pt_y(p: Pt) -> F32:
  match p:
    case Pt{a, b}:
      b

# The number at an index of a list (linear walk, the attention example's
# `at`).

def at_go(xs: List<&2, F32>, +k: Nat, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      Bool.pick(F32, Nat.is_eq(i, k), h, at_go(t, k, (1n + i : Nat)))

def at(xs: List<&2, F32>, k: Nat) -> F32:
  at_go(xs, k, 0n)

# ---- the denoiser: linear, SiLU, linear, SiLU, linear ----

def B1P() -> Type:
  T.Two<T.Lin<inwidth(), hidden()>, Unit>

def B1S() -> Type:
  T.Two<M2(n(), inwidth()), M2(n(), hidden())>

def B2P() -> Type:
  T.Two<B1P, T.Lin<hidden(), hidden()>>

def B2S() -> Type:
  T.Two<B1S, M2(n(), hidden())>

def B3P() -> Type:
  T.Two<B2P, Unit>

def B3S() -> Type:
  T.Two<B2S, M2(n(), hidden())>

def B4P() -> Type:
  T.Two<B3P, T.Lin<hidden(), d()>>

def B4S() -> Type:
  T.Two<B3S, M2(n(), hidden())>

def B1() -> T.Layer<B1P, X(), M2(n(), hidden()), B1S>:
  T.Seq(~T.Lin<inwidth(), hidden()>, ~Unit, ~X(), ~M2(n(), hidden()),
    ~M2(n(), hidden()), ~M2(n(), inwidth()), ~M2(n(), hidden()),
    ~T.LinearFirst(n(), inwidth(), hidden(), 0n), ~T.SiLU(n(), hidden()))

def B2() -> T.Layer<B2P, X(), M2(n(), hidden()), B2S>:
  T.Seq(~B1P, ~T.Lin<hidden(), hidden()>, ~X(), ~M2(n(), hidden()),
    ~M2(n(), hidden()), ~B1S, ~M2(n(), hidden()), ~B1(), ~T.Linear(n(), hidden(), hidden(), 0n))

def B3() -> T.Layer<B3P, X(), M2(n(), hidden()), B3S>:
  T.Seq(~B2P, ~Unit, ~X(), ~M2(n(), hidden()), ~M2(n(), hidden()),
    ~B2S, ~M2(n(), hidden()), ~B2(), ~T.SiLU(n(), hidden()))

def Model() -> T.Layer<B4P, X(), Y(), B4S>:
  T.Seq(~B3P, ~T.Lin<hidden(), d()>, ~X(), ~M2(n(), hidden()), ~Y(),
    ~B3S, ~M2(n(), hidden()), ~B3(), ~T.Linear(n(), hidden(), d(), 0n))

# ---- the data: two moons, half a circle each, on a fixed grid ----

# one moon point: the upper half of a circle, or the lower one flipped
def moon_point(is_upper: Bool, +k: Nat) -> Pt:
  match is_upper:
    case True{}:
      +th = ((3.1415927 * (F32.from_nat(k) + 0.5 : F32)) / 64.0 : F32)
      Pt{F32.cos(th), F32.sin(th)}
    case False{}:
      +th = ((3.1415927 * (F32.from_nat(k) + 0.5 : F32)) / 64.0 : F32)
      Pt{(1.0 - F32.cos(th) : F32), (0.5 - F32.sin(th) : F32)}

def moons_go(+left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +p1 = moon_point(True{}, Nat.sub(64n, left))
      +p2 = moon_point(False{}, Nat.sub(64n, left))
      a1 = List.append(&2, F32, acc, Con{pt_x(p1), Nil{}})
      a2 = List.append(&2, F32, a1, Con{pt_y(p1), Nil{}})
      a3 = List.append(&2, F32, a2, Con{pt_x(p2), Nil{}})
      moons_go(rest, List.append(&2, F32, a3, Con{pt_y(p2), Nil{}}))

# the 256 numbers of the two moons, the upper moon first
def moons() -> List<&2, F32>:
  moons_go(64n, Nil{})

# ---- the model input: x_t beside its time embedding ----

def input_row(+s: Nat, +r: Nat, +a: F32, +b: F32, +x0: List<&2, F32>,
  +te: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  +i0 = Nat.mul(r, 2n)
  ax = (a * at(x0, i0) : F32)
  bx = (b * G.gauss_at(s, i0) : F32)
  ay = (a * at(x0, (1n + i0 : Nat)) : F32)
  by = (b * G.gauss_at(s, (1n + i0 : Nat)) : F32)
  a1 = List.append(&2, F32, acc, Con{(ax + bx : F32), Nil{}})
  a2 = List.append(&2, F32, a1, Con{(ay + by : F32), Nil{}})
  a3 = List.append(&2, F32, a2, Con{at(te, 0n), Nil{}})
  a4 = List.append(&2, F32, a3, Con{at(te, 1n), Nil{}})
  a5 = List.append(&2, F32, a4, Con{at(te, 2n), Nil{}})
  a6 = List.append(&2, F32, a5, Con{at(te, 3n), Nil{}})
  a7 = List.append(&2, F32, a6, Con{at(te, 4n), Nil{}})
  a8 = List.append(&2, F32, a7, Con{at(te, 5n), Nil{}})
  a9 = List.append(&2, F32, a8, Con{at(te, 6n), Nil{}})
  List.append(&2, F32, a9, Con{at(te, 7n), Nil{}})

def input_go(+s: Nat, +a: F32, +b: F32, +x0: List<&2, F32>, +te: List<&2, F32>,
  +left: Nat, +r: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      input_go(s, a, b, x0, te, rest, (1n + r : Nat),
        input_row(s, r, a, b, x0, te, acc))

# the input of training step s at noise level t: 128 rows of 10 numbers,
# x_t then its embedding, flat in row-major order
def input_of(+s: Nat, +t: Nat, +a: F32, +b: F32, +x0: List<&2, F32>) -> List<&2, F32>:
  input_go(s, a, b, x0, G.temb_of(t, 64n, 4n), 128n, 0n, Nil{})

# ---- the training loop ----

def cfg() -> T.OptCfg:
  T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

def mse(y: Y(), tgt: Y()) -> T.LossOut<Y()>:
  T.MSE(n(), d(), y, tgt)

# the starting weights: a fixed small pattern, not zeros
def value(+i: Nat, +j: Nat, +seed: Nat) -> F32:
  F32.sub(F32.mul(F32.from_nat(Nat.mod(Nat.add(Nat.add(Nat.mul(i, 3n), Nat.mul(j, 5n)), seed), 7n)),
    0.1), 0.2)

def row_vals(+c: Nat, +i: Nat, +j: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match c:
    case 0n:
      acc
    case 1n+rest:
      row_vals(rest, i, (1n + j : Nat), seed,
        List.append(&2, F32, acc, Con{value(i, j, seed), Nil{}}))

def rows_vals(+r: Nat, +c: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match r:
    case 0n:
      acc
    case 1n+rest:
      rows_vals(rest, c, (1n + i : Nat), seed,
        List.append(&2, F32, acc, row_vals(c, i, 0n, seed, Nil{})))

def mat_of(+r: Nat, +c: Nat, +seed: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, rows_vals(r, c, 0n, seed, Nil{}))

def params() -> B4P():
  T.Two{
    T.Two{
      T.Two{T.Two{T.Lin{mat_of(hidden(), inwidth(), 1n), TA.Mat.fill(1n, hidden(), 0.0)}, Unit{}},
      T.Lin{mat_of(hidden(), hidden(), 2n), TA.Mat.fill(1n, hidden(), 0.0)}},
    Unit{}},
  T.Lin{mat_of(d(), hidden(), 3n), TA.Mat.fill(1n, d(), 0.0)}}

# ᾱ_t of the linear schedule, by index
def abar(+t: Nat) -> F32:
  at(G.lin_alphas(64n), Nat.sub(t, 1n))

# ᾱ_{t-1}, the rung below: ᾱ_0 = 1 at the bottom (the schedule has no
# entry before the first), the entry before otherwise
def abar_below(+t: Nat) -> F32:
  Bool.pick(F32, Nat.is_eq(t, 1n), 1.0, at(G.lin_alphas(64n), Nat.sub(t, 2n)))

# sqrt(ᾱ_t) and sqrt(1-ᾱ_t), the forward process's two coefficients, over
# the linear schedule of mlbend-generative (gentle at the top: a 64-step
# sampler needs every reverse scale small, see the package's note)
def coef_a(+t: Nat) -> F32:
  F32.sqrt(abar(t))

def coef_b(+t: Nat) -> F32:
  F32.sqrt((1.0 - abar(t) : F32))

def do_step(+s: Nat, tr: T.Train<B4P()>) -> T.Train<B4P()>:
  +t = Nat.add(Nat.mod(s, 64n), 1n)
  T.Train.step(~B4P(), ~X(), ~Y(), ~B4S(), ~Y(), ~Model(), ~mse, tr,
    TA.Mat.from_list(n(), inwidth(), input_of(s, t, coef_a(t), coef_b(t), moons())),
    TA.Mat.from_list(n(), d(), G.noise_of(s, 256n)))

def should_print(+i: Nat) -> Bool:
  Bool.or(Bool.or(Nat.is_eq(i, 1n), Nat.is_eq(Nat.mod((1n + i : Nat), 40n), 0n)),
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

def loop(left: Nat, +i: Nat, tr: T.Train<B4P()>) -> IO(T.Train<B4P()>):
  match left tr:
    case 0n T.Train{p, g, st, c, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        return T.Train{p, g, st, c, q1, q2, l}
    case 1n+rest T.Train{p, g, st, c, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        Unit <- print_or(should_print(i), l)
        loop(rest, (1n + i : Nat), do_step(i, T.Train{p, g, st, c, q1, q2, l}))

# ---- the sample: the ancestral DDPM loop, from pure noise to a moon ----
#
# The parameters are affine (they hold arrays), so they are not copied but
# returned: `Layer.fwd` answers `Fwd{p, y, s}` with the parameters inside,
# and the loop threads them through a record, the way bendtorch's own
# eval loop does.

# the parameters and the current sample, threaded through the loop

type PS is Type:
  PS{p: B4P(), x: List<&2, F32>}

# one denoise: the parameters come back beside the predicted noise

type PE is Type:
  PE{p: B4P(), eps: List<&2, F32>}

# the flat list of a matrix (Mat.to_list answers ML{mat, list})
def flat_of(m: TA.ML<n(), d()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def eps_list(m: Y()) -> List<&2, F32>:
  flat_of(TA.Mat.to_list(n(), d(), m))

# the denoiser's forward pass over one input list
def fwd_eps_of(r: T.Fwd<B4P(), Y(), B4S()>) -> PE:
  match r:
    case T.Fwd{p2, y, s}:
      PE{p2, eps_list(y)}

# the denoiser's answer for one sample (a = 1, b = 0: the sample is not
# noised again, the embedding rides beside it)
def predict(p: B4P(), +t: Nat, +x: List<&2, F32>) -> PE:
  fwd_eps_of(T.Layer.fwd(B4P(), X(), Y(), B4S(), Model(), p,
    TA.Mat.from_list(n(), inwidth(), input_of((1000000n + t : Nat), t, 1.0, 0.0, x))))

# One element of the new sample. The mean is scaled by 1/sqrt(1-beta_t)
# and the draw is added after it, outside the scale (issue #14): Ho et
# al.'s Algorithm 2 adds sigma_t·z to the already-scaled mean, and this
# is exactly what makes DDIM with eta = 1 reproduce the step.
def rev_elem(+coef: F32, +scale: F32, +sig: F32, +xv: F32, +eps: F32, +z: F32) -> F32:
  ((scale * (xv - (coef * eps : F32) : F32) : F32) + (sig * z : F32) : F32)

def rev_go(+coef: F32, +scale: F32, +sig: F32, +x: List<&2, F32>, +eps: List<&2, F32>,
  +z: List<&2, F32>, +left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +j = Nat.sub(256n, left)
      rev_go(coef, scale, sig, x, eps, z, rest,
        List.append(&2, F32, acc,
          Con{rev_elem(coef, scale, sig, at(x, j), at(eps, j), at(z, j)), Nil{}}))

# one reverse step, over the denoise's answer: the new parameters beside
# the new sample
def rev_pe(pe: PE, +coef: F32, +scale: F32, +sig: F32, +z: List<&2, F32>,
  +x: List<&2, F32>) -> PS:
  match pe:
    case PE{p2, eps}:
      PS{p2, rev_go(coef, scale, sig, x, eps, z, 256n, Nil{})}

# One reverse step of the ancestral loop. The mean is the DDPM step,
# x0-substituted (its two forms are the same expression); the noise is
# the exact posterior's sigma = sqrt(beta_tilde_t), issue #14 — the one
# DDIM with eta = 1 reproduces exactly, which is the equivalence
# reference/test_ddim.py checks.
def reverse_step(+t: Nat, st: PS) -> PS:
  match st:
    case PS{p, +x}:
      +beta = G.lin_beta_at(t, 64n)
      +abar_t = abar(t)
      +coef = (beta / F32.sqrt((1.0 - abar_t : F32)) : F32)
      +scale = (1.0 / F32.sqrt((1.0 - beta : F32)) : F32)
      +sig = F32.sqrt((beta * ((1.0 - abar_below(t) : F32) / (1.0 - abar_t : F32)) : F32))
      rev_pe(predict(p, t, x), coef, scale, sig, G.noise_of((1000000n + t : Nat), 256n), x)

def sample_go(t: Nat, st: PS) -> PS:
  match t:
    case 0n:
      st
    case 1n++rest:
      sample_go(rest, reverse_step(t, st))



# x_32 from pure noise (a counter past the training ones), then down to x_0
def sample(p: B4P()) -> PS:
  sample_go(64n, PS{p, G.noise_of(2000000n, 256n)})

def sample_x(st: PS) -> List<&2, F32>:
  match st:
    case PS{p, x}:
      x

# ---- the DDIM sampler (Song et al., 2020): the same model, every k-th rung ----
#
# DDIM walks a subsequence of the schedule: of the 64 rungs it takes
# every k-th one (k = 4 below), so the model runs 16 times instead of
# 64. The two rung lists — the ᾱ of each rung and its index — are built
# by the package's strided walk (which emits ascending; List.reverse
# turns it around into reverse time) and walked in lockstep. eta (η)
# picks the stochasticity: at η = 0 a step reads no noise at all — the
# z list is empty and σ·0.0 = 0 — so the sample is a pure function of
# the model and the starting noise; at η = 1 over every rung a step IS
# the ancestral one above, because that σ² is exactly the posterior's
# β̃_t (issue #14, checked in reference/test_ddim.py).

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
      G.noise_of((1000000n + (salt + t : Nat) : Nat), 256n)
    case False{}:
      Nil{}

# one rung: the model's noise there, folded with the rung below into
# the sample (the package's ddim does the folding and keeps the shape)
def ddim_pe(pe: PE, +abar_now: F32, +abar_prev: F32, +eta_now: F32, +z: List<&2, F32>,
  +x: List<&2, F32>) -> PS:
  match pe:
    case PE{p2, eps}:
      PS{p2, G.ddim(G.step_of(abar_now, abar_prev, eta_now), x, eps, z)}

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
  ddim_go(rung_abars(k), rung_ts(k), PS{p, G.noise_of(2000000n, 256n)},
    eta_now, salt, draw)

# ---- the statistics, printed ----

def sum2_go(+arr: List<&2, F32>, +left: Nat, +sx: F32, +sy: F32) -> Pt:
  match left:
    case 0n:
      Pt{sx, sy}
    case 1n+rest:
      +i = Nat.mul(Nat.sub(128n, left), 2n)
      sum2_go(arr, rest,
        (sx + at(arr, i) : F32),
        (sy + at(arr, (1n + i : Nat)) : F32))

def div2_of(p: Pt) -> Pt:
  match p:
    case Pt{sx, sy}:
      Pt{(sx / 128.0 : F32), (sy / 128.0 : F32)}

def means(arr: List<&2, F32>) -> Pt:
  div2_of(sum2_go(arr, 128n, 0.0, 0.0))

def var2_go(+arr: List<&2, F32>, +left: Nat, +mx: F32, +my: F32, +vx: F32, +vy: F32) -> Pt:
  match left:
    case 0n:
      Pt{vx, vy}
    case 1n+rest:
      +i = Nat.mul(Nat.sub(128n, left), 2n)
      +dx = (at(arr, i) - mx : F32)
      +dy = (at(arr, (1n + i : Nat)) - my : F32)
      var2_go(arr, rest, mx, my, (vx + (dx * dx : F32) : F32), (vy + (dy * dy : F32) : F32))

def vars_of(arr: List<&2, F32>, m: Pt) -> Pt:
  match m:
    case Pt{mx, my}:
      var2_go(arr, 128n, mx, my, 0.0, 0.0)

def sqrt2_of(p: Pt) -> Pt:
  match p:
    case Pt{vx, vy}:
      Pt{F32.sqrt((vx / 128.0 : F32)), F32.sqrt((vy / 128.0 : F32))}

def stds(+arr: List<&2, F32>) -> Pt:
  sqrt2_of(vars_of(arr, means(arr)))

def show_pair(tag: String, p: Pt) -> String:
  match p:
    case Pt{a, b}:
      String.append(tag, String.append(" ", String.append(F32.show(a),
        String.append(" ", F32.show(b)))))

def params_of(tr: T.Train<B4P()>) -> B4P():
  match tr:
    case T.Train{p, g, st, c, q1, q2, l}:
      p

def pair_text(p: Pt) -> String:
  match p:
    case Pt{a, b}:
      String.append(F32.show(a), String.append(" ", F32.show(b)))

def print_points_go(+x: List<&2, F32>, +k: Nat, +acc: String) -> String:
  match k:
    case 0n:
      acc
    case 1n+rest:
      +i = Nat.mul(Nat.sub(128n, k), 2n)
      rest_txt = print_points_go(x, rest, acc)
      String.append(rest_txt, String.append(" ",
        String.append(F32.show(at(x, i)), String.append(" ", F32.show(at(x, (1n + i : Nat)))))))

def print_ddim(p: B4P()) -> IO(Unit):
  +dx = sample_x(ddim_sample(p, eta(), stride(), False{}, 0n))
  do IO<Unit>:
    IO.print(show_pair("ddim_mean:", means(dx)))
    IO.print(show_pair("ddim_std:", stds(dx)))
    IO.print(String.append("ddim_samples:", print_points_go(dx, 128n, "")))
    return Unit{}

# the ancestral cloud's statistics, then the DDIM one's (the DDIM loop
# takes the parameters back out of the ancestral one's state: they are
# affine, so one sampler hands them to the next)
def report_sample(st: PS) -> IO(Unit):
  match st:
    case PS{p, +x}:
      do IO<Unit>:
        IO.print(show_pair("sample_mean:", means(x)))
        IO.print(show_pair("sample_std:", stds(x)))
        IO.print(String.append("samples:", print_points_go(x, 128n, "")))
        Unit <- print_ddim(p)
        return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    tr : T.Train<B4P()> <- loop(steps(), 0n,
      T.Train.new(~B4P(), ~X(), ~Y(), ~B4S(), ~Model(), cfg(), params()))
    IO.print(show_pair("data_mean:", means(moons())))
    IO.print(show_pair("data_std:", stds(moons())))
    Unit <- report_sample(sample(params_of(tr)))
    return Unit{}
