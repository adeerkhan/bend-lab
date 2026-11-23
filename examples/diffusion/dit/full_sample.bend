# DDIM sampler for the full-width Bendlang DiT denoiser.
# Loads a checkpoint trained by full_main.bend, runs the 16-step DDIM walk with
# the clamped start, and prints the 4096 sample pixels for the legibility gate.
# The model and all helpers are the ViT's sampler, pointed at full_main's Model.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import bendtorch@2.1.0.0/main.bend as T
import mlbend-generative@0.1.1.0/main.bend as G
import ./full_main.bend as M

def RW() -> Nat: 1024n
def FE() -> Nat: 20n
def PX() -> Nat: 4n

# ---- helpers (from the ViT sampler; identical token layout) ----

def at(xs: List<&2, F32>, +k: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      Bool.pick(F32, Nat.is_eq(k, 0n), h, at(t, Nat.sub(k, 1n)))

def con_go(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      con_go(t, Con{h, acc})

def cat_go(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      cat_go(t, List.append(&2, F32, acc, Con{h, Nil{}}))

def snoc(xs: List<&2, F32>, +v: F32) -> List<&2, F32>:
  List.append(&2, F32, xs, Con{v, Nil{}})

def drop64_go(xs: List<&2, F32>, left: Nat) -> List<&2, F32>:
  match xs left:
    case Nil{} _:
      Nil{}
    case _ 0n:
      xs
    case Con{h, t} 1n+rest:
      drop64_go(t, rest)
    case _ _:
      Nil{}

def drop64(xs: List<&2, F32>) -> List<&2, F32>:
  drop64_go(xs, 64n)

def pad64_go(xs: List<&2, F32>, +k: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs k:
    case Nil{} _:
      List.reverse(&2, F32, acc)
    case Con{h, t} 0n:
      List.reverse(&2, F32, acc)
    case Con{h, t} 1n+q:
      pad64_go(t, q, Con{h, acc})

def pad64(xs: List<&2, F32>) -> List<&2, F32>:
  pad64_go(xs, 64n, Nil{})


def take4(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Con{a, Con{b, Con{c, Con{d, t}}}}:
      Con{a, Con{b, Con{c, Con{d, Nil{}}}}}
    case _:
      Nil{}

def patch_rev(left: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      patch_rev(rest, drop64(xs), con_go(take4(pad64(xs), Nil{}), acc))
    case _:
      acc

def patch_all(left: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  List.reverse(&2, F32, patch_rev(left, xs, acc))

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

def rows_go(patched: List<&2, F32>, +te: List<&2, F32>, +k: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match patched:
    case Nil{}:
      acc
    case Con{a, Con{b, Con{c, Con{d, t}}}}:
      r1 = snoc(snoc(snoc(snoc(Nil{}, a), b), c), d)
      r2 = cat_go(te, r1)
      r3 = cat_go(pos_of(k), r2)
      rows_go(t, te, Nat.mod((1n + k : Nat), 16n), List.append(&2, F32, acc, r3))
    case _:
      acc

def tinput(digits: List<&2, F32>, +t: Nat) -> List<&2, F32>:
  rows_go(patch_all(64n, digits, Nil{}), G.temb_of(t, 64n, 4n), 0n, Nil{})

def abar(+t: Nat) -> F32:
  at(G.lin_alphas(64n), Nat.sub(t, 1n))

def abar_below(+t: Nat) -> F32:
  Bool.pick(F32, Nat.is_eq(t, 1n), 1.0, at(G.lin_alphas(64n), Nat.sub(t, 2n)))

# ---- sampler state ----

type VS is Type:
  VS{p: M.ModelP(), x: List<&2, F32>}

type VE is Type:
  VE{p: M.ModelP(), eps: List<&2, F32>}

def flat_m2(m: TA.ML<RW(), PX()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def eps_of(f: T.Fwd<M.ModelP, M.Y, M.ModelS>) -> VE:
  match f:
    case T.Fwd{p, y, s}:
      VE{p, flat_m2(TA.Mat.to_list(RW(), PX(), y))}

def pairs(+left: Nat, +v: F32, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+q:
      pairs(q, v, Con{v, acc})

def predict(p: M.ModelP(), +t: Nat, +x: List<&2, F32>) -> VE:
  eps_of(T.Layer.fwd(M.ModelP(), M.X, M.Y, M.ModelS(), M.Model(), p,
    T.Two{TA.Mat.from_list(64n, 3n, pairs(64n, 0.25, Nil{})),
      TA.Mat.from_list(RW(), FE(), tinput(x, t))}))

def rev_elem(+c1: F32, +c2: F32, +sa: F32, +sm: F32, +sig: F32, +xv: F32, +eps: F32, +z: F32) -> F32:
  x0 = ((xv - (sm * eps : F32) : F32) / sa : F32)
  xc = F32.clamp(x0, 0.0, 1.0)
  (((c1 * xc : F32) + (c2 * xv : F32) : F32) + (sig * z : F32) : F32)

def rev_go(+c1: F32, +c2: F32, +sa: F32, +sm: F32, +sig: F32, x: List<&2, F32>, eps: List<&2, F32>, z: List<&2, F32>, left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match x eps z left:
    case Con{xv, xt} Con{ev, et} Con{zv, zt} 1n+rest:
      rev_go(c1, c2, sa, sm, sig, xt, et, zt, rest, Con{rev_elem(c1, c2, sa, sm, sig, xv, ev, zv), acc})
    case _ _ _ _:
      acc

def rev_pe(ve: VE, +c1: F32, +c2: F32, +sa: F32, +sm: F32, +sig: F32, +z: List<&2, F32>, +x: List<&2, F32>) -> VS:
  match ve:
    case VE{p2, eps}:
      VS{p2, List.reverse(&2, F32, rev_go(c1, c2, sa, sm, sig, x, eps, z, 4096n, Nil{}))}

def rev_step(+t: Nat, st: VS) -> VS:
  match st:
    case VS{p, +x}:
      +beta = G.lin_beta_at(t, 64n)
      +abar_t = abar(t)
      +bar_prev = abar_below(t)
      +c1 = (beta * (F32.sqrt(bar_prev) / (1.0 - abar_t) : F32) : F32)
      +c2 = (F32.sqrt((1.0 - beta : F32)) * ((1.0 - bar_prev : F32) / (1.0 - abar_t : F32)) : F32)
      +sa = F32.sqrt(abar_t)
      +sm = F32.sqrt((1.0 - abar_t : F32))
      +sig = F32.sqrt((beta * ((1.0 - bar_prev : F32) / (1.0 - abar_t : F32)) : F32))
      rev_pe(predict(p, t, x), c1, c2, sa, sm, sig, G.noise_of((1000000n + t : Nat), 4096n), x)

def sample_go(t: Nat, st: VS) -> VS:
  match t:
    case 0n:
      st
    case 1n++rest:
      sample_go(rest, rev_step(t, st))

def sample(p: M.ModelP()) -> VS:
  sample_go(64n, VS{p, patch_all(64n, G.noise_of(2000000n, 4096n), Nil{})})

# ---- DDIM 16-step ----

def clip_eps_go(left: Nat, x: List<&2, F32>, eps: List<&2, F32>, +sa: F32, +sm: F32, +acc: List<&2, F32>) -> List<&2, F32>:
  match left x eps:
    case 1n+rest Con{+xv, xt} Con{ev, et}:
      x0 = ((xv - (sm * ev : F32) : F32) / sa : F32)
      xc = F32.clamp(x0, 0.0, 1.0)
      clip_eps_go(rest, xt, et, sa, sm, Con{((xv - (sa * xc : F32) : F32) / sm : F32), acc})
    case _ _ _:
      acc

def clip_eps(+abar_now: F32, +x: List<&2, F32>, +eps: List<&2, F32>) -> List<&2, F32>:
  +sa = F32.sqrt(abar_now)
  +sm = F32.sqrt((1.0 - abar_now : F32))
  List.reverse(&2, F32, clip_eps_go(4096n, x, eps, sa, sm, Nil{}))

def rung_abars(+k: Nat) -> List<&2, F32>:
  List.reverse(&2, F32, G.strided_from(F32, G.lin_alphas(64n), k, Nat.mod(63n, k)))

def rung_ts(+k: Nat) -> List<&2, Nat>:
  List.reverse(&2, Nat, G.strided_from(Nat, List.range(64n), k, Nat.mod(63n, k)))

def head_or_one(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      1.0
    case Con{h, t}:
      h

def ddim_pe(ve: VE, +a: F32, +ar: List<&2, F32>, +x: List<&2, F32>) -> VS:
  match ve:
    case VE{p2, eps}:
      VS{p2, G.ddim(G.step_of(a, head_or_one(ar), 0.0), x, clip_eps(a, x, eps), Nil{})}

def ddim_go(abars: List<&2, F32>, ts: List<&2, Nat>, st: VS) -> VS:
  match abars:
    case Nil{}:
      st
    case Con{a, +ar}:
      match ts:
        case Nil{}:
          st
        case Con{t, tr}:
          match st:
            case VS{p, +x}:
              +now = (1n + t : Nat)
              ddim_go(ar, tr, ddim_pe(predict(p, now, x), a, ar, x))

def ddim_sample(p: M.ModelP()) -> VS:
  ddim_go(rung_abars(4n), rung_ts(4n), VS{p, patch_all(64n, G.noise_of(2000000n, 4096n), Nil{})})

# ---- unpatch to pixel order and print ----

def unpatch1(left: Nat, +t: List<&2, F32>, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +k = (Nat.add(Nat.mul(4n, Nat.div(Nat.div(i, 8n), 2n)), Nat.div(Nat.mod(i, 8n), 2n)) : Nat)
      +d = (Nat.add(Nat.mul(2n, Nat.mod(Nat.div(i, 8n), 2n)), Nat.mod(i, 2n)) : Nat)
      unpatch1(rest, t, (1n + i : Nat), List.append(&2, F32, acc, Con{at(t, Nat.add(Nat.mul(4n, k), d)), Nil{}}))
    case _:
      acc

def unpatch_all(left: Nat, +t: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      unpatch_all(rest, drop64(t), List.append(&2, F32, acc, unpatch1(64n, t, 0n, Nil{})))
    case _:
      acc

def show_cloud(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_cloud(t)))

def arg1(args: List<&1, String>) -> String:
  match args:
    case Con{h, t}:
      h
    case Nil{}:
      ""

def params_of(tr: T.Train<M.ModelP>) -> IO(M.ModelP):
  match tr:
    case T.Train{p, g, st, c, p1, p2, l}:
      IO.pure(M.ModelP, p)

def show_vs(st: VS) -> IO(Unit):
  match st:
    case VS{p2, x}:
      +px = unpatch_all(64n, x, Nil{})
      IO.print(String.append("samples: ", show_cloud(px)))

def from_params(p: M.ModelP) -> IO(Unit):
  show_vs(ddim_sample(p))

def after_load(xs: List<&2, F32>) -> IO(Unit):
  from_params(M.restore_params(xs))

def run(path: String) -> IO(Unit):
  do IO<Unit>:
    xs : List<&2, F32> <- M.floats_at(path)
    after_load(xs)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<&1, String> <- IO.args()
    run(arg1(args))
