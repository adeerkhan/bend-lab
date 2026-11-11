# BendTorch's GPU lane, on a Linear layer of Mamba-3's shape.
#
# `G.LinearNBF` runs its products as bangs in a binary built with CUDA; the
# same program with `+RTS --gpu off -RTS` runs the bangs on the CPU threads.
# Prints the number of differing words between the CPU and GPU forward,
# input gradient and weight gradient: 0 means the bits agree.
#
# The published packages are used because the cache has them; the local
# bendtorch-devel worktree at 65c8038 has the same lane over tensor-gpu
# 0.1.5.2, which is not in the cache.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bendtorch-gpu@2.0.0.0/main.bend as G
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32))

def fill(k: Nat, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (i + 1 : U32), seed, Array.set(F32, a, i, val(i, seed)))

def mat(+r: Nat, +c: Nat, +seed: U32) -> TA.Mat<r, c>:
  TA.Mat{fill(Nat.mul(r, c), 0, seed, Array.new(F32, TA.cap_depth(Nat.mul(r, c)), 0.0))}

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def cmp(k: Nat, +i: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmp(p, (i + 1 : U32), (acc + Bool.pick(U32, U32.is_eq(bits(x), bits(y)), 0, 1) : U32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def diff(+cnt: Nat, x: Array<F32>, y: Array<F32>) -> U32:
  cmp(cnt, 0, 0, Array.get(F32, x, 0), Array.get(F32, y, 0))

type R<-P: Type, -n: Nat, -i: Nat, -o: Nat> is Type:
  R{y: TA.Mat<n, o>, dx: TA.Mat<n, i>, g: P}

def run_b(-P: Type, -n: Nat, -i: Nat, -o: Nat, y: TA.Mat<n, o>, r: T.Bwd<P, TA.Mat<n, i>>) -> R<P, n, i, o>:
  match r:
    case T.Bwd{p, dx, g}: R{y, dx, g}

def run_f(~n: Nat, ~i: Nat, ~o: Nat, ~P: Type, ~l: T.Layer<P, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>, dy: TA.Mat<n, o>, g: P, r: T.Fwd<P, TA.Mat<n, o>, TA.Mat<n, i>>) -> R<P, n, i, o>:
  match r:
    case T.Fwd{p, y, s}: run_b(P, n, i, o, y, T.Layer.bwd(P, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>, l, p, s, dy, g))

def run(~n: Nat, ~i: Nat, ~o: Nat, ~P: Type, ~l: T.Layer<P, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>, p: P, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: P) -> R<P, n, i, o>:
  run_f(~n, ~i, ~o, ~P, ~l, dy, g, T.Layer.fwd(P, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>, l, p, x))

def d_nb(+n: Nat, +i: Nat, +o: Nat, a: R<TA.Mat<o, i>, n, i, o>, b: R<TA.Mat<o, i>, n, i, o>) -> U32:
  match a b:
    case R{TA.Mat{y1}, TA.Mat{d1}, TA.Mat{w1}} R{TA.Mat{y2}, TA.Mat{d2}, TA.Mat{w2}}:
      ((diff(Nat.mul(n, o), y1, y2) + diff(Nat.mul(n, i), d1, d2) : U32) + diff(Nat.mul(o, i), w1, w2) : U32)

def go(p: TA.Mat<304n, 64n>, x: TA.Mat<8n, 64n>,
    dy: TA.Mat<8n, 304n>, g: TA.Mat<304n, 64n>,
    p2: TA.Mat<304n, 64n>, x2: TA.Mat<8n, 64n>,
    dy2: TA.Mat<8n, 304n>, g2: TA.Mat<304n, 64n>) -> IO(Unit):
  do IO<Unit>:
    IO.print(U32.show(d_nb(8n, 64n, 304n,
      run(~8n, ~64n, ~304n, ~TA.Mat<304n, 64n>, ~T.LinearNB(8n, 64n, 304n, 0n), p, x, dy, g),
      run(~8n, ~64n, ~304n, ~TA.Mat<304n, 64n>, ~G.LinearNBF(8n, 64n, 304n, 0n), p2, x2, dy2, g2))))
    return Unit{}

def main() -> IO(Unit):
  go(mat(304n, 64n, 1), mat(8n, 64n, 2), mat(8n, 304n, 3),
    mat(304n, 64n, 5), mat(304n, 64n, 1), mat(8n, 64n, 2),
    mat(8n, 304n, 3), mat(304n, 64n, 5))
