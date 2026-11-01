# Traps

The Bend and pipeline traps that cost real time here, each with a minimal
program and the error or the measurement it produced. Every error text below
was printed by `bend` 2.0.38 on this repository's pinned install.

## A `match` cannot inspect a computed value

```bend
def same(n: Nat, m: Nat) -> Nat:
  match Nat.is_eq(n, m):
    case true:
      1
    case false:
      0
```

```
SOME PROOFS FAIL
Error:
- message  : a parameter or field scrutinee (a match cannot scrutinize a computed value: give it its own def)
Location:
3>|   match Nat.is_eq(n, m):
  |         ^^^^^^^^^^^^^^^
```

Give the value its own def, or compute it at the call site and pass the `Bool`
to a def that matches on its parameter: `same_len_of(Nat.is_eq(n, m), n, m)`.

## A negative literal is not a term

```bend
def half() -> F32:
  -0.5
```

```
SOME PROOFS FAIL
Error:
- expected : a name
- observed : '0'
Location:
3>|   -0.5
  |    ^
```

Write `(0.0 - 0.5 : F32)`.

## An import path segment must start with a letter

```bend
import traps/2d/mod as M
```

```
SOME PROOFS FAIL
Error:
- expected : an import of a .bend file
- observed : 'traps/2d/mod'
Location:
1>| import traps/2d/mod as M
  |        ^
```

This is why `examples/diffusion/2d` became `two_moons` (the task id stayed
`diffusion-2d`).

## A literal too large for its type

```bend
def main() -> IO<Unit>:
  x = Nat.mul(281474976710656, 2)
```

```
SOME PROOFS FAIL
Error:
- expected : a u32 literal up to 4294967295 (got 281474976710656)
- observed : '281474976710656'
```

`Nat` arithmetic fail-stops past 2^48-1 at run time; keep counters in `U32` and
mix with `Nat.mul` only where the bound is real.

## A constructor that was never declared

```bend
match y:
  case Some{v}:
    ...
```

```
SOME PROOFS FAIL
Error:
- message  : a declared constructor (unknown: Some)
```

Constructors come from a declaration in the program or a package (`Option` is
the standard one; `Option.some`/`Option.none` are the shapes it provides), not
from another language's spelling.

## A def with no return type is a law

```bend
def broken(): F32 = true
```

```
SOME PROOFS FAIL
Error:
- expected : '->' (a def with no return type fills a law; no law named broken is in scope)
- observed : ':'
```

Write `def broken() -> F32:`. The error names the reason, which is why the
`--fast` gate is a useful first stop: it fails in a second with this text.

## A build must run under the guard

`bend -o` builds the C program in one pass and its memory grows with the model.
On this machine a 512-row model and its data peak at 4.3 GB and the full
example reached 9.3 GB before the kernel killed it — with nothing else OOM-killed
first, because the limit was the whole machine. Run heavy builds as

```sh
BEND_MEM=9G scripts/bend-guard.sh bend -o out main.bend
```

A build that dies inside the cgroup is a smaller build, never a bigger limit.

## `IO.now` is milliseconds, and a small phase is `0`

The stage split times phases with `IO.now`, which is whole milliseconds. A phase
below one millisecond prints `0`, and the page prints `<1 ms` rather than
inventing precision. A pure phase that returns `Unit` is erased by the compiler
and also times as `0`: the work must land in a value that is used.

## stdout is block-buffered

A program's output arrives in blocks, so a run that prints one line per step
shows nothing until the block fills — and a killed build looks silent rather
than slow. This repository learned it on the ViT example; `scripts/vit.sh` runs
the program under `stdbuf -oL` and `setsid` for that reason. A log that stays
empty for minutes is buffering, not a hang.

## `List.append` into a growing list is O(n²)

The first plumbing split measured a step at **543 ms** with **77%** in the input
and target build: the builders appended into a growing list, and the tokeniser
scanned each digit with `at()` 64 times. Cons and reverse once, and walk the
list linearly: the same step is **131 ms** with **5%** in that build
(`results/PLUMBING.md`).

## One timestep per image, not per batch

Not an error but a silent 0.4-point loss: with one `t` per batch the ViT scores
0.359–0.441; with one `t` per image it scores 0.770–0.778. The first version of
that comparison was miswired (the "random" arm drew `t` per image) — the
correction sits above the table in ISSUE-016. Any ablation needs its wiring
read twice before its numbers are believed.
