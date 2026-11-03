# The machine's practical ceilings, measured: a write, a read and a copy of one
# array, through the same `Array.get`/`Array.set` the pipeline pays for.
#
#   bendenv run -- bend bench/peaks/main.bend
#
# These are not DRAM's raw peaks. Bend's `Array` is a tree with a log-depth
# descent, so the figure below is the rate a stage can actually reach when it
# walks an array cell by cell; a kernel that streams contiguously would be
# faster, and the report says so rather than pretend otherwise.
#
# Every phase is repeated `passes()` times inside ONE timed region: one pass is
# 2-5 ms on this machine and `IO.now()` counts whole milliseconds, so a single
# pass would be a 20% number. The reader divides by `passes` and says so.
#
# The read phase accumulates, so it measures a reducing walk (an F32 add chain
# plus the tree descent), not a bandwidth; `bench_peaks.py` publishes it as
# `reduce_gbs` and charges a read-only stage at the copy's per-byte rate.
#
# Every phase prints its value after the timer, so the compiler cannot erase
# the work as unused.
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def n() -> Nat:
  4194304n

def passes() -> Nat:
  200n

def cap() -> Nat:
  TA.cap_depth(n())

def bytes() -> Nat:
  Nat.mul(n(), 4n)

type RA is Type:
  RA{a: Array<F32>, s: F32}

type CA is Type:
  CA{src: Array<F32>, dst: Array<F32>}

def write_go(k: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: write_go(p, (i + 1 : U32), Array.set(F32, a, i, 1.0))

# The cell is a parameter, so the tuple Array.get hands back is matched where it
# arrives rather than after a computed call. One fetch past the end is masked by
# Array.get and never consumed.
# hands the array back beside the sum: a read pass consumes the array through
# Array.get, so a repeat loop needs it returned rather than duplicated
def read_go(k: Nat, +i: U32, r: Array<F32> & F32, +acc: F32) -> RA:
  match k r:
    case 0n Tuple{a, _}: RA{a, acc}
    case 1n+p Tuple{a, +x}: read_go(p, (i + 1 : U32), Array.get(F32, a, (i + 1 : U32)), (acc + x : F32))

def copy_go(k: Nat, +i: U32, r: Array<F32> & F32, dst: Array<F32>) -> CA:
  match k r:
    case 0n Tuple{a, _}: CA{a, dst}
    case 1n+p Tuple{src, +x}: copy_go(p, (i + 1 : U32), Array.get(F32, src, (i + 1 : U32)), Array.set(F32, dst, i, x))

def head_go(r: Array<F32> & F32) -> F32:
  match r:
    case Tuple{_, +x}: x

def head(a: Array<F32>) -> F32:
  head_go(Array.get(F32, a, (0 : U32)))

# one pass, the array handed back beside its sum: an Array is linear, so a
# repeat loop threads it rather than duplicating it
def rd_one(a: Array<F32>) -> RA:
  read_go(n(), 0, Array.get(F32, a, (0 : U32)), 0.0)

def write_r(+left: Nat, a: Array<F32>) -> Array<F32>:
  match left:
    case 0n: a
    case 1n+p: write_r(p, write_go(n(), (0 : U32), a))

def read_r(+left: Nat, ra: RA, +acc: F32) -> F32:
  match left ra:
    case 0n RA{a, +s}: acc
    case 1n+p RA{a, +s}: read_r(p, rd_one(a), (acc + s : F32))

# a copy alternates its two arrays, so a pass allocates nothing: src is read,
# dst is written, and the next pass swaps them
def ca_head(ca: CA) -> F32:
  match ca:
    case CA{src, dst}: head(dst)

def swap(ca: CA) -> CA:
  match ca:
    case CA{a, b}: CA{b, a}

def copy_r(+left: Nat, ca: CA) -> CA:
  match left ca:
    case 0n c: c
    case 1n+p CA{src, dst}:
      copy_r(p, swap(copy_go(n(), 0, Array.get(F32, src, (0 : U32)), dst)))

def show_ms(+name: String, +ms: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(ms))))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("n: ", Nat.show(n())))
    Unit <- IO.print(String.append("bytes: ", Nat.show(bytes())))
    Unit <- IO.print(String.append("passes: ", Nat.show(passes())))

    a0 : Array<F32> <- IO.pure(Array<F32>, Array.new(F32, cap(), 0.0))
    t0 : Nat <- IO.now()
    a1 : Array<F32> <- IO.pure(Array<F32>, write_r(passes(), a0))
    t1 : Nat <- IO.now()
    Unit <- show_ms("write", Nat.sub(t1, t0))
    Unit <- IO.print(String.append("write_head: ", F32.show(1.0)))

    t2 : Nat <- IO.now()
    s : F32 <- IO.pure(F32, read_r(passes(), RA{a1, 0.0}, 0.0))
    t3 : Nat <- IO.now()
    Unit <- show_ms("read", Nat.sub(t3, t2))
    Unit <- IO.print(String.append("read_head: ", F32.show(s)))

    c0 : Array<F32> <- IO.pure(Array<F32>, Array.new(F32, cap(), 0.0))
    c1 : Array<F32> <- IO.pure(Array<F32>, write_r(passes(), c0))
    b0 : Array<F32> <- IO.pure(Array<F32>, Array.new(F32, cap(), 0.0))
    t4 : Nat <- IO.now()
    ca : CA <- IO.pure(CA, copy_r(passes(), CA{c1, b0}))
    t5 : Nat <- IO.now()
    Unit <- show_ms("copy", Nat.sub(t5, t4))
    IO.print(String.append("copy_head: ", F32.show(ca_head(ca))))
    return Unit{}
