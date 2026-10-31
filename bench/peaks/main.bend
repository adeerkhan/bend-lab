# The machine's practical ceilings, measured: a write, a read and a copy of one
# array, through the same `Array.get`/`Array.set` the pipeline pays for.
#
#   bendenv run -- bend bench/peaks/main.bend
#
# These are not DRAM's raw peaks. Bend's `Array` is a tree with a log-depth
# descent, so the figure below is the rate a stage can actually reach when it
# walks an array cell by cell; a kernel that streams contiguously would be
# faster, and the report says so rather than pretend otherwise. The compute
# ceiling comes from `bench/matmul` (`results/bench-matmul.json`), and launch
# latency is not measured here: it needs a bang, and a plain program has none.
#
# Every phase prints its value after the timer, so the compiler cannot erase
# the work as unused.
import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA

def n() -> Nat:
  4194304n

def cap() -> Nat:
  TA.cap_depth(n())

def bytes() -> Nat:
  Nat.mul(n(), 4n)

def write_go(k: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: write_go(p, (i + 1 : U32), Array.set(F32, a, i, 1.0))

# The cell is a parameter, so the tuple Array.get hands back is matched where it
# arrives rather than after a computed call. One fetch past the end is masked by
# Array.get and never consumed.
def read_go(k: Nat, +i: U32, r: Array<F32> & F32, +acc: F32) -> F32:
  match k r:
    case 0n _: acc
    case 1n+p Tuple{a, +x}: read_go(p, (i + 1 : U32), Array.get(F32, a, (i + 1 : U32)), (acc + x : F32))

def copy_go(k: Nat, +i: U32, r: Array<F32> & F32, dst: Array<F32>) -> Array<F32>:
  match k r:
    case 0n _: dst
    case 1n+p Tuple{src, +x}: copy_go(p, (i + 1 : U32), Array.get(F32, src, (i + 1 : U32)), Array.set(F32, dst, i, x))

def head_go(r: Array<F32> & F32) -> F32:
  match r:
    case Tuple{_, +x}: x

def head(a: Array<F32>) -> F32:
  head_go(Array.get(F32, a, (0 : U32)))

def show_ms(+name: String, +ms: Nat) -> IO(Unit):
  IO.print(String.append(name, String.append("_ms: ", Nat.show(ms))))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("n: ", Nat.show(n())))
    Unit <- IO.print(String.append("bytes: ", Nat.show(bytes())))

    a0 : Array<F32> <- IO.pure(Array<F32>, Array.new(F32, cap(), 0.0))
    t0 : Nat <- IO.now()
    a1 : Array<F32> <- IO.pure(Array<F32>, write_go(n(), (0 : U32), a0))
    t1 : Nat <- IO.now()
    Unit <- show_ms("write", Nat.sub(t1, t0))
    Unit <- IO.print(String.append("write_head: ", F32.show(1.0)))

    t2 : Nat <- IO.now()
    s : F32 <- IO.pure(F32, read_go(n(), 0, Array.get(F32, a1, (0 : U32)), 0.0))
    t3 : Nat <- IO.now()
    Unit <- show_ms("read", Nat.sub(t3, t2))
    Unit <- IO.print(String.append("read_head: ", F32.show(s)))

    c0 : Array<F32> <- IO.pure(Array<F32>, Array.new(F32, cap(), 0.0))
    c1 : Array<F32> <- IO.pure(Array<F32>, write_go(n(), (0 : U32), c0))
    b0 : Array<F32> <- IO.pure(Array<F32>, Array.new(F32, cap(), 0.0))
    t4 : Nat <- IO.now()
    b1 : Array<F32> <- IO.pure(Array<F32>, copy_go(n(), 0, Array.get(F32, c1, (0 : U32)), b0))
    t5 : Nat <- IO.now()
    Unit <- show_ms("copy", Nat.sub(t5, t4))
    Unit <- IO.print(String.append("copy_head: ", F32.show(head(b1))))
    return Unit{}
