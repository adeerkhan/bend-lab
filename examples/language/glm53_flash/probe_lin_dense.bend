# The dense linear core against lin_fast/lin_bwd at 128 positions, one head.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./lin_fast.bend as LF
import ./lin_bwd.bend as L
import ./lin_dense.bend as LD

def val(+i: Nat) -> F32:
  +k = Nat.mod(Nat.add(Nat.mul(i, 73n), 19n), 211n)
  (F32.from_nat(k) / 105.5 - 1.0 : F32)

def gen(+left: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      gen(p, (i + 1n : Nat), seed, Con{val(Nat.add(i, seed)), acc})

def ml(ml: TA.ML<128n, 8n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ones_row(+left: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ones_row(p, Con{1.0, acc})

def ones_sq(+n: Nat) -> TA.Mat<n, n>:
  TA.Mat.from_list(n, n, ones_row(Nat.mul(n, n), Nil{}))

def ml2(ml: TA.ML<128n, 128n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def maxdiff(xs: List<&2, F32>, ys: List<&2, F32>, acc: F32) -> F32:
  match xs ys:
    case Nil{} _:
      acc
    case _ Nil{}:
      acc
    case Con{h, t} Con{g, u}:
      maxdiff(t, u, F32.max(acc, F32.abs((h - g : F32))))

def diff(+tag: String, xs: List<&2, F32>, ys: List<&2, F32>) -> IO(Unit):
  IO.print(String.append(tag, String.append("_diff: ", F32.show(maxdiff(xs, ys, 0.0)))))

def run_bw(ob: L.BW, nb: LD.DBW<128n, 8n>) -> IO(Unit):
  match ob nb:
    case L.BW{odq, odk, odv} LD.DBW{ndq, ndk, ndv}:
      do IO<Unit>:
        diff("n128_dq", odq, ml(TA.Mat.to_list(128n, 8n, ndq)))
        diff("n128_dk", odk, ml(TA.Mat.to_list(128n, 8n, ndk)))
        diff("n128_dv", odv, ml(TA.Mat.to_list(128n, 8n, ndv)))
        return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    diff("n128_y", LF.lin_y_fast(128n, 8n, gen(1024n, 0n, 0n, Nil{}),
      gen(1024n, 0n, 1000n, Nil{}), gen(1024n, 0n, 2000n, Nil{})),
      ml(TA.Mat.to_list(128n, 8n, LD.lin_y(128n, 8n,
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 0n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 1000n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 2000n, Nil{}))))))
    run_bw(LF.lin_bwd_fast(128n, 8n, gen(1024n, 0n, 0n, Nil{}),
      gen(1024n, 0n, 1000n, Nil{}), gen(1024n, 0n, 2000n, Nil{}),
      gen(1024n, 0n, 3000n, Nil{})),
      LD.lin_bwd(128n, 8n, TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 3000n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 0n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 1000n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 2000n, Nil{}))))
    diff("n128_uncased", LF.lin_y_fast(128n, 8n, gen(1024n, 0n, 0n, Nil{}),
      gen(1024n, 0n, 1000n, Nil{}), gen(1024n, 0n, 2000n, Nil{})),
      ml(TA.Mat.to_list(128n, 8n, LD.lin_y_mask(128n, 8n,
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 0n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 1000n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 2000n, Nil{})),
        ones_sq(128n)))))
