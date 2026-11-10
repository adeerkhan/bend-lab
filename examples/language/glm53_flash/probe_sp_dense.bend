# The dense sparse core against the gathered list core: y, dq, dk and dv at
# 32 and 128 positions, on the same deterministic inputs.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./attn_c.bend as AC
import ./lin_bwd.bend as L
import ./sp_dense.bend as SD

def val(+i: Nat) -> F32:
  +k = Nat.mod(Nat.add(Nat.mul(i, 73n), 19n), 211n)
  (F32.from_nat(k) / 105.5 - 1.0 : F32)

def gen(+left: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      gen(p, (i + 1n : Nat), seed, Con{val(Nat.add(i, seed)), acc})

def ml32(ml: TA.ML<32n, 8n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml128(ml: TA.ML<128n, 8n>) -> List<&2, F32>:
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

def diff32(ob: L.BW, nb: SD.DBW<32n, 8n>) -> IO(Unit):
  match ob nb:
    case L.BW{odq, odk, odv} SD.DBW{ndq, ndk, ndv}:
      do IO<Unit>:
        diff("n32_dq", odq, ml32(TA.Mat.to_list(32n, 8n, ndq)))
        diff("n32_dk", odk, ml32(TA.Mat.to_list(32n, 8n, ndk)))
        diff("n32_dv", odv, ml32(TA.Mat.to_list(32n, 8n, ndv)))
        return Unit{}

def diff128(ob: L.BW, nb: SD.DBW<128n, 8n>) -> IO(Unit):
  match ob nb:
    case L.BW{odq, odk, odv} SD.DBW{ndq, ndk, ndv}:
      do IO<Unit>:
        diff("n128_dq", odq, ml128(TA.Mat.to_list(128n, 8n, ndq)))
        diff("n128_dk", odk, ml128(TA.Mat.to_list(128n, 8n, ndk)))
        diff("n128_dv", odv, ml128(TA.Mat.to_list(128n, 8n, ndv)))
        return Unit{}

def run32() -> IO(Unit):
  do IO<Unit>:
    diff("n32_y", AC.sp_y_go(32n, 0n, gen(256n, 0n, 0n, Nil{}), gen(256n, 0n, 1000n, Nil{}),
      gen(256n, 0n, 2000n, Nil{}), Nil{}),
      ml32(TA.Mat.to_list(32n, 8n, SD.y_of(32n, 8n,
        TA.Mat.from_list(32n, 8n, gen(256n, 0n, 0n, Nil{})),
        TA.Mat.from_list(32n, 8n, gen(256n, 0n, 1000n, Nil{})),
        TA.Mat.from_list(32n, 8n, gen(256n, 0n, 2000n, Nil{})), SD.mask_mat(32n)))))
    diff32(AC.sp_t_go(32n, 0n, gen(256n, 0n, 3000n, Nil{}), gen(256n, 0n, 0n, Nil{}),
      gen(256n, 0n, 1000n, Nil{}), gen(256n, 0n, 2000n, Nil{}), AC.zeros(256n, Nil{}),
      AC.zeros(256n, Nil{}), AC.zeros(256n, Nil{})),
      SD.bwd(32n, 8n, TA.Mat.from_list(32n, 8n, gen(256n, 0n, 3000n, Nil{})),
        TA.Mat.from_list(32n, 8n, gen(256n, 0n, 0n, Nil{})),
        TA.Mat.from_list(32n, 8n, gen(256n, 0n, 1000n, Nil{})),
        TA.Mat.from_list(32n, 8n, gen(256n, 0n, 2000n, Nil{})), SD.mask_mat(32n)))

def run128() -> IO(Unit):
  do IO<Unit>:
    diff("n128_y", AC.sp_y_go(128n, 0n, gen(1024n, 0n, 0n, Nil{}), gen(1024n, 0n, 1000n, Nil{}),
      gen(1024n, 0n, 2000n, Nil{}), Nil{}),
      ml128(TA.Mat.to_list(128n, 8n, SD.y_of(128n, 8n,
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 0n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 1000n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 2000n, Nil{})), SD.mask_mat(128n)))))
    diff128(AC.sp_t_go(128n, 0n, gen(1024n, 0n, 3000n, Nil{}), gen(1024n, 0n, 0n, Nil{}),
      gen(1024n, 0n, 1000n, Nil{}), gen(1024n, 0n, 2000n, Nil{}), AC.zeros(1024n, Nil{}),
      AC.zeros(1024n, Nil{}), AC.zeros(1024n, Nil{})),
      SD.bwd(128n, 8n, TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 3000n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 0n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 1000n, Nil{})),
        TA.Mat.from_list(128n, 8n, gen(1024n, 0n, 2000n, Nil{})), SD.mask_mat(128n)))

def main() -> IO(Unit):
  do IO<Unit>:
    run32()
    run128()
