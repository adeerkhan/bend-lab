# The fourth hybrid block's sparse attention, full prefix versus one row at a
# time, on the trained weights.
#
# The full path is the reference: RMSNorm, the qkv projection, the split, RoPE
# per head, `sp_y4`, and the output projection, all on 192 rows. The step path
# does the same for one row, carrying the post-RoPE K/V cache, and must produce
# every row's residual output. This is the one-block attention gate; the MLP
# step and the four-block decode follow.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./attn_mh192.bend as A
import ./attn_c.bend as AC
import ./lin_bwd.bend as L

def EPS() -> F32:
  0.000001

def val(+i: Nat, +seed: Nat) -> F32:
  +h = Nat.mod(Nat.add(Nat.mul(i, 37n), Nat.add(Nat.mul(seed, 101n), 11n)), 1009n)
  (((F32.from_nat(h) / 1009.0 : F32) - 0.5 : F32))

def ramp_go(+left: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+p:
      ramp_go(p, (i + 1n : Nat), seed, Con{val(i, seed), acc})

def ramp(+n: Nat, +seed: Nat) -> List<&2, F32>:
  List.reverse(&2, F32, ramp_go(n, 0n, seed, Nil{}))

def take(xs: List<&2, F32>, +off: Nat, +n: Nat) -> List<&2, F32>:
  List.take(&2, F32, List.drop(&2, F32, xs, off), n)

def join_one(+a: List<&2, F32>, +b: List<&2, F32>,
    +c: List<&2, F32>, +d: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, a, List.append(&2, F32, b,
    List.append(&2, F32, c, d)))

def rope32(+x: List<&2, F32>, +t: Nat) -> List<&2, F32>:
  join_one(A.rope_row(take(x, 0n, 8n), t), A.rope_row(take(x, 8n, 8n), t),
    A.rope_row(take(x, 16n, 8n), t), A.rope_row(take(x, 24n, 8n), t))

def sp_y32(+t: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>) -> List<&2, F32>:
  +q0 = A.gather_go(Nat.add(t, 1n), 0n, 0n, q, Nil{})
  +k0 = A.gather_go(Nat.add(t, 1n), 0n, 0n, k, Nil{})
  +v0 = A.gather_go(Nat.add(t, 1n), 0n, 0n, v, Nil{})
  +q1 = A.gather_go(Nat.add(t, 1n), 0n, 1n, q, Nil{})
  +k1 = A.gather_go(Nat.add(t, 1n), 0n, 1n, k, Nil{})
  +v1 = A.gather_go(Nat.add(t, 1n), 0n, 1n, v, Nil{})
  +q2 = A.gather_go(Nat.add(t, 1n), 0n, 2n, q, Nil{})
  +k2 = A.gather_go(Nat.add(t, 1n), 0n, 2n, k, Nil{})
  +v2 = A.gather_go(Nat.add(t, 1n), 0n, 2n, v, Nil{})
  +q3 = A.gather_go(Nat.add(t, 1n), 0n, 3n, q, Nil{})
  +k3 = A.gather_go(Nat.add(t, 1n), 0n, 3n, k, Nil{})
  +v3 = A.gather_go(Nat.add(t, 1n), 0n, 3n, v, Nil{})
  join_one(AC.srow_at(t, q0, k0, v0), AC.srow_at(t, q1, k1, v1),
    AC.srow_at(t, q2, k2, v2), AC.srow_at(t, q3, k3, v3))

# ---- the full-prefix attention residual ----

def rms_fin192(r: T.Fwd<TA.Mat<1n, 32n>, TA.Mat<192n, 32n>, TA.Mat<192n, 32n>>) -> List<&2, F32>:
  match r:
    case T.Fwd{pp, y, s}:
      A.mlD(TA.Mat.to_list(192n, 32n, y))

def rms192(ln: TA.Mat<1n, 32n>, +x: List<&2, F32>) -> List<&2, F32>:
  rms_fin192(T.rms_fwd(192n, 32n, EPS(), ln, TA.Mat.from_list(192n, 32n, x)))

def mm_fin96(r: TA.MMulNT<192n, 32n, 96n>) -> List<&2, F32>:
  match r:
    case TA.MMulNT{x, w, y}:
      A.mlW(TA.Mat.to_list(192n, 96n, y))

def lin192_96(w: TA.Mat<96n, 32n>, +x: List<&2, F32>) -> List<&2, F32>:
  mm_fin96(TA.Mat.matmul_nt(192n, 32n, 96n, 0n, TA.Mat.from_list(192n, 32n, x), w))

def mm_fin32(r: TA.MMulNT<192n, 32n, 32n>) -> List<&2, F32>:
  match r:
    case TA.MMulNT{x, w, y}:
      A.mlD(TA.Mat.to_list(192n, 32n, y))

def lin192_32(w: TA.Mat<32n, 32n>, +x: List<&2, F32>) -> List<&2, F32>:
  mm_fin32(TA.Mat.matmul_nt(192n, 32n, 32n, 0n, TA.Mat.from_list(192n, 32n, x), w))

def qkv_fin(y: A.QKV()) -> List<&2, F32> & (List<&2, F32> & List<&2, F32>):
  match y:
    case T.Two{qm, T.Two{km, vm}}:
      Tuple{A.mlD(TA.Mat.to_list(192n, 32n, qm)),
        Tuple{A.mlD(TA.Mat.to_list(192n, 32n, km)),
          A.mlD(TA.Mat.to_list(192n, 32n, vm))}}

def split_fin(r: T.Fwd<Unit, A.QKV(), Unit>) -> List<&2, F32> & (List<&2, F32> & List<&2, F32>):
  match r:
    case T.Fwd{pp, y, s}:
      qkv_fin(y)

def split192(c: TA.Mat<192n, 96n>) -> List<&2, F32> & (List<&2, F32> & List<&2, F32>):
  split_fin(A.split_fwd(c))

def full_b(outw: TA.Mat<32n, 32n>, +x: List<&2, F32>,
    +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>) -> List<&2, F32>:
  +y = A.sp_y4(q, k, v)
  +o = lin192_32(outw, y)
  L.add_lists(x, o)

def full_a(outw: TA.Mat<32n, 32n>, +x: List<&2, F32>,
    r: List<&2, F32> & (List<&2, F32> & List<&2, F32>)) -> List<&2, F32>:
  match r:
    case Tuple{q, Tuple{k, v}}:
      full_b(outw, x, q, k, v)

def full_attn(ln: TA.Mat<1n, 32n>, qkvw: TA.Mat<96n, 32n>,
    outw: TA.Mat<32n, 32n>, +x: List<&2, F32>) -> List<&2, F32>:
  +xn = rms192(ln, x)
  +qkv = lin192_96(qkvw, xn)
  full_a(outw, x, split192(TA.Mat.from_list(192n, 96n, qkv)))

# ---- the step path ----

type SpCache is Type:
  SpCache{q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>}

type SR is Type:
  SR{acc: List<&2, F32>}

def ml1d(ml: TA.ML<1n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1w(ml: TA.ML<1n, 96n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def rms_fin1(r: T.Fwd<TA.Mat<1n, 32n>, TA.Mat<1n, 32n>, TA.Mat<1n, 32n>>) -> List<&2, F32> & TA.Mat<1n, 32n>:
  match r:
    case T.Fwd{pp, y, s}:
      Tuple{ml1d(TA.Mat.to_list(1n, 32n, y)), pp}

def rms1(ln: TA.Mat<1n, 32n>, +x: List<&2, F32>) -> List<&2, F32> & TA.Mat<1n, 32n>:
  rms_fin1(T.rms_fwd(1n, 32n, EPS(), ln, TA.Mat.from_list(1n, 32n, x)))

def mm1_fin96(r: TA.MMulNT<1n, 32n, 96n>) -> List<&2, F32> & TA.Mat<96n, 32n>:
  match r:
    case TA.MMulNT{x, w, y}:
      Tuple{ml1w(TA.Mat.to_list(1n, 96n, y)), w}

def lin1_96(w: TA.Mat<96n, 32n>, +x: List<&2, F32>) -> List<&2, F32> & TA.Mat<96n, 32n>:
  mm1_fin96(TA.Mat.matmul_nt(1n, 32n, 96n, 0n, TA.Mat.from_list(1n, 32n, x), w))

def mm1_fin32(r: TA.MMulNT<1n, 32n, 32n>) -> List<&2, F32> & TA.Mat<32n, 32n>:
  match r:
    case TA.MMulNT{x, w, y}:
      Tuple{ml1d(TA.Mat.to_list(1n, 32n, y)), w}

def lin1_32(w: TA.Mat<32n, 32n>, +x: List<&2, F32>) -> List<&2, F32> & TA.Mat<32n, 32n>:
  mm1_fin32(TA.Mat.matmul_nt(1n, 32n, 32n, 0n, TA.Mat.from_list(1n, 32n, x), w))

def step_c(lnb: TA.Mat<1n, 32n>, qkvb: TA.Mat<96n, 32n>,
    +q2: List<&2, F32>, +k2: List<&2, F32>,
    +v2: List<&2, F32>, +x: List<&2, F32>, +t: Nat,
    r: List<&2, F32> & TA.Mat<32n, 32n>) -> (TA.Mat<1n, 32n> & (TA.Mat<96n, 32n> & TA.Mat<32n, 32n>)) & (List<&2, F32> & SpCache):
  match r:
    case Tuple{o, outb}:
      Tuple{Tuple{lnb, Tuple{qkvb, outb}},
        Tuple{L.add_lists(x, o), SpCache{q2, k2, v2}}}

def step_b(lnb: TA.Mat<1n, 32n>,
    outw: TA.Mat<32n, 32n>, +qc: List<&2, F32>, +kc: List<&2, F32>,
    +vc: List<&2, F32>, +x: List<&2, F32>, +t: Nat,
    r: List<&2, F32> & TA.Mat<96n, 32n>) -> (TA.Mat<1n, 32n> & (TA.Mat<96n, 32n> & TA.Mat<32n, 32n>)) & (List<&2, F32> & SpCache):
  match r:
    case Tuple{+qkv, qkvb}:
      +qrow = rope32(take(qkv, 0n, 32n), t)
      +krow = rope32(take(qkv, 32n, 32n), t)
      +vrow = take(qkv, 64n, 32n)
      +q2 = List.append(&2, F32, qc, qrow)
      +k2 = List.append(&2, F32, kc, krow)
      +v2 = List.append(&2, F32, vc, vrow)
      step_c(lnb, qkvb, q2, k2, v2, x, t,
        lin1_32(outw, sp_y32(t, q2, k2, v2)))

def step_a(qkvw: TA.Mat<96n, 32n>,
    outw: TA.Mat<32n, 32n>, +qc: List<&2, F32>, +kc: List<&2, F32>,
    +vc: List<&2, F32>, +x: List<&2, F32>, +t: Nat,
    r: List<&2, F32> & TA.Mat<1n, 32n>) -> (TA.Mat<1n, 32n> & (TA.Mat<96n, 32n> & TA.Mat<32n, 32n>)) & (List<&2, F32> & SpCache):
  match r:
    case Tuple{xn, lnb}:
      step_b(lnb, outw, qc, kc, vc, x, t, lin1_96(qkvw, xn))

def step_row(w: TA.Mat<1n, 32n> & (TA.Mat<96n, 32n> & TA.Mat<32n, 32n>),
    +x: List<&2, F32>, +t: Nat, c: SpCache) -> (TA.Mat<1n, 32n> & (TA.Mat<96n, 32n> & TA.Mat<32n, 32n>)) & (List<&2, F32> & SpCache):
  match w c:
    case Tuple{ln, Tuple{qkvw, outw}} SpCache{qc, kc, vc}:
      step_a(qkvw, outw, qc, kc, vc, x, t, rms1(ln, x))

def step_go(+left: Nat, +t: Nat, +x: List<&2, F32>, st: SR,
    r: (TA.Mat<1n, 32n> & (TA.Mat<96n, 32n> & TA.Mat<32n, 32n>)) & (List<&2, F32> & SpCache)) -> SR:
  match left st r:
    case 0n st0 _:
      st0
    case 1n SR{acc} Tuple{w2, Tuple{y2, c2}}:
      SR{List.append(&2, F32, acc, y2)}
    case 2n+p SR{acc} Tuple{w2, Tuple{y2, c2}}:
      +acc2 = List.append(&2, F32, acc, y2)
      step_go((1n+p), (t + 1n : Nat), x, SR{acc2},
        step_row(w2, take(x, Nat.mul((t + 1n : Nat), 32n), 32n),
          (t + 1n : Nat), c2))
    case _ st1 _:
      st1

def ys_of(st: SR) -> List<&2, F32>:
  match st:
    case SR{acc}:
      acc

def maxdiff(xs: List<&2, F32>, ys: List<&2, F32>, acc: F32) -> F32:
  match xs ys:
    case Nil{} _:
      acc
    case _ Nil{}:
      acc
    case Con{h, t} Con{g, u}:
      maxdiff(t, u, F32.max(acc, F32.abs((h - g : F32))))

def run(lnf: TA.Mat<1n, 32n>, qkvf: TA.Mat<96n, 32n>,
    outf: TA.Mat<32n, 32n>, lns: TA.Mat<1n, 32n>,
    qkvs: TA.Mat<96n, 32n>, outs: TA.Mat<32n, 32n>) -> IO(Unit):
  +x = ramp(Nat.mul(192n, 32n), 7n)
  +yfull = full_attn(lnf, qkvf, outf, x)
  +ystep = ys_of(step_go(192n, 0n, x, SR{Nil{}},
    step_row(Tuple{lns, Tuple{qkvs, outs}}, take(x, 0n, 32n), 0n,
      SpCache{Nil{}, Nil{}, Nil{}})))
  +d = maxdiff(ystep, yfull, 0.0)
  do IO<Unit>:
    Unit <- IO.print(String.append("rows 192  attention max|d| ", F32.show(d)))
    Unit <- IO.print(Bool.pick(String, F32.is_le(d, 1.0e-04),
      "GLM BLOCK STEP PASS", "GLM BLOCK STEP FAIL"))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    lnf : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/ln1.bin", 0)
    qkvf : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/qkv.bin", 0)
    outf : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/out.bin", 0)
    lns : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/ln1.bin", 0)
    qkvs : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/qkv.bin", 0)
    outs : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/out.bin", 0)
    run(lnf, qkvf, outf, lns, qkvs, outs)
