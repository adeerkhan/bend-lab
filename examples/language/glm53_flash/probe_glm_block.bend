# The fourth hybrid block, full prefix versus one row at a time, on the
# trained weights.
#
# The full path is the reference block (`SpBlock`). The step path does the
# same for one row: RMSNorm, the qkv projection, the split, RoPE per head,
# the sparse softmax over the post-RoPE K/V cache, the output projection, and
# the SwiGLU MLP. This is the one-block gate; the four-block decode follows.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./attn_mh192.bend as A
import ./attn_c.bend as AC
import ./lin_bwd.bend as L
import ./model_h192.bend as M
import ./weights_h.bend as W

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

def block_fin(r: T.Fwd<M.BlockP(), TA.Mat<192n, 32n>, M.SpBlockS()>) -> List<&2, F32>:
  match r:
    case T.Fwd{pp, y, s}:
      A.mlD(TA.Mat.to_list(192n, 32n, y))

def y_block(p: M.BlockP(), x: TA.Mat<192n, 32n>) -> List<&2, F32>:
  block_fin(T.Layer.fwd(M.BlockP(), TA.Mat<192n, 32n>, TA.Mat<192n, 32n>,
    M.SpBlockS(), M.SpBlock(), p, x))

def ml1d(ml: TA.ML<1n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1w(ml: TA.ML<1n, 96n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml1h(ml: TA.ML<1n, 128n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def mm1_fin128(r: TA.MMulNT<1n, 32n, 128n>) -> List<&2, F32> & TA.Mat<128n, 32n>:
  match r:
    case TA.MMulNT{x, w, y}:
      Tuple{ml1h(TA.Mat.to_list(1n, 128n, y)), w}

def lin1_128(w: TA.Mat<128n, 32n>, +x: List<&2, F32>) -> List<&2, F32> & TA.Mat<128n, 32n>:
  mm1_fin128(TA.Mat.matmul_nt(1n, 32n, 128n, 0n, TA.Mat.from_list(1n, 32n, x), w))

def mm1_fin_down(r: TA.MMulNT<1n, 128n, 32n>) -> List<&2, F32> & TA.Mat<32n, 128n>:
  match r:
    case TA.MMulNT{x, w, y}:
      Tuple{ml1d(TA.Mat.to_list(1n, 32n, y)), w}

def lin1_down(w: TA.Mat<32n, 128n>, +x: List<&2, F32>) -> List<&2, F32> & TA.Mat<32n, 128n>:
  mm1_fin_down(TA.Mat.matmul_nt(1n, 128n, 32n, 0n, TA.Mat.from_list(1n, 128n, x), w))

def silu_mul(xs: List<&2, F32>, ys: List<&2, F32>,
    acc: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Nil{} _:
      List.reverse(&2, F32, acc)
    case _ Nil{}:
      List.reverse(&2, F32, acc)
    case Con{+g, t} Con{u, v}:
      +den = ((1.0 + F32.exp((0.0 - g : F32))) : F32)
      +sg = ((g / den : F32))
      silu_mul(t, v, Con{(sg * u : F32), acc})

# ---- the step path ----

type SpCache is Type:
  SpCache{q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>}

type SR is Type:
  SR{acc: List<&2, F32>}

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

type AttW is Type:
  AttW{ln: TA.Mat<1n, 32n>, qkv: TA.Mat<96n, 32n>, out: TA.Mat<32n, 32n>}

type MlpW is Type:
  MlpW{ln2: TA.Mat<1n, 32n>, gate: TA.Mat<128n, 32n>,
       up: TA.Mat<128n, 32n>, down: TA.Mat<32n, 128n>}

def step_c(lnb: TA.Mat<1n, 32n>, qkvb: TA.Mat<96n, 32n>,
    +q2: List<&2, F32>, +k2: List<&2, F32>,
    +v2: List<&2, F32>, +x: List<&2, F32>,
    r: List<&2, F32> & TA.Mat<32n, 32n>) -> AttW & (List<&2, F32> & SpCache):
  match r:
    case Tuple{o, outb}:
      Tuple{AttW{lnb, qkvb, outb},
        Tuple{L.add_lists(x, o), SpCache{q2, k2, v2}}}

def step_b(lnb: TA.Mat<1n, 32n>,
    outw: TA.Mat<32n, 32n>, +qc: List<&2, F32>, +kc: List<&2, F32>,
    +vc: List<&2, F32>, +x: List<&2, F32>, +t: Nat,
    r: List<&2, F32> & TA.Mat<96n, 32n>) -> AttW & (List<&2, F32> & SpCache):
  match r:
    case Tuple{+qkv, qkvb}:
      +qrow = rope32(take(qkv, 0n, 32n), t)
      +krow = rope32(take(qkv, 32n, 32n), t)
      +vrow = take(qkv, 64n, 32n)
      +q2 = List.append(&2, F32, qc, qrow)
      +k2 = List.append(&2, F32, kc, krow)
      +v2 = List.append(&2, F32, vc, vrow)
      step_c(lnb, qkvb, q2, k2, v2, x,
        lin1_32(outw, sp_y32(t, q2, k2, v2)))

def step_a(qkvw: TA.Mat<96n, 32n>,
    outw: TA.Mat<32n, 32n>, +qc: List<&2, F32>, +kc: List<&2, F32>,
    +vc: List<&2, F32>, +x: List<&2, F32>, +t: Nat,
    r: List<&2, F32> & TA.Mat<1n, 32n>) -> AttW & (List<&2, F32> & SpCache):
  match r:
    case Tuple{xn, lnb}:
      step_b(lnb, outw, qc, kc, vc, x, t, lin1_96(qkvw, xn))

def mlp_d(attW2: AttW, cache: SpCache, +x1: List<&2, F32>,
    ln2b: TA.Mat<1n, 32n>, gateb: TA.Mat<128n, 32n>,
    upb: TA.Mat<128n, 32n>,
    r: List<&2, F32> & TA.Mat<32n, 128n>) -> (AttW & MlpW) & (List<&2, F32> & SpCache):
  match r:
    case Tuple{d, downb}:
      Tuple{Tuple{attW2, MlpW{ln2b, gateb, upb, downb}},
        Tuple{L.add_lists(x1, d), cache}}

def mlp_c(attW2: AttW, cache: SpCache, +x1: List<&2, F32>,
    +g: List<&2, F32>, ln2b: TA.Mat<1n, 32n>,
    gateb: TA.Mat<128n, 32n>, down: TA.Mat<32n, 128n>,
    r: List<&2, F32> & TA.Mat<128n, 32n>) -> (AttW & MlpW) & (List<&2, F32> & SpCache):
  match r:
    case Tuple{u, upb2}:
      mlp_d(attW2, cache, x1, ln2b, gateb, upb2,
        lin1_down(down, silu_mul(g, u, Nil{})))

def mlp_b(attW2: AttW, cache: SpCache, +x1: List<&2, F32>,
    +xn2: List<&2, F32>, ln2b: TA.Mat<1n, 32n>,
    up: TA.Mat<128n, 32n>, down: TA.Mat<32n, 128n>,
    r: List<&2, F32> & TA.Mat<128n, 32n>) -> (AttW & MlpW) & (List<&2, F32> & SpCache):
  match r:
    case Tuple{g, gateb}:
      mlp_c(attW2, cache, x1, g, ln2b, gateb, down, lin1_128(up, xn2))

def mlp_a(attW2: AttW, cache: SpCache, +x1: List<&2, F32>,
    gate: TA.Mat<128n, 32n>, up: TA.Mat<128n, 32n>,
    down: TA.Mat<32n, 128n>,
    r: List<&2, F32> & TA.Mat<1n, 32n>) -> (AttW & MlpW) & (List<&2, F32> & SpCache):
  match r:
    case Tuple{+xn2, ln2b}:
      mlp_b(attW2, cache, x1, xn2, ln2b, up, down, lin1_128(gate, xn2))

def step_post(mlpW: MlpW,
    r: AttW & (List<&2, F32> & SpCache)) -> (AttW & MlpW) & (List<&2, F32> & SpCache):
  match mlpW r:
    case MlpW{ln2, gate, up, down} Tuple{attW2, Tuple{+x1, cache}}:
      mlp_a(attW2, cache, x1, gate, up, down, rms1(ln2, x1))

def step_row(attW: AttW, mlpW: MlpW, +x: List<&2, F32>, +t: Nat,
    c: SpCache) -> (AttW & MlpW) & (List<&2, F32> & SpCache):
  match attW c:
    case AttW{ln, qkvw, outw} SpCache{qc, kc, vc}:
      step_post(mlpW, step_a(qkvw, outw, qc, kc, vc, x, t, rms1(ln, x)))

def step_go(+left: Nat, +t: Nat, +x: List<&2, F32>, st: SR,
    r: (AttW & MlpW) & (List<&2, F32> & SpCache)) -> SR:
  match left st r:
    case 0n st0 _:
      st0
    case 1n SR{acc} Tuple{Tuple{_, _}, Tuple{y2, _}}:
      SR{List.append(&2, F32, acc, y2)}
    case 2n+p SR{acc} Tuple{Tuple{attW2, mlpW2}, Tuple{y2, c2}}:
      +acc2 = List.append(&2, F32, acc, y2)
      step_go((1n+p), (t + 1n : Nat), x, SR{acc2},
        step_row(attW2, mlpW2, take(x, Nat.mul((t + 1n : Nat), 32n), 32n),
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

def run(lns: TA.Mat<1n, 32n>, qkvs: TA.Mat<96n, 32n>,
    outs: TA.Mat<32n, 32n>, ln2s: TA.Mat<1n, 32n>,
    gates: TA.Mat<128n, 32n>, ups: TA.Mat<128n, 32n>,
    downs: TA.Mat<32n, 128n>, p3: M.BlockP()) -> IO(Unit):
  +x = ramp(Nat.mul(192n, 32n), 7n)
  +ystep = ys_of(step_go(192n, 0n, x, SR{Nil{}},
    step_row(AttW{lns, qkvs, outs}, MlpW{ln2s, gates, ups, downs},
      take(x, 0n, 32n), 0n, SpCache{Nil{}, Nil{}, Nil{}})))
  +yblock = y_block(p3, TA.Mat.from_list(192n, 32n, x))
  +d = maxdiff(ystep, yblock, 0.0)
  do IO<Unit>:
    Unit <- IO.print(String.append("block max|d| ", F32.show(d)))
    Unit <- IO.print(Bool.pick(String, F32.is_le(d, 1.0e-04),
      "GLM BLOCK STEP PASS", "GLM BLOCK STEP FAIL"))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    lns : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/ln1.bin", 0)
    qkvs : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/qkv.bin", 0)
    outs : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/out.bin", 0)
    ln2s : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/ln2.bin", 0)
    gates : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/gate.bin", 0)
    ups : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, "results/glm53-hybrid192-ep4/model-400/b3/up.bin", 0)
    downs : TA.Mat<32n, 128n> <- T.Mat.load(32n, 128n, "results/glm53-hybrid192-ep4/model-400/b3/down.bin", 0)
    p3 : W.BlockP() <- W.load_block("results/glm53-hybrid192-ep4/model-400/b3")
    run(lns, qkvs, outs, ln2s, gates, ups, downs, p3)
