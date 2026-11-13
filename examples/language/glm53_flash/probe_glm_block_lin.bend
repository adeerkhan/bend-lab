# The third hybrid block (linear attention) one row at a time, on the trained
# weights, against the full `LinBlock`.
#
# The attention state is the linear prefix state per head: K = sum elu(k_t) and
# KV = sum elu(k_t) v_t^T. The SwiGLU MLP is the same step chain as the sparse
# block. This is the linear half of the four-block decode gate.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./model_h192.bend as M
import ./weights_h.bend as W
import ./attn_mh192.bend as A
import ./lin_bwd.bend as L
import ./probe_glm_block.bend as P

def take(xs: List<&2, F32>, +off: Nat, +n: Nat) -> List<&2, F32>:
  P.take(xs, off, n)

type LinCache is Type:
  LinCache{k0: List<&2, F32>, kv0: List<&2, F32>,
           k1: List<&2, F32>, kv1: List<&2, F32>,
           k2: List<&2, F32>, kv2: List<&2, F32>,
           k3: List<&2, F32>, kv3: List<&2, F32>}

def lin_head(+q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, K: List<&2, F32>,
    KV: List<&2, F32>) -> List<&2, F32> & (List<&2, F32> & List<&2, F32>):
  +qe = L.elu_row(q)
  +ke = L.elu_row(k)
  +K2 = L.add_lists(K, ke)
  +KV2 = L.add_outer(KV, ke, v, 8n)
  +c = L.clamp_den(L.dot(qe, K2))
  +num = L.matvec_t(KV2, qe, 8n)
  Tuple{K2, Tuple{KV2, L.scale_list(num, (1.0 / c : F32))}}

def lin_h3(+q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, K0b: List<&2, F32>, KV0b: List<&2, F32>,
    K1b: List<&2, F32>, KV1b: List<&2, F32>,
    K2b: List<&2, F32>, KV2b: List<&2, F32>,
    +y0: List<&2, F32>, +y1: List<&2, F32>, +y2: List<&2, F32>,
    r: List<&2, F32> & (List<&2, F32> & List<&2, F32>)) -> LinCache & List<&2, F32>:
  match r:
    case Tuple{K3b, Tuple{KV3b, y3}}:
      Tuple{LinCache{K0b, KV0b, K1b, KV1b, K2b, KV2b, K3b, KV3b},
        List.append(&2, F32, y0, List.append(&2, F32, y1,
          List.append(&2, F32, y2, y3)))}

def lin_h2(+q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, K0b: List<&2, F32>, KV0b: List<&2, F32>,
    K1b: List<&2, F32>, KV1b: List<&2, F32>, K3: List<&2, F32>,
    KV3: List<&2, F32>, +y0: List<&2, F32>, +y1: List<&2, F32>,
    r: List<&2, F32> & (List<&2, F32> & List<&2, F32>)) -> LinCache & List<&2, F32>:
  match r:
    case Tuple{K2b, Tuple{KV2b, y2}}:
      lin_h3(q, k, v, K0b, KV0b, K1b, KV1b, K2b, KV2b, y0, y1, y2,
        lin_head(take(q, 24n, 8n), take(k, 24n, 8n), take(v, 24n, 8n), K3, KV3))

def lin_h1(+q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, K0b: List<&2, F32>, KV0b: List<&2, F32>,
    K2: List<&2, F32>, KV2: List<&2, F32>, K3: List<&2, F32>,
    KV3: List<&2, F32>, +y0: List<&2, F32>,
    r: List<&2, F32> & (List<&2, F32> & List<&2, F32>)) -> LinCache & List<&2, F32>:
  match r:
    case Tuple{K1b, Tuple{KV1b, y1}}:
      lin_h2(q, k, v, K0b, KV0b, K1b, KV1b, K3, KV3, y0, y1,
        lin_head(take(q, 16n, 8n), take(k, 16n, 8n), take(v, 16n, 8n), K2, KV2))

def lin_h0(+q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, K1: List<&2, F32>, KV1: List<&2, F32>,
    K2: List<&2, F32>, KV2: List<&2, F32>, K3: List<&2, F32>,
    KV3: List<&2, F32>,
    r: List<&2, F32> & (List<&2, F32> & List<&2, F32>)) -> LinCache & List<&2, F32>:
  match r:
    case Tuple{K0b, Tuple{KV0b, y0}}:
      lin_h1(q, k, v, K0b, KV0b, K2, KV2, K3, KV3, y0,
        lin_head(take(q, 8n, 8n), take(k, 8n, 8n), take(v, 8n, 8n), K1, KV1))

def lin_y32(+t: Nat, +q: List<&2, F32>, +k: List<&2, F32>,
    +v: List<&2, F32>, c: LinCache) -> LinCache & List<&2, F32>:
  match c:
    case LinCache{K0, KV0, K1, KV1, K2, KV2, K3, KV3}:
      lin_h0(q, k, v, K1, KV1, K2, KV2, K3, KV3,
        lin_head(take(q, 0n, 8n), take(k, 0n, 8n), take(v, 0n, 8n), K0, KV0))

def mlp_d(attW2: P.AttW, cache: LinCache, +x1: List<&2, F32>,
    ln2b: TA.Mat<1n, 32n>, gateb: TA.Mat<128n, 32n>,
    upb: TA.Mat<128n, 32n>,
    r: List<&2, F32> & TA.Mat<32n, 128n>) -> (P.AttW & P.MlpW) & (List<&2, F32> & LinCache):
  match r:
    case Tuple{d, downb}:
      Tuple{Tuple{attW2, P.MlpW{ln2b, gateb, upb, downb}},
        Tuple{L.add_lists(x1, d), cache}}

def mlp_c(attW2: P.AttW, cache: LinCache, +x1: List<&2, F32>,
    +g: List<&2, F32>, ln2b: TA.Mat<1n, 32n>,
    gateb: TA.Mat<128n, 32n>, down: TA.Mat<32n, 128n>,
    r: List<&2, F32> & TA.Mat<128n, 32n>) -> (P.AttW & P.MlpW) & (List<&2, F32> & LinCache):
  match r:
    case Tuple{u, upb2}:
      mlp_d(attW2, cache, x1, ln2b, gateb, upb2,
        P.lin1_down(down, P.silu_mul(g, u, Nil{})))

def mlp_b(attW2: P.AttW, cache: LinCache, +x1: List<&2, F32>,
    +xn2: List<&2, F32>, ln2b: TA.Mat<1n, 32n>,
    up: TA.Mat<128n, 32n>, down: TA.Mat<32n, 128n>,
    r: List<&2, F32> & TA.Mat<128n, 32n>) -> (P.AttW & P.MlpW) & (List<&2, F32> & LinCache):
  match r:
    case Tuple{g, gateb}:
      mlp_c(attW2, cache, x1, g, ln2b, gateb, down, P.lin1_128(up, xn2))

def mlp_a(attW2: P.AttW, cache: LinCache, +x1: List<&2, F32>,
    gate: TA.Mat<128n, 32n>, up: TA.Mat<128n, 32n>,
    down: TA.Mat<32n, 128n>,
    r: List<&2, F32> & TA.Mat<1n, 32n>) -> (P.AttW & P.MlpW) & (List<&2, F32> & LinCache):
  match r:
    case Tuple{+xn2, ln2b}:
      mlp_b(attW2, cache, x1, xn2, ln2b, up, down, P.lin1_128(gate, xn2))

def step_post(mlpW: P.MlpW,
    r: P.AttW & (List<&2, F32> & LinCache)) -> (P.AttW & P.MlpW) & (List<&2, F32> & LinCache):
  match mlpW r:
    case P.MlpW{ln2, gate, up, down} Tuple{attW2, Tuple{+x1, cache}}:
      mlp_a(attW2, cache, x1, gate, up, down, P.rms1(ln2, x1))

def step_out(lnb: TA.Mat<1n, 32n>, qkvb: TA.Mat<96n, 32n>,
    c2: LinCache, +x: List<&2, F32>,
    r: List<&2, F32> & TA.Mat<32n, 32n>) -> P.AttW & (List<&2, F32> & LinCache):
  match r:
    case Tuple{o, outb}:
      Tuple{P.AttW{lnb, qkvb, outb},
        Tuple{L.add_lists(x, o), c2}}

def step_after_y(lnb: TA.Mat<1n, 32n>,
    qkvb: TA.Mat<96n, 32n>, outw: TA.Mat<32n, 32n>,
    +x: List<&2, F32>, r: LinCache & List<&2, F32>) -> P.AttW & (List<&2, F32> & LinCache):
  match r:
    case Tuple{c2, y32}:
      step_out(lnb, qkvb, c2, x, P.lin1_32(outw, y32))

def step_after_qkv(lnb: TA.Mat<1n, 32n>, outw: TA.Mat<32n, 32n>,
    +x: List<&2, F32>, +t: Nat, c: LinCache,
    r: List<&2, F32> & TA.Mat<96n, 32n>) -> P.AttW & (List<&2, F32> & LinCache):
  match r:
    case Tuple{+qkv, qkvb}:
      +qrow = P.rope32(take(qkv, 0n, 32n), t)
      +krow = P.rope32(take(qkv, 32n, 32n), t)
      +vrow = take(qkv, 64n, 32n)
      step_after_y(lnb, qkvb, outw, x, lin_y32(t, qrow, krow, vrow, c))

def step_after_rms(qkvw: TA.Mat<96n, 32n>, outw: TA.Mat<32n, 32n>,
    +x: List<&2, F32>, +t: Nat, c: LinCache,
    r: List<&2, F32> & TA.Mat<1n, 32n>) -> P.AttW & (List<&2, F32> & LinCache):
  match r:
    case Tuple{xn, lnb}:
      step_after_qkv(lnb, outw, x, t, c, P.lin1_96(qkvw, xn))

def step_row(attW: P.AttW, mlpW: P.MlpW, +x: List<&2, F32>, +t: Nat,
    c: LinCache) -> (P.AttW & P.MlpW) & (List<&2, F32> & LinCache):
  match attW c:
    case P.AttW{ln, qkvw, outw} LinCache{k0, kv0, k1, kv1, k2, kv2, k3, kv3}:
      step_post(mlpW, step_after_rms(qkvw, outw, x, t,
        LinCache{k0, kv0, k1, kv1, k2, kv2, k3, kv3}, P.rms1(ln, x)))

def block_fin(r: T.Fwd<M.BlockP(), TA.Mat<192n, 32n>, M.LinBlockS()>) -> List<&2, F32>:
  match r:
    case T.Fwd{pp, y, s}:
      A.mlD(TA.Mat.to_list(192n, 32n, y))

type LR is Type:
  LR{acc: List<&2, F32>}

def step_go(+left: Nat, +t: Nat, +x: List<&2, F32>, st: LR,
    r: (P.AttW & P.MlpW) & (List<&2, F32> & LinCache)) -> LR:
  match left st r:
    case 0n st0 _:
      st0
    case 1n LR{acc} Tuple{Tuple{_, _}, Tuple{y2, _}}:
      LR{List.append(&2, F32, acc, y2)}
    case 2n+p LR{acc} Tuple{Tuple{attW2, mlpW2}, Tuple{y2, c2}}:
      +acc2 = List.append(&2, F32, acc, y2)
      step_go((1n+p), (t + 1n : Nat), x, LR{acc2},
        step_row(attW2, mlpW2, take(x, Nat.mul((t + 1n : Nat), 32n), 32n),
          (t + 1n : Nat), c2))
    case _ st1 _:
      st1

def ys_of(st: LR) -> List<&2, F32>:
  match st:
    case LR{acc}:
      acc

def run(lns: TA.Mat<1n, 32n>, qkvs: TA.Mat<96n, 32n>,
    outs: TA.Mat<32n, 32n>, ln2s: TA.Mat<1n, 32n>,
    gates: TA.Mat<128n, 32n>, ups: TA.Mat<128n, 32n>,
    downs: TA.Mat<32n, 128n>, p2: M.BlockP()) -> IO(Unit):
  +x = P.ramp(Nat.mul(192n, 32n), 7n)
  +ystep = ys_of(step_go(192n, 0n, x, LR{Nil{}},
    step_row(P.AttW{lns, qkvs, outs}, P.MlpW{ln2s, gates, ups, downs},
      take(x, 0n, 32n), 0n, LinCache{L.zline(8n), L.zline(64n), L.zline(8n), L.zline(64n),
        L.zline(8n), L.zline(64n), L.zline(8n), L.zline(64n)})))
  +yfull = block_fin(T.Layer.fwd(M.BlockP(), TA.Mat<192n, 32n>,
    TA.Mat<192n, 32n>, M.LinBlockS(), M.LinBlock(), p2,
    TA.Mat.from_list(192n, 32n, x)))
  +d = P.maxdiff(ystep, yfull, 0.0)
  do IO<Unit>:
    Unit <- IO.print(String.append("lin block max|d| ", F32.show(d)))
    Unit <- IO.print(Bool.pick(String, F32.is_le(d, 1.0e-04),
      "GLM LIN BLOCK PASS", "GLM LIN BLOCK FAIL"))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    lns : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, "results/glm53-hybrid192-ep4/model-400/b2/ln1.bin", 0)
    qkvs : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, "results/glm53-hybrid192-ep4/model-400/b2/qkv.bin", 0)
    outs : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, "results/glm53-hybrid192-ep4/model-400/b2/out.bin", 0)
    ln2s : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, "results/glm53-hybrid192-ep4/model-400/b2/ln2.bin", 0)
    gates : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, "results/glm53-hybrid192-ep4/model-400/b2/gate.bin", 0)
    ups : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, "results/glm53-hybrid192-ep4/model-400/b2/up.bin", 0)
    downs : TA.Mat<32n, 128n> <- T.Mat.load(32n, 128n, "results/glm53-hybrid192-ep4/model-400/b2/down.bin", 0)
    p2 : W.BlockP() <- W.load_block("results/glm53-hybrid192-ep4/model-400/b2")
    run(lns, qkvs, outs, ln2s, gates, ups, downs, p2)
