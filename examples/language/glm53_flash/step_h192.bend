# The four-block hybrid decoded one token at a time, on the trained weights.
#
# The prefill runs the real prompt rows through the four block step functions
# (three linear, one sparse), then the decode loop feeds the generated token
# back at the next position. The golden is the recorded full-prefix generator's
# 48 tokens for the same episode.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./probe_glm_block.bend as P
import ./probe_glm_block_lin.bend as Lp
import ./lin_bwd.bend as L

def POS() -> Nat:
  192n

def NEW() -> Nat:
  48n

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def nth(+n: Nat, xs: List<&2, Nat>) -> Nat:
  match n xs:
    case 0n Con{h, t}:
      h
    case 1n+p Con{h, t}:
      nth(p, t)
    case _ _:
      0n

def head_u32(xs: List<&2, U32>) -> U32:
  match xs:
    case Con{h, t}:
      h
    case Nil{}:
      0

def ml260(ml: TA.ML<1n, 260n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def ml260x32(ml: TA.ML<260n, 32n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def skip(+i: Nat) -> Bool:
  Bool.or(Bool.or(Nat.is_eq(i, 0n), Nat.is_eq(i, 1n)), Nat.is_eq(i, 3n))

def argmax_go(xs: List<&2, F32>, +i: Nat, +best_i: Nat, +best_v: F32) -> Nat:
  match xs:
    case Nil{}:
      best_i
    case Con{+h, t}:
      +score = Bool.pick(F32, skip(i), (0.0 - 1.0e30 : F32), h)
      +gt = F32.is_gt(score, best_v)
      argmax_go(t, (i + 1n : Nat), Bool.pick(Nat, gt, i, best_i),
        Bool.pick(F32, gt, score, best_v))

def argmax(xs: List<&2, F32>) -> Nat:
  argmax_go(xs, 0n, 4n, (0.0 - 1.0e30 : F32))

type MdlW is Type:
  MdlW{embL: List<&2, F32>, embM: TA.Mat<260n, 32n>, lnf: TA.Mat<1n, 32n>,
       a0: P.AttW, m0: P.MlpW, a1: P.AttW, m1: P.MlpW,
       a2: P.AttW, m2: P.MlpW, a3: P.AttW, m3: P.MlpW}

type Cache is Type:
  Cache{c0: Lp.LinCache, c1: Lp.LinCache, c2: Lp.LinCache, c3: P.SpCache}

def head_fin(r: TA.MMulNT<1n, 32n, 260n>) -> List<&2, F32> & TA.Mat<260n, 32n>:
  match r:
    case TA.MMulNT{x, w, y}:
      Tuple{ml260(TA.Mat.to_list(1n, 260n, y)), w}

def step_fin_b(embL: List<&2, F32>,
    a0b: P.AttW, m0b: P.MlpW, a1b: P.AttW, m1b: P.MlpW,
    a2b: P.AttW, m2b: P.MlpW, a3b: P.AttW, m3b: P.MlpW,
    c0b: Lp.LinCache, c1b: Lp.LinCache, c2b: Lp.LinCache,
    c3b: P.SpCache, lnfb: TA.Mat<1n, 32n>,
    r: List<&2, F32> & TA.Mat<260n, 32n>) -> MdlW & (List<&2, F32> & Cache):
  match r:
    case Tuple{logits, embMb}:
      Tuple{MdlW{embL, embMb, lnfb, a0b, m0b, a1b, m1b, a2b, m2b,
        a3b, m3b}, Tuple{logits, Cache{c0b, c1b, c2b, c3b}}}

def step_fin_a(embL: List<&2, F32>, embM: TA.Mat<260n, 32n>,
    a0b: P.AttW, m0b: P.MlpW, a1b: P.AttW, m1b: P.MlpW,
    a2b: P.AttW, m2b: P.MlpW, a3b: P.AttW, m3b: P.MlpW,
    c0b: Lp.LinCache, c1b: Lp.LinCache, c2b: Lp.LinCache,
    c3b: P.SpCache, r: List<&2, F32> & TA.Mat<1n, 32n>) -> MdlW & (List<&2, F32> & Cache):
  match r:
    case Tuple{xn, lnfb}:
      step_fin_b(embL, a0b, m0b, a1b, m1b, a2b, m2b, a3b, m3b,
        c0b, c1b, c2b, c3b, lnfb,
        head_fin(TA.Mat.matmul_nt(1n, 32n, 260n, 0n,
          TA.Mat.from_list(1n, 32n, xn), embM)))

def step_sp(embL: List<&2, F32>, embM: TA.Mat<260n, 32n>,
    lnf: TA.Mat<1n, 32n>,
    a0b: P.AttW, m0b: P.MlpW, a1b: P.AttW, m1b: P.MlpW,
    a2b: P.AttW, m2b: P.MlpW,
    c0b: Lp.LinCache, c1b: Lp.LinCache, c2b: Lp.LinCache,
    +x3: List<&2, F32>, +t: Nat,
    r: (P.AttW & P.MlpW) & (List<&2, F32> & P.SpCache)) -> MdlW & (List<&2, F32> & Cache):
  match r:
    case Tuple{Tuple{a3b, m3b}, Tuple{x4, c3b}}:
      step_fin_a(embL, embM, a0b, m0b, a1b, m1b, a2b, m2b, a3b, m3b,
        c0b, c1b, c2b, c3b, P.rms1(lnf, x4))

def step_b2(embL: List<&2, F32>, embM: TA.Mat<260n, 32n>,
    lnf: TA.Mat<1n, 32n>, a0b: P.AttW, m0b: P.MlpW,
    a1b: P.AttW, m1b: P.MlpW, a3: P.AttW, m3: P.MlpW,
    c0b: Lp.LinCache, c1b: Lp.LinCache, c3: P.SpCache, +t: Nat,
    r: (P.AttW & P.MlpW) & (List<&2, F32> & Lp.LinCache)) -> MdlW & (List<&2, F32> & Cache):
  match r:
    case Tuple{Tuple{a2b, m2b}, Tuple{+x3, c2b}}:
      step_sp(embL, embM, lnf, a0b, m0b, a1b, m1b, a2b, m2b,
        c0b, c1b, c2b, x3, t, P.step_row(a3, m3, x3, t, c3))

def step_b1(embL: List<&2, F32>, embM: TA.Mat<260n, 32n>,
    lnf: TA.Mat<1n, 32n>, a0b: P.AttW, m0b: P.MlpW,
    a2: P.AttW, m2: P.MlpW, a3: P.AttW, m3: P.MlpW,
    c0b: Lp.LinCache, c2: Lp.LinCache, c3: P.SpCache, +t: Nat,
    r: (P.AttW & P.MlpW) & (List<&2, F32> & Lp.LinCache)) -> MdlW & (List<&2, F32> & Cache):
  match r:
    case Tuple{Tuple{a1b, m1b}, Tuple{x2, c1b}}:
      step_b2(embL, embM, lnf, a0b, m0b, a1b, m1b, a3, m3,
        c0b, c1b, c3, t, Lp.step_row(a2, m2, x2, t, c2))

def step_model(w: MdlW, +token: Nat, +t: Nat,
    c: Cache) -> MdlW & (List<&2, F32> & Cache):
  match w c:
    case MdlW{+embL, embM, lnf, a0, m0, a1, m1, a2, m2, a3, m3}
         Cache{c0, c1, c2, c3}:
      +x = P.take(embL, Nat.mul(token, 32n), 32n)
      step_b1(embL, embM, lnf, a1, m1, a2, m2, a3, m3, c1, c2, c3, t,
        Lp.step_row(a0, m0, x, t, c0))

type PS is Type:
  PS{w: MdlW, cache: Cache}

def prefill_go(+left: Nat, +t: Nat, +tokens: List<&2, Nat>,
    last: List<&2, F32>,
    r: MdlW & (List<&2, F32> & Cache)) -> PS & List<&2, F32>:
  match left last r:
    case 0n last0 Tuple{w, Tuple{logits, c}}:
      Tuple{PS{w, c}, last0}
    case 1n last0 Tuple{w2, Tuple{logits, c2}}:
      Tuple{PS{w2, c2}, logits}
    case 2n+p last0 Tuple{w2, Tuple{logits, c2}}:
      prefill_go((1n+p), (t + 1n : Nat), tokens, logits,
        step_model(w2, nth((t + 1n : Nat), tokens), (t + 1n : Nat), c2))
    case _ last1 Tuple{w, Tuple{logits, c}}:
      Tuple{PS{w, c}, last1}

def decode_go(+left: Nat, +t: Nat, +tok: Nat,
    +acc: List<&2, Nat>,
    r: MdlW & (List<&2, F32> & Cache)) -> List<&2, Nat>:
  match left acc r:
    case 0n acc0 _:
      acc0
    case 1n acc0 Tuple{w2, Tuple{logits, c2}}:
      List.append(&2, Nat, acc0, Con{argmax(logits), Nil{}})
    case 2n+p acc0 Tuple{w2, Tuple{logits, c2}}:
      +next = argmax(logits)
      +acc2 = List.append(&2, Nat, acc0, Con{next, Nil{}})
      decode_go((1n+p), (t + 1n : Nat), next, acc2,
        step_model(w2, next, (t + 1n : Nat), c2))
    case _ acc1 _:
      acc1

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(Nat.show(h), String.append(" ", show_nats(t)))

def decode_fin(+toks: List<&2, Nat>, +ep: Nat) -> IO(Unit):
  do IO<Unit>:
    Unit <- IO.print(String.append("tokens: ", show_nats(toks)))
    Unit <- IO.print(String.append("episode: ", String.append(Nat.show(ep),
      String.append(" ", show_nats(toks)))))
    return Unit{}

def finish_gen(ps: PS, +tok0: Nat, +len: Nat, +ep: Nat) -> IO(Unit):
  match ps:
    case PS{w, cache}:
      decode_fin(decode_go(47n, len, tok0, Con{tok0, Nil{}},
        step_model(w, tok0, len, cache)), ep)

def finish(ps_last: PS & List<&2, F32>, +len: Nat, +ep: Nat) -> IO(Unit):
  match ps_last:
    case Tuple{ps, last}:
      finish_gen(ps, argmax(last), len, ep)

def wp(+base: String, +name: String) -> String:
  String.append(base, name)

def run_episode(+weights: String, +tokens: List<&2, Nat>, +len: Nat,
    +ep: Nat) -> IO(Unit):
  do IO<Unit>:
    emb : TA.Mat<260n, 32n> <- T.Mat.load(260n, 32n, wp(weights, "/emb.bin"), 0)
    embM : TA.Mat<260n, 32n> <- T.Mat.load(260n, 32n, wp(weights, "/emb.bin"), 0)
    embL : List<&2, F32> <- IO.pure(List<&2, F32>, ml260x32(TA.Mat.to_list(260n, 32n, emb)))
    lnf : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wp(weights, "/lnf.bin"), 0)
    ln10 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wp(weights, "/b0/ln1.bin"), 0)
    qkv0 : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, wp(weights, "/b0/qkv.bin"), 0)
    out0 : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, wp(weights, "/b0/out.bin"), 0)
    ln20 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wp(weights, "/b0/ln2.bin"), 0)
    gate0 : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, wp(weights, "/b0/gate.bin"), 0)
    up0 : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, wp(weights, "/b0/up.bin"), 0)
    down0 : TA.Mat<32n, 128n> <- T.Mat.load(32n, 128n, wp(weights, "/b0/down.bin"), 0)
    ln11 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wp(weights, "/b1/ln1.bin"), 0)
    qkv1 : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, wp(weights, "/b1/qkv.bin"), 0)
    out1 : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, wp(weights, "/b1/out.bin"), 0)
    ln21 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wp(weights, "/b1/ln2.bin"), 0)
    gate1 : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, wp(weights, "/b1/gate.bin"), 0)
    up1 : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, wp(weights, "/b1/up.bin"), 0)
    down1 : TA.Mat<32n, 128n> <- T.Mat.load(32n, 128n, wp(weights, "/b1/down.bin"), 0)
    ln12 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wp(weights, "/b2/ln1.bin"), 0)
    qkv2 : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, wp(weights, "/b2/qkv.bin"), 0)
    out2 : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, wp(weights, "/b2/out.bin"), 0)
    ln22 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wp(weights, "/b2/ln2.bin"), 0)
    gate2 : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, wp(weights, "/b2/gate.bin"), 0)
    up2 : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, wp(weights, "/b2/up.bin"), 0)
    down2 : TA.Mat<32n, 128n> <- T.Mat.load(32n, 128n, wp(weights, "/b2/down.bin"), 0)
    ln13 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wp(weights, "/b3/ln1.bin"), 0)
    qkv3 : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, wp(weights, "/b3/qkv.bin"), 0)
    out3 : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, wp(weights, "/b3/out.bin"), 0)
    ln23 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, wp(weights, "/b3/ln2.bin"), 0)
    gate3 : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, wp(weights, "/b3/gate.bin"), 0)
    up3 : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, wp(weights, "/b3/up.bin"), 0)
    down3 : TA.Mat<32n, 128n> <- T.Mat.load(32n, 128n, wp(weights, "/b3/down.bin"), 0)
    a0 : P.AttW <- IO.pure(P.AttW, P.AttW{ln10, qkv0, out0})
    m0 : P.MlpW <- IO.pure(P.MlpW, P.MlpW{ln20, gate0, up0, down0})
    a1 : P.AttW <- IO.pure(P.AttW, P.AttW{ln11, qkv1, out1})
    m1 : P.MlpW <- IO.pure(P.MlpW, P.MlpW{ln21, gate1, up1, down1})
    a2 : P.AttW <- IO.pure(P.AttW, P.AttW{ln12, qkv2, out2})
    m2 : P.MlpW <- IO.pure(P.MlpW, P.MlpW{ln22, gate2, up2, down2})
    a3 : P.AttW <- IO.pure(P.AttW, P.AttW{ln13, qkv3, out3})
    m3 : P.MlpW <- IO.pure(P.MlpW, P.MlpW{ln23, gate3, up3, down3})
    w : MdlW <- IO.pure(MdlW, MdlW{embL, embM, lnf, a0, m0, a1, m1, a2, m2, a3, m3})
    cache : Cache <- IO.pure(Cache,
      Cache{Lp.LinCache{L.zline(8n), L.zline(64n), L.zline(8n), L.zline(64n),
                        L.zline(8n), L.zline(64n), L.zline(8n), L.zline(64n)},
            Lp.LinCache{L.zline(8n), L.zline(64n), L.zline(8n), L.zline(64n),
                        L.zline(8n), L.zline(64n), L.zline(8n), L.zline(64n)},
            Lp.LinCache{L.zline(8n), L.zline(64n), L.zline(8n), L.zline(64n),
                        L.zline(8n), L.zline(64n), L.zline(8n), L.zline(64n)},
            P.SpCache{Nil{}, Nil{}, Nil{}}})
    finish(prefill_go(len, 0n, tokens, Nil{},
      step_model(w, nth(0n, tokens), 0n, cache)), len, ep)

type Run is Type:
  Run{split: String, data: String, weights: String, count: Nat}

def run_default() -> Run:
  Run{"dev", "examples/language/glm53_flash/data",
    "results/glm53-hybrid192-ep4/model-400", 64n}

def nat_or(m: Maybe<&2, Nat>, d: Nat) -> Nat:
  match m:
    case Some{x}:
      x
    case None{}:
      d

def run_go(args: List<String>, r: Run) -> Run:
  match args:
    case Nil{}:
      r
    case Con{a, rest}:
      match a:
        case "--split":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{split, data, weights, count}:
                  run_go(more, Run{b, data, weights, count})
        case "--data":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{split, data, weights, count}:
                  run_go(more, Run{split, b, weights, count})
        case "--weights":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{split, data, weights, count}:
                  run_go(more, Run{split, data, b, count})
        case "--episodes":
          match rest:
            case Nil{}:
              r
            case Con{b, more}:
              match r:
                case Run{split, data, weights, count}:
                  run_go(more, Run{split, data, weights,
                    nat_or(Nat.read(b), count)})
        case _:
          run_go(rest, r)

def run_of(args: List<String>) -> Run:
  match args:
    case Nil{}:
      run_default()
    case Con{program, rest}:
      run_go(rest, run_default())

def eps_go(+left: Nat, +ep: Nat, +weights: String, +data: String,
    +split: String, +lengths: List<&2, Nat>) -> IO(Unit):
  match left:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        row : List<&2, U32> <- T.read_at(
          String.append(data, String.append("/eval-", String.append(split, ".bin"))),
          U32.from_nat(Nat.mul(ep, POS())), U32.from_nat(POS()))
        tokens : List<&2, Nat> <- IO.pure(List<&2, Nat>, nats(row))
        run_episode(weights, tokens, nth(ep, lengths), ep)
        eps_go(p, (ep + 1n : Nat), weights, data, split, lengths)

def run_args(r: Run) -> IO(Unit):
  match r:
    case Run{+split, +data, +weights, +count}:
      do IO<Unit>:
        lengths : List<&2, U32> <- T.read_at(
          String.append(data, String.append("/eval-", String.append(split, "-len.bin"))),
          0, U32.from_nat(count))
        eps_go(count, 0n, weights, data, split, nats(lengths))

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    run_args(run_of(args))
