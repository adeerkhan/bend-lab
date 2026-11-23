# DiT attention probe: QKV projection + FullAttention + output projection,
# as the `f` of an adaLN-conditioned DiT block. b=2 images of t=3 tokens,
# width w=4, one kv head (kv=1, g=1, hd=4, kvw=4), scale 1/sqrt(4)=0.5.
# All layers are BendTorch's; this is a forward+backward parity probe.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import bendtorch@2.1.0.0/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

def mat(+r: Nat, +c: Nat, +s: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, vals(Nat.mul(r, c), s))

def tag(+t: String, +name: String) -> String:
  String.append(t, String.append(".", name))

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def L44() -> Type:
  T.Lin<4n, 4n>

def Fl() -> T.Layer<L44, M(6n, 4n), M(6n, 4n), M(6n, 4n)>:
  T.Linear(6n, 4n, 4n, 0n)

def lin(+i: Nat, +o: Nat, +s: Nat) -> T.Lin<i, o>:
  T.Lin{mat(o, i, s), mat(1n, o, Nat.add(s, 40n))}

def show_lin(+i: Nat, +o: Nat, +t: String, l: T.Lin<i, o>) -> IO(Unit):
  match l:
    case T.Lin{w, b}:
      do IO<Unit>:
        Unit <- T.Mat.print(o, i, tag(t, "dw"), w)
        T.Mat.print(1n, o, tag(t, "db"), b)

# ---- QKV: fork h into (q, (k, v)), each a Linear(4,4) ----

def KVP() -> Type:
  T.Two<L44, L44>

def KVY() -> Type:
  T.Two<M(6n, 4n), M(6n, 4n)>

def KVL() -> T.Layer<KVP, M(6n, 4n), KVY, KVY>:
  T.Fork(~L44, ~L44, ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n),
    ~(m => T.Mat.dup(6n, 4n, m)), ~(a => b => T.Mat.sum2(6n, 4n, a, b)), ~Fl(), ~Fl())

def QKVP() -> Type:
  T.Two<L44, KVP>

def QKVY() -> Type:
  T.Two<M(6n, 4n), KVY>

def QKVS() -> Type:
  T.Two<M(6n, 4n), KVY>

def QKV() -> T.Layer<QKVP, M(6n, 4n), QKVY, QKVS>:
  T.Fork(~L44, ~KVP, ~M(6n, 4n), ~M(6n, 4n), ~KVY, ~M(6n, 4n), ~KVY,
    ~(m => T.Mat.dup(6n, 4n, m)), ~(a => b => T.Mat.sum2(6n, 4n, a, b)), ~Fl(), ~KVL())

# ---- Core: FullAttention then output projection ----

def CoreP() -> Type:
  T.Two<Unit, L44>

def CoreS() -> Type:
  T.Two<T.AttnS, M(6n, 4n)>

def Core() -> T.Layer<CoreP, QKVY, M(6n, 4n), CoreS>:
  T.Seq(~Unit, ~L44, ~QKVY, ~M(6n, 4n), ~M(6n, 4n), ~T.AttnS, ~M(6n, 4n),
    ~T.FullAttention(6n, 4n, 4n, 2n, 3n, 1n, 1n, 4n, 0.5, {==}, {==}, {==}), ~Fl())

def AttnP() -> Type:
  T.Two<QKVP, CoreP>

def AttnS() -> Type:
  T.Two<QKVS, CoreS>

def AttnF() -> T.Layer<AttnP, M(6n, 4n), M(6n, 4n), AttnS>:
  T.Seq(~QKVP, ~CoreP, ~M(6n, 4n), ~QKVY, ~M(6n, 4n), ~QKVS, ~CoreS, ~QKV(), ~Core())

def attn_p() -> AttnP:
  T.Two{T.Two{lin(4n, 4n, 3n), T.Two{lin(4n, 4n, 7n), lin(4n, 4n, 11n)}}, T.Two{Unit{}, lin(4n, 4n, 15n)}}

def dp2(kv: KVP) -> IO(Unit):
  match kv:
    case T.Two{k, v}:
      do IO<Unit>:
        Unit <- show_lin(4n, 4n, "attn.k", k)
        show_lin(4n, 4n, "attn.v", v)

def dpc(c: CoreP) -> IO(Unit):
  match c:
    case T.Two{u, o}:
      show_lin(4n, 4n, "attn.o", o)

def dpq(q: QKVP) -> IO(Unit):
  match q:
    case T.Two{qproj, kv}:
      do IO<Unit>:
        Unit <- show_lin(4n, 4n, "attn.q", qproj)
        dp2(kv)

def dp(r: AttnP) -> IO(Unit):
  match r:
    case T.Two{q, c}:
      do IO<Unit>:
        Unit <- dpq(q)
        dpc(c)

def show_bwd(r: T.Bwd<AttnP, M(6n, 4n)>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- T.Mat.print(6n, 4n, "attn.dh", dx)
        dp(g)

def fwd_show(r: T.Fwd<AttnP, M(6n, 4n), AttnS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(6n, 4n, "attn.y", y)
        show_bwd(T.Layer.bwd(AttnP, M(6n, 4n), M(6n, 4n), AttnS, AttnF, p, s, mat(6n, 4n, 97n), T.Layer.zeros(AttnP, M(6n, 4n), M(6n, 4n), AttnS, AttnF)))

def main() -> IO(Unit):
  fwd_show(T.Layer.fwd(AttnP, M(6n, 4n), M(6n, 4n), AttnS, AttnF, attn_p, mat(6n, 4n, 33n)))
