# Wider DiT attention: the same QKV -> FullAttention -> output branch as
# attn_probe.bend, instantiated at b=2, t=4, w=8 (kv=2 heads of hd=4, kvw=8),
# via top-level Nat constants — the pattern the vit example uses. Forward and
# backward printed as raw F32 words for reference/dit_wide_attn_check.py.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import bendtorch@2.1.0.0/main.bend as T

def B() -> Nat: 2n
def TK() -> Nat: 4n
def N() -> Nat: 8n
def W() -> Nat: 8n
def KVW() -> Nat: 8n
def KV() -> Nat: 2n
def G() -> Nat: 1n
def HD() -> Nat: 4n

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

def lin(+i: Nat, +o: Nat, +s: Nat) -> T.Lin<i, o>:
  T.Lin{mat(o, i, s), mat(1n, o, Nat.add(s, 40n))}

def show_lin(+i: Nat, +o: Nat, +t: String, l: T.Lin<i, o>) -> IO(Unit):
  match l:
    case T.Lin{w, b}:
      do IO<Unit>:
        Unit <- T.Mat.print(o, i, tag(t, "dw"), w)
        T.Mat.print(1n, o, tag(t, "db"), b)

def dup(m: M(N(), W())) -> T.Two<M(N(), W()), M(N(), W())>:
  T.Mat.dup(N(), W(), m)

def add(a: M(N(), W()), b: M(N(), W())) -> M(N(), W()):
  T.Mat.sum2(N(), W(), a, b)

def KVL() -> T.Layer<T.Two<T.Lin<W(), W()>, T.Lin<W(), W()>>, M(N(), W()), T.Two<M(N(), W()), M(N(), W())>, T.Two<M(N(), W()), M(N(), W())>>:
  T.Fork(~T.Lin<W(), W()>, ~T.Lin<W(), W()>, ~M(N(), W()), ~M(N(), W()), ~M(N(), W()), ~M(N(), W()), ~M(N(), W()),
    ~(m => dup(m)), ~(a => b => add(a, b)),
    ~T.Linear(N(), W(), W(), 0n), ~T.Linear(N(), W(), W(), 0n))

def QKV() -> T.Layer<T.Two<T.Lin<W(), W()>, T.Two<T.Lin<W(), W()>, T.Lin<W(), W()>>>, M(N(), W()), T.Two<M(N(), W()), T.Two<M(N(), W()), M(N(), W())>>, T.Two<M(N(), W()), T.Two<M(N(), W()), M(N(), W())>>>:
  T.Fork(~T.Lin<W(), W()>, ~T.Two<T.Lin<W(), W()>, T.Lin<W(), W()>>, ~M(N(), W()), ~M(N(), W()), ~T.Two<M(N(), W()), M(N(), W())>, ~M(N(), W()), ~T.Two<M(N(), W()), M(N(), W())>,
    ~(m => dup(m)), ~(a => b => add(a, b)),
    ~T.Linear(N(), W(), W(), 0n), ~KVL())

def AttnP() -> Type:
  T.Two<T.Two<T.Lin<W(), W()>, T.Two<T.Lin<W(), W()>, T.Lin<W(), W()>>>, T.Two<Unit, T.Lin<W(), W()>>>

def AttnState() -> Type:
  T.Two<T.Two<M(N(), W()), T.Two<M(N(), W()), M(N(), W())>>, T.Two<T.AttnS, M(N(), W())>>

def Core() -> T.Layer<T.Two<Unit, T.Lin<W(), W()>>, T.Two<M(N(), W()), T.Two<M(N(), KVW()), M(N(), KVW())>>, M(N(), W()), T.Two<T.AttnS, M(N(), W())>>:
  T.Seq(~Unit, ~T.Lin<W(), W()>, ~T.Two<M(N(), W()), T.Two<M(N(), KVW()), M(N(), KVW())>>, ~M(N(), W()), ~M(N(), W()), ~T.AttnS, ~M(N(), W()),
    ~T.FullAttention(N(), W(), KVW(), B(), TK(), KV(), G(), HD(), 0.5, {==}, {==}, {==}), ~T.Linear(N(), W(), W(), 0n))

def AttnF() -> T.Layer<AttnP(), M(N(), W()), M(N(), W()), AttnState()>:
  T.Seq(~T.Two<T.Lin<W(), W()>, T.Two<T.Lin<W(), W()>, T.Lin<W(), W()>>>, ~T.Two<Unit, T.Lin<W(), W()>>,
    ~M(N(), W()), ~T.Two<M(N(), W()), T.Two<M(N(), W()), M(N(), W())>>, ~M(N(), W()),
    ~T.Two<M(N(), W()), T.Two<M(N(), W()), M(N(), W())>>, ~T.Two<T.AttnS, M(N(), W())>,
    ~QKV(), ~Core())

def attn_p() -> AttnP():
  T.Two{T.Two{lin(W(), W(), 3n), T.Two{lin(W(), W(), 7n), lin(W(), W(), 11n)}}, T.Two{Unit{}, lin(W(), W(), 15n)}}

def show_kv(kv: T.Two<T.Lin<W(), W()>, T.Lin<W(), W()>>) -> IO(Unit):
  match kv:
    case T.Two{k, v}:
      do IO<Unit>:
        Unit <- show_lin(W(), W(), "wide.k", k)
        show_lin(W(), W(), "wide.v", v)

def show_q(q: T.Two<T.Lin<W(), W()>, T.Two<T.Lin<W(), W()>, T.Lin<W(), W()>>>) -> IO(Unit):
  match q:
    case T.Two{qp, kv}:
      do IO<Unit>:
        Unit <- show_lin(W(), W(), "wide.q", qp)
        show_kv(kv)

def show_c(c: T.Two<Unit, T.Lin<W(), W()>>) -> IO(Unit):
  match c:
    case T.Two{u, o}:
      show_lin(W(), W(), "wide.o", o)

def show_p(r: AttnP()) -> IO(Unit):
  match r:
    case T.Two{q, c}:
      do IO<Unit>:
        Unit <- show_q(q)
        show_c(c)

def show_bwd(r: T.Bwd<AttnP(), M(N(), W())>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- T.Mat.print(N(), W(), "wide.dh", dx)
        show_p(g)

def show_fwd(r: T.Fwd<AttnP(), M(N(), W()), AttnState()>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(N(), W(), "wide.y", y)
        show_bwd(T.Layer.bwd(AttnP(), M(N(), W()), M(N(), W()), AttnState(), AttnF(), p, s,
          mat(N(), W(), 97n), T.Layer.zeros(AttnP(), M(N(), W()), M(N(), W()), AttnState(), AttnF())))

def main() -> IO(Unit):
  show_fwd(T.Layer.fwd(AttnP(), M(N(), W()), M(N(), W()), AttnState(), AttnF(), attn_p, mat(N(), W(), 33n)))
