# Isolation probe: LayerNorm -> attention -> residual, WITHOUT the adaLN gate
# or the (m, sh) modulation. y = h + AttnF(LN(h)). Same shapes as the DiT block
# (b=2, t=3, w=4, one kv head, scale 0.5), same dy. This isolates whether the
# 5.45x gradient gap in block.bend comes from depth (LN stacked on attention)
# or from the gate/modulation interaction.
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

def vals2(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.add(Nat.mul(Nat.mul(i, i), 5n), Nat.mul(i, 3n)), 17n)) - 8.0 : F32) / 4.0 : F32), vals2(rest, Nat.add(i, 1n))}

def mat2(+r: Nat, +c: Nat, +s: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, vals2(Nat.mul(r, c), s))

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

def show_ln(+d: Nat, +t: String, l: T.LN<d>) -> IO(Unit):
  match l:
    case T.LN{g, b}:
      do IO<Unit>:
        Unit <- T.Mat.print(1n, d, tag(t, "dg"), g)
        T.Mat.print(1n, d, tag(t, "db"), b)

def ln4() -> T.LN<4n>:
  T.LN{mat(1n, 4n, 61n), mat(1n, 4n, 67n)}

# ---- attention f (identical to attn_probe.bend) ----

def AttnP() -> Type:
  T.Two<T.Two<T.Lin<4n, 4n>, T.Two<T.Lin<4n, 4n>, T.Lin<4n, 4n>>>, T.Two<Unit, T.Lin<4n, 4n>>>

def AttnState() -> Type:
  T.Two<T.Two<M(6n, 4n), T.Two<M(6n, 4n), M(6n, 4n)>>, T.Two<T.AttnS, M(6n, 4n)>>

def KVL() -> T.Layer<T.Two<T.Lin<4n, 4n>, T.Lin<4n, 4n>>, M(6n, 4n), T.Two<M(6n, 4n), M(6n, 4n)>, T.Two<M(6n, 4n), M(6n, 4n)>>:
  T.Fork(~T.Lin<4n, 4n>, ~T.Lin<4n, 4n>, ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n),
    ~(m => T.Mat.dup(6n, 4n, m)), ~(a => b => T.Mat.sum2(6n, 4n, a, b)),
    ~T.Linear(6n, 4n, 4n, 0n), ~T.Linear(6n, 4n, 4n, 0n))

def QKV() -> T.Layer<T.Two<T.Lin<4n, 4n>, T.Two<T.Lin<4n, 4n>, T.Lin<4n, 4n>>>, M(6n, 4n), T.Two<M(6n, 4n), T.Two<M(6n, 4n), M(6n, 4n)>>, T.Two<M(6n, 4n), T.Two<M(6n, 4n), M(6n, 4n)>>>:
  T.Fork(~T.Lin<4n, 4n>, ~T.Two<T.Lin<4n, 4n>, T.Lin<4n, 4n>>, ~M(6n, 4n), ~M(6n, 4n), ~T.Two<M(6n, 4n), M(6n, 4n)>, ~M(6n, 4n), ~T.Two<M(6n, 4n), M(6n, 4n)>,
    ~(m => T.Mat.dup(6n, 4n, m)), ~(a => b => T.Mat.sum2(6n, 4n, a, b)),
    ~T.Linear(6n, 4n, 4n, 0n), ~KVL())

def AttnF() -> T.Layer<AttnP, M(6n, 4n), M(6n, 4n), AttnState>:
  T.Seq(~T.Two<T.Lin<4n, 4n>, T.Two<T.Lin<4n, 4n>, T.Lin<4n, 4n>>>, ~T.Two<Unit, T.Lin<4n, 4n>>,
    ~M(6n, 4n), ~T.Two<M(6n, 4n), T.Two<M(6n, 4n), M(6n, 4n)>>, ~M(6n, 4n),
    ~T.Two<M(6n, 4n), T.Two<M(6n, 4n), M(6n, 4n)>>, ~T.Two<T.AttnS, M(6n, 4n)>,
    ~QKV(),
    ~T.Seq(~Unit, ~T.Lin<4n, 4n>, ~T.Two<M(6n, 4n), T.Two<M(6n, 4n), M(6n, 4n)>>, ~M(6n, 4n), ~M(6n, 4n), ~T.AttnS, ~M(6n, 4n),
      ~T.FullAttention(6n, 4n, 4n, 2n, 3n, 1n, 1n, 4n, 0.5, {==}, {==}, {==}), ~T.Linear(6n, 4n, 4n, 0n)))

def attn_p() -> AttnP:
  T.Two{T.Two{lin(4n, 4n, 3n), T.Two{lin(4n, 4n, 7n), lin(4n, 4n, 11n)}}, T.Two{Unit{}, lin(4n, 4n, 15n)}}

# ---- LayerNorm -> attention -> residual (no gate, no m/sh) ----

def LNAttnP() -> Type:
  T.Two<T.LN<4n>, AttnP>

def LNAttnS() -> Type:
  T.Two<M(6n, 4n), AttnState>

def LNAttnIn() -> T.Layer<LNAttnP, M(6n, 4n), M(6n, 4n), LNAttnS>:
  T.Seq(~T.LN<4n>, ~AttnP, ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~AttnState,
    ~T.LayerNorm(6n, 4n, 0.00001), ~AttnF())

def LNAttn() -> T.Layer<T.Two<T.Two<Unit, LNAttnP>, Unit>, M(6n, 4n), M(6n, 4n), T.Two<T.Two<Unit, LNAttnS>, Unit>>:
  T.Residual(6n, 4n, LNAttnP, LNAttnS, LNAttnIn())

def p_all() -> T.Two<T.Two<Unit, LNAttnP>, Unit>:
  T.Two{T.Two{Unit{}, T.Two{ln4(), attn_p()}}, Unit{}}

# ---- display ----

def show_kv(kv: T.Two<T.Lin<4n, 4n>, T.Lin<4n, 4n>>) -> IO(Unit):
  match kv:
    case T.Two{k, v}:
      do IO<Unit>:
        Unit <- show_lin(4n, 4n, "lnattn.k", k)
        show_lin(4n, 4n, "lnattn.v", v)

def show_q(q: T.Two<T.Lin<4n, 4n>, T.Two<T.Lin<4n, 4n>, T.Lin<4n, 4n>>>) -> IO(Unit):
  match q:
    case T.Two{qp, kv}:
      do IO<Unit>:
        Unit <- show_lin(4n, 4n, "lnattn.q", qp)
        show_kv(kv)

def show_c(c: T.Two<Unit, T.Lin<4n, 4n>>) -> IO(Unit):
  match c:
    case T.Two{u, o}:
      show_lin(4n, 4n, "lnattn.o", o)

def show_attn(r: AttnP) -> IO(Unit):
  match r:
    case T.Two{q, c}:
      do IO<Unit>:
        Unit <- show_q(q)
        show_c(c)

def show_dkv(dkv: T.Two<T.Lin<4n, 4n>, T.Lin<4n, 4n>>) -> IO(Unit):
  match dkv:
    case T.Two{dk, dv}:
      do IO<Unit>:
        Unit <- show_lin(4n, 4n, "lnattn.k", dk)
        show_lin(4n, 4n, "lnattn.v", dv)

def show_dqkv(dqkv: T.Two<T.Lin<4n, 4n>, T.Two<T.Lin<4n, 4n>, T.Lin<4n, 4n>>>) -> IO(Unit):
  match dqkv:
    case T.Two{dq, dkv}:
      do IO<Unit>:
        Unit <- show_lin(4n, 4n, "lnattn.q", dq)
        show_dkv(dkv)

def show_dcore(c: T.Two<Unit, T.Lin<4n, 4n>>) -> IO(Unit):
  match c:
    case T.Two{u, dow}:
      show_lin(4n, 4n, "lnattn.o", dow)

def show_dattn(dattn: AttnP) -> IO(Unit):
  match dattn:
    case T.Two{dqkv, c}:
      do IO<Unit>:
        Unit <- show_dqkv(dqkv)
        show_dcore(c)

def show_grads(g: T.Two<T.Two<Unit, LNAttnP>, Unit>) -> IO(Unit):
  match g:
    case T.Two{inner, u}:
      match inner:
        case T.Two{u2, st}:
          match st:
            case T.Two{dln, dattn}:
              do IO<Unit>:
                Unit <- show_ln(4n, "lnattn.ln", dln)
                show_dattn(dattn)

def show_bwd(r: T.Bwd<T.Two<T.Two<Unit, LNAttnP>, Unit>, M(6n, 4n)>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- T.Mat.print(6n, 4n, "lnattn.dh", dx)
        show_grads(g)

def show_fwd(r: T.Fwd<T.Two<T.Two<Unit, LNAttnP>, Unit>, M(6n, 4n), T.Two<T.Two<Unit, LNAttnS>, Unit>>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(6n, 4n, "lnattn.y", y)
        show_bwd(T.Layer.bwd(T.Two<T.Two<Unit, LNAttnP>, Unit>, M(6n, 4n), M(6n, 4n), T.Two<T.Two<Unit, LNAttnS>, Unit>, LNAttn, p, s,
          mat(6n, 4n, 97n), T.Layer.zeros(T.Two<T.Two<Unit, LNAttnP>, Unit>, M(6n, 4n), M(6n, 4n), T.Two<T.Two<Unit, LNAttnS>, Unit>, LNAttn)))

def main() -> IO(Unit):
  show_fwd(T.Layer.fwd(T.Two<T.Two<Unit, LNAttnP>, Unit>, M(6n, 4n), M(6n, 4n), T.Two<T.Two<Unit, LNAttnS>, Unit>, LNAttn, p_all, mat2(6n, 4n, 33n)))
