# Full conditioned DiT block at width 8, two kv heads of hd 4:
#   y = h + Tile(g) * f( LayerNorm(h) * Tile(m) + Tile(sh) )
# f = QKV -> FullAttention -> output projection (wide_attn.bend), the (g,m,sh)
# conditioning is three Linear(CD, W) of silu(cond). All helpers are generalised
# from block.bend by replacing the width literal 4 with the top-level C().
# Forward and backward printed as raw F32 words for a PyTorch twin.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import bendtorch@2.1.0.0/main.bend as T

def B() -> Nat: 2n
def TK() -> Nat: 3n
def N() -> Nat: 6n
def C() -> Nat: 8n
def CD() -> Nat: 3n
def KVW() -> Nat: 8n
def KV() -> Nat: 2n
def G() -> Nat: 1n
def HD() -> Nat: 4n

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

def vals2(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.add(Nat.mul(Nat.mul(i, i), 5n), Nat.mul(i, 3n)), 17n)) - 8.0 : F32) / 4.0 : F32), vals2(rest, Nat.add(i, 1n))}

def mat(+r: Nat, +c: Nat, +s: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, vals(Nat.mul(r, c), s))

def mat2(+r: Nat, +c: Nat, +s: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, vals2(Nat.mul(r, c), s))

def tag(+t: String, +name: String) -> String:
  String.append(t, String.append(".", name))

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

def dup(m: M(N(), C())) -> T.Two<M(N(), C()), M(N(), C())>:
  T.Mat.dup(N(), C(), m)

def add(a: M(N(), C()), b: M(N(), C())) -> M(N(), C()):
  T.Mat.sum2(N(), C(), a, b)

# ---- width-generalised Rewire / Fma ----

def Rewire(~X: Type, ~Y: Type, ~f: X -> Y, ~b: Y -> X) -> T.Layer<Unit, X, Y, Unit>:
  T.Layer{p => x => T.Fwd{p, f(x), Unit{}}, p => s => dy => g => T.Bwd{p, b(dy), g}, op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

def sw2(-A: Type, -B: Type, -Cc: Type, a: A, r: T.Two<B, Cc>) -> T.Two<B, T.Two<A, Cc>>:
  match r:
    case T.Two{b, c}:
      T.Two{b, T.Two{a, c}}

def sw(-A: Type, -B: Type, -Cc: Type, x: T.Two<A, T.Two<B, Cc>>) -> T.Two<B, T.Two<A, Cc>>:
  match x:
    case T.Two{a, r}:
      sw2(A, B, Cc, a, r)

def X3() -> Type:
  T.Two<M(N(), C()), T.Two<M(N(), C()), M(N(), C())>>

def MulS() -> Type:
  T.Two<M(N(), C()), M(N(), C())>

def FmaP() -> Type:
  T.Two<Unit, T.Two<T.Two<Unit, Unit>, Unit>>

def FmaS() -> Type:
  T.Two<Unit, T.Two<T.Two<Unit, MulS()>, Unit>>

def MA() -> T.Layer<T.Two<Unit, Unit>, X3(), T.Two<M(N(), C()), M(N(), C())>, T.Two<Unit, MulS()>>:
  T.Both(~Unit, ~Unit, ~M(N(), C()), ~T.Two<M(N(), C()), M(N(), C())>, ~M(N(), C()), ~M(N(), C()), ~Unit, ~MulS(), ~T.Id(~M(N(), C())), ~T.Mul2(N(), C()))

def MAA() -> T.Layer<T.Two<T.Two<Unit, Unit>, Unit>, X3(), M(N(), C()), T.Two<T.Two<Unit, MulS()>, Unit>>:
  T.Seq(~T.Two<Unit, Unit>, ~Unit, ~X3(), ~T.Two<M(N(), C()), M(N(), C())>, ~M(N(), C()), ~T.Two<Unit, MulS()>, ~Unit, ~MA(), ~T.Add2(N(), C()))

def Fma() -> T.Layer<FmaP, X3(), M(N(), C()), FmaS()>:
  T.Seq(~Unit, ~T.Two<T.Two<Unit, Unit>, Unit>, ~X3(), ~X3(), ~M(N(), C()), ~Unit, ~T.Two<T.Two<Unit, MulS()>, Unit>,
    ~Rewire(~X3(), ~X3(), ~(x => sw(M(N(), C()), M(N(), C()), M(N(), C()), x)), ~(x => sw(M(N(), C()), M(N(), C()), M(N(), C()), x))),
    ~MAA())

def unitsF() -> FmaP:
  T.Two{Unit{}, T.Two{T.Two{Unit{}, Unit{}}, Unit{}}}

# ---- attention f (wide, from wide_attn.bend) ----

def KVL() -> T.Layer<T.Two<T.Lin<C(), C()>, T.Lin<C(), C()>>, M(N(), C()), T.Two<M(N(), C()), M(N(), C())>, T.Two<M(N(), C()), M(N(), C())>>:
  T.Fork(~T.Lin<C(), C()>, ~T.Lin<C(), C()>, ~M(N(), C()), ~M(N(), C()), ~M(N(), C()), ~M(N(), C()), ~M(N(), C()),
    ~(m => dup(m)), ~(a => b => add(a, b)),
    ~T.Linear(N(), C(), C(), 0n), ~T.Linear(N(), C(), C(), 0n))

def QKV() -> T.Layer<T.Two<T.Lin<C(), C()>, T.Two<T.Lin<C(), C()>, T.Lin<C(), C()>>>, M(N(), C()), T.Two<M(N(), C()), T.Two<M(N(), C()), M(N(), C())>>, T.Two<M(N(), C()), T.Two<M(N(), C()), M(N(), C())>>>:
  T.Fork(~T.Lin<C(), C()>, ~T.Two<T.Lin<C(), C()>, T.Lin<C(), C()>>, ~M(N(), C()), ~M(N(), C()), ~T.Two<M(N(), C()), M(N(), C())>, ~M(N(), C()), ~T.Two<M(N(), C()), M(N(), C())>,
    ~(m => dup(m)), ~(a => b => add(a, b)),
    ~T.Linear(N(), C(), C(), 0n), ~KVL())

def AttnP() -> Type:
  T.Two<T.Two<T.Lin<C(), C()>, T.Two<T.Lin<C(), C()>, T.Lin<C(), C()>>>, T.Two<Unit, T.Lin<C(), C()>>>

def AttnState() -> Type:
  T.Two<T.Two<M(N(), C()), T.Two<M(N(), C()), M(N(), C())>>, T.Two<T.AttnS, M(N(), C())>>

def Core() -> T.Layer<T.Two<Unit, T.Lin<C(), C()>>, T.Two<M(N(), C()), T.Two<M(N(), KVW()), M(N(), KVW())>>, M(N(), C()), T.Two<T.AttnS, M(N(), C())>>:
  T.Seq(~Unit, ~T.Lin<C(), C()>, ~T.Two<M(N(), C()), T.Two<M(N(), KVW()), M(N(), KVW())>>, ~M(N(), C()), ~M(N(), C()), ~T.AttnS, ~M(N(), C()),
    ~T.FullAttention(N(), C(), KVW(), B(), TK(), KV(), G(), HD(), 0.5, {==}, {==}, {==}), ~T.Linear(N(), C(), C(), 0n))

def AttnF() -> T.Layer<AttnP(), M(N(), C()), M(N(), C()), AttnState()>:
  T.Seq(~T.Two<T.Lin<C(), C()>, T.Two<T.Lin<C(), C()>, T.Lin<C(), C()>>>, ~T.Two<Unit, T.Lin<C(), C()>>,
    ~M(N(), C()), ~T.Two<M(N(), C()), T.Two<M(N(), C()), M(N(), C())>>, ~M(N(), C()),
    ~T.Two<M(N(), C()), T.Two<M(N(), C()), M(N(), C())>>, ~T.Two<T.AttnS, M(N(), C())>,
    ~QKV(), ~Core())

def attn_p() -> AttnP():
  T.Two{T.Two{lin(C(), C(), 3n), T.Two{lin(C(), C(), 7n), lin(C(), C(), 11n)}}, T.Two{Unit{}, lin(C(), C(), 15n)}}

# ---- conditioning: silu(cond) -> (g, (m, sh)) via Linear(CD, C) ----

def LCD() -> Type:
  T.Lin<CD(), C()>

def Lc() -> T.Layer<LCD, M(B(), CD()), M(B(), C()), M(B(), CD())>:
  T.Linear(B(), CD(), C(), 0n)

def dupc(m: M(B(), CD())) -> T.Two<M(B(), CD()), M(B(), CD())>:
  T.Mat.dup(B(), CD(), m)

def addc(a: M(B(), CD()), b: M(B(), CD())) -> M(B(), CD()):
  T.Mat.sum2(B(), CD(), a, b)

def K2() -> T.Layer<T.Two<LCD, LCD>, M(B(), CD()), T.Two<M(B(), C()), M(B(), C())>, T.Two<M(B(), CD()), M(B(), CD())>>:
  T.Fork(~LCD, ~LCD, ~M(B(), CD()), ~M(B(), C()), ~M(B(), C()), ~M(B(), CD()), ~M(B(), CD()),
    ~(m => dupc(m)), ~(a => b => addc(a, b)), ~Lc, ~Lc)

def K3P() -> Type:
  T.Two<LCD, T.Two<LCD, LCD>>

def K3S() -> Type:
  T.Two<M(B(), CD()), T.Two<M(B(), CD()), M(B(), CD())>>

def C3() -> Type:
  T.Two<M(B(), C()), T.Two<M(B(), C()), M(B(), C())>>

def K3() -> T.Layer<K3P, M(B(), CD()), C3, K3S>:
  T.Fork(~LCD, ~T.Two<LCD, LCD>, ~M(B(), CD()), ~M(B(), C()), ~T.Two<M(B(), C()), M(B(), C())>, ~M(B(), CD()), ~T.Two<M(B(), CD()), M(B(), CD())>,
    ~(m => dupc(m)), ~(a => b => addc(a, b)), ~Lc, ~K2)

def KP() -> Type:
  T.Two<Unit, K3P>

def KS() -> Type:
  T.Two<M(B(), CD()), K3S>

def K() -> T.Layer<KP, M(B(), CD()), C3, KS>:
  T.Seq(~Unit, ~K3P, ~M(B(), CD()), ~M(B(), CD()), ~C3, ~M(B(), CD()), ~K3S, ~T.SiLU(B(), CD()), ~K3)

# ---- mod: (m, (sh, h)) to sh_t + m_t * LayerNorm(h) ----

def TLP() -> Type:
  T.Two<Unit, T.LN<C()>>

def TLS() -> Type:
  T.Two<Unit, M(N(), C())>

def TL() -> T.Layer<TLP, T.Two<M(B(), C()), M(N(), C())>, T.Two<M(N(), C()), M(N(), C())>, TLS>:
  T.Both(~Unit, ~T.LN<C()>, ~M(B(), C()), ~M(N(), C()), ~M(N(), C()), ~M(N(), C()), ~Unit, ~M(N(), C()),
    ~T.Tile(B(), TK(), C()), ~T.LayerNorm(N(), C(), 0.00001))

def MX() -> Type:
  T.Two<M(B(), C()), T.Two<M(B(), C()), M(N(), C())>>

def M1P() -> Type:
  T.Two<Unit, TLP>

def M1S() -> Type:
  T.Two<Unit, TLS>

def M1() -> T.Layer<M1P, MX, X3(), M1S>:
  T.Both(~Unit, ~TLP, ~M(B(), C()), ~T.Two<M(B(), C()), M(N(), C())>, ~M(N(), C()), ~T.Two<M(N(), C()), M(N(), C())>, ~Unit, ~TLS,
    ~T.Tile(B(), TK(), C()), ~TL)

def MP() -> Type:
  T.Two<M1P, FmaP>

def MS() -> Type:
  T.Two<M1S, FmaS()>

def Mod() -> T.Layer<MP, MX, M(N(), C()), MS>:
  T.Seq(~M1P, ~FmaP, ~MX, ~X3(), ~M(N(), C()), ~M1S, ~FmaS(), ~M1, ~Fma())

def mod_p() -> MP:
  T.Two{T.Two{Unit{}, T.Two{Unit{}, T.LN{mat(1n, C(), 61n), mat(1n, C(), 67n)}}}, unitsF}

# ---- block: (cond, h) -> h + Tile(g) * f(mod) ----

def HDup() -> T.Layer<T.Two<Unit, Unit>, M(N(), C()), T.Two<M(N(), C()), M(N(), C())>, T.Two<Unit, Unit>>:
  T.Fork(~Unit, ~Unit, ~M(N(), C()), ~M(N(), C()), ~M(N(), C()), ~Unit, ~Unit,
    ~(m => dup(m)), ~(a => b => add(a, b)), ~T.Id(~M(N(), C())), ~T.Id(~M(N(), C())))

def BX() -> Type:
  T.Two<M(B(), CD()), M(N(), C())>

def B1Y() -> Type:
  T.Two<C3, T.Two<M(N(), C()), M(N(), C())>>

def B1P() -> Type:
  T.Two<KP, T.Two<Unit, Unit>>

def B1S() -> Type:
  T.Two<KS, T.Two<Unit, Unit>>

def B1() -> T.Layer<B1P, BX, B1Y, B1S>:
  T.Both(~KP, ~T.Two<Unit, Unit>, ~M(B(), CD()), ~M(N(), C()), ~C3, ~T.Two<M(N(), C()), M(N(), C())>, ~KS, ~T.Two<Unit, Unit>, ~K, ~HDup())

def B2X() -> Type:
  T.Two<M(B(), C()), T.Two<M(N(), C()), MX>>

def rw3(g: M(B(), C()), m: M(B(), C()), sh: M(B(), C()), hh: T.Two<M(N(), C()), M(N(), C())>) -> B2X:
  match hh:
    case T.Two{h1, h2}:
      T.Two{g, T.Two{h2, T.Two{m, T.Two{sh, h1}}}}

def rw2(g: M(B(), C()), r: T.Two<M(B(), C()), M(B(), C())>, hh: T.Two<M(N(), C()), M(N(), C())>) -> B2X:
  match r:
    case T.Two{m, sh}:
      rw3(g, m, sh, hh)

def rw1(c: C3, hh: T.Two<M(N(), C()), M(N(), C())>) -> B2X:
  match c:
    case T.Two{g, r}:
      rw2(g, r, hh)

def rw(x: B1Y) -> B2X:
  match x:
    case T.Two{c, hh}:
      rw1(c, hh)

def un3(g: M(B(), C()), h2: M(N(), C()), m: M(B(), C()), r: T.Two<M(B(), C()), M(N(), C())>) -> B1Y:
  match r:
    case T.Two{sh, h1}:
      T.Two{T.Two{g, T.Two{m, sh}}, T.Two{h1, h2}}

def un2(g: M(B(), C()), h2: M(N(), C()), r: MX) -> B1Y:
  match r:
    case T.Two{m, r2}:
      un3(g, h2, m, r2)

def un1(g: M(B(), C()), r: T.Two<M(N(), C()), MX>) -> B1Y:
  match r:
    case T.Two{h2, r2}:
      un2(g, h2, r2)

def un(x: B2X) -> B1Y:
  match x:
    case T.Two{g, r}:
      un1(g, r)

def MFP() -> Type:
  T.Two<MP, AttnP()>

def MFS() -> Type:
  T.Two<MS, AttnState()>

def MF() -> T.Layer<MFP, MX, M(N(), C()), MFS>:
  T.Seq(~MP, ~AttnP(), ~MX, ~M(N(), C()), ~M(N(), C()), ~MS, ~AttnState(), ~Mod, ~AttnF())

def HMP() -> Type:
  T.Two<Unit, MFP>

def HMS() -> Type:
  T.Two<Unit, MFS>

def HM() -> T.Layer<HMP, T.Two<M(N(), C()), MX>, T.Two<M(N(), C()), M(N(), C())>, HMS>:
  T.Both(~Unit, ~MFP, ~M(N(), C()), ~MX, ~M(N(), C()), ~M(N(), C()), ~Unit, ~MFS, ~T.Id(~M(N(), C())), ~MF)

def B2P() -> Type:
  T.Two<Unit, HMP>

def B2S() -> Type:
  T.Two<Unit, HMS>

def B2() -> T.Layer<B2P, B2X, X3(), B2S>:
  T.Both(~Unit, ~HMP, ~M(B(), C()), ~T.Two<M(N(), C()), MX>, ~M(N(), C()), ~T.Two<M(N(), C()), M(N(), C())>, ~Unit, ~HMS, ~T.Tile(B(), TK(), C()), ~HM)

def B3P() -> Type:
  T.Two<B2P, FmaP>

def B3S() -> Type:
  T.Two<B2S, FmaS()>

def B3() -> T.Layer<B3P, B2X, M(N(), C()), B3S>:
  T.Seq(~B2P, ~FmaP, ~B2X, ~X3(), ~M(N(), C()), ~B2S, ~FmaS(), ~B2, ~Fma())

def B4P() -> Type:
  T.Two<Unit, B3P>

def B4S() -> Type:
  T.Two<Unit, B3S>

def B4() -> T.Layer<B4P, B1Y, M(N(), C()), B4S>:
  T.Seq(~Unit, ~B3P, ~B1Y, ~B2X, ~M(N(), C()), ~Unit, ~B3S, ~Rewire(~B1Y, ~B2X, ~(x => rw(x)), ~(x => un(x))), ~B3)

def BP() -> Type:
  T.Two<B1P, B4P>

def BS() -> Type:
  T.Two<B1S, B4S>

def Block() -> T.Layer<BP, BX, M(N(), C()), BS>:
  T.Seq(~B1P, ~B4P, ~BX, ~B1Y, ~M(N(), C()), ~B1S, ~B4S, ~B1, ~B4)

def block_p() -> BP:
  T.Two{T.Two{T.Two{Unit{}, T.Two{lin(CD(), C(), 7n), T.Two{lin(CD(), C(), 9n), lin(CD(), C(), 11n)}}}, T.Two{Unit{}, Unit{}}},
    T.Two{Unit{}, T.Two{T.Two{Unit{}, T.Two{Unit{}, T.Two{mod_p, attn_p()}}}, unitsF}}}

def show_grads(g: BP) -> IO(Unit):
  match g:
    case T.Two{b1, b4}:
      match b1:
        case T.Two{kp, u}:
          match kp:
            case T.Two{u2, k3}:
              match k3:
                case T.Two{wg, r}:
                  match r:
                    case T.Two{wm, wsh}:
                      do IO<Unit>:
                        Unit <- show_lin(CD(), C(), "wide.g.dw", wg)
                        Unit <- show_lin(CD(), C(), "wide.m.dw", wm)
                        show_lin(CD(), C(), "wide.sh.dw", wsh)

def show_dx(dx: BX) -> IO(Unit):
  match dx:
    case T.Two{dc, dh}:
      do IO<Unit>:
        Unit <- T.Mat.print(B(), CD(), "wide.dcond", dc)
        T.Mat.print(N(), C(), "wide.dh", dh)

def show_bwd(r: T.Bwd<BP, BX>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- show_dx(dx)
        show_grads(g)

def show_fwd(r: T.Fwd<BP, M(N(), C()), BS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(N(), C(), "wide.y", y)
        show_bwd(T.Layer.bwd(BP, BX, M(N(), C()), BS, Block, p, s, mat(N(), C(), 97n), T.Layer.zeros(BP, BX, M(N(), C()), BS, Block)))

def main() -> IO(Unit):
  show_fwd(T.Layer.fwd(BP, BX, M(N(), C()), BS, Block, block_p,
    T.Two{mat2(B(), CD(), 23n), mat2(N(), C(), 33n)}))
