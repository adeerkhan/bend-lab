# The four-block hybrid at 8 positions, for a forward check: three linear blocks and one sparse
# (window 16, stride 16), each with a SwiGLU 128, final RMSNorm, tied head.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./attn_mh8.bend as A

def ROWS() -> Nat:
  A.N()

def DIM() -> Nat:
  A.D()

def HID() -> Nat:
  128n

def VOCAB() -> Nat:
  260n

def EPS() -> F32:
  0.000001

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def A(c: Nat) -> Type:
  TA.Mat<ROWS(), c>

def AttP() -> Type:
  A.AttP()

def NAttP() -> Type:
  T.Two<M(1n, DIM()), AttP()>

def LinNAttS() -> Type:
  T.Two<A(DIM()), A.LinAttS()>

def LinNAtt() -> T.Layer<NAttP(), A(DIM()), A(DIM()), LinNAttS()>:
  T.Seq(~M(1n, DIM()), ~AttP(), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~A.LinAttS(),
    ~T.RMSNorm(ROWS(), DIM(), EPS()), ~A.LinAttention())

def SpNAtt() -> T.Layer<NAttP(), A(DIM()), A(DIM()), LinNAttS()>:
  T.Seq(~M(1n, DIM()), ~AttP(), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~A.LinAttS(),
    ~T.RMSNorm(ROWS(), DIM(), EPS()), ~A.SpAttention())

# ---- SwiGLU 128 ----

def GateP() -> Type:
  T.Two<M(HID(), DIM()), Unit>

def GateS() -> Type:
  T.Two<A(DIM()), A(HID())>

def Gate() -> T.Layer<GateP(), A(DIM()), A(HID()), GateS()>:
  T.Seq(~M(HID(), DIM()), ~Unit, ~A(DIM()), ~A(HID()), ~A(HID()), ~A(DIM()), ~A(HID()),
    ~T.LinearNB(ROWS(), DIM(), HID(), 0n), ~T.SiLU(ROWS(), HID()))

def GUP() -> Type:
  T.Two<GateP(), M(HID(), DIM())>

def GUY() -> Type:
  T.Two<A(HID()), A(HID())>

def GUS() -> Type:
  T.Two<GateS(), A(DIM())>

def GU() -> T.Layer<GUP(), A(DIM()), GUY(), GUS()>:
  T.Fork(~T.Two<M(HID(), DIM()), Unit>, ~M(HID(), DIM()), ~A(DIM()), ~A(HID()), ~A(HID()),
    ~GateS(), ~A(DIM()),
    ~(m => T.Mat.dup(ROWS(), DIM(), m)), ~(x => y => T.Mat.sum2(ROWS(), DIM(), x, y)),
    ~Gate(), ~T.LinearNB(ROWS(), DIM(), HID(), 0n))

def GLUS() -> Type:
  T.Two<GUS(), GUY()>

def GLU() -> T.Layer<T.Two<GUP(), Unit>, A(DIM()), A(HID()), GLUS()>:
  T.Seq(~GUP(), ~Unit, ~A(DIM()), ~GUY(), ~A(HID()), ~GUS(), ~GUY(),
    ~GU(), ~T.Mul2(ROWS(), HID()))

def MlpP() -> Type:
  T.Pair3(GUP(), Unit, M(DIM(), HID()))

def MlpS() -> Type:
  T.Two<GLUS(), A(HID())>

def Mlp() -> T.Layer<MlpP(), A(DIM()), A(DIM()), MlpS()>:
  T.Seq(~T.Two<GUP(), Unit>, ~M(DIM(), HID()), ~A(DIM()), ~A(HID()), ~A(DIM()),
    ~GLUS(), ~A(HID()), ~GLU(), ~T.LinearNB(ROWS(), HID(), DIM(), 0n))

def NMlpP() -> Type:
  T.Two<M(1n, DIM()), MlpP()>

def NMlpS() -> Type:
  T.Two<A(DIM()), MlpS()>

def NMlp() -> T.Layer<NMlpP(), A(DIM()), A(DIM()), NMlpS()>:
  T.Seq(~M(1n, DIM()), ~MlpP(), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~MlpS(),
    ~T.RMSNorm(ROWS(), DIM(), EPS()), ~Mlp())

# ---- the blocks ----

def BlockP() -> Type:
  T.Two<T.ResidualType(NAttP()), T.ResidualType(NMlpP())>

def LinBlockS() -> Type:
  T.Two<T.ResidualType(LinNAttS()), T.ResidualType(NMlpS())>

def SpBlockS() -> Type:
  T.Two<T.ResidualType(LinNAttS()), T.ResidualType(NMlpS())>

def LinBlock() -> T.Layer<BlockP(), A(DIM()), A(DIM()), LinBlockS()>:
  T.Seq(~T.ResidualType(NAttP()), ~T.ResidualType(NMlpP()), ~A(DIM()), ~A(DIM()), ~A(DIM()),
    ~T.ResidualType(LinNAttS()), ~T.ResidualType(NMlpS()),
    ~T.Residual(ROWS(), DIM(), NAttP(), LinNAttS(), LinNAtt()),
    ~T.Residual(ROWS(), DIM(), NMlpP(), NMlpS(), NMlp()))

def SpBlock() -> T.Layer<BlockP(), A(DIM()), A(DIM()), SpBlockS()>:
  T.Seq(~T.ResidualType(NAttP()), ~T.ResidualType(NMlpP()), ~A(DIM()), ~A(DIM()), ~A(DIM()),
    ~T.ResidualType(LinNAttS()), ~T.ResidualType(NMlpS()),
    ~T.Residual(ROWS(), DIM(), NAttP(), LinNAttS(), SpNAtt()),
    ~T.Residual(ROWS(), DIM(), NMlpP(), NMlpS(), NMlp()))

# ---- four blocks, three linear and one sparse ----

def B3P() -> Type:
  T.Two<BlockP(), T.Two<BlockP(), BlockP()>>

def B3S() -> Type:
  T.Two<LinBlockS(), T.Two<LinBlockS(), SpBlockS()>>

def B3() -> T.Layer<B3P(), A(DIM()), A(DIM()), B3S()>:
  T.Seq(~BlockP(), ~T.Two<BlockP(), BlockP>(), ~A(DIM()), ~A(DIM()), ~A(DIM()),
    ~LinBlockS(), ~T.Two<LinBlockS(), SpBlockS>(),
    ~LinBlock(), ~T.Seq(~BlockP(), ~BlockP(), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~LinBlockS(), ~SpBlockS(),
      ~LinBlock(), ~SpBlock()))

def BlocksP() -> Type:
  T.Two<BlockP(), B3P()>

def BlocksS() -> Type:
  T.Two<LinBlockS(), B3S()>

def Blocks() -> T.Layer<BlocksP(), A(DIM()), A(DIM()), BlocksS()>:
  T.Seq(~BlockP(), ~B3P(), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~LinBlockS(), ~B3S(),
    ~LinBlock(), ~B3())

def BodyP() -> Type:
  T.Two<BlocksP(), M(1n, DIM())>

def BodyS() -> Type:
  T.Two<BlocksS(), A(DIM())>

def Body() -> T.Layer<BodyP(), A(DIM()), A(DIM()), BodyS()>:
  T.Seq(~BlocksP(), ~M(1n, DIM()), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~BlocksS(), ~A(DIM()),
    ~Blocks(), ~T.RMSNorm(ROWS(), DIM(), EPS()))

def P() -> Type:
  T.Two<M(VOCAB(), DIM()), BodyP()>

def X() -> Type:
  List<&2, Nat>

def Y() -> Type:
  A(VOCAB())

def S() -> Type:
  T.Three<List<&2, Nat>, BodyS(), A(DIM())>

def Model() -> T.Layer<P(), X(), Y(), S()>:
  T.Tie(~M(VOCAB(), DIM()), ~BodyP(), ~X(), ~A(DIM()), ~A(DIM()), ~Y(),
    ~List<&2, Nat>, ~BodyS(), ~A(DIM()),
    ~T.Embedding(ROWS(), VOCAB(), DIM()), ~Body(),
    ~T.LinearNB(ROWS(), DIM(), VOCAB(), 0n))
