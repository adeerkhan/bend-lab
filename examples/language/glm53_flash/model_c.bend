# The integrated Stage C model: one batch, 64 positions, one head of width 8,
# two blocks (linear attention, then sparse attention), each with a SwiGLU
# hidden 32 MLP, a final RMSNorm and the tied head.  The MoE and the
# HyperConnection are not part of this stage.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./attn_c.bend as C

def ROWS() -> Nat:
  64n

def DIM() -> Nat:
  8n

def HID() -> Nat:
  32n

def VOCAB() -> Nat:
  260n

def EPS() -> F32:
  0.000001

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def A(c: Nat) -> Type:
  TA.Mat<ROWS(), c>

# ---- SwiGLU ----

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

# ---- one block, two attention kinds ----

def NAttP() -> Type:
  T.Two<M(1n, DIM()), C.AttP()>

def LinNAttS() -> Type:
  T.Two<A(DIM()), C.LinAttS()>

def SpNAttS() -> Type:
  T.Two<A(DIM()), C.SpAttS()>

def LinNAtt() -> T.Layer<NAttP(), A(DIM()), A(DIM()), LinNAttS()>:
  T.Seq(~M(1n, DIM()), ~C.AttP(), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~C.LinAttS(),
    ~T.RMSNorm(ROWS(), DIM(), EPS()), ~C.LinAttention())

def SpNAtt() -> T.Layer<NAttP(), A(DIM()), A(DIM()), SpNAttS()>:
  T.Seq(~M(1n, DIM()), ~C.AttP(), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~C.SpAttS(),
    ~T.RMSNorm(ROWS(), DIM(), EPS()), ~C.SpAttention())

def BlockP() -> Type:
  T.Two<T.ResidualType(NAttP()), T.ResidualType(NMlpP())>

def LinBlockS() -> Type:
  T.Two<T.ResidualType(LinNAttS()), T.ResidualType(NMlpS())>

def SpBlockS() -> Type:
  T.Two<T.ResidualType(SpNAttS()), T.ResidualType(NMlpS())>

def LinBlock() -> T.Layer<BlockP(), A(DIM()), A(DIM()), LinBlockS()>:
  T.Seq(~T.ResidualType(NAttP()), ~T.ResidualType(NMlpP()), ~A(DIM()), ~A(DIM()), ~A(DIM()),
    ~T.ResidualType(LinNAttS()), ~T.ResidualType(NMlpS()),
    ~T.Residual(ROWS(), DIM(), NAttP(), LinNAttS(), LinNAtt()),
    ~T.Residual(ROWS(), DIM(), NMlpP(), NMlpS(), NMlp()))

def SpBlock() -> T.Layer<BlockP(), A(DIM()), A(DIM()), SpBlockS()>:
  T.Seq(~T.ResidualType(NAttP()), ~T.ResidualType(NMlpP()), ~A(DIM()), ~A(DIM()), ~A(DIM()),
    ~T.ResidualType(SpNAttS()), ~T.ResidualType(NMlpS()),
    ~T.Residual(ROWS(), DIM(), NAttP(), SpNAttS(), SpNAtt()),
    ~T.Residual(ROWS(), DIM(), NMlpP(), NMlpS(), NMlp()))

# ---- the model ----

def BodyP() -> Type:
  T.Two<T.Two<BlockP(), BlockP()>, M(1n, DIM())>

def BodyS() -> Type:
  T.Two<T.Two<LinBlockS(), SpBlockS()>, A(DIM())>

def Body() -> T.Layer<BodyP(), A(DIM()), A(DIM()), BodyS()>:
  T.Seq(~T.Two<BlockP(), BlockP()>, ~M(1n, DIM()), ~A(DIM()), ~A(DIM()), ~A(DIM()),
    ~T.Two<LinBlockS(), SpBlockS()>, ~A(DIM()),
    ~T.Seq(~BlockP(), ~BlockP(), ~A(DIM()), ~A(DIM()), ~A(DIM()), ~LinBlockS(), ~SpBlockS(),
      ~LinBlock(), ~SpBlock()),
    ~T.RMSNorm(ROWS(), DIM(), EPS()))

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
