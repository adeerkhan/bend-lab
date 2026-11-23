# DiT denoiser on the digits, first stage: the model builds and runs.
#
# Unlike the ViT task, which bakes the timestep into each token's features, the
# DiT block needs a separate per-image condition vector. This model takes
#   X = (cond: Mat<B, CD>, hfeat: Mat<N, FE>)
# embeds the token features, runs the verified conditioned DiT block from
# wide_block.bend, and applies an output head. The patch/timestep data adapter
# and the training loop are added once this forward check is green.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import bendtorch@2.1.0.0/main.bend as T
import ./wide_block.bend as W

def FE() -> Nat:
  8n

def PX() -> Nat:
  4n

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

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

# ---- model: (cond, token features) -> per-patch noise ----

def EmbP() -> Type:
  M(W.C(), FE())

def HeadP() -> Type:
  M(PX(), W.C())

def ModelP() -> Type:
  T.Two<T.Two<T.Two<Unit, EmbP>, W.BP>, HeadP>

def ModelS() -> Type:
  T.Two<T.Two<T.Two<Unit, M(W.N(), FE())>, W.BS>, M(W.N(), W.C())>

def X() -> Type:
  T.Two<M(W.B(), W.CD()), M(W.N(), FE())>

def Y() -> Type:
  M(W.N(), PX())

def Embed() -> T.Layer<EmbP, M(W.N(), FE()), M(W.N(), W.C()), M(W.N(), FE())>:
  T.LinearNB(W.N(), FE(), W.C(), 0n)

def Pre() -> T.Layer<T.Two<Unit, EmbP>, X(), W.BX, T.Two<Unit, M(W.N(), FE())>>:
  T.Both(~Unit, ~EmbP, ~M(W.B(), W.CD()), ~M(W.N(), FE()), ~M(W.B(), W.CD()), ~M(W.N(), W.C()),
    ~Unit, ~M(W.N(), FE()), ~T.Id(~M(W.B(), W.CD())), ~Embed())

def Body() -> T.Layer<T.Two<T.Two<Unit, EmbP>, W.BP>, X(), M(W.N(), W.C()), T.Two<T.Two<Unit, M(W.N(), FE())>, W.BS>>:
  T.Seq(~T.Two<Unit, EmbP>, ~W.BP, ~X(), ~W.BX, ~M(W.N(), W.C()),
    ~T.Two<Unit, M(W.N(), FE())>, ~W.BS, ~Pre(), ~W.Block())

def Head() -> T.Layer<HeadP, M(W.N(), W.C()), Y, M(W.N(), W.C())>:
  T.LinearNB(W.N(), W.C(), PX(), 0n)

def Model() -> T.Layer<ModelP, X(), Y, ModelS>:
  T.Seq(~T.Two<T.Two<Unit, EmbP>, W.BP>, ~HeadP, ~X(), ~M(W.N(), W.C()), ~Y,
    ~T.Two<T.Two<Unit, M(W.N(), FE())>, W.BS>, ~M(W.N(), W.C()), ~Body(), ~Head())

def params() -> ModelP:
  T.Two{T.Two{T.Two{Unit{}, mat(W.C(), FE(), 3n)}, W.block_p()}, mat(PX(), W.C(), 97n)}

# ---- display ----

def show_y(r: T.Fwd<ModelP, Y, ModelS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(W.N(), PX(), "model.y", y)
        IO.print(String.append("params ", U32.show(T.Layer.count(ModelP(), X(), Y, ModelS(), Model()))))

def main() -> IO(Unit):
  show_y(T.Layer.fwd(ModelP(), X(), Y, ModelS(), Model(), params,
    T.Two{mat2(W.B(), W.CD(), 23n), mat(W.N(), FE(), 33n)}), )
