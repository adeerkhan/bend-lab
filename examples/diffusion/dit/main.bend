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

# ---- training: MSE against target noise, AdamW ----

def mse(y: Y, tgt: Y) -> T.LossOut<Y>:
  T.MSE(W.N(), PX(), y, tgt)

def cfg() -> T.OptCfg:
  T.CAdam{0.003, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

def new_train() -> T.Train<ModelP>:
  T.Train.new(~ModelP(), ~X(), ~Y(), ~ModelS(), ~Model(), cfg(), params)

def step_once(tr: T.Train<ModelP>, x: X, tgt: Y) -> T.Train<ModelP>:
  T.Train.step(~ModelP(), ~X(), ~Y(), ~ModelS(), ~Y(), ~Model(), ~mse, tr, x, tgt)

def loss_of(tr: T.Train<ModelP>) -> F32:
  match tr:
    case T.Train{p, g, st, c, p1, p2, +l}:
      l

def train_go(left: Nat, +i: Nat, tr: T.Train<ModelP>) -> IO(T.Train<ModelP>):
  match left tr:
    case 0n T.Train{p, g, st, c, p1, p2, l}:
      IO.pure(T.Train<ModelP>, T.Train{p, g, st, c, p1, p2, l})
    case 1n+rest T.Train{p, g, st, c, p1, p2, +l}:
      do IO<T.Train<ModelP>>:
        Unit <- IO.print(String.append("loss: ", F32.show(l)))
        x : X = T.Two{mat2(W.B(), W.CD(), 23n), mat(W.N(), FE(), 33n)}
        tgt : Y = mat(W.N(), PX(), 97n)
        train_go(rest, (1n + i : Nat), step_once(T.Train{p, g, st, c, p1, p2, l}, x, tgt))

def show_y(r: T.Fwd<ModelP, Y, ModelS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(W.N(), PX(), "model.y", y)
        IO.print(String.append("params ", U32.show(T.Layer.count(ModelP(), X(), Y, ModelS(), Model()))))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- show_y(T.Layer.fwd(ModelP(), X(), Y, ModelS(), Model(), params,
      T.Two{mat2(W.B(), W.CD(), 23n), mat(W.N(), FE(), 33n)}))
    tr : T.Train<ModelP> <- train_go(6n, 0n, new_train())
    IO.print(String.append("final_loss: ", F32.show(loss_of(tr))))
