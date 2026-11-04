# `Train.new(cfg)` builds the optimizer state; `Train` carries no config; and
# `Train.step(+cfg)` builds the optimizer from the config it is handed. So a
# step silently uses the config at the call site and ignores the one the Train
# was made with (ISSUE-037).
#
#   bendenv run -- bend bench/train_cfg/main.bend
#
# A 1 x 4 linear, a fixed input and target, 20 AdamW steps. Three Trains, each
# made with its own lr, all stepped with the SAME config (lr 0.001): the three
# losses are identical to the last digit. Then three Trains made with the same
# lr, stepped with their own: the three losses separate. The first table is the
# bug, the second is the control.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def one() -> Nat:
  1n

def w() -> Nat:
  4n

def X() -> Type:
  TA.Mat<one(), w()>

def Y() -> Type:
  TA.Mat<one(), w()>

def P() -> Type:
  T.Lin<w(), w()>

def Mdl() -> T.Layer<P(), X(), Y(), Y()>:
  T.Linear(one(), w(), w(), 0n)

def mse(y: Y(), t: Y()) -> T.LossOut<Y()>:
  T.MSE(one(), w(), y, t)

def vals() -> List<&2, F32>:
  Con{1.0, Con{2.0, Con{3.0, Con{4.0, Nil{}}}}}

def params() -> P():
  T.Lin{TA.Mat.fill(w(), w(), 0.0), TA.Mat.fill(one(), w(), 0.0)}

def cfg_lr(+lr: F32) -> T.OptCfg:
  T.CAdamW{lr, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

# one step, the config named at the call site the way the examples do it
def step(+lr: F32, tr: T.Train<P()>) -> T.Train<P()>:
  T.Train.step(~P(), ~X(), ~Y(), ~Y(), ~Y(), ~Mdl(), ~mse, tr,
    TA.Mat.from_list(one(), w(), vals()), TA.Mat.from_list(one(), w(), vals()))

def loop(+lr: F32, +left: Nat, tr: T.Train<P()>) -> F32:
  match left tr:
    case 0n T.Train{p, g, st, c, q1, q2, +l}:
      l
    case 1n+rest T.Train{p, g, st, c, q1, q2, +l}:
      loop(lr, rest, step(lr, T.Train{p, g, st, c, q1, q2, l}))

# the Train's own lr is `built`; the step's is `stepped`
def run(+built: F32, +stepped: F32) -> IO(Unit):
  do IO<Unit>:
    tr : T.Train<P()> <- IO.pure(T.Train<P()>,
      T.Train.new(~P(), ~X(), ~Y(), ~Y(), ~Mdl(), cfg_lr(built), params()))
    IO.print(String.append(String.append("built ", F32.show(built)),
      String.append(String.append(String.append(" stepped ", F32.show(stepped)),
        " loss "), F32.show(loop(stepped, 20n, tr)))))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print("the config Train.step is handed decides, whatever Train.new got:")
    Unit <- run(0.001, 0.001)
    Unit <- run(0.01, 0.001)
    Unit <- run(0.1, 0.001)
    Unit <- IO.print("control: the step's config is the one that acts")
    Unit <- run(0.001, 0.001)
    Unit <- run(0.001, 0.01)
    Unit <- run(0.001, 0.1)
    IO.print("done")
    return Unit{}
