# Boundary probe (B4): A at the -A_floor clamp, trap at 0/1, softplus
# at large/small dd_dt.  Prints the Bend port's values for the ladders;
# the gate compares against the twin at 1e-5/1e-6.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def show_row(+tag: String, +xs: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    M.show_tag(tag, xs)
    return Unit{}

def a_ladder() -> List<&2, F32>:
  Con{(0.0 - 10000.0 : F32), Con{(0.0 - 9999.0 : F32), Con{(0.0 - 1000.1 : F32), Con{(0.0 - 100.0 : F32),
    Con{0.0, Con{1.0, Con{10.0, Nil{}}}}}}}}

def sp_ladder() -> List<&2, F32>:
  Con{(0.0 - 100.0 : F32), Con{(0.0 - 20.0 : F32), Con{(0.0 - 1.0 : F32), Con{0.0,
    Con{1.0, Con{20.0, Con{50.0, Con{88.0, Con{100.0, Nil{}}}}}}}}}}

def trap_ladder() -> List<&2, F32>:
  Con{(0.0 - 1000.0 : F32), Con{(0.0 - 100.0 : F32), Con{(0.0 - 20.0 : F32), Con{0.0,
    Con{20.0, Con{100.0, Con{1000.0, Nil{}}}}}}}}

def neg_htail_row(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{M.neg_htail(h), neg_htail_row(t)}

def sigmoid_row_l(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{M.sigmoid(h), sigmoid_row_l(t)}

def softplus_row_l(+xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{M.softplus(h), softplus_row_l(t)}

def show_pair(+tag: String, +xs: List<&2, F32>,
    +ys: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    M.show_tag(String.append(tag, "_x"), xs)
    M.show_tag(String.append(tag, "_y"), ys)
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    +a_y : List<&2, F32> <- IO.pure(List<&2, F32>, neg_htail_row(a_ladder()))
    +sp_y : List<&2, F32> <- IO.pure(List<&2, F32>, softplus_row_l(sp_ladder()))
    +trap_y : List<&2, F32> <- IO.pure(List<&2, F32>, sigmoid_row_l(trap_ladder()))
    show_pair("clamp_A", a_ladder(), a_y)
    show_pair("clamp_softplus", sp_ladder(), sp_y)
    show_pair("clamp_sigmoid", trap_ladder(), trap_y)
    return Unit{}
