# The Stage B backward at the smallest scale: t = 2, one head, hd = 8.
# Prints the forward and the three gradients for a general dout, then dv/dq/dk
# for a one-hot dout at t = 0 (where the t > 0 dv must be zero).
import Base
import ./lin_bwd.bend as L

def qv() -> List<&2, F32>:
  Con{0.1, Con{(0.0 - 0.2 : F32), Con{0.3, Con{(0.0 - 0.4 : F32),
  Con{0.5, Con{(0.0 - 0.6 : F32), Con{0.7, Con{(0.0 - 0.8 : F32),
  Con{0.2, Con{0.4, Con{(0.0 - 0.1 : F32), Con{0.6,
  Con{(0.0 - 0.3 : F32), Con{0.8, Con{(0.0 - 0.5 : F32), Con{0.9, Nil{}}}}}}}}}}}}}}}}}

def kv() -> List<&2, F32>:
  Con{0.3, Con{(0.0 - 0.1 : F32), Con{0.2, Con{0.5,
  Con{(0.0 - 0.4 : F32), Con{0.1, Con{(0.0 - 0.2 : F32), Con{0.6,
  Con{(0.0 - 0.2 : F32), Con{0.7, Con{0.1, Con{(0.0 - 0.3 : F32),
  Con{0.4, Con{(0.0 - 0.6 : F32), Con{0.2, Con{0.5, Nil{}}}}}}}}}}}}}}}}}


def vv() -> List<&2, F32>:
  Con{1.0, Con{(0.0 - 2.0 : F32), Con{0.5, Con{0.25,
  Con{(0.0 - 0.75 : F32), Con{1.5, Con{(0.0 - 0.5 : F32), Con{2.0,
  Con{0.5, Con{0.25, Con{(0.0 - 1.0 : F32), Con{2.0,
  Con{(0.0 - 0.25 : F32), Con{0.75, Con{1.25, Con{(0.0 - 1.5 : F32), Nil{}}}}}}}}}}}}}}}}}


def dya() -> List<&2, F32>:
  Con{0.1, Con{0.2, Con{(0.0 - 0.1 : F32), Con{0.3,
  Con{(0.0 - 0.2 : F32), Con{0.4, Con{(0.0 - 0.3 : F32), Con{0.5,
  Con{(0.0 - 0.1 : F32), Con{0.3, Con{0.2, Con{(0.0 - 0.4 : F32),
  Con{0.5, Con{(0.0 - 0.2 : F32), Con{0.1, Con{(0.0 - 0.3 : F32), Nil{}}}}}}}}}}}}}}}}}


def dyb() -> List<&2, F32>:
  Con{1.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0,
  Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Con{0.0, Nil{}}}}}}}}}}}}}}}}}


def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

def show_bw(+tag: String, bw: L.BW) -> IO(Unit):
  match bw:
    case L.BW{dq, dk, dv}:
      do IO<Unit>:
        IO.print(String.append(String.append(tag, "_dq: "), show(dq)))
        IO.print(String.append(String.append(tag, "_dk: "), show(dk)))
        IO.print(String.append(String.append(tag, "_dv: "), show(dv)))
        return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("y_a: ", show(L.lin_y(2n, 8n, qv(), kv(), vv()))))
    show_bw("a", L.lin_bwd(2n, 8n, qv(), kv(), vv(), dya()))
    show_bw("b", L.lin_bwd(2n, 8n, qv(), kv(), vv(), dyb()))
