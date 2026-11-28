# Parity probe for the local four-token depthwise convolution.
#
#   reference/.venv/bin/python reference/test_short_conv.py
#
# The checker writes x, w0..w3 and dy under /tmp/opencode/short_conv/ and
# compares y, dx and dw0..dw3 with PyTorch conv1d(groups=D, padding=3).
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import bendtorch@2.1.0.0/main.bend as T
import ./short_conv.bend as C

def B() -> Nat:
  2n

def SL() -> Nat:
  5n

def D() -> Nat:
  3n

def N() -> Nat:
  Nat.mul(B(), SL())

def P() -> String:
  "/tmp/opencode/short_conv/"

def bwd_bind(-Z: Type, b: C.ConvBwd(B(), SL(), D()),
  k: TA.Mat<N(), D()> -> TA.Mat<1n, D()> -> TA.Mat<1n, D()> ->
     TA.Mat<1n, D()> -> TA.Mat<1n, D()> -> Z) -> Z:
  match b:
    case T.Two{T.Two{dx, T.Two{dw0, dw1}}, T.Two{dw2, dw3}}:
      k(dx, dw0, dw1, dw2, dw3)

def main() -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<N(), D()> <- T.Mat.load(N(), D(), String.append(P(), "x.bin"), 0)
    w0a : TA.Mat<1n, D()> <- T.Mat.load(1n, D(), String.append(P(), "w0.bin"), 0)
    w1a : TA.Mat<1n, D()> <- T.Mat.load(1n, D(), String.append(P(), "w1.bin"), 0)
    w2a : TA.Mat<1n, D()> <- T.Mat.load(1n, D(), String.append(P(), "w2.bin"), 0)
    w3a : TA.Mat<1n, D()> <- T.Mat.load(1n, D(), String.append(P(), "w3.bin"), 0)
    w0b : TA.Mat<1n, D()> <- T.Mat.load(1n, D(), String.append(P(), "w0.bin"), 0)
    w1b : TA.Mat<1n, D()> <- T.Mat.load(1n, D(), String.append(P(), "w1.bin"), 0)
    w2b : TA.Mat<1n, D()> <- T.Mat.load(1n, D(), String.append(P(), "w2.bin"), 0)
    w3b : TA.Mat<1n, D()> <- T.Mat.load(1n, D(), String.append(P(), "w3.bin"), 0)
    dy : TA.Mat<N(), D()> <- T.Mat.load(N(), D(), String.append(P(), "dy.bin"), 0)
    C.dup_bind(IO(Unit), N(), D(), x, xf => xb =>
      do IO<Unit>:
        y : TA.Mat<N(), D()> <- IO.pure(TA.Mat<N(), D()>,
          C.conv_fwd(B(), SL(), D(), w0a, w1a, w2a, w3a, xf))
        yp : Unit <- T.Mat.print(N(), D(), "y", y)
        bwd_bind(IO(Unit), C.conv_bwd(B(), SL(), D(), w0b, w1b, w2b, w3b, xb, dy),
          dx => dw0 => dw1 => dw2 => dw3 =>
            do IO<Unit>:
              dxp : Unit <- T.Mat.print(N(), D(), "dx", dx)
              d0p : Unit <- T.Mat.print(1n, D(), "dw0", dw0)
              d1p : Unit <- T.Mat.print(1n, D(), "dw1", dw1)
              d2p : Unit <- T.Mat.print(1n, D(), "dw2", dw2)
              T.Mat.print(1n, D(), "dw3", dw3)))
