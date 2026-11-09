# Load the four-block hybrid parameters.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_h.bend as M

def P() -> Type:
  M.P()

def BlockP() -> Type:
  M.BlockP()

def load_block(+dir: String) -> IO(BlockP()):
  do IO<BlockP()>:
    ln1 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(dir, "/ln1.bin"), 0)
    qkv : TA.Mat<96n, 32n> <- T.Mat.load(96n, 32n, String.append(dir, "/qkv.bin"), 0)
    out : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, String.append(dir, "/out.bin"), 0)
    ln2 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(dir, "/ln2.bin"), 0)
    gate : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, String.append(dir, "/gate.bin"), 0)
    up : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, String.append(dir, "/up.bin"), 0)
    down : TA.Mat<32n, 128n> <- T.Mat.load(32n, 128n, String.append(dir, "/down.bin"), 0)
    return T.Two{
      T.Two{T.Two{Unit{}, T.Two{ln1, T.Two{qkv, T.Two{Unit{}, T.Two{Unit{}, out}}}}}, Unit{}},
      T.Two{T.Two{Unit{}, T.Two{ln2, T.Two{T.Two{T.Two{T.Two{gate, Unit{}}, up}, Unit{}}, down}}}, Unit{}}}

def load_params(+dir: String) -> IO(P()):
  do IO<P()>:
    emb : TA.Mat<260n, 32n> <- T.Mat.load(260n, 32n, String.append(dir, "/emb.bin"), 0)
    b0 : BlockP() <- load_block(String.append(dir, "/b0"))
    b1 : BlockP() <- load_block(String.append(dir, "/b1"))
    b2 : BlockP() <- load_block(String.append(dir, "/b2"))
    b3 : BlockP() <- load_block(String.append(dir, "/b3"))
    lnf : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(dir, "/lnf.bin"), 0)
    return T.Two{emb, T.Two{T.Two{b0, T.Two{b1, T.Two{b2, b3}}}, lnf}}
