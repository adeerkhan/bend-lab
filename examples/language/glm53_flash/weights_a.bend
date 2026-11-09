# Load the Stage A parameter tree from the twin's files.
#
# The files are little-endian F32, one per matrix, in the layout
# `reference/test_glm53_layers.py` and `reference/glm53_dense_data.py` write;
# the q/k files are already in BendTorch's half-split RoPE order.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./model_a.bend as M

def P() -> Type:
  M.P()

def BlockP() -> Type:
  M.BlockP()

def load_block(+dir: String) -> IO(BlockP()):
  do IO<BlockP()>:
    ln1 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(dir, "/ln1.bin"), 0)
    wq : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, String.append(dir, "/wq.bin"), 0)
    wk : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, String.append(dir, "/wk.bin"), 0)
    wv : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, String.append(dir, "/wv.bin"), 0)
    wo : TA.Mat<32n, 32n> <- T.Mat.load(32n, 32n, String.append(dir, "/wo.bin"), 0)
    ln2 : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(dir, "/ln2.bin"), 0)
    wg : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, String.append(dir, "/wg.bin"), 0)
    wu : TA.Mat<128n, 32n> <- T.Mat.load(128n, 32n, String.append(dir, "/wu.bin"), 0)
    wd : TA.Mat<32n, 128n> <- T.Mat.load(32n, 128n, String.append(dir, "/wd.bin"), 0)
    return T.Two{
      T.Two{T.Two{Unit{}, T.Two{ln1, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{T.Two{Unit{}, T.Two{Unit{}, Unit{}}}, T.Two{Unit{}, wo}}}}}, Unit{}},
      T.Two{T.Two{Unit{}, T.Two{ln2, T.Two{T.Two{T.Two{T.Two{wg, Unit{}}, wu}, Unit{}}, wd}}}, Unit{}}}

def load_params(+dir: String) -> IO(P()):
  do IO<P()>:
    emb : TA.Mat<260n, 32n> <- T.Mat.load(260n, 32n, String.append(dir, "/emb.bin"), 0)
    b0 : BlockP() <- load_block(String.append(dir, "/b0"))
    b1 : BlockP() <- load_block(String.append(dir, "/b1"))
    b2 : BlockP() <- load_block(String.append(dir, "/b2"))
    b3 : BlockP() <- load_block(String.append(dir, "/b3"))
    lnf : TA.Mat<1n, 32n> <- T.Mat.load(1n, 32n, String.append(dir, "/lnf.bin"), 0)
    return T.Two{emb, T.Two{T.Two{T.Two{b0, b1}, T.Two{b2, b3}}, lnf}}
