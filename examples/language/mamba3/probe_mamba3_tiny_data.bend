# The tiny-LM data: read the byte corpus back through Bend and print the
# first tokens of each split.  The gate is `reference/test_mamba3_tiny_lm.py`
# (it also checks the manifest hashes against the files).
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./mamba3_siso.bend as M

def ddir() -> String:
  ".scratch/mamba3_tiny"

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{b, t}:
      Con{U32.to_nat(b), nats(t)}

def main() -> IO(Unit):
  do IO<Unit>:
    tr : List<&2, U32> <- T.read_at(String.append(ddir(), "/train_tokens.bin"), 0, 8192)
    ho : List<&2, U32> <- T.read_at(String.append(ddir(), "/heldout_tokens.bin"), 0, 4096)
    M.show_tag("tiny_train_tokens", M.nat_row_f32(16n, nats(tr), Nil{}))
    M.show_tag("tiny_heldout_tokens", M.nat_row_f32(16n, nats(ho), Nil{}))
    return Unit{}
