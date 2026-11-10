# The sparse mask at 64 positions: 16/16, 32/32 and the 8/8 can-fail.
import Base
import bendtorch@2.0.0.0/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./sp_dense.bend as SD

def ml(ml: TA.ML<64n, 64n>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

def show(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show(t)))

def show_mask(+tag: String, m: TA.Mat<64n, 64n>) -> IO(Unit):
  IO.print(String.append(tag, String.append(": ", show(ml(TA.Mat.to_list(64n, 64n, m))))))

def main() -> IO(Unit):
  do IO<Unit>:
    show_mask("m1616", SD.mask_mat_ws(64n, 16n, 16n))
    show_mask("m3232", SD.mask_mat_ws(64n, 32n, 32n))
    show_mask("m88", SD.mask_mat_ws(64n, 8n, 8n))
