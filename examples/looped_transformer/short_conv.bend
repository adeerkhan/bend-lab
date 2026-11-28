# The four-token causal depthwise convolution on a row-major batch matrix.
#
# The reference applies Conv1d(dim, dim, 4, groups=dim, padding=3, bias=False)
# to K and V before attention. The released bendtorch@2.1.0.0 has no grouped
# Conv1d, so this module writes it as a dense patch matrix:
#
#   patches r x (4*d): tap j of row (b, t) is x[b, t - (3 - j)]
#   weights (4*d) x d: block j is the row vector w_j on its diagonal
#   y = patches * weights
#
# The backward is dP = dy * W^T, dW = P^T * dy, then the adjoint of the patch
# map. The parity probe checks y, dx and the four dw rows against PyTorch.
import Base
import bendtorch@2.1.0.0/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA

def row_bind(-Z: Type, +r: Nat, +c: Nat, m: TA.MRow<r, c>,
  k: TA.Mat<r, c> -> List<&2, F32> -> Z) -> Z:
  match m:
    case TA.MRow{m, row}:
      k(m, row)

def dup_two(-Z: Type, +r: Nat, +c: Nat,
  d: T.Two<TA.Mat<r, c>, TA.Mat<r, c>>,
  k: TA.Mat<r, c> -> TA.Mat<r, c> -> Z) -> Z:
  match d:
    case T.Two{a, b}:
      k(a, b)

def dup_bind(-Z: Type, +r: Nat, +c: Nat, m: TA.Mat<r, c>,
  k: TA.Mat<r, c> -> TA.Mat<r, c> -> Z) -> Z:
  dup_two(Z, r, c, T.Mat.dup(r, c, m), k)

def maybe_f32(-Z: Type, m: Maybe<&2, F32>, k: F32 -> Z, d: Unit -> Z) -> Z:
  match m:
    case Some{v}:
      k(v)
    case None{}:
      d(Unit{})

def zlist(+d: Nat) -> List<&2, F32>:
  List.replicate(F32, d, 0.0)

def rows_of(left: Nat, +N: Nat, +d: Nat, +r: U32, x: TA.Mat<N, d>,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match left:
    case 0n:
      List.reverse(&2, List<&2, F32>, acc)
    case 1n+q:
      row_bind(List<&2, List<&2, F32>>, N, d,
        TA.Mat.read_row(N, d, x, r),
        m => row => rows_of(q, N, d, (r + 1 : U32), m, Con{row, acc}))

def maybe_row(-Z: Type, m: Maybe<&2, List<&2, F32>>, k: List<&2, F32> -> Z,
  d: Unit -> Z) -> Z:
  match m:
    case Some{v}:
      k(v)
    case None{}:
      d(Unit{})

def row_list_at(+rows: List<&2, List<&2, F32>>, +i: Nat) -> List<&2, F32>:
  maybe_row(List<&2, F32>, List.get(&2, List<&2, F32>, rows, i), x => x, u => Nil{})

# One patch row: the four taps, oldest first. Positions before zero keep a
# zero tap so every row has the same length.
def patch_row(+rows: List<&2, List<&2, F32>>, +base: U32, +t: Nat,
  +d: Nat) -> List<&2, F32>:
  match t:
    case 0n:
      List.append(&2, F32, zlist(d), List.append(&2, F32, zlist(d),
        List.append(&2, F32, zlist(d), row_list_at(rows, U32.to_nat(base)))))
    case 1n:
      List.append(&2, F32, zlist(d), List.append(&2, F32, zlist(d),
        List.append(&2, F32, row_list_at(rows, U32.to_nat(base)),
          row_list_at(rows, (U32.to_nat(base) + 1n : Nat)))))
    case 2n:
      List.append(&2, F32, zlist(d), List.append(&2, F32,
        row_list_at(rows, U32.to_nat(base)),
        List.append(&2, F32, row_list_at(rows, (U32.to_nat(base) + 1n : Nat)),
          row_list_at(rows, (U32.to_nat(base) + 2n : Nat)))))
    case 3n+q:
      List.append(&2, F32, row_list_at(rows, (U32.to_nat(base) + q : Nat)),
        List.append(&2, F32, row_list_at(rows, ((U32.to_nat(base) + q : Nat) + 1n : Nat)),
          List.append(&2, F32, row_list_at(rows, ((U32.to_nat(base) + q : Nat) + 2n : Nat)),
            row_list_at(rows, ((U32.to_nat(base) + q : Nat) + 3n : Nat)))))

def patch_pos(left: Nat, +N: Nat, +rows: List<&2, List<&2, F32>>,
  +base: U32, +pos: Nat, +d: Nat, p: TA.Mat<N, Nat.mul(4n, d)>) ->
  TA.Mat<N, Nat.mul(4n, d)>:
  match left:
    case 0n:
      p
    case 1n+q:
      patch_pos(q, N, rows, base, (pos + 1n : Nat), d,
        TA.Mat.write_row(N, Nat.mul(4n, d), p, (base + U32.from_nat(pos) : U32),
          patch_row(rows, base, pos, d)))

def patch_batch(left: Nat, +N: Nat, +rows: List<&2, List<&2, F32>>,
  +base: U32, +t: Nat, +d: Nat, p: TA.Mat<N, Nat.mul(4n, d)>) ->
  TA.Mat<N, Nat.mul(4n, d)>:
  match left:
    case 0n:
      p
    case 1n+q:
      patch_batch(q, N, rows, (base + U32.from_nat(t) : U32), t, d,
        patch_pos(t, N, rows, base, 0n, d, p))

def patch(+b: Nat, +t: Nat, +d: Nat,
  x: TA.Mat<Nat.mul(b, t), d>) -> TA.Mat<Nat.mul(b, t), Nat.mul(4n, d)>:
  rows = rows_of(Nat.mul(b, t), Nat.mul(b, t), d, 0, x, Nil{})
  patch_batch(b, Nat.mul(b, t), rows, 0, t, d,
    TA.Mat.zeros(Nat.mul(b, t), Nat.mul(4n, d)))

def wlist(+d: Nat, w: TA.Mat<1n, d>) -> List<&2, F32>:
  row_bind(List<&2, F32>, 1n, d, TA.Mat.read_row(1n, d, w, 0), m => row => row)

def wrow(+d: Nat, +j: Nat, +c: Nat, +l0: List<&2, F32>, +l1: List<&2, F32>,
  +l2: List<&2, F32>, +l3: List<&2, F32>) -> List<&2, F32>:
  match j:
    case 0n:
      List.set(&2, F32, zlist(d), c,
        maybe_f32(F32, List.get(&2, F32, l0, c), v => v, u => 0.0))
    case 1n:
      List.set(&2, F32, zlist(d), c,
        maybe_f32(F32, List.get(&2, F32, l1, c), v => v, u => 0.0))
    case 2n:
      List.set(&2, F32, zlist(d), c,
        maybe_f32(F32, List.get(&2, F32, l2, c), v => v, u => 0.0))
    case 3n+q:
      List.set(&2, F32, zlist(d), c,
        maybe_f32(F32, List.get(&2, F32, l3, c), v => v, u => 0.0))

def weight_go(left: Nat, +r: Nat, +d: Nat, +l0: List<&2, F32>, +l1: List<&2, F32>,
  +l2: List<&2, F32>, +l3: List<&2, F32>,
  w: TA.Mat<Nat.mul(4n, d), d>) -> TA.Mat<Nat.mul(4n, d), d>:
  match left:
    case 0n:
      w
    case 1n+q:
      weight_go(q, (r + 1n : Nat), d, l0, l1, l2, l3,
        TA.Mat.write_row(Nat.mul(4n, d), d, w, U32.from_nat(r),
          wrow(d, Nat.div(r, d), Nat.mod(r, d), l0, l1, l2, l3)))

def weight_matrix(+d: Nat, w0: TA.Mat<1n, d>, w1: TA.Mat<1n, d>,
  w2: TA.Mat<1n, d>, w3: TA.Mat<1n, d>) -> TA.Mat<Nat.mul(4n, d), d>:
  weight_go(Nat.mul(4n, d), 0n, d, wlist(d, w0), wlist(d, w1),
    wlist(d, w2), wlist(d, w3), TA.Mat.zeros(Nat.mul(4n, d), d))

def mul_bind(-Z: Type, +n: Nat, +k: Nat, +m: Nat, r: TA.MMul<n, k, m>,
  f: TA.Mat<n, m> -> Z) -> Z:
  match r:
    case TA.MMul{a, b, c}:
      f(c)

def mulnt_bind(-Z: Type, +n: Nat, +k: Nat, +m: Nat, r: TA.MMulNT<n, k, m>,
  f: TA.Mat<n, m> -> Z) -> Z:
  match r:
    case TA.MMulNT{a, b, c}:
      f(c)

def multn_bind(-Z: Type, +n: Nat, +k: Nat, +m: Nat, r: TA.MMulTN<n, k, m>,
  f: TA.Mat<n, m> -> Z) -> Z:
  match r:
    case TA.MMulTN{a, b, c}:
      f(c)

# dW's diagonal block j: rows j*d+c, column c.
def diag_go(left: Nat, +c: Nat, +j: Nat, +d: Nat, dw: TA.Mat<Nat.mul(4n, d), d>,
  acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+q:
      row_bind(List<&2, F32>, Nat.mul(4n, d), d,
        TA.Mat.read_row(Nat.mul(4n, d), d, dw,
          U32.from_nat(Nat.add(Nat.mul(j, d), c))),
        m => row =>
          diag_go(q, (c + 1n : Nat), j, d, m,
            Con{maybe_f32(F32, List.get(&2, F32, row, c), v => v, u => 0.0), acc}))

def diag_block(+j: Nat, +d: Nat, dw: TA.Mat<Nat.mul(4n, d), d>) -> TA.Mat<1n, d>:
  TA.Mat.from_list(1n, d, diag_go(d, 0n, j, d, dw, Nil{}))

# One d x d block of dP's columns j*d .. (j+1)*d-1.
def block_go(left: Nat, +b: Nat, +t: Nat, +r: Nat, +j: Nat, +d: Nat,
  dp: TA.Mat<Nat.mul(b, t), Nat.mul(4n, d)>, out: TA.Mat<Nat.mul(b, t), d>) ->
  TA.Mat<Nat.mul(b, t), d>:
  match left:
    case 0n:
      out
    case 1n+q:
      row_bind(TA.Mat<Nat.mul(b, t), d>, Nat.mul(b, t), Nat.mul(4n, d),
        TA.Mat.read_row(Nat.mul(b, t), Nat.mul(4n, d), dp, U32.from_nat(r)),
        m => row =>
          block_go(q, b, t, (r + 1n : Nat), j, d, m,
            TA.Mat.write_row(Nat.mul(b, t), d, out, U32.from_nat(r),
              List.take(&2, F32, List.drop(&2, F32, row, Nat.mul(j, d)), d))))

def block_cols(+b: Nat, +t: Nat, +j: Nat, +d: Nat,
  dp: TA.Mat<Nat.mul(b, t), Nat.mul(4n, d)>) -> TA.Mat<Nat.mul(b, t), d>:
  block_go(Nat.mul(b, t), b, t, 0n, j, d, dp, TA.Mat.zeros(Nat.mul(b, t), d))

# The adjoint of the patch map: block j is moved back by 3 - j positions.
def unshift_data(left: Nat, +N: Nat, +rows: List<&2, List<&2, F32>>,
  +dst: U32, +src: U32, +d: Nat, acc: TA.Mat<N, d>) -> TA.Mat<N, d>:
  match left:
    case 0n:
      acc
    case 1n+q:
      unshift_data(q, N, rows, (dst + 1 : U32), (src + 1 : U32), d,
        TA.Mat.write_row(N, d, acc, dst, row_list_at(rows, U32.to_nat(src))))

def unshift_batch(left: Nat, +N: Nat, +rows: List<&2, List<&2, F32>>,
  +base: U32, +t: Nat, +d: Nat, +s: Nat, acc: TA.Mat<N, d>) -> TA.Mat<N, d>:
  match left:
    case 0n:
      acc
    case 1n+q:
      unshift_batch(q, N, rows, (base + U32.from_nat(t) : U32), t, d, s,
        unshift_data(Nat.sub(t, s), N, rows, base, (base + U32.from_nat(s) : U32),
          d, acc))

def unshift(+b: Nat, +t: Nat, +d: Nat, +s: Nat,
  x: TA.Mat<Nat.mul(b, t), d>) -> TA.Mat<Nat.mul(b, t), d>:
  rows = rows_of(Nat.mul(b, t), Nat.mul(b, t), d, 0, x, Nil{})
  unshift_batch(b, Nat.mul(b, t), rows, 0, t, d, s,
    TA.Mat.zeros(Nat.mul(b, t), d))

def ConvBwd(+b: Nat, +t: Nat, +d: Nat) -> Type:
  T.Two<T.Two<TA.Mat<Nat.mul(b, t), d>, T.Two<TA.Mat<1n, d>, TA.Mat<1n, d>>>,
    T.Two<TA.Mat<1n, d>, TA.Mat<1n, d>>>

def conv_fwd(+b: Nat, +t: Nat, +d: Nat, w0: TA.Mat<1n, d>, w1: TA.Mat<1n, d>,
  w2: TA.Mat<1n, d>, w3: TA.Mat<1n, d>, x: TA.Mat<Nat.mul(b, t), d>) ->
  TA.Mat<Nat.mul(b, t), d>:
  mul_bind(TA.Mat<Nat.mul(b, t), d>, Nat.mul(b, t), Nat.mul(4n, d), d,
    TA.Mat.matmul(Nat.mul(b, t), Nat.mul(4n, d), d, 0n,
      patch(b, t, d, x), weight_matrix(d, w0, w1, w2, w3)), y => y)

def conv_bwd(+b: Nat, +t: Nat, +d: Nat, w0: TA.Mat<1n, d>, w1: TA.Mat<1n, d>,
  w2: TA.Mat<1n, d>, w3: TA.Mat<1n, d>, x: TA.Mat<Nat.mul(b, t), d>,
  dy: TA.Mat<Nat.mul(b, t), d>) -> ConvBwd(b, t, d):
  dup_bind(ConvBwd(b, t, d), Nat.mul(b, t), d, dy, dy1 => dy2 =>
    mulnt_bind(ConvBwd(b, t, d), Nat.mul(b, t), d, Nat.mul(4n, d),
      TA.Mat.matmul_nt(Nat.mul(b, t), d, Nat.mul(4n, d), 0n, dy1,
        weight_matrix(d, w0, w1, w2, w3)), dp =>
      multn_bind(ConvBwd(b, t, d), Nat.mul(4n, d), Nat.mul(b, t), d,
        TA.Mat.matmul_tn(Nat.mul(4n, d), Nat.mul(b, t), d, 0n,
          patch(b, t, d, x), dy2), dw =>
          dup_bind(ConvBwd(b, t, d), Nat.mul(b, t), Nat.mul(4n, d), dp, dpA => dpB =>
            dup_bind(ConvBwd(b, t, d), Nat.mul(b, t), Nat.mul(4n, d), dpA, dpC => dpD =>
              dup_bind(ConvBwd(b, t, d), Nat.mul(b, t), Nat.mul(4n, d), dpC, dpE => dpF =>
                dup_bind(ConvBwd(b, t, d), Nat.mul(4n, d), d, dw, dwA => dwB =>
                  dup_bind(ConvBwd(b, t, d), Nat.mul(4n, d), d, dwA, dwC => dwD =>
                    dup_bind(ConvBwd(b, t, d), Nat.mul(4n, d), d, dwC, dwE => dwF =>
                      T.Two{T.Two{
                        T.Mat.sum2(Nat.mul(b, t), d,
                          T.Mat.sum2(Nat.mul(b, t), d,
                            unshift(b, t, d, 3n, block_cols(b, t, 0n, d, dpB)),
                            unshift(b, t, d, 2n, block_cols(b, t, 1n, d, dpD))),
                          T.Mat.sum2(Nat.mul(b, t), d,
                            unshift(b, t, d, 1n, block_cols(b, t, 2n, d, dpE)),
                            unshift(b, t, d, 0n, block_cols(b, t, 3n, d, dpF)))),
                        T.Two{diag_block(0n, d, dwB), diag_block(1n, d, dwD)}},
                        T.Two{diag_block(2n, d, dwE), diag_block(3n, d, dwF)}})))))))))

def cbwd_bind(-Z: Type, +b: Nat, +t: Nat, +d: Nat, r: ConvBwd(b, t, d),
  k: TA.Mat<Nat.mul(b, t), d> -> TA.Mat<1n, d> -> TA.Mat<1n, d> ->
     TA.Mat<1n, d> -> TA.Mat<1n, d> -> Z) -> Z:
  match r:
    case T.Two{T.Two{dx, T.Two{dw0, dw1}}, T.Two{dw2, dw3}}:
      k(dx, dw0, dw1, dw2, dw3)

def ConvP(+d: Nat) -> Type:
  T.Two<T.Two<TA.Mat<1n, d>, TA.Mat<1n, d>>,
    T.Two<TA.Mat<1n, d>, TA.Mat<1n, d>>>

def RowParam(+d: Nat) -> T.Layer<TA.Mat<1n, d>, Unit, Unit, Unit>:
  T.Layer{
    p => x => T.Fwd{p, Unit{}, Unit{}},
    p => s => dy => g => T.Bwd{p, Unit{}, g},
    op => p => g => st => T.mat_step(1n, d, op, p, g, st),
    u => TA.Mat.zeros(1n, d),
    p => T.Mat.dump(1n, d, p),
    p => xs => off => T.Mat.restore(1n, d, p, xs, off),
    p => xs => off => T.Mat.store(1n, d, p, xs, off),
    u => U32.from_nat(d)
  }

def ConvP32() -> Type:
  TA.Mat<1n, 128n>

def unpack_bind(-Z: Type, w: TA.Mat<1n, 128n>,
  k: TA.Mat<1n, 32n> -> TA.Mat<1n, 32n> -> TA.Mat<1n, 32n> -> TA.Mat<1n, 32n> -> Z) -> Z:
  row_bind(Z, 1n, 128n, TA.Mat.read_row(1n, 128n, w, 0),
    m => +row =>
      k(TA.Mat.from_list(1n, 32n, List.take(&2, F32, row, 32n)),
        TA.Mat.from_list(1n, 32n,
          List.take(&2, F32, List.drop(&2, F32, row, 32n), 32n)),
        TA.Mat.from_list(1n, 32n,
          List.take(&2, F32, List.drop(&2, F32, row, 64n), 32n)),
        TA.Mat.from_list(1n, 32n,
          List.take(&2, F32, List.drop(&2, F32, row, 96n), 32n))))

def pack_bind(-Z: Type, w0: TA.Mat<1n, 32n>, w1: TA.Mat<1n, 32n>,
  w2: TA.Mat<1n, 32n>, w3: TA.Mat<1n, 32n>,
  k: TA.Mat<1n, 128n> -> Z) -> Z:
  k(TA.Mat.from_list(1n, 128n,
    List.append(&2, F32, wlist(32n, w0),
      List.append(&2, F32, wlist(32n, w1),
        List.append(&2, F32, wlist(32n, w2), wlist(32n, w3))))))

def conv_layer_fwd(+b: Nat, +t: Nat, p: ConvP32(),
  x: TA.Mat<Nat.mul(b, t), 32n>) ->
  T.Fwd<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>, TA.Mat<Nat.mul(b, t), 32n>>:
  dup_bind(T.Fwd<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>, TA.Mat<Nat.mul(b, t), 32n>>,
    1n, 128n, p, pk => pu =>
    dup_bind(T.Fwd<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>, TA.Mat<Nat.mul(b, t), 32n>>,
      Nat.mul(b, t), 32n, x, xf => xs =>
      unpack_bind(T.Fwd<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>, TA.Mat<Nat.mul(b, t), 32n>>,
        pu, w0 => w1 => w2 => w3 =>
        T.Fwd{pk, conv_fwd(b, t, 32n, w0, w1, w2, w3, xf), xs})))

def conv_layer_bwd(+b: Nat, +t: Nat, p: ConvP32(),
  x: TA.Mat<Nat.mul(b, t), 32n>, dy: TA.Mat<Nat.mul(b, t), 32n>, g: ConvP32()) ->
  T.Bwd<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>>:
  dup_bind(T.Bwd<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>>, 1n, 128n, p, pk => pu =>
    unpack_bind(T.Bwd<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>>, pu,
      w0 => w1 => w2 => w3 =>
      cbwd_bind(T.Bwd<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>>, b, t, 32n,
        conv_bwd(b, t, 32n, w0, w1, w2, w3, x, dy),
        dx => dw0 => dw1 => dw2 => dw3 =>
          pack_bind(T.Bwd<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>>,
            dw0, dw1, dw2, dw3, dwp =>
            T.Bwd{pk, dx, T.Mat.sum2(1n, 128n, g, dwp)}))))

def ShortConv32(+b: Nat, +t: Nat) ->
  T.Layer<ConvP32(), TA.Mat<Nat.mul(b, t), 32n>, TA.Mat<Nat.mul(b, t), 32n>,
    TA.Mat<Nat.mul(b, t), 32n>>:
  T.Layer{
    p => x => conv_layer_fwd(b, t, p, x),
    p => s => dy => g => conv_layer_bwd(b, t, p, s, dy, g),
    op => p => g => st => T.mat_step(1n, 128n, op, p, g, st),
    u => TA.Mat.zeros(1n, 128n),
    p => T.Mat.dump(1n, 128n, p),
    p => xs => off => T.Mat.restore(1n, 128n, p, xs, off),
    p => xs => off => T.Mat.store(1n, 128n, p, xs, off),
    u => U32.from_nat(128n)
  }
