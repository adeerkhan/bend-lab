# The tiny-LM's shape invariants, as laws over the constants the model uses.
#
# The model names its shapes in mamba3_siso.bend (n_heads, n_groups,
# d_state_w, state_rows, head_state, all_state, seq_len) and computes its
# offsets with them. Each law below checks every index the shape has: the four
# query heads, the 32 state rows, the 2048 state cells, and the positions of
# the layer (8) and of the tiny-LM (128). They are finite checks by
# computation, not general arithmetic lemmas: they catch a constant or an
# offset that drifts, which is what a shape change looks like.
#
#   bend examples/language/mamba3/mamba3_laws.bend --verdict
import Base
import ./mamba3_siso.bend as S

# and_bool takes the check as a value so no match has a computed scrutinee
def and_bool(b: Bool, c: Bool) -> Bool:
  match b:
    case True{}:
      c
    case False{}:
      False{}

# ---- GQA: the group a query head reads ----
#
# Nat.div(h, 2) < 2 for h = 0..3, the four query heads.
def gqa_go(+left: Nat, +h: Nat) -> Bool:
  match left:
    case 0n:
      True{}
    case 1n+p:
      and_bool(Nat.is_lt(Nat.div(h, S.n_groups()), S.n_groups()),
        gqa_go(p, Nat.add(h, 1n)))

law gqa_heads:
  {gqa_go(S.n_heads(), 0n) == True{} : Bool}

def gqa_heads():
  {==}

# ---- the state's rows: p * 16 + 16 <= 512 for p = 0..31 ----
def rows_go(+left: Nat, +p: Nat) -> Bool:
  match left:
    case 0n:
      True{}
    case 1n+q:
      and_bool(Nat.is_le(Nat.add(Nat.mul(p, S.d_state_w()), S.d_state_w()),
          S.head_state()), rows_go(q, Nat.add(p, 1n)))

law state_rows_cover:
  {rows_go(S.state_rows(), 0n) == True{} : Bool}

def state_rows_cover():
  {==}

# ---- the heads' blocks: h * 512 + 512 <= 2048 for h = 0..3 ----
def heads_go(+left: Nat, +h: Nat) -> Bool:
  match left:
    case 0n:
      True{}
    case 1n+p:
      and_bool(Nat.is_le(Nat.add(Nat.mul(h, S.head_state()), S.head_state()),
          S.all_state()), heads_go(p, Nat.add(h, 1n)))

law state_heads_cover:
  {heads_go(S.n_heads(), 0n) == True{} : Bool}

def state_heads_cover():
  {==}

# ---- the scan's read: t + 1 written rows <= the sequence ----
def reads_go(+left: Nat, +t: Nat, +n: Nat) -> Bool:
  match left:
    case 0n:
      True{}
    case 1n+p:
      and_bool(Nat.is_le(Nat.add(t, 1n), n), reads_go(p, Nat.add(t, 1n), n))

# the layer's gate (8 positions) and the tiny-LM's (128)
law scan_reads_cover:
  {reads_go(S.seq_len(), 0n, S.seq_len()) == True{} : Bool} &
  {reads_go(128n, 0n, 128n) == True{} : Bool}

def scan_reads_cover():
  ({==}, {==})
