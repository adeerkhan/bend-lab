# Bend 2.0.38 pitfalls this project paid for

Status: committed to `docs/bend-2.0.38-pitfalls.md`. Written 2026-11-13 from the
Mamba-3 layer gate (`mamba3_layer.bend`), the Mamba-3 SISO work
(`mamba3_siso.bend`), the Mamba-3 decode step (`step_mamba3.bend`, added the
same day) and the GLM-5.3-Flash step decoder (`step_h192.bend`). Bend is
pinned at 2.0.38; nothing here is a claim about any other version.

Each entry is framed the same way: given a goal like X, here is what rusts.
The examples are real sites in this repository; "the workaround" means the
shape the code was rewritten into, "the structural constraint" means the
thing no rewriting inside this project can remove today.

## ISSUE-120 — matches cannot scrutinize computed values; helpers take the value as a parameter

Given a goal like "call a function and branch on its result with `match`",
here is what rusts. `match f(x):` is rejected with "a parameter or field
scrutinee (a match cannot scrutinize a computed value: give it its own def)",
and binding the call to a local first does not help: `let y = f(x); match y`
reports the same for a local binder. The scrutinee of a `match` must be a
def parameter or a pattern variable.

Example. `loss_masked.bend` computes `Nat.is_eq(y, ignore)` at the call site
and passes the `Bool` into `mask_one(masked: Bool, ...)`, whose body matches
on the parameter (`examples/language/glm53_flash/loss_masked.bend:27-42`).
`mamba3_layer.bend` does the same for the layer forward: `fwd` binds
`f : M.FWD = M.forward(...)` plainly, passes the owned value to `bp_of(f)`
and `fwdfin(...)`, and the `match` lives on those parameters
(`examples/language/mamba3/mamba3_layer.bend:225-232`, `:29-36`, `:157-161`).
The same shape forced the whole `T.Layer` wiring: every `match` in the file
sits on a parameter.

The fix was mechanical but pervasive: hoist every computed value that needs
scrutinizing into a helper def that receives it as an argument, and keep the
call sites reading `helper(computed, ...)`. Two side effects came free with
the rule: no forward references (defs must precede their use, so file order
is dependency order), and a topological sort over the module was needed
before it compiled at all.

Take-away. The workaround is reliable and general: pass the value in, match
on the parameter. The structural constraint is that a `match` can never see
a call or a local binder, so the helper-def shape is not stylistic, it is
mandatory. Every future gate in this repository that wants to inspect a
step result must budget for the extra defs.

## ISSUE-121 — multi-scrutinee matches require the full cross product; `_` covers only one side

Given a goal like "match two or three things at once without spelling out
every constructor combination", here is what rusts. The exhaustiveness
checker composes the constructor sets of all scrutinees: `case 0n _:` plus
`case 1n+p Con{v, t}:` still leaves `(1n+p, Nil{})` out, reported as
"expected: cases for Nil". A wildcard on one axis discharges only that axis.

Example. `store_step` in `mamba3_layer.bend` matches `left src:` and must
carry all three groups explicitly: `case 0n _:`, `case 1n+p Nil{}:` and
`case 1n+p Con{v, t}:` (`examples/language/mamba3/mamba3_layer.bend:198-207`).
`tp_add_list` does the two-list version: `case Nil{} _:`, `case _ Nil{}:`,
`case Con{x, t1} Con{y, t2}:` (`:73-80`). The three-scrutinee case is
`prefill_go` in `step_h192.bend`: `match left last r:` enumerates
`0n last0 Tuple{...}`, `1n last0 ...`, `2n+p last0 ...` and the
`_ last1 ...` remainder (`examples/language/glm53_flash/step_h192.bend:152-161`);
`nth` shows the same trick with a trailing `_ _:` (`:27-34`).

The fix was to write the cross product, or to fall back to a wildcard on the
unmatched scrutinee axis and accept the extra cases. There is no shorter
spelling.

Take-away. The workaround is reliable but noisy: big matches grow
combinatorially with each scrutinee added. The structural constraint is that
exhaustiveness is checked per constructor combination, not per axis, so a
"rest" case is not expressible as one wildcard per axis.

## ISSUE-122 — decreasing self-calls read arguments left to right; the shrinking argument must come first

Given a goal like "write a recursive loop whose arguments change, with the
countdown somewhere in the middle of the parameter list", here is what
rusts. The self-call is rejected as not decreasing with "expected: a
decreasing self-call (arguments are read left to right: each passed
unchanged until one shrinks)", even when a later argument clearly shrinks.
The check is positional: each argument is passed unchanged until the first
argument that shrinks.

Example. The array-store helpers in `mamba3_layer.bend` were restructured to
put the `Nat` countdown first: `li_fill(+left: Nat, cur: ..., +off: U32,
acc: ...)` (`:185-196`) and `store_step(+left: Nat, xs: ..., +off: U32,
src: ...)` (`:198-207`); every call site now reads
`li_fill(511n, Array.get(F32, xs, off), ...)`. The same shape is all over
`mamba3_siso.bend`: `bw_go(+left: Nat, +t: Nat, ...)`
(`:985`), `loss_go` (`:1654`), `dy_all_go` (`:1664`), `dWh_go` (`:1675`),
and the token loops in `step_h192.bend` (`prefill_go`, `decode_go`, `:149`,
`:163`).

The fix was to move the shrinking argument to position zero and match
helpers against it. Loops that originally carried the weight list or the
state tensor first had to be rewritten.

Take-away. The workaround is reliable: shrinking argument first, everything
else after. The structural constraint is that the checker ignores decreasing
arguments to the right of a non-shrinking one, so argument order in recursive
defs is semantic, not cosmetic.

## ISSUE-123 — a shared (`+`) binding requires the `Data` sort; Bend types cannot be shared

Given a goal like "bind the forward result once and reuse it several times
without copying", here is what rusts. `+f : FWD = M.forward(...)` is
rejected with "Note: +f can be used many times, so its type must be Data".
Values of a `type ... is Type` cannot be `+`-bound; only lists
(`List<&2, F32>`), `F32`, `Nat` and the `&`-shaped tuples may be.

Example. `mamba3_layer.bend` keeps the layer forward result unshared:
`f : M.FWD = M.forward(...)` is a plain (owned) bind, and the `FWD` is moved
into `bp_of(f)` exactly once (`:229-232`). What may be shared are the weight
lists: `tp_pl` builds `T.PL{TP{w_in, ...}, tp_flat(w_in, ...)}` from the
same `+`-bound list views passed twice, once into the `TP` construction and
once into the flattening call (`:55-60`); `fwd` and `bwd` do the same with
`+w_in`, `+E`, `+Wh` and friends. The `FWD` record itself is passed by
value and rebuilt by `bp_of` into `Tuple{M.FWD{pre, z, ...}, y}` from its
`+`-shared field views (`:29-36`).

The fix was to restructure: thread owned `Type` values through helper defs,
or rebuild from shared field views in one construction.

Take-away. The workaround is reliable for lists and tuples: bind them with
`+` and reuse freely. The structural constraint is that a `Type` value is
affine with no sharing escape hatch, so any structure meant to be read twice
must either be a list, a tuple of Data, or rebuilt from Data parts. This
rule, not taste, is why the block is flat `List<&2, F32>` instead of a
record of tensors.

## ISSUE-124 — the tuple type is `A & B`, the constructor is `Tuple{...}`; not `Tuple<A, B>`; `match` scrutinee order rules

Given a goal like "write the return type of a function returning a pair, or
pattern-match a nested tuple", here is what rusts. `def f() -> Tuple<A, B>`
is rejected with "a declared datatype (unknown: Tuple)". The type spelling is
`A & B`, and the value constructor is `Tuple{a, b}`. In a `match`, the
scrutinees are listed in order and the patterns in each case are written in
that same order, left to right.

Example. `bp_of` returns `M.FWD & List<&2, F32>`
(`examples/language/mamba3/mamba3_layer.bend:29`) and is called through
`fwdfin`, whose `case Tuple{f2, y}:` matches the constructor in the order
the return type declares (`:157-161`). `head_fin` in
`step_h192.bend` returns `List<&2, F32> & TA.Mat<260n, 32n>` and its one
case is `Tuple{ml260(...), w}` (`:77-80`); callers nest the matches, for
instance `match r: case Tuple{logits, embMb}:` in `step_fin_b`
(`:88-91`) and the doubly nested `match r: ... case Tuple{Tuple{a3b, m3b},
Tuple{x4, c3b}}:` in `step_sp` (`:112-113`). The `match left last r:` in
`prefill_go` shows the ordering rule at three scrutinees deep: every case
writes the `left` pattern, then the `last` pattern, then the tuple
(`:152-161`).

The fix was spelling and discipline: `A & B` in type position, `Tuple{...}`
in value position, and case patterns kept in scrutinee order.

Take-away. The workaround is reliable and the severity is low: it is a
spelling and ordering convention, not a semantic trap. The structural note
worth remembering is that the order of the scrutinee list is the order of
the patterns, so moving a scrutinee means rewriting every case.

## ISSUE-125 — `do IO<Unit>:` needs typed binds: `x : T <- expr`, and `IO.pure` needs a Type explicitly

Given a goal like "load ten files, build a record, run the trainer, print
the result", here is what rusts. Inside `do IO<Unit>:`, a pure construction
on its own line misparses: the block is rejected with "expected: a pattern",
showing the fully expanded continuation as the observed text. A bind must be
typed (`wi : TA.Mat<304n, 64n> <- T.Mat.load(...)`), and a pure value must
enter the block through an explicit `IO.pure` with the target type spelled
out: `x : T <- IO.pure(T, pure_expr)`. `IO.pure` without the type, or a bare
constructor line, is not legal. IO actions themselves may appear bare in the
block; pure constructions may not.

Example. `step_h192.bend` is the reference shape: every load is a typed
bind (`row : List<&2, U32> <- T.read_at(...)` in `eps_go`, `:332`), and
every pure build is hoisted into a typed `IO.pure`:
`embL : List<&2, F32> <- IO.pure(List<&2, F32>, ml260x32(...))`,
`a0 : P.AttW <- IO.pure(P.AttW, P.AttW{ln10, qkv0, out0})`,
`w : MdlW <- IO.pure(MdlW, MdlW{...})`, and the `Cache` zeros likewise
(`:212`, `:242-258`). `mamba3_layer.bend` does the same with the trainer:
`cp : TP <- IO.pure(TP, step_c(10n, build_tp(...), 0.001))`,
`tp : TP <- IO.pure(TP, build_tp(...))`,
`trf : T.Train<TP> <- IO.pure(T.Train<TP>, do_train(...))`
(`:591`, `:605-606`). Note also that `do` takes `IO<T>`, never `IO(T)`
(already recorded in the pitfalls list in `AGENTS.md`).

The fix was to hoist every pure construction into a top-level def (or a
named `IO.pure` bind inside the block) and to spell the type on every
`IO.pure`.

Take-away. The workaround is reliable: loads and pure builds each get one
typed bind line. The structural constraint is the block parser itself, not
the IO semantics; `io.print`/`IO.pass` cannot be used to share anything, so
sharing an intermediate result always goes through a typed bind.

## ISSUE-126 — `T.Train.step`'s per-step loss can only be read inside the loop; the gate compares only what is accessible

Given a goal like "log the loss of every training step and compare the whole
curve against the custom trainer", here is what rusts. `T.Train.step`
returns `Train{p, g, st, cfg, p1, p2, loss}` carrying that step's loss, but
Bend gives no one-way access to it: the `Train` value is consumed by the
match-on-def rules (ISSUE-120), there are no accessor defs on `Train`, and
apart from matching the final returned value the loop internals are not
reachable from outside. A per-step loss list therefore cannot leave
`T.Train` without forking bendtorch.

Example. `mamba3_layer.bend` threads the trainer through `train_full`
(`:453-463`) and can only print the final loss: `show_final` matches the
returned `T.Train{p, g, st, cfg, p1, p2, l}` and emits `layer_loss` from
`l` (`:552-566`). The gate compares the final-step loss (custom trainer
`2.1401443`, layer `2.1404066`, about `1.2e-04` relative, gate `1e-3`) and
the generated tokens exactly (`[3, 6, 0, 5, 0, 7, 0, 7]`), and says so
explicitly; the per-step list is not compared because it is not readable.

The fix was to narrow the gate's claim to what is observable from outside
the loop (final loss, tokens, weight heads at `<= 1.3e-05`) and to state
that limitation in the gate output and in this repository's records.

Take-away. There is no reliable workaround inside Bend 2.0.38 and no
in-repo patch: this is a structural boundary around `T.Train`. Claims about
per-step behaviour have to be made from inside a `T.Layer`/`T.Train` fork
or not at all; gates here are written against the final step.

## ISSUE-127 — state carried as records: the decode step, written anyway

Given a goal like "generate one token at a time from the trained Mamba-3 LM,
carrying the scan state forward instead of recomputing the prefix", here is
what rusts. The scan state is records (`FWD`, `SC`, `ST`) plus a 2048-float
array, and Bend 2.0.38 gives no compositional way to carry them across a
token loop: records of `Type` cannot be `+`-bound (ISSUE-123), a `match`
cannot sit on a computed value (ISSUE-120), arrays cannot be `+`-bound either
(`Array<F32> is not Data`), and the whole wiring has to be spelled out as
def-after-def with the shrinking argument first (ISSUE-122). The
`match-on-computed` rule also forces every intermediate record to be consumed
by a parameter-match helper, which is why the module reads as a chain of
`*_pb2`/`layer1_finish`/`head_done`-style defs.

The implementation paid the cost and it worked. `step_mamba3.bend`
(2026-11-13) carries `ST{l1, l2}` (two `ST2` records, each holding a 2048-float
`Array<F32>`, a 16-float angle list and the previous row's `x`/`k`/`DT`/`trap`
pending tensors) across tokens, threading the state unmatched through the
recursive def's `r` argument. One wrinkle was the reference pass's trapezoid:
row `t`'s `scale` uses row `t+1`'s `DT`/`trap`, which an online decoder cannot
know at emission time, so every row is folded with `scale = 0` (the
final-row rule, exactly what a full-prefix pass does for its last row) and,
when the next row's `DT`/`trap` arrive, an additive rank-one correction
`S[v, q] += (scale * x_v) * k_q` is applied to the carried array, in the same
float op order as `row_set`. The angle has no future dependency and threads
directly. `make mamba3-step` prints PASS: the sixteen greedy tokens equal the
golden `[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` exactly.

Take-away. The reliable paths are (a) accept the O(n^2) recompute where
exactness matters, (b) write the inline per-token scan as one large
hand-rolled module with the state flattened into lists and array arguments,
budgeting 1-2k lines for the ISSUE-120/121/122 shapes, or (c) wait for
first-class state records or explicit mutable state in the language. What
does not survive contact is a "small clean" design: the module is large,
every helper def exists because of one of the rules above, and nothing
about the wiring is optional. The interpreter path (`bend file.bend`)
overflows the Bend VM around seq len >= 64; the compiled path (`bend -o`
then the binary) completes: 64 tokens exact (reference 17.7 s, step
0.45 s), 256 tokens exact (reference 29 m 17 s, step 1.2 s).
`ulimit -s unlimited` does not help the interpreter. The step itself runs
in 0.08 s at 16 tokens.

## ISSUE-128 — `match left xs:` with an `xs` passed through requires the explicit cross product

Given a goal like "write a small helper that consumes a `Nat` countdown and
leaves the `xs` list untouched except to thread it through", here is what
rusts. Even in the narrowest one-call case, Bend 2.0.38 demands the same
explicit cross product as a general multi-scrutinee match: `case 0n _:`
alone is rejected as missing `cases for Con`, and `1n+p Nil{}` must be
listed explicitly. There is no pass-through exemption for a scrutinee that
is merely carried, not consumed.

Example. `store_step` in `mamba3_layer.bend` (`match left src:`, where
`src` is threaded straight into the recursive call) is the canonical case:
it carries `case 0n _:`, `case 1n+p Nil{}:` and `case 1n+p Con{v, t}:`
and nothing shorter compiles (`:198-207`). The same wall appeared at a
`match left xs:` site in the short-lived `head_rows` helper in
`step_mamba3.bend` (the deleted placeholder, ISSUE-127), which was the
ledger entry's original observation.

The fix was the same enumeration as ISSUE-121: every constructor pair gets
its own case group, with a wildcard only on the axis that is allowed to be
covered wholesale.

Take-away. The workaround is reliable: enumerate the cross product and move
on; the cost is a few extra case lines, not semantics. Distinct from
ISSUE-121 (where the missing cases are the point of the check), this is the
low-severity reminder that even a pass-through scrutinee keeps the full
constructor obligation, so small helpers are never as small as they look.

## Upstream requests this would inspire

Plain text, for the maintainer to weigh; nothing here has been filed
anywhere, and no code is proposed for pushing.

1. Let a `match` sit directly on a computed value, or make local binders
   legal scrutinees, so the helper-def ceremony disappears (ISSUE-120).
2. In a multi-scrutinee match, let one wildcard discharge a whole axis and
   report the remaining missing cases per axis, instead of the raw cross
   product (ISSUE-121 and ISSUE-128).
3. Detect a decreasing self-call argument anywhere in the argument list
   instead of strictly left to right, so argument order is not semantic
   (ISSUE-122).
4. Allow `+` sharing (or an explicit immutable borrow) for records built
   from `Type` when every field is `Data`, so a layer's forward result can
   be reused without rebuilding (ISSUE-123).
5. Either bless `Tuple<A, B>` as a synonym for `A & B` with the `Tuple{...}`
   constructor, or make the "unknown datatype" error name the right spelling
   (ISSUE-124).
6. Inside `do IO<T>:`, accept pure expressions as statements by lifting
   them, or fail with a message that names the `IO.pure(T, ...)` idiom; and
   let `IO.pure` infer its type argument (ISSUE-125).
7. Give `T.Train` accessor defs (or an iteration/fold over steps) so the
   per-step loss is readable from outside the loop without forking
   BendTorch (ISSUE-126).
8. First-class state records, or explicit mutable state threading, so an
   incremental decode step can be written as a small composition of defs
   instead of one hand-rolled module (ISSUE-127).
9. Allow `A & B` binders as scrutinees in multi-scrutinee matches without
   extra case groups, or at least confirm the one-call shape is intended
   (ISSUE-128).
