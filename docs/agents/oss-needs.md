# What the open-source Base is missing for ML ports

One file. What the Bend ecosystem ships today for the numerical primitives the
ports keep hand-writing, what is missing, and what shape a `Base` would have
to take to close the gap. It accompanies `docs/bend-2.0.38-pitfalls.md` and
`.local/issues-hit.md` (ISSUE-108, ISSUE-109). Nothing here is filed as an
upstream issue; the two asks that were filed are `bendlang/bend#2146` and
`bendlang/bend#2147`.

Search date: 2026-11-13. Bend 2.0.38, bend-ml 2.1.0.0 tensor-array 0.1.20.0.

## What exists, where

### Base (shipped with the compiler), `bend2/base.bend` (~3212 lines, ~411 defs)

`F32` has `exp`, `log`, `tanh`, `cos`, `sin`, `sqrt`, `mod`, `is_lt`,
`from_nat`. That is the whole numerical surface. There is **no scan, no
softmax, no argmax, no cross-entropy** in Base:

- `bend base F32 | grep -c 'argmax\|softmax'` → `0`
- No `scan` or `cumsum` exists anywhere in Base, tensor-array or bendtorch.

### bend-ml tensor-array (`tensor-array/main.bend`), 0.1.20.0

- `softmax` / `softmax2` — lines 334-339, list substrate, `maxl`/`expl`.
- cross-entropy — `ce_row2`/`ce_row`/`ce_rows` (lines 381-394), and
  `Mat.softmax_ce` (line 1412) producing `MCE{loss, grad}`.
- `argmax` / `argmax2` — lines 491-502 (tests also use these).

### bendtorch (`torch/main.bend`), 2.1.0.0

- in-place row softmax — line 1513 (`softmax in place over the n numbers of p from off`).
- log-softmax — line 1587.
- softmax backward — line 1565 (`dx = y * (dy - sum(dy * y))`).
- `T.Train` carries a single `loss: F32` (final step only; see ISSUE-126 and
  `bendtorch#86`).

### Reinforcement

Nothing else. The gradient machinery is BendTorch's; Base is intentionally a
compiler library, not a tensor library.

## What is missing and is therefore hand-written

1. **A scan / cumsum / in-place-state primitive.** The Mamba-3 port keeps the
   2048-float structured state in one `Array<F32>` and walks it with a
   hand-written `Nat` recursion (`pass_c` in `mamba3_siso.bend`, and the GLM
   port's decode loop in `glm53_flash/step_h192.bend`). There is no way to
   say `scan state f init xs`, so every port unrolls its own. Cost of the
   naive list loop on the training probe (80 tokens, ten SGD steps, three
   forward passes and one backward per step, RTX 5060):
   `/usr/bin/time -v bend examples/language/mamba3/probe_mamba3_train.bend` →
   **4.58 s**, peak RSS **354,264 KiB** (ISSUE-108). The
   array-state version now in `pass_c` is `3.64-3.67 s` against `9.81-10.50 s`
   for the naive loop, but that win was hand-bought, not language-given.
2. **A token `argmax` / softmax in Base.** Both ports hand-write them:
   `mamba3_siso.bend` (`argmax` ~1611, `softmax_list` ~1614, `ce_loss`
   ~1619), `glm53_flash/step_h192.bend` (`argmax` 66).
3. **A tied-`Var`/`Id` shared-shape story for in-place weight updates** so a
   training loop can hold weights, grads and an optimizer state without
   rebuilding records (see ISSUE-105, `bendlang/bend#2146`).

## What `scan` in Base could look like

Two viable designs, either is enough:

```text
def scan(~A: Type, ~X: Type, ~S: Type, ~acc: (S, X) -> S, +z: S,
         xs: List<&2, X>) -> S                    # the mathematical form
```
or a state-threaded loop marker (`Scan` law that the checker can see
descending), mirroring how `IO` is a monad in Base. The second is more in
keeping with the filled-law discipline; the first is more in keeping with
the list substrate. Neither exists today; nothing is filed against them in
`bendlang/bend`.

## What `softmax` / CE / argmax in Base should (not) look like

The strong case to leave them out: tensor-array already ships them, tested,
and `Mat.softmax_ce` computes loss and gradient in one pass. Duplicating
that in Base would fork the numerics. The useful Base addition would only
be a **stable `softmax` over a `List<&2, F32>` and an `argmax`**, as
conveniences, with the same signatures as tensor-array so ports can swap
implementations. Cross-entropy belongs in tensor-array, full stop.

## Work-in-progress tag

Searching `bendlang/bend` on 2026-11-13 finds **no** open or closed issue
proposing a scan/cumsum primitive, a Base softmax/argmax, or a Base
cross-entropy. The only adjacent activity is complexity/flattening O(n^2)
work (parser/checker) and one metal dispatcher crash; none of it touches
numerical primitives. The nearest live artifacts are the ports above and the
hand-rolled loops they contain.

## Recommendation

No local attempt at a `Base` PR: the change is a compiler-language decision,
and this repository does not edit `bend2/base.bend`. The honest next move,
if the maintainers ever want it, is one proposal issue on `bendlang/bend`
with the two designs in "What `scan` in Base could look like", scoped as
design-first. The ports keep their hand-rolled loops and their cost
measurements; they stay the oracle for `pass_c`.
