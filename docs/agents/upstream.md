# Upstream issues and pull requests

Every artifact this repository has filed or opened upstream, newest work first.
The **full bendlang/bend list, with the fixing pull request for each issue**, is the README's
`Upstream` section; this file adds the BendTorch and bend-ml artifacts and what is still open there.
An artifact that this repository fixed itself is not opened upstream. **BendTorch 2.1.0.0 is the lane's target** (`bendtorch@2.1.0.0`,
tensor-array `0.1.20.0`, tensor-gpu `0.1.5.2`).

## Live check (2026-11-17)

The statuses below were read from the GitHub API on 2026-11-17, not from the
last entry in the ledger. Corrections to the older table:

- **BendTorch#28, #26 are closed** (completed / not planned), not open.
- **BendTorch#36 is open**: merged into devel `65c8038`, but nuxyel kept it
  open for the promised four-batch UNet run from us; that run is still owed.
- **bend-ml#35 is closed** by `Mat.reshape` in #36; `Mat.rows/cols` stay out.
- **bendlang/bend#1998, #1980, #1632, #1961 are closed** (completed or
  duplicate). Only #2146, #2147 and #2150 remain open language design asks.
- **BendTorch#42 merged** into devel `8999c69`; #39 is closed as completed.
  nuxyel added the `gpu.bend` twin (0 ulps on four shapes), seven mutants and
  the docs on top. Our follow-up for the two gaps he recorded in
  `docs/limitations.md` is PR [#44](https://github.com/nuxyel/BendTorch/pull/44):
  geometry guarded by construction (kernel + surplus, stride - 1), as in
  Conv2d. `check-maxpool2d` and `check-gpu-layers` PASS on the new head.
- **bendlang/bend#1403 closed as not worth fixing** by icolas-abril: a Nat
  match with thousands of literal rows costs quadratic checker memory, but
  a few hundred rows check in ~0.2 s and a fix would grow `bend.ts` past
  its size cap; succ-pattern rows check in 0.2 s / 83 MB at 3000 rows.
- **bendlang/bend v2.0.40 is tagged upstream**; this lab stays pinned at 2.0.38 —
  a version change is a banner, not a delta.

## Open

| Artifact | What it asks |
|---|---|
| [BendTorch#39](https://github.com/nuxyel/BendTorch/issues/39) | `MaxPool2d` with the arg-max backward, and a no-bias `Conv2d`: the image reference's two layer gaps. The no-bias half is merged as #40. `MaxPool2d` is PR [#42](https://github.com/nuxyel/BendTorch/pull/42), rebased stand-alone on devel `58f346f`: `(n,c,h,w,k,s)`, the arg-max recomputed from the input, ties first-in-window, overlap adds; y and dX are 0.000e+00 against `F.max_pool2d(..., return_indices=True)`. Its two laws (`mp_count`, `mp_scatter`) make 46 in all; `--verdict` ALL PROOFS CHECK. |
| [BendTorch#40](https://github.com/nuxyel/BendTorch/pull/40) | Merged into devel `c62e1e9`: `Conv2dNB`, with the GPU twin and six mutants. (In the Open table above only for history; it is closed.) |
| [BendTorch#36](https://github.com/nuxyel/BendTorch/issues/36) | **Open** (live 2026-11-17): merged on devel `65c8038`, kept open for our four-batch UNet run; the k=1 and repeated-pair k=4 numbers are already reported in the issue. |
| [BendTorch#29](https://github.com/nuxyel/BendTorch/issues/29) | `Conv2d`: the adjoint law is still tested rather than stated. **Open.** |
| [BendTorch#28](https://github.com/nuxyel/BendTorch/issues/28) | **Closed (completed)** 2026-11-11: the triangle sums landed in devel `e6e2b2a` (laws `causal_pv`, `causal_kv`); the mask was never the cost, the core's 64 tiny products were. |
| [BendTorch#26](https://github.com/nuxyel/BendTorch/issues/26) | **Closed (not planned)** 2026-11-07: launches stay countable only under `nsys`, so `launches`/`unclaimed` remain `unavailable`, never zero. |
| [aaaxn/bendtorch#86](https://github.com/aaaxn/bendtorch/issues/86) | Question, not a defect: is keeping only the last-step `loss: F32` on `T.Train` intentional? The per-step curve is not readable from outside the loop. |
| [bendlang/bend#2146](https://github.com/bendlang/bend/issues/2146) | Record ergonomics: field projection, match on a local binder, or Data-shared records. Carries the three repros from `.local/mamba3-repro/` (`r.a` is not a term; `match` cannot scrutinize a local binder; `+r: R` is rejected). |
| [bendlang/bend#2147](https://github.com/bendlang/bend/issues/2147) | No forward references and no mutual recursion: the two-def `f`/`g` minimal case. A two-pass compilation or a sanctioned iterative encoding was asked; the local pattern is one self-recursive def. |
| [bendlang/bend#2150](https://github.com/bendlang/bend/issues/2150) | The intended encoding for evolving state across an online loop (records + `Array<F32>` threaded through one self-recursive def); carries two verified repros and the pointer to the working 16-token mamba3 step gate. |
| [bendlang/bend#1632](https://github.com/bendlang/bend/issues/1632) | Clearer diagnostics for common mistakes. Items 9 and 10 are ours; the 84 rejections in `reference/corpus/qwenbend/` are ready-made inputs. |
| [bendlang/bend#1998](https://github.com/bendlang/bend/issues/1998) | A match binder that shadows a parameter prints the name twice and no scope. |
| [bendlang/bend#1980](https://github.com/bendlang/bend/issues/1980) | `Array<T>` accepts a mixed-depth `ANode` the compiled runtime cannot represent. |
| [bend-ml#35](https://github.com/nuxyel/bend-ml/issues/35) | **Closed (completed)** 2026-11-11 by `Mat.reshape` in #36 (`6b6b343`); flat `Array.get`/`Array.set` covers cell reads. `Mat.rows/cols` stay out until content laws land. |
| [bend-ml#36](https://github.com/nuxyel/bend-ml/pull/36) | Merged into devel `6b6b343`: `Mat.reshape`, the same array under a new shape with `r1*c1 == r2*c2`, and the law `reshape_len`. |
| [bend-ml#26](https://github.com/nuxyel/bend-ml/issues/26) | The value law for `Array.set`/`Array.get`. nuxyel proved `size_set`, `get_after_set`, `get_other_set` and `ASet.Apart` in tensor-array devel `f882c77`; the remaining `Mat.to_list (Mat.of_list rs cs xs) == xs` corollary is ours, with a 64x64 NumPy round-trip test. Open until that PR lands. |

## Closed

| Artifact | Outcome |
|---|---|
| [BendTorch#34](https://github.com/nuxyel/BendTorch/pull/34) | Merged: `Tile`, a per-image row broadcast, with `tile_row` and `tile_sum`. |
| [BendTorch#33](https://github.com/nuxyel/BendTorch/pull/33) | Merged (80b7da0): `RoPE2d`, with `rope2_partition` and the two `tw = 1` laws. |
| [BendTorch#32](https://github.com/nuxyel/BendTorch/pull/32) | Merged: the `Conv2d` round trip at any stride and padding, with the division lemma behind it. |
| [BendTorch#27](https://github.com/nuxyel/BendTorch/pull/27) | Merged 2026-11-03; released as 2.0.0. The optimizer configuration lives in the `Train`, `Train.with_cfg` changes it, `Opt.guard` refuses a kind mismatch with a NaN optimizer (four laws). Our sweep on it: three rates, three curves. |
| [BendTorch#22](https://github.com/nuxyel/BendTorch/pull/22) | Merged 2026-11-03 as part of 2.0.0: `Conv2d` in both lanes, `im2col`/`col2im`, two laws, the PyTorch checks. Rebased onto #27 and merged with its three review notes. |
| [BendTorch#24](https://github.com/nuxyel/BendTorch/issues/24) | Closed by 2.0.0. It was worse than the learning rate: a kind mismatch left the loss unchanged with no error, and `OptCfg.ok` ran only in `Train.new`. |
| [BendTorch#23](https://github.com/nuxyel/BendTorch/pull/23) | Closed 2026-11-03 as agreed: v1.3.0 wired the demo and `test_fullattn.py` into `make check` differently; nothing in ours to keep. |
| [BendTorch#19](https://github.com/nuxyel/BendTorch/issues/19) | Closed: `Tile` (#34) broadcasts a per-image row; adaLN is a composition of `Tile` and a per-image vector. |
| [BendTorch#18](https://github.com/nuxyel/BendTorch/issues/18) | Closed by `RoPE2d` (#33), with `rope2_partition` and the `tw = 1` laws. |
| [BendTorch#17](https://github.com/nuxyel/BendTorch/issues/17) | Closed by `FullAttention` in v1.3.0. |
| [BendTorch#15](https://github.com/nuxyel/BendTorch/issues/15) | Closed 2026-11-03: with `Conv2d`, `GroupNorm` and `FullAttention` in, an unconditional UNet is expressible. A follow-up (#29) carries the general `k = s` round trip and the adjoint law. |

## Repros and their upstream state

| Item | Repro | State |
|---|---|---|
| ISSUE-026 the TEM port's missing layers | `papers/TEM/README.md` | BendTorch #33 (`RoPE2d`), #34 (`Tile`) and #42 (`MaxPool2d`); two merged, `MaxPool2d` in review |
| ISSUE-028 RoPE is one rotation over the flat row | `rope2d_check.py` in BendTorch PR #33 | BendTorch #18, closed by `RoPE2d` (#33) |
| ISSUE-029 no per-image vector broadcast | `papers/TEM/README.md`, the adaLN section | BendTorch #19, closed by `Tile` (#34) plus a composition |
| ISSUE-030 a forward reference is an unfilled law | the minimal polymorphic def in bendlang/bend #1632 | bendlang/bend #1632, proposed; local repro kept |
| ISSUE-031 a shadowing match binder loses its scope | the minimal match in bendlang/bend #1998 | bendlang/bend #1998, filed |
| ISSUE-064 `Nat.mul` in a type argument overflows the checker stack | bendlang/bend #1632, item 9 | local repro; compiler limit, not patched here |
| ISSUE-069 `--check-only` exits 1 on any `bendtorch-gpu` import | `examples/language/mamba3/probe_bendtorch_gpu.bend` | BendTorch, measured; #41 (release 2.1.0) closed 2026-11-12 — recheck on 2.1.0.0 still owed |
| ISSUE-102 the published before-RL 0/24 does not reproduce | `results/glm53-dense/eval-dev.json` | reference discrepancy; recorded, not tuned |
| ISSUE-104, 105, 107, 127 | `bendlang/bend#2146` (records), `bendlang/bend#2147` (forward references / mutual recursion), `bendlang/bend#2150` (state threading) | language design limits; design around them; asks filed 2026-11-13 |
| ISSUE-063 / bend-ml#26 | the U32 facts and the round-trip laws in bend-ml#26 | U32 facts landed; the `Mat.to_list ∘ Mat.of_list` corollary remains upstream |
