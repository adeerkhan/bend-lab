# Upstream issues and pull requests

Every artifact this repository has filed or opened upstream, newest work first.
The **full bendlang/bend list, with the fixing pull request for each issue**, is the README's
`Upstream` section; this file adds the BendTorch and bend-ml artifacts and what is still open there.
An artifact that this repository fixed itself is not opened upstream. **BendTorch 2.0.0 is the pin's
target** (`bendtorch@2.0.0.0`, tensor-array `0.1.18.0`).

## Open

| Artifact | What it asks |
|---|---|
| [BendTorch#39](https://github.com/nuxyel/BendTorch/issues/39) | `MaxPool2d` with the arg-max backward, and a no-bias `Conv2d`: the image reference's two layer gaps. The no-bias half is #40. `MaxPool2d` is implemented on `feat/maxpool2d` (rebased on devel bf02f19, still stacked on the Conv2dNB commit, no PR until #40 is reviewed): `(n,c,h,w,k,s)`, the arg-max recomputed from the input, ties first-in-window, overlap adds; y and dX are 0.000e+00 against `F.max_pool2d(..., return_indices=True)`. Its two laws are in (`mp_count`, `mp_scatter`), 44 in all, `--verdict` ALL PROOFS CHECK; the PR body is written and waits for #40. |
| [BendTorch#40](https://github.com/nuxyel/BendTorch/pull/40) | `Conv2dNB`: `Conv2d` without the bias, open PR. y, dX and dW are 0.000e+00 against `F.conv2d(..., bias=None)` on the three shapes the Conv2d check uses. |
| [BendTorch#36](https://github.com/nuxyel/BendTorch/issues/36) | Gradient accumulation over k micro-batches, so a typed batch of 2 can reach an effective 8-16; the k=4 arm is still blocked on a Bend syntax error. |
| [BendTorch#29](https://github.com/nuxyel/BendTorch/issues/29) | `Conv2d`: the adjoint law is still tested rather than stated. |
| [BendTorch#28](https://github.com/nuxyel/BendTorch/issues/28) | The attention **block** is 3.2x its arithmetic at the ViT's shapes. **Answered:** masked full attention is **9.55 ms** a pass against causal's **10.65 ms**, so the mask is not the cost — the core is 64 tiny 16x16 products instead of one large one, and the lever is fewer, larger calls. |
| [BendTorch#26](https://github.com/nuxyel/BendTorch/issues/26) | A step's kernel launches cannot be counted from a program, only under `nsys`: the report's `launches` and `unclaimed` lines are permanently `unavailable`, so "fewer launches" cannot be a measured lever. Asks for one count per `Train.step`. |
| [BendTorch#18](https://github.com/nuxyel/BendTorch/issues/18) | RoPE: the only rotation is 1D over rows, so an image transformer cannot rotate by (y, x). Needed only by a conditional UNet. |
| [BendTorch#19](https://github.com/nuxyel/BendTorch/issues/19) | adaLN: no layer broadcasts a per-image vector over its tokens. Needed only by a conditional UNet. |
| [bendlang/bend#1632](https://github.com/bendlang/bend/issues/1632) | Clearer diagnostics for common mistakes. Items 9 and 10 are ours; the 84 rejections in `reference/corpus/qwenbend/` are ready-made inputs. |
| [bendlang/bend#1998](https://github.com/bendlang/bend/issues/1998) | A match binder that shadows a parameter prints the name twice and no scope. |
| [bendlang/bend#1980](https://github.com/bendlang/bend/issues/1980) | `Array<T>` accepts a mixed-depth `ANode` the compiled runtime cannot represent. |
| [bend-ml#35](https://github.com/nuxyel/bend-ml/issues/35) | `Mat` reshape and row/column slicing, and an `ML` accessor, so pixel-unshuffle and max-pool stop going through `Mat.to_list` and linked-list `nth` (O(n²)). The reshape is #36; row/column slicing stays, as the matcher plus `Array.get` may cover it. |
| [bend-ml#36](https://github.com/nuxyel/bend-ml/pull/36) | `Mat.reshape`, the same array under a new shape with `r1*c1 == r2*c2` erased, and the law `reshape_len`; open PR. |
| [bend-ml#26](https://github.com/nuxyel/bend-ml/issues/26) | The value law (get after set) as a Base question; the round-trip corollary waits on two Base lemmas. Our last word is on the record. |

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
| [BendTorch#17](https://github.com/nuxyel/BendTorch/issues/17) | Closed by `FullAttention` in v1.3.0. |
| [BendTorch#15](https://github.com/nuxyel/BendTorch/issues/15) | Closed 2026-11-03: with `Conv2d`, `GroupNorm` and `FullAttention` in, an unconditional UNet is expressible. A follow-up (#29) carries the general `k = s` round trip and the adjoint law. |
