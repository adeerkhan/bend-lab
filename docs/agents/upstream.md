# Upstream issues and pull requests

Every artifact this repository has filed or opened upstream, newest work first.
The local issue numbers in `.local/issues-hit.md` map to these; a local issue
with no entry here was fixed in this repository. **BendTorch 2.0.0 is the pin's
target** (`bendtorch@2.0.0.0`, tensor-array `0.1.18.0`).

## Open

| Artifact | What it asks |
|---|---|
| [BendTorch#28](https://github.com/nuxyel/BendTorch/issues/28) | The attention **block** is 3.2x its arithmetic at the ViT's shapes. **Answered:** masked full attention is **9.55 ms** a pass against causal's **10.65 ms**, so the mask is not the cost — the core is 64 tiny 16x16 products instead of one large one, and the lever is fewer, larger calls. |
| [BendTorch#26](https://github.com/nuxyel/BendTorch/issues/26) | A step's kernel launches cannot be counted from a program, only under `nsys`: the report's `launches` and `unclaimed` lines are permanently `unavailable`, so "fewer launches" cannot be a measured lever. Asks for one count per `Train.step`. |
| [BendTorch#18](https://github.com/nuxyel/BendTorch/issues/18) | RoPE: the only rotation is 1D over rows, so an image transformer cannot rotate by (y, x). Needed only by a conditional UNet. |
| [BendTorch#19](https://github.com/nuxyel/BendTorch/issues/19) | adaLN: no layer broadcasts a per-image vector over its tokens. Needed only by a conditional UNet. |
| [bendlang/bend#1632](https://github.com/bendlang/bend/issues/1632) | Clearer diagnostics for common mistakes. Items 9 and 10 are ours; the 84 rejections in `reference/corpus/qwenbend/` are ready-made inputs. |
| [bendlang/bend#1998](https://github.com/bendlang/bend/issues/1998) | A match binder that shadows a parameter prints the name twice and no scope. |
| [bendlang/bend#1980](https://github.com/bendlang/bend/issues/1980) | `Array<T>` accepts a mixed-depth `ANode` the compiled runtime cannot represent. |
| [bend-ml#26](https://github.com/nuxyel/bend-ml/issues/26) | The value law (get after set) as a Base question; the round-trip corollary waits on two Base lemmas. Our last word is on the record. |

## Closed

| Artifact | Outcome |
|---|---|
| [BendTorch#27](https://github.com/nuxyel/BendTorch/pull/27) | Merged 2026-11-03; released as 2.0.0. The optimizer configuration lives in the `Train`, `Train.with_cfg` changes it, `Opt.guard` refuses a kind mismatch with a NaN optimizer (four laws). Our sweep on it: three rates, three curves. |
| [BendTorch#22](https://github.com/nuxyel/BendTorch/pull/22) | Merged 2026-11-03 as part of 2.0.0: `Conv2d` in both lanes, `im2col`/`col2im`, two laws, the PyTorch checks. Rebased onto #27 and merged with its three review notes. |
| [BendTorch#24](https://github.com/nuxyel/BendTorch/issues/24) | Closed by 2.0.0. It was worse than the learning rate: a kind mismatch left the loss unchanged with no error, and `OptCfg.ok` ran only in `Train.new`. |
| [BendTorch#23](https://github.com/nuxyel/BendTorch/pull/23) | Closed 2026-11-03 as agreed: v1.3.0 wired the demo and `test_fullattn.py` into `make check` differently; nothing in ours to keep. |
| [BendTorch#17](https://github.com/nuxyel/BendTorch/issues/17) | Closed by `FullAttention` in v1.3.0. |
| [BendTorch#15](https://github.com/nuxyel/BendTorch/issues/15) | Closed 2026-11-03: with `Conv2d`, `GroupNorm` and `FullAttention` in, an unconditional UNet is expressible. A follow-up (#29) carries the general `k = s` round trip and the adjoint law. |
