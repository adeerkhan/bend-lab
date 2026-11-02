# Upstream issues and pull requests

Every artifact this repository has filed or opened upstream, newest work first.
The local issue numbers in `.local/issues-hit.md` map to these; a local issue
with no entry here was fixed in this repository.

## Open

| Artifact | What it asks |
|---|---|
| [BendTorch#24](https://github.com/nuxyel/BendTorch/issues/24) | `Train.step`'s config silently overrides `Train.new`'s; a step at 0.001 on a Train made at 0.1 reads the same loss for every Train. |
| [BendTorch#22](https://github.com/nuxyel/BendTorch/pull/22) | `Conv2d` in both lanes, with im2col and col2im on the device. Rebased on v1.3.0; waiting on review. |
| [BendTorch#23](https://github.com/nuxyel/BendTorch/pull/23) | The full-attention demo runs in `make check`. |
| [BendTorch#18](https://github.com/nuxyel/BendTorch/issues/18) | RoPE: the only rotation is 1D over rows, so an image transformer cannot rotate by (y, x). |
| [BendTorch#19](https://github.com/nuxyel/BendTorch/issues/19) | adaLN: no layer broadcasts a per-image vector over its tokens. |
| [BendTorch#15](https://github.com/nuxyel/BendTorch/issues/15) | `Conv2d` and `GroupNorm` (the UNet work). |
| [bend-ml#26](https://github.com/nuxyel/bend-ml/issues/26) | The value law (get after set) as a Base question. We adopt nuxyel's proof; the PR waits on his word. |
| [bend-ml#23](https://github.com/nuxyel/bend-ml/issues/23) | A counter-RNG fill bang (uniform/gauss) for tensor-gpu; the CIFAR step's target noise now uses it. |
| [bendlang/bend#1632](https://github.com/bendlang/bend/issues/1632) | Clearer diagnostics for common mistakes. Items 9 and 10 are ours; item 11 is the Nat past 2^48 - 1 stop with no location. |
| [bendlang/bend#1998](https://github.com/bendlang/bend/issues/1998) | A match binder that shadows a parameter prints the name twice and no scope. |
| [bendlang/bend#1980](https://github.com/bendlang/bend/issues/1980) | `Array<T>` accepts a mixed-depth `ANode` the compiled runtime cannot represent. |
| [bendlang/bend#1934](https://github.com/bendlang/bend/issues/1934) | Whether Base should state `Array`'s own laws. |

## Closed

| Artifact | Outcome |
|---|---|
| [BendTorch#17](https://github.com/nuxyel/BendTorch/issues/17) | Closed by `FullAttention` in v1.3.0. |
| [BendTorch#12](https://github.com/nuxyel/BendTorch/issues/12) | Retracted and closed by us: `package.sh` does publish the fast and gpu lanes. |
| [bend-ml#27](https://github.com/nuxyel/bend-ml/pull/27) | The tensor-array round trip at 64x64 and what a shape test can see. |
| [bend-ml#20](https://github.com/nuxyel/bend-ml/issues/20) | The autodiff design question; closed, BendTorch is the answer. |
| [bendlang/bend#1961](https://github.com/bendlang/bend/issues/1961) | The import-digit message; fixed upstream. |
| [bendlang/bend#1894](https://github.com/bendlang/bend/issues/1894) | The run path's stack overflow; closed as a known limit. |
