# The lucidrains rectified-flow port, 16x16

This directory ports the layers of
[`lucidrains/rectified-flow-pytorch`](../../../reference/lucidrains/rectified-flow-pytorch/)
to Bend, one layer at a time, against a NumPy twin. The reference recipe
(64x64, dim 64, batch 16, 70k steps, EMA) is an 18-hour run on this machine and
is not the target: this example keeps the same layers at **16x16, batch 2,
dim 8, two levels** (64 -> 16 -> 8 -> 16 in the U), full attention at both
stages, with the XAttnRes history mix. The cap is **one hour** for a full run.
Learning Bend is the goal; the quality target is a small, gated example, not
state of the art.

## What is here

| file | what it is |
|---|---|
| `config.bend` | the toy shape: 2 images, 16x16, 8 channels, 8x8 after the first down |
| `model.bend` | `Down2` (pixel unshuffle), the XAttnRes max-pool resize and zero-query mix, `Pool2` with its arg-max backward, the learned-query mix forward and backward |
| `probe_down.bend` | `Down2` on a ramp: `0 8 128 136 1 9 129 137` |
| `probe_xattn.bend` | `0.5 * (maxpool(history) + x)`: `68 69 70 71 72 73 74 75` |
| `probe_pool_bwd.bend` | the scatter backward: `dx@136: 0 1 2 3 4 5 6 7` |
| `probe_mix.bend`, `probe_mix_trivial.bend` | the learned-query mix: `0 2 4 6 8 10 12 14`, and the trivial `0.5` / `1.5` cases |
| `probe_softmax.bend` | BendTorch's softmax is row-wise: `0.5 0.5 0.8808 0.1192` |

The whole-array gate is `reference/test_rectified_flow_layers.py`; it builds
`reference/probe_rectified_flow_layers.bend` once and compares fourteen outputs
(forward and backward for `Down2` and `Pool2`, the zero-query and learned
mixes, the mix's saved weights and its five gradients, and the softmax axis)
with its own NumPy twin:

```console
$ reference/.venv/bin/python reference/test_rectified_flow_layers.py
ok   down: 4096 values, max |diff| 0.000e+00 (atol 0)
...
14/14 gates passed
```

`make check` runs it with the rest. Building a probe needs the pinned Bend and
a C compiler (this machine's `~/.local/bin/clang` wrapper needs
`LD_LIBRARY_PATH=$HOME/.local/lib/llvm22`; `check_all.py` supplies it for the
build checks and leaves `--verdict` alone).

## The mix, and the bug the whole-array gate found

The reference's XAttnRes scores each history entry with a learned pseudo-query
and a softmax, instead of averaging blindly:

    logits  = [kh . q, kx . q] / sqrt(8)
    weights = softmax(logits, axis = 1)
    out     = weights[0] * vh + weights[1] * xc

The first row-only probes passed because row 0 reads the query at offset 0.
`dot8` then indexed the 1x8 query at `off + k`, so **every row after the first
read past the query's end and scored 0**, and the mix silently became a
50/50 average. The whole-array `mix2` gate (`kx = 2 * kh`, so each row's
weights differ) caught it; `dot8` now reads the query at 0..7 and moves only
the key/value row with `off`.

The backward uses BendTorch's own `Softmax` backward on the saved weights, so
the forward and backward share one definition. The query-gradient gate caught
a missing recursion in `dq_sum` while it was being written: the sum stopped
after row 0, which the eight-value gate saw at once (max |diff| 689). At this
point the query is an input; it becomes a parameter with the XAttn wrapper.

## What is next

1. the XAttnRes wrapper: normalization, the LoRA key, the projection, and the
   query as a parameter (the backward is in place);
2. the memory-KV `FullAttention`: four learned KV rows appended to each batch,
   Q padded, the extra outputs dropped;
3. the DAG U-Net that carries the history gradients across stages;
4. the `rf16` task in `reference/experiment.py` and a texture band in the
   metrics, so the port gets a report card instead of prose.

The layer gaps upstream are BendTorch
[#39](https://github.com/nuxyel/BendTorch/issues/39) (`MaxPool2d` with an
arg-max backward, a no-bias `Conv2d`) and bend-ml
[#35](https://github.com/nuxyel/bend-ml/issues/35) (`Mat` reshape/slice, so a
pixel unshuffle is not O(n^2) list code).

The PyTorch reference run (32,874,625 parameters, the full Oxford recipe) was
smoke-tested and then stopped deliberately: there is **no reference result**
for the 16x16 target yet. The environment and the vendored commit are in
`reference/lucidrains/rectified-flow-pytorch/VENDORED.md`.
