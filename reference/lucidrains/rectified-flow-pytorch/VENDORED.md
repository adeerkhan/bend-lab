# Vendored: lucidrains/rectified-flow-pytorch

This directory is a **verbatim copy** of the package and the Oxford trainer from
[`lucidrains/rectified-flow-pytorch`](https://github.com/lucidrains/rectified-flow-pytorch) at commit

```
3ca498a138b4335c6e3c485d3b1c7446a7cb2ece   2026-11-03   "add ability to use a welford online normalizer with rectified flow or nano flow"
```

cloned on 2026-11-08. Nothing here is modified: the hashes in `VENDORED.sha256`
are the upstream bytes.

## What is copied, and what is not

Copied verbatim:

- `rectified_flow_pytorch/` — the whole package. `__init__.py` imports every
  submodule (`nano_flow`, `mean_flow`, `fit`, ...), so the package is the unit
  that imports unchanged.
- `train_oxford.py` — the reference's own Oxford-flowers recipe: 64x64,
  `dim=64`, batch 16, 70,000 steps, `predict='clean'`, `auto_normalize=True`,
  EMA. This is the recipe the comparison uses.
- `LICENSE` — MIT, Copyright (c) 2024 Phil Wang.

Not copied: the repository's other training scripts, tests, `images/`, `rf.png`
and `pyproject.toml`. They are not needed to import the package or to run the
Oxford recipe.

## Why

The flow-matching comparison needs the reference implementation, not a NumPy
mirror. The package's `Unet` is not the small U-Net of
`examples/diffusion/unet`: it is `dim=64` with `dim_mults=(1,2,4,8)`, RMSNorm
residual blocks, full/linear attention, cross-attention residuals, and a time
MLP. The comparison script is `reference/rectified_flow_compare.py`.

## Check the copy

```console
$ cd reference/lucidrains/rectified-flow-pytorch && sha256sum -c VENDORED.sha256
```
