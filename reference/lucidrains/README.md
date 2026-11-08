# The lucidrains references (vendored)

Two verbatim copies, both MIT licensed. They are **references**, not
dependencies: no Bend package imports them, `make check` does not run them, and
the Bend layer still writes no tensor, no autodiff and no optimizer.

| directory | upstream | commit | its gate |
|---|---|---|---|
| `denoising_diffusion_pytorch/` | [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | [`VENDORED.md`](VENDORED.md) | `reference/lucidrains_compare.py`, `reference/digits_ablation.py` |
| `rectified-flow-pytorch/` | [lucidrains/rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) | `rectified-flow-pytorch/VENDORED.md` | `reference/rectified_flow_compare.py` |

It exists because the diffusion examples needed a second opinion that is not
my own NumPy mirror. The mirror agrees with the example by construction; this
library is what the field actually uses, and comparing against it is how the
digits denoiser's failure was found (the local issue ledger, ISSUE-001).

## The environments

PyTorch is not in `reference/.venv` (that venv is NumPy-only, and it is what
`make check` uses). Each reference has its own, gitignored:

- **`reference/.venv-lucidrains`** — the denoising-diffusion reference. It was
  removed in the 2026-11-08 cleanup, so recreate it before running
  `lucidrains_compare.py` or `digits_ablation.py`:

  ```console
  $ python3.14 -m venv reference/.venv-lucidrains
  $ reference/.venv-lucidrains/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  $ reference/.venv-lucidrains/bin/pip install -r reference/lucidrains/requirements.txt
  ```

  The CPU index keeps it to ~200 MB instead of pulling the CUDA build.

- **`reference/.venv-rf`** — the rectified-flow reference (Python 3.13,
  torch 2.11.0+cu128, the CUDA build the RTX 5060 needs). It is what the
  port's comparison runs use, and it is pinned in `requirements-rf.txt`:

  ```console
  $ python3.13 -m venv reference/.venv-rf
  $ reference/.venv-rf/bin/pip install -r reference/lucidrains/requirements-rf.txt
  ```

## What runs against them

- `reference/lucidrains_compare.py` — trains the same denoiser task
  (noise prediction, T = 64, linear schedule) on the same 1024 optdigits rows
  as `examples/diffusion/digits/main.bend`, prints both schedules side by side,
  trains, samples with DDIM (16 calls) and with the ancestral loop (64), and
  writes `results/lucidrains-diffusion/`.

  ```console
  $ reference/.venv-lucidrains/bin/python reference/lucidrains_compare.py
  $ STEPS=200 reference/.venv-lucidrains/bin/python reference/lucidrains_compare.py   # smoke
  ```

- `reference/rectified_flow_compare.py` — trains the rectified-flow reference
  on the pinned 7,067/102 flowers split, 64x64, one grayscale channel, and
  writes `results/lucidrains-rectified-flow/` (gitignored). The full recipe is
  70k steps and was stopped after the smoke test; the 16x16 port is the
  target. `--sample` without `--checkpoint` loads the newest checkpoint under
  `--checkpoints`; `--resume` is a warm start (the vendored `Trainer` has no
  step resume):

  ```console
  $ reference/.venv-rf/bin/python reference/rectified_flow_compare.py --steps 200
  $ reference/.venv-rf/bin/python reference/rectified_flow_compare.py --sample --draws 64
  ```

- `reference/digits_ablation.py` — the controlled experiment that isolated
  the digits failure: the example's exact MLP and optimizer with init, input range and
  schedule changed one at a time.

  ```console
  $ reference/.venv-lucidrains/bin/python reference/digits_ablation.py
  ```

## What was taken from it (not copied, learned)

The parts that changed the Bend side, with the files they come from:

| Reference | Where | What Bend did |
|---|---|---|
| zero-mean init at `1/sqrt(fan_in)` | PyTorch `nn.Linear` default | `examples/diffusion/digits` `value()` rebuilt (ISSUE-001) |
| x̂₀ clamped to [-1, 1] before the posterior mean | `p_mean_variance`, `ddim_sample` (`maybe_clip`) | open — see the local issue ledger H2 |
| data normalized to [-1, 1] | `normalize_to_neg_one_to_one` | open — H3 |
| `linear_beta_schedule` = `linspace(1e-4·1000/T, 0.02·1000/T, T)` | `linear_beta_schedule` | divergence recorded (ISSUE-002) |
| sinusoidal embedding injected per block | `Unet` + `SinusoidalPosEmb` | ours concatenates 8 values into the input row — H4 |

The DDIM step itself was **already right**: `mlbend-generative`'s `sigma_of`
and `step_of` expand to the reference's `sigma` and `c` exactly (checked line
by line when the files were read, 2026-10-27).

Licence: MIT, Copyright (c) 2020 Phil Wang — see [`LICENSE`](LICENSE).
