<p align="center">
  <img src="docs/assets/bend-hero.png" alt="bend-lab: machine learning, built in Bend" width="100%">
</p>

# bend-lab

**Machine learning, built in Bend.** Models are written from scratch in
[Bend](https://github.com/bendlang/bend) and trained on
[BendTorch](https://github.com/nuxyel/BendTorch). The repository is an
experiment corpus, not a library. bend-ml owns the tensors and the laws,
BendTorch owns the training, and this repository uses them, measures them and
records the results.

Every number comes from a program in the repository and is checked three ways:
against a NumPy twin written at the same operation order, against an outside
implementation where one exists (scikit-learn, PyTorch, the lucidrains UNet),
and against an earlier version of itself through frozen probes. When the three
disagree, the disagreement is the result. A number that was not measured is not
written down.

<p align="center">
  <a href="https://github.com/adeerkhan/bend-lab/actions/workflows/ci.yml"><img src="https://github.com/adeerkhan/bend-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

## Results

![Legibility by arm](docs/assets/figures/legibility.webp)

The diffusion track is complete through a ViT denoiser: 2×2 patches, 16 tokens,
three pre-LN blocks, 100,736 parameters and 3000 steps on the first 1024
optdigits images. Legibility is the mean best absolute correlation of a sample
with any training digit. Random images score 0.326, the digits MLP scores 0.408,
and the gate is 0.75.

| arm (same data, sampler and starting noise) | DDIM 16 | ancestral 64 | DDIM MSE | ancestral MSE |
|---|---:|---:|---:|---:|
| **Bend-ViT** | **0.787** | **0.811** | 0.0598 | 0.0561 |
| PyTorch, same architecture | 0.790 | 0.779 | 0.0583 | 0.0633 |
| lucidrains UNet (reference) | 0.932 | 0.940 | 0.0202 | 0.0183 |

The Bend model is within 0.003 of the PyTorch twin on DDIM and ahead of it on
the ancestral sampler, with the same 4096 starting samples. Its cloud mean and
standard deviation are 0.2957 / 0.3395 against the data's 0.3076 / 0.3767, and
4087 of 4096 pixels land in [0, 1].

The gap closed in the training loop, not in the architecture. Bend drew one
timestep per batch while the reference draws one per image, worth +0.36
legibility. The evidence is in `results/diffusion-vit/`, and the three-arm
comparison is `results/compare/index.html`.

## Diffusion

![ViT loss](docs/assets/figures/loss.webp)

The loss falls from 1.009 to 0.071 over 3000 steps, with a minimum of 0.058.
The frozen probe pins the forward pass, one ancestral step, the 16-step DDIM
cloud and the eta=0 invariance to the NumPy twin; the legibility gate is 0.75.
The same task also runs on a dense MLP over real digits, and on two moons with
an ancestral sampler and DDIM.

![DDIM samples](docs/assets/figures/samples.webp)

## Flow matching

![Rectified flow loss](docs/assets/figures/flow.webp)

Rectified flow replaces the diffusion chain with one vector field and an Euler
ODE: `x_t = noise*(1-t) + data*t`, target `v = data - noise`, MSE loss, and a
sample integrated over 32 steps of `linspace(0, 1)`. The 2D example uses three
linear layers with SiLU, full batch, 6000 AdamW steps, and reaches a last-100
loss of 0.886. Its samples have mean 0.049 / -0.048 and standard deviation
0.610 / 0.675 against the data's 0.000 / 0.000 and 0.577 / 0.659, with 112 of
128 inside the data's box. A frozen probe and NumPy twin for this example are
still to be added.

## Running on your machine

A Bend program's shapes are type-level constants, so the model is chosen before
the compile. Two commands make that choice explicit instead of accidental:

```
make doctor                 # RAM, cores, GPU, the build budget, the preset that fits
make flow64 PRESET=tiny    # build and run the 64x64 flow inside that budget
```

`make doctor` reads the machine and prints a build budget (55% of RAM, capped at
12 GB). Every build runs under `scripts/bend-guard.sh`, which puts it in a memory
cgroup with swap off: a compile that grows past `BEND_MEM` (default 6 GB) is
killed with a message, and the desktop stays up. That guard exists because it
did not, twice: `bend -o` on the 64x64 flow reached 9.3 GB and took the session
into swap with it. The measured costs and the bisect are in `docs/NOTES.md`.

The preset is a property of the program, not a runtime flag:

| preset | images/step | width | rows | compile memory |
|---|---|---|---|---|
| `tiny` | 1 | 64 | 64 | over 7.5 GB, not yet buildable |
| `small` | 1 | 96 | 64 | not measured |
| `medium` | 2 | 128 | 128 | not measured |
| `full` | 8 | 128 | 512 | past 8 GB, killed |

None of them builds on a 15 GB machine yet. The cost is not the model's size:
a one-block, 64-wide model costs more to compile than the 512-row three-block
one did before the file was split, which points at the number of large recursive
functions in one program rather than at the dimensions. The ViT example, the
same architecture at 512 rows, does build in 82 s under 8 GB, so the target is
reachable; splitting training and sampling into two programs is the next
experiment. Nothing here is hidden behind a default that silently OOMs.


## Layout

```
examples/          what is run: one directory per task, with main.bend, README.md and plot.svg
  classification/    knn, logistic, mlp, naive_bayes, naive_bayes_heart, crossval
  diffusion/         two_moons, flow_matching, digits, vit
  transformers/      attention, periodic
  unsupervised/      kmeans, pca
{metrics,hf,datasets,preprocess,split,select,cluster,pipeline,report,generative}/
                   the ten packages published to BendHub
reference/         what proves it: twins, probes, gates, comparisons, the task registry
results/           what it proved: run.json, REPORT.md, plots, the three-arm comparison
docs/              PLAN, AUDIT, NOTES, the agent handoff, assets/ (hero and figures)
papers/            paper comparisons: the claim, the reduced protocol, the two arms
bench/             compute-only benchmarks
```

The rule for the tree: the package directories are what gets published,
`examples/` is what runs, `reference/` is what proves it, and `results/` is what
it proved. Moving the packages under `packages/` and the verification layer
under `apparatus/` is the next structural step. Those moves touch `bend.toml`
and many paths, so they go in one commit with `make check` behind them.

## Quickstart

Bend 2.0.38, Lean 4.34.0, clang and the Python environment, one command:

```sh
make setup                                               # toolchain, lock, environment
bendenv run -- bend examples/diffusion/vit/main.bend     # train and sample the ViT
make check                                               # the gate: proofs, kernels, twins, comparisons
```

`make check` is the gate. It checks the proofs and the kernel with Lean, runs
the programs that must be refused, and compares every measured number with a
NumPy or scikit-learn reference. On a cold machine it builds each program once,
which is the slow part.

## The corpus

| example | data | what is checked | result |
|---|---|---|---|
| `diffusion/vit` | optdigits, 1024 × 8×8 | legibility of 4096 samples, three arms | **0.787 / 0.811** (torch 0.790 / 0.779) |
| `diffusion/digits` | optdigits, 1024 × 8×8 | first loss against the twin, cloud statistics | 1.0039 (twin agrees to 2.9e-8) |
| `cifar` | cifar10, 256 × 32×32×3 | first loss and data mean against the twin | 1.0014 (twin agrees to 2.5e-5) |
| `diffusion/two_moons` | two moons, 128 × 2 | loss curve, cloud mean and std | 152.95 → 0.065 |
| `diffusion/flow_matching` | two moons, 128 × 2 | last-100 loss, sample mean/std, box | 0.886, 112/128 in the box |
| `transformers/periodic` | periodic 0…7, 32 × 8 | loss and next-token accuracy | 2.359 → 0.002, accuracy 1.00 |
| `transformers/attention` | synthetic 4 × 8 | every output against NumPy | checksum 1.76295, delta 3.3e-9 |
| `classification/crossval` | iris, 150 × 4 | five stratified folds, no leakage | 0.9307 ± 0.006 (sklearn 0.9307) |
| `classification/knn` | breast cancer, 683 × 9 | accuracy on the same split | 0.978102 (sklearn 0.978102) |
| `classification/naive_bayes` | breast cancer, 683 × 9 | accuracy on the same split | 0.985401 (sklearn 0.985401) |
| `classification/naive_bayes_heart` | heart failure, 299 × 12 | accuracy on the same split | 0.950000 (sklearn 0.950000) |
| `classification/mlp` | iris, 150 × 4 | accuracy, 400 SGD steps | 0.973333 (sklearn 0.973333) |
| `classification/logistic` | iris, 150 × 4 | the loss over four steps | 1.0986 → 1.0672 |
| `unsupervised/kmeans` | wine, 6497 × 11 | inertia and cluster sizes | 50645.64, [2384, 990, 3123] |
| `unsupervised/pca` | wine, 6497 × 11 | explained variance ratio | 0.275443 (sklearn 0.275443) |

`make results` writes the full table, with the delta and tolerance of every
task, to `results/RESULTS.md`. The dataset, seed, hub hashes and git state of
each run are in `results/latest/`.

## Method

1. An example in `examples/<area>/<task>/main.bend` trains and prints its
   numbers.
2. A twin in `reference/` mirrors it in NumPy at the same operation order.
3. A gate in `reference/test_<task>.py` compares the two and runs in
   `make check`.
4. A result in `results/<task>/` records the run, the report card and the plots.

A number that was not measured does not go in a README, a report card or a
result.

Run one task, or the whole corpus, through the experiment layer:

```sh
reference/.venv/bin/python reference/experiment.py --list
reference/.venv/bin/python reference/experiment.py --task iris-crossval
```

Each run writes a report card and a manifest: the git commit, the Bend version,
every package at its hub hash, the machine, the seed and the tolerances. The
README figures are generated from those result files with `make figures`.

## The GPU lane

`bend.toml` pins `bendtorch@1.1.1.0` and `bendtorch-gpu@1.1.1.0`. The ViT builds
with eight `LinearNBF` and one `F.Attention`, links `libnvrtc`, and executes on
an RTX 5060. `nvidia-smi` shows the process with 104 MiB on the card, and the
frozen values are bit-identical to the CPU lane.

The earlier 5m44 either way was measured against the data path, not the device.
After the builder fix the step is 131 ms and 95% of it is forward and backward,
so the device lane is worth re-measuring. The split is in
`results/PLUMBING.md`.

## Papers

Paper comparisons live in `papers/`, one folder per paper, with the paper's
claims, the reduced protocol used here, and the Python and Bend arms. The first
entry, TEM, is a negative result: the 64x64 contrastive protocol collapses to a
constant energy, the denoising variants move toward the data but need far more
compute, and the Bend port is blocked by the missing full attention, RoPE and
adaLN. Paper numbers are never presented as reproduced.

## Upstream

Everything this repository found in Bend, filed at
[bendlang/bend](https://github.com/bendlang/bend), and what came back. Every
issue carries the program that found it; every pull request was tested against
the programs in `examples/` and `reference/` here.

**Open**

| Issue | What it found |
|---|---|
| [#2059](https://github.com/bendlang/bend/issues/2059) | Base states no bit facts about `U32` (mask, halving, the branch bit, the subtraction bound), so the value round-trip law for `Mat` stays "tested, not proved". Six lemmas are proved in the meantime and offered with the program inline. |
| [#1998](https://github.com/bendlang/bend/issues/1998) | A match binder that shadows a parameter is reported with the name printed twice and no scope. |

**Closed**

| Issue | Outcome |
|---|---|
| [#1980](https://github.com/bendlang/bend/issues/1980) | `Array<T>` accepts a mixed-depth `ANode` the compiled runtime cannot represent. Closed as a duplicate. |
| [#1962](https://github.com/bendlang/bend/issues/1962) | `Array.new`'s depth argument was undocumented, and a length gave "an array past the deepest block class 31". Fixed by [#1963](https://github.com/bendlang/bend/pull/1963) and [#1970](https://github.com/bendlang/bend/pull/1970). |
| [#1961](https://github.com/bendlang/bend/issues/1961) | `import` refused a path segment starting with a digit while the error said digits are allowed. Fixed by [#1964](https://github.com/bendlang/bend/pull/1964). |
| [#1894](https://github.com/bendlang/bend/issues/1894) | The run path overflows its stack on a 32k-element list while `--check-only` and `-o` are fine. Fixed by [#1921](https://github.com/bendlang/bend/pull/1921). |
| [#1868](https://github.com/bendlang/bend/issues/1868) | A pure let inside a do block was refused with a message pointing at the previous statement. Fixed by [#1889](https://github.com/bendlang/bend/pull/1889), after [#1870](https://github.com/bendlang/bend/pull/1870). |
| [#1863](https://github.com/bendlang/bend/issues/1863) | The GPU build recipe existed only in a comment in `cli_build`. Fixed by [#1888](https://github.com/bendlang/bend/pull/1888), after [#1871](https://github.com/bendlang/bend/pull/1871). |
| [#1848](https://github.com/bendlang/bend/issues/1848) | The two spellings of an effect (`IO<Unit>` and `IO(Unit)`) both named the wrong side. Closed; [#1835](https://github.com/bendlang/bend/pull/1835) was the attempt. |
| [#1789](https://github.com/bendlang/bend/issues/1789) | Base could not read a number from a string. Fixed in Base. |
| [#1716](https://github.com/bendlang/bend/issues/1716) | A bare constructor name as a match pattern silently became a catch-all when the datatype was imported. Fixed by [#1738](https://github.com/bendlang/bend/pull/1738), after [#1717](https://github.com/bendlang/bend/pull/1717). |

**Pull requests opened from here**

Merged: [#1880](https://github.com/bendlang/bend/pull/1880) `Array.get_unchecked`, an `@unsafe` read without the
wrap mask; [#1845](https://github.com/bendlang/bend/pull/1845) `File.read` fills its buffer instead of answering one
`read(2)`; [#1748](https://github.com/bendlang/bend/pull/1748) the gate counts caps with one tokenizer;
[#1655](https://github.com/bendlang/bend/pull/1655) an async-signal-safe SIGSEGV/SIGBUS trap;
[#1654](https://github.com/bendlang/bend/pull/1654) the bug form asks for `bend version`;
[#1653](https://github.com/bendlang/bend/pull/1653) effects scan with the source's own comment rules.

Closed with the maintainer's own fix landing instead: [#1820](https://github.com/bendlang/bend/pull/1820) the native
spin's poll period, [#1769](https://github.com/bendlang/bend/pull/1769) the gate pin's columns,
[#1715](https://github.com/bendlang/bend/pull/1715) native C intrinsics for `Nat.pow`, `U32.log2` and `Map.msb.u`,
[#1652](https://github.com/bendlang/bend/pull/1652) and [#1651](https://github.com/bendlang/bend/pull/1651) the JS
lane, and the seven diagnostic prototypes
[#1731](https://github.com/bendlang/bend/pull/1731)–[#1737](https://github.com/bendlang/bend/pull/1737) for
[#1632](https://github.com/bendlang/bend/issues/1632).

## Status

`make check` is 121/121. Diffusion is complete through the ViT, with the gate at
0.75 and the Bend model measured at 0.795 (DDIM) and 0.807 (DDPM). Flow matching
has a measured 2D example and its frozen gate (`test_flow2d.py`, six gates). CIFAR-10 has its
3072-output DDPM: 256 images, 40 steps, build 50.8 s and run 65.4 s, first loss
and data mean gated against the NumPy twin and `RESULTS.md` carrying the rows. The
flowers U-Net is at five levels (64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 4 -> ... -> 64),
298,439 parameters, held-out legibility 0.547, gated against NumPy and PyTorch. The
rectified-flow port's layers are gated whole-array against a NumPy twin
(`test_rectified_flow_layers.py`, fourteen gates). The GLM Stage A harness is
now a trainable dense baseline: 74,144 parameters, a 20-step AdamW twin gate,
a 400-step record, and a frozen greedy dev/confirm manifest (`test_glm53_dense.py`).

## Licence

MIT. See `LICENSE`.
