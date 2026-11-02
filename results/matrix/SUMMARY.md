# The matrix

`make matrix` over the declared cells: 17 run, 17 wrote a record, 0 failed. References: frozen.

A lane is a build fact: `gpu` links `libnvrtc`, `cpu` must not.

| task | lane | seed | linked | record | verdict |
|---|---|---:|---|---|---|
| `vit-step` | cpu | 0 | none | `results/matrix/cells/vit-step-20261101T170451Z.json` | ok |
| `vit-step` | gpu | 0 | libnvrtc.so.13 | `results/matrix/cells/vit-step-20261101T170630Z.json` | ok |
| `iris-crossval` | cpu | 42 | none | `results/matrix/cells/iris-crossval-20261101T170712Z.json` | ok |
| `iris-crossval` | cpu | 43 | none | `results/matrix/cells/iris-crossval-20261101T170754Z.json` | ok |
| `iris-crossval` | cpu | 44 | none | `results/matrix/cells/iris-crossval-20261101T170836Z.json` | ok |
| `attention-parity` | cpu | 0 | none | `results/matrix/cells/attention-parity-20261101T170837Z.json` | ok |
| `wine-pca` | cpu | 0 | none | `results/matrix/cells/wine-pca-20261101T170919Z.json` | ok |
| `cifar-train` | cpu | 0 | none | `results/matrix/cells/cifar-train-20261101T171338Z.json` | ok |
| `vit-first` | cpu | 0 | none | `results/matrix/cells/vit-first-20261101T171455Z.json` | ok |
| `transformer-periodic` | cpu | 0 | none | `results/matrix/cells/transformer-periodic-20261101T171501Z.json` | ok |
| `diffusion-2d` | cpu | 0 | none | `results/matrix/cells/diffusion-2d-20261101T171528Z.json` | ok |
| `iris-mlp` | cpu | 0 | none | `results/matrix/cells/iris-mlp-20261101T171608Z.json` | ok |
| `wine-kmeans` | cpu | 0 | none | `results/matrix/cells/wine-kmeans-20261101T171649Z.json` | ok |
| `cancer-knn` | cpu | 0 | none | `results/matrix/cells/cancer-knn-20261101T171734Z.json` | ok |
| `cancer-naive-bayes` | cpu | 0 | none | `results/matrix/cells/cancer-naive-bayes-20261101T171813Z.json` | ok |
| `heart-naive-bayes` | cpu | 0 | none | `results/matrix/cells/heart-naive-bayes-20261101T171852Z.json` | ok |
| `cifar-10` | cpu | 0 | none | `results/matrix/cells/cifar-10-20261101T172041Z.json` | ok |

## Tasks with no cell, and why

- `diffusion-digits` — 226 s: 226 s a cell (build 77 + run 148); gated by reference/test_diffusion_digits.py
- `diffusion-vit` — 724 s: 724 s a cell (build 100 + run 625, 3000 steps); gated by reference/test_diffusion_vit.py (first loss, forward, one step, DDIM, eta, legibility)
- `flowers-flow` — 1204 s: 1,204 s a cell (build 28 + run 1,176, 6,000 steps at the corpus preset); the task run is its own gate, and `make experiment TASK=flowers-flow` names the same numbers on its page

## Cells a lane does not have

- `iris-crossval` / gpu: plain Bend, no device path: the GPU lane is BendTorch's
- `wine-pca` / gpu: plain Bend, no device path: the GPU lane is BendTorch's
- `attention-parity` / gpu: plain Bend, no device path: the GPU lane is BendTorch's
- `cifar-train` / gpu: the device lane's exp differs from the host's; no measured tolerance for a 600-step curve yet
- `cifar-10` / gpu: the device lane's exp differs from the host's; the task has no measured device tolerance yet
- `vit-first` / gpu: the twin is one host window and the device lane's exp is not the host's; no measured tolerance for out_sq there yet
- `diffusion-2d` / gpu: the device lane's exp differs from the host's; no measured tolerance for this task's gate yet
- `flowers-flow` / gpu: the example imports bendtorch only: there is no device path for a cell to link libnvrtc against
