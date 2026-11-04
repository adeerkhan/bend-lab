# The matrix

`make matrix` over the declared cells: 19 run, 19 wrote a record, 0 failed. References: frozen.

A lane is a build fact: `gpu` links `libnvrtc`, `cpu` must not.

| task | lane | threads | seed | linked | record | verdict |
|---|---|---|---:|---|---|---|
| `vit-step` | cpu | default | 0 | none | `results/matrix/cells/vit-step-20261104T091213Z.json` | ok |
| `vit-step` | gpu | default | 0 | libnvrtc.so.13 | `results/matrix/cells/vit-step-20261104T091442Z.json` | ok |
| `iris-crossval` | cpu | default | 42 | none | `results/matrix/cells/iris-crossval-20261104T091539Z.json` | ok |
| `iris-crossval` | cpu | default | 43 | none | `results/matrix/cells/iris-crossval-20261104T091636Z.json` | ok |
| `iris-crossval` | cpu | default | 44 | none | `results/matrix/cells/iris-crossval-20261104T091733Z.json` | ok |
| `attention-parity` | cpu | 1 | 0 | none | `results/matrix/cells/attention-parity-20261104T091734Z.json` | ok |
| `attention-parity` | cpu | 16 | 0 | none | `results/matrix/cells/attention-parity-20261104T091734Z.json` | ok |
| `wine-pca` | cpu | default | 0 | none | `results/matrix/cells/wine-pca-20261104T091831Z.json` | ok |
| `cifar-train` | cpu | default | 0 | none | `results/matrix/cells/cifar-train-20261104T092428Z.json` | ok |
| `vit-first` | cpu | default | 0 | none | `results/matrix/cells/vit-first-20261104T092617Z.json` | ok |
| `flow64-first` | cpu | default | 0 | none | `results/matrix/cells/flow64-first-20261104T092644Z.json` | ok |
| `transformer-periodic` | cpu | default | 0 | none | `results/matrix/cells/transformer-periodic-20261104T092653Z.json` | ok |
| `diffusion-2d` | cpu | default | 0 | none | `results/matrix/cells/diffusion-2d-20261104T092730Z.json` | ok |
| `iris-mlp` | cpu | default | 0 | none | `results/matrix/cells/iris-mlp-20261104T092826Z.json` | ok |
| `wine-kmeans` | cpu | default | 0 | none | `results/matrix/cells/wine-kmeans-20261104T092919Z.json` | ok |
| `cancer-knn` | cpu | default | 0 | none | `results/matrix/cells/cancer-knn-20261104T093019Z.json` | ok |
| `cancer-naive-bayes` | cpu | default | 0 | none | `results/matrix/cells/cancer-naive-bayes-20261104T093111Z.json` | ok |
| `heart-naive-bayes` | cpu | default | 0 | none | `results/matrix/cells/heart-naive-bayes-20261104T093202Z.json` | ok |
| `cifar-10` | cpu | default | 0 | none | `results/matrix/cells/cifar-10-20261104T093432Z.json` | ok |

## Tasks with no cell, and why

- `diffusion-digits` — 226 s: 226 s a cell (build 77 + run 148); gated by reference/test_diffusion_digits.py
- `diffusion-vit` — 724 s: 724 s a cell (build 100 + run 625, 3000 steps); gated by reference/test_diffusion_vit.py (first loss, forward, one step, DDIM, eta, legibility)
- `flowers-flow` — 1204 s: 1,204 s a cell (build 28 + run 1,176, 6,000 steps at the corpus preset): the forward and the sampler are gated elementwise by reference/test_flow64.py, whose cheap cell is `flow64-first` (52 s); the 6,000-step run's legibility stays the task's own measured gate

## Cells a lane does not have

- `iris-crossval` / gpu: plain Bend, no device path: the GPU lane is BendTorch's
- `wine-pca` / gpu: plain Bend, no device path: the GPU lane is BendTorch's
- `attention-parity` / gpu: plain Bend, no device path: the GPU lane is BendTorch's
- `cifar-train` / gpu: the device lane's exp differs from the host's; no measured tolerance for a 600-step curve yet
- `cifar-10` / gpu: the device lane's exp differs from the host's; the task has no measured device tolerance yet
- `vit-first` / gpu: the twin is one host window and the device lane's exp is not the host's; no measured tolerance for out_sq there yet
- `diffusion-2d` / gpu: the device lane's exp differs from the host's; no measured tolerance for this task's gate yet
- `flowers-flow` / gpu: no device run is measured: the example's own build links no libnvrtc (ldd on the probe binary), and the twin is one host window
