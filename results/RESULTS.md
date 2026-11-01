# Results

The latest report card of every task, written by `reference/experiment.py`.
`fetch` is the time to fill and hash the local cache; the Bend program reads
the same bytes through `MLBEND_DATA`, so on a warm cache it is 0 and `compute`
is the run itself. `Python` is the reference computation. The build time is
Bend's compiler and is reported separately.

Machine: Linux-7.2.9-200.fc44.x86_64-x86_64-with-glibc2.43 · 28 CPUs · Bend 2.0.38

| task | dataset | hash | metric | seeds | Bend | reference | delta | fetch ms | Bend ms | compute ms | Python ms | ok |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `attention-parity` | synthetic-attention (4x8) | `439ef1cd` | checksum | 1 | 1.76295 | 1.76295 | 3.342e-09 | 0 | 1 | 1 | 3 | ok |
| `cancer-knn` | breast_cancer (683x9) | `4c35090c` | accuracy | 1 | 0.978102 | 0.978102 | 1.022e-08 | 1 | 196 | 195 | 1617 | ok |
| `cancer-naive-bayes` | breast_cancer (683x9) | `4c35090c` | accuracy | 1 | 0.985401 | 0.985401 | -9.854e-09 | 0 | 22 | 22 | 535 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | first_loss | 1 | 1.00145 | 1.00145 | -7.65e-08 | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | data_mean | 1 | 0.474855 | 0.474855 | 1.689e-09 | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | final_loss | 1 | 1.01675 | 1.01675 | - | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | sample_mean | 1 | 0.493899 | 0.493899 | - | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | sample_std | 1 | 0.428974 | 0.428963 | - | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-train` | cifar10 (256x3072) | `80f3ca81` | loss_drop | 1 | 1.00627 | 1.00696 | -0.0006853 | 67 | 94427 | 94360 | 34120 | ok |
| `cifar-train` | cifar10 (256x3072) | `80f3ca81` | final_loss | 1 | 0.996027 | 0.99535 | - | 67 | 94427 | 94360 | 34120 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | first_loss | 1 | 152.954 | 152.954 | -1.483e-05 | 0 | 31919 | 31919 | 36063 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | final_loss | 1 | 0.0649613 | 0.0569123 | - | 0 | 31919 | 31919 | 36063 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | sample_mean | 1 | 0.385003 | 0.385002 | - | 0 | 31919 | 31919 | 36063 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | sample_std | 1 | 0.599225 | 0.599225 | - | 0 | 31919 | 31919 | 36063 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | first_loss | 1 | 1.00389 | 1.00389 | -2.907e-08 | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | data_mean | 1 | 0.307565 | 0.307565 | 4.587e-09 | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | final_loss | 1 | 0.0416787 | 0.0444931 | - | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | sample_mean | 1 | 0.38456 | 0.380702 | - | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | sample_std | 1 | 0.354907 | 0.359419 | - | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | first_loss | 1 | 1.00919 | 1.00697 | 0.002219 | 5 | 1493151 | 1493146 | 75 | FAIL |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | data_mean | 1 | 0.307565 | 0.307565 | 4.587e-09 | 5 | 1493151 | 1493146 | 75 | ok |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | final_loss | 1 | 0.0711534 | - | - | 5 | 1493151 | 1493146 | 75 | ok |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | sample_mean | 1 | 0.295717 | - | - | 5 | 1493151 | 1493146 | 75 | ok |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | sample_std | 1 | 0.339538 | - | - | 5 | 1493151 | 1493146 | 75 | ok |
| `heart-naive-bayes` | heart_failure (299x12) | `e0771c8c` | accuracy | 1 | 0.95 | 0.95 | 0 | 1 | 9 | 8 | 769 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | mean_accuracy | 5 | 0.930667 ± 0.006 | 0.930667 ± 0.006 | -5.867e-08 ± 1.4e-08 | 0 | 76 | 76 | 763 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_min | 5 | 0.84 ± 0.037 | 0.84 ± 0.037 | -2e-08 ± 1.8e-08 | 0 | 76 | 76 | 763 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_max | 5 | 0.986667 ± 0.018 | 0.986667 ± 0.018 | -1.067e-08 ± 1.5e-08 | 0 | 76 | 76 | 763 | ok |
| `iris-mlp` | iris (150x4) | `6f608b71` | accuracy | 1 | 0.973333 | 0.973333 | 2.667e-08 | 0 | 482 | 482 | 271 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | first_loss | 1 | 2.35914 | 2.35914 | 6.813e-07 | 0 | 26 | 26 | 41 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | final_loss | 1 | 0.00211434 | 0.1351 | - | 0 | 26 | 26 | 41 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | accuracy | 1 | 1 | 0.9375 | 0.0625 | 0 | 26 | 26 | 41 | ok |
| `vit-step` | synthetic-vit-step (1024x64) | `8a7882ec` | loss_after_3 | 1 | 0.921376 | - | - | 0 | 31422 | 31422 | 0 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | inertia | 1 | 50645.6 | 50645.6 | -0.00456 | 1 | 2289 | 2288 | 1073 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | cluster_sizes | 1 | [2384, 990, 3123] | [2384, 990, 3123] | 0 | 1 | 2289 | 2288 | 1073 | ok |
| `wine-pca` | wine (6497x11) | `7dcde7c7` | explained_variance_ratio | 1 | 0.275443 | 0.275443 | 3.956e-07 | 0 | 4016 | 4016 | 1170 | ok |

Bend and reference are mean ± sample deviation over the seeds when a task
has more than one; a single-seed task has no deviation. The dataset, seeds,
hub hashes and the git state of every run are in
`results/latest/<task>.json`; the timestamped reports are in `results/raw/`.
