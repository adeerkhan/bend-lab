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
| `cancer-knn` | breast_cancer (683x9) | `4c35090c` | accuracy | 1 | 0.978102 | 0.978102 | 1.022e-08 | 1 | 381 | 380 | 1318 | ok |
| `cancer-naive-bayes` | breast_cancer (683x9) | `4c35090c` | accuracy | 1 | 0.985401 | 0.985401 | -9.854e-09 | 0 | 22 | 22 | 535 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | first_loss | 1 | 1.00145 | 1.00145 | -7.65e-08 | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | data_mean | 1 | 0.474855 | 0.474855 | 1.689e-09 | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | final_loss | 1 | 1.01675 | 1.01675 | - | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | sample_mean | 1 | 0.493899 | 0.493899 | - | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-10` | cifar10 (256x3072) | `80f3ca81` | sample_std | 1 | 0.428974 | 0.428963 | - | 65 | 60864 | 60799 | 3008 | ok |
| `cifar-train` | cifar10 (256x3072) | `80f3ca81` | loss_drop | 1 | 1.00627 | 1.00696 | -0.0006853 | 78 | 94412 | 94334 | 34107 | ok |
| `cifar-train` | cifar10 (256x3072) | `80f3ca81` | final_loss | 1 | 0.996027 | 0.99535 | - | 78 | 94412 | 94334 | 34107 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | first_loss | 1 | 152.954 | 152.954 | -1.483e-05 | 0 | 103705 | 103705 | 207300 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | final_loss | 1 | 0.0649613 | 0.0569123 | - | 0 | 103705 | 103705 | 207300 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | sample_mean | 1 | 0.385003 | 0.385002 | - | 0 | 103705 | 103705 | 207300 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | sample_std | 1 | 0.599225 | 0.599225 | - | 0 | 103705 | 103705 | 207300 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | first_loss | 1 | 1.00389 | 1.00389 | -2.907e-08 | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | data_mean | 1 | 0.307565 | 0.307565 | 4.587e-09 | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | final_loss | 1 | 0.0416787 | 0.0444931 | - | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | sample_mean | 1 | 0.38456 | 0.380702 | - | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-digits` | optdigits (1024x64) | `e1b683cc` | sample_std | 1 | 0.354907 | 0.359419 | - | 2 | 148399 | 148397 | 12189 | ok |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | first_loss | 1 | 1.00919 | 1.00919 | -6.271e-09 | 1 | 624838 | 624837 | 42 | ok |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | data_mean | 1 | 0.307565 | 0.307565 | 4.587e-09 | 1 | 624838 | 624837 | 42 | ok |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | final_loss | 1 | 0.0711534 | - | - | 1 | 624838 | 624837 | 42 | ok |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | sample_mean | 1 | 0.295717 | - | - | 1 | 624838 | 624837 | 42 | ok |
| `diffusion-vit` | optdigits (1024x64) | `e1b683cc` | sample_std | 1 | 0.339538 | - | - | 1 | 624838 | 624837 | 42 | ok |
| `flow64-first` | flowers64-noise (2x4096) | `ab9a8682` | params | 1 | 413440 | 413440 | 0 | 0 | 2935 | 2935 | 317 | ok |
| `flow64-first` | flowers64-noise (2x4096) | `ab9a8682` | out_sq | 1 | 0.0290745 | 0.0290745 | -2.097e-08 | 0 | 2935 | 2935 | 317 | ok |
| `flow64-first` | flowers64-noise (2x4096) | `ab9a8682` | step_sq | 1 | 1.0198 | 1.0198 | -1.948e-06 | 0 | 2935 | 2935 | 317 | ok |
| `flow64-first` | flowers64-noise (2x4096) | `ab9a8682` | euler32_sq | 1 | 1.04371 | 1.04371 | -5.233e-07 | 0 | 2935 | 2935 | 317 | ok |
| `flowers-flow` | flowers64 (3874x64) | `566936b6` | params | 1 | 413440 | 413440 | 0 | 0 | 1209530 | 1209530 | 410 | ok |
| `flowers-flow` | flowers64 (3874x64) | `566936b6` | legibility_heldout | 1 | 0.484633 | 0.489027 | -0.004394 | 0 | 1209530 | 1209530 | 410 | ok |
| `flowers-flow` | flowers64 (3874x64) | `566936b6` | ceiling | 1 | 0.5621 | 0.5621 | - | 0 | 1209530 | 1209530 | 410 | ok |
| `flowers-flow` | flowers64 (3874x64) | `566936b6` | final_loss | 1 | 0.291029 | - | - | 0 | 1209530 | 1209530 | 410 | ok |
| `glm53-hybrid128` | glm53-corpus128 (128x1) | `57bd157b` | params | 1 | 74144 | 74144 | 0 | 0 | 1542 | 1542 | 0 | ok |
| `glm53-hybrid128` | glm53-corpus128 (128x1) | `57bd157b` | first_loss | 1 | 5.56347 | - | - | 0 | 1542 | 1542 | 0 | ok |
| `glm53-hybrid128` | glm53-corpus128 (128x1) | `57bd157b` | final_loss | 1 | 5.56347 | - | - | 0 | 1542 | 1542 | 0 | ok |
| `heart-naive-bayes` | heart_failure (299x12) | `e0771c8c` | accuracy | 1 | 0.95 | 0.95 | 0 | 1 | 9 | 8 | 769 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | mean_accuracy | 5 | 0.930667 ± 0.006 | 0.930667 ± 0.006 | -5.867e-08 ± 1.4e-08 | 1 | 68 | 67 | 1672 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_min | 5 | 0.84 ± 0.037 | 0.84 ± 0.037 | -2e-08 ± 1.8e-08 | 1 | 68 | 67 | 1672 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_max | 5 | 0.986667 ± 0.018 | 0.986667 ± 0.018 | -1.067e-08 ± 1.5e-08 | 1 | 68 | 67 | 1672 | ok |
| `iris-mlp` | iris (150x4) | `6f608b71` | accuracy | 1 | 0.973333 | 0.973333 | 2.667e-08 | 0 | 482 | 482 | 271 | ok |
| `mamba3-tiny-lm` | mamba3-byte-corpus-64 (64x128) | `3f42d353` | final_loss | 1 | 4.33305 | 4.33305 | -9.889e-07 | 0 | 523270 | 523270 | 4838 | ok |
| `mamba3-tiny-lm` | mamba3-byte-corpus-64 (64x128) | `3f42d353` | bits_per_byte | 1 | 6.24765 | 6.24765 | -1.623e-06 | 0 | 523270 | 523270 | 4838 | ok |
| `mamba3-tiny-lm` | mamba3-byte-corpus-64 (64x128) | `3f42d353` | heldout_hits | 1 | 1279 | 1279 | 0 | 0 | 523270 | 523270 | 4838 | ok |
| `mamba3-tiny-lm` | mamba3-byte-corpus-64 (64x128) | `3f42d353` | train_hits | 1 | 512 | 512 | 0 | 0 | 523270 | 523270 | 4838 | ok |
| `mamba3-tiny-lm` | mamba3-byte-corpus-64 (64x128) | `3f42d353` | w_sumsq | 1 | 2800.74 | 2800.74 | -0.0007304 | 0 | 523270 | 523270 | 4838 | ok |
| `mamba3-tiny-lm` | mamba3-byte-corpus-64 (64x128) | `3f42d353` | heldout_acc | 1 | 0.314715 | 0.314715 | 1.307e-08 | 0 | 523270 | 523270 | 4838 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | first_loss | 1 | 2.35914 | 2.35914 | 6.813e-07 | 0 | 26 | 26 | 41 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | final_loss | 1 | 0.00211434 | 0.1351 | - | 0 | 26 | 26 | 41 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | accuracy | 1 | 1 | 0.9375 | 0.0625 | 0 | 26 | 26 | 41 | ok |
| `unet-flowers` | flowers64 (7066x4096) | `87ee501c` | params | 1 | 298439 | - | - | 0 | unavailable | unavailable | unavailable | ok |
| `unet-flowers` | flowers64 (7066x4096) | `87ee501c` | final_loss | 1 | 0.0259002 | - | - | 0 | unavailable | unavailable | unavailable | ok |
| `unet-flowers` | flowers64 (7066x4096) | `87ee501c` | draws | 1 | 64 | - | - | 0 | unavailable | unavailable | unavailable | ok |
| `unet-flowers` | flowers64 (7066x4096) | `87ee501c` | legibility_held | 1 | 0.547348 | 0.573776 | -0.02643 | 0 | unavailable | unavailable | unavailable | ok |
| `unet-flowers` | flowers64 (7066x4096) | `87ee501c` | legibility_ceiling | 1 | 0.573776 | - | - | 0 | unavailable | unavailable | unavailable | ok |
| `unet-flowers` | flowers64 (7066x4096) | `87ee501c` | legibility_floor | 1 | 0.058161 | - | - | 0 | unavailable | unavailable | unavailable | ok |
| `unet-flowers` | flowers64 (7066x4096) | `87ee501c` | span | 1 | 0.948743 | - | - | 0 | unavailable | unavailable | unavailable | ok |
| `unet-flowers` | flowers64 (7066x4096) | `87ee501c` | spectrum_high | 1 | 0.0344778 | - | - | 0 | unavailable | unavailable | unavailable | ok |
| `unet-flowers` | flowers64 (7066x4096) | `87ee501c` | spectrum_low | 1 | 0.00174826 | - | - | 0 | unavailable | unavailable | unavailable | ok |
| `vit-first` | optdigits (1024x64) | `e1b683cc` | loss_at_0 | 1 | 1.0093 | 1.0093 | -2.403e-08 | 3 | 1062 | 1059 | 37 | ok |
| `vit-first` | optdigits (1024x64) | `e1b683cc` | tgt_sq | 1 | 0.999309 | 0.999309 | 1.388e-08 | 3 | 1062 | 1059 | 37 | ok |
| `vit-first` | optdigits (1024x64) | `e1b683cc` | out_sq | 1 | 0.00931402 | 0.00931402 | -3.017e-10 | 3 | 1062 | 1059 | 37 | ok |
| `vit-step` | synthetic-vit-step (1024x64) | `8a7882ec` | loss_after_3 | 1 | 0.921376 | - | - | 0 | 31422 | 31422 | 0 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | inertia | 1 | 50645.6 | 50645.6 | -0.00456 | 1 | 2289 | 2288 | 1073 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | cluster_sizes | 1 | [2384, 990, 3123] | [2384, 990, 3123] | 0 | 1 | 2289 | 2288 | 1073 | ok |
| `wine-pca` | wine (6497x11) | `7dcde7c7` | explained_variance_ratio | 1 | 0.275443 | 0.275443 | 3.956e-07 | 0 | 4016 | 4016 | 1170 | ok |

Bend and reference are mean ± sample deviation over the seeds when a task
has more than one; a single-seed task has no deviation. The dataset, seeds,
hub hashes and the git state of every run are in
`results/latest/<task>.json`; the timestamped reports are in `results/raw/`.
