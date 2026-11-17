# Lane compliance matrix

Generated 2026-11-17T14:08:43+00:00 by `make lane-matrix` from `results/latest/*.json` —
no task was run to produce this page. A cell is compliant when its
record passed and a frozen reference is on file; per-lane gate suites
(for example `docs/mamba3-lanes.md`) must still pass before numbers
are published. A mismatch across lanes is a banner, not a delta.

**20/20 cells compliant**

| task | lane | trainer | bend | schema | seeds | frozen | record |
|---|---|---|---|---|---|---|---|
| cancer-knn | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| cifar-10 | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| cifar-train | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| diffusion-2d | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| diffusion-vit | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| flow64-first | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| flowers-flow | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| glm53-hybrid128 | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| iris-crossval | cpu | bend | 2.0.38 | mlbend.experiment/3 | 5 | yes | PASS |
| mamba3-tiny-lm | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| vit-first | cpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| vit-step | gpu | bend | 2.0.38 | mlbend.experiment/3 | 1 | yes | PASS |
| attention-parity | unrecorded | bend | 2.0.38 | mlbend.experiment/1 | 1 | yes | PASS |
| cancer-naive-bayes | unrecorded | bend | 2.0.38 | mlbend.experiment/1 | 1 | yes | PASS |
| diffusion-digits | unrecorded | bend | 2.0.38 | mlbend.experiment/1 | 1 | yes | PASS |
| heart-naive-bayes | unrecorded | bend | 2.0.38 | mlbend.experiment/1 | 1 | yes | PASS |
| iris-mlp | unrecorded | bend | 2.0.38 | mlbend.experiment/1 | 1 | yes | PASS |
| transformer-periodic | unrecorded | bend | 2.0.38 | mlbend.experiment/1 | 1 | yes | PASS |
| wine-kmeans | unrecorded | bend | 2.0.38 | mlbend.experiment/1 | 1 | yes | PASS |
| wine-pca | unrecorded | bend | 2.0.38 | mlbend.experiment/1 | 1 | yes | PASS |

## What this page does not say

It does not rank lanes, replace the gates, or compare numbers across
lanes. It says only that each recorded lane has an independent
reference and a passing run.
