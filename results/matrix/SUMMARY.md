# The matrix

`make matrix` over the declared cells: 7 run, 7 wrote a record, 0 failed. References: frozen.

A lane is a build fact: `gpu` links `libnvrtc`, `cpu` must not.

| task | lane | seed | linked | record | verdict |
|---|---|---:|---|---|---|
| `vit-step` | cpu | 0 | none | `results/matrix/cells/vit-step-20261101T080715Z.json` | ok |
| `vit-step` | gpu | 0 | libnvrtc.so.13 | `results/matrix/cells/vit-step-20261101T080855Z.json` | ok |
| `iris-crossval` | cpu | 42 | none | `results/matrix/cells/iris-crossval-20261101T080937Z.json` | ok |
| `iris-crossval` | cpu | 43 | none | `results/matrix/cells/iris-crossval-20261101T081019Z.json` | ok |
| `iris-crossval` | cpu | 44 | none | `results/matrix/cells/iris-crossval-20261101T081101Z.json` | ok |
| `attention-parity` | cpu | 0 | none | `results/matrix/cells/attention-parity-20261101T081102Z.json` | ok |
| `wine-pca` | cpu | 0 | none | `results/matrix/cells/wine-pca-20261101T081145Z.json` | ok |

## Cells a lane does not have

- `iris-crossval` / gpu: plain Bend, no device path: the GPU lane is BendTorch's
- `wine-pca` / gpu: plain Bend, no device path: the GPU lane is BendTorch's
- `attention-parity` / gpu: plain Bend, no device path: the GPU lane is BendTorch's
