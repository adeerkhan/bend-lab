# cifar-train: the same denoiser, trained so its loss can fall

`examples/cifar`'s model and data, with the two things its 40-step plumbing run
lacks (ISSUE-036): a level drawn from {16, 32, 48} each step — 200 visits a
level in 600 steps instead of one in forty — and lr 1e-3, the rate the sweep
measured as the one that converges. The draw is a counter, `t = 16 + 16·(s mod 3)`,
so the run is reproducible; every "random" number is a counter as everywhere
else in this corpus.

```console
$ bendenv run -- bend examples/cifar_train/main.bend
```

| | Bend | NumPy twin |
|---|---:|---:|
| first loss | 1.0022775 | 1.0022774934 |
| final loss (report) | 0.9960275 | 0.9953497 |
| **`loss_drop`** (first over last) | **1.0062749** | 1.0069602 |

`loss_drop` is the gate, at 2e-3 relative — the measured delta between the two
implementations is 6.85e-4, and 600 AdamW steps carry their ulp differences
that far. `final_loss` is report-only: the loss falls by 0.6% in 600 steps,
which is a falling curve and not a trained model. The 0.529 that a fixed noise
draw reaches in the same budget is one pair memorised 600 times, not a
denoiser.

The task is `cifar-train` in `reference/experiment.py`: the same dataset pin,
`bench/plumbing_cifar` as its stage split, `bounds.cifar_step` as its floors,
and `reference/test_cifar.py`'s own loop as the twin.

```console
$ make experiment TASK=cifar-train
$ make report TASK=cifar-train
```

One step at these sizes (`bench/plumbing_cifar`, three runs): input 41 ms,
forward 36, target 4, loss and backward 44, 126 ms in all. The curve and the record are
in [`results/cifar-train/`](../../../results/cifar-train/).
