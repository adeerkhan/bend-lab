# mamba3 ten-step train — recorded curve

Recorded by `reference/test_mamba3_train_curve.py` (regenerated on every run).

Command: `reference/.venv/bin/python reference/test_mamba3_train_curve.py`
Build 9.3 s, run 20.41 s wall. Bend 2.0.38, reference/.venv.

Custom trainer (ten steps, lr 1e-3), ten losses:

```
2.1568611 2.1549881 2.1531188 2.1512539 2.1493928 2.1475356 2.1456823 2.1438324 2.1419866 2.1401443
```

- `train_loss[0]` 2.1568611 → `train_loss[9]` 2.1401443
- `train_tokens` [3, 6, 0, 5, 0, 7, 0, 7]
- layer final loss (`layer_loss`): 2.1404066
- relative gap |layer-custom|/|layer|: 1.23e-04 (gate 1e-03) — PASS
- tokens equal: yes
- w_in head: max rel 1.27e-07 (gate 1e-05)
- E head: max rel 7.06e-08 (gate 1e-05)
- Wh head: max rel 3.64e-08 (gate 1e-05)
- w_out head: max rel 8.93e-07 (gate 1e-05)

Per-step losses are not readable from outside `T.Train` (bendtorch#86 —
`Train` carries a single `loss: F32`, the final step only), so no
per-step curve can be recorded from the layer path; this file records
the custom curve and the final-step comparison only.
