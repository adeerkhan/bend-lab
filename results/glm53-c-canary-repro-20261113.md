# GLM-5.3-Flash Stage-C learning canary — fresh run, 2026-11-13

The 16-label overfit canary from `results/glm53-c/canary-16.json`, re-run on
this lane. Bend and the PyTorch autograd twin both learn the 16 labels in
50 steps at lr 0.03. The two implementations agree to F32 tolerance, the
loss is monotonically decreasing, and the saved weights/grads match the
twin's flat tensors.

Machine: this lane, RTX 5060, Bend 2.0.38 (`~/.bend`),
`LD_LIBRARY_PATH=$HOME/.local/lib/llvm22`, torch (CPU, 1 thread) via
`reference/.venv-rf/bin/python`.

## Commands

```text
# bend train (build 21 s, run 27 s)
env LD_LIBRARY_PATH=$HOME/.local/lib/llvm22 BEND_MEM=6G \
  scripts/bend-guard.sh ~/.bend/bin/bend -o /tmp/opencode/glm53/train_c_lr \
  examples/language/glm53_flash/train_c.bend
mkdir -p /tmp/opencode/glm53/c_canary
/tmp/opencode/glm53/train_c_lr --steps 50 --lr-ppm 30000 --mask 16 \
  --data examples/language/glm53_flash/data/c --out /tmp/opencode/glm53/c_canary

# twin (run 5 s)
reference/.venv-rf/bin/python reference/glm53_c_twin.py \
  --steps 50 --lr 0.03 --mask 16 --data examples/language/glm53_flash/data/c
```

## Numbers

| metric | bend | twin | rel |
|---|---:|---:|---:|
| training loss, step 50 | 0.04192596 | 0.0419212505 | 1.1e-04 |
| post-update eval loss (`grad_loss` vs twin final) | 0.03683044 | 0.0368300416 | 1.1e-05 |
| weights, max rel | 5.544e-05 | — | — |
| grads, max rel | 1.182e-03 | — | — |

The recorded artifact (`results/glm53-c/canary-16.json`, from the earlier
run on this lane) reports bend `final_loss` 0.04192061 and
`final_weights_loss` 0.03682779 (rel 1.5e-05 / 6.1e-05 to the twin);
today's numbers are the same trajectory (rel 1.1e-04 / 1.1e-05). The
`/tmp/opencode/glm53/c_canary` directory holds `final_weights.bin` and
`final_grads.bin` (16672 bytes each = 4168 F32 values, the model's full
count).

## What this does and does not show

It shows the Stage-C trainer learns the first 16 labels of the fixed
batch on this lane, end-to-end, and the Bend and PyTorch paths agree. It
does **not** touch the published before-RL 0/24 number on the three
trained families (ISSUE-102): that evaluation needs the 25.7M checkpoint
and the RL rollout, neither of which is in this tree; see the ledger's
ISSUE-102 entry and `docs/agents/upstream.md` for the recorded numbers.
