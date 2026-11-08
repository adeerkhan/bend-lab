# Vendored: vukrosic/glm-5.3-flash-from-scratch

A copy of the educational 25.7M-parameter model and its small training/evaluation
harness, from commit

```
96d9c7c3bf850ca2500a41b8184b4efb8e07890f   2026-08-28   "Add pretraining research experiments and final course slides"
```

MIT licensed; see `LICENSE`. The upstream repository is
https://github.com/vukrosic/glm-5.3-flash-from-scratch.

This is **not** the 320B released GLM-5.3-Flash. The upstream README is explicit
that it is a scaled, educational implementation: 12 layers, width 192, 8 routed
experts at top-2, four residual streams, a byte tokenizer, and synthetic Python
task families. It is vendored as a reference and a twin; no Bend package imports
it and `make check` does not run its training.

Copied: `glm53_flash/`, `tests/`, `scripts/train_pretrain.py`,
`scripts/evaluate.py`, `experiments/full_25m_vision_digit_pilot.py`,
`requirements.txt`, `LICENSE`, `README.md`. Omitted: `artifacts/`, `slides/`,
`thumbnails/`, the RL/vision scripts and the research experiments that are not
needed for the Stage A/B/C twins.

The upstream pins `torch==2.5.1` and Python 3.12. The local twin uses
`reference/.venv-rf` (Python 3.13, torch 2.11.0+cu128), because torch 2.5.1
predates the RTX 5060's compute capability. The model and its 12 tests pass
under that environment; exact published digits are not reproduced because the
released checkpoint weights are not distributed.
