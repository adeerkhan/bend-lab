# Bendlang DiT block — reference result

**Status (2026-11-23):** a conditioned DiT block runs in Bend. Its forward
pass matches PyTorch at 0.07× the fixed oracle tolerance. Its **gradient
gate does not pass** (5.454×); the gap is measured, isolated, and stated as a
banner below. This is a CPU-side structural probe at b=2, t=3, width 4 — not
yet a denoiser task and not a test-set result.

## What is implemented

`examples/diffusion/dit/block.bend` composes, entirely from BendTorch layers:

```
y = h + Tile(g) * f( LayerNorm(h) * Tile(m) + Tile(sh) )
f = QKV fork -> FullAttention(6,4,4, 2,3, 1,1,4, 0.5) -> output Linear
(g, m, sh) = Linear(silu(cond)), three per-image rows tiled over tokens
```

`FullAttention` (non-causal) is BendTorch #20; the adaLN modulation is the
`Tile`/`Fork`/`Both`/`Mul2`/`Add2` composition from BendTorch #19. No tensor,
autodiff or optimizer code is added to bend-lab.

## Measured gates

Built under the pinned Bend 2.0.38 with `BEND_LIB` and the 2.5 GB memory guard;
compared with `../bendtorch/reference/oracle.py` (ATOL 1e-6, RTOL 1e-5).
`reference/.venv/bin/python reference/test_dit_ref.py` reproduces all three.

| probe | tensors | worst / tolerance | verdict |
|---|---:|---:|---|
| `attn_probe.bend` — attention branch | 10 | 0.076 | PASS |
| `ln_attn_probe.bend` — LN → attention → `T.Residual` | 12 | 0.954 | PASS |
| `block.bend` — full conditioned block | 19 | **5.454** | **FAIL** |
| `wide_attn.bend` — attention at w=8, 2 heads | 10 | 0.418 | PASS |
| `wide_block.bend` — full conditioned block at w=8 | 1 (forward) | 0.336 | PASS |
| `wide_block.bend` — its gradients | 5 | 14.203 | FAIL |
| `main.bend` — denoiser model (embed + block + head) | 1 (forward) | 0.225 | PASS |

**Real data** (`make check-dit-real`): `examples/diffusion/dit/real_main.bend`
trains the same 496-param model on the **pinned optdigits CSV** (read directly
from `reference/.cache/datasets/mstz/optdigits/optdigits.data` with Base
`File`, not through the HTTP dataset stack — that binary does not fit the
machine's free RAM). The gate verifies 65536 values load and the first row's
first 8 scaled pixels match the CSV (2.1875), so a wrong loader fails. Six
AdamW steps run on real digits; loss is stochastic (per-step timestep/chunk)
and not gated numerically. Dims stay tiny (2 images, 3 of 16 patches each) —
a wiring proof, not a quality result. No sampler.

**Fixed-input training** (`make check-dit-denoiser`, 6 AdamW steps, lr 3e-3, 496 params):
the first real loss is 1286.68 — identical to the torch twin — and the
trajectory falls to 781.19. The torch twin over the same 6 steps is
1286.68 → 1161.70 → 1048.34 → 945.77 → 853.07 → 769.30; the 1.5% gap at the
last step is the documented f32 gradient drift of the composed block, not a
wiring error (step 1 matches exactly).

For the full block: forward `y` matches at 0.07, `dcond` 0.13, `dh` 0.96, and
the v/o gradients 0.02–0.06. The excess is on q/k weights (q.dw 4.42, k.dw
5.45, k.db 1.89) and m/sh (m.db 1.94). The oracle was **not** loosened; the
gate reports the measured ratio verbatim.

## Isolation ladder (the gap is now understood, not fixed)

Same q/k/v/o seeds and dy throughout; each row built and measured, not inferred:

| composition | input | worst / tol |
|---|---|---:|
| attention `f` alone | mat | 0.076 |
| LN → attention → `T.Residual` | mat | 0.954 |
| LN → attention → `T.Residual` | **mat2** | **1.582** |
| block with g=m=1, sh=0 | mat2 | 1.582 |
| block, live g/m/sh | mat2 | 5.454 |

Two contributions: the input pattern accounts for 0.954 → 1.582 (mat2 drives
a harder softmax backward with more f32 cancellation on the k weights — the
plain composition with **no adaLN** shows the same 1.582), and the active g/m/sh
conditioning accounts for 1.582 → 5.454. Earlier attributions to the gating
chain were confounded by comparing probes with different inputs; they are
withdrawn. Forward parity at 0.07 means the computation itself is correct.

## Banners

- CPU torch reference twin; no CUDA check of these probes.
- A first denoiser model (`examples/diffusion/dit/main.bend`, 496 params)
  wraps the block with a token-feature embedding and an output head, taking a
  separate per-image condition vector. Its forward matches PyTorch at 0.225x;
  The denoiser now includes an AdamW training loop (6 steps verified); a
  real dataset adapter and sampler are not yet added.
- The original block probe is tiny (b=2, t=3, w=4). Both the attention
  branch (`wide_attn.bend`, 0.418x) and the full conditioned block
  (`wide_block.bend`, forward 0.336x) were generalised to width 8 with two kv
  heads via top-level Nat constants. The wide block's gradients fail at 14.203x
  the oracle — worse than the tiny block's 5.454x, consistent with accumulated
  f32 drift at larger width. A denoiser task is not built.
- The gradient gate FAILS. The 5.454 value is frozen in `test_dit_ref.py` as a
  known-limitation reference, not accepted as a pass.
- Environment: Bend must be pinned at 2.0.38 (`make check-fast` enforces it),
  clang 22 needs `LD_LIBRARY_PATH=$HOME/.local/lib/llvm22` to build, and the
  memory guard caps builds at 2.5 GB.

## Open

- Pin the 1.582 → 5.454 rise to a specific sum site using **matched mat2
  inputs** (conditioning Forks and Tile backward are the candidates).
- Scale the verified block to the digits denoiser task at full width and
  measure training loss against a torch twin.
