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
- Tiny shapes (b=2, t=3, w=4). A full-width denoiser task is not built.
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
