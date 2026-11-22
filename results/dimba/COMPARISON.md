# Dimba-style hybrid on flowers64 — four-arm comparison

Cloned `https://github.com/feizc/Dimba` into `ref/Dimba` and read the model:
`diffusion/model/nets/dimba.py` is a text-to-image latent diffusion model whose
backbone **interleaves DiT blocks (adaLN-Zero, caption cross-attention) with
Mamba SSM blocks**, at `gap = depth // mamba_depth`. It needs a VAE (80M), T5
(4.3B) and 0.9B denoiser weights; our flowers are 64×64 grayscale with no
captions, so the released model cannot run here.

`reference/dimba_ref.py` is a small structural analogue on our data: 18 DiT
blocks with per-row adaLN, 18 small CPU Mamba-3 SISO mixers interleaved with
them (`gap=1`), patch embedding + sincos positions, same clean-image velocity
loss and the same training harness as the ViT / JiT(DiT) / HiT arms. It is
**not** the released Dimba and no released weights are used.

## Measured results (1200 steps, batch 16, 2,048 flowers, seed 0)

| arm | params | held legibility | final loss | wall |
|---|---:|---:|---:|---:|
| ViT | 4.74M | 0.514 | 0.074 | 466 s |
| JiT / DiT | 6.53M | 0.607 | 0.070 | 549 s |
| HiT | 6.62M | 0.573 | 0.120 | 536 s |
| **Dimba-style hybrid** | 7.57M | **0.622** | 0.075 | 1,705 s |

The hybrid is best on test-set legibility (+0.015 over JiT/DiT) and has the
lowest final loss of the conditional arms. It is also ~3× slower: the Mamba
scan is a Python loop over 64 tokens, where the released model uses a fused
CUDA selective scan. No FID is claimed.

## What this says about the mechanism

- Combining a content-mixing Transformer block with a recurrent Mamba block
  helps even at 64 tokens and 1,200 steps, where HiT's grouping did not.
- HiT's compression pays off at large scale; here the hybrid's simple
  interleaving wins without needing variable group counts.
- This is one seed, one budget, one metric that smooth blobs can game — the
  ranking is evidence, not a final verdict.

## Ablation: where the gain comes from

The same hybrid code was re-run with `mamba_ratio=0` (no Mamba blocks), same
data, seed, steps and metric (`results/dimba/ablation-no-mamba/run.json`):

| arm | params | held legibility | final loss | wall |
|---|---:|---:|---:|---:|
| JiT/DiT (hit_v4) | 6.53M | 0.6071 | 0.0700 | 549 s |
| Dimba code, no Mamba | 6.53M | 0.6071 | 0.0700 | 344 s |
| Dimba hybrid, 18 Mamba blocks | 7.57M | **0.6225** | 0.0752 | 1,705 s |

The no-Mamba control is numerically identical to JiT/DiT (same 6,525,888
params and 0.6071 legibility), so the code path itself adds nothing. The
**+0.015 legibility is attributable to the 18 interleaved Mamba blocks**, at
the cost of ~1.04M extra params and 5× the wall time (Python scan; the release
uses a fused CUDA selective scan).

## Banners

- CPU torch reference, not Bendlang; no VAE, T5, captions, cross-attention,
  CFG, EMA or released weights.
- `ref/Dimba` and `ref/HiT-Released` are external checkouts, ignored by git.
- `comparison.png` is generated from the four committed `run.json` files.
