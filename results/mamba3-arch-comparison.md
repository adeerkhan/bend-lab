# The architecture comparison (D) — not run, with the measured reason

The goal asks (stretch, only with A–C green): equal parameters, same data,
seed and schedule — Mamba-3, Mamba-1 (the MambaViT mixer) and a tiny
Transformer — on the flowers 5-class set (held-out accuracy) and the byte
corpus (held-out bits/byte).

A–C are green (2026-11-15), but the comparison was **not run**, and no
numbers for it exist. The honest budget, from this session's measurements on
this lane:

| anchor | measurement | command |
|---|---|---|
| Mamba-3 tiny-LM training | 1297.2 s wall for 20 steps of one 128-position example (64.9 s/step) | `reference/test_mamba3_tiny_opt.py` |
| Mamba-1 (MambaViT mixer) subset | 1869.8 s wall for 160 train + 256 held-out MNIST images (4.49 s/image) | `reference/test_mamba_vit_train.py` |
| full 60,000-image MNIST epoch in Bend | on the order of days | extrapolation from the above, recorded as unavailable, not as a number |
| flowers 5-class data | 3,670 images at `/tmp/opencode/flowers/flower_photos` (unpinned, from another session); the pinned Oxford-102 cache (`~/.cache/bend-lab/flowers`, 15 files) is a different set |
| byte corpus held-out | bigram 3.3569, unigram 4.8093 bits/byte on the 4,096-byte split (`reference/test_mamba3_tiny_lm.py` baselines) |

Why it was not run:

1. **The Transformer arm does not exist in Bend in this repository.** glm53's
   attention block is a different shape on a different task; a tiny
   Transformer for this comparison (attention + MLP, equal parameters,
   the same data path and schedule) is a separate project, not an afternoon
   of wiring.
2. **The flowers path for a Bend trainer does not exist either** (no flowers
   task/adapter with pinned data for a fresh 5-class classifier run). The
   Oxford-102 cache is available, and the 5-class folder set is unpinned.
3. **The parameters cannot be equalised honestly without a search.** The
   Mamba-3 LM here is 71,952 parameters (vocab 127, d_model 64, two layers,
   no tied embedding); the Mamba-1 MNIST arm is 83,050 (d_model 32, eight
   layers); a comparable Mamba-1 flowers classifier would be a different
   model again. "Equal parameters" therefore means three different models
   retuned until the counts match — a design task, not an experiment.
4. **The budget.** Even an A3-scale payment per arm (5 steps of batch 32 on
   160 train images) is ~12 min per architecture on the Mamba-1 arm and
   potentially much longer for a fresh Transformer, and a 5-step payment
   teaches nothing about held-out generalisation on flowers or bytes; the
   meaningful budget is hours-to-days of Bend runtime for three arms plus
   three trainers plus three data paths. Nothing was promised beyond the
   budget, and none of it is measured — so none of it is claimed.

The two numbers that exist and are relevant neighbours (measured, with
commands): the Mamba-1 MNIST full-epoch number is **94.61 %** (the twin, not
Bend — `reference/mamba_vit_twin.py --epochs 1`, 4 min 05 s on the RTX 5060),
and the byte-corpus baselines sit at **3.3569** (bigram) and **4.8093**
(unigram) bits/byte. Neither is an architecture-comparison number.

Unblock path: a Bend Transformer for this shape, a flowers task/adapter with
pinned data (Oxford-102 cache), and a session sized for hours of Bend
runtime per arm. Until then, D is recorded as **unavailable**, not estimated.

## 2026-11-18 — the arm that does exist, and a measured step cost

The "no Transformer arm" reason above is still right for a *pure* attention
stack, and there is still no Llama trainer at this size in the lab. What does
exist is an attention-family LM on the same corpus and almost the same
parameter count: `glm53-hybrid128` (four blocks, three linear and one sparse
attention, width 32, four heads, **74,144 parameters**) reads the same
`corpus128-b32` windows at 128 positions, batch 1. The Mamba-3 tiny-LM is
**71,952 parameters** at 128 positions, batch 1, from the same file. Three
percent apart in parameters is close enough to time both; it is not close
enough to compare quality without a matched run.

Same session, 2026-11-18 16:24 to 16:26, load average 2.9, Bend 2.0.38,
native builds:

| arm | one training step | command |
|---|---|---|
| Mamba-3 SISO, two blocks, d_model 64 | **10,860 ms** (32,581 ms / 3) | `bench/plumbing_mamba_tiny/main.bend`, `full_step_x3_ms` |
| GLM-5.3-Flash hybrid, four blocks, width 32 | **1,245 ms** | `bend examples/language/glm53_flash/train_h128.bend -o /tmp/opencode/glm1.bin`, then `glm1.bin --steps 1 --no-val --no-save --trace` |

The Mamba step is **8.7x** the hybrid's at almost the same parameter count,
the same 128 positions, the same corpus and the same machine, minutes apart.
It is not the optimizer (32 ms / 3) and not the input (0 ms): the scan phase
accounts for 8,972 ms of the 10,860, with the per-token preprocessing inside
that noise now. This is a **cost** comparison only: the two losses (4.838 and
5.563) sit on different output vocabularies and are not comparable.

What it does not answer, and what would:

1. **Quality at equal tokens.** One pass of the corpus is 409,600 bytes =
   3,200 windows of 128: about 9.7 h for Mamba and 1.1 h for the hybrid at
   today's steps. Same held-out split, same metric (bits/byte), same
   optimizer and schedule — and the last one is the missing piece: the two
   trainers use different AdamW settings and schedules today, so the curves
   would rank the optimizer, not the architecture.
2. **Quality at equal wall-clock**, which is the practical question: two
   hours each gives the hybrid about 5,800 steps against Mamba's 660. The
   8.7x is exactly what that measures.
3. **The asymptotic claim.** Mamba's case is cost linear in sequence length
   against attention's quadratic, and 128 positions is where attention is
   cheapest. A step-cost curve at 128, 256 and 512 positions for both arms
   would test the reason to use Mamba at all; it is three rebuilds per arm
   (the shapes are type-level) and one step each.

For reference, the pretrained SmolLM2 models in `nuxyel/llama.bend`
(134.5M to 1.7B parameters, trained by Hugging Face on the order of a
trillion tokens, BPE) are not a comparison point for either arm here: a
different scale, a different tokenizer, and a repository that is inference
only. Its 76 ms per token at 135M on one thread measures *that engine*
against llama.cpp (2.6x to 3.9x slower), not Llama against Mamba.
