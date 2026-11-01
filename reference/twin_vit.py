"""The NumPy twin of the Bend ViT denoiser (issue #16, milestone M2).

  reference/.venv/bin/python reference/twin_vit.py

A mirror of `reference/probe_vit_model.bend`, in float32, at the same
operation order:

  20 features per token (4 patched pixels, 8 time, 8 position)
  embed      LinearNB 20 -> 64          y = x @ W.T          (no bias)
  block x 3  LayerNorm -> attention -> residual
             LayerNorm -> 64 -> 128 -> 64 GELU -> residual
  attention  q, k, v each x @ W.T, one head, CAUSAL (BendTorch's
             `Attention` masks the future: query i reads keys 0..i),
             scores = (q @ k.T) * 0.125, softmax over the visible keys,
             out = scores @ v, then x @ Wo.T
  LayerNorm  y = ((x - mean) * rstd) * g + b, rstd = 1/sqrt(var + eps)
  GELU       0.5x (1 + tanh(0.7978845608 (x + 0.044715 x^3)))
  head       LinearNB 64 -> 4

The weights are named as in the file: embed, then per block (att q/k/v/out,
norms, mlp), then head. `from_flat` takes the flat list a Bend `Layer.dump`
writes, in the loader's order, so a run of the Bend probe and a run of this
twin can be compared without trusting either side's initialisation.

Nothing here is a gate yet: the gates land with `reference/test_diffusion_vit.py`.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

F = np.float32
W, HI, PX, FE, TK = 64, 128, 4, 20, 16
SCALE = F(0.125)
EPS = F(1e-5)


def patch(x):
    """64 pixels -> 64 values in token order (16 tokens of 4, 2x2 patches)."""
    idx = []
    for k in range(16):
        r, c = k // 4, k % 4
        b = 16 * r + 2 * c
        idx += [b, b + 1, b + 8, b + 9]
    return x[..., idx]


def unpatch(p):
    out = np.zeros_like(p)
    for i in range(64):
        k = 4 * (i // 16) + ((i % 8) // 2)
        d = 2 * ((i // 8) % 2) + (i % 2)
        out[..., i] = p[..., 4 * k + d]
    return out


def layernorm(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    r = F(1.0) / np.sqrt(v + EPS)
    return F(F(F(x - m) * r) * g + b)


def gelu(x):
    return F(F(F(0.5) * x) * F(F(1.0) + np.tanh(F(F(0.7978845608028654) * F(x + F(F(0.044715) * x ** 3))))))


def causal_mask(t=TK):
    """BendTorch's `Attention` is causal: query i reads keys 0..i only.

    `qrow` builds i + 1 scores and `prow` softmaxes exactly those, leaving
    the rest of the row as zeros, so masking the upper triangle with -inf
    is the same mathematics (and the same float32 result: exp(-inf) = 0).
    """
    return np.triu(np.ones((t, t), dtype=bool), 1)


_MASK = causal_mask()


def attention(x, wq, wk, wv, wo):
    q = F(x @ wq.T)
    k = F(x @ wk.T)
    v = F(x @ wv.T)
    s = F(F(q @ k.swapaxes(-1, -2)) * SCALE)
    s = np.where(_MASK[: s.shape[-1], : s.shape[-1]], F(-np.inf), s).astype(F)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    a = F(e / e.sum(-1, keepdims=True))
    return F(F(a @ v) @ wo.T)


def block(x, w):
    h = layernorm(x, w["ln1g"], w["ln1b"])
    x = F(x + attention(h, w["wq"], w["wk"], w["wv"], w["wo"]))
    h = layernorm(x, w["ln2g"], w["ln2b"])
    m = F(h @ w["w1"].T)
    m = gelu(m)
    m = F(m @ w["w2"].T)
    return F(x + m)


def forward(weights, tokens):
    """tokens: (B, 16, 20) float32 -> (B, 16, 4) noise."""
    x = F(tokens @ weights["embed"].T)
    for i in range(3):
        x = block(x, {k: v for k, v in weights[f"b{i}"].items()})
    y = layernorm(x, weights["head_ln_g"], weights["head_ln_b"])
    return F(y @ weights["head"].T)


def from_flat(flat, blocks=3):
    """The flat list a Bend `Layer.dump` writes, in the loader's order.

    Seq loads left then right; Residual is Seq(Fork(Id, f), Add2), so the
    norm's (g, b) come before the layer it wraps; the attention is
    Fork(q, Fork(k, v)) then its output projection; the MLP is 64->128 then
    128->64. LayerNorm is LN{g, b}. The counts add up to the model's
    100,736: 1,280 + 3 * (128 + 4*4096 + 128 + 2*8192) + 384.
    """
    v = np.asarray(flat, dtype=F)
    i = 0
    def take(n):
        nonlocal i
        out = v[i:i + n]
        i += n
        return out
    w = {"embed": take(W * FE).reshape(W, FE)}
    for b in range(blocks):
        d = {}
        d["ln1g"] = take(W); d["ln1b"] = take(W)
        d["wq"] = take(W * W).reshape(W, W)
        d["wk"] = take(W * W).reshape(W, W)
        d["wv"] = take(W * W).reshape(W, W)
        d["wo"] = take(W * W).reshape(W, W)
        d["ln2g"] = take(W); d["ln2b"] = take(W)
        d["w1"] = take(HI * W).reshape(HI, W)
        d["w2"] = take(W * HI).reshape(W, HI)
        w[f"b{b}"] = d
    w["head_ln_g"] = take(W); w["head_ln_b"] = take(W)
    w["head"] = take(PX * W).reshape(PX, W)
    if i != len(v):
        raise ValueError(f"from_flat: consumed {i} of {len(v)}")
    return w


def count(blocks=3):
    """The model's parameter count, from the same structure."""
    return W * FE + blocks * (2 * W + 4 * W * W + 2 * W + 2 * HI * W) + 2 * W + PX * W


def clip_eps(x, eps, abar):
    """The reference's rederive: clamp the start, then the noise it implies."""
    sa = F(np.sqrt(abar).astype(F))
    sm = F(np.sqrt(F(F(1.0) - abar)).astype(F))
    x0 = F(F(x - F(sm * eps)) / sa)
    xc = np.clip(x0, F(0.0), F(1.0)).astype(F)
    return F(F(x - F(sa * xc)) / sm)


def ddim_cloud(w, k=4, z=None):
    """The 16-step DDIM cloud, eta = 0, over all 64 images (token order).

    `z` is the sampler noise: at eta = 0 its coefficient is zero, so any z
    must give the same cloud (the invariance gate).
    """
    from test_diffusion_digits import ALPHAS, noise_of
    from test_diffusion import ddim, rungs, step_of
    pos = rungs(k)
    x = patch(noise_of(2000000, 4096).reshape(64, 64)).astype(F)   # pixel -> token order
    for i, idx in enumerate(pos):
        abar = ALPHAS[idx]
        prev = ALPHAS[pos[i + 1]] if i + 1 < len(pos) else F(1.0)
        t = idx + 1
        eps = forward(w, tokens_from_token(x, t)).reshape(64, 64)
        eps = clip_eps(x, eps, abar)
        zz = np.zeros((64, 64), dtype=F) if z is None else z
        x = ddim(step_of(abar, prev, 0.0), x, eps, zz)
    return x.reshape(-1)


def legibility(samples, data, d=64):
    s = np.asarray(samples).reshape(-1, d).astype(np.float64)
    t = np.asarray(data).reshape(-1, d).astype(np.float64)
    s = s - s.mean(1, keepdims=True)
    s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)
    t = t - t.mean(1, keepdims=True)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    c = np.abs(s @ t.T).max(1)
    return float(c.mean()), float(c.min()), float(c.max())


FREQS = (F(0.001), F(0.01), F(0.1), F(1.0))


def temb(t, steps=64):
    """mlbend-generative's temb_of(t, steps, 4): sin/cos of (t/steps) * f."""
    tau = F(F(t) / F(steps))
    out = []
    for f in FREQS:
        out += [F(np.sin(F(tau * f))), F(np.cos(F(tau * f)))]
    return np.array(out, dtype=F)


def pos(k):
    """The position feature of token k: sin/cos of k * f (not k/steps)."""
    kk = F(k)
    out = []
    for f in FREQS:
        out += [F(np.sin(F(kk * f))), F(np.cos(F(kk * f)))]
    return np.array(out, dtype=F)


def tokens_of(pixels, t):
    """(B, 64) pixel-ordered digits -> (B, 16, 20) token features."""
    b = pixels.shape[0]
    pat = patch(pixels)                       # (B, 64) token order
    tok = pat.reshape(b, TK, PX)
    te = np.tile(temb(t), (b, TK, 1))
    po = np.tile(np.stack([pos(k) for k in range(TK)]), (b, 1, 1))
    return F(np.concatenate([tok, te, po], axis=-1))


def target_of(pixels):
    """The noise target in the same token order: (B, 16, 4)."""
    b = pixels.shape[0]
    return patch(pixels).reshape(b, TK, PX)


def tokens_from_token(x_tok, t):
    """(B, 64) already in token order -> (B, 16, 20) model input."""
    b = x_tok.shape[0]
    tok = x_tok.reshape(b, TK, PX)
    te = np.tile(temb(t), (b, TK, 1))
    po = np.tile(np.stack([pos(k) for k in range(TK)]), (b, 1, 1))
    return F(np.concatenate([tok, te, po], axis=-1))


def frozen_eps(t=64, seed=1):
    """The forward the frozen probe prints: t = 64 on the starting cloud."""
    from test_diffusion_digits import noise_of
    cloud = noise_of(2000000, 4096).reshape(64, 64)     # pixel order
    tok = patch(cloud)                                   # token order
    return forward(bend_init(seed), tokens_from_token(tok, t)).reshape(4096)


def bend_init(seed=1):
    """Bend's `init_params`: noise_of(seed, count) * 0.1, in loader order."""
    from test_diffusion_digits import noise_of
    flat = F(noise_of(seed, count()) * F(0.1))
    return from_flat(flat)


def t_of(s, i):
    """Bend's `t_img(s, i)`: (i + s*64) * 2654435761 in U32, >> 26, mod 64, plus 1."""
    return 1 + ((((i + s * 64) * 2654435761) & 0xFFFFFFFF) >> 26) % 64


def levels_of(s):
    """The level each image gets in Bend's step `s`: image i takes `t_of(s, 63 - i)`.

    `ab_img` and `te_img` both end in `List.reverse` over lists that are already
    ascending (`ab_rev`/`te_rev` cons from the last image down), so the example
    hands image 0 the pair and the time embedding of image 63, image 1 those of
    image 62, and so on (ISSUE-040). The pairing of pixels, coefficients and
    embedding is consistent within an image, so this is a permutation of which
    level each image draws, not a wrong objective — but the twin has to mirror
    it, because the loss is measured on that pairing.
    """
    return [t_of(s, 63 - i) for i in range(64)]


def loss_pieces(s, seed=1):
    """The three numbers of Bend's step `s`: its loss, mean(target^2), mean(out^2).

    The window is the per-image timestep one (ISSUE-016): chunk
    `(s div 64) mod 16`, a level per image from the same Weyl sequence, the
    noise of seed s. The two mean squares are returned beside the loss because
    the loss is dominated by the target's own mean square (~1.0), so a wrong
    pairing moves it only in the fourth digit while the model's mean square
    moves in the second: `vit-first` gates all three.
    """
    from test_diffusion_digits import ALPHAS, noise_of, load
    x0 = load()
    chunk = (s * 2246822519 // 1048576 % 16) * 64
    batch = x0[chunk:chunk + 64]
    nz = noise_of(s, 4096).reshape(64, 64)
    ts = levels_of(s)
    ab = np.array([[np.sqrt(ALPHAS[t - 1]), np.sqrt(1.0 - ALPHAS[t - 1])] for t in ts], dtype=F)
    mixed = F(ab[:, 0:1] * batch + ab[:, 1:2] * nz)
    tok = patch(mixed).reshape(64, TK, PX)
    te = np.stack([temb(t) for t in ts])
    po = np.tile(np.stack([pos(k) for k in range(TK)]), (64, 1, 1))
    tokens = F(np.concatenate([tok, np.repeat(te[:, None, :], TK, axis=1), po], axis=-1))
    out = forward(bend_init(seed), tokens)
    tgt = target_of(nz)
    return (float(np.mean(F(out - tgt) ** 2)), float(np.mean(F(tgt * tgt))),
            float(np.mean(F(out * out))))


def loss_at(s, seed=1):
    """The loss Bend's step `s` prints; `loss_pieces` has it with its two sides."""
    return loss_pieces(s, seed)[0]


def first_loss(seed=1):
    """The loss of step 0 with ONE timestep for the whole batch (t = 1).

    This is the pre-ISSUE-016 wiring, kept for `main`'s shape check: the
    example has run one level per image since ISSUE-016, so the gate's window
    is `loss_at(0)` and this function is not it."""
    from test_diffusion_digits import noise_of, load, ALPHAS
    x0 = load()
    batch = x0[:64]                                   # first 4096 pixels
    nz = noise_of(0, 4096).reshape(64, 64)
    a = F(np.sqrt(ALPHAS[0]).astype(F))               # t = 1
    b = F(np.sqrt(F(1.0) - ALPHAS[0]).astype(F))
    mixed = F(F(a * batch) + F(b * nz))
    w = bend_init(seed)
    out = forward(w, tokens_of(mixed, 1))
    tgt = target_of(nz)
    loss = float(np.mean(F(out - tgt) ** 2))
    return loss, out, tgt, w


def main():
    """A shape check with deterministic pseudo-weights until the loader order
    is wired: 64 digits of 20 features in, 64 x 16 x 4 out."""
    rng = np.random.default_rng(0)
    w = {"embed": F(rng.standard_normal((W, FE)) * 0.1),
         "head": F(rng.standard_normal((PX, W)) * 0.1),
         "head_ln_g": np.ones(W, dtype=F), "head_ln_b": np.zeros(W, dtype=F)}
    for i in range(3):
        w[f"b{i}"] = {
            "wq": F(rng.standard_normal((W, W)) * 0.1),
            "wk": F(rng.standard_normal((W, W)) * 0.1),
            "wv": F(rng.standard_normal((W, W)) * 0.1),
            "wo": F(rng.standard_normal((W, W)) * 0.1),
            "w1": F(rng.standard_normal((HI, W)) * 0.1),
            "w2": F(rng.standard_normal((W, HI)) * 0.1),
            "ln1g": np.ones(W, dtype=F), "ln1b": np.zeros(W, dtype=F),
            "ln2g": np.ones(W, dtype=F), "ln2b": np.zeros(W, dtype=F),
        }
    tokens = F(rng.standard_normal((64, TK, FE)))
    out = forward(w, tokens)
    print(f"twin forward: {tokens.shape} -> {out.shape}, finite {np.isfinite(out).all()}, "
          f"rms {float(np.sqrt((out ** 2).mean())):.4f}")
    x = F(rng.standard_normal(64))
    print(f"patch roundtrip exact: {np.array_equal(unpatch(patch(x)), x)}")
    print(f"parameter count from the structure: {count()} (Bend reports 100736)")
    loss, out, tgt, w = first_loss()
    print(f"twin step-0 loss: {loss:.6f}   (Bend prints its own on the same window)")
    c = ddim_cloud(bend_init())
    print(f"twin ddim cloud: mean {float(c.mean()):+.4f}  std {float(c.std()):.4f}  "
          f"in [0,1] {int(((c >= 0) & (c <= 1)).sum())}/4096")
    e = frozen_eps()
    print(f"twin frozen eps: {e.shape}, first {float(e[0]):.6f}, rms {float(np.sqrt((e**2).mean())):.4f}")


if __name__ == "__main__":
    main()
