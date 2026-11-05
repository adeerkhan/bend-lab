"""The NumPy twin of the 64x64 flow (ISSUE-043).

  reference/.venv/bin/python reference/twin_flow64.py

A mirror of `reference/probe_flow64.bend` and of
`examples/diffusion/flowers_flow/main.bend`, in float32, at the same operation
order. The preset is `corpus` (`config.bend`):

  128 rows of 64 pixels, one token per row: two images per step, and the
              rows run 0..127 across both (ISSUE-044)
  features    64 pixels, then 8 position, then 8 time values per token
  embed       LinearNB 80 -> 128            y = x @ W.T        (no bias)
  block x 3   LayerNorm -> attention -> residual
              LayerNorm -> 128 -> 256 -> 128 GELU -> residual
  attention   q, k, v each x @ W.T, one head, head dim 128, scale 0.125,
              CAUSAL (BendTorch's `Attention` masks the future: query i reads
              keys 0..i), batched in two windows of 64 rows
  LayerNorm   y = ((x - mean) * rstd) * g + b, rstd = 1/sqrt(var + 1e-5)
  GELU        0.5x (1 + tanh(0.7978845608 (x + 0.044715 x^3)))
  head        LayerNorm, then LinearNB 128 -> 64
  sampler     x <- x + v(x, t) / 32, t = i / 32 (the example's `upd` and
              `step_vs`), over the preset's 32 steps

`from_flat` takes the flat list the probe's `Layer.dump` prints, in the loader's
order, so a run of the Bend probe and a run of this twin can be compared
without trusting either side's initialisation. `count()` is the structure's
parameter count (413,440), which the probe prints.

The three numbers the gates are set from land in `reference/test_flow64.py`.
"""
import numpy as np

F = np.float32
W, HI, PX, TK, B, FE = 128, 256, 64, 64, 2, 80
RW = B * TK                      # 128 rows: two images of 64 rows
SCALE = F(0.125)                 # the example's `sc`: not 1/sqrt(head dim)
EPS = F(1e-5)                    # LayerNorm's eps
EULER = 32                       # config.bend's EULER, the sampler's divisor
DT = F(1.0 / EULER)              # 0.03125, the example's `upd`
TAU = F(6.2831855)               # the feature angle's constant
FREQS = (F(0.5), F(1.0), F(2.0), F(4.0))
BLOCKS = 3


def layernorm(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    r = F(1.0) / np.sqrt(v + EPS)
    return F(F(F(x - m) * r) * g + b)


def gelu(x):
    return F(F(F(0.5) * x) * F(F(1.0) + np.tanh(F(F(0.7978845608028654)
                                                 * F(x + F(F(0.044715) * x ** 3))))))


def causal(t=TK):
    return np.triu(np.ones((t, t), dtype=bool), 1)


_MASK = causal()


def attention(x, wq, wk, wv, wo):
    """x is (128, W): the projection runs over all rows, the attention in two
    windows of TK rows (`Attention`'s b = 2, t = 64)."""
    q = F(x @ wq.T)
    k = F(x @ wk.T)
    v = F(x @ wv.T)
    s = F(F(q @ k.swapaxes(-1, -2)) * SCALE)
    s = np.where(_MASK, F(-np.inf), s).astype(F)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    a = F(e / e.sum(-1, keepdims=True))
    return F(F(a @ v) @ wo.T)


def block(x, w):
    h = layernorm(x, w["ln1g"], w["ln1b"])
    x = F(x + attention(h.reshape(B, TK, W), w["wq"], w["wk"], w["wv"], w["wo"])
          .reshape(RW, W))
    h = layernorm(x, w["ln2g"], w["ln2b"])
    m = F(h @ w["w1"].T)
    m = gelu(m)
    m = F(m @ w["w2"].T)
    return F(x + m)


def forward(weights, tokens, blocks=BLOCKS):
    """tokens: (128, 80) float32 -> (128, 64) the velocity field."""
    x = F(tokens @ weights["embed"].T)
    for i in range(blocks):
        x = block(x, weights[f"b{i}"])
    y = layernorm(x, weights["head_ln_g"], weights["head_ln_b"])
    return F(y @ weights["head"].T)


def from_flat(flat, blocks=BLOCKS):
    """The flat list a Bend `Layer.dump` writes, in the loader's order.

    Seq loads left then right; Residual is Seq(Fork(Id, f), Add2), so the norm
    comes before the layer it wraps; the attention is Fork(q, Fork(k, v)) then
    its output projection; the MLP is 128 -> 256 then 256 -> 128. LayerNorm is
    LN{g, b}. The counts add up to `count()`: 10,240 + 3 * (2*128 + 4*16384 +
    2*128 + 2*32768) + 256 + 8,192.
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
        d["ln1g"] = take(W)
        d["ln1b"] = take(W)
        d["wq"] = take(W * W).reshape(W, W)
        d["wk"] = take(W * W).reshape(W, W)
        d["wv"] = take(W * W).reshape(W, W)
        d["wo"] = take(W * W).reshape(W, W)
        d["ln2g"] = take(W)
        d["ln2b"] = take(W)
        d["w1"] = take(HI * W).reshape(HI, W)
        d["w2"] = take(W * HI).reshape(W, HI)
        w[f"b{b}"] = d
    w["head_ln_g"] = take(W)
    w["head_ln_b"] = take(W)
    w["head"] = take(PX * W).reshape(PX, W)
    if i != len(v):
        raise ValueError(f"from_flat: consumed {i} of {len(v)}")
    return w


def count(blocks=BLOCKS):
    """The model's parameter count, from the same structure."""
    return W * FE + blocks * (2 * W + 4 * W * W + 2 * W + 2 * HI * W) + 2 * W + PX * W


def features(t, k):
    """One token's 16 features, in the example's order.

    `feats` conses the position block and then the time block in front of the
    accumulator and `lr_of` reverses the whole list, so a token reads its
    position (sin, cos of k/TK at 0.5, 1, 2, 4 turns) and then its level.

    Rows 64..127 therefore read positions 1.0..1.98: the example's token
    builder never restarts the row for the second image (ISSUE-044), and the
    twin mirrors what the program computes.
    """
    u = F(F(k) / F(TK))
    out = []
    for f in FREQS:
        a = F(F(TAU * f) * u)
        out += [F(np.sin(a)), F(np.cos(a))]
    for f in FREQS:
        a = F(F(TAU * f) * F(t))
        out += [F(np.sin(a)), F(np.cos(a))]
    return np.array(out, dtype=F)


def tokens_of(t, x):
    """(8192,) noise in row order -> (128, 80) the model input at level t."""
    rows = np.asarray(x, dtype=F).reshape(RW, TK)
    feats = np.stack([features(t, k % TK) for k in range(RW)])
    return F(np.concatenate([rows, feats], axis=-1))


def fwd_flat(w, t, x):
    return forward(w, tokens_of(t, x)).reshape(-1)


def euler(w, x0, steps=EULER):
    """The example's sampler: t = i / 32 and dt = 1/32 for every step count.

    `step_vs` divides by the preset's EULER (32) rather than by the number of
    steps it is given, so the 8- and 16-step runs are prefixes of the 32-step
    one, not coarser samplers.
    """
    x = np.asarray(x0, dtype=F).copy()
    for i in range(steps):
        t = F(F(i) / F(EULER))
        v = fwd_flat(w, t, x)
        x = F(x + F(DT * v))
    return x


def noise(s=2000000, m=B * TK * TK):
    """mlbend-generative's `noise_of`, the probe's starting cloud."""
    from test_diffusion_digits import noise_of
    return noise_of(s, m).astype(F)


def bend_init(seed=1):
    """The example's `init_params`: noise_of(seed, count) * 0.1, loader order.

    The gates do not use this (they take the probe's `Layer.dump`); a cell
    needs it because the program builds its own starting weights.
    """
    from test_diffusion_digits import noise_of
    return from_flat(F(noise_of(seed, count()) * F(0.1)))


def main():
    from test_diffusion_digits import noise_of
    rng = np.random.default_rng(0)
    w = {"embed": F(rng.standard_normal((W, FE)) * 0.1),
         "head": F(rng.standard_normal((PX, W)) * 0.1),
         "head_ln_g": np.ones(W, dtype=F), "head_ln_b": np.zeros(W, dtype=F)}
    for i in range(BLOCKS):
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
    tok = tokens_of(F(0.5), noise_of(2000000, B * TK * TK))
    out = forward(w, tok)
    print(f"twin forward: {tok.shape} -> {out.shape}, "
          f"finite {np.isfinite(out).all()}, rms {float(np.sqrt((out ** 2).mean())):.4f}")
    print(f"parameter count from the structure: {count()} (the probe prints its own)")
    feats = features(F(0.5), 0)
    print(f"token 0's features: {' '.join(f'{float(v):+.4f}' for v in feats)}")
    x = euler(w, noise_of(2000000, B * TK * TK), 32)
    print(f"twin euler32: mean {float(x.mean()):+.4f}  std {float(x.std()):.4f}  "
          f"finite {np.isfinite(x).all()}")


if __name__ == "__main__":
    main()
