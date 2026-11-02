"""The digits diffusion model: BendTorch against a NumPy twin.

  reference/.venv/bin/python reference/test_diffusion_digits.py

`examples/diffusion/digits/main.bend` reads the first 1024 rows of
mstz/optdigits (pinned revision, 8x8 digits, pixels scaled to [0, 1]),
trains a three-linear-layer SiLU denoiser full-batch by AdamW to predict
the noise of the 64-step linear schedule, then samples 64 fresh digits
two ways: the ancestral loop over all 64 rungs and DDIM over every 4th
rung (issue #12). This twin does the same in float32 with the same
counter-based noise, computes in the same operation order, and compares:

  first_loss   the loss of the first step — the whole deterministic
               pipeline (data, noise, schedule, weights, forward, loss)
               in one number, gated
  data_mean    the dataset's own mean, gated
  final_loss   the loss of the last step — printed, not gated: the two
               optimizers round differently and the gap grows with the
               steps
  sample_*     the ancestral grid's statistics — report mode: the
               sampling loop runs the model 64 times, and one ulp of
               exp/sin/cos drift amplified through it is a different grid
  ddim_*       the DDIM grid's statistics — report mode, for the same
               reason

Plus the frozen probe (reference/probe_digits.bend): the two samplers
with the deterministic starting weights, no training, compared with this
twin element by element — statistics are order-invariant, so only the
reads can catch a reordered array (issue #13), and a second DDIM run
with the draw on and another salt must be bit-identical at eta = 0. The
probe prints the model's output for the first step too, and both sides
step with it: that tight gate is about the sampler's arithmetic and the
array's order, not about the matmul (BendTorch's dot accumulates in four
partial sums where numpy's BLAS does not — measured 7 ulps on eps,
reported rather than gated).

The row order of the staged input was wrong until 2f2631b (one pixel was
skipped per row), which the gate caught. Writes results/diffusion-digits/:
loss.tsv, grid.svg, run.json. `numpy_answer` is what
`reference/experiment.py`'s diffusion-digits task compares against.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

import artifacts

# The step algebra is one definition, mirrored once: mlbend-generative's
# `sigma_of`, `step_of`, `ddim` and the strided walk over the same T = 64
# linear schedule. test_diffusion.py carries that mirror; the loops below
# are this example's own (4096 pixels a cloud).
from test_diffusion import ddim, rungs, step_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

N = 1024            # training digits
D = 64              # pixels a digit
T = 64              # diffusion steps
STEPS = 3000        # training steps
H = 64              # hidden width
S = 64              # sampled digits
LR, B1, B2, EPS, WD = 0.01, 0.9, 0.999, 1e-8, 0.0
URL = ("https://huggingface.co/datasets/mstz/optdigits/resolve/"
       "4f20cc827b763fb69e2a80e18e892e4e11498aec/optdigits.data")
SHA = "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd"

F = np.float32
np.seterr(over="ignore")


def load():
    cache = os.path.join(ROOT, "reference/.cache/datasets/mstz/optdigits/optdigits.data")
    if os.path.exists(cache):
        text = open(cache).read()
    else:
        import urllib.request
        text = urllib.request.urlopen(URL, timeout=120).read().decode()
    rows = [l for l in text.split("\n") if l.strip()]
    data = np.array([[int(c) for c in r.split(",")] for r in rows[:N]], dtype=F)
    return (data[:, :-1] / F(16.0)).astype(F)


# ---- the twin's counter RNG and schedule (mlbend-generative mirrored) ----

def mix32(z):
    z = np.uint32(z)
    a = np.uint32(z ^ (z >> np.uint32(16)))
    b = np.uint32(a * np.uint32(2246822519))
    c = np.uint32(b ^ (b >> np.uint32(13)))
    e = np.uint32(c * np.uint32(3266489917))
    return np.uint32(e ^ (e >> np.uint32(16)))


def hash_of(s, i):
    k = np.uint32(np.uint32(s) * np.uint32(2654435769))
    return np.uint32(k + np.uint32(np.uint32(i) + np.uint32(305419898)))


def uniforms(s, i):
    k = np.uint32(np.uint32(s) * np.uint32(2654435769))
    h = mix32(np.uint32(k + np.uint32(np.uint32(i) + np.uint32(305419898))))
    m = np.uint32((h & np.uint32(16777215)) + np.uint32(1))
    return (m.astype(F) / F(16777216.0)).astype(F)


def noise_of(s, m):
    # Box-Muller over counter uniforms, vectorized; element-wise float32
    # ops in the scalar twins' order, so the bits agree
    j = np.arange(m)
    pair = j - j % 2
    u = uniforms(s, np.arange(m + 1))
    r = np.sqrt(F(0.0) - F(2.0 * np.log(u[pair]).astype(F))).astype(F)
    ang = (F(6.2831855) * u[pair + 1]).astype(F)
    z = np.where(j % 2 == 0, (r * np.cos(ang)).astype(F), (r * np.sin(ang)).astype(F))
    return z.astype(F)


def lin_beta_at(t, steps=T):
    return F(F(0.0001) + F(F(0.2799) * F(F(t) / F(steps))))


ALPHAS = np.array([f for f in (np.cumprod([F(1.0 - lin_beta_at(t)) for t in range(1, T + 1)],
                                          dtype=F))], dtype=F).flatten()


def temb_of(t):
    tau = F(F(t) / F(64.0))
    freqs = [F(0.001), F(0.01), F(0.1), F(1.0)]
    out = []
    for f in freqs:
        out += [np.sin(F(tau * f)).astype(F), np.cos(F(tau * f)).astype(F)]
    return np.array(out, dtype=F)


def input_of(s, t, a, b, x0, rows):
    eps = noise_of(s, rows * D)
    xt = F(F(a * x0.reshape(rows * D)) + F(b * eps)).reshape(rows, D)
    te = np.tile(temb_of(t), (rows, 1))
    return np.concatenate([xt, te], axis=1).astype(F)


def silu(x):
    return F(x / F(1.0 + np.exp(np.float32(-x))))


def value(i, j, seed):
    """The example's value(): a 97-value hash on [-1, 1), zero-mean."""
    m = (i * 3 + j * 5 + seed * 7) % 97
    return F(F(F(m) - F(48.0)) / F(48.0))


def init(seed, r, c):
    """The example's params(): value() scaled by 1/sqrt(fan_in), PyTorch's
    Linear init made deterministic (see ISSUE-001)."""
    a = F(F(1.0) / np.sqrt(F(c)))
    return F(np.array([[value(i, j, seed) for j in range(c)] for i in range(r)], dtype=F) * a)


def forward(params, inp):
    h = silu(F(inp @ params["w1"].T + params["b1"]))
    h = silu(F(h @ params["w2"].T + params["b2"]))
    return F(h @ params["w3"].T + params["b3"])


def init_params():
    """The starting weights — the example's params(), frozen (no training)."""
    return {"w1": init(1, H, D + 8), "b1": np.zeros(H, dtype=F),
            "w2": init(2, H, H), "b2": np.zeros(H, dtype=F),
            "w3": init(3, D, H), "b3": np.zeros(D, dtype=F)}


def train(x0):
    params = init_params()
    moment = {k: np.zeros_like(v) for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    curve = []
    for s in range(STEPS):
        t = s % T + 1
        abar = ALPHAS[t - 1]
        inp = input_of(s, t, np.sqrt(abar).astype(F),
                       np.sqrt(F(1.0 - abar)).astype(F), x0, N)
        tgt = noise_of(s, N * D).reshape(N, D)
        z1 = F(inp @ params["w1"].T + params["b1"])
        h1 = silu(z1)
        z2 = F(h1 @ params["w2"].T + params["b2"])
        h2 = silu(z2)
        out = F(h2 @ params["w3"].T + params["b3"])
        diff = F(out - tgt)
        curve.append(float(np.mean(F(diff * diff))))
        d_out = F(F(2.0 / F(N * D)) * diff)
        grads = {"w3": F(d_out.T @ h2), "b3": d_out.sum(axis=0).astype(F)}
        # the SiLU derivative is taken at the PRE-activation, as bendtorch's
        # silu_d does; taking it at the activated value was ISSUE-006
        s2 = F(1.0 / F(1.0 + np.exp(np.float32(-z2))))
        d_h2 = F(F(F(d_out @ params["w3"]) * s2) * F(1.0 + F(z2 * F(1.0 - s2))))
        grads["w2"] = F(d_h2.T @ h1)
        grads["b2"] = d_h2.sum(axis=0).astype(F)
        s1 = F(1.0 / F(1.0 + np.exp(np.float32(-z1))))
        d_h1 = F(F(F(d_h2 @ params["w2"]) * s1) * F(1.0 + F(z1 * F(1.0 - s1))))
        grads["w1"] = F(d_h1.T @ inp)
        grads["b1"] = d_h1.sum(axis=0).astype(F)
        for k in params:
            g = grads[k]
            moment[k] = F(B1 * moment[k] + F(1.0 - B1) * g)
            velocity[k] = F(B2 * velocity[k] + F(1.0 - B2) * F(g * g))
            mhat = F(moment[k] / F(1.0 - B1 ** (s + 1)))
            vhat = F(velocity[k] / F(1.0 - B2 ** (s + 1)))
            params[k] = F(params[k] - F(LR * F(F(mhat / F(np.sqrt(vhat) + EPS)) + WD * params[k])))
    return params, curve


def rev_step_eps(x, t, eps):
    """The step's arithmetic around a model output already in hand: the
    model's implied start, clamped to the data range [0, 1], then the
    posterior mean formed from it (ISSUE-005) — the example's order."""
    beta = lin_beta_at(t)
    abar = ALPHAS[t - 1]
    prev = ALPHAS[t - 2] if t > 1 else F(1.0)
    c1 = F(beta * F(np.sqrt(prev).astype(F) / F(F(1.0) - abar)))
    c2 = F(np.sqrt(F(F(1.0) - beta)).astype(F) * F(F(F(1.0) - prev) / F(F(1.0) - abar)))
    sa = F(np.sqrt(abar).astype(F))
    sm = F(np.sqrt(F(F(1.0) - abar)).astype(F))
    sig = np.sqrt(F(beta * F(F(F(1.0) - prev) / F(F(1.0) - abar)))).astype(F)
    z = noise_of(1000000 + t, S * D)
    x0 = F(F(x - F(sm * eps)) / sa)
    xc = np.clip(x0, F(0.0), F(1.0)).astype(F)
    return F(F(F(c1 * xc) + F(c2 * x)) + F(sig * z))


def rev_step(params, x, t):
    """One ancestral step, the example's reverse_step order."""
    inp = input_of(1000000 + t, t, F(1.0), F(0.0), x, S)
    eps = forward(params, inp).reshape(S * D)
    return rev_step_eps(x, t, eps)


def sample(params):
    """The ancestral loop with the posterior's sigma (issue #14): at t=1
    that variance is 0, so the last step adds nothing."""
    x = noise_of(2000000, S * D).copy()
    for t in range(T, 0, -1):
        x = rev_step(params, x, t)
    return x


def ddim_sample(params, eta, k, draw=True, salt=0):
    """The example's DDIM loop: every k-th rung, top first, the two rung
    lists in lockstep. The step itself (sigma, the five coefficients,
    the elementwise fold) is mlbend-generative's, mirrored once in
    test_diffusion.py over the same T = 64 linear schedule."""
    pos = rungs(k)
    x = noise_of(2000000, S * D).copy()
    for i, idx in enumerate(pos):
        abar = ALPHAS[idx]
        prev = ALPHAS[pos[i + 1]] if i + 1 < len(pos) else F(1.0)
        t = idx + 1
        z = noise_of(1000000 + salt + t, S * D) if draw else np.zeros(S * D, dtype=F)
        inp = input_of(1000000 + t, t, F(1.0), F(0.0), x, S)
        eps = forward(params, inp).reshape(S * D)
        # the clamped start, and the noise it re-derives (ISSUE-005)
        sa = F(np.sqrt(abar).astype(F))
        sm = F(np.sqrt(F(F(1.0) - abar)).astype(F))
        x0 = F(F(x - F(sm * eps)) / sa)
        xc = np.clip(x0, F(0.0), F(1.0)).astype(F)
        eps = F(F(x - F(sa * xc)) / sm)
        x = ddim(step_of(abar, prev, eta), x, eps, z)
    return x


def numpy_answer():
    """The twin's numbers, as reference/experiment.py's runner compares them."""
    x0 = load()
    params, curve = train(x0)
    xs = sample(params)
    return {"first_loss": float(curve[0]), "final_loss": float(curve[-1]),
            "data_mean": float(x0.mean()),
            "sample_mean": float(xs.mean()), "sample_std": float(xs.std())}


def run_bend():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-digits-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/diffusion/digits/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    t0 = time.time()
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    return out.stdout, time.time() - t0


def parse(out):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M)]
    vals = {k: float(re.search(rf"^{k}: ([0-9.eE+-]+)$", out, re.M).group(1))
            for k in ("data_mean", "data_std", "sample_mean", "sample_std",
                      "ddim_mean", "ddim_std")}
    samples = np.array([float(v) for v in
                        re.search(r"^samples: (.+)$", out, re.M).group(1).split()], dtype=F)
    ddim = np.array([float(v) for v in
                     re.search(r"^ddim_samples: (.+)$", out, re.M).group(1).split()], dtype=F)
    return losses, vals, samples, ddim


def grid_svg(path, data, samples):
    cell, pad = 14, 2
    rows = 8
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{2*rows*cell+3*pad}" height="{2*rows*cell+3*pad}" viewBox="0 0 {2*rows*cell+3*pad} {2*rows*cell+3*pad}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="4" y="10" font-family="sans-serif" font-size="9">data (left) vs Bend samples (right)</text>']
    def draw(img, ox, oy):
        px = np.clip(img.reshape(8, 8), 0, 1)
        for r in range(8):
            for c in range(8):
                v = int(round(255 * (1 - float(px[r, c]))))
                # the cell offset is c*(cell/8); a stray c*cell//8*8//8 here
                # made every cell 1.7x too wide (ISSUE-003)
                parts.append(f'<rect x="{ox + c*(cell/8):.1f}" y="{oy + r*(cell/8):.1f}" '
                             f'width="{cell/8:.1f}" height="{cell/8:.1f}" fill="rgb({v},{v},{v})"/>')
    for i in range(rows * rows):
        ox = pad + (i % rows) * cell
        oy = pad + 12 + (i // rows) * cell
        draw(data[i] if i < len(data) else np.zeros(64), ox, oy)
        draw(samples[i] if i < len(samples) else np.zeros(64),
             pad + rows * cell + pad + (i % rows) * cell, oy)
    parts.append("</svg>")
    return artifacts.save_text(path, "\n".join(parts))


def legibility(samples, data):
    """The best absolute correlation between a sample and any training digit.

    The statistics say the cloud is in the data's range; this says whether a
    sample looks like a digit at all. Random images score ~0.33 on this data,
    the dense MLP 0.41, a ViT-style denoiser 0.80, the reference's UNet 0.93
    (reference/digits_vit_probe.py). Reported, not gated yet: the threshold is
    set when the denoiser that can meet it lands (ISSUE-008)."""
    s = np.asarray(samples).reshape(-1, D).astype(np.float64)
    d = np.asarray(data).reshape(-1, D).astype(np.float64)
    s = s - s.mean(1, keepdims=True)
    s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)
    d = d - d.mean(1, keepdims=True)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    c = np.abs(s @ d.T).max(1)
    return float(c.mean()), float(c.min()), float(c.max())


def run_probe():
    """The frozen probe: the model's output and the samplers with the
    starting weights, one line of 4096 numbers each (issue #13 —
    statistics are order-blind, only the reads are not)."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-digits-probe")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "reference/probe_digits.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the probe did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the probe did not run:\n" + out.stdout + out.stderr)
    lines = {}
    for line in out.stdout.strip().split("\n"):
        if ": " in line:
            tag, rest = line.split(": ", 1)
            lines[tag] = np.array([float(v) for v in rest.split()], dtype=F).astype(F)
    return lines


def rel(a, b):
    """The worst element-wise relative gap, in float64 so it is exact."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m = np.maximum(np.abs(a), np.abs(b))
    m = np.where(m > 0, m, 1.0)
    return float(np.max(np.abs(a - b) / m))


def maxabs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


# Measured on this probe before the gates were set (2026-10-27): the
# step with the model's output shared sits at 3.8e-06 absolute — one ulp
# of the 30-magnitude values, from a fused multiply-add on one side —
# and the two frozen clouds sit at 5.1e-04 and 3.9e-04 relative, which
# is what 64 (and 16) forward passes of a 7-ulp model gap cost. Each
# gate keeps ~5x over its own measurement; a reversed array would be
# ~2 in both metrics.
TOL_STEP = 1e-5
TOL_CLOUD = 5e-3


def main():
    out, seconds = run_bend()
    losses, vals, bend_samples, bend_ddim = parse(out)
    print("\n".join([l for l in out.strip().split("\n")
                     if not l.startswith(("samples:", "ddim_samples:"))]))

    x0 = load()
    params, curve = train(x0)
    xs = sample(params).reshape(S, D)
    dxs = ddim_sample(params, 0.0, 4, draw=False).reshape(S, D)

    fail = 0
    if abs(losses[0] - curve[0]) <= 1e-2 * max(1.0, abs(curve[0])):
        print(f"ok   the first loss agrees: {losses[0]} vs {curve[0]}")
    else:
        print(f"ERROR the first loss differs: {losses[0]} vs {curve[0]}")
        fail += 1
    dm = float(x0.mean())
    if abs(vals["data_mean"] - dm) < 1e-5:
        print(f"ok   the data mean agrees: {vals['data_mean']} vs {dm:.7f}")
    else:
        print(f"ERROR the data mean differs: {vals['data_mean']} vs {dm:.7f}")
        fail += 1
    print(f"     final_loss (report): {losses[-1]} vs {curve[-1]}")
    print(f"     sample_mean (report): {vals['sample_mean']} vs {float(xs.mean()):.4f}")
    print(f"     sample_std  (report): {vals['sample_std']} vs {float(xs.std()):.4f}")
    print(f"     ddim_mean   (report): {vals['ddim_mean']} vs {float(dxs.mean()):.4f}")
    print(f"     ddim_std    (report): {vals['ddim_std']} vs {float(dxs.std()):.4f}")
    corr_mean, corr_min, corr_max = legibility(bend_samples, x0)
    rng = np.random.default_rng(0)
    base = legibility(rng.random((S, D)).astype(F), x0)[0]
    print(f"     sample_corr (report): {corr_mean:.3f} "
          f"(min {corr_min:.3f}, max {corr_max:.3f}); random images {base:.3f}")

    # the frozen probe (issues #12, #13): the samplers against the twin,
    # element by element. The step gate feeds Bend's own model output to
    # both sides, so what it gates is the sampler's arithmetic and the
    # array's order — the matmul gap is a report line instead. Mean and
    # std are order-invariant: only the reads can catch a reordered array.
    p0 = init_params()
    frozen = run_probe()
    start = noise_of(2000000, S * D)
    twin_step = rev_step_eps(start, 64, frozen["eps"])
    twin_anc = sample(p0).astype(F)
    twin_ddim = ddim_sample(p0, 0.0, 4, draw=False, salt=0).astype(F)
    eps_twin = forward(p0, input_of(1000000 + 64, 64, F(1.0), F(0.0), start, S)).reshape(S * D)
    print(f"     eps (report): Bend vs twin maxabs {maxabs(frozen['eps'], eps_twin):.3g}, "
          f"worst relative {rel(frozen['eps'], eps_twin):.3g}")

    if maxabs(frozen["step"], twin_step) <= TOL_STEP:
        print(f"ok   one frozen step vs the twin (shared model output): "
              f"worst absolute {maxabs(frozen['step'], twin_step):.3g}")
    else:
        print(f"ERROR one frozen step differs from the twin: "
              f"worst absolute {maxabs(frozen['step'], twin_step):.3g}")
        fail += 1
    if rel(frozen["frozen"], twin_anc) <= TOL_CLOUD:
        print(f"ok   the frozen ancestral cloud vs the twin: "
              f"worst relative {rel(frozen['frozen'], twin_anc):.3g}")
    else:
        print(f"ERROR the frozen ancestral cloud differs from the twin: "
              f"worst relative {rel(frozen['frozen'], twin_anc):.3g}")
        fail += 1
    if rel(frozen["frozen_ddim"], twin_ddim) <= TOL_CLOUD:
        print(f"ok   the frozen DDIM cloud vs the twin: "
              f"worst relative {rel(frozen['frozen_ddim'], twin_ddim):.3g}")
    else:
        print(f"ERROR the frozen DDIM cloud differs from the twin: "
              f"worst relative {rel(frozen['frozen_ddim'], twin_ddim):.3g}")
        fail += 1
    if np.array_equal(frozen["frozen_ddim"], frozen["frozen_ddim2"]):
        print("ok   eta = 0 reads no noise: another salt and the draw are bit-identical")
    else:
        print(f"ERROR eta = 0 moved with the draw: "
              f"worst relative {rel(frozen['frozen_ddim'], frozen['frozen_ddim2']):.3g}")
        fail += 1

    res = os.path.join(ROOT, "results/diffusion-digits")
    os.makedirs(res, exist_ok=True)
    idx = [k - 1 for k in range(1, STEPS + 1)
           if k == 1 or (k + 1) % 100 == 0 or k + 1 == STEPS]
    rows = ["step\tbend\tnumpy"]
    for i, s in enumerate(idx):
        rows.append(f"{s}\t{losses[i] if i < len(losses) else float('nan')}\t{curve[s]}")
    wrote = artifacts.save_text(os.path.join(res, "loss.tsv"), "\n".join(rows) + "\n")
    wrote |= grid_svg(os.path.join(res, "grid.svg"), x0.reshape(N, D)[:64],
                      bend_samples.reshape(S, D))
    wrote |= artifacts.save_json(os.path.join(res, "run.json"), {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "example": "examples/diffusion/digits/main.bend",
        "dataset": {"url": URL, "sha256": SHA, "rows": N},
        "seconds": seconds,
        "bend": {"first_loss": losses[0], "final_loss": losses[-1],
                 "sample_corr": corr_mean, **vals},
        "numpy": {"first_loss": curve[0], "final_loss": curve[-1],
                  "sample_mean": float(xs.mean()), "sample_std": float(xs.std()),
                  "ddim_mean": float(dxs.mean()), "ddim_std": float(dxs.std())}})
    print(f"{'wrote' if wrote else 'unchanged'} {res}/loss.tsv, grid.svg, run.json")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
