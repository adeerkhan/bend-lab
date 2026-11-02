"""The diffusion model: BendTorch against a NumPy twin.

  reference/.venv/bin/python reference/test_diffusion.py

`examples/diffusion/two_moons/main.bend` trains a three-linear-layer denoiser with
SiLU, by AdamW, to predict the noise of a DDPM forward process over a
two-moons dataset of 128 points, then samples two ways: the ancestral
loop over all 64 rungs and DDIM over every 4th rung (issue #12).
Every "random" number is a pure function of a counter (murmur3's fmix32 over the
step salt and the index, Box-Muller for the Gaussians), so this twin draws
the same numbers, computes in float32 with the same operation order, and
compares the curves:

  first_loss   the loss of the first step — the whole deterministic
               pipeline (data, noise, schedule, weights, forward, loss)
               in one number, gated
  final_loss   the loss of the last step — printed, not gated: the two
               optimizers round differently and the gap grows with the
               steps
  data_*       the dataset's own statistics, gated
  sample_*     the DDPM cloud's statistics — printed: the sampling
               loop runs the model 32 times, and one ulp of exp/sin/cos
               drift amplified through it is a different cloud
  ddim_*       the DDIM cloud's statistics (eta = 0, stride 4) — printed
               for the same reason; the sampler's algebra is gated
               instead by reference/test_ddim.py

Writes results/diffusion/: loss.tsv, scatter.svg, run.json.
"""
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

import artifacts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

N = 128            # points
T = 64             # diffusion steps
STEPS = 4000       # training steps
H = 64             # hidden width
LR, B1, B2, EPS, WD = 0.01, 0.9, 0.999, 1e-8, 0.0

F = np.float32


# ---- the counters' hash and the Gaussian it answers, mirrored ----

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


def uniform(s, i):
    h = mix32(hash_of(s, i))
    m = np.uint32((h & np.uint32(16777215)) + np.uint32(1))
    return F(F(m) / F(16777216.0))


def gauss_at(s, j):
    pair = j - (j % 2)
    ua = uniform(s, pair)
    ub = uniform(s, pair + 1)
    r = np.sqrt(F(0.0) - F(2.0 * np.log(ua)))
    ang = F(6.2831855 * ub)
    return F(r * np.cos(ang)) if j % 2 == 0 else F(r * np.sin(ang))


def noise_of(s):
    return np.array([gauss_at(s, j) for j in range(2 * N)], dtype=F)


# ---- the data, the schedule, the embedding, mirrored ----

def moons():
    # interleaved the way the Bend walk builds it: u0, l0, u1, l1, ...
    out = []
    for k in range(64):
        th = F(F(F(3.1415927) * F(F(k) + F(0.5))) / F(64.0))
        out += [np.cos(th).astype(F), np.sin(th).astype(F)]
        out += [F(1.0 - np.cos(th)), F(0.5 - np.sin(th))]
    return np.array(out, dtype=F)


def lin_beta_at(t, steps=T):
    return F(F(0.0001) + F(F(0.2799) * F(F(t) / F(steps))))


def lin_alphas(steps=T):
    acc, out = F(1.0), []
    for t in range(1, steps + 1):
        acc = F(acc * F(1.0 - lin_beta_at(t, steps)))
        out.append(acc)
    return np.array(out, dtype=F)


ALPHAS = lin_alphas()


def temb_of(t):
    tau = F(F(t) / F(64.0))
    freqs = [F(0.001), F(0.01), F(0.1), F(1.0)]
    out = []
    for f in freqs:
        out += [np.sin(F(tau * f)).astype(F), np.cos(F(tau * f)).astype(F)]
    return np.array(out, dtype=F)


# ---- the model, mirrored ----

def value(i, j, seed):
    m = (i * 3 + j * 5 + seed) % 7
    return F(F(F(m) * F(0.1)) - F(0.2))


def init(seed, rows, cols):
    return np.array([[value(i, j, seed) for j in range(cols)] for i in range(rows)], dtype=F)


def silu(x):
    return F(x / F(1.0 + np.exp(np.float32(-x))))


def forward(params, inp):
    h = silu(F(inp @ params["w1"].T + params["b1"]))
    h = silu(F(h @ params["w2"].T + params["b2"]))
    return F(h @ params["w3"].T + params["b3"])


def input_of(s, t, a, b, x0):
    eps = noise_of(s)
    xt = F(F(a * x0) + F(b * eps))
    xt = xt.reshape(N, 2)
    te = np.tile(temb_of(t), (N, 1))
    return np.concatenate([xt, te], axis=1).astype(F)


def init_params():
    """The starting weights — the example's params(), frozen (no training)."""
    return {
        "w1": init(1, H, 10), "b1": np.zeros(H, dtype=F),
        "w2": init(2, H, H), "b2": np.zeros(H, dtype=F),
        "w3": init(3, 2, H), "b3": np.zeros(2, dtype=F),
    }


def train(x0):
    params = init_params()
    moment = {k: np.zeros_like(v) for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    curve = []
    for s in range(STEPS):
        t = s % T + 1
        abar = ALPHAS[t - 1]
        inp = input_of(s, t, np.sqrt(abar).astype(F),
                       np.sqrt(F(1.0 - abar)).astype(F), x0)
        tgt = noise_of(s).reshape(N, 2)
        out = forward(params, inp)
        diff = F(out - tgt)
        loss = F(np.mean(F(diff * diff)))
        curve.append(float(loss))
        d_out = F(F(2.0 / F(2 * N)) * diff)
        # the hidden activations, for the gradients
        z1 = F(inp @ params["w1"].T + params["b1"])
        h1 = silu(z1)
        z2 = F(h1 @ params["w2"].T + params["b2"])
        h2 = silu(z2)
        # silu backward: dy * s * (1 + x * (1 - s)), s = sigmoid(x), and x is
        # the PRE-activation as bendtorch's silu_d has it (ISSUE-006)
        grads = {}
        grads["w3"] = F(d_out.T @ h2)
        grads["b3"] = d_out.sum(axis=0).astype(F)
        s2 = F(1.0 / F(1.0 + np.exp(np.float32(-z2))))
        d_pre2 = F(d_out @ params["w3"])
        d_h2 = F(F(d_pre2 * s2) * F(1.0 + F(z2 * F(1.0 - s2))))
        grads["w2"] = F(d_h2.T @ h1)
        grads["b2"] = d_h2.sum(axis=0).astype(F)
        s1 = F(1.0 / F(1.0 + np.exp(np.float32(-z1))))

        d_pre1 = F(d_h2 @ params["w2"])
        d_h1 = F(F(d_pre1 * s1) * F(1.0 + F(z1 * F(1.0 - s1))))
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


def sample(params):
    """The ancestral loop: the DDPM mean, the posterior's sigma (issue #14).

    sigma^2 = beta_t·(1-ᾱ_{t-1})/(1-ᾱ_t) is the exact posterior's, which
    at t=1 is 0 (ᾱ_0 = 1), so the last step adds nothing and the example
    needs no `last` flag. This is the sigma DDIM with eta = 1 reproduces.
    """
    x = noise_of(2000000).copy()
    for t in range(T, 0, -1):
        beta = lin_beta_at(t)
        abar = ALPHAS[t - 1]
        prev = ALPHAS[t - 2] if t > 1 else F(1.0)
        coef = F(beta / np.sqrt(F(1.0 - abar)).astype(F))
        scale = F(1.0 / np.sqrt(F(1.0 - beta)).astype(F))
        sig = np.sqrt(F(beta * F(F(F(1.0) - prev) / F(F(1.0) - abar)))).astype(F)
        inp = input_of(1000000 + t, t, F(1.0), F(0.0), x)
        eps = forward(params, inp).reshape(2 * N)
        z = noise_of(1000000 + t)
        # the draw rides outside `scale`: Algorithm 2 adds sigma_t*z to
        # the already-scaled mean (issue #14)
        x = F(F(scale * F(x - F(coef * eps))) + F(sig * z))
    return x


# ---- the DDIM sampler, mirrored (issue #12): mlbend-generative's
# sigma_of / step_of / ddim over the example's rung lists ----

def sigma_of(abar, abar_prev, eta):
    above = np.maximum(F(0.0), F(F(1.0) - abar_prev))
    below = np.maximum(F(0.0), F(F(1.0) - F(abar / abar_prev)))
    return F(eta * F(np.sqrt(F(above / F(F(1.0) - abar))) * np.sqrt(below)))


def step_of(abar, abar_prev, eta):
    """The five coefficients, in the order step_of computes them."""
    sig = sigma_of(abar, abar_prev, eta)
    c0 = F(F(1.0) / np.sqrt(abar))
    c1 = np.sqrt(np.maximum(F(0.0), F(F(1.0) - abar)))
    c2 = np.sqrt(np.maximum(F(0.0), abar_prev))
    c3 = np.sqrt(np.maximum(F(0.0), F(F(1.0) - F(abar_prev + F(sig * sig)))))
    return c0, c1, c2, c3, sig


def ddim(co, x, eps, z):
    """One step over the sample: x0 folded into x_prev, elementwise."""
    c0, c1, c2, c3, sig = co
    x0 = F(F(x - F(c1 * eps)) * c0)
    return F(F(F(x0 * c2) + F(c3 * eps)) + F(sig * z))


def rungs(k):
    """The example's rung positions: skip = (T-1) mod k anchors the top at ᾱ_T."""
    return list(range((T - 1) % k, T, k))[::-1]


def ddim_sample(params, eta, k, draw=True, salt=0):
    """The example's DDIM loop: every k-th rung, top first, in lockstep."""
    pos = rungs(k)
    x = noise_of(2000000).copy()
    for i, idx in enumerate(pos):
        abar = ALPHAS[idx]
        prev = ALPHAS[pos[i + 1]] if i + 1 < len(pos) else F(1.0)
        t = idx + 1
        z = noise_of(1000000 + salt + t) if draw else np.zeros(2 * N, dtype=F)
        inp = input_of(1000000 + t, t, F(1.0), F(0.0), x)
        eps = forward(params, inp).reshape(2 * N)
        x = ddim(step_of(abar, prev, eta), x, eps, z)
    return x


def stats(x):
    pts = x.reshape(N, 2)
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()),
            float(np.sqrt(pts[:, 0].var())), float(np.sqrt(pts[:, 1].var())))


# ---- the Bend side ----

def run_bend():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-diffusion-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/diffusion/two_moons/main.bend")],
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
    pairs = {k: [float(v) for v in re.findall(rf"^{k}: ([0-9.eE+-]+) ([0-9.eE+-]+)$", out, re.M)[0]]
             for k in ("data_mean", "data_std", "sample_mean", "sample_std",
                       "ddim_mean", "ddim_std")}
    final = float(re.search(r"^final_loss: ([0-9.eE+-]+)$", out, re.M).group(1))
    samples = np.array([float(v) for v in
                        re.search(r"^samples: (.+)$", out, re.M).group(1).split()], dtype=F)
    ddim_samples = np.array([float(v) for v in
                             re.search(r"^ddim_samples: (.+)$", out, re.M).group(1).split()],
                            dtype=F)
    return losses, final, pairs, samples, ddim_samples


# ---- the plot ----

def scatter_svg(path, data, bend_pts, numpy_pts, ddim_pts):
    W, Hh = 640, 400
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" viewBox="0 0 {W} {Hh}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="16" y="22" font-family="sans-serif" font-size="14">'
             f'two moons: data (grey), Bend DDPM (red), NumPy DDPM (blue), Bend DDIM (green)</text>']
    def dot(p, color):
        x = 160 + p[0] * 100
        y = 250 - p[1] * 100
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}" opacity="0.7"/>')
    for p in data.reshape(N, 2):
        dot(p, "#7f8c8d")
    for p in bend_pts.reshape(N, 2):
        dot(p, "#c0392b")
    for p in numpy_pts.reshape(N, 2):
        dot(p, "#2471a3")
    for p in ddim_pts.reshape(N, 2):
        dot(p, "#1e8449")
    parts.append("</svg>")
    return artifacts.save_text(path, "\n".join(parts))


def loss_svg(path, steps_idx, bend_curve, numpy_curve):
    W, Hh = 640, 300
    top = max(max(bend_curve), max(numpy_curve)) * 1.05
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" viewBox="0 0 {W} {Hh}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="16" y="22" font-family="sans-serif" font-size="14">the loss of the printed steps</text>']
    def line(curve, color):
        pts = []
        for i, v in enumerate(curve):
            x = 50 + i * (W - 80) / max(1, len(curve) - 1)
            y = 260 - v * 220 / top
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    line(bend_curve, "#c0392b")
    line(numpy_curve, "#2471a3")
    parts.append(f'<line x1="50" y1="260" x2="{W-30}" y2="260" stroke="#333"/>'
                 '<text x="55" y="278" font-family="sans-serif" font-size="10" fill="#c0392b">Bend</text>'
                 '<text x="85" y="278" font-family="sans-serif" font-size="10" fill="#2471a3">NumPy</text></svg>')
    return artifacts.save_text(path, "\n".join(parts))


def numpy_answer():
    """The twin's numbers, for the experiment runner."""
    x0 = moons()
    params, curve = train(x0)
    ss = stats(sample(params))
    dm = stats(x0)
    return {"first_loss": curve[0], "final_loss": curve[-1],
            "sample_mean": ss[0], "sample_std": ss[2],
            "data_mean": dm[0], "data_std": dm[2]}


def main():
    out, seconds = run_bend()
    losses, final, pairs, bend_samples, bend_ddim = parse(out)
    # the two sample lines carry 256 numbers each: everything but them
    print("\n".join(l for l in out.strip().split("\n")
                    if not l.startswith(("samples:", "ddim_samples:"))))

    x0 = moons()
    params, curve = train(x0)
    xs = sample(params)
    # the example's report settings: eta() = 0, stride() = 4, no draw
    xd = ddim_sample(params, 0.0, 4, draw=False)

    # the printed steps: the Bend loop prints at i = 1, then every 40th
    # step boundary, then the last
    idx = [k - 1 for k in range(1, STEPS + 1)
           if k == 1 or (k + 1) % 40 == 0 or k + 1 == STEPS]
    idx = sorted(set(idx))
    want = [curve[i] for i in idx]

    fail = 0
    # the first loss: the whole deterministic pipeline in one number
    if abs(losses[0] - curve[0]) > 1e-3:
        print(f"ERROR the first loss differs: {losses[0]} vs {curve[0]}")
        fail += 1
    else:
        print(f"ok   the first loss agrees: {losses[0]} vs {curve[0]}")
    print(f"     final_loss (report): {final} vs {curve[-1]}")
    dm = stats(x0)
    ok = (abs(pairs["data_mean"][0] - dm[0]) < 1e-5 and abs(pairs["data_mean"][1] - dm[1]) < 1e-5
          and abs(pairs["data_std"][0] - dm[2]) < 1e-5 and abs(pairs["data_std"][1] - dm[3]) < 1e-5)
    if not ok:
        print(f"ERROR the data statistics differ: {pairs['data_mean']} {pairs['data_std']} vs {dm}")
        fail += 1
    else:
        print(f"ok   the data statistics agree: {pairs['data_mean']} {pairs['data_std']}")
    ss = stats(xs)
    print(f"     sample_mean (report): {pairs['sample_mean']} vs [{ss[0]:.6}, {ss[1]:.6}]")
    print(f"     sample_std  (report): {pairs['sample_std']} vs [{ss[2]:.6}, {ss[3]:.6}]")
    sd = stats(xd)
    print(f"     ddim_mean   (report): {pairs['ddim_mean']} vs [{sd[0]:.6}, {sd[1]:.6}]")
    print(f"     ddim_std    (report): {pairs['ddim_std']} vs [{sd[2]:.6}, {sd[3]:.6}]")

    res = os.path.join(ROOT, "results/diffusion")
    os.makedirs(res, exist_ok=True)
    rows = ["step\tbend\tnumpy"]
    for i, s in enumerate(idx):
        rows.append(f"{s}\t{losses[i]}\t{want[i]}")
    wrote = artifacts.save_text(os.path.join(res, "loss.tsv"), "\n".join(rows) + "\n")
    wrote |= scatter_svg(os.path.join(res, "scatter.svg"), x0, bend_samples, xs, bend_ddim)
    wrote |= loss_svg(os.path.join(res, "loss.svg"), idx, losses, want)
    wrote |= artifacts.save_json(os.path.join(res, "run.json"), {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "example": "examples/diffusion/two_moons/main.bend",
        "architecture": "linear(10,32), SiLU, linear(32,32), SiLU, linear(32,2)",
        "process": f"DDPM, T={T}, linear beta 0.0001..0.3, eps-prediction; "
                   "DDIM eta=0, every 4th rung (16 model calls)",
        "machine": platform.platform(), "seconds": seconds,
        "bend": {"first_loss": losses[0], "final_loss": final,
                 "data_mean": pairs["data_mean"], "data_std": pairs["data_std"],
                 "sample_mean": pairs["sample_mean"], "sample_std": pairs["sample_std"],
                 "ddim_mean": pairs["ddim_mean"], "ddim_std": pairs["ddim_std"],
                 "samples": [float(v) for v in bend_samples[:8]] + ["..."]},
        "numpy": {"first_loss": curve[0], "final_loss": curve[-1],
                  "data_stats": dm, "sample_stats": ss, "ddim_stats": sd}})
    print(f"{'wrote' if wrote else 'unchanged'} {res}/loss.tsv, loss.svg, scatter.svg, run.json")

    if fail:
        print(f"{fail} checks differ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
