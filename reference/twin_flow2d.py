"""The 2D rectified flow's NumPy twin: the same dumped weights, the same forward, Euler step and sample.

The model is `examples/diffusion/flow_matching/main.bend`'s: three linear layers with SiLU
(3 -> H -> H -> 2, where the three inputs are x_t, y_t and the level), and Euler with dt = 1/32. `H` is
read from the example's `hidden()` (128 at the recorded setting), so the twin follows a width change instead of
keeping a stale one (ISSUE-062).

The weights come from the probe's own `Layer.dump` (flat, in the loader's order: w1, b1, w2, b2, w3, b3),
so neither side's initialisation is trusted. SiLU is BendTorch's `x / (1 + exp(-x))`, not `x * sigmoid(x)`.
"""
import numpy as np

from flow_config import hidden

F = np.float32
N, D, IN, EULER = 128, 2, 3, 32
HID = hidden()
# the dump's parts, in the loader's order (Lin's w then b, three layers)
PARTS = ((HID, IN), (1, HID), (HID, HID), (1, HID), (D, HID), (1, D))
DT = F(1.0 / EULER)


def silu(x):
    return (x / (F(1.0) + np.exp(-x))).astype(F)


class Model:
    def __init__(self, flat):
        f = np.asarray(flat, dtype=F)
        off, parts = 0, []
        for shape in PARTS:
            size = int(np.prod(shape))
            parts.append(f[off:off + size].reshape(shape))
            off += size
        if off != f.size:
            raise SystemExit(f"twin_flow2d: the dump has {f.size} values, the layers take {off}")
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = parts

    def forward(self, x):
        """(128, 3) rows of (x_t, y_t, t) -> (128, 2) the velocity."""
        h = silu(x @ self.w1.T + self.b1)
        h = silu(h @ self.w2.T + self.b2)
        return (h @ self.w3.T + self.b3).astype(F)


def parse(out):
    """The probe's lines as arrays: `tag: v1 v2 ...`."""
    got = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        tag, rest = line.split(":", 1)
        got[tag.strip()] = np.array([float(v) for v in rest.split()], dtype=F) if rest.split() else np.array([], dtype=F)
    for key in ("params",):
        if key in got:
            got[key] = int(got[key][0])
    return got


def euler(model, x, steps):
    """Euler over `steps` of the 32 the preset divides by: t = i/32, dt = 1/32."""
    x = np.asarray(x, dtype=F).reshape(N, D).copy()
    for i in range(steps):
        t = F(F(i) / F(EULER))
        inp = np.concatenate([x, np.full((N, 1), t, dtype=F)], axis=1)
        x = (x + DT * model.forward(inp)).astype(F)
    return x.reshape(-1)


def compare(out):
    """Every gate the probe supports, as (name, ok, detail) rows.

    `fwd` is one *step* at t = 0.5, not the velocity: the example's `fwd_v` flattens the
    forward's output and returns `x + dt*v` (via `euler_x_go`), so the twin compares the
    same thing. `euler k` is k steps in one call, because the level is i/32 — restarting
    would sample t = 0 again.
    """
    from test_diffusion_digits import noise_of
    got = parse(out)
    model = Model(got["w"])
    x0 = np.asarray(noise_of(2000000, 256), dtype=F).reshape(N, D)
    inp = np.concatenate([x0, np.full((N, 1), F(0.5), dtype=F)], axis=1)
    rows = []
    def add(name, want, have, tol=1e-5):
        a = np.asarray(want, dtype=F).reshape(-1); b = np.asarray(have, dtype=F).reshape(-1)
        if a.size != b.size:
            rows.append((name, False, f"size {a.size} vs {b.size}")); return
        d = float(np.abs(a - b).max())
        rows.append((name, d <= tol, f"maxabs {d:.3e}  tol {tol:.0e}"))
    add("input", inp.reshape(-1), got["inp"])
    add("forward (one step at t = 0.5)", x0 + DT * model.forward(inp), got["fwd"])
    add("step (one step at t = 0)", euler(model, x0, 1), got["step"])
    add("euler 8", euler(model, x0, 8), got["euler8"])
    add("euler 16", euler(model, x0, 16), got["euler16"])
    add("euler 32", euler(model, x0, 32), got["euler32"])
    return rows
