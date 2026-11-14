"""The depth-4 reference: four independent SISO blocks, residual around each.

  reference/.venv-rf/bin/python reference/mamba3_deep_twin.py

Writes `reference/mamba3_deep_golden.npz`: for the prompt tokens 0..7, the
stack's hidden rows, logits and copy-task loss, plus every layer's weights
(so the Bend probe loads exactly the same numbers).  The training and
generation halves are NOT written yet: the port's 4-layer backward and
generation chains are not built, so there is no honest quantity to compare.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
import mamba3_twin as W  # noqa: E402

V, T, DM, NL = 8, 8, 64, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_module(seed):
    torch.manual_seed(seed)
    m = W.Mamba3(d_model=DM, d_state=16, expand=2, headdim=32, ngroups=1,
                 is_mimo=False, device=DEV, dtype=torch.float32)
    return m.float()


def init_extra():
    g = torch.Generator(device=DEV)
    g.manual_seed(321)
    E = torch.randn(V, DM, device=DEV, generator=g) * 0.5
    Wh = torch.randn(V, DM, device=DEV, generator=g) * 0.5
    nw = 1.0 + torch.randn(DM, device=DEV, generator=g) * 0.1
    return E, Wh, nw


def deep_forward(ms, E, Wh, nw, tokens):
    u = E[torch.tensor(tokens, device=DEV)].unsqueeze(0)
    h = u
    for m in ms:
        h = h + W.twin_forward(m, h)["y"]
    hn = W.rms(h, nw, eps=1e-5)
    logits = hn @ Wh.T
    targets = torch.arange(T, device=DEV)
    loss = F.cross_entropy(logits.reshape(T, V), targets)
    return {"u": u, "h": h, "hn": hn, "logits": logits, "loss": loss}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "reference/mamba3_deep_golden.npz"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    ms = [make_module(71 + i) for i in range(NL)]
    E, Wh, nw = init_extra()
    out = deep_forward(ms, E, Wh, nw, list(range(T)))
    arrays = {"E": E.cpu().numpy(), "Wh": Wh.cpu().numpy(), "nw": nw.cpu().numpy(),
              "h": out["h"].detach().cpu().numpy(),
              "hn": out["hn"].detach().cpu().numpy(),
              "logits": out["logits"].detach().cpu().numpy(),
              "loss": np.asarray([float(out["loss"].detach())], dtype=np.float32),
              "layers": np.asarray([NL], dtype=np.float32)}
    for i, m in enumerate(ms):
        for key, arr in (
            ("in_proj", m.in_proj.weight), ("dt_bias", m.dt_bias),
            ("B_bias", m.B_bias.squeeze(1)), ("C_bias", m.C_bias.squeeze(1)),
            ("B_norm_w", m.B_norm.weight), ("C_norm_w", m.C_norm.weight),
            ("D", m.D), ("out_proj", m.out_proj.weight)):
            arrays[f"l{i+1}_{key}"] = arr.detach().cpu().numpy()
    np.savez(args.out, **arrays)
    print(f"wrote {args.out}")
    print("loss %.6f  logits max|.| %.6f" %
          (float(out["loss"].detach()), float(out["logits"].abs().max())))


if __name__ == "__main__":
    main()
