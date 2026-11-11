"""A two-layer SISO Mamba-3 LM in PyTorch: the full-model twin and its goldens.

  reference/.venv-rf/bin/python reference/mamba3_lm_twin.py

Two independent blocks, residual around each, a final RMSNorm and an untied
`8 x 64` head.  Vocabulary 8, seqlen 8 for the copy task, prompt 4 then 12
greedy tokens for generation.  The block itself is `mamba3_twin.twin_forward`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
import mamba3_twin as W  # noqa: E402

V, T, DM = 8, 8, 64


def make_module(seed):
    torch.manual_seed(seed)
    return W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32, ngroups=1,
                    is_mimo=False).cuda().float()


def init_extra():
    g = torch.Generator(device="cuda")
    g.manual_seed(321)
    E = torch.randn(V, DM, device="cuda", generator=g) * 0.5
    Wh = torch.randn(V, DM, device="cuda", generator=g) * 0.5
    nw = 1.0 + torch.randn(DM, device="cuda", generator=g) * 0.1
    return E, Wh, nw


def lm_forward(m1, m2, E, Wh, nw, tokens):
    u = E[torch.tensor(tokens, device="cuda")].unsqueeze(0)
    f1 = W.twin_forward(m1, u)
    h1 = u + f1["y"]
    f2 = W.twin_forward(m2, h1)
    h2 = h1 + f2["y"]
    hn = W.rms(h2, nw, eps=1e-5)
    logits = hn @ Wh.T
    return {"u": u, "h1": h1, "h2": h2, "hn": hn, "logits": logits}


def params(m):
    return {
        "in_proj": m.in_proj.weight.detach().cpu().numpy(),
        "dt_bias": m.dt_bias.detach().cpu().numpy(),
        "B_bias": m.B_bias.detach().squeeze(1).cpu().numpy(),
        "C_bias": m.C_bias.detach().squeeze(1).cpu().numpy(),
        "B_norm_w": m.B_norm.weight.detach().cpu().numpy(),
        "C_norm_w": m.C_norm.weight.detach().cpu().numpy(),
        "D": m.D.detach().cpu().numpy(),
        "out_proj": m.out_proj.weight.detach().cpu().numpy(),
    }


def train(steps=10, lr=1e-3):
    torch.set_num_threads(1)
    m1, m2 = make_module(71), make_module(72)
    E, Wh, nw = init_extra()
    E = E.clone().requires_grad_(True)
    Wh = Wh.clone().requires_grad_(True)
    nw = nw.clone().requires_grad_(True)
    tokens = list(range(T))
    targets = torch.arange(T, device="cuda")
    losses, out_tokens = [], None
    logits = None
    for _ in range(steps):
        out = lm_forward(m1, m2, E, Wh, nw, tokens)
        logits = out["logits"]
        loss = F.cross_entropy(logits.reshape(T, V), targets)
        loss.backward()
        with torch.no_grad():
            for m in (m1, m2):
                for p in m.parameters():
                    p -= lr * p.grad
                    p.grad = None
            E -= lr * E.grad
            Wh -= lr * Wh.grad
            nw -= lr * nw.grad
            E.grad = None
            Wh.grad = None
            nw.grad = None
        losses.append(float(loss))
    with torch.no_grad():
        out = lm_forward(m1, m2, E, Wh, nw, tokens)
        out_tokens = out["logits"].reshape(T, V).argmax(-1)
    return m1, m2, E.detach(), Wh.detach(), nw.detach(), losses, out_tokens


def generate(m1, m2, E, Wh, nw, steps=12):
    tokens = [0, 1, 2, 3]
    first = None
    for i in range(steps):
        out = lm_forward(m1, m2, E, Wh, nw, tokens)
        logits = out["logits"][0, -1]
        if i == 0:
            first = logits.detach().cpu().numpy().reshape(-1)
        tokens.append(int(logits.argmax()))
    return tokens, first


def main():
    torch.set_num_threads(1)
    m1, m2 = make_module(71), make_module(72)
    E0, Wh0, nw0 = init_extra()
    with torch.no_grad():
        out = lm_forward(m1, m2, E0, Wh0, nw0, list(range(T)))
        loss = F.cross_entropy(out["logits"].reshape(T, V), torch.arange(T, device="cuda"))
    m1t, m2t, Et, Wht, nwt, losses, toks = train()
    gen, gen0 = generate(m1t, m2t, Et, Wht, nwt)

    arrays = {"E0": E0.cpu().numpy(), "Wh0": Wh0.cpu().numpy(),
              "nw0": nw0.cpu().numpy(),
              "lm_hn": out["hn"].detach().cpu().numpy(),
              "lm_logits": out["logits"].detach().cpu().numpy(),
              "lm_loss": np.asarray([float(loss)], dtype=np.float32),
              "lm_train_loss": np.asarray(losses, dtype=np.float32),
              "lm_train_tokens": toks.cpu().numpy().astype(np.float32),
              "lm_gen_tokens": np.asarray(gen, dtype=np.float32),
              "lm_gen_logits0": gen0}
    for tag, m in (("l1", m1), ("l2", m2)):
        for key, value in params(m).items():
            arrays[f"{tag}_{key}"] = value
    out_path = os.path.join(ROOT, "reference/mamba3_lm_golden.npz")
    np.savez(out_path, **arrays)
    print(f"wrote {out_path}")
    print(f"lm_loss {float(loss):.8e}")
    print(f"train_loss {np.asarray(losses)}")
    print(f"train_tokens {toks.cpu().numpy().astype(int).tolist()}")
    print(f"gen_tokens {gen}")
    print(f"block params {sum(p.numel() for p in m1.parameters())} x 2, "
          f"plus {V*DM} embedding, {V*DM} head, {DM} norm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
