"""The PyTorch-autograd twin of the dense Stage A baseline.

  reference/.venv-rf/bin/python reference/glm53_dense_twin.py --steps 20 --golden reference/glm53_dense_golden.npz

It loads the NumPy tensors written by `reference/glm53_dense_data.py` (the
same weights and the same two fixed 64-byte rows the Bend program reads),
trains with AdamW for the requested steps, and writes the loss of every step
plus the final weights and the gradients at the final weights.  The q/k rows
are unpermuted on the way in, as `glm53_twin.py` does; the final tensors are
flattened back into the file layout so the comparison is whole-tensor and
order-exact.

  reference/.venv/bin/python reference/glm53_dense_data.py --prepare
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import glm53_dense_data as D  # noqa: E402

V, DIM, LAYERS, H, HD, FF, T, B = D.V, D.DIM, D.LAYERS, D.H, D.HD, D.FF, D.T, D.B
N = D.N


def unpermute_rows(w):
    out = w.clone()
    for h in range(H):
        for j in range(HD):
            out[h * HD + j] = w[h * HD + D.P[j]]
    return out


def load_weights(directory):
    w = {}
    for name, shape in D.shapes():
        t = torch.from_numpy(
            np.fromfile(os.path.join(directory, name + ".bin"), dtype="<f4").reshape(shape).copy()
        )
        if name.endswith(("wq", "wk")):
            t = unpermute_rows(t)
        w[name] = t.clone().requires_grad_(True)
    return w


def rmsnorm(x, g):
    return g * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))


def rope_ref(x):
    length, width = x.shape[1], x.shape[-1]
    positions = torch.arange(length, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, width, 2, dtype=torch.float32) / width))
    angles = positions[:, None] * freqs[None, :]
    cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def causal_attention(q, k, v):
    scores = torch.einsum("bthd,bshd->bhts", q, k) * (1.0 / math.sqrt(HD))
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", probs, v)
    return out.reshape(B, T, DIM)


def forward(w, ids):
    x = w["emb"][ids].reshape(N, DIM)
    for i in range(LAYERS):
        h = rmsnorm(x, w[f"b{i}/ln1"])
        q = (h @ w[f"b{i}/wq"].T).reshape(B, T, H, HD)
        k = (h @ w[f"b{i}/wk"].T).reshape(B, T, H, HD)
        v = (h @ w[f"b{i}/wv"].T).reshape(B, T, H, HD)
        q, k = rope_ref(q), rope_ref(k)
        a = causal_attention(q, k, v).reshape(N, DIM)
        x = x + a @ w[f"b{i}/wo"].T
        h = rmsnorm(x, w[f"b{i}/ln2"])
        gate = h @ w[f"b{i}/wg"].T
        up = h @ w[f"b{i}/wu"].T
        x = x + (F.silu(gate) * up) @ w[f"b{i}/wd"].T
    x = rmsnorm(x, w["lnf"])
    return x @ w["emb"].T


def masked_ce(logits, labels):
    valid = labels != D.IGNORE
    return F.cross_entropy(logits[valid], labels[valid])


def flat_tensor(w):
    values = {name: t.detach().cpu().numpy() for name, t in w.items()}
    return D.flatten(values, store_qk=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--data", type=Path, default=Path(D.DATA))
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    D.prepare(str(args.data))
    ids = torch.from_numpy(np.fromfile(args.data / "train_batch.bin", dtype=np.uint8).astype(np.int64))
    labels = torch.from_numpy(np.fromfile(args.data / "train_labels.bin", dtype=np.uint8).astype(np.int64))
    w = load_weights(str(args.data))
    opt = torch.optim.AdamW(list(w.values()), lr=args.lr, betas=(args.beta1, args.beta2),
                            eps=args.eps, weight_decay=args.weight_decay)
    losses = []
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        logits = forward(w, ids)
        loss = masked_ce(logits, labels)
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    logits = forward(w, ids)
    final_loss = masked_ce(logits, labels)
    final_loss.backward()
    final_weights = flat_tensor(w)
    final_grads = D.flatten({name: t.grad.detach().cpu().numpy() for name, t in w.items()}, store_qk=True)
    for i, value in enumerate(losses, 1):
        print(f"step {i}: {value:.9g}")
    print(f"final_loss: {float(final_loss.detach()):.9g}")
    print(f"weights: {final_weights.size} values, grads: {final_grads.size} values")
    if args.golden is not None:
        np.savez(args.golden, losses=np.asarray(losses, dtype=np.float32),
                 weights=final_weights.astype(np.float32), grads=final_grads.astype(np.float32),
                 final_loss=np.asarray(float(final_loss.detach()), dtype=np.float32))
        print(f"wrote {args.golden}")
    print(json.dumps({"steps": args.steps, "first_loss": losses[0], "final_loss": losses[-1]}))


if __name__ == "__main__":
    main()
