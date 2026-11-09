"""The PyTorch-autograd twin of the integrated Stage C model.

  reference/.venv-rf/bin/python reference/glm53_c_twin.py --steps 20 \
      --golden reference/glm53_c_golden.npz

One batch, 64 positions, one head of width 8, two blocks: linear attention
then sparse attention (window 16, stride 16), each with a SwiGLU hidden 32,
a final RMSNorm and the tied head.  The weights, the batch and the flatten
order are `reference/glm53_c_data.py`'s.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import glm53_c_data as D  # noqa: E402

N, DIM, HID, V, T = 64, 8, 32, 260, 64
SCALE = DIM ** -0.5


def load_weights(directory):
    w = {}
    for name, shape in D.shapes():
        w[name] = torch.from_numpy(
            np.fromfile(os.path.join(directory, name + ".bin"), dtype="<f4").reshape(shape).copy()
        ).requires_grad_(True)
    return w


def rmsnorm(x, g):
    return g * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))


def rope(x):
    positions = torch.arange(T, dtype=torch.float32)[:, None]
    freqs = 1.0 / (10000.0 ** (torch.arange(0, DIM, 2, dtype=torch.float32) / DIM))[None, :]
    angles = positions * freqs
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[:, 0::2], x[:, 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def linear_attn(q, k, v):
    q, k = rope(q), rope(k)
    qe, ke = F.elu(q) + 1.0, F.elu(k) + 1.0
    K = torch.zeros(DIM)
    KV = torch.zeros(DIM, DIM)
    ys = []
    for t in range(T):
        K = K + ke[t]
        KV = KV + torch.outer(ke[t], v[t])
        ys.append((KV.T @ qe[t]) / (qe[t] @ K).clamp_min(1e-6))
    return torch.stack(ys)


def sparse_attn(q, k, v):
    q, k = rope(q), rope(k)
    ys = []
    for t in range(T):
        idx = [s for s in range(t + 1) if s % 16 == 0 or t - s < 16]
        scores = torch.stack([q[t] @ k[s] for s in idx]) * SCALE
        p = torch.softmax(scores, dim=0)
        ys.append(sum(p[j] * v[s] for j, s in enumerate(idx)))
    return torch.stack(ys)


def attention(x, w, prefix, fn):
    h = rmsnorm(x, w[prefix + "/ln1"])
    qkv = h @ w[prefix + "/qkv"].T
    q, k, v = qkv[:, :DIM], qkv[:, DIM:2 * DIM], qkv[:, 2 * DIM:]
    a = fn(q, k, v) @ w[prefix + "/out"].T
    x = x + a
    h = rmsnorm(x, w[prefix + "/ln2"])
    gate = h @ w[prefix + "/gate"].T
    up = h @ w[prefix + "/up"].T
    return x + (F.silu(gate) * up) @ w[prefix + "/down"].T


def forward(w, ids):
    x = w["emb"][ids]
    x = attention(x, w, "b0", linear_attn)
    x = attention(x, w, "b1", sparse_attn)
    x = rmsnorm(x, w["lnf"])
    return x @ w["emb"].T


def flat(w, grads=False):
    values = []
    for name, _ in D.shapes():
        t = w[name].grad if grads else w[name].detach()
        values.append(t.cpu().numpy().reshape(-1).astype(np.float32))
    return np.concatenate(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--data", type=Path, default=Path(D.DATA))
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    D.prepare(str(args.data))
    ids = torch.from_numpy(np.fromfile(args.data / "c_batch.bin", dtype=np.uint8).astype(np.int64))
    labels = torch.from_numpy(np.fromfile(args.data / "c_labels.bin", dtype=np.uint8).astype(np.int64))
    if args.mask < T:
        labels = labels.clone()
        labels[args.mask:] = -100
    w = load_weights(str(args.data))
    opt = torch.optim.AdamW(list(w.values()), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    losses = []
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(w, ids), labels, ignore_index=-100)
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step}: {float(loss.detach()):.9g}")
    opt.zero_grad(set_to_none=True)
    final = F.cross_entropy(forward(w, ids), labels, ignore_index=-100)
    final.backward()
    for i, value in enumerate(losses, 1):
        print(f"step {i}: {value:.9g}")
    print(f"final_loss: {float(final.detach()):.9g}")
    print(f"weights: {flat(w).size}, grads: {flat(w, True).size}")
    if args.golden:
        np.savez(args.golden, losses=np.asarray(losses, dtype=np.float32),
                 weights=flat(w), grads=flat(w, True),
                 final_loss=np.asarray(float(final.detach()), dtype=np.float32))
        print(f"wrote {args.golden}")


if __name__ == "__main__":
    main()
