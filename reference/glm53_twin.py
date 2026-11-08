"""The PyTorch twin of the Stage A/B/C tiny GLM harness.

  reference/.venv-rf/bin/python reference/glm53_twin.py --prep DIR

The tiny harness is width 32, four blocks, four heads of 8, 64 positions,
SwiGLU hidden 128, vocabulary 260, tied embedding/head. The Bend side loads
the files written here with `T.Mat.load`; the q/k files are stored in BendTorch's
half-split RoPE order and are unpermuted here for the reference's interleaved
RoPE. The permutation is:

    pairs_ref = (0,1), (2,3), (4,5), (6,7)
    pairs_bend = (0,4), (1,5), (2,6), (3,7)
    P = [0, 4, 1, 5, 2, 6, 3, 7]   (w_perm[p] = w[P_inv[p]])

so `rope_bend(P q)` equals `P rope_ref(q)`; q.k is unchanged after the same
permutation of q and k.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

V, DIM, LAYERS, H, HD, FF, T, B = 260, 32, 4, 4, 8, 128, 64, 2
N = B * T
EPS, BASE, SCALE = 1e-6, 10000.0, 1.0 / math.sqrt(HD)
P = [0, 4, 1, 5, 2, 6, 3, 7]
P_INV = [P.index(i) for i in range(len(P))]


def block_shapes(i: int):
    return [
        (f"b{i}/ln1", (DIM,)), (f"b{i}/wq", (DIM, DIM)), (f"b{i}/wk", (DIM, DIM)),
        (f"b{i}/wv", (DIM, DIM)), (f"b{i}/wo", (DIM, DIM)), (f"b{i}/ln2", (DIM,)),
        (f"b{i}/wg", (FF, DIM)), (f"b{i}/wu", (FF, DIM)), (f"b{i}/wd", (DIM, FF)),
    ]


def shapes():
    out = [("emb", (V, DIM))]
    for i in range(LAYERS):
        out += block_shapes(i)
    return out + [("lnf", (DIM,))]


def permute_rows(w: torch.Tensor) -> torch.Tensor:
    """Store a q/k projection in BendTorch's half-split RoPE order."""
    out = w.clone()
    for h in range(H):
        for j in range(HD):
            out[h * HD + j] = w[h * HD + P_INV[j]]
    return out


def unpermute_rows(w: torch.Tensor) -> torch.Tensor:
    out = w.clone()
    for h in range(H):
        for j in range(HD):
            out[h * HD + j] = w[h * HD + P[j]]
    return out


def init_weights(seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    w = {}
    for name, shape in shapes():
        if name.endswith(("ln1", "ln2", "lnf")):
            w[name] = torch.ones(shape)
        else:
            w[name] = torch.randn(*shape, generator=g) * 0.02
    return w


def write_weights(w, directory: Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, t in w.items():
        path = directory / (name + ".bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        t = t.detach().clone()
        if name.endswith(("wq", "wk")):
            t = permute_rows(t)
        path.write_bytes(t.numpy().astype("<f4").tobytes())


def load_twin(directory: Path):
    w = {}
    for name, shape in shapes():
        t = torch.from_numpy(
            __import__("numpy").fromfile(Path(directory) / (name + ".bin"), dtype="<f4").reshape(shape)
        ).clone()
        if name.endswith(("wq", "wk")):
            t = unpermute_rows(t)
        w[name] = t.requires_grad_(True)
    return w


def rmsnorm(x, g):
    return g * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


def rope_ref(x):
    """The reference's interleaved even/odd rotation on [B, T, H, HD]."""
    length, width = x.shape[1], x.shape[-1]
    positions = torch.arange(length, dtype=torch.float32)
    freqs = 1.0 / (BASE ** (torch.arange(0, width, 2, dtype=torch.float32) / width))
    angles = positions[:, None] * freqs[None, :]
    cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def causal_attention(q, k, v):
    # q, k, v: [B, T, H, HD]
    scores = torch.einsum("bthd,bshd->bhts", q, k) * SCALE
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
    """Mean over non-ignored rows; label 260 is ignore."""
    valid = labels != V
    logits = logits[valid]
    labels = labels[valid]
    return F.cross_entropy(logits, labels)


def make_batch(seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, V, (N,), generator=g)
    labels = torch.randint(4, V, (N,), generator=g)
    return ids, labels


def write_batch(directory: Path, ids, labels):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "batch.bin").write_bytes(ids.to(torch.uint8).numpy().tobytes())
    (directory / "labels.bin").write_bytes(labels.to(torch.uint8).numpy().tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    w = init_weights(args.seed)
    write_weights(w, args.prep)
    ids, labels = make_batch()
    write_batch(args.prep, ids, labels)
    alt = ids.clone()
    alt[N // 2:] = torch.roll(alt[N // 2:], 1)
    (args.prep / "batch_alt.bin").write_bytes(alt.to(torch.uint8).numpy().tobytes())
    print(f"wrote {len(w)} tensors and batches to {args.prep}")


if __name__ == "__main__":
    main()
