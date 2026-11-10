"""The PyTorch-autograd twin for 20 AdamW steps of the 128 hybrid.

  reference/.venv-rf/bin/python reference/glm53_h_train128.py \
      --golden reference/glm53_h_train128_golden.npz

The same weights, corpus rows and AdamW settings as `train_h128.bend` (lr
3e-3, betas (0.9, 0.95), eps 1e-8, weight decay 0.1).  Saves the 20 losses and
the final weights in `glm53_h_data.shapes()` order; a moved-lr run is the
can-fail.
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
sys.path.insert(0, os.path.join(ROOT, "reference/glm53_flash"))
import glm53_h_data as D  # noqa: E402
import glm53_h_bwd_data as B  # noqa: E402
from glm53_flash.model import LinearAttention, SparseAttention, ModelConfig  # noqa: E402

N, DIM, H, V, IGNORE = 128, 32, 4, 260, 260
STEPS = 20


def modules():
    cfg = ModelConfig(dim=DIM, heads=H, expert_hidden=128, experts=8, streams=4,
                      sparse_window=16, sparse_stride=16, max_sequence_length=192)
    return [LinearAttention(cfg), LinearAttention(cfg), LinearAttention(cfg), SparseAttention(cfg)]


def params():
    w = D.weights()
    return {name: torch.nn.Parameter(torch.from_numpy(w[name].copy())) for name, _ in D.shapes()}


def rmsnorm(x, g):
    return g * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))


def forward(p, ids):
    x = p["emb"][ids]
    ms = modules()
    for i, m in enumerate(ms):
        m.qkv.weight = p[f"b{i}/qkv"]
        m.out.weight = p[f"b{i}/out"]
        h = rmsnorm(x, p[f"b{i}/ln1"])
        x = x + m(h.unsqueeze(0)).squeeze(0)
        h = rmsnorm(x, p[f"b{i}/ln2"])
        gate = h @ p[f"b{i}/gate"].T
        up = h @ p[f"b{i}/up"].T
        x = x + (F.silu(gate) * up) @ p[f"b{i}/down"].T
    x = rmsnorm(x, p["lnf"])
    return x @ p["emb"].T


def run(lr, golden=None):
    torch.set_num_threads(1)
    p = params()
    opt = torch.optim.AdamW(list(p.values()), lr=lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.1)
    raw = np.fromfile(os.path.join(ROOT, "examples/language/glm53_flash/data/corpus128-b32",
                                   "corpus128-train_batch.bin"), dtype=np.uint8)
    labels_raw = np.fromfile(os.path.join(ROOT, "examples/language/glm53_flash/data/corpus128-b32",
                                          "corpus128-train_labels.bin"), dtype=np.uint8)
    losses = []
    for step in range(STEPS):
        ids = torch.from_numpy(raw[step * N:(step + 1) * N].astype(np.int64))
        labels = torch.from_numpy(labels_raw[step * N:(step + 1) * N].astype(np.int64))
        logits = forward(p, ids)
        valid = labels != IGNORE
        loss = F.cross_entropy(logits[valid], labels[valid])
        losses.append(float(loss.detach()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    flat = np.concatenate([p[name].detach().reshape(-1).numpy().astype(np.float32)
                           for name, _ in D.shapes()])
    print("losses", " ".join(f"{v:.7f}" for v in losses))
    if golden:
        np.savez(golden, losses=np.array(losses, dtype=np.float32), weights=flat)
        print(f"wrote {golden}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=os.path.abspath, default=None)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()
    run(args.lr, args.golden)


if __name__ == "__main__":
    main()
