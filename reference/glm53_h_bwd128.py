"""The PyTorch-autograd golden for the four-block hybrid's gradients at N=8.

  reference/.venv-rf/bin/python reference/glm53_h_bwd.py \
      --golden reference/glm53_h_bwd_golden.npz

The same weights (`glm53_h_data.weights()`), the same first eight corpus
tokens and next-token labels, the same four blocks (linear, linear, linear,
sparse) and the same masked cross-entropy as `train_h128.bend`.  The gradients
are flattened in `glm53_h_data.shapes()` order, which is the parameter tree's
dump order.  `labels_bad` moves one label by +1: its gradient is the can-fail.
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
        a = m(h.unsqueeze(0)).squeeze(0)
        x = x + a
        h = rmsnorm(x, p[f"b{i}/ln2"])
        gate = h @ p[f"b{i}/gate"].T
        up = h @ p[f"b{i}/up"].T
        x = x + (F.silu(gate) * up) @ p[f"b{i}/down"].T
    x = rmsnorm(x, p["lnf"])
    return x @ p["emb"].T


def loss_of(p, ids, labels):
    logits = forward(p, ids)
    valid = labels != IGNORE
    return F.cross_entropy(logits[valid], labels[valid]), logits


def flat_grads(p):
    return np.concatenate([p[name].grad.detach().reshape(-1).numpy().astype(np.float32)
                           for name, _ in D.shapes()])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=os.path.abspath, default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    ids_np, labels_np, bad_np = B.batch_n(128)
    ids = torch.from_numpy(ids_np.astype(np.int64))
    labels = torch.from_numpy(labels_np.astype(np.int64))
    bad = torch.from_numpy(bad_np.astype(np.int64))
    p = params()
    loss, logits = loss_of(p, ids, labels)
    loss.backward()
    grad = flat_grads(p)
    for value in p.values():
        value.grad = None
    bad_loss, _ = loss_of(p, ids, bad)
    bad_loss.backward()
    bad_grad = flat_grads(p)
    print(f"loss {float(loss):.9f} bad_loss {float(bad_loss):.9f}")
    print(f"logits {logits.numel()} values, max |logit| {float(logits.abs().max()):.6e}")
    print(f"grad {grad.size} values, max |grad| {float(np.abs(grad).max()):.6e}")
    print(f"bad_grad finite {bool(np.isfinite(bad_grad).all())}, "
          f"diff from good {float(np.abs(grad - bad_grad).max()):.6e}")
    if args.golden:
        np.savez(args.golden, loss=np.float32(loss.detach()),
                 bad_loss=np.float32(bad_loss.detach()), grad=grad, bad_grad=bad_grad)
        print(f"wrote {args.golden}")


if __name__ == "__main__":
    main()
