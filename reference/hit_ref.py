#!/usr/bin/env python3
"""A small faithful HiT reference on flowers64, with a matched plain-ViT arm.

  reference/.venv/bin/python reference/hit_ref.py --arm hit --steps 300 --out results/hit

Implements the mechanism of arXiv:2611.22962 (HiT) at a size this machine can
train: 64x64 flowers, patch 8 -> N=64 tokens, two 2x grouping stages
(64 -> ~32 -> ~16), width 64, 10 blocks. Equations follow the paper:

  affinity      S_ij = (q_i.k_j + q_j.k_i)/2, q,k L2-normalized   (Eq. 2)
  group affinity s(U,V) = mean_{i in U, j in V} S_ij              (Eq. 8)
  soft weights  e_ig = mean_{j in G_g} S_ij, A = softmax(e/Tr)    (Eq. 9)
  pooling       P_ig = A_ig / sum_j A_jg,  Z = P^T H              (Eq. 3)
  ungroup       H~ = A M(Z) + H W^T                               (Eq. 3)
  compression   L = r/(r-1)[(r-1) fh fs + (1-fh)(1-fs)], fh const (Eq. 10)
  soft count    K~ = Nb - sum_m sigmoid((s_m - tau)/Tb)           (Eq. 11)
  denoise       L_diff = MSE(pred, x) / max(1-t, 0.05)^2          (Eq. 12)

The hard tree is computed with S.detach() (no gradient through topology, as
in the paper); merge affinities and soft weights stay differentiable.

This is a REFERENCE, not a Bend port. It exists to answer, cheaply, whether
the grouping mechanism learns coherent regions on these flowers before any
Bendlang implementation is paid for. Numbers here are CPU torch 2.15.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from flowers_report import legibility  # noqa: E402
from unet_flowers_eval import spectrum  # noqa: E402

DATA = os.path.expanduser("~/.cache/bend-lab/flowers")
EDGE = 64
PATCH = 8
GRID = EDGE // PATCH          # 8
TOKENS = GRID * GRID          # 64
PATCH_DIM = PATCH * PATCH     # 64
TR = 0.05                     # paper: routing softmax temperature
TB = 0.1                      # paper: soft group-count temperature
LAMBDA = 0.01                 # paper: compression-loss weight
AMIN = 0.05                   # paper: velocity-denominator floor
RATIO = 2.0                   # two successive 2x grouping stages


def load_flowers(n_train=1024, n_hold=102):
    tr = np.fromfile(os.path.join(DATA, "flowers64-trainfull-gray.u8"),
                     dtype=np.uint8).reshape(-1, EDGE, EDGE)[:n_train].astype(np.float32) / 255.0
    ho = np.fromfile(os.path.join(DATA, "flowers64-hold-gray.u8"),
                     dtype=np.uint8).reshape(-1, EDGE, EDGE)[:n_hold].astype(np.float32) / 255.0
    return tr, ho


def patches(x):
    # (B,64,64) -> (B,64 patches,64 values), row-major
    B = x.shape[0]
    x = x.reshape(B, GRID, PATCH, GRID, PATCH).permute(0, 1, 3, 2, 4).reshape(B, TOKENS, PATCH_DIM)
    return x


def unpatch(z):
    B = z.shape[0]
    return z.reshape(B, GRID, GRID, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(B, EDGE, EDGE)


class Block(nn.Module):
    def __init__(self, d, heads=4, mlp=2):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.heads = heads

    def forward(self, x):
        B, N, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).reshape(B, N, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)          # bidirectional (the paper's ViT)
        y = self.proj(y.transpose(1, 2).reshape(B, N, d))
        x = x + y
        return x + self.ff(self.ln2(x))


class Blocks(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class Group(nn.Module):
    """A HiT grouping stage: hard average-linkage tree + soft pooling/ungrouping."""
    def __init__(self, d, ratio=RATIO):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.skip = nn.Linear(d, d)
        with torch.no_grad():
            self.skip.weight.copy_(torch.eye(d))
            self.skip.bias.zero_()
        self.ratio = ratio
        self.register_buffer("tau", torch.tensor(-1.0))  # frozen after calibration

    def affinity(self, H):
        q = F.normalize(self.q(H), dim=-1)
        k = F.normalize(self.k(H), dim=-1)
        S = 0.5 * (q @ k.transpose(-1, -2) + k @ q.transpose(-1, -2))  # (B,N,N)
        return S

    def tree(self, S, tau):
        """Average-linkage tree with merges at affinity >= tau (Eq. 8).

        scipy's `linkage(method='average')` is textbook average linkage; its
        merge heights are the mean pairwise affinities. Hard labels come from
        fcluster at distance 1-tau. Merge affinities are recomputed from S over
        the merged members so the soft count (Eq. 11) stays differentiable.
        Topology (which pairs merge) is fixed in backprop, as in the paper.
        """
        B, N, _ = S.shape
        Sd = S.detach().cpu().numpy()
        hard, all_affs, fh = [], [], []
        for b in range(B):
            if N <= 1:
                hard.append(torch.zeros(N, dtype=torch.long, device=S.device))
                all_affs.append([])
                fh.append(1.0)
                continue
            iu = np.triu_indices(N, 1)
            dist = 1.0 - Sd[b][iu]
            Zc = scipy_linkage(dist, method="average")
            # hard labels at threshold tau
            lab = fcluster(Zc, 1.0 - tau, criterion="distance")
            _, inv = np.unique(lab, return_inverse=True)
            hard.append(torch.tensor(inv, dtype=torch.long, device=S.device))
            fh.append(len(np.unique(inv)) / N)
            # track members by cluster id (originals 0..N-1, new ids N..N+m-1)
            members = {i: [i] for i in range(N)}
            affs = []
            for m in range(N - 1):
                i, j = int(Zc[m, 0]), int(Zc[m, 1])
                ci, cj = members[i], members[j]
                # differentiable mean affinity of the two merged groups
                idx = torch.tensor(ci + cj, device=S.device, dtype=torch.long)
                sub = S[b][idx][:, idx]
                affs.append(sub.mean())
                members[N + m] = ci + cj
            all_affs.append(affs)
        return torch.stack(hard), all_affs, fh

    def forward(self, H):
        S = self.affinity(H)
        B, N, d = H.shape
        # tau is frozen after one-time calibration on fixed noisy examples
        # (paper A.3); the compression loss, not tau, adapts routing during training.
        tau = float(self.tau.item()) if self.tau.item() >= 0 else 0.0
        hard, merge_tree, fh = self.tree(S, tau)
        K = int(hard.max().item()) + 1

        # soft weights (Eq. 9), differentiable through S
        e = torch.zeros(B, N, K, device=H.device)
        for b in range(B):
            for g in range(K):
                idx = (hard[b] == g).nonzero(as_tuple=True)[0]
                if idx.numel():
                    e[b, :, g] = S[b][:, idx].mean(dim=1)
        A = torch.softmax(e / TR, dim=-1)                       # (B,N,K)
        P = A / (A.sum(dim=1, keepdim=True) + 1e-8)             # column-normalized
        Z = torch.bmm(P.transpose(1, 2), H)                     # (B,K,d)
        # soft group count (Eq. 11): every merge contributes sigmoid((s-tau)/Tb)
        fs = torch.zeros(B, device=H.device)
        for b in range(B):
            affs = merge_tree[b]
            if affs:
                affs = torch.stack(affs)
                Ktilde = N - torch.sigmoid((affs - tau) / TB).sum()
                fs[b] = Ktilde / N
        comp = (self.ratio / (self.ratio - 1)) * (
            (self.ratio - 1) * torch.tensor(fh, device=H.device).mean() * fs.mean()
            + (1 - torch.tensor(fh, device=H.device).mean()) * (1 - fs.mean()))
        return Z, A, comp, hard, K


class HiT(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.embed = nn.Linear(PATCH_DIM, d)
        self.temb = nn.Linear(1, d)
        self.enc = Blocks(2, d)
        self.g1 = Group(d)
        self.mid = Blocks(2, d)
        self.g2 = Group(d)
        self.core = Blocks(2, d)
        self.dec_mid = Blocks(2, d)
        self.dec = Blocks(2, d)
        self.head = nn.Linear(d, PATCH_DIM)

    def forward(self, x, t, return_groups=False):
        # x: (B,64,64) in [0,1]; t: (B,) in [0,1]
        H = self.embed(patches(x)) + self.temb(t.reshape(-1, 1).float()).unsqueeze(1)
        e1 = self.enc(H)                                  # (B, N=64, d)
        Z1, A1, c1, hard1, K1 = self.g1(e1)               # (B, K1, d); A1 (B,N,K1)
        e2 = self.mid(Z1)                                 # intermediate encoder at K1
        Z2, A2, c2, hard2, K2 = self.g2(e2)               # (B, K2, d); A2 (B,K1,K2)
        core = self.core(Z2)                              # compact bottleneck at K2
        # ungroup level 2: restore K1 positions
        H2 = torch.bmm(A2, core) + e2 @ self.g2.skip.weight.T + self.g2.skip.bias
        H2 = self.dec_mid(H2)                             # intermediate decoder at K1
        # ungroup level 1: restore N positions
        H3 = torch.bmm(A1, H2) + e1 @ self.g1.skip.weight.T + self.g1.skip.bias
        H3 = self.dec(H3)                                 # full-resolution decoder
        pred = unpatch(self.head(H3))
        out = (pred, c1 + c2)
        if return_groups:
            out += (hard1, K1, hard2, K2, A1, A2, e1, e2)
        return out


class ViT(nn.Module):
    """Matched plain transformer: same patch/width/blocks, no grouping."""
    def __init__(self, d=64):
        super().__init__()
        self.embed = nn.Linear(PATCH_DIM, d)
        self.temb = nn.Linear(1, d)
        self.blocks = Blocks(10, d)
        self.head = nn.Linear(d, PATCH_DIM)

    def forward(self, x, t, return_groups=False):
        H = self.embed(patches(x)) + self.temb(t.reshape(-1, 1).float()).unsqueeze(1)
        H = self.blocks(H)
        pred = unpatch(self.head(H))
        return (pred, H.new_zeros(())) + ((None,) * 8 if return_groups else ())


def calibrate(model, tr, n=256, t=0.5):
    """Freeze each router's tau so mean group count ~= N/ratio (paper A.3).

    Calibrate sequentially: stage 2 sees the features stage 1 produces.
    Uses fixed noisy examples with a fixed seed; tau is then never updated.
    """
    torch.manual_seed(1234)
    x = torch.from_numpy(tr[:n])
    tt = torch.full((n,), float(t))
    xt = tt.reshape(-1, 1, 1) * x + (1 - t) * torch.randn_like(x)
    model.eval()
    with torch.no_grad():
        H = model.embed(patches(xt)) + model.temb(tt.reshape(-1, 1).float()).unsqueeze(1)
        e1 = model.enc(H)
        if hasattr(model, "g1"):
            S = model.g1.affinity(e1)
            N = S.shape[1]
            target = max(1, round(N / model.g1.ratio))
            best_tau, best_err = 0.0, 1e9
            for cand in np.linspace(-0.2, 0.95, 47):
                i = np.triu_indices(N, 1)
                ds = 1.0 - S.detach().cpu().numpy()[0][i]
                Zc = scipy_linkage(ds, method="average")
                labs = fcluster(Zc, 1.0 - float(cand), criterion="distance")
                err = abs(len(np.unique(labs)) - target)
                if err < best_err:
                    best_err, best_tau = err, float(cand)
            model.g1.tau.fill_(best_tau)
            Z1, _, _, _, _ = model.g1(e1)
            e2 = model.mid(Z1)
            S2 = model.g2.affinity(e2)
            N2 = S2.shape[1]
            target2 = max(1, round(N2 / model.g2.ratio))
            best2, err2 = 0.0, 1e9
            for cand in np.linspace(-0.2, 0.95, 47):
                i = np.triu_indices(N2, 1)
                ds = 1.0 - S2.detach().cpu().numpy()[0][i]
                Zc = scipy_linkage(ds, method="average")
                labs = fcluster(Zc, 1.0 - float(cand), criterion="distance")
                err = abs(len(np.unique(labs)) - target2)
                if err < err2:
                    err2, best2 = err, float(cand)
            model.g2.tau.fill_(best2)
            print(f"calibrated tau: L1={best_tau:.3f} (target {target}), "
                  f"L2={best2:.3f} (target {target2})", flush=True)
    return model


def sample(model, x0, steps=32):
    model.eval()
    x = x0
    for t in np.linspace(0, 1, steps + 1)[:-1]:
        with torch.no_grad():
            pred, _ = model(x, torch.full((x.shape[0],), float(t)))
        v = (pred - x) / max(1 - t, AMIN)
        x = x + v * (1.0 / steps)
    return torch.clamp(x, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["hit", "vit"], default="hit")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--generated_samples", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups-at", type=float, default=0.75)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tr, ho = load_flowers(args.n_train)
    tr_t = torch.from_numpy(tr)
    model = HiT() if args.arm == "hit" else ViT()
    if args.arm == "hit":
        model = calibrate(model, tr)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    params = sum(p.numel() for p in model.parameters())
    os.makedirs(args.out, exist_ok=True)
    print(f"arm={args.arm} params={params} steps={args.steps} batch={args.batch} "
          f"train={len(tr)} hold={len(ho)}", flush=True)

    t0 = time.time()
    losses = []
    model.train()
    for step in range(args.steps):
        idx = np.random.randint(0, len(tr), args.batch)
        x = tr_t[idx]
        tt = torch.rand(args.batch)
        xt = tt.reshape(-1, 1, 1) * x + (1 - tt.reshape(-1, 1, 1)) * torch.randn_like(x)
        pred, comp = model(xt, tt)
        denom = torch.clamp(1 - tt, min=AMIN).reshape(-1, 1, 1)
        loss = ((pred - x) ** 2).mean() / (denom ** 2).mean() + LAMBDA * comp
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % max(1, args.steps // 10) == 0:
            print(f"step {step+1}/{args.steps} loss {losses[-1]:.5f}", flush=True)

    # sample from noise on held-out-shaped starting clouds
    B = args.generated_samples
    x0 = torch.randn(B, EDGE, EDGE)
    generated_samples = sample(model, x0).numpy()
    held = legibility(generated_samples.reshape(B, -1), ho.reshape(-1, EDGE * EDGE))
    train_leg = legibility(generated_samples.reshape(B, -1), tr.reshape(-1, EDGE * EDGE))
    positive_control = legibility(ho.reshape(-1, EDGE * EDGE), tr.reshape(-1, EDGE * EDGE))
    noise_leg = legibility(np.random.randn(B, EDGE * EDGE).astype(np.float32),
                           tr.reshape(-1, EDGE * EDGE))
    sp = spectrum(generated_samples)
    rec = {
        "arm": args.arm, "paper": "arXiv:2611.22962", "params": params,
        "steps": args.steps, "batch": args.batch, "seed": args.seed,
        "patch": PATCH, "tokens": TOKENS, "width": 64,
        "losses": losses, "first_loss": losses[0], "final_loss": losses[-1],
        "legibility_test": float(held.mean()), "legibility_training": float(train_leg.mean()),
        "positive_control": float(positive_control.mean()), "negative_control": float(noise_leg.mean()),
        "spectrum": [float(x) for x in sp], "wall_seconds": time.time() - t0,
        "generated_samples": B,
    }

    group_data = None
    if args.arm == "hit":
        # group maps at t=0.75 on three clean flowers (paper Figure 4)
        imgs = torch.from_numpy(tr[:3])
        tt = torch.full((3,), args.groups_at)
        xt = tt.reshape(-1, 1, 1) * imgs + (1 - args.groups_at) * torch.randn_like(imgs)
        out = model(xt, tt, return_groups=True)
        hard1, K1, hard2, K2, A1, A2 = out[2], out[3], out[4], out[5], out[6], out[7]
        group_data = {
            "images": tr[:3].tolist(),
            "hard1": hard1.tolist(), "K1": K1,
            "hard2": hard2.tolist(), "K2": K2,
            "A1": A1.tolist(),
            "t": args.groups_at,
        }
        rec["group_counts"] = {"level1": K1, "level2": K2}

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    json.dump(rec, open(os.path.join(args.out, "run.json"), "w"), indent=1)
    if group_data:
        json.dump(group_data, open(os.path.join(args.out, "groups.json"), "w"))
    np.save(os.path.join(args.out, "generated_samples.npy"), generated_samples)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in rec.items() if k != "losses"}, indent=1), flush=True)
    print(f"wrote {args.out}/run.json", flush=True)


if __name__ == "__main__":
    main()
