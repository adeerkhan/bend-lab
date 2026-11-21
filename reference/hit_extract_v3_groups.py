#!/usr/bin/env python3
"""Extract v3 HiT group boundaries on real flowers at t=0.75.

  reference/.venv/bin/python reference/hit_extract_v3_groups.py results/hit-v3/hit

Mirrors the released Figure 4 protocol: noisy inputs at t=0.75, fixed pool,
hard average-linkage groups from the trained routers, written to groups4.json
for hit_plot_groups_v3.py. The threshold is frozen (no retune) as in the recipe.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.cluster.hierarchy import _hierarchy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from hit_v3 import HiT, load, patchify, TOKENS  # noqa: E402

T, POOL = 0.75, 128


def groups(q, k, tau):
    S = 0.5 * (q @ k.transpose(-1, -2) + k @ q.transpose(-1, -2))
    Sd = S.detach().cpu().numpy()
    iu = np.triu_indices(S.shape[1], 1)
    hard, counts = [], []
    for b in range(S.shape[0]):
        d = np.maximum(0, 1 - Sd[b][iu].astype(np.float64))
        z = _hierarchy.nn_chain(d, S.shape[1], 2)
        sc = 1 - z[:, 2]
        parent = np.arange(2 * S.shape[1] - 1)
        for m in range(S.shape[1] - 1):
            if sc[m] >= tau:
                a, c = int(z[m, 0]), int(z[m, 1])
                while parent[a] != a: a = parent[a]
                while parent[c] != c: c = parent[c]
                parent[c] = a = min(a, c)
        lab = np.zeros(S.shape[1], np.int64); nxt = 0; lookup = {}
        for i in range(S.shape[1]):
            r = i
            while parent[r] != r: r = parent[r]
            if r not in lookup: lookup[r] = nxt; nxt += 1
            lab[i] = lookup[r]
        hard.append(lab); counts.append(int(nxt))
    return np.array(hard), counts


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "results/hit-v3/hit"
    m = HiT()
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), weights_only=True))
    m.eval()
    tr, _ = load(2048)
    torch.manual_seed(7)
    x = torch.from_numpy(tr[:POOL])
    xt = T * x + (1 - T) * torch.randn_like(x)
    with torch.no_grad():
        H = m.embed(patchify(xt)) + m.ctx.expand(POOL, -1, -1)
        cond = m.t(torch.full((POOL,), float(T)).reshape(-1, 1)) + m.y(torch.zeros(POOL, 1))
        e1 = m.enc(H, cond)
        q1, k1 = m.g1.project(e1)
        h1, c1 = groups(q1, k1, m.g1.tau.item())
        Z1 = m.g1(e1)[0]
        e2 = m.mid(Z1, cond)
        q2, k2 = m.g2.project(e2)
        h2, c2 = groups(q2, k2, m.g2.tau.item())
    # pick flowers whose affinities have the widest spread (most informative groups)
    S1 = 0.5 * (q1 @ k1.transpose(-1, -2) + k1 @ q1.transpose(-1, -2)).detach().cpu().numpy()
    informative = np.array([S1[i][np.triu_indices(TOKENS, 1)].std() for i in range(POOL)])
    idx = np.argsort(-informative)[:3].tolist()
    data = dict(images=tr[idx].tolist(), indices=idx, t=T,
                tau=[m.g1.tau.item(), m.g2.tau.item()],
                hard1=[h1[i].tolist() for i in idx], K1=[c1[i] for i in idx],
                hard2=[h2[i].tolist() for i in idx], K2=[c2[i] for i in idx])
    out = os.path.join(run, "groups4.json")
    json.dump(data, open(out, "w"))
    print(f"wrote {out}: flowers {idx}, K1={[c1[i] for i in idx]}, K2={[c2[i] for i in idx]}")


if __name__ == "__main__":
    main()
