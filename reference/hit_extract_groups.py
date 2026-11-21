#!/usr/bin/env python3
"""Figure-4-style HiT group extraction with a fixed noisy pool.

  reference/.venv/bin/python reference/hit_extract_groups.py results/hit-reference/hit

Calibrates the routing threshold once on a fixed noisy pool at t=0.75 (the
paper's Figure-4 regime), then picks three flowers with the widest affinity
spread so the learned groups are actually informative. One noise draw is used
for calibration and extraction, so the result is reproducible and consistent.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from hit_ref import HiT, load_flowers, patches  # noqa: E402

T = 0.75
POOL = 96


def groups_for(router, feats, tau):
    S = router.affinity(feats)
    N = S.shape[1]
    iu = np.triu_indices(N, 1)
    hard, K = [], []
    for b in range(feats.shape[0]):
        Zc = scipy_linkage(1.0 - S[b].detach().cpu().numpy()[iu], method="average")
        lab = fcluster(Zc, 1.0 - tau, criterion="distance")
        _, inv = np.unique(lab, return_inverse=True)
        hard.append(inv); K.append(int(inv.max()) + 1)
    return np.array(hard), K, S


def calibrate(router, feats, target, taus):
    N = feats.shape[1]
    best_tau, best_err = 0.0, 1e9
    for cand in taus:
        _, K, _ = groups_for(router, feats[:16], float(cand))
        err = abs(int(np.median(K)) - target)
        if err < best_err:
            best_err, best_tau = err, float(cand)
    return best_tau


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "results/hit-reference/hit"
    model = HiT()
    model.load_state_dict(torch.load(os.path.join(run, "model.pt"), weights_only=True))
    model.eval()
    tr, _ = load_flowers(1024)
    torch.manual_seed(7)
    xt = T * torch.from_numpy(tr[:POOL]) + (1 - T) * torch.randn(POOL, 64, 64)

    with torch.no_grad():
        H = model.embed(patches(xt)) + model.temb(torch.full((POOL,), float(T)).reshape(-1, 1)).unsqueeze(1)
        e1 = model.enc(H)
        taus = np.linspace(-0.3, 0.3, 61)
        tau1 = calibrate(model.g1, e1, 32, taus)
        hard1, K1, S1 = groups_for(model.g1, e1, tau1)
        Z1, A1, _, _, _ = model.g1(e1)
        e2 = model.mid(Z1)
        tau2 = calibrate(model.g2, e2, 16, taus)
        hard2, K2, S2 = groups_for(model.g2, e2, tau2)
        A2 = model.g2.affinity(e2)

    # pick 3 flowers by affinity spread (informative grouping)
    spread = S1.reshape(POOL, -1).std(1)
    idx = np.argsort(-spread)[:3].tolist()
    data = {
        "images": tr[idx].tolist(), "indices": idx, "t": T,
        "tau": [tau1, tau2],
        "spread": [float(spread[i]) for i in idx],
        "hard1": [hard1[i].tolist() for i in idx], "K1": [K1[i] for i in idx],
        "hard2": [hard2[i].tolist() for i in idx], "K2": [K2[i] for i in idx],
        "A1": [A1[i].detach().numpy().tolist() for i in idx],
        "A2": [A2[i].detach().numpy().tolist() for i in idx],
    }
    out = os.path.join(run, "groups4.json")
    json.dump(data, open(out, "w"))
    print(f"wrote {out}: flowers {idx}, tau=({tau1:.3f},{tau2:.3f}), "
          f"K1={[K1[i] for i in idx]}, K2={[K2[i] for i in idx]}")


if __name__ == "__main__":
    main()
