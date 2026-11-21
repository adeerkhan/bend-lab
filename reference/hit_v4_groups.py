#!/usr/bin/env python3
"""Group boundaries from the v4 model's OWN forward(return_aux=True).

  reference/.venv/bin/python reference/hit_v4_groups.py results/hit-v4/hit

No separate grouping function: the assignments come from the same code path
used in training, at t=0.75, thresholds frozen.
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from hit_v4 import HiT, load, GRID

run = sys.argv[1] if len(sys.argv) > 1 else "results/hit-v4/hit"
m = HiT(); m.load_state_dict(torch.load(os.path.join(run, "model.pt"), weights_only=True)); m.eval()
tr, _ = load(2048)
torch.manual_seed(7)
idx = [45, 29, 26]
x = torch.from_numpy(tr[idx]); t = 0.75
tt = torch.full((3,), float(t))
xt = tt.reshape(-1, 1, 1) * x + (1 - t) * torch.randn_like(x)
with torch.no_grad():
    pred, comp, aux = m(xt, tt, torch.zeros(3), return_aux=True)
h1 = aux["hard1"].cpu().numpy(); c1 = aux["counts1"].cpu().numpy()
h2 = aux["hard2"].cpu().numpy(); c2 = aux["counts2"].cpu().numpy()
data = dict(images=tr[idx].tolist(), indices=idx, t=t,
            tau=[aux["tau1"], aux["tau2"]],
            hard1=[h1[i].tolist() for i in range(3)], K1=[int(c1[i]) for i in range(3)],
            hard2=[h2[i].tolist() for i in range(3)], K2=[int(c2[i]) for i in range(3)])
out = os.path.join(run, "groups4.json")
json.dump(data, open(out, "w"))
print(f"wrote {out}: K1={data['K1']} K2={data['K2']} tau={data['tau']}")
