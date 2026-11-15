"""The MambaViT A3 twin run: the same subset, the same schedule, in PyTorch.

  reference/.venv-rf/bin/python reference/mamba_vit_twin_train.py [--dtype float64]

Reads `.scratch/mv_mnist/manifest.json` (written by
`reference/mamba_vit_train_data.py`), loads the dumped seed-0 init into
`mamba_vit_twin.MambaViT`, and trains exactly the schedule the Bend trainer
runs:

  the first `train_used` images, in file order, batch 32, no shuffling
  AdamW lr 1e-3, weight_decay 0.01, betas (0.9, 0.999), eps 1e-8
  gradient clipping by global norm at 1.0, constant lr, `steps` steps =
  one pass over the used subset

and writes `twin.json` (config, per-step losses, train/test accuracy, the
final per-tensor checksums and the sha256 of the float32 final parameters)
plus `twin_losses.bin` (the step losses, float32). The dtype is float64 by
default: the twin is the oracle, and float64 keeps its own rounding out of a
1e-3 comparison with the Bend trainer's float32.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from mamba_vit_twin import MambaViT  # noqa: E402

OUT = os.path.join(ROOT, ".scratch/mv_mnist")
N_LAYER = 8
LR, WD, BETA1, BETA2, EPS, CLIP = 1e-3, 0.01, 0.9, 0.999, 1e-8, 1.0


def build(m, manifest, dtype):
    m = m.to(dtype)
    shapes = {n: p.shape for n, p in m.named_parameters()}
    state = {}
    for p in manifest["params"]:
        a = np.fromfile(os.path.join(OUT, p["file"]), dtype="<f4").reshape(-1)
        state[p["name"]] = torch.from_numpy(
            a.astype(np.float64 if dtype == torch.float64 else np.float32)
        ).reshape(shapes[p["name"]])
    m.load_state_dict(state, strict=True)
    return m


def checksums(m, dtype):
    """per-tensor sum of squares, from the float32 view of the parameters"""
    out = {}
    for name, p in m.named_parameters():
        a = p.detach().to(torch.float32).numpy().reshape(-1).astype(np.float64)
        out[name] = float((a * a).sum())
    return out


def evaluate(m, x, y):
    m.eval()
    hit = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], 128):
            out = m(x[i:i + 128])
            hit += int((out.argmax(-1) == y[i:i + 128]).sum())
    m.train()
    return hit / x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=0,
                    help="override the manifest's step count (smoke tests)")
    args = ap.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dev = torch.device(args.device)

    manifest = json.load(open(os.path.join(OUT, "manifest.json")))
    b = manifest["budget"]
    steps, batch, used, test_used = b["steps"], b["batch"], b["train_used"], b["test_used"]
    if args.steps:
        steps = args.steps
        used = steps * batch
    if used != steps * batch:
        raise SystemExit(f"budget: train_used {used} != steps {steps} * batch {batch}")

    x = torch.from_numpy(np.fromfile(os.path.join(OUT, "train_images.bin"),
                                     dtype="<f4").astype(np.float32)).reshape(-1, 784)
    y = torch.from_numpy(np.fromfile(os.path.join(OUT, "train_labels.bin"),
                                     dtype=np.uint8).astype(np.int64))
    xt = torch.from_numpy(np.fromfile(os.path.join(OUT, "test_images.bin"),
                                      dtype="<f4").astype(np.float32)).reshape(-1, 784)
    yt = torch.from_numpy(np.fromfile(os.path.join(OUT, "test_labels.bin"),
                                      dtype=np.uint8).astype(np.int64))
    x = x[:used].to(dev).to(dtype).reshape(used, 1, 28, 28)
    y = y[:used].to(dev)
    xt = xt[:test_used].to(dev).to(dtype).reshape(test_used, 1, 28, 28)
    yt = yt[:test_used].to(dev)

    torch.manual_seed(manifest["seed"])
    m = build(MambaViT(image_size=28, patch_size=4, num_classes=10, channels=1,
                       n_layer=N_LAYER, dim=32, pool="mean"), manifest, dtype).to(dev)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD,
                            betas=(BETA1, BETA2), eps=EPS)

    losses, normsv, hits = [], [], 0
    for s in range(steps):
        xb = x[s * batch:(s + 1) * batch]
        yb = y[s * batch:(s + 1) * batch]
        out = m(xb)
        loss = F.cross_entropy(out, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
        opt.step()
        hits += int((out.argmax(-1) == yb).sum())
        losses.append(float(loss.detach()))
        normsv.append(float(gn.detach()))
        print(f"step {s} loss {losses[-1]:.7f} gn {float(gn):.6f} "
              f"acc {hits / ((s + 1) * batch):.4f}")

    train_acc = evaluate(m, x, y)
    test_acc = evaluate(m, xt, yt)
    print(f"train acc {train_acc:.4f} ({int(round(train_acc * used))}/{used})  "
          f"test acc {test_acc:.4f} ({int(round(test_acc * test_used))}/{test_used})")

    np.asarray(losses, dtype="<f4").tofile(os.path.join(OUT, "twin_losses.bin"))
    params32 = np.concatenate([p.detach().to(torch.float32).numpy().reshape(-1)
                               for _, p in m.named_parameters()]).astype("<f4")
    params32.tofile(os.path.join(OUT, "twin_params_final.bin"))
    rec = {
        "config": {"seed": manifest["seed"], "dtype": args.dtype, "device": args.device,
                   "torch": torch.__version__, "steps": steps, "batch": batch,
                   "train_used": used, "test_used": test_used, "lr": LR, "weight_decay": WD,
                   "betas": [BETA1, BETA2], "eps": EPS, "clip": CLIP, "shuffle": False,
                   "loss": "cross_entropy (mean over the batch)"},
        "losses": losses,
        "grad_norms": normsv,
        "train_acc": train_acc, "test_acc": test_acc,
        # the accuracy of the forward that consumed each batch, before its
        # update: the same definition the Bend trainer prints, which counts a
        # hit when it does the step (it does not re-run the train set)
        "train_hits_preupdate": hits,
        "train_acc_preupdate": hits / used,
        "train_hits": int(round(train_acc * used)), "train_total": used,
        "test_hits": int(round(test_acc * test_used)), "test_total": test_used,
        "final_sumsq": checksums(m, dtype),
        "final_sha256": hashlib.sha256(params32.tobytes()).hexdigest(),
        "final_param_count": int(params32.size),
    }
    with open(os.path.join(OUT, "twin.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("final sha256", rec["final_sha256"][:16], "params", rec["final_param_count"])


if __name__ == "__main__":
    main()
