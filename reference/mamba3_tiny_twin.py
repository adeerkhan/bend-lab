"""The tiny-LM twin: the byte corpus in a two-layer SISO Mamba-3 LM, PyTorch.

  reference/.venv-rf/bin/python reference/mamba3_tiny_twin.py --mode dump
  reference/.venv-rf/bin/python reference/mamba3_tiny_twin.py --mode train
  reference/.venv-rf/bin/python reference/mamba3_tiny_twin.py --mode eval

Reads `.scratch/mamba3_tiny/manifest.json` (written by
`reference/mamba3_tiny_data.py`), builds the two-layer SISO LM (the same
block the copy task gates) at d_model 64, d_state 16, expand 2, headdim 32,
vocab 127 (the manifest's observed maximum + 1), and either dumps the fixed
seed-101/102 init the Bend trainer reads, trains it on the train split, or
evaluates the trained model on the held-out split.

The training schedule is the Bend trainer's, exactly:

  examples in file order, batch 1, targets are the next byte within the
  example (the last position is dropped, 127 targets per example)
  AdamW lr = warmup(5) then cosine from 1e-3 to 1e-4, betas (0.9, 0.999),
  eps 1e-8, weight_decay 0.01, gradient clipping by global norm at 1.0,
  bias correction, decoupled decay, torch's step order
  `steps` steps (the first `steps` train examples)

`dump` writes `.scratch/mamba3_tiny/w_init/*.bin` (one file per parameter,
the manifest's flat order) plus `w_init_flat.bin` and a sha256 table in the
manifest of `w_init/manifest.json`; `train` writes `twin.json` (per-step
losses, grad norms, final per-tensor sums of squares, held-out bits/byte,
held-out token accuracy, train accuracy on the used examples) and
`twin_losses.bin`.

Bits/byte = the summed per-token cross entropy in nats over the held-out
positions divided by the number of held-out target bytes times ln(2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
import mamba3_twin as W  # noqa: E402

OUT = os.path.join(ROOT, ".scratch/mamba3_tiny")
V, DM, POS = 127, 64, 128
TARGETS = POS - 1  # the last position of an example has no next byte
SEED1, SEED2, SEEDX = 101, 102, 20261115
PEAK, FLOOR, WARMUP = 1e-3, 1e-4, 5
WD, BETA1, BETA2, EPS, CLIP = 0.01, 0.9, 0.999, 1e-8, 1.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_module(seed):
    torch.manual_seed(seed)
    return W.PureMamba3(d_model=DM, d_state=16, expand=2, headdim=32,
                        ngroups=1, is_mimo=False).float()


def init_extra():
    g = torch.Generator()
    g.manual_seed(SEEDX)
    E = torch.randn(V, DM, generator=g) * 0.5
    Wh = torch.randn(V, DM, generator=g) * 0.02
    nw = 1.0 + torch.randn(DM, generator=g) * 0.1
    return E, Wh, nw


def lm_forward(m1, m2, E, Wh, nw, tokens):
    u = E[torch.tensor(tokens, dtype=torch.long)].unsqueeze(0)
    f1 = W.twin_forward(m1, u)
    h1 = u + f1["y"]
    f2 = W.twin_forward(m2, h1)
    h2 = h1 + f2["y"]
    hn = W.rms(h2, nw, eps=1e-5)
    logits = hn @ Wh.T
    return {"u": u, "h1": h1, "h2": h2, "hn": hn, "logits": logits}


def params(m):
    return {
        "in_proj": m.in_proj.weight.detach().cpu().numpy(),
        "dt_bias": m.dt_bias.detach().cpu().numpy(),
        "B_bias": m.B_bias.detach().squeeze(1).cpu().numpy(),
        "C_bias": m.C_bias.detach().squeeze(1).cpu().numpy(),
        "B_norm_w": m.B_norm.weight.detach().cpu().numpy(),
        "C_norm_w": m.C_norm.weight.detach().cpu().numpy(),
        "D": m.D.detach().cpu().numpy(),
        "out_proj": m.out_proj.weight.detach().cpu().numpy(),
    }


def lr_of(step, steps):
    if step < WARMUP:
        return PEAK * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, steps - WARMUP)
    return FLOOR + (PEAK - FLOOR) * 0.5 * (1.0 + math.cos(math.pi * t))


def load_manifest():
    return json.load(open(os.path.join(OUT, "manifest.json")))


def read_tokens(split):
    raw = np.fromfile(os.path.join(OUT, f"{split}_tokens.bin"), dtype=np.uint8)
    n = raw.size // POS
    return raw.reshape(n, POS)


def dump():
    os.makedirs(os.path.join(OUT, "w_init"), exist_ok=True)
    m1, m2 = make_module(SEED1), make_module(SEED2)
    E, Wh, nw = init_extra()
    files = {}
    flat = []
    order = []
    for tag, m in (("l1", m1), ("l2", m2)):
        for name, a in params(m).items():
            f = os.path.join("w_init", f"w_{tag}_{name}.bin")
            np.ascontiguousarray(a.astype("<f4")).tofile(os.path.join(OUT, f))
            files[f"{tag}_{name}"] = f
            order.append((f"{tag}_{name}", f))
            flat.append(a.astype("<f4").reshape(-1))
    for name, a in (("E", E), ("Wh", Wh), ("nw", nw)):
        f = os.path.join("w_init", f"w_{name}.bin")
        np.ascontiguousarray(a.detach().numpy().astype("<f4")).tofile(
            os.path.join(OUT, f))
        files[name] = f
        order.append((name, f))
        flat.append(a.detach().numpy().astype("<f4").reshape(-1))
    flat = np.concatenate(flat).astype("<f4")
    flat.tofile(os.path.join(OUT, "w_init_flat.bin"))
    man = {
        "seed": {"l1": SEED1, "l2": SEED2, "extra": SEEDX},
        "vocab": V, "d_model": DM, "positions": POS,
        "targets_per_example": TARGETS,
        "order": order,
        "param_count": int(flat.size),
        "schedule": {"peak": PEAK, "floor": FLOOR, "warmup": WARMUP,
                     "weight_decay": WD, "betas": [BETA1, BETA2], "eps": EPS,
                     "clip": CLIP},
        "files": {},
    }
    for key, f in files.items():
        man["files"][key] = {"file": f.replace(os.sep, "/"),
                             "sha256": sha256(os.path.join(OUT, f))}
    man["files"]["w_init_flat"] = {"file": "w_init_flat.bin",
                                   "sha256": sha256(os.path.join(OUT, "w_init_flat.bin")),
                                   "count": int(flat.size)}
    with open(os.path.join(OUT, "w_init", "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"wrote {OUT}/w_init ({len(order)} tensors, {flat.size} parameters)")


def load_w(name, ref):
    a = np.fromfile(os.path.join(OUT, "w_init", f"w_{name}.bin"),
                    dtype="<f4").astype(np.float32)
    return torch.from_numpy(a).reshape(ref.shape)


def build(manifest, steps):
    m1, m2 = make_module(SEED1), make_module(SEED2)
    for tag, m in (("l1", m1), ("l2", m2)):
        sd = {"in_proj.weight": load_w(f"{tag}_in_proj", m.in_proj.weight),
              "dt_bias": load_w(f"{tag}_dt_bias", m.dt_bias),
              "B_bias": load_w(f"{tag}_B_bias", m.B_bias.squeeze(1)).unsqueeze(1),
              "C_bias": load_w(f"{tag}_C_bias", m.C_bias.squeeze(1)).unsqueeze(1),
              "B_norm.weight": load_w(f"{tag}_B_norm_w", m.B_norm.weight),
              "C_norm.weight": load_w(f"{tag}_C_norm_w", m.C_norm.weight),
              "D": load_w(f"{tag}_D", m.D),
              "out_proj.weight": load_w(f"{tag}_out_proj", m.out_proj.weight)}
        m.load_state_dict(sd, strict=True)
    E, Wh, nw = init_extra()
    E = load_w("E", E).clone().requires_grad_(True)
    Wh = load_w("Wh", Wh).clone().requires_grad_(True)
    nw = load_w("nw", nw).clone().requires_grad_(True)
    return m1, m2, E, Wh, nw


STEPS = 20


def train(steps):
    manifest = load_manifest()
    m1, m2, E, Wh, nw = build(manifest, steps)
    tokens = read_tokens("train")
    held = read_tokens("heldout")
    opt = torch.optim.AdamW(list(m1.parameters()) + list(m2.parameters())
                            + [E, Wh, nw], lr=PEAK, weight_decay=WD,
                            betas=(BETA1, BETA2), eps=EPS)

    def set_lr(s):
        for g in opt.param_groups:
            g["lr"] = lr_of(s, steps)

    losses, norms = [], []
    hits, total = 0, 0
    for s in range(steps):
        set_lr(s)
        tgt = torch.tensor(tokens[s, 1:], dtype=torch.long)
        out = lm_forward(m1, m2, E, Wh, nw, tokens[s])
        loss = F.cross_entropy(out["logits"][0, :TARGETS], tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(m1.parameters())
                                            + list(m2.parameters())
                                            + [E, Wh, nw], CLIP)
        opt.step()
        with torch.no_grad():
            hits += int((out["logits"][0, :TARGETS].argmax(-1) == tgt).sum())
            total += TARGETS
        losses.append(float(loss.detach()))
        norms.append(float(gn))
        print(f"step {s} lr {lr_of(s, steps):.6f} loss {losses[-1]:.7f} "
              f"gn {float(gn):.4f}")

    final_train_acc = hits / total
    held_ce_nats, held_hits = 0.0, 0
    with torch.no_grad():
        for i in range(held.shape[0]):
            out = lm_forward(m1, m2, E, Wh, nw, held[i])
            lp = out["logits"][0, :TARGETS]
            tgt = torch.tensor(held[i, 1:], dtype=torch.long)
            held_ce_nats += float(F.cross_entropy(lp, tgt, reduction="sum"))
            held_hits += int((lp.argmax(-1) == tgt).sum())
    bits = held_ce_nats / (held.shape[0] * TARGETS * math.log(2))
    held_acc = held_hits / (held.shape[0] * TARGETS)

    np.asarray(losses, dtype="<f4").tofile(os.path.join(OUT, "twin_losses.bin"))
    # the flat layout the Bend trainer and the gate read, in the manifest order
    tens = {}
    for t, m in (("l1", m1), ("l2", m2)):
        for name, p in m.named_parameters():
            short = name.replace(".weight", "").replace("B_norm", "B_norm_w").replace(
                "C_norm", "C_norm_w").replace("B_bias", "B_bias").replace("C_bias", "C_bias")
            tens[f"{t}_{short}"] = p.detach().numpy().astype("<f4")
    for name, p in (("E", E), ("Wh", Wh), ("nw", nw)):
        tens[name] = p.detach().numpy().astype("<f4")
    man = json.load(open(os.path.join(OUT, "w_init", "manifest.json")))
    flat = np.concatenate([tens[name].reshape(-1) for name, _ in man["order"]]).astype("<f4")
    np.ascontiguousarray(flat).tofile(os.path.join(OUT, "twin_params_final.bin"))
    checksums = {}
    for name, a in tens.items():
        checksums[name] = float((a.astype(np.float64) ** 2).sum())
    rec = {
        "config": {"seed": {"l1": SEED1, "l2": SEED2, "extra": SEEDX},
                   "dtype": "float32", "device": "cpu", "torch": torch.__version__,
                   "steps": steps, "batch": 1, "positions": POS,
                   "targets": TARGETS, "vocab": V, "peak_lr": PEAK,
                   "floor_lr": FLOOR, "warmup": WARMUP, "weight_decay": WD,
                   "betas": [BETA1, BETA2], "eps": EPS, "clip": CLIP,
                   "loss": "cross_entropy (mean over the 127 targets)"},
        "losses": losses, "grad_norms": norms,
        "train_hits": int(hits), "train_total": int(total),
        "train_acc": float(final_train_acc),
        "heldout": {"ce_nats": held_ce_nats,
                    "bytes": int(held.shape[0] * TARGETS),
                    "bits_per_byte": float(bits),
                    "hits": int(held_hits),
                    "total": int(held.shape[0] * TARGETS),
                    "accuracy": float(held_acc)},
        "final_sumsq": checksums,
        "final_param_count": int(flat.size),
    }
    with open(os.path.join(OUT, "twin.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"held-out bits/byte {bits:.6f}  accuracy {held_acc:.4f} "
          f"({held_hits}/{held.shape[0] * TARGETS})")
    print(f"train acc (used examples) {final_train_acc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dump", "train", "eval"], required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    if args.mode == "dump":
        dump()
    elif args.mode == "train":
        train(args.steps)
    else:
        raise SystemExit("eval is folded into train; rerun with --mode train")


if __name__ == "__main__":
    main()
