"""The config LMs for the decode-step gates (C1/C2/C3): one two-layer LM per config.

  reference/.venv-rf/bin/python reference/mamba3_cfg_lm.py --config siso --mode gen --pack varlen --length 16
  reference/.venv-rf/bin/python reference/mamba3_cfg_lm.py --config gqa --mode dump
  reference/.venv-rf/bin/python reference/mamba3_cfg_lm.py --config gqa --mode gen --length 16
  reference/.venv-rf/bin/python reference/mamba3_cfg_lm.py --config on --mode gen --length 64

Builds the two-layer Mamba-3 LM (the copy-task model shape: residual blocks,
final RMSNorm, untied head, vocab 8, d_model 64) in one of the three
breadth configurations:

  siso  ngroups 1, rope_fraction 0.5 (the SISO config; used with the
        copy-task train weights for the varlen pack)
  gqa   ngroups 2, rope_fraction 1.0 (the GQA row; fresh seed 71/72 init)
  on    ngroups 1, is_outproj_norm=True (the outnorm row; fresh seed 71/72
        init; norm.weight = linspace(0.85, 1.15) like the single-block twin)

Modes:

  dump  writes `.scratch/<cfg>_lm_data/` (two layers' weights, E/Wh/nw, a
        manifest.json with sha256s) — the files the Bend probe loads
  gen   greedy decode from the 4-token copy prompt (or the packed
        varlen prompt `[0,1,2,3,0,1,2,3]` with `--pack varlen`, the state
        reset at position 4 and the continuation in the second segment) to
        `--length` total tokens (prompt included); prints
        `gen_tokens: <ints>` and writes `gen<length>.json` next to the
        dump (tokens + the first- and last-position logits)

The seeds are fixed (layers 71/72, E/Wh/nw generator seed 321), so every
number in a gate carries the command that printed it. The SISO pack uses
the copy-task train weights (`reference/mamba3_data/train_*.bin`), the
same files the Bend probe loads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import mamba3_twin as W  # noqa: E402
import mamba3_gqa_twin as WG  # noqa: E402
import mamba3_outnorm_twin as WO  # noqa: E402

V, DM = 8, 64
SEED_L1, SEED_L2, SEED_X = 71, 72, 321


def cfg_module(cfg):
    torch.manual_seed(SEED_L1)
    m1 = cfg_module_one(cfg, SEED_L1)
    m2 = cfg_module_one(cfg, SEED_L2)
    return m1, m2


def cfg_module_one(cfg, seed):
    torch.manual_seed(seed)
    if cfg == "siso":
        return W.Mamba3(d_model=DM, d_state=16, expand=2, headdim=32,
                        ngroups=1, is_mimo=False).float()
    if cfg == "gqa":
        return WG.Mamba3(d_model=DM, d_state=16, expand=2, headdim=32,
                         ngroups=2, rope_fraction=1.0,
                         is_mimo=False, device="cpu",
                         dtype=torch.float32).float()
    if cfg == "on":
        m = WO.Mamba3(d_model=DM, d_state=16, expand=2, headdim=32,
                      ngroups=1, rope_fraction=0.5,
                      is_outproj_norm=True, is_mimo=False,
                      device="cpu", dtype=torch.float32).float()
        with torch.no_grad():
            m.norm.weight.copy_(torch.linspace(0.85, 1.15, m.d_inner))
        return m
    raise SystemExit(f"unknown config {cfg}")


def cfg_forward(cfg, m, u):
    if cfg == "siso":
        return W.twin_forward(m, u)
    if cfg == "gqa":
        return WG.twin_forward(m, u)
    return WO.twin_forward(m, u)


def extra_params():
    g = torch.Generator()
    g.manual_seed(SEED_X)
    E = torch.randn(V, DM, generator=g) * 0.5
    Wh = torch.randn(V, DM, generator=g) * 0.5
    nw = 1.0 + torch.randn(DM, generator=g) * 0.1
    return E, Wh, nw


def load_siso_lm():
    """the copy-task two-layer LM dump (reference/mamba3_lm_data), the same
    files the Bend probe loads"""
    src = os.path.join(ROOT, "reference/mamba3_lm_data")
    m1 = cfg_module_one("siso", SEED_L1)
    m2 = cfg_module_one("siso", SEED_L2)
    for tag, m in (("l1", m1), ("l2", m2)):
        a = np.fromfile(os.path.join(src, f"{tag}_in_proj.bin"), dtype="<f4")
        sd = {"in_proj.weight": torch.from_numpy(a.reshape(304, 64).copy()),
              "dt_bias": torch.from_numpy(np.fromfile(
                  os.path.join(src, f"{tag}_dt_bias.bin"), dtype="<f4").copy()),
              "B_bias": torch.from_numpy(np.fromfile(
                  os.path.join(src, f"{tag}_B_bias.bin"), dtype="<f4").reshape(4, 16)
                  .copy()).unsqueeze(1),
              "C_bias": torch.from_numpy(np.fromfile(
                  os.path.join(src, f"{tag}_C_bias.bin"), dtype="<f4").reshape(4, 16)
                  .copy()).unsqueeze(1),
              "B_norm.weight": torch.from_numpy(np.fromfile(
                  os.path.join(src, f"{tag}_B_norm_w.bin"), dtype="<f4").copy()),
              "C_norm.weight": torch.from_numpy(np.fromfile(
                  os.path.join(src, f"{tag}_C_norm_w.bin"), dtype="<f4").copy()),
              "D": torch.from_numpy(np.fromfile(
                  os.path.join(src, f"{tag}_D.bin"), dtype="<f4").copy()),
              "out_proj.weight": torch.from_numpy(np.fromfile(
                  os.path.join(src, f"{tag}_out_proj.bin"), dtype="<f4").reshape(64, 128).copy())}
        m.load_state_dict(sd, strict=True)
    E = torch.from_numpy(np.fromfile(os.path.join(src, "E0.bin"),
                                     dtype="<f4").reshape(8, 64).copy())
    Wh = torch.from_numpy(np.fromfile(os.path.join(src, "Wh0.bin"),
                                      dtype="<f4").reshape(8, 64).copy())
    nw = torch.from_numpy(np.fromfile(os.path.join(src, "nw0.bin"),
                                      dtype="<f4").copy())
    return m1, m2, E, Wh, nw


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(cfg, out):
    os.makedirs(out, exist_ok=True)
    m1, m2 = cfg_module(cfg)
    E, Wh, nw = extra_params()
    files = {}

    def put(name, arr):
        f = os.path.join(out, name)
        np.ascontiguousarray(arr.astype("<f4")).tofile(f)
        files[name] = sha256(f)

    for tag, m in (("l1", m1), ("l2", m2)):
        put(f"{tag}_in_proj.bin", m.in_proj.weight.detach().numpy())
        put(f"{tag}_dt_bias.bin", m.dt_bias.detach().numpy())
        put(f"{tag}_B_bias.bin", m.B_bias.detach().squeeze(1).numpy())
        put(f"{tag}_C_bias.bin", m.C_bias.detach().squeeze(1).numpy())
        put(f"{tag}_B_norm_w.bin", m.B_norm.weight.detach().numpy())
        put(f"{tag}_C_norm_w.bin", m.C_norm.weight.detach().numpy())
        put(f"{tag}_D.bin", m.D.detach().numpy())
        put(f"{tag}_out_proj.bin", m.out_proj.weight.detach().numpy())
        if cfg == "on":
            put(f"{tag}_norm_w.bin", m.norm.weight.detach().numpy())
    put("E.bin", E.numpy())
    put("Wh.bin", Wh.numpy())
    put("nw.bin", nw.numpy())
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"config": cfg, "seeds": {"l1": SEED_L1, "l2": SEED_L2,
                                            "extra": SEED_X},
                   "vocab": V, "d_model": DM,
                   "files": files}, f, indent=1)
    print(f"wrote {out} ({len(files)} files)")


def lm_h2(cfg, m1, m2, E, tokens, pack):
    """hidden states at every position; a packed varlen prompt runs each
    segment through the model independently and concatenates"""
    if pack == "varlen":
        segs = [tokens[:4], tokens[4:]]
    else:
        segs = [tokens]
    outs = []
    for seg in segs:
        u = E[torch.tensor(seg, dtype=torch.long)].unsqueeze(0)
        f1 = cfg_forward(cfg, m1, u)
        h1 = u + f1["y"]
        f2 = cfg_forward(cfg, m2, h1)
        outs.append(h1 + f2["y"])
    return torch.cat(outs, dim=1)


def gen(cfg, length, pack, out):
    if cfg == "siso":
        m1, m2, E, Wh, nw = load_siso_lm()
    else:
        m1, m2 = cfg_module(cfg)
        E, Wh, nw = extra_params()
    tokens = [0, 1, 2, 3, 0, 1, 2, 3] if pack == "varlen" else [0, 1, 2, 3]
    while len(tokens) < length:
        h2 = lm_h2(cfg, m1, m2, E, tokens, pack)
        hn = W.rms(h2, nw, eps=1e-5)
        nxt = hn[0, -1] @ Wh.T
        tokens.append(int(nxt.argmax()))
    h2 = lm_h2(cfg, m1, m2, E, tokens, pack)
    hn = W.rms(h2, nw, eps=1e-5)
    logits = hn[0] @ Wh.T
    print("gen_tokens:", " ".join(str(t) for t in tokens))
    print(f"gen_len {len(tokens)}")
    if out is not None:
        with open(out, "w") as f:
            json.dump({"config": cfg, "pack": pack, "length": length,
                       "tokens": tokens,
                       "last_logits": logits[-1].tolist()}, f)
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["siso", "gqa", "on"], required=True)
    ap.add_argument("--mode", choices=["dump", "gen"], required=True)
    ap.add_argument("--pack", choices=["plain", "varlen"], default="plain")
    ap.add_argument("--length", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, ".scratch", f"{args.config}_lm_data")
    if args.mode == "dump":
        dump(args.config, out_dir)
    else:
        out = args.out or os.path.join(out_dir, f"gen{args.length}.json")
        gen(args.config, args.length, args.pack, out)


if __name__ == "__main__":
    main()
