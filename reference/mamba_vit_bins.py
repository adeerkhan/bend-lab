"""Dump MambaViT weights, one forward's intermediates and one backward's
gradients for the Bend gates.

  reference/.venv-rf/bin/python reference/mamba_vit_bins.py

Writes `.scratch/mamba_vit/*.bin` (one array per file, row-major float32):

  * the parameters, `w_<module>_<param>` (A1a/A1b);
  * the forward: `img`, `emb`, `tokens`, `block0..7`, `block0..7_res`,
    `final`, `pooled`, `logits` (A1a/A1b);
  * the loss used for A2 and its backward: `dy`, the gradient of
    `loss = sum(logits * dy)` with respect to every parameter
    (`g_<module>_<param>`, the parameter names above) and with respect to
    the intermediates the probe compares: `g_emb`, `g_tokens`,
    `g_block0..7`, `g_final`, `g_pooled`.

`dy` is drawn from its own generator, after the forward, so adding the
backward left every forward number above byte-identical.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from mamba_vit_twin import MambaViT  # noqa: E402

OUT = os.path.join(ROOT, ".scratch/mamba_vit")
N_LAYER = 8


def save(name, arr):
    np.ascontiguousarray(np.asarray(arr, dtype="<f4")).tofile(os.path.join(OUT, name + ".bin"))


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    m = MambaViT(image_size=28, patch_size=4, num_classes=10, channels=1,
                 n_layer=N_LAYER, dim=32, pool="mean").eval()
    # a fixed deterministic image; the Bend probe reads this same img.bin
    img = (torch.arange(28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28) * 0.17).sin() * 0.5
    save("img", img.reshape(-1))

    # the forward, kept in the graph: every intermediate the probe compares
    # carries its gradient
    emb = m.patches(img)                                   # (1,49,32)
    emb.retain_grad()
    toks = torch.cat((m.cls_token.expand(1, -1, -1), emb), 1)
    toks = toks + m.pos_embedding[:, : toks.shape[1]]
    toks.retain_grad()
    with torch.no_grad():
        save("emb", emb[0])
        save("tokens", toks[0])                            # (50,32)
    res = None
    outs = []
    x = toks
    for i, layer in enumerate(m.layers):
        x, res = layer(x, res)
        x.retain_grad()
        outs.append(x)
        with torch.no_grad():
            save(f"block{i}", x[0])
            save(f"block{i}_res", res[0])
    res = x + res
    xf = m.norm_f(res)
    xf.retain_grad()
    with torch.no_grad():
        save("final", xf[0])
    pooled = xf.mean(dim=1)
    pooled.retain_grad()
    with torch.no_grad():
        save("pooled", pooled[0])
    logits = m.mlp_head(pooled)
    with torch.no_grad():
        save("logits", logits[0])
        print("img", img.numel(), "tokens", tuple(xf.shape), "logits", tuple(logits.shape))

    # A2: a fixed scalar loss, and the gradient of everything it depends on.
    # dy comes from its own generator so the forward files stay identical.
    gen = torch.Generator().manual_seed(1234)
    dy = torch.randn(10, generator=gen)
    save("dy", dy)
    loss = (logits[0] * dy).sum()
    m.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.grad is None:
                raise SystemExit(f"no gradient for {name}")
            save("g_" + name.replace(".", "_"), p.grad)
        save("g_emb", emb.grad[0])
        save("g_tokens", toks.grad[0])
        for i, o in enumerate(outs):
            save(f"g_block{i}", o.grad[0])
        save("g_final", xf.grad[0])
        save("g_pooled", pooled.grad[0])
        print("loss", float(loss), "dy", np.round(dy.numpy(), 5))
        print("|g_pos_embedding|max", float(m.pos_embedding.grad.abs().max()),
              "|g_cls|max", float(m.cls_token.grad.abs().max()))

    # weights, one file per parameter tensor
    for name, p in m.named_parameters():
        save("w_" + name.replace(".", "_"), p.detach().numpy())
    files = [f for f in os.listdir(OUT) if f.endswith(".bin")]
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump({"files": len(files), "image": "sin(arange(784) * 0.17) * 0.5",
                   "torch": torch.__version__}, f, indent=2)
        f.write("\n")
    print("wrote", len(files), "files to", OUT)
    print("logits", np.round(logits[0].detach().numpy(), 5))


if __name__ == "__main__":
    main()
