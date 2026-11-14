"""Dump MambaViT weights and one forward's intermediates for the Bend gate.

  reference/.venv-rf/bin/python reference/mamba_vit_bins.py

Writes `.scratch/mamba_vit/npz`-style `.bin` files (one array per file) and
the reference intermediates for a fixed MNIST image: the patch tokens after
the embedding, every block's output, the final pooled vector and the logits.
The Bend probe loads these and must match every intermediate at 1e-5.
"""
from __future__ import annotations

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
    # a fixed small image: the first MNIST test image, as 0..1 floats
    import torchvision
    from torchvision import transforms
    ds = torchvision.datasets.MNIST("/tmp/opencode/mnist", train=False, download=True,
                                    transform=transforms.ToTensor())
    img = ds[0][0]  # (1,28,28) in 0..1
    img = img.unsqueeze(0) - 0.5
    save("img", img.reshape(-1))
    with torch.no_grad():
        emb = m.patches(img)[0]                                # (49,32)
        x = torch.cat((m.cls_token.expand(1, -1, -1), emb.unsqueeze(0)), 1)
        x = x + m.pos_embedding[:, : x.shape[1]]
        save("emb", emb)
        save("tokens", x[0])                                   # (50,32)
        res = None
        for i, layer in enumerate(m.layers):
            x, res = layer(x, res)
            save(f"block{i}", x[0])
            for nm, t in (("norm", layer.norm), ("mixer", layer.mixer)):
                pass
            save(f"block{i}_res", res[0])
        res = x + res
        x = m.norm_f(res)
        save("final", x[0])
        pooled = x.mean(dim=1)
        save("pooled", pooled[0])
        logits = m.mlp_head(pooled)
        save("logits", logits[0])
        print("img", img.numel(), "tokens", tuple(x.shape), "logits", tuple(logits.shape))
    # weights, one file per parameter tensor
    for name, p in m.named_parameters():
        save("w_" + name.replace(".", "_"), p.detach().numpy())
    print("wrote", len([f for f in os.listdir(OUT) if f.endswith(".bin")]), "files to", OUT)
    print("logits", np.round(logits[0].numpy(), 5))


if __name__ == "__main__":
    main()
