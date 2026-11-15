"""The MambaViT A3 subset: the MNIST slice, the seed-0 init, and the manifest.

  reference/.venv-rf/bin/python reference/mamba_vit_train_data.py

Writes `.scratch/mv_mnist/`:

  train_images.bin   2048 x 784 float32, the first MNIST training images, 0..1
  train_labels.bin   2048 uint8
  test_images.bin     512 x 784 float32, the first MNIST test images, 0..1
  test_labels.bin     512 uint8
  w_init/w_<name>.bin  the seed-0 MambaViT init, one file per parameter, in the
                       order `named_parameters()` walks (the flat order both
                       the twin and the Bend trainer read)
  manifest.json        source, counts, sha256 of every file, the parameter
                       table (name, shape, count, flat offset) and the budget

The images are `ToTensor`'s 0..1 floats, not the A1b dump's centered `-0.5`
image: A3 trains the twin's own preprocessing, so the dump and the subset
differ on purpose. The held-out slice is the first 512 MNIST *test* images and
never overlaps the train slice.

The budget is a fixed, honest fraction of the 2048-image subset. Measured
on this lane 2026-11-15 (`reference/test_mamba_vit_train.py`): 1869.8 s wall
for the 160 training + 256 held-out images (4.49 s per image), so the
2048-image subset is on the order of hours and a 60,000-image epoch on the
order of days. The run therefore uses the first `TRAIN_USED` images for
`STEPS` steps and the first `TEST_USED` held-out images, and both sides see
exactly the same slice.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from mamba_vit_twin import MambaViT  # noqa: E402

OUT = os.path.join(ROOT, ".scratch/mv_mnist")
DATA = "/tmp/opencode/mnist"
SEED = 0
TRAIN_IMAGES = 2048
TEST_IMAGES = 512
BATCH = 32
STEPS = 5
TRAIN_USED = BATCH * STEPS       # 160
TEST_USED = 256
N_LAYER = 8


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(os.path.join(OUT, "w_init"), exist_ok=True)
    import torchvision
    from torchvision import transforms

    tf = transforms.Compose([transforms.ToTensor()])
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True, transform=tf)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True, transform=tf)

    # the fixed slice: the first TRAIN_IMAGES / TEST_IMAGES examples, in file order
    tr_img = torch.stack([tr[i][0] for i in range(TRAIN_IMAGES)]).reshape(TRAIN_IMAGES, -1)
    tr_lab = torch.tensor([tr[i][1] for i in range(TRAIN_IMAGES)], dtype=torch.uint8)
    te_img = torch.stack([te[i][0] for i in range(TEST_IMAGES)]).reshape(TEST_IMAGES, -1)
    te_lab = torch.tensor([te[i][1] for i in range(TEST_IMAGES)], dtype=torch.uint8)
    for name, arr in (("train_images", tr_img.numpy().astype("<f4")),
                      ("test_images", te_img.numpy().astype("<f4"))):
        np.ascontiguousarray(arr).tofile(os.path.join(OUT, name + ".bin"))
    for name, arr in (("train_labels", tr_lab.numpy()), ("test_labels", te_lab.numpy())):
        np.ascontiguousarray(arr).tofile(os.path.join(OUT, name + ".bin"))

    # the seed-0 init, dumped in named_parameters() order
    torch.manual_seed(SEED)
    m = MambaViT(image_size=28, patch_size=4, num_classes=10, channels=1,
                 n_layer=N_LAYER, dim=32, pool="mean")
    params = []
    off = 0
    for name, p in m.named_parameters():
        a = np.ascontiguousarray(p.detach().to(torch.float32).numpy().reshape(-1))
        np.ascontiguousarray(a.astype("<f4")).tofile(
            os.path.join(OUT, "w_init", "w_" + name.replace(".", "_") + ".bin"))
        params.append({"name": name, "file": "w_init/w_" + name.replace(".", "_") + ".bin",
                       "shape": list(p.shape), "count": int(a.size), "offset": int(off)})
        off += int(a.size)

    counts = {"train_images": TRAIN_IMAGES, "train_labels": TRAIN_IMAGES,
              "test_images": TEST_IMAGES, "test_labels": TEST_IMAGES}
    files = {}
    for name, n in counts.items():
        p = os.path.join(OUT, name + ".bin")
        files[name] = {"file": name + ".bin", "count": n, "sha256": sha256(p)}
    flat = np.concatenate([np.fromfile(os.path.join(OUT, p["file"]), dtype="<f4")
                           for p in params]).astype("<f4")
    flat.tofile(os.path.join(OUT, "w_init_flat.bin"))
    files["w_init_flat"] = {"file": "w_init_flat.bin", "count": int(flat.size),
                            "sha256": sha256(os.path.join(OUT, "w_init_flat.bin"))}
    assert flat.size == off, (flat.size, off)
    manifest = {
        "source": "torchvision MNIST, /tmp/opencode/mnist, ToTensor (0..1 float32)",
        "seed": SEED,
        "subset": {"train": "the first %d MNIST training images, in file order" % TRAIN_IMAGES,
                   "test": "the first %d MNIST test images, in file order" % TEST_IMAGES},
        "files": files,
        "images_shape": [TRAIN_IMAGES, 784],
        "params": params,
        "param_count": off,
        "budget": {
            "batch": BATCH, "steps": STEPS, "train_used": TRAIN_USED, "test_used": TEST_USED,
            "note": ("the 2048-image subset is the fixed data slice; the run trains on its "
                     "first %d images (%d steps of %d) and evaluates on the first %d held-out "
                     "images; the measured Bend run is 1869.8 s wall for 160 train + 256 held-out "
                     "images (4.49 s per image), so the subset is on the order of hours "
                     "and a full epoch on the order of days" % (TRAIN_USED, STEPS, BATCH, TEST_USED))},
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", OUT, "params", off, "train", TRAIN_IMAGES, "test", TEST_IMAGES)
    print("param files", len(params))


if __name__ == "__main__":
    main()
