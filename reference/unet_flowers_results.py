"""Turn a `train.bend` run into the record: numbers, loss curve, draws, stats.

  reference/.venv/bin/python reference/unet_flowers_results.py /tmp/opencode/unet_flowers.log

Reads the program's stdout and writes, under `results/diffusion-unet/`:

  run.json        the numbers, one object (losses, stats, draws as bytes)
  loss.svg        the loss curve
  samples.svg     the 16 draws, 4x4, each 64x64 (one embedded PNG)
  stats.svg       the draws' mean and std against the export's

and prints the same numbers, so a run can be quoted without the files. The data
is `flowers64-trainfull-gray.u8`: the 7,067 training flowers at 64x64, one grayscale byte a pixel.
"""
import json
import os
import re
import struct
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("OUT") or os.path.join(ROOT, "results", "diffusion-unet", "trained")
DATA = os.path.join(os.path.expanduser("~"), ".cache", "bend-lab", "flowers",
                    "flowers64-trainfull-gray.u8")
HW, RES = 4096, 64


def parse(text):
    def one(key, cast=float):
        m = re.search(rf"^{key}: ([0-9.eE+-]+)$", text, re.M)
        return cast(m.group(1)) if m else None

    conf = {k: one(k, int) for k in ("images", "channels", "steps", "euler", "params")}
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", text, re.M)]
    final = one("final_loss")
    draws = []
    for line in re.findall(r"^sample: (.*)$", text, re.M):
        vals = np.array([float(v) for v in line.split()], dtype=np.float32)
        draws.extend(vals.reshape(-1, RES, RES))
    if not losses or not draws:
        raise SystemExit("no loss lines or no sample lines: is the run finished?")
    return conf, losses, final, np.array(draws)


def export_stats():
    if not os.path.exists(DATA):
        return None
    b = np.fromfile(DATA, dtype=np.uint8).astype(np.float32) / 255.0
    return float(b.mean()), float(b.std())


def png(gray):
    """a minimal 8-bit grayscale PNG, no dependencies."""
    h, w = gray.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in np.ascontiguousarray(gray, dtype=np.uint8))
    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def grid(draws, per_row=4, gap=6):
    """the draws as one grayscale image, 4x4, in [0, 255]."""
    n = draws.shape[0]
    rows = (n + per_row - 1) // per_row
    w = per_row * RES + (per_row + 1) * gap
    h = rows * RES + (rows + 1) * gap
    img = np.full((h, w), 16, dtype=np.uint8)
    for k in range(n):
        r, c = divmod(k, per_row)
        y, x = gap + r * (RES + gap), gap + c * (RES + gap)
        img[y:y + RES, x:x + RES] = np.clip(draws[k] * 255.0, 0, 255).astype(np.uint8)
    return img


def loss_svg(path, losses, final):
    w, h, left, top, bottom = 640, 220, 60, 20, 40
    pw, ph = w - left - 20, h - top - bottom
    pts = losses + ([final] if final is not None else [])
    lo, hi = min(pts), max(pts)
    hi = hi if hi > lo else lo + 1.0
    p = []
    for i, v in enumerate(pts):
        x = left + pw * i / max(1, len(pts) - 1)
        y = top + ph * (1.0 - (v - lo) / (hi - lo))
        p.append(f"{x:.1f},{y:.1f}")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="#101418"/>',
           f'<polyline fill="none" stroke="#5ee6c8" stroke-width="2" points="{" ".join(p)}"/>',
           f'<text x="{left}" y="{top - 6}" fill="#98a2b3" font-family="monospace" font-size="12">'
           f'loss {pts[0]:.4f} -&gt; {pts[-1]:.4f} (min {min(pts):.4f}), {len(pts)} points</text>',
           f'<text x="{left}" y="{h - 14}" fill="#98a2b3" font-family="monospace" font-size="11">step 0</text>',
           f'<text x="{w - 120}" y="{h - 14}" fill="#98a2b3" font-family="monospace" font-size="11">'
           f'step {len(pts) - 1}</text>', "</svg>"]
    open(path, "w").write("\n".join(out) + "\n")


def samples_svg(path, draws):
    img = grid(draws)
    import base64
    b64 = base64.b64encode(png(img)).decode()
    w, h = img.shape[1], img.shape[0]
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" fill="#101418"/>'
            f'<image x="0" y="0" width="{w}" height="{h}" href="data:image/png;base64,{b64}"/>')
    open(path, "w").write(head + "</svg>\n")


def stats_svg(path, data_mean, data_std, sample_mean, sample_std):
    w, h, left, top, rowh = 560, 170, 120, 30, 46
    rows = [("mean", data_mean, sample_mean, "#5b8cff", "#5ee6c8"),
            ("std", data_std, sample_std, "#5b8cff", "#5ee6c8")]
    hi = max(abs(v) for _, a, b, _, _ in rows for v in (a, b)) or 1.0
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         f'<rect width="{w}" height="{h}" fill="#101418"/>']
    for i, (label, a, b, ca, cb) in enumerate(rows):
        y = top + i * rowh
        for j, (v, tone, name) in enumerate(((a, ca, "data"), (b, cb, "drawn"))):
            yy = y + j * 14
            bw = (w - left - 90) * abs(v) / hi
            o.append(f'<rect x="{left}" y="{yy}" width="{bw:.1f}" height="10" fill="{tone}" opacity="0.9"/>')
            o.append(f'<text x="{left + bw + 6}" y="{yy + 9}" fill="#e6edf3" font-family="monospace" '
                     f'font-size="10">{v:.4f}</text>')
            o.append(f'<text x="{left - 8}" y="{yy + 9}" fill="#98a2b3" text-anchor="end" '
                     f'font-family="monospace" font-size="10">{label} {name}</text>')
    o.append("</svg>")
    open(path, "w").write("\n".join(o) + "\n")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        raise SystemExit("usage: unet_flowers_results.py <log>")
    text = open(path).read()
    conf, losses, final, draws = parse(text)
    ex = export_stats()
    sm, ss = float(draws.mean()), float(draws.std())
    os.makedirs(OUT, exist_ok=True)
    record = {
        "task": "unet-flowers",
        "program": "examples/diffusion/unet/train.bend",
        "images": conf["images"], "channels": conf["channels"],
        "steps": conf["steps"], "euler": conf["euler"], "params": conf["params"],
        "first_loss": losses[0], "final_loss": final, "losses": losses,
        "draws": len(draws), "sample_mean": sm, "sample_std": ss,
        "data_mean": ex[0] if ex else None, "data_std": ex[1] if ex else None,
        "draws_u8": [[int(v) for v in np.clip(d * 255.0, 0, 255).astype(np.uint8).reshape(-1)]
                     for d in draws],
    }
    json.dump(record, open(os.path.join(OUT, "run.json"), "w"))
    loss_svg(os.path.join(OUT, "loss.svg"), losses, final)
    samples_svg(os.path.join(OUT, "samples.svg"), draws)
    stats_svg(os.path.join(OUT, "stats.svg"), record["data_mean"] or 0.0, record["data_std"] or 0.0,
              sm, ss)
    print(json.dumps({k: record[k] for k in ("images", "channels", "steps", "euler", "params",
                                            "first_loss", "final_loss", "draws",
                                            "sample_mean", "sample_std", "data_mean", "data_std")},
                     indent=2))
    print(f"wrote {os.path.relpath(OUT, ROOT)}/{{run.json,loss.svg,samples.svg,stats.svg}}")


if __name__ == "__main__":
    main()
