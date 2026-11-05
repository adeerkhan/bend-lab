"""Turn the UNet probe's full planes into a record: numbers, a U diagram, the feature maps.

  reference/.venv/bin/python reference/unet_results.py

Runs `examples/diffusion/unet/main.bend --dump` and writes

  results/diffusion-unet/run.json       the numbers, one object
  results/diffusion-unet/level.svg      the two-level U with its shapes and operations
  results/diffusion-unet/features.svg   x, y, z and out, both channels, as heatmaps
  results/diffusion-unet/stats.svg      |value| per stage: mean, std, max

The weights are the example's deterministic pattern, so the pictures are of the *arithmetic*: a
block that computed the wrong thing shows it as a broken plane (the 64x64 input is a signed ramp,
x is 0.01*i with alternating sign, and the skips add it back).
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

import test_unet as twin

OUT = os.environ.get("OUT") or os.path.join(ROOT, "results", "diffusion-unet")
F = np.float32
SIZES = {"y": (32, 32), "z": (32, 32), "out": (64, 64)}


def parse(text):
    planes = {}
    for key in ("y_all", "z_all", "out_all"):
        m = re.search(rf"^{key}: (.*)$", text, re.M)
        if not m:
            raise SystemExit(f"no {key}: line: run the binary with --dump")
        planes[key[:-4]] = np.array([float(v) for v in m.group(1).split()], dtype=F)
    cells = {}
    for key in ("y", "z", "out"):
        m = re.search(rf"^{key}: (.*)$", text, re.M)
        cells[key] = [float(v) for v in m.group(1).split()] if m else []
    adj = re.search(r"^adjoint: (PASS|FAIL)$", text, re.M)
    return planes, cells, (adj.group(1) if adj else "missing")


def colour(v, lo, hi):
    """per-panel diverging scale: low -> blue, middle -> the page, high -> amber."""
    mid = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo), 1e-9)
    t = max(-1.0, min(1.0, (v - mid) / half))
    a = abs(t)
    base, tip = (0x10, 0x14, 0x18), ((0xff, 0xb4, 0x54) if t >= 0 else (0x5b, 0x8c, 0xff))
    r, g, b = (int(base[i] + (tip[i] - base[i]) * a) for i in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def features_svg(path, planes):
    cell, gap, label, colw = 3, 22, 16, 64 * 3
    x = twin.gen(64 * 64 * 2).reshape(64, 64, 2)     # the probe does not dump its input
    data = [("x", x), ("y", planes["y"].reshape(32, 32, 2)),
            ("z", planes["z"].reshape(32, 32, 2)), ("out", planes["out"].reshape(64, 64, 2))]
    w = 2 * colw + 3 * gap
    h = sum(a.shape[0] * cell + label + gap for _, a in data) + gap
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         f'<rect width="{w}" height="{h}" fill="#101418"/>']
    y0 = gap
    for name, arr in data:
        rows, cols = arr.shape[0], arr.shape[1]
        for ch in range(2):
            panel = arr[:, :, ch]
            x0 = gap + ch * (colw + gap)
            lo, hi = float(panel.min()), float(panel.max())
            o.append(f'<text x="{x0}" y="{y0 + 12}" fill="#98a2b3" font-family="monospace" '
                     f'font-size="11">{name}.{ch}  {rows}x{cols}  [{lo:.2f}, {hi:.2f}]</text>')
            for r in range(rows):
                for c in range(cols):
                    o.append(f'<rect x="{x0 + c * cell}" y="{y0 + label + r * cell}" '
                             f'width="{cell}" height="{cell}" '
                             f'fill="{colour(float(panel[r, c]), lo, hi)}"/>')
        y0 += rows * cell + label + gap
    o.append("</svg>")
    open(path, "w").write("\n".join(o) + "\n")


def level_svg(path, stats):
    w, h = 640, 240
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         f'<rect width="{w}" height="{h}" fill="#101418"/>',
         '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
         '<path d="M0,0 L8,3 L0,6 z" fill="#98a2b3"/></marker></defs>']
    bw, bh = 138, 44

    def box(x, y, title, sub, tone="#5ee6c8"):
        o.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="#1a2028" '
                 f'stroke="{tone}" stroke-width="1.5"/>')
        o.append(f'<text x="{x + bw // 2}" y="{y + 18}" fill="#e6edf3" font-family="monospace" '
                 f'font-size="11" text-anchor="middle">{title}</text>')
        o.append(f'<text x="{x + bw // 2}" y="{y + 34}" fill="#98a2b3" font-family="monospace" '
                 f'font-size="10" text-anchor="middle">{sub}</text>')

    def arrow(x1, y1, x2, y2, dash=""):
        o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#98a2b3" '
                 f'stroke-width="1.2" marker-end="url(#a)" {dash}/>')

    xs = [12, 170, 328, 486]
    box(xs[0], 30, "x", "64x64x2", "#5b8cff")
    box(xs[1], 30, "conv1 s2 + GN + SiLU", "32x32x2  (y)")
    box(xs[2], 30, "conv3 s2 + GN + SiLU", "16x16x2  (low)")
    box(xs[3], 30, "up2 (nearest)", "16 -> 32")
    for a, b in zip(xs, xs[1:]):
        arrow(a + bw, 52, b, 52)
    box(xs[3], 140, "conv4 + GN + SiLU", "32x32x2  (z)")
    box(xs[2], 140, "up2 (nearest)", "32 -> 64")
    box(xs[1], 140, "conv2 + GN + SiLU", "64x64x2  (out)")
    box(xs[0], 140, "out = conv + x", "the second skip", "#ffb454")
    arrow(xs[3] + bw // 2, 74, xs[3] + bw // 2, 140)
    for a, b in ((xs[3], xs[2]), (xs[2], xs[1]), (xs[1], xs[0])):
        arrow(a, 162, b + bw, 162)

    o.append(f'<path d="M{xs[1] + bw // 2},30 C{xs[1] + bw // 2},4 {xs[3] + bw // 2},4 '
             f'{xs[3] + bw // 2},30" fill="none" stroke="#ffb454" stroke-width="1.2" '
             f'stroke-dasharray="4 3" marker-end="url(#a)"/>')
    o.append(f'<text x="{(xs[1] + xs[3]) // 2 + bw // 2}" y="12" fill="#ffb454" '
             f'font-family="monospace" font-size="10" text-anchor="middle">y skip</text>')
    o.append(f'<path d="M{xs[0] + 20},30 C{xs[0] + 20},100 {xs[1] + 20},100 {xs[1] + 20},140" '
             f'fill="none" stroke="#ffb454" stroke-width="1.2" stroke-dasharray="4 3" '
             f'marker-end="url(#a)"/>')
    o.append(f'<text x="{xs[0] + 26}" y="100" fill="#ffb454" font-family="monospace" '
             f'font-size="10">x skip</text>')

    o.append(f'<text x="12" y="200" fill="#98a2b3" font-family="monospace" font-size="10">'
             + "  |  ".join(f'{k} {v["min"]:+.2f}..{v["max"]:+.2f}' for k, v in stats.items())
             + "</text>")
    o.append("</svg>")
    open(path, "w").write("\n".join(o) + "\n")


def stats_svg(path, stats):
    names = list(stats)
    w, left, top, rowh = 520, 110, 24, 34
    series = [("mean |v|", "absmean", "#5ee6c8"), ("std", "std", "#5b8cff"),
              ("max |v|", "absmax", "#ffb454")]
    hi = max(stats[k][key] for k in names for _, key, _ in series) or 1.0
    h = top + len(names) * rowh + 24
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         f'<rect width="{w}" height="{h}" fill="#101418"/>']
    for gi, k in enumerate(names):
        base_y = top + gi * rowh
        for si, (_, key, tone) in enumerate(series):
            v = stats[k][key]
            y = base_y + si * 8
            bwid = (w - left - 96) * v / hi
            o.append(f'<rect x="{left}" y="{y}" width="{bwid:.1f}" height="6" fill="{tone}" opacity="0.9"/>')
            o.append(f'<text x="{left + bwid + 5}" y="{y + 7}" fill="#e6edf3" '
                     f'font-family="monospace" font-size="9">{v:.2f}</text>')
        o.append(f'<text x="10" y="{base_y + 14}" fill="#e6edf3" font-family="monospace" '
                 f'font-size="11">{k}</text>')
    ly = h - 10
    for i, (label, _, tone) in enumerate(series):
        o.append(f'<rect x="{left + i * 110}" y="{ly - 8}" width="9" height="9" fill="{tone}"/>')
        o.append(f'<text x="{left + i * 110 + 14}" y="{ly}" fill="#98a2b3" '
                 f'font-family="monospace" font-size="10">{label}</text>')
    o.append("</svg>")
    open(path, "w").write("\n".join(o) + "\n")


def main():
    text = subprocess.run([twin.build(), "--dump"], capture_output=True, text=True,
                          env=twin.ENV, cwd=ROOT)
    if text.returncode != 0:
        raise SystemExit("the unet probe did not run:\n" + text.stdout[-2000:] + text.stderr[-2000:])
    planes, cells, adjoint = parse(text.stdout)
    stats = {}
    for k, a in planes.items():
        stats[k] = {"shape": list(SIZES[k]) + [2], "cells": int(a.size),
                    "mean": float(a.mean()), "std": float(a.std()),
                    "min": float(a.min()), "max": float(a.max()),
                    "absmean": float(np.abs(a).mean()), "absmax": float(np.abs(a).max())}
    os.makedirs(OUT, exist_ok=True)
    record = {"task": "unet-two-level", "program": "examples/diffusion/unet/main.bend",
              "levels": [{"stage": "x", "shape": [64, 64, 2]},
                         {"stage": "y", "shape": [32, 32, 2]},
                         {"stage": "low", "shape": [16, 16, 2]},
                         {"stage": "z", "shape": [32, 32, 2]},
                         {"stage": "out", "shape": [64, 64, 2]}],
              "adjoint": adjoint, "first8": cells, "planes": stats}
    json.dump(record, open(os.path.join(OUT, "run.json"), "w"), indent=2)
    features_svg(os.path.join(OUT, "features.svg"), planes)
    level_svg(os.path.join(OUT, "level.svg"), stats)
    stats_svg(os.path.join(OUT, "stats.svg"), stats)
    print(json.dumps({"adjoint": adjoint, "planes": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                                         for kk, vv in v.items()} for k, v in stats.items()}},
                     indent=2))
    print(f"wrote {os.path.relpath(OUT, ROOT)}/{{run.json,level.svg,features.svg,stats.svg}}")


if __name__ == "__main__":
    main()
