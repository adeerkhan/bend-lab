"""Time the Bend Mamba-3 SISO block beside the PyTorch twin and the Triton module.

  reference/.venv-rf/bin/python reference/mamba3_compare.py

Every row says its lane and its measurement boundary.  The Bend rows are
built C binaries (bend -o .c, gcc -O2), timed as one process from cold data,
so they include the Mat loads and the list conversion.  The torch rows are
warm in-process loops with CUDA events or perf_counter, so they exclude model
construction and imports.  The two boundaries are not the same; the report
keeps them apart instead of pretending they are one measurement.
"""
from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference/mamba3"))
import mamba3_twin as W  # noqa: E402

BEND = os.environ.get("BEND", "bend")
SCRATCH = os.path.join(ROOT, ".scratch/mamba3_compare")
GOLDEN = os.path.join(ROOT, "reference/mamba3_golden.npz")
DATA = os.path.join(ROOT, "reference/mamba3_data")
LM_GOLDEN = os.path.join(ROOT, "reference/mamba3_lm_golden.npz")
LM_DATA = os.path.join(ROOT, "reference/mamba3_lm_data")
PROGRAMS = ["bench_mamba3_fwd", "bench_mamba3_fwd_n", "bench_mamba3_bwd",
            "bench_mamba3_bwd_n", "probe_mamba3_train", "probe_mamba3_gen",
            "probe_mamba3_lm"]


def build():
    os.makedirs(SCRATCH, exist_ok=True)
    out = {}
    for name in PROGRAMS:
        src = os.path.join(ROOT, "examples/language/mamba3", name + ".bend")
        c = os.path.join(SCRATCH, name + ".c")
        exe = os.path.join(SCRATCH, name)
        subprocess.run([BEND, src, "-o", c], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        subprocess.run(["gcc", "-O2", c, "-lm", "-o", exe], check=True,
                       capture_output=True, text=True)
        out[name] = exe
    return out


def time_exe(exe, reps, parse_checksum=False):
    times = []
    last = ""
    for _ in range(reps):
        t = time.monotonic()
        p = subprocess.run([exe], cwd=ROOT, capture_output=True, text=True,
                           check=True)
        times.append(time.monotonic() - t)
        last = p.stdout.strip()
    return statistics.median(times), last


def peak_rss(exe):
    p = subprocess.run(["/usr/bin/time", "-f", "%M", exe], cwd=ROOT,
                       capture_output=True, text=True)
    return int(p.stderr.strip().splitlines()[-1]) / 1024.0


def make_module(device):
    torch.manual_seed(71)
    torch.set_num_threads(1)
    return W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32, ngroups=1,
                    is_mimo=False).to(device).float()


def load_inputs(device):
    z = np.load(GOLDEN)
    return (torch.tensor(z["u"], device=device),
            torch.tensor(z["dy"], device=device))


def bench_forward(device, reps=20):
    m = make_module(device)
    u, _ = load_inputs(device)
    for _ in range(5):
        W.twin_forward(m, u)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        W.twin_forward(m, u)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000.0


def bench_fwd_bwd(device, reps=20):
    m = make_module(device)
    u, dy = load_inputs(device)
    for _ in range(5):
        m.zero_grad(set_to_none=True)
        f = W.twin_forward(m, u)
        (f["y"] * dy).sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        m.zero_grad(set_to_none=True)
        f = W.twin_forward(m, u)
        (f["y"] * dy).sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000.0


def train(device, steps=10, lr=1e-3):
    torch.manual_seed(71)
    torch.set_num_threads(1)
    m = W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32, ngroups=1,
                 is_mimo=False).to(device).float()
    z = np.load(GOLDEN)
    E = torch.tensor(z["E0"], device=device).clone().requires_grad_(True)
    Wh = torch.tensor(z["WhT0"], device=device).clone().requires_grad_(True)
    targets = torch.arange(8, device=device)
    for _ in range(steps):
        f = W.twin_forward(m, E.unsqueeze(0))
        logits = f["y"].reshape(8, 64) @ Wh.T
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        with torch.no_grad():
            for p in m.parameters():
                p -= lr * p.grad
                p.grad = None
            E -= lr * E.grad
            Wh -= lr * Wh.grad
            E.grad = None
            Wh.grad = None
    return m, E, Wh


def generate(m, E, Wh, device, steps=12):
    toks = [0, 1, 2, 3]
    for _ in range(steps):
        u = E[torch.tensor(toks, device=device)].unsqueeze(0)
        y1 = W.twin_forward(m, u)["y"]
        y2 = W.twin_forward(m, y1)["y"]
        toks.append(int((y2[0, -1] @ Wh.T).argmax()))
    return toks


def time_training(device, reps=3):
    times = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        m, E, Wh = train(device)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times), m, E, Wh


def time_generation(device, reps=3):
    m, E, Wh = train(device)
    times = []
    toks = None
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        toks = generate(m, E, Wh, device)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times), toks


def module_forward(device, reps=20):
    m = make_module(device)
    u, _ = load_inputs(device)
    for _ in range(5):
        m(u)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        m(u)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000.0


def make_lm(device):
    torch.manual_seed(71)
    m1 = W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32, ngroups=1,
                  is_mimo=False).to(device).float()
    torch.manual_seed(72)
    m2 = W.Mamba3(d_model=64, d_state=16, expand=2, headdim=32, ngroups=1,
                  is_mimo=False).to(device).float()
    z = np.load(LM_GOLDEN)
    E = torch.tensor(z["E0"], device=device).clone().requires_grad_(True)
    Wh = torch.tensor(z["Wh0"], device=device).clone().requires_grad_(True)
    nw = torch.tensor(z["nw0"], device=device).clone().requires_grad_(True)
    return m1, m2, E, Wh, nw


def lm_forward_device(m1, m2, E, Wh, nw, tokens, device):
    u = E[torch.tensor(tokens, device=device)].unsqueeze(0)
    f1 = W.twin_forward(m1, u)
    h1 = u + f1["y"]
    f2 = W.twin_forward(m2, h1)
    h2 = h1 + f2["y"]
    hn = W.rms(h2, nw, eps=1e-5)
    return hn, hn @ Wh.T


def lm_train(device, steps=10, lr=1e-3):
    m1, m2, E, Wh, nw = make_lm(device)
    targets = torch.arange(8, device=device)
    for _ in range(steps):
        _, logits = lm_forward_device(m1, m2, E, Wh, nw, list(range(8)), device)
        loss = F.cross_entropy(logits.reshape(8, 8), targets)
        loss.backward()
        with torch.no_grad():
            for m in (m1, m2):
                for p in m.parameters():
                    p -= lr * p.grad
                    p.grad = None
            E -= lr * E.grad
            Wh -= lr * Wh.grad
            nw -= lr * nw.grad
            E.grad = None
            Wh.grad = None
            nw.grad = None
    return m1, m2, E.detach(), Wh.detach(), nw.detach()


def lm_generate(m1, m2, E, Wh, nw, device, steps=12):
    toks = [0, 1, 2, 3]
    for _ in range(steps):
        _, logits = lm_forward_device(m1, m2, E, Wh, nw, toks, device)
        toks.append(int(logits[0, -1].argmax()))
    return toks


def time_lm(device, reps=3):
    m1, m2, E, Wh, nw = make_lm(device)
    for _ in range(3):
        lm_forward_device(m1, m2, E, Wh, nw, list(range(8)), device)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        lm_forward_device(m1, m2, E, Wh, nw, list(range(8)), device)
    if device == "cuda":
        torch.cuda.synchronize()
    fwd = (time.perf_counter() - t) / 20 * 1000.0
    tt = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        mt = lm_train(device)
        if device == "cuda":
            torch.cuda.synchronize()
        tt.append(time.perf_counter() - t)
    tg = []
    toks = None
    for _ in range(reps):
        m1t, m2t, Et, Wht, nwt = mt
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        toks = lm_generate(m1t, m2t, Et, Wht, nwt, device)
        if device == "cuda":
            torch.cuda.synchronize()
        tg.append(time.perf_counter() - t)
    return fwd, statistics.median(tt), statistics.median(tg), toks


def main():
    if not os.path.exists(GOLDEN):
        raise SystemExit("run reference/mamba3_twin.py first")
    subprocess.run([sys.executable, os.path.join(ROOT, "reference/mamba3_bins.py"),
                    GOLDEN, DATA], check=True, capture_output=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "reference/mamba3_bins.py"),
                    LM_GOLDEN, LM_DATA], check=True, capture_output=True)
    print("building the Bend C binaries ...", flush=True)
    exes = build()

    t_fwd, c_fwd = time_exe(exes["bench_mamba3_fwd"], 5)
    t_fwd20, c_fwd20 = time_exe(exes["bench_mamba3_fwd_n"], 5)
    t_bwd, c_bwd = time_exe(exes["bench_mamba3_bwd"], 5)
    t_bwd10, c_bwd10 = time_exe(exes["bench_mamba3_bwd_n"], 3)
    t_train, c_train = time_exe(exes["probe_mamba3_train"], 3)
    t_gen, c_gen = time_exe(exes["probe_mamba3_gen"], 3)
    t_lm, c_lm = time_exe(exes["probe_mamba3_lm"], 1)
    rss_fwd = peak_rss(exes["bench_mamba3_fwd"])
    rss_bwd = peak_rss(exes["bench_mamba3_bwd"])
    rss_train = peak_rss(exes["probe_mamba3_train"])
    rss_gen = peak_rss(exes["probe_mamba3_gen"])
    rss_lm = peak_rss(exes["probe_mamba3_lm"])

    bend_fwd = t_fwd * 1000.0
    bend_fwd_slope = (t_fwd20 - t_fwd) / 19.0 * 1000.0
    bend_bwd = t_bwd * 1000.0
    bend_bwd_slope = (t_bwd10 - t_bwd) / 9.0 * 1000.0

    print("\nBend native C binaries, one process from cold data")
    print(f"forward one call            {bend_fwd:8.2f} ms   ({c_fwd})")
    print(f"forward slope over 20       {bend_fwd_slope:8.2f} ms/call   ({c_fwd20})")
    print(f"forward+backward one call   {bend_bwd:8.2f} ms   ({c_bwd})")
    print(f"fwd+bwd slope over 10       {bend_bwd_slope:8.2f} ms/call   ({c_bwd10})")
    print(f"ten-step copy task          {t_train:8.3f} s    ({c_train})")
    print(f"twelve generation steps     {t_gen:8.3f} s    ({c_gen})")
    print(f"peak RSS fwd {rss_fwd:.1f} MB, fwd+bwd {rss_bwd:.1f} MB, "
          f"train {rss_train:.1f} MB, gen {rss_gen:.1f} MB")
    print(f"full two-layer LM probe     {t_lm:8.3f} s    ({c_lm[:40]}...)")
    print(f"peak RSS full LM            {rss_lm:8.1f} MB")

    print("\nPyTorch twin, warm in-process, 20 reps")
    print(f"CPU forward                 {bench_forward('cpu'):8.3f} ms")
    print(f"CPU forward+backward        {bench_fwd_bwd('cpu'):8.3f} ms")
    print(f"CUDA forward                {bench_forward('cuda'):8.3f} ms")
    print(f"CUDA forward+backward       {bench_fwd_bwd('cuda'):8.3f} ms")
    print(f"CUDA public module bf16 fwd {module_forward('cuda'):8.3f} ms")

    torch.cuda.reset_peak_memory_stats()
    m = make_module("cuda")
    u, dy = load_inputs("cuda")
    f = W.twin_forward(m, u)
    (f["y"] * dy).sum().backward()
    torch.cuda.synchronize()
    print(f"CUDA peak allocated         {torch.cuda.max_memory_allocated()/1e6:8.3f} MB")
    print(f"CUDA peak reserved          {torch.cuda.max_memory_reserved()/1e6:8.3f} MB")

    tc, _, _, _ = time_training("cpu")
    tg, _, _, _ = time_training("cuda")
    gc, toks_c = time_generation("cpu")
    gg, toks_g = time_generation("cuda")
    print("\nPyTorch twin training and generation")
    print(f"CPU ten-step copy task      {tc:8.3f} s")
    print(f"CUDA ten-step copy task     {tg:8.3f} s")
    print(f"CPU twelve generation steps {gc:8.3f} s   {toks_c}")
    print(f"CUDA twelve generation steps {gg:8.3f} s   {toks_g}")

    lc = time_lm("cpu")
    lg = time_lm("cuda")
    print("\nPyTorch full two-layer LM (golden initial weights, ten SGD steps)")
    print(f"CPU forward                 {lc[0]:8.3f} ms")
    print(f"CPU ten-step + generation   {lc[1]:8.3f} s + {lc[2]:8.3f} s   {lc[3]}")
    print(f"CUDA forward                {lg[0]:8.3f} ms")
    print(f"CUDA ten-step + generation  {lg[1]:8.3f} s + {lg[2]:8.3f} s   {lg[3]}")

    params = sum(p.numel() for p in make_module("cpu").parameters())
    print(f"\nblock parameters {params}, plus 512 embedding and 512 head")
    print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
    cpu = ""
    with open("/proc/cpuinfo") as fh:
        for line in fh:
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    print(f"CPU {cpu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
