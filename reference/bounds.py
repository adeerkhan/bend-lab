"""The floors of a task's stages, and the score against the machine's peaks (B3).

  reference/bounds.py --task vit-step      # write results/bounds-<task>.json
  reference/bounds.py --list               # the task families that have a floor

A bound is what a stage must move and compute to run alone, at best: bytes read,
bytes written, and multiply-add FLOPs (two per multiply-add). It is a **lower**
bound, so a measured stage below it means the declaration is wrong; a stage
without a bound prints `unavailable`, never zero. The formulas are the ones in
the measurement plan's appendix C, and each function says what it excludes
(bias adds, softmax's exp, temporaries kept in registers).
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BYTES = 4  # F32


def linear(n, i, o):
    """y = x·Wᵀ + b: read x, the weights and the bias; write y; 2·n·i·o flops."""
    return {"read": (n * i + i * o + o) * BYTES, "write": n * o * BYTES, "flops": 2 * n * i * o}


def matmul(n, k, m):
    """C = A·B without a bias."""
    return {"read": (n * k + k * m) * BYTES, "write": n * m * BYTES, "flops": 2 * n * k * m}


def mlp(n, dims):
    """A chain of linears: `dims` is [i, h1, ..., o]; the sum of the parts."""
    total = {"read": 0, "write": 0, "flops": 0}
    for i, o in zip(dims, dims[1:]):
        part = linear(n, i, o)
        for key in total:
            total[key] += part[key]
    return total


def attention(b, t, h, hd, kv, g, causal=True):
    """Scores + softmax + P·V, one step: q, k, v in, o out.

    `g` query heads per key/value head. Excludes the softmax's exp and the
    masking; the score matrix is not counted as a temporary (a fused kernel
    keeps it on chip), so this is a lower bound.
    """
    read = (b * t * h * hd + 2 * b * t * kv * hd) * BYTES
    write = b * t * h * hd * BYTES
    rows = t * (t + 1) // 2 if causal else t * t
    flops = 2 * b * h * rows * hd * 2  # scores and P·V
    return {"read": read, "write": write, "flops": flops}


def conv2d(n, h, w, c, co, k, p=0, s=1):
    """im2col + linear, fused: read the image and the weights, write y."""
    oh = (h + 2 * p - k) // s + 1
    ow = (w + 2 * p - k) // s + 1
    rows = n * oh * ow
    ckk = c * k * k
    return {
        "read": (n * h * w * c + ckk * co) * BYTES,
        "write": rows * co * BYTES,
        "flops": 2 * rows * ckk * co,
    }


def conv2d_patch(n, h, w, c, co, k, p=0, s=1):
    """The patch matrix's round trip: what a materialised im2col adds, both ways."""
    oh = (h + 2 * p - k) // s + 1
    ow = (w + 2 * p - k) // s + 1
    rows = n * oh * ow
    return {"extra_bytes": 2 * rows * c * k * k * BYTES}


def knn(n_query, n_train, d, k):
    """Brute-force distances: every query against every training row, then k rows out."""
    return {
        "read": (n_query * n_train * d + n_query * k * d) * BYTES,
        "write": n_query * k * BYTES,
        "flops": n_query * n_train * d,
    }


def vit_step(params=100736, rows=1024, fe=20, px=4, tokens=16):
    """One ViT training step at the plumbing benchmark's sizes.

    Every weight meets every **row** once per forward, and a row is one token
    of one image: `rows` is 1024 (64 images x 16 tokens), not 16. The dense
    lower bound is 2·params·rows flops; the backward is counted at one more
    forward (4·params·rows including the forward), and the optimizer's own pass
    is not counted. The embedding rows a token selects are ignored (they are
    read once, not once per weight). Excludes attention's quadratic term (16
    tokens per image: small) and every temporary a kernel keeps.

    Counting `tokens` here instead of `rows` understated the floor by 64x and
    made the page read "0.5% of its floor" for a forward that is at a third of
    it; corrected 2026-11-03 (ISSUE-049).
    """
    return {
        "run/step/input": {"read": rows * fe * BYTES, "write": rows * fe * BYTES, "flops": 0},
        "run/step/forward": {
            "read": (params + rows * fe) * BYTES,
            "write": rows * px * BYTES,
            "flops": 2 * params * rows,
            "counts": "matmul flops only",
        },
        "run/step/loss_backward_opt": {
            "read": (params + rows * px) * BYTES,
            "write": params * BYTES,
            "flops": 4 * params * rows,
            "counts": "matmul flops only",
        },
    }


def cifar_step(params=807168, rows=256, fe=3080, px=3072):
    """One CIFAR MLP training step (examples/cifar_train).

    The model is not token-parallel: each of the 256 rows meets every weight
    once, so the dense lower bound is 2·params·rows flops for the forward and
    twice that for the backward. The three linears are 3080→128→128→3072 and
    `params` counts their weights and biases (807,168 from the probe). The
    optimizer's own pass is not counted, and neither are the input builder's
    temporaries or the loss's elementwise work.
    """
    return {
        "run/step/input": {"read": rows * fe * BYTES, "write": rows * fe * BYTES, "flops": 0},
        "run/step/forward": {
            "read": (params + rows * fe) * BYTES,
            "write": rows * px * BYTES,
            "flops": 2 * params * rows,
            "counts": "matmul flops only",
        },
        "run/step/target": {"read": rows * px * BYTES, "write": rows * px * BYTES, "flops": 0},
        "run/step/loss_backward_opt": {
            "read": (params + rows * px) * BYTES,
            "write": params * BYTES,
            "flops": 4 * params * rows,
            "counts": "matmul flops only",
        },
    }


def flow64_step(params=413440, rows=128, fe=80, px=64):
    """One 64x64 rectified-flow step (examples/diffusion/flowers_flow).

    `rows` is RW() = images per step x tokens per image (2 x 64 at the `corpus`
    preset), `fe` = the token width (64 pixels + 8 time + 8 position) and `px`
    the predicted pixels. The model is the ViT shape at 64 tokens: every row
    meets every weight once per forward, so the dense floor is 2*params*rows
    flops, doubled for the backward. `params` is the count the probe prints
    (413,440). The Euler loop's repeated forward is not counted here; the
    optimizer's own pass is not either.
    """
    return {
        "run/step/input": {"read": rows * fe * BYTES, "write": rows * fe * BYTES, "flops": 0},
        "run/step/forward": {
            "read": (params + rows * fe) * BYTES,
            "write": rows * px * BYTES,
            "flops": 2 * params * rows,
            "counts": "matmul flops only",
        },
        "run/step/loss_backward_opt": {
            "read": (params + rows * px) * BYTES,
            "write": params * BYTES,
            "flops": 4 * params * rows,
            "counts": "matmul flops only",
        },
    }


TASK_BOUNDS = {
    "vit-step": vit_step,
    "diffusion-vit": vit_step,
    "cifar-train": cifar_step,
    "flowers-flow": flow64_step,
}


def refusal(record, peaks):
    """Why this peaks block cannot score this record, or None when it can.

    A ceiling is only a ceiling for the machine, the Bend build and the thread
    count it was measured at. Comparing across any of those is a banner, never
    a floor: a stage run on one thread scored against a 16-thread peak is not
    0.5% of anything.
    """
    if not peaks:
        return "no peaks file for this machine"
    if peaks.get("schema") != "mlbend.peaks/2":
        return ("peaks measured with the old probe (one cold pass, no thread count): "
                "run `make peaks`")
    checks = peaks.get("checks") or {}
    if checks.get("sample_enough") is False:
        return "the peaks samples are under the clock's resolution"
    want_bend = (record.get("bend") or {}).get("version")
    got_bend = peaks.get("bend")
    if want_bend and got_bend and want_bend != got_bend:
        return f"peaks measured with bend {got_bend}, this record with {want_bend}"
    want_threads = (record.get("variant") or {}).get("threads")
    got_threads = peaks.get("threads")
    if want_threads is not None and got_threads is not None and want_threads != got_threads:
        return f"peaks measured at {got_threads} threads, this record at {want_threads}"
    return None


def peaks_and_reason(record):
    """(the peaks that may score this record, why not when they may not)."""
    own = record.get("peaks") or {}
    own_reason = refusal(record, own) if own else None
    if own and not own_reason:
        return own, None
    machine = machine_key(record.get("machine") or {})
    reason = own_reason or "no peaks file for this machine"
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "peaks-*.json"))):
        try:
            with open(path) as f:
                found = json.load(f)
        except (OSError, ValueError):
            continue
        if machine_key(found.get("machine") or {}) != machine:
            continue
        found = dict(found)
        found["source_file"] = os.path.relpath(path, ROOT)
        bad = refusal(record, found)
        if not bad:
            return found, None
        reason = bad
    return {}, reason


def peaks_of(record):
    """The record's peaks, or a file measured on the same machine, same build."""
    return peaks_and_reason(record)[0]


def machine_key(info):
    """The same device: architecture, processor and cores.

    The OS build is not part of it: a kernel update does not change the
    hardware, and the page names the peaks file it used so a reader can judge.
    """
    return (info.get("machine") or "", info.get("processor") or "", info.get("cpu_count"))


def sidecar(task):
    return os.path.join(ROOT, "results", f"bounds-{task}.json")


def bounds_for(record):
    """The bounds of a task: its sidecar (the current declaration) first.

    A bound is a formula over the task's shapes, not a measurement, so a fixed
    formula has to reach every page: the record keeps the copy it was made with
    as evidence, and the page names the sidecar it used. Falling back to the
    record's own copy would keep a corrected floor invisible until the task was
    re-run.
    """
    task = (record.get("task") or {}).get("name")
    if task and os.path.isfile(sidecar(task)):
        with open(sidecar(task)) as f:
            return json.load(f)
    return record.get("bounds") or {}


def floor_us(record, path):
    """The floor of one stage in microseconds, or None when it cannot be scored.

    Reads and writes are charged at their own measured ceilings (the copy
    ceiling is the fallback where a device reports none); flops at the measured
    arithmetic rate. A declared bound whose peak is missing scores nothing.
    """
    part = bounds_for(record).get(path)
    if not part:
        return None
    peaks = peaks_of(record)
    times = []
    if part.get("read"):
        rate = peaks.get("read_gbs") or peaks.get("copy_gbs")
        if not rate:
            return None
        times.append(part["read"] / (rate * 1e9) * 1e6)
    if part.get("write"):
        rate = peaks.get("write_gbs") or peaks.get("copy_gbs")
        if not rate:
            return None
        times.append(part["write"] / (rate * 1e9) * 1e6)
    if part.get("flops"):
        rate = peaks.get("flops_gflops")
        if not rate:
            return None
        times.append(part["flops"] / (rate * 1e9) * 1e6)
    return max(times) if times else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="write results/bounds-<task>.json from its floor function")
    parser.add_argument("--list", action="store_true", help="the tasks that have a floor")
    args = parser.parse_args()
    if args.list:
        for name in sorted(TASK_BOUNDS):
            print(name)
        return 0
    if not args.task:
        parser.error("give --task or --list")
    if args.task not in TASK_BOUNDS:
        raise SystemExit(f"no floor for {args.task}; known: {', '.join(sorted(TASK_BOUNDS))}")
    bounds = TASK_BOUNDS[args.task]()
    path = sidecar(args.task)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"bounds: {path}")
    for stage, part in bounds.items():
        print(f"  {stage:<32} read {part['read']:>10}  write {part['write']:>9}  flops {part['flops']:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
