"""A compute-only benchmark: bend-ml-tensor-array against NumPy.

  reference/.venv/bin/python reference/bench_matmul.py

The task reports in `results/` include the dataset fetch, so they compare a
whole program and not the kernel. This benchmark has neither: fifty
sequential 128 x 128 float32 matrix products, once in Bend over
`bend-ml-tensor-array` and once in NumPy. It writes `results/bench-matmul.json`
and `results/BENCH.md`.
"""
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

PROGRAM = os.path.join(ROOT, "bench/matmul/main.bend")
THREADS = None
SIZE = 128
PRODUCTS = 50
REPEATS = 5
FLOPS = 2 * SIZE ** 3 * PRODUCTS


def bend_time():
    """Build, then time the program; return the minimum run in seconds."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-matmul-bench")
    build = subprocess.run([BEND, "-o", binary, PROGRAM], capture_output=True, text=True,
                           env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the benchmark did not build:\n" + build.stdout + build.stderr)
    times, checksum = [], None
    rts = [] if not THREADS else ["+RTS", "--threads", str(THREADS), "-RTS"]
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = subprocess.run([binary] + rts, capture_output=True, text=True, env=ENV, cwd=ROOT)
        times.append(time.perf_counter() - start)
        if out.returncode != 0:
            raise SystemExit("the benchmark did not run:\n" + out.stdout + out.stderr)
        checksum = float(re.search(r"checksum: ([0-9.eE+-]+)", out.stdout).group(1))
    return min(times), times, checksum


def numpy_time():
    """Time the same products in NumPy; return the minimum run in seconds."""
    a = np.full((SIZE, SIZE), 1.0, dtype=np.float32)
    b = np.full((SIZE, SIZE), 2.0, dtype=np.float32)
    times, checksum = [], None
    for _ in range(REPEATS):
        start = time.perf_counter()
        c = a
        for _ in range(PRODUCTS):
            c = a @ b
        times.append(time.perf_counter() - start)
        checksum = float(c[0, 0])
    return min(times), times, checksum


def main():
    global THREADS
    if "--threads" in sys.argv:
        THREADS = int(sys.argv[sys.argv.index("--threads") + 1])
    bend_best, bend_all, bend_checksum = bend_time()
    numpy_best, numpy_all, numpy_checksum = numpy_time()
    if abs(bend_checksum - numpy_checksum) > 1e-3:
        raise SystemExit(f"the checksums differ: {bend_checksum} vs {numpy_checksum}")

    result = {
        "schema": "mlbend.benchmark/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": "matmul",
        "program": os.path.relpath(PROGRAM, ROOT),
        "size": [SIZE, SIZE],
        "products": PRODUCTS,
        "dtype": "F32",
        "repeats": REPEATS,
        "threads": THREADS,
        "checksum": bend_checksum,
        "bend": {
            "seconds_min": round(bend_best, 6),
            "seconds_all": [round(t, 6) for t in bend_all],
            "gflops": round(FLOPS / bend_best / 1e9, 3),
        },
        "numpy": {
            "seconds_min": round(numpy_best, 6),
            "seconds_all": [round(t, 6) for t in numpy_all],
            "gflops": round(FLOPS / numpy_best / 1e9, 3),
        },
        "speed_ratio": round(numpy_best / bend_best, 3),
        "machine": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "bench-matmul.json"), "w") as f:
        json.dump(result, f, indent=2)

    line = (f"| `{result['name']}` | {SIZE}x{SIZE} x{PRODUCTS} | "
            f"{result['bend']['seconds_min'] * 1000:.1f} ms | "
            f"{result['numpy']['seconds_min'] * 1000:.1f} ms | "
            f"{result['bend']['gflops']:.1f} | {result['numpy']['gflops']:.1f} | "
            f"{result['speed_ratio']:.2f}x |")
    text = "\n".join([
        "# Kernel benchmarks",
        "",
        "Compute only: no dataset, no IO. `make bench` writes this file.",
        "",
        f"Machine: {result['machine']['platform']} · {result['machine']['cpu_count']} CPUs · "
        f"NumPy {result['machine']['numpy']}",
        "",
        "| kernel | shape x rounds | Bend | NumPy | Bend GFLOP/s | NumPy GFLOP/s | speed (NumPy/Bend) |",
        "|---|---|---:|---:|---:|---:|---:|",
        line,
        "",
        "NumPy uses its BLAS and every thread it likes; the Bend product here is",
        "sequential (`par = 0`). The comparison is one thread of Bend against",
        "whatever NumPy chooses, which is the honest reading of the two",
        "programs as they are run.",
        "",
    ])
    with open(os.path.join(out_dir, "BENCH.md"), "w") as f:
        f.write(text)
    print(text)
    print("results/bench-matmul.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
