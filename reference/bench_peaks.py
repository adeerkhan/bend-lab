"""The machine's measured ceilings, saved for the report (B3).

  reference/bench_peaks.py

Runs `bench/peaks/main.bend` three times, keeps each phase's median, and writes
`results/peaks-<machine>.json`. The array rates are the ones a Bend stage can
reach walking an array cell by cell; they are not DRAM's raw peaks, and the file
says so. The compute ceiling is `bench/matmul`'s measured rate, and launch
latency is not measured: it needs a bang, and the report prints `unprobed`
rather than a number nobody measured.
"""
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

PROGRAM = os.path.join(ROOT, "bench/peaks/main.bend")
REPEATS = 3
LINE = re.compile(r"^([a-z_0-9]+): ([0-9.eE+-]+)$")


def run_once(binary):
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the peaks probe did not run:\n" + out.stdout + out.stderr)
    values = {}
    for line in out.stdout.splitlines():
        match = LINE.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def rate(byte_count, ms):
    """GB/s from bytes and whole milliseconds; None below the clock's resolution."""
    if not ms or ms <= 0:
        return None
    return round(byte_count / (ms * 1e6), 3)


def matmul_flops():
    path = os.path.join(ROOT, "results/bench-matmul.json")
    if not os.path.isfile(path):
        return None, None
    with open(path) as f:
        record = json.load(f)
    return record.get("bend", {}).get("gflops"), f"results/bench-matmul.json ({record.get('size')} {record.get('dtype')})"


def slug():
    return re.sub(r"[^A-Za-z0-9._-]+", "-", platform.platform()).strip("-")


def main():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-peaks")
    build = subprocess.run([BEND, "-o", binary, PROGRAM], capture_output=True, text=True,
                           env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the peaks probe did not build:\n" + build.stdout + build.stderr)
    runs = [run_once(binary) for _ in range(REPEATS)]
    median = {key: statistics.median(run[key] for run in runs) for key in sorted(runs[0])}
    bytes_per_pass = int(median["bytes"])
    flops, flops_source = matmul_flops()
    record = {
        "schema": "mlbend.peaks/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "bend": subprocess.run([BEND, "version"], capture_output=True, text=True,
                               env=ENV, timeout=30).stdout.split()[-1],
        "read_gbs": rate(bytes_per_pass, median["read_ms"]),
        "write_gbs": rate(bytes_per_pass, median["write_ms"]),
        "copy_gbs": rate(2 * bytes_per_pass, median["copy_ms"]),
        "flops_gflops": flops,
        "flops_dtype": "f32" if flops is not None else None,
        "launch_us": None,
        "source": {"program": "bench/peaks/main.bend", "bytes": bytes_per_pass,
                   "repeats": REPEATS, "flops": flops_source},
        "median_ms": {key: value for key, value in median.items() if key != "bytes"},
        "notes": [
            "read, write and copy are through Array.get/Array.set, not raw DRAM",
            "copy counts a read and a write of every byte",
            "flops is bench/matmul's measured rate at its shape",
            "launch latency needs a bang: unprobed",
        ],
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"peaks-{slug()}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"peaks: {path}")
    print(f"  read  {record['read_gbs']} GB/s   write {record['write_gbs']} GB/s   copy {record['copy_gbs']} GB/s")
    print(f"  flops {record['flops_gflops']} GFLOP/s ({flops_source or 'unprobed'})   launch unprobed")
    print(f"  medians: " + ", ".join(f"{k}={v:.0f}" for k, v in record["median_ms"].items() if k.endswith("_ms")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
