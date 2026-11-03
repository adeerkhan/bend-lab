"""The machine's measured ceilings, saved for the report (B3).

  reference/bench_peaks.py [--threads N]

Runs `bench/peaks/main.bend` three times, keeps each phase's median, and writes
`results/peaks-<machine>.json`.

Two rules this file exists to keep:

- **A sample has to beat the clock.** `IO.now()` counts whole milliseconds, so
  the probe repeats every phase `passes` times inside one timed region; the rate
  is `bytes x passes / total_ms`, and the record keeps both the total (what was
  measured) and the per-pass figure (what a stage is charged). A sample under
  `MIN_SAMPLE_MS` is recorded but flagged, because a 3 ms median is a 30% number.
- **A ceiling has a thread count.** The runtime's `--threads` is passed through
  and recorded; a stage measured at a different thread count is not comparable
  to this file, and `bounds.peaks_of` refuses the mismatch instead of scoring a
  floor from it.

The array rates are the ones a Bend stage can reach walking an array cell by
cell; they are not DRAM's raw peaks, and the file says so. The reducing read
pays an F32 add chain on top of the walk, so it is published as `reduce_gbs`;
a read-only stage is charged at the copy's per-byte rate. The compute ceiling is
`bench/matmul`'s measured rate at its shape, with its own thread count, and
launch latency is not measured: it needs a bang, and the report prints
`unprobed` rather than a number nobody measured.
"""
import argparse
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
MIN_SAMPLE_MS = 100          # a sample shorter than this is flagged, not trusted
LINE = re.compile(r"^([a-z_0-9]+): ([0-9.eE+-]+)$")


def rts(threads):
    """The runtime's options, the way the run ought to be labelled."""
    return [] if not threads else ["+RTS", "--threads", str(threads), "-RTS"]


def run_once(binary, threads=None):
    out = subprocess.run([binary] + rts(threads), capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the peaks probe did not run:\n" + out.stdout + out.stderr)
    values = {}
    for line in out.stdout.splitlines():
        match = LINE.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def rate(byte_count, ms):
    """GB/s from bytes and milliseconds; None when the sample is empty."""
    if not ms or ms <= 0:
        return None
    return round(byte_count / (ms * 1e6), 3)


def matmul_source():
    path = os.path.join(ROOT, "results/bench-matmul.json")
    if not os.path.isfile(path):
        return None, None, None
    with open(path) as f:
        record = json.load(f)
    return (record.get("bend", {}).get("gflops"),
            f"results/bench-matmul.json ({record.get('size')} {record.get('dtype')})",
            record.get("threads"))


def slug():
    return re.sub(r"[^A-Za-z0-9._-]+", "-", platform.platform()).strip("-")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=None,
                        help="the runtime's thread count (passed to the program and recorded)")
    args = parser.parse_args()

    binary = os.path.join(tempfile.mkdtemp(), "mlbend-peaks")
    build = subprocess.run([BEND, "-o", binary, PROGRAM], capture_output=True, text=True,
                           env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the peaks probe did not build:\n" + build.stdout + build.stderr)

    runs = [run_once(binary, args.threads) for _ in range(REPEATS)]
    median = {key: statistics.median(run[key] for run in runs) for key in sorted(runs[0])}
    bytes_per_pass = int(median["bytes"])
    passes = int(median["passes"])

    # what was timed, and the per-pass figure a stage is charged
    total_ms = {phase: median[f"{phase}_ms"] for phase in ("write", "read", "copy")}
    pass_ms = {phase: total_ms[phase] / passes for phase in total_ms}

    write_gbs = rate(bytes_per_pass, pass_ms["write"])
    copy_gbs = rate(2 * bytes_per_pass, pass_ms["copy"])
    # a read-only pass is charged at the copy's per-byte rate; the reducing walk
    # below is what a stage that accumulates pays, and it is not a bandwidth
    read_gbs = round(copy_gbs / 2, 3) if copy_gbs else None
    reduce_gbs = rate(bytes_per_pass, pass_ms["read"])

    flops, flops_source, flops_threads = matmul_source()
    checks = {
        "read_le_write": read_gbs is not None and write_gbs is not None and read_gbs <= write_gbs,
        "reduce_le_read": reduce_gbs is not None and read_gbs is not None and reduce_gbs <= read_gbs,
        "sample_ms": {phase: total_ms[phase] for phase in total_ms},
        "sample_enough": all(value >= MIN_SAMPLE_MS for value in total_ms.values()),
    }
    record = {
        "schema": "mlbend.peaks/2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "bend": subprocess.run([BEND, "version"], capture_output=True, text=True,
                               env=ENV, timeout=30).stdout.split()[-1],
        "threads": args.threads,
        "clock_ms": 1,
        "read_gbs": read_gbs,
        "write_gbs": write_gbs,
        "copy_gbs": copy_gbs,
        "reduce_gbs": reduce_gbs,
        "flops_gflops": flops,
        "flops_dtype": "f32" if flops is not None else None,
        "flops_threads": flops_threads,
        "launch_us": None,
        "source": {"program": "bench/peaks/main.bend", "bytes": bytes_per_pass,
                   "passes": passes, "repeats": REPEATS, "flops": flops_source},
        "total_ms": total_ms,
        "pass_ms": {phase: round(value, 3) for phase, value in pass_ms.items()},
        "checks": checks,
        "notes": [
            "read, write and copy are through Array.get/Array.set, not raw DRAM",
            "each phase is repeated inside one timed region: IO.now() counts whole ms",
            "copy counts a read and a write of every byte; read_gbs charges a read at the copy's per-byte rate",
            "reduce_gbs is a reducing walk (an F32 add chain), not a bandwidth",
            f"a sample under {MIN_SAMPLE_MS} ms is flagged in checks.sample_enough",
            "threads is the runtime option the probe ran with; a record made at another count is not comparable",
            "flops is bench/matmul's measured rate at its shape, with its thread count",
            "launch latency needs a bang: unprobed",
        ],
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"peaks-{slug()}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"peaks: {path}")
    print(f"  read  {read_gbs} GB/s (copy-charged)   write {write_gbs} GB/s   copy {copy_gbs} GB/s")
    print(f"  reduce {reduce_gbs} GB/s (an add chain, not a bandwidth)")
    print(f"  flops {flops} GFLOP/s ({flops_source or 'unprobed'}"
          f"{'' if flops_threads is None else f', {flops_threads} threads'})   launch unprobed")
    print(f"  threads {args.threads if args.threads else 'the runtime default'}"
          f"   passes {passes}   samples {json.dumps(checks['sample_ms'])}"
          f"   enough {checks['sample_enough']}")
    print(f"  checks: read<=write {checks['read_le_write']}, reduce<=read {checks['reduce_le_read']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
