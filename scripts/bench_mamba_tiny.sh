#!/usr/bin/env bash
# scripts/bench_mamba_tiny.sh — one line for the tiny-LM step, and a gate.
#
#   scripts/bench_mamba_tiny.sh
#
# Builds bench/plumbing_mamba_tiny/main.bend, runs it, and prints
#
#   ms_per_step=10860.3 loss=4.838168 load=0.4 tokens=128 ok
#
# ms_per_step is the program's own clock: full_step_x3_ms / 3, three steps from
# the same weights. Loading and the build are outside it. The gate fails if any
# line of bench/expected-mamba-tiny.txt is missing from the output (the loss,
# the backward and scan heads, the parameter count): a change in what the model
# computes cannot pass as a timing.
#
# The harness refuses to measure when the one-minute load average is above
# MAX_LOAD (default 1.0), unless FORCE=1. A number measured over that ceiling is
# a reference, not a benchmark, and the row it writes must say so.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BEND=${BEND_BIN:-$HOME/.bend/bin/bend}
MAX_LOAD=${MAX_LOAD:-1.0}
FORCE=${FORCE:-0}
EXPECTED=bench/expected-mamba-tiny.txt

load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
if [ "$FORCE" != 1 ]; then
  if ! awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN{exit !(l <= m)}'; then
    echo "refusing: load average $load > MAX_LOAD=$MAX_LOAD (FORCE=1 runs anyway)"
    exit 2
  fi
fi

[ -f "$EXPECTED" ] || { echo "no $EXPECTED"; exit 2; }
[ -x "$BEND" ] || { echo "no bend at $BEND (BEND_BIN=)"; exit 2; }

export PATH="$HOME/.bend/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/llvm22${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BEND_LIB="$PWD/.bend/env"
export BEND_NO_TELEMETRY=1
export MLBEND_DATA="$PWD/reference/.cache/datasets"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if ! scripts/bend-guard.sh "$BEND" bench/plumbing_mamba_tiny/main.bend -o "$tmp/bench" >"$tmp/build.log" 2>&1; then
  cat "$tmp/build.log"
  echo "build failed"
  exit 2
fi
if ! "$tmp/bench" >"$tmp/out.txt" 2>&1; then
  cat "$tmp/out.txt"
  echo "run failed"
  exit 2
fi

fail=0
while IFS= read -r want; do
  case $want in
    ''|'#'*) continue ;;
  esac
  if ! grep -Fxq -- "$want" "$tmp/out.txt"; then
    echo "WRONG OUTPUT: missing '$want'"
    fail=1
  fi
done < "$EXPECTED"

x3=$(sed -n 's/^full_step_x3_ms: //p' "$tmp/out.txt")
loss=$(sed -n 's/^full_step_loss: //p' "$tmp/out.txt")
if [ -z "$x3" ] || [ -z "$loss" ]; then
  cat "$tmp/out.txt"
  echo "no full_step_x3_ms/full_step_loss"
  exit 2
fi
if [ "$fail" != 0 ]; then
  cat "$tmp/out.txt"
  exit 1
fi

awk -v x="$x3" -v l="$loss" -v ld="$load" \
  'BEGIN { printf "ms_per_step=%.1f loss=%s load=%s tokens=128 ok\n", x / 3, l, ld }'
