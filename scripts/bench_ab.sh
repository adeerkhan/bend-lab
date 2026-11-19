#!/usr/bin/env bash
# scripts/bench_ab.sh — A/B two builds of the same bench, one protocol.
#
#   scripts/bench_ab.sh <a.bin> <b.bin> [rounds]
#
# Both binaries must be builds of bench/plumbing_mamba_tiny/main.bend (the
# gate in scripts/bench_mamba_tiny.sh prints one line for the current tree).
# The script refuses on a machine above MAX_LOAD (default 1.0) unless FORCE=1,
# checks that both print the same full_step_loss (otherwise this is not an A/B
# of one program), then alternates the two in shuffled order for `rounds`
# rounds and prints the medians of full_step_x3_ms, divided by three.
#
# It prints the row to paste into bench/history.md. Rows measured with
# FORCE=1 above the load ceiling are references, not idle measurements, and
# the load column says so.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

A=${1:?usage: scripts/bench_ab.sh <a.bin> <b.bin> [rounds]}
B=${2:?usage: scripts/bench_ab.sh <a.bin> <b.bin> [rounds]}
R=${3:-5}
MAX_LOAD=${MAX_LOAD:-1.0}
FORCE=${FORCE:-0}

for f in "$A" "$B"; do
  [ -x "$f" ] || { echo "not executable: $f"; exit 2; }
done

load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
if [ "$FORCE" != 1 ]; then
  if ! awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN{exit !(l <= m)}'; then
    echo "refusing: load average $load > MAX_LOAD=$MAX_LOAD (FORCE=1 runs anyway)"
    exit 2
  fi
fi

loss_a=$("$A" | sed -n 's/^full_step_loss: //p')
loss_b=$("$B" | sed -n 's/^full_step_loss: //p')
if [ -z "$loss_a" ] || [ -z "$loss_b" ]; then
  echo "a binary printed no full_step_loss (is it the plumbing bench?)"
  exit 2
fi
if [ "$loss_a" != "$loss_b" ]; then
  echo "not an A/B of one program: loss $loss_a vs $loss_b"
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

for _ in $(seq 1 "$R"); do
  printf 'a\nb\n' | shuf | while read -r w; do
    bin=$([ "$w" = a ] && echo "$A" || echo "$B")
    x=$("$bin" | sed -n 's/^full_step_x3_ms: //p')
    [ -n "$x" ] || { echo "no full_step_x3_ms from $bin"; exit 2; }
    echo "$x" >>"$tmp/$w"
  done || exit 2
done

med() { sort -n "$1" | awk '{a[NR]=$1} END {print (NR % 2) ? a[(NR+1)/2] : (a[NR/2] + a[NR/2+1]) / 2}'; }
ma=$(med "$tmp/a")
mb=$(med "$tmp/b")

na=$(basename "$A")
nb=$(basename "$B")
today=$(date +%F)
awk -v a="$ma" -v b="$mb" -v r="$R" -v ld="$load" -v la="$loss_a" -v na="$na" -v nb="$nb" -v d="$today" '
  BEGIN {
    printf "medians over %d rounds, ms per step (three steps divided by three):\n", r
    printf "  a: %.1f   b: %.1f   (loss %s both)\n", a / 3, b / 3, la
    if (a > 0 && b > 0) printf "  b / a: %.3f\n", b / a
    printf "history row:\n"
    printf "| %s | %s vs %s | %.1f | %.1f | %d | %s |\n", d, na, nb, a / 3, b / 3, r, ld
  }'
