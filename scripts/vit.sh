#!/usr/bin/env bash
# scripts/vit.sh — the ViT pipeline, one stage per command.
#
#   ./scripts/vit.sh m1          build the example for STEPS (default 3000) and train
#   ./scripts/vit.sh m1-results  parse that log into results/diffusion-vit/
#   ./scripts/vit.sh gates       the frozen probe against the NumPy twin
#   ./scripts/vit.sh arms        torch-ViT and the lucidrains UNet
#   ./scripts/vit.sh compare     one JSON and one page from the three arms
#   ./scripts/vit.sh check       make check
#   ./scripts/vit.sh status      what finished, what is still running
#   ./scripts/vit.sh stop        stop what this script started
#
# Why it exists (2026-10-28): four training runs died to process management —
# a `timeout` on the tool call, SIGTERM when a foreground command was
# cancelled, and a wrong `pkill` — and every one of them lost 25-100 minutes
# because a run that is killed cannot be sampled. So each stage runs under
# setsid+nohup in its own session (a cancelled shell cannot reach it), writes
# $LOG/<stage>.log with `stdbuf -oL` so progress is visible immediately, and
# drops $LOG/<stage>.done with its exit code so nothing has to be guessed.
#
# STEPS is the example's `--steps` flag, so one binary serves every length:
# the stage builds `main.bend` once and starts it with `--steps $STEPS`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${VIT_LOG:-$ROOT/results/vit-logs}"
mkdir -p "$LOG"
export BEND_LIB="${BEND_LIB:-$ROOT/.bend/env}"
export BEND_NO_TELEMETRY=1
BEND="${BEND_BIN:-$HOME/.bend/bin/bend}"
PY="$ROOT/reference/.venv/bin/python"
TORCH_PY="$ROOT/reference/.venv-lucidrains/bin/python"
STEPS="${STEPS:-3000}"
THREADS="${THREADS:-3}"
# a second, shorter run alongside the headline one: STAGE=-q
STAGE="${STAGE:-}"

start() {           # start <stage> <command...>
  local stage=$1; shift
  local pidf="$LOG/$stage.pid"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "$stage: already running (pid $(cat "$pidf"))"; return 0
  fi
  rm -f "$LOG/$stage.done"
  setsid nohup bash -c "stdbuf -oL -eL $* > '$LOG/$stage.log' 2>&1; echo \$? > '$LOG/$stage.done'" \
    >/dev/null 2>&1 &
  echo $! > "$pidf"
  echo "$stage: started (pid $!), log $LOG/$stage.log"
}

finish() {          # finish <stage> -> its exit code, or 1 while it runs
  if [ ! -f "$LOG/$1.done" ]; then
    echo "$1: still running ($LOG/$1.log)"; return 1
  fi
  local rc; rc="$(cat "$LOG/$1.done")"
  echo "$1: exit $rc"
  return "$rc"
}

case "${1:-}" in
m1)
  "$BEND" "$ROOT/examples/diffusion/vit/main.bend" -o "$LOG/vit-m1$STAGE.bin" > "$LOG/m1$STAGE.build.log" 2>&1
  echo "built $LOG/vit-m1$STAGE.bin ($STEPS steps)"
  start "m1$STAGE" "'$LOG/vit-m1$STAGE.bin' --steps $STEPS"
  ;;
m1-results)
  finish "m1$STAGE" || exit 1
  OUT="${OUT:-$ROOT/results/diffusion-vit}" VIT_STEPS="$STEPS" \
    "$PY" "$ROOT/reference/vit_results.py" "$LOG/m1$STAGE.log"
  ;;
gates)
  start gates "'$PY' '$ROOT/reference/vit_gate.py'"
  ;;
arms)
  start torch-vit "env HEADS=1 POS=features CAUSAL=1 LR=3e-3 BATCH=64 STEPS=$STEPS THREADS=$THREADS '$TORCH_PY' '$ROOT/reference/digits_vit_probe.py' '$ROOT/results/compare/torch-vit.json'"
  start unet "env THREADS=$THREADS '$TORCH_PY' '$ROOT/reference/lucidrains_compare.py'"
  ;;
compare)
  finish torch-vit || true
  finish unet || true
  "$PY" "$ROOT/reference/compare_vit.py"
  ;;
check)
  start check "env BEND_BIN='$BEND' make -C '$ROOT' check"
  ;;
status)
  for f in "$LOG"/*.done; do
    [ -e "$f" ] && echo "$(basename "$f" .done): done, exit $(cat "$f")"
  done
  for f in "$LOG"/*.pid; do
    [ -e "$f" ] || continue
    p="$(cat "$f")"
    kill -0 "$p" 2>/dev/null && echo "$(basename "$f" .pid): running (pid $p)"
  done
  ;;
stop)
  for f in "$LOG"/*.pid; do
    [ -e "$f" ] || continue
    p="$(cat "$f")"
    kill "$p" 2>/dev/null && echo "stopped $(basename "$f" .pid) ($p)"
    rm -f "$f"
  done
  ;;
*)
  sed -n '2,20p' "$0"
  exit 1
  ;;
esac
