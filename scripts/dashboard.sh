#!/usr/bin/env bash
# scripts/dashboard.sh — the run dashboard in the background.
#
#   ./scripts/dashboard.sh start     # http://127.0.0.1:8787/
#   ./scripts/dashboard.sh status
#   ./scripts/dashboard.sh stop
#
# PORT and DASH_LOG set the port and the log. The process runs under
# setsid+nohup, so a cancelled shell does not take it down.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8787}"
LOG="${DASH_LOG:-$ROOT/results/dashboard.log}"
PIDF="$ROOT/results/dashboard.pid"
PY="$ROOT/reference/.venv/bin/python"
URL="http://127.0.0.1:$PORT/"

alive() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case "${1:-start}" in
start)
  if alive; then
    echo "dashboard: already running (pid $(cat "$PIDF")) at $URL"
    exit 0
  fi
  mkdir -p "$(dirname "$LOG")"
  setsid nohup "$PY" "$ROOT/reference/dashboard.py" --port "$PORT" \
    >"$LOG" 2>&1 </dev/null &
  echo $! >"$PIDF"
  for _ in $(seq 1 20); do
    sleep 0.2
    if curl -fsS -o /dev/null "$URL" 2>/dev/null; then
      echo "dashboard: $URL (pid $(cat "$PIDF"), log $LOG)"
      exit 0
    fi
  done
  echo "dashboard: did not answer on $URL; see $LOG" >&2
  exit 1
  ;;
stop)
  if alive; then
    kill "$(cat "$PIDF")"
    echo "dashboard: stopped (pid $(cat "$PIDF"))"
  else
    echo "dashboard: not running"
  fi
  rm -f "$PIDF"
  ;;
status)
  if alive; then
    echo "dashboard: running (pid $(cat "$PIDF")) at $URL"
  else
    echo "dashboard: not running"
    exit 1
  fi
  ;;
*)
  echo "usage: $0 {start|stop|status}" >&2
  exit 2
  ;;
esac
