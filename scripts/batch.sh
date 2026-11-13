#!/usr/bin/env bash
# scripts/batch.sh — one matrix, one summary.
#
#   scripts/batch.sh                 # every row
#   ROWS="mamba3 check-fast" scripts/batch.sh
#
# Each row runs through `make session`, so it is in the dashboard with its
# command and system samples. A failed row is recorded and the batch goes on.
# The summary is `results/batch-<date>.md`; a missing sample count is
# "unavailable", never zero.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/llvm22${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BEND_NO_TELEMETRY=1
BEND="${BEND_BIN:-$HOME/.bend/bin/bend}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="results/batch-$STAMP.md"
LOGS="/tmp/opencode/batch-$STAMP"
mkdir -p "$LOGS" results

if ! pgrep -f "reference/dashboard.py" >/dev/null 2>&1; then
  ./scripts/dashboard.sh start >/dev/null 2>&1 || true
fi

echo "| row | command | status | seconds | samples | note |" > "$OUT"
echo "|---|---|---|---:|---|---|" >> "$OUT"

run() {
  local name="$1" cmd="$2" pass="${3:-}"
  if [ -n "${ROWS:-all}" ] && [ "${ROWS:-all}" != "all" ]; then
    case " $ROWS " in *" $name "*) ;; *) return 0 ;; esac
  fi
  local log="$LOGS/$name.log"
  local start end secs code samples note
  start="$(date +%s)"
  make session TASK="$name" CMD="$cmd" >"$log" 2>&1
  code=$?
  end="$(date +%s)"
  secs=$((end - start))
  samples="$(sqlite3 results/runs.sqlite3 \
    "select count(*) from samples where run_id=(select max(id) from runs where task='$name');" 2>/dev/null || true)"
  if [ -z "$samples" ] || [ "$samples" = "0" ]; then samples="unavailable"; fi
  note="-"
  if [ "$code" -ne 0 ]; then
    note="$(grep -m1 -E 'FAIL|Error|error:' "$log" 2>/dev/null | tr -d '|' | cut -c1-120)"
    [ -z "$note" ] && note="exit $code"
  elif [ -n "$pass" ] && ! grep -q "$pass" "$log"; then
    code=1
    note="missing '$pass'"
  fi
  if [ "$code" -eq 0 ]; then
    echo "| $name | \`$cmd\` | PASS | $secs | $samples | $note |" >> "$OUT"
  else
    echo "| $name | \`$cmd\` | FAIL | $secs | $samples | ${note:-exit $code} |" >> "$OUT"
  fi
  echo "$name: $([ "$code" -eq 0 ] && echo PASS || echo FAIL) (${secs}s)"
}

notrun() {
  local name="$1" reason="$2"
  if [ -n "${ROWS:-all}" ] && [ "${ROWS:-all}" != "all" ]; then
    case " $ROWS " in *" $name "*) ;; *) return 0 ;; esac
  fi
  echo "| $name | - | not run | - | - | $reason |" >> "$OUT"
  echo "$name: not run ($reason)"
}

# A. Mamba-3, existing gates
run mamba3 "make mamba3" "MAMBA3 PASS"
run mamba3-bench "make mamba3-bench" "bench mamba3:"
run n4n16 "$BEND examples/language/mamba3/probe_mamba3_n.bend" "y16:"
run gpu-probe "$BEND examples/language/mamba3/probe_bendtorch_gpu.bend" ""

# B. Mamba-3, the tests that remain (added as each gate lands)
# run mamba3-layer "make mamba3-layer" "MAMBA3 LAYER PASS"
notrun mamba3-step "placeholder gate exists but no step kernel; see .local/issues-hit.md ISSUE-127"
# run mamba3-varlen "make mamba3-varlen" "MAMBA3 VARLEN PASS"
# run mamba3-gqa "make mamba3-gqa" "MAMBA3 GQA PASS"
# run mamba3-outnorm "make mamba3-outnorm" "MAMBA3 OUTNORM PASS"
# run mamba3-deep "make mamba3-deep" "MAMBA3 DEEP PASS"

notrun mamba3-varlen "depends on the step path"
notrun mamba3-gqa "depends on the step path"
notrun mamba3-outnorm "depends on the step path"
notrun mamba3-deep "depends on the step path"
notrun mimo "needs its own oracle; after rows 1-6"
notrun tiny-lm "over the timebox; named corpus subset not defined"

# C. The other gates
run versions "make versions" "A review lane is one row"
run check-fast "make check-fast" "31/31 checks passed"
run full "make check" "141/141 checks passed"
run unet "make unet-metric" "UNET METRIC PASS"
run glm-decode "make glm-decode" "GLM DECODE PASS"
run glm-192 "make glm-h192-step" "wrote results/glm53-hybrid192-step/eval-"
run flow64 "make experiment TASK=flow64-first" "PASS"
run vit "make experiment TASK=vit-first" "PASS"
run glm53 "make experiment TASK=glm53-hybrid128" "PASS"
run classical "make experiment TASK=iris-crossval" "PASS"

echo "wrote $OUT"
