#!/usr/bin/env bash
# One gate for humans and CI: cheapest first, a timing per stage, and the same
# command in both places.
#
#   make validate                  # environment, the cheap gate, the full gate, the smoke
#   make validate QUICK=1          # environment and the cheap gate, for the edit loop
#   make validate LONG=1           # everything above, then the long runs
#   make validate ARGS=--quick     # the same as QUICK=1, spelled out
#
# A cheap breakage has to fail here in under a minute, so the proofs run before
# the programs that take minutes: `--fast` checks every package's laws and the
# frozen references and stops at the first failure. The full `make check` keeps
# running the numbers against NumPy; it is the same gate CI runs.
#
# Locally, wrap it so the builds run inside the memory cgroup:
#   BEND_MEM=9G scripts/bend-guard.sh make validate
# CI has no systemd-run and runs it unguarded on its own runner.
set -u

cd "$(dirname "$0")/.." || exit 2

QUICK="${QUICK:-0}"
LONG="${LONG:-0}"
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --long) LONG=1 ;;
        *) echo "usage: validate.sh [--quick|--long]" >&2; exit 2 ;;
    esac
done

PY=reference/.venv/bin/python
FAILED=""
START_ALL=$SECONDS
LOG=/tmp/mlbend-validate.log
export BEND_NO_TELEMETRY=1

stage() {
    # stage <name> <command...>: one timed stage, its tail on failure
    local name="$1" start=$SECONDS
    shift
    printf 'stage %-12s ' "$name"
    if "$@" >"$LOG" 2>&1; then
        printf 'ok    %3ss\n' "$((SECONDS - start))"
        return 0
    fi
    printf 'FAIL  %3ss\n' "$((SECONDS - start))"
    tail -25 "$LOG" | sed 's/^/    /'
    FAILED="$FAILED $name"
    return 1
}

# 1. the lock and one example over it: seconds
stage env-check make env-check || { echo; echo "validate: FAILED:$FAILED"; exit 1; }

# 2. the cheap gate: the proofs, the frozen references, no @unsafe
stage check-fast "$PY" reference/check_all.py --fast || { echo; echo "validate: FAILED:$FAILED"; exit 1; }

if [ "$QUICK" -eq 1 ]; then
    printf '\nvalidate --quick: %ss, passed\n' "$((SECONDS - START_ALL))"
    exit 0
fi

# 3. the full gate: the same command CI runs
stage check make check || { echo; echo "validate: FAILED:$FAILED"; exit 1; }

# 4. a staged task end to end against the frozen reference: builds, runs, pages
stage smoke "$PY" reference/experiment.py --task vit-step --seeds 0 \
    --references frozen --no-latest --out results/raw \
    || { echo; echo "validate: FAILED:$FAILED"; exit 1; }

if [ "$LONG" -eq 1 ]; then
    # 5. the long runs: the corpus, the benchmarks, the plumbing split, flow64
    stage experiments make experiments || true
    stage bench make bench || true
    stage plumbing make plumbing || true
    stage flow64 make flow64 || true
fi

if [ -n "$FAILED" ]; then
    echo
    echo "validate: FAILED:$FAILED"
    exit 1
fi
printf '\nvalidate: %ss, every stage passed%s\n' "$((SECONDS - START_ALL))" \
    "$([ "$LONG" -eq 1 ] && echo ' (long)')"
