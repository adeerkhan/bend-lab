#!/usr/bin/env bash
# scripts/mamba3_ab.sh — the Mamba-3 forward benchmark, current scan vs a revision.
#
#   scripts/mamba3_ab.sh [rev]
#
# rev defaults to 8792d35, the per-head array core. Both sides build the same
# bench_mamba3_fwd_n.bend; each binary runs five times, interleaved, so a
# machine-state drift hits both. Numbers from different sessions are not a delta.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REV="${1:-8792d35}"
BEND="${BEND_BIN:-$HOME/.bend/bin/bend}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/mamba3-ab.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/rev"
git show "$REV:examples/language/mamba3/mamba3_siso.bend" > "$TMP/rev/mamba3_siso.bend"
git show "$REV:examples/language/mamba3/bench_mamba3_fwd_n.bend" > "$TMP/rev/bench_mamba3_fwd_n.bend"

"$BEND" -o "$TMP/rev.bin" "$TMP/rev/bench_mamba3_fwd_n.bend"
"$BEND" -o "$TMP/cur.bin" examples/language/mamba3/bench_mamba3_fwd_n.bend

echo "revision $REV against the working tree; 20 forwards at seqlen 8"
for i in 1 2 3 4 5; do
  /usr/bin/time -f "rev  $i  %e s wall  %U s user" "$TMP/rev.bin"
  /usr/bin/time -f "cur  $i  %e s wall  %U s user" "$TMP/cur.bin"
done
