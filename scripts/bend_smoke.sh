#!/usr/bin/env bash
# scripts/bend_smoke.sh — is this Bend install whole?
#
#   scripts/bend_smoke.sh [bend]
#
# Compiles and runs a three-line program that calls F32.from_nat. This is the
# check that caught a 2.0.40 base.bend left under a 2.0.38 binary: every live
# F32.from_nat call failed with "a live call into the law", while `bend
# version` still said 2.0.38. A version string is not an install check.
set -euo pipefail

BEND=${1:-${BEND_BIN:-$HOME/.bend/bin/bend}}
[ -x "$BEND" ] || { echo "no bend at $BEND"; exit 2; }

export LD_LIBRARY_PATH="$HOME/.local/lib/llvm22${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BEND_NO_TELEMETRY=1

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/smoke.bend" <<'EOF'
import Base

def pick(n: Nat) -> F32:
  F32.from_nat(n)

def main() -> IO(Unit):
  do IO<Unit>:
    x : F32 <- IO.pure(F32, pick(3n))
    Unit <- IO.print(F32.show(x))
    return Unit{}
EOF

"$BEND" "$tmp/smoke.bend" -o "$tmp/smoke" >/dev/null
out=$("$tmp/smoke")
if [ "$out" != "3" ]; then
  echo "smoke test printed '$out', wanted 3: the install is not whole"
  exit 1
fi
echo "smoke ok: F32.from_nat prints $out"
