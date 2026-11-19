#!/usr/bin/env bash
# scripts/bend-latest.sh — is there a Bend release newer than the pin?
#
#   scripts/bend-latest.sh            report pinned and latest
#   scripts/bend-latest.sh --install  install the latest side by side and run
#                                     the model shape laws with it
#
# The pin is BEND_VERSION in the Makefile. This script never runs
# `bend update`: that replaces the shared ~/.bend in place, and a half-applied
# update (a new bend2/ under the old binary) is exactly what broke the pin
# once. scripts/bend-side.sh installs the whole release under a private HOME.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PIN=$(sed -n 's/^BEND_VERSION := //p' Makefile)
[ -n "$PIN" ] || { echo "no BEND_VERSION in the Makefile"; exit 2; }

LATEST=$(curl -fsSL https://api.github.com/repos/bendlang/bend/releases/latest 2>/dev/null |
  sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p' | head -1)
[ -n "$LATEST" ] || { echo "could not read the latest release from GitHub"; exit 2; }

echo "pinned=$PIN latest=$LATEST"
if [ "$PIN" = "$LATEST" ]; then
  exit 0
fi

if [ "${1:-}" != "--install" ]; then
  echo "newer than the pin; run scripts/bend-latest.sh --install to test it side by side"
  exit 0
fi

WRAP=$(scripts/bend-side.sh "$LATEST" | tail -1) || exit 1
echo "$WRAP"
echo "model shape laws with Bend $LATEST:"
if BEND_LIB="$ROOT/.bend/env" "$WRAP" examples/language/mamba3/mamba3_laws.bend --verdict; then
  echo "Bend $LATEST passes the model shape laws"
else
  echo "Bend $LATEST does not pass them; the pin stays at $PIN"
  exit 1
fi
