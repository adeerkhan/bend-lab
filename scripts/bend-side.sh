#!/usr/bin/env bash
# scripts/bend-side.sh — install a Bend release next to the pin, not over it.
#
#   scripts/bend-side.sh <version> [sha256]
#
# Downloads bend-<version>-linux-x64.tar.gz and installs bin/bend, bend2/ and
# guide/ together at ~/.bend-side/<version>/.bend/, with a wrapper at
# ~/.bend-side/bend-<version>. The compiler finds its bend2/ and guide/ next
# to its own bin/bend, so the wrapper needs no HOME change and the real HOME's
# clang and Lean stay usable. The pin's ~/.bend is never touched. Verifies the version and runs scripts/bend_smoke.sh through
# the wrapper, so a partial install (the failure mode of `bend update`) is
# caught here rather than in a gate.
#
# The sha256 is printed; pass it back on a second run to verify the download.
set -euo pipefail

V=${1:?usage: scripts/bend-side.sh <version> [sha256]}
WANT_SHA=${2:-}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIDE=$HOME/.bend-side/$V
WRAP=$HOME/.bend-side/bend-$V

if [ -e "$SIDE" ]; then
  echo "already installed: $SIDE" >&2
  echo "$WRAP"
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

url="https://github.com/bendlang/bend/releases/download/v$V/bend-$V-linux-x64.tar.gz"
echo "downloading $url" >&2
curl --proto '=https' --tlsv1.2 -fsSL -o "$tmp/bend.tar.gz" "$url"
got=$(sha256sum "$tmp/bend.tar.gz" | cut -d' ' -f1)
echo "sha256=$got" >&2
if [ -n "$WANT_SHA" ] && [ "$got" != "$WANT_SHA" ]; then
  echo "wanted sha256=$WANT_SHA" >&2
  exit 1
fi

tar -xzf "$tmp/bend.tar.gz" -C "$tmp"
[ -d "$tmp/bend/bend2" ] && [ -d "$tmp/bend/guide" ] && [ -f "$tmp/bend/bin/bend" ] || {
  echo "unexpected archive layout: no bend/bin/bend, bend/bend2, bend/guide" >&2
  exit 1
}

mkdir -p "$SIDE/.bend/bin"
mv "$tmp/bend/bend2" "$tmp/bend/guide" "$SIDE/.bend/"
mv "$tmp/bend/bin/bend" "$SIDE/.bend/bin/bend"
printf '#!/bin/sh\nexec %s/.bend/bin/bend "$@"\n' "$SIDE" > "$WRAP"
chmod +x "$WRAP"

gotv=$("$SIDE/.bend/bin/bend" version | awk '{print $NF}')
[ "$gotv" = "$V" ] || { echo "version says $gotv, wanted $V" >&2; exit 1; }

"$ROOT/scripts/bend_smoke.sh" "$WRAP"
echo "installed: Bend $V" >&2
echo "$WRAP"
