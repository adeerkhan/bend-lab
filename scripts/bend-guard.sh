#!/usr/bin/env bash
# Run a Bend command inside a memory cgroup, so a runaway compile is killed
# instead of taking the desktop with it.
#
#   scripts/bend-guard.sh bend -o out main.bend
#   scripts/bend-guard.sh bend -o out main.bend
#
# Why: `bend -o` builds the C or JS program in one pass and its memory grows
# with the model. On a 15 GB machine with a desktop, a build that wants 9 GB
# pushes the kernel into swap and freezes everything; nothing is OOM-killed
# early because the limit is the whole machine. A cgroup with MemorySwapMax=0
# makes the build the first thing to die, and the host stays up. Measured on
# this repository: model + data at 512 rows peaks at 4.3 GB, the training step
# pushes it past 8 GB, and the full example reached 9.3 GB before it was killed.
#
# Set BEND_MEM (default 4G) to the largest build you want to allow. Keep it below the free RAM on this box: a build that dies here is better than a frozen desktop. If
# systemd-run is missing, the command runs unguarded and says so.
set -u

MEM="${BEND_MEM:-4G}"

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "bend-guard: no systemd-run; running unguarded (a runaway build can freeze this machine)" >&2
  exec "$@"
fi

# --user keeps it out of root's slice; --scope ties the unit's life to this
# command; -q drops systemd's own chatter and leaves the program's output.
exec systemd-run --user --scope -q \
  -p MemoryMax="$MEM" -p MemorySwapMax=0 \
  -p MemoryAccounting=yes \
  --unit="bend-guard-$$" -- "$@"
