#!/usr/bin/env bash
# Sample 64 Euler draws from a trained UNet checkpoint, then score them.
#
#   CKPT=.scratch/unet_ckpt make unet-sample
#
# This is inference only: --steps 0 restores the checkpoint, prints final_loss
# and draws, and never updates a weight. The draws go to
# results/diffusion-unet/sampled/run.json and the held-out gate reads that file.
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${CKPT:-.scratch/unet_ckpt}"
BIN="${UNET_BIN:-.scratch/unet_four_train.bin}"
LOG="${UNET_SAMPLE_LOG:-.scratch/unet_sample.log}"
OUT="${UNET_SAMPLE_OUT:-results/diffusion-unet/sampled}"

test -f "$CKPT" || { echo "no checkpoint: $CKPT" >&2; exit 1; }
test -x "$BIN" || { echo "no binary: $BIN (rebuild train.bend)" >&2; exit 1; }

export PATH="$HOME/.bend/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/llvm22${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BEND_LIB="$PWD/.bend/env"
export MLBEND_DATA="${MLBEND_DATA:-$HOME/.cache/bend-lab}"
export BEND_NO_TELEMETRY=1

echo "sampling from $CKPT with --steps 0; log $LOG"
"$BIN" --resume "$CKPT" --steps 0 > "$LOG" 2>&1

OUT="$OUT" reference/.venv/bin/python reference/unet_flowers_results.py "$LOG"
UNET_RUN="$OUT/run.json" reference/.venv/bin/python reference/test_unet_flowers_metric.py
