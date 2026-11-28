#!/usr/bin/env bash
# One-shot, idempotent setup for the MLBend checks. No sudo.
#
#   scripts/setup.sh
#
# Pins: Bend 2.0.38 (checked by SHA256) and Lean v4.34.0 (the kernel behind
# `bend --verdict`).
set -euo pipefail

BEND_VERSION=2.0.38
LEAN_VERSION=4.34.0
BENDENV_COMMIT=f750578f677c89689900ef29f07c2c2f67bf0ee3

cd "$(dirname "$0")/.."
export BEND_NO_TELEMETRY=1
HOME_BEND=${BEND_HOME:-$HOME/.bend}
export PATH="$HOME_BEND/bin:$HOME/.elan/bin:$PATH"
if [ -d "$HOME/.local/lib/llvm22" ]; then
  export LD_LIBRARY_PATH="$HOME/.local/lib/llvm22${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

say() { printf '\n== %s\n' "$*"; }
have_bend() { command -v bend >/dev/null 2>&1 && [ "$(bend version 2>/dev/null | awk '{print $NF}')" = "$BEND_VERSION" ]; }
have_lean() { command -v lean >/dev/null 2>&1 && lean --version 2>/dev/null | grep -q "version $LEAN_VERSION"; }
have_bendenv() { [ -x "$HOME_BEND/bin/bendenv" ] && [ "$(cat "$HOME_BEND/.bendenv-commit" 2>/dev/null)" = "$BENDENV_COMMIT" ] && "$HOME_BEND/bin/bendenv" --help >/dev/null 2>&1; }

# --- 1. Bend, pinned and verified -------------------------------------------------------
say "Bend $BEND_VERSION"
if have_bend; then
  echo "already installed"
else
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  OS=linux; ARCH=x64;   SHA=f19d088b079c9e1a8834ecd153b7d58c11a5f8f9ab1bdb78dbc96a6bf49aaa81 ;;
    Linux-aarch64) OS=linux; ARCH=arm64; SHA=1f7c6e74a6ef7412cf08eee0e4d45fe5960f89d66fdaca2466f6a4856f0d400a ;;
    *) echo "unsupported platform $(uname -s)-$(uname -m): Bend runs on Linux (use WSL on Windows)"; exit 1 ;;
  esac
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  NAME="bend-$BEND_VERSION-$OS-$ARCH.tar.gz"
  curl --proto '=https' --tlsv1.2 -fsSL -o "$TMP/$NAME" "https://github.com/bendlang/bend/releases/download/v$BEND_VERSION/$NAME"
  echo "$SHA  $TMP/$NAME" | sha256sum -c - >/dev/null || { echo "SHA256 mismatch for $NAME"; exit 1; }
  tar -xzf "$TMP/$NAME" -C "$TMP" 2>/dev/null
  mkdir -p "$HOME_BEND/bin"
  rm -rf "$HOME_BEND/bend2" "$HOME_BEND/guide"
  mv "$TMP/bend/bend2" "$TMP/bend/guide" "$HOME_BEND/"
  mv -f "$TMP/bend/bin/bend" "$HOME_BEND/bin/bend"
fi
have_bend || { echo "ERROR: bend $BEND_VERSION not found after install"; exit 1; }
bend version

# --- 2. bendenv (the lockfile and environment checker) ----------------------------
say "bendenv $BENDENV_COMMIT"
if have_bendenv; then
  echo "already installed"
else
  command -v cargo >/dev/null 2>&1 || { echo "ERROR: cargo is required to install bendenv"; exit 1; }
  cargo install --force --locked --git https://github.com/adeerkhan/bend-env.git \
    --rev "$BENDENV_COMMIT" bendenv --root "$HOME_BEND"
  printf '%s\n' "$BENDENV_COMMIT" > "$HOME_BEND/.bendenv-commit"
fi
have_bendenv || { echo "ERROR: bendenv $BENDENV_COMMIT not found after install"; exit 1; }
"$HOME_BEND/bin/bendenv" --help >/dev/null

say "Bend environment"
"$HOME_BEND/bin/bendenv" lock --check
"$HOME_BEND/bin/bendenv" sync --locked
"$HOME_BEND/bin/bendenv" check

# --- 3. Lean (the kernel behind --verdict) -----------------------------------------------
say "Lean $LEAN_VERSION"
if have_lean; then
  echo "already installed"
else
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --no-modify-path --default-toolchain "leanprover/lean4:v$LEAN_VERSION"
fi
have_lean || { echo "ERROR: lean $LEAN_VERSION not found after install"; exit 1; }
lean --version

# --- 4. Python environment --------------------------------------------------------------
say "Python environment"
[ -d reference/.venv ] || python3 -m venv reference/.venv
reference/.venv/bin/pip install -q --upgrade pip
reference/.venv/bin/pip install -q -r reference/requirements.txt
reference/.venv/bin/python -c "import numpy, sklearn; print('numpy', numpy.__version__, '| sklearn', sklearn.__version__)"

say "Oxford Flowers fixture"
reference/.venv/bin/python reference/flowers_export.py --setup

# --- 4b. the clean-runner fixtures the full check reads -------------------------------
say "Clean-runner fixtures"
mkdir -p /tmp/opencode .scratch
if [ ! -e reference/.venv-rf ]; then
  ln -s .venv reference/.venv-rf
  echo "linked reference/.venv-rf -> .venv"
fi
reference/.venv/bin/python reference/materialize_datasets.py

GLM_DATA=examples/language/glm53_flash/data
if [ ! -f "$GLM_DATA/corpus128-b32/corpus128.json" ] || \
   [ ! -f "$GLM_DATA/corpus128-holdout-reverse/corpus128.json" ]; then
  reference/.venv/bin/python reference/glm53_dense_data.py --prepare --directory "$GLM_DATA"
  reference/.venv/bin/python reference/glm53_dense_data.py --windows \
    --directory "$GLM_DATA/corpus128-b32" --examples 16000 --batch 32
  reference/.venv/bin/python reference/glm53_dense_data.py --windows \
    --directory "$GLM_DATA/corpus128-holdout-reverse" --examples 20000 --batch 32 \
    --holdout reverse
else
  echo "glm53 corpus128-b32 already present"
fi

if [ ! -f .scratch/mamba3_tiny/w_init/manifest.json ]; then
  reference/.venv/bin/python reference/mamba3_tiny_data.py
  reference/.venv-rf/bin/python reference/mamba3_tiny_twin.py --mode dump
else
  echo "mamba3 tiny twin dump already present"
fi

if [ ! -f .scratch/mamba_vit/manifest.json ]; then
  reference/.venv-rf/bin/python reference/mamba_vit_bins.py
else
  echo "mamba-vit dump already present"
fi

# --- 5. clang (Bend builds binaries with it) ----------------------------------------------
say "clang"
if command -v clang >/dev/null 2>&1; then
  clang --version | head -1
else
  echo "ERROR: clang is not on the path: Bend needs clang 14 or newer to build"
  echo "       binaries (the checks build every program before it runs it)."
  exit 1
fi
