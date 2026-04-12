#!/bin/bash
# brev-env.sh — Universal Brev environment bootstrap for Cosmos recipes
#
# Source this file from any recipe demo.sh AFTER the GPU pre-flight check.
# It handles every environment difference between Brev providers (Nebius,
# Hyperstack, local) so recipe scripts can focus purely on inference.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#
#   # In your demo.sh:
#   export COSMOS_REPO_URL="https://github.com/nvidia-cosmos/cosmos-reason2.git"
#   export COSMOS_DIR="cosmos-reason2"           # directory name under workspace/$HOME
#   export COSMOS_UV_EXTRA="cu128"               # uv sync --extra <value>; omit for no extra
#   export COSMOS_EXTRA_DEPS="fiftyone"          # space-separated; installed via uv pip
#   export COSMOS_NEED_FFMPEG="1"               # set to "1" if recipe needs ffmpeg
#   source "$COOKBOOK/deploy/shared/brev-env.sh"
#
#   # brev-env.sh exports these for use in demo.sh:
#   BREV_HOME        — verified HOME for the instance user
#   BREV_COSMOS_DIR  — full path to the cloned model repo
#
# ── What it solves ────────────────────────────────────────────────────────────
#
#   HOME mismatch        brev exec passes the local machine's HOME. This script
#                        detects the actual remote HOME via whoami.
#
#   Missing git-lfs      Bare Nebius/Hyperstack instances don't have git-lfs.
#                        Installs before clone if absent.
#
#   Missing ffmpeg       Installed on demand when COSMOS_NEED_FFMPEG=1.
#
#   uv not in PATH       Installs uv and sources its env file so it's available
#                        immediately without a new shell.
#
#   Empty /workspace     brev create bare instances don't run brev.yaml setup.
#                        Falls back to $HOME/<COSMOS_DIR> transparently.
#
#   pip vs uv pip        uv venvs don't include pip. System pip is PEP 668-
#                        blocked on Ubuntu 24.04. All package installs use
#                        uv pip to target the active venv correctly.
#
#   HF auth ordering     huggingface-cli lives inside the uv venv. This script
#                        runs HF auth AFTER venv activation, never before.

set -e

# ── 1. Detect actual HOME ─────────────────────────────────────────────────────
# brev exec inherits the local caller's HOME. Detect the real remote HOME.

_brev_actual_home=$(eval echo "~$(whoami)")
if [ -z "$_brev_actual_home" ] || [ ! -d "$_brev_actual_home" ]; then
  _brev_actual_home="$HOME"
fi
export HOME="$_brev_actual_home"
export BREV_HOME="$HOME"

# ── 2. Validate required inputs ───────────────────────────────────────────────

if [ -z "$COSMOS_REPO_URL" ]; then
  echo "[brev-env] ERROR: COSMOS_REPO_URL must be set before sourcing brev-env.sh"
  exit 1
fi
if [ -z "$COSMOS_DIR" ]; then
  echo "[brev-env] ERROR: COSMOS_DIR must be set before sourcing brev-env.sh"
  exit 1
fi

# ── 3. Resolve COSMOS target directory (workspace-aware) ─────────────────────
# Priority: caller-provided BREV_COSMOS_DIR > /workspace/<dir> > $HOME/<dir>

if [ -n "$BREV_COSMOS_DIR" ] && [ -d "$BREV_COSMOS_DIR" ]; then
  : # caller pre-set a valid path
elif [ -d "/workspace/$COSMOS_DIR" ]; then
  BREV_COSMOS_DIR="/workspace/$COSMOS_DIR"
else
  BREV_COSMOS_DIR="$HOME/$COSMOS_DIR"
fi
export BREV_COSMOS_DIR

# ── 4. System dependencies ────────────────────────────────────────────────────

_need_apt=0
command -v git-lfs &>/dev/null  || _need_apt=1
[ "${COSMOS_NEED_FFMPEG:-0}" = "1" ] && { command -v ffmpeg &>/dev/null || _need_apt=1; }

if [ "$_need_apt" = "1" ]; then
  echo "[brev-env] Installing system dependencies..."
  sudo apt-get update -q
  _pkgs="git git-lfs"
  [ "${COSMOS_NEED_FFMPEG:-0}" = "1" ] && _pkgs="$_pkgs ffmpeg"
  sudo apt-get install -y -q $_pkgs
fi
git lfs install --skip-repo 2>/dev/null || true

# ── 5. uv ─────────────────────────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
  echo "[brev-env] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Source uv's env file so it's on PATH in this shell without a new login
source "$HOME/.local/bin/env" 2>/dev/null || true
if ! command -v uv &>/dev/null; then
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── 6. Clone or update model repo ────────────────────────────────────────────

if [ ! -d "$BREV_COSMOS_DIR" ]; then
  echo "[brev-env] Cloning $COSMOS_REPO_URL → $BREV_COSMOS_DIR"
  git clone "$COSMOS_REPO_URL" "$BREV_COSMOS_DIR"
  git -C "$BREV_COSMOS_DIR" lfs pull 2>/dev/null || true
fi

# ── 7. Python virtual environment ────────────────────────────────────────────

cd "$BREV_COSMOS_DIR"
if [ ! -d ".venv" ]; then
  echo "[brev-env] Creating Python environment..."
  if [ -n "$COSMOS_UV_EXTRA" ]; then
    uv sync --extra "$COSMOS_UV_EXTRA" 2>&1 | tail -5
  elif [ -f "pyproject.toml" ]; then
    uv sync 2>&1 | tail -5
  elif [ -f "requirements.txt" ]; then
    uv venv .venv
    uv pip install -r requirements.txt 2>&1 | tail -5
  fi
fi
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
echo "[brev-env] CUDA available ✓"

# ── 8. Recipe-specific extra packages (uv pip — NOT bare pip) ────────────────
# uv venvs do not ship pip. Bare 'pip install' falls back to system pip which
# is PEP 668-blocked on Ubuntu 24.04. Always use 'uv pip install'.

if [ -n "$COSMOS_EXTRA_DEPS" ]; then
  echo "[brev-env] Installing extra packages: $COSMOS_EXTRA_DEPS"
  uv pip install -q $COSMOS_EXTRA_DEPS
fi

# ── 9. HuggingFace auth ───────────────────────────────────────────────────────
# huggingface-cli lives in the venv — must run AFTER venv activation (step 7).

if [ -z "$HF_TOKEN" ]; then
  echo "HuggingFace token required. Enter your token (hf_...):"
  read -r -s HF_TOKEN
  export HF_TOKEN
fi
echo "$HF_TOKEN" | huggingface-cli login --token 2>/dev/null || \
  huggingface-cli login --token "$HF_TOKEN"

# ── Ready ─────────────────────────────────────────────────────────────────────

echo ""
echo "[brev-env] ✓ Environment ready"
echo "[brev-env]   Provider:   $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo unknown)"
echo "[brev-env]   User:       $(whoami)  HOME=$BREV_HOME"
echo "[brev-env]   COSMOS_DIR: $BREV_COSMOS_DIR"
echo "[brev-env]   Python:     $(python --version 2>&1)"
echo ""
