#!/usr/bin/env bash
# deploy/shared/brev-env.sh
# Universal Brev environment bootstrap for Cosmos Cookbook recipes.
#
# Fixes all 6 bugs documented in the April 2026 Brev GPU Sprint:
#   Bug 1: Wrong HOME — brev exec inherits local machine HOME
#   Bug 2: Empty /workspace/ — brev.yaml setup: block does not run on bare creates
#   Bug 3: Missing git-lfs — not present on bare Nebius/Hyperstack Ubuntu images
#   Bug 4: HF login before venv — huggingface-cli only exists inside the uv venv
#   Bug 5: bare pip blocked (PEP 668) — Ubuntu 24.04 blocks pip install in uv venv
#   Bug 6: uv missing after HOME fix — uv installed to wrong HOME path
#
# NOTE: These are documented workarounds for Brev behavior observed in April 2026.
#       If Brev changes its internal behavior, these workarounds may need revision.
#
# Usage (in your demo.sh):
#   export COSMOS_REPO_URL="https://github.com/nvidia-cosmos/cosmos-reason2.git"
#   export COSMOS_DIR="cosmos-reason2"        # dir name under /workspace or $HOME
#   export COSMOS_UV_EXTRA="cu128"            # passed to: uv sync --extra <value>
#   export COSMOS_EXTRA_DEPS="fiftyone"       # installed via: uv pip install <value>
#   source "$COOKBOOK/deploy/shared/brev-env.sh"
#
# After sourcing, the following are available:
#   BREV_HOME         — verified remote HOME (/home/ubuntu, /home/shadeform, etc.)
#   BREV_COSMOS_DIR   — full path to the cloned model repo
#   python / uv / huggingface-cli — all on PATH from the active venv
#
# Optional env vars (set before sourcing):
#   COSMOS_NEED_FFMPEG=1   — install ffmpeg (required for Transfer1, Transfer2.5 recipes)
#   BREV_COSMOS_DIR        — pre-set to force a specific repo path
#   COSMOS_UV_EXTRA        — omit for repos without CUDA extras (falls back to plain uv sync)
#   COSMOS_EXTRA_DEPS      — space-separated list: "fiftyone opencv-python"

set -euo pipefail

# ---------------------------------------------------------------------------
# Bug 1 fix: Detect the real HOME for the remote user, not the local machine's.
# brev exec inherits the local HOME which may be /root or a macOS path — both
# wrong. Use `id -un` (more portable than whoami) to find the actual user, then
# derive HOME from /etc/passwd.
# ---------------------------------------------------------------------------
ACTUAL_USER="$(id -un)"
ACTUAL_HOME="$(getent passwd "$ACTUAL_USER" | cut -d: -f6)"

if [[ "$HOME" != "$ACTUAL_HOME" ]]; then
    echo "[brev-env] Fixing HOME: was '$HOME', setting to '$ACTUAL_HOME'"
    export HOME="$ACTUAL_HOME"
fi
export BREV_HOME="$HOME"

# ---------------------------------------------------------------------------
# Bug 3 fix: Install git-lfs if absent. Must happen before any git clone that
# needs LFS objects. Bare Nebius and Hyperstack images ship without it.
# ---------------------------------------------------------------------------
if ! command -v git-lfs &>/dev/null; then
    echo "[brev-env] Installing git-lfs..."
    sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
    git lfs install --skip-repo
fi

# Install ffmpeg if the recipe needs video processing (Transfer1, Transfer2.5)
if [[ "${COSMOS_NEED_FFMPEG:-0}" == "1" ]]; then
    if ! command -v ffmpeg &>/dev/null; then
        echo "[brev-env] Installing ffmpeg..."
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    fi
fi

# ---------------------------------------------------------------------------
# Bug 6 fix: Install uv to the correct HOME. The installer writes to
# $HOME/.local/bin/ and sources $HOME/.local/bin/env. If HOME was wrong
# during install the binary goes to the wrong path. We've already fixed HOME
# above, so this install is safe.
#
# SECURITY NOTE: This uses curl | sh (supply chain risk). The installer is
# from astral.sh (the uv maintainer). If your environment requires hardened
# supply chain controls, replace this with: apt-get install uv, or install
# from a pinned release URL with checksum verification.
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[brev-env] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Source uv's env — handles the case where uv was just installed this session
if [[ -f "$HOME/.local/bin/env" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

# Verify uv is now available
if ! command -v uv &>/dev/null; then
    echo "[brev-env] ERROR: uv still not found after install. PATH=$PATH" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Bug 2 fix: Clone the model repo. brev.yaml setup: block does NOT run on bare
# `brev create`. /workspace/ exists but is empty. We try /workspace/ first
# (preferred on Brev), then fall back to $HOME.
# ---------------------------------------------------------------------------
: "${COSMOS_REPO_URL:?COSMOS_REPO_URL must be set before sourcing brev-env.sh}"
: "${COSMOS_DIR:?COSMOS_DIR must be set before sourcing brev-env.sh}"

if [[ -n "${BREV_COSMOS_DIR:-}" ]]; then
    # Caller pre-set the path — honor it
    REPO_ROOT="$BREV_COSMOS_DIR"
elif [[ -d "/workspace" ]] && mkdir -p "/workspace/.write-test" 2>/dev/null && rmdir "/workspace/.write-test" 2>/dev/null; then
    # Verify /workspace is actually writable (NFS mounts can pass -w but fail on write)
    REPO_ROOT="/workspace/$COSMOS_DIR"
else
    REPO_ROOT="$HOME/$COSMOS_DIR"
fi

if [[ ! -d "$REPO_ROOT/.git" ]]; then
    echo "[brev-env] Cloning $COSMOS_REPO_URL → $REPO_ROOT"
    git clone "$COSMOS_REPO_URL" "$REPO_ROOT"
    pushd "$REPO_ROOT" >/dev/null
    echo "[brev-env] Running git lfs pull..."
    git lfs pull
    # Verify LFS objects were actually downloaded (not just pointer stubs)
    LFS_OBJECTS=$(git lfs ls-files 2>/dev/null | wc -l)
    if [[ "$LFS_OBJECTS" -gt 0 ]]; then
        STUB_COUNT=$(git lfs ls-files --error-unmatch 2>&1 | grep -c "error" || true)
        if [[ "$STUB_COUNT" -gt 0 ]]; then
            echo "[brev-env] WARNING: $STUB_COUNT LFS objects may be stub pointers. Run: git lfs pull --include='*'" >&2
        fi
    fi
    popd >/dev/null
else
    echo "[brev-env] Repo already cloned at $REPO_ROOT"
fi

export BREV_COSMOS_DIR="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Bug 5 fix: Install Python dependencies via uv, not bare pip.
# Ubuntu 24.04 marks the system Python as externally-managed (PEP 668) and
# blocks bare `pip install` inside an activated uv venv. Always use `uv pip`.
# ---------------------------------------------------------------------------
pushd "$BREV_COSMOS_DIR" >/dev/null

if [[ -n "${COSMOS_UV_EXTRA:-}" ]]; then
    echo "[brev-env] Running: uv sync --extra $COSMOS_UV_EXTRA"
    uv sync --extra "$COSMOS_UV_EXTRA" || { echo "[brev-env] FATAL: uv sync failed. Check pyproject.toml and CUDA extras." >&2; exit 1; }
else
    echo "[brev-env] Running: uv sync"
    uv sync || { echo "[brev-env] FATAL: uv sync failed. Check pyproject.toml." >&2; exit 1; }
fi

# Activate the venv — must happen before HF auth (Bug 4 fix)
# shellcheck source=/dev/null
source .venv/bin/activate

popd >/dev/null

# Install extra deps via uv pip (never bare pip)
if [[ -n "${COSMOS_EXTRA_DEPS:-}" ]]; then
    echo "[brev-env] Installing extra deps: $COSMOS_EXTRA_DEPS"
    # shellcheck disable=SC2086
    uv pip install $COSMOS_EXTRA_DEPS
fi

# ---------------------------------------------------------------------------
# Bug 4 fix: Authenticate with HuggingFace AFTER the venv is active.
# huggingface-cli is installed inside the venv, not system-wide. Calling it
# before activation means the binary doesn't exist yet.
#
# SECURITY: Token is piped via stdin (not passed as a CLI arg) to avoid
# exposure in process listings and shell history.
# ---------------------------------------------------------------------------
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[brev-env] Logging in to HuggingFace..."
    echo "$HF_TOKEN" | huggingface-cli login --token-stdin
else
    echo "[brev-env] FATAL: HF_TOKEN is not set." >&2
    echo "[brev-env]   All Cosmos models require a HuggingFace token with gated model access." >&2
    echo "[brev-env]   Set it before running: export HF_TOKEN=hf_..." >&2
    exit 1
fi

echo "[brev-env] Environment ready."
echo "[brev-env]   BREV_HOME=$BREV_HOME"
echo "[brev-env]   BREV_COSMOS_DIR=$BREV_COSMOS_DIR"
echo "[brev-env]   Python=$(python --version 2>&1)"
echo "[brev-env]   uv=$(uv --version)"
