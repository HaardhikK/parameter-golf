#!/usr/bin/env bash
# setup.sh — one-command setup for Parameter Golf on a fresh Ubuntu 22.04/24.04 machine
# Works on RunPod containers (root, no sudo needed) and local Ubuntu installs.
# Usage: bash setup.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[setup] $*"; }

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
log "Installing system packages..."
apt-get update -qq
apt-get install -y -q \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates \
    gnupg

# ---------------------------------------------------------------------------
# 2. Node.js LTS (via NodeSource)
# ---------------------------------------------------------------------------
if ! command -v node &>/dev/null; then
    log "Installing Node.js LTS..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
    apt-get install -y -q nodejs
else
    log "Node.js already installed: $(node --version)"
fi

log "Node $(node --version)  |  npm $(npm --version)"

# ---------------------------------------------------------------------------
# 3. Claude Code CLI
# ---------------------------------------------------------------------------
if ! command -v claude &>/dev/null; then
    log "Installing Claude Code globally..."
    npm install -g @anthropic-ai/claude-code --silent
else
    log "Claude Code already installed: $(claude --version 2>/dev/null || echo 'present')"
fi

# ---------------------------------------------------------------------------
# 4. Python virtual environment + dependencies
# ---------------------------------------------------------------------------
cd "$REPO_DIR"

if [ ! -d ".venv" ]; then
    log "Creating Python virtual environment..."
    python3 -m venv .venv
fi

log "Activating venv and installing Python dependencies..."
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q

# On the official RunPod Parameter Golf template, PyTorch with CUDA is
# already installed system-wide. Skip reinstalling to avoid downgrades.
if python3 -c "import torch" 2>/dev/null; then
    log "torch already importable — installing remaining requirements (skipping torch)..."
    grep -v '^torch' requirements.txt | pip install -q -r /dev/stdin
else
    log "Installing all requirements (including torch)..."
    pip install -q -r requirements.txt
fi

# ---------------------------------------------------------------------------
# 5. zstandard — required for int6+zstd quantization
# ---------------------------------------------------------------------------
if ! python3 -c "import zstandard" 2>/dev/null; then
    log "Installing zstandard..."
    pip install -q zstandard
else
    log "zstandard already installed"
fi

# ---------------------------------------------------------------------------
# 6. Flash Attention (optional — Hopper H100 only, skip gracefully)
# ---------------------------------------------------------------------------
if python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    log "flash_attn_interface available — FA3 mode enabled for H100"
else
    log "flash_attn_interface not found — will use PyTorch SDPA fallback (fine for non-Hopper GPUs)"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log ""
log "=== Setup complete ==="
log ""
log "Next steps:"
log "  1. Activate the venv:  source .venv/bin/activate"
log "  2. Download data:      python3 data/cached_challenge_fineweb.py --variant sp1024"
log "  3. Check artifact:     SIZE_ONLY=1 python train_gpt.py"
log "  4. Run 8×H100:         See info_runpod.md for the full submission command"
log ""
log "See info_runpod.md for full instructions."
