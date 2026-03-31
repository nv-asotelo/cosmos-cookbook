#!/usr/bin/env bash
# =============================================================================
# Cosmos Reason2 — Worker Safety Inference Launchable
# =============================================================================
# Zero-to-FiftyOne for a brand new Brev user. One command:
#
#   export HF_TOKEN=hf_your_token_here
#   brev org set asotelo-test-org
#   brev create worker-safety \
#     --type massedcompute_A100_sxm4_80G_DGX \
#     --startup-script @launch.sh
#
#   Then locally: brev port-forward worker-safety -p 5151:5151
#   Open browser: http://localhost:5151
#
# Prerequisites (one-time, on your local machine):
#   1. brev login                            (browser OAuth)
#   2. Accept NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-2B
#   3. export HF_TOKEN=hf_...               (from huggingface.co/settings/tokens)
#
# Cost: ~$1.49/hr (massedcompute A100 80GB). Total runtime: ~2 hours.
# =============================================================================

set -uo pipefail   # note: NOT -e, so non-critical steps don't kill the run

LOG="$HOME/cosmos-setup.log"
VENV="$HOME/cosmos-reason2/.venv"
PYTHON="$VENV/bin/python3"
export PATH="$HOME/.local/bin:$PATH"

# HF_TOKEN: read from environment (set before brev create) or first arg
HF_TOKEN="${HF_TOKEN:-${1:-}}"
if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN not set. Re-run with: export HF_TOKEN=hf_... then brev create ..."
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u)] === Cosmos Reason2 Worker Safety Launchable ==="
echo "[$(date -u)] Instance: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# Step 1: System dependencies
# ---------------------------------------------------------------------------
echo "[$(date -u)] === Step 1: System dependencies ==="
sudo apt-get update -y
sudo apt-get install -y curl ffmpeg git git-lfs
git lfs install

# ---------------------------------------------------------------------------
# Step 2: Install uv
# ---------------------------------------------------------------------------
echo "[$(date -u)] === Step 2: Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# Step 3: Clone cosmos-reason2 (skip LFS — model downloaded separately)
# ---------------------------------------------------------------------------
echo "[$(date -u)] === Step 3: Clone cosmos-reason2 ==="
if [ ! -d "$HOME/cosmos-reason2" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-reason2.git $HOME/cosmos-reason2
fi
cd $HOME/cosmos-reason2

# ---------------------------------------------------------------------------
# Step 4: Python environment (CUDA 12.8)
# ---------------------------------------------------------------------------
echo "[$(date -u)] === Step 4: uv sync (CUDA 12.8 — ~3GB, ~10 min) ==="
uv sync --extra cu128
echo "[$(date -u)] CUDA check:"
$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------------------
# Step 5: Install fiftyone (uv pip — NOT pip or python -m pip)
# ---------------------------------------------------------------------------
echo "[$(date -u)] === Step 5: Install fiftyone ==="
uv pip install fiftyone
$PYTHON -c "import fiftyone; print('fiftyone:', fiftyone.__version__)"

# ---------------------------------------------------------------------------
# Step 6: Download model weights (~4.6GB)
# ---------------------------------------------------------------------------
echo "[$(date -u)] === Step 6: Download Cosmos-Reason2-2B weights ==="
$PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/Cosmos-Reason2-2B',
    local_dir='$HOME/cosmos-reason2/models/Cosmos-Reason2-2B',
    token='$HF_TOKEN'
)
print('Model download complete')
"
echo "[$(date -u)] Model size: $(du -sh $HOME/cosmos-reason2/models/Cosmos-Reason2-2B | cut -f1)"

# ---------------------------------------------------------------------------
# Step 7: Clone cosmos-cookbook and copy recipe files
# ---------------------------------------------------------------------------
echo "[$(date -u)] === Step 7: Get recipe files ==="
if [ ! -d "$HOME/cosmos-cookbook" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone \
    https://github.com/nvidia-cosmos/cosmos-cookbook.git \
    $HOME/cosmos-cookbook
fi
cp $HOME/cosmos-cookbook/docs/recipes/inference/reason2/worker_safety/worker_safety.py \
   $HOME/cosmos-reason2/worker_safety.py
cp $HOME/cosmos-cookbook/docs/recipes/inference/reason2/worker_safety/run_headless.py \
   $HOME/cosmos-reason2/run_headless.py
cp -r $HOME/cosmos-cookbook/docs/recipes/inference/reason2/worker_safety/assets \
   $HOME/cosmos-reason2/assets 2>/dev/null || true
echo "[$(date -u)] Recipe files copied OK"

# ---------------------------------------------------------------------------
# Step 8: Run inference headlessly via run_headless.py
# ---------------------------------------------------------------------------
echo "[$(date -u)] === Step 8: Run inference (40 videos) ==="
cd $HOME/cosmos-reason2
$PYTHON run_headless.py --results ~/inference_results.json

# ---------------------------------------------------------------------------
# Step 9: Launch FiftyOne App for browser viewing
# ---------------------------------------------------------------------------
echo "[$(date -u)] === Step 9: Launching FiftyOne on port 5151 ==="
echo "[$(date -u)] On your LOCAL machine run:"
echo "    brev port-forward worker-safety -p 5151:5151"
echo "    Then open: http://localhost:5151"

$PYTHON run_headless.py --results ~/inference_results.json --serve --port 5151

echo "[$(date -u)] === ALL DONE ==="
