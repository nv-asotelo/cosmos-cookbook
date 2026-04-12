#!/bin/bash
# demo.sh — Isaac GR00T-Mimic / Cosmos Transfer 1
# Headless video generation step for synthetic manipulation motion.
#
# SCOPE: This script runs the Cosmos Transfer 1 VIDEO GENERATION step only.
# Isaac Sim / Omniverse (simulation and motion export) must be run separately
# on a compatible NVIDIA Omniverse workstation. Export your control video from
# Isaac Sim and set BYO_VIDEO to that path before running this script.
#
# Usage:
#   export HF_TOKEN=hf_...
#   export BYO_VIDEO=/path/to/your/control_video.mp4   # Isaac Sim export
#   bash deploy/transfer1/gr00t-mimic/demo.sh
#
# Output:
#   /tmp/gr00t_mimic_output/output.mp4
#   /tmp/gr00t_mimic_results.json

set -e

COSMOS_TRANSFER1="${COSMOS_TRANSFER1:-/workspace/cosmos-transfer1}"
COOKBOOK="${COOKBOOK:-/workspace/cosmos-cookbook}"
OUTPUT_DIR="/tmp/gr00t_mimic_output"
RESULTS_JSON="/tmp/gr00t_mimic_results.json"

# BYO_VIDEO: path to a control video exported from Isaac Sim / Omniverse
# Set this to your actual file before running. Supported formats: .mp4, .avi, .mov
BYO_VIDEO="${BYO_VIDEO:-/path/to/your/isaac_sim_control_video.mp4}"

# ── Pre-flight ───────────────────────────────────────────────────────────────

echo "=== Pre-flight checks ==="

# GPU check
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader
echo ""

# VRAM check — Cosmos Transfer 1 requires >= 70000 MiB
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$VRAM_FREE" -lt 70000 ]; then
  echo "ERROR: Only ${VRAM_FREE} MiB VRAM free. Cosmos Transfer 1 requires >= 70000 MiB (H100-80GB)."
  exit 1
fi
echo "VRAM: ${VRAM_FREE} MiB free ✓"

# BYO_VIDEO check
if [ "$BYO_VIDEO" = "/path/to/your/isaac_sim_control_video.mp4" ] || [ ! -f "$BYO_VIDEO" ]; then
  echo ""
  echo "=========================================================="
  echo "ERROR: No valid control video provided."
  echo ""
  echo "This recipe requires a control video exported from NVIDIA Omniverse / Isaac Sim."
  echo "Isaac Sim is not available in headless cloud environments."
  echo ""
  echo "To use this demo:"
  echo "  1. On an NVIDIA Omniverse workstation, run Isaac Sim with the GR00T-Mimic blueprint"
  echo "     https://build.nvidia.com/nvidia/isaac-gr00t-synthetic-manipulation"
  echo "  2. Export a manipulation control video (RGB or depth/segmentation map)"
  echo "  3. Transfer the video to this machine"
  echo "  4. Re-run with: export BYO_VIDEO=/path/to/your/video.mp4"
  echo ""
  echo "Blueprint GitHub:"
  echo "  https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation"
  echo "=========================================================="
  exit 1
fi
echo "Control video: $BYO_VIDEO ✓"

# HF token check
if [ -z "$HF_TOKEN" ]; then
  echo ""
  echo "HuggingFace token required for model checkpoint downloads. Enter your token (hf_...):"
  read -r -s HF_TOKEN
  export HF_TOKEN
fi
echo "HF_TOKEN: set ✓"
echo ""

# ── Step 1: System dependencies ──────────────────────────────────────────────

echo "=== Step 1: System dependencies ==="
if ! command -v ffmpeg &>/dev/null || ! command -v git-lfs &>/dev/null; then
  sudo apt-get update -q
  sudo apt-get install -y -q curl ffmpeg git git-lfs wget
fi
git lfs install --skip-repo 2>/dev/null || true
echo "System deps ✓"

# ── Step 2: uv ───────────────────────────────────────────────────────────────

echo "=== Step 2: uv ==="
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env" 2>/dev/null || true
echo "uv $(uv --version) ✓"

# ── Step 3: Clone cosmos-transfer1 ───────────────────────────────────────────

echo "=== Step 3: cosmos-transfer1 ==="
if [ ! -d "$COSMOS_TRANSFER1" ]; then
  git clone https://github.com/nvidia-cosmos/cosmos-transfer1.git "$COSMOS_TRANSFER1"
  git -C "$COSMOS_TRANSFER1" lfs pull
  echo "Cloned ✓"
else
  echo "Already present ✓"
fi

# ── Step 4: HF login ─────────────────────────────────────────────────────────

echo "=== Step 4: HuggingFace auth ==="
echo "$HF_TOKEN" | huggingface-cli login --token 2>/dev/null || \
  huggingface-cli login --token "$HF_TOKEN"
echo "Authenticated ✓"

# ── Step 5: Python environment ───────────────────────────────────────────────

echo "=== Step 5: Python environment ==="
cd "$COSMOS_TRANSFER1"
if [ -f "pyproject.toml" ]; then
  uv sync 2>&1 | tail -5
  source .venv/bin/activate 2>/dev/null || true
elif [ -f "setup.py" ] || [ -f "requirements.txt" ]; then
  pip install -e . 2>&1 | tail -5
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
echo "CUDA available ✓"

# ── Step 6: Model checkpoints ────────────────────────────────────────────────

echo "=== Step 6: Model checkpoints ==="
CHECKPOINT_DIR="$COSMOS_TRANSFER1/checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
  echo "Downloading Cosmos Transfer 1 checkpoints (~100 GB — this will take a while)..."
  # Follow the cosmos-transfer1 INSTALL.md checkpoint download procedure
  if [ -f "$COSMOS_TRANSFER1/INSTALL.md" ]; then
    echo "See $COSMOS_TRANSFER1/INSTALL.md for checkpoint download instructions."
  fi
  huggingface-cli download nvidia/Cosmos-Transfer1-7B \
    --repo-type model \
    --local-dir "$CHECKPOINT_DIR" 2>/dev/null || \
  huggingface-cli download nvidia/Cosmos-Transfer1 \
    --repo-type model \
    --local-dir "$CHECKPOINT_DIR" 2>/dev/null || \
  echo "WARNING: Could not auto-download checkpoints. Follow INSTALL.md manually."
else
  echo "Checkpoints present ✓"
fi

# ── Step 7: Prepare output directory ─────────────────────────────────────────

echo "=== Step 7: Output directory ==="
mkdir -p "$OUTPUT_DIR"
echo "Output dir: $OUTPUT_DIR ✓"

# Build controlnet spec JSON for Cosmos Transfer 1
CONTROL_SPEC="$OUTPUT_DIR/control_spec.json"
cat > "$CONTROL_SPEC" <<JSONEOF
{
  "prompt": "A photorealistic manipulation scene. The robot arm performs a precise grasping and placement task in a well-lit workspace. Physics are accurate and motion is smooth.",
  "negative_prompt": "blurry, jittery, unrealistic physics, floating objects, distorted limbs",
  "input_video_path": "${BYO_VIDEO}",
  "guidance": 7.0,
  "sigma_max": 90.0,
  "seg": {
    "control_weight": 0.9
  },
  "depth": {
    "control_weight": 0.9
  }
}
JSONEOF
echo "Control spec written: $CONTROL_SPEC ✓"

# ── Step 8: Run inference ─────────────────────────────────────────────────────

echo ""
echo "=== Step 8: Running Cosmos Transfer 1 inference ==="
echo "Input:  $BYO_VIDEO"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$COSMOS_TRANSFER1"

START_NS=$(date +%s%N)

PYTHONPATH="$(pwd)" torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --video_save_folder "$OUTPUT_DIR" \
  --controlnet_specs "$CONTROL_SPEC" \
  --offload_text_encoder_model \
  --offload_guardrail_models \
  --num_gpus 1

END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

# ── Step 9: Collect metrics ───────────────────────────────────────────────────

echo ""
echo "=== Step 9: Metrics ==="

# Count output frames by probing generated video
OUTPUT_VIDEO=$(find "$OUTPUT_DIR" -name "*.mp4" -newer "$CONTROL_SPEC" | head -1)
FRAMES_GENERATED=0
if [ -n "$OUTPUT_VIDEO" ] && command -v ffprobe &>/dev/null; then
  FRAMES_GENERATED=$(ffprobe -v error -select_streams v:0 \
    -count_packets -show_entries stream=nb_read_packets \
    -of csv=p=0 "$OUTPUT_VIDEO" 2>/dev/null || echo "0")
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

# Compute derived metrics (integer math via awk)
if [ "$FRAMES_GENERATED" -gt 0 ] 2>/dev/null; then
  TIME_PER_FRAME=$(awk "BEGIN {printf \"%.2f\", $ELAPSED_MS / $FRAMES_GENERATED}")
  THROUGHPUT_FPS=$(awk "BEGIN {printf \"%.3f\", $FRAMES_GENERATED / ($ELAPSED_MS / 1000)}")
else
  TIME_PER_FRAME="N/A"
  THROUGHPUT_FPS="N/A"
  FRAMES_GENERATED=0
fi

STATUS="success"
if [ -z "$OUTPUT_VIDEO" ]; then
  STATUS="no_output_video_found"
fi

cat > "$RESULTS_JSON" <<JSONEOF
{
  "recipe": "gr00t-mimic",
  "model": "https://github.com/nvidia-cosmos/cosmos-transfer1",
  "model_family": "transfer1",
  "gpu": "${GPU_NAME}",
  "dataset": "BYO (Isaac Sim / Omniverse export)",
  "wall_time_ms": ${ELAPSED_MS},
  "samples_processed": 1,
  "frames_generated_total": ${FRAMES_GENERATED},
  "time_per_frame_ms": "${TIME_PER_FRAME}",
  "throughput_fps": "${THROUGHPUT_FPS}",
  "output_dir": "${OUTPUT_DIR}/",
  "output_video": "${OUTPUT_VIDEO}",
  "status": "${STATUS}"
}
JSONEOF

echo "Wall time:        ${ELAPSED_MS} ms"
echo "Frames generated: ${FRAMES_GENERATED}"
echo "Time/frame:       ${TIME_PER_FRAME} ms"
echo "Throughput:       ${THROUGHPUT_FPS} fps"
echo ""
echo "=== Done ==="
echo "Output video:  $OUTPUT_VIDEO"
echo "Results JSON:  $RESULTS_JSON"
