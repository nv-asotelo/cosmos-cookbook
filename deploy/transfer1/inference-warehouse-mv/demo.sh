#!/bin/bash
# demo.sh — Sim2Real Multi-View Warehouse / Cosmos Transfer 1
# Headless video generation for warehouse sim-to-real domain adaptation.
#
# Processes 6 synchronized camera views from the bundled SURF_Booth_030825 dataset
# through Cosmos Transfer 1 with edge + depth controls, producing realistic
# warehouse videos suitable for training detection and tracking models.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash deploy/transfer1/inference-warehouse-mv/demo.sh
#
# Optional — process only Camera_00 for a quick test:
#   export SINGLE_CAMERA=Camera_00
#
# Output:
#   /tmp/warehouse_mv_output/{camera}/output.mp4
#   /tmp/warehouse_mv_results.json

set -e

COSMOS_TRANSFER1="${COSMOS_TRANSFER1:-/workspace/cosmos-transfer1}"
COOKBOOK="${COOKBOOK:-/workspace/cosmos-cookbook}"
ASSETS_DIR="$COOKBOOK/scripts/examples/transfer1/inference-warehouse-mv/assets/SURF_Booth_030825"
OUTPUT_DIR="/tmp/warehouse_mv_output"
RESULTS_JSON="/tmp/warehouse_mv_results.json"

# Set SINGLE_CAMERA to process only one camera (e.g., Camera_00) for quick testing
SINGLE_CAMERA="${SINGLE_CAMERA:-}"

WAREHOUSE_PROMPT="The camera provides a clear view of the warehouse interior, showing rows of shelves stacked with boxes, baskets and other objects. There are workers and robots walking around, moving boxes or operating machinery. The lighting is bright and even, with overhead fluorescent lights illuminating the space. The floor is clean and well-maintained, with clear pathways between the shelves. The atmosphere is busy but organized, with workers and humanoids moving efficiently around the warehouse."

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
  echo "ERROR: Only ${VRAM_FREE} MiB VRAM free. Cosmos Transfer 1 requires >= 70000 MiB (H100/A100-80GB)."
  exit 1
fi
echo "VRAM: ${VRAM_FREE} MiB free ✓"

# HF token check
if [ -z "$HF_TOKEN" ]; then
  echo ""
  echo "HuggingFace token required for model checkpoints. Enter your token (hf_...):"
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

# Copy cookbook recipe assets into cosmos-transfer1 examples directory
mkdir -p "$COSMOS_TRANSFER1/examples/cookbook"
if [ -d "$COOKBOOK/scripts/examples/transfer1" ]; then
  cp -r "$COOKBOOK/scripts/examples/transfer1/"* \
    "$COSMOS_TRANSFER1/examples/cookbook/" 2>/dev/null || true
  echo "Recipe assets copied ✓"
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

# ── Step 7: Validate dataset ─────────────────────────────────────────────────

echo "=== Step 7: Validate warehouse dataset ==="
if [ ! -d "$ASSETS_DIR" ]; then
  echo "ERROR: SURF_Booth_030825 dataset not found at: $ASSETS_DIR"
  echo ""
  echo "This dataset is bundled with the recipe. Ensure cosmos-cookbook is cloned"
  echo "with LFS assets pulled:"
  echo "  git lfs pull"
  echo ""
  echo "Or provide your own multi-camera warehouse dataset:"
  echo "  huggingface-cli download nvidia/PhysicalAI-SmartSpaces --repo-type dataset"
  exit 1
fi
echo "Dataset: $ASSETS_DIR ✓"

# Determine which cameras to process
if [ -n "$SINGLE_CAMERA" ]; then
  CAMERAS=("$SINGLE_CAMERA")
  echo "Single-camera mode: $SINGLE_CAMERA"
else
  CAMERAS=(Camera_00 Camera_01 Camera_02 Camera_03 Camera_04 Camera_05)
  echo "Processing all 6 cameras: ${CAMERAS[*]}"
fi

# ── Step 8: Run inference per camera ─────────────────────────────────────────

echo ""
echo "=== Step 8: Running Cosmos Transfer 1 per camera ==="
echo ""

mkdir -p "$OUTPUT_DIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
cd "$COSMOS_TRANSFER1"

TOTAL_FRAMES=0
TOTAL_ELAPSED_MS=0
CAMERAS_PROCESSED=0
CAMERAS_FAILED=0
CAMERA_RESULTS=""

for CAMERA in "${CAMERAS[@]}"; do
  CAMERA_DIR="$ASSETS_DIR/$CAMERA"
  RGB_VIDEO="$CAMERA_DIR/rgb.mp4"
  DEPTH_VIDEO="$CAMERA_DIR/depth.mp4"
  CAM_OUTPUT_DIR="$OUTPUT_DIR/$CAMERA"
  CONTROL_SPEC="$OUTPUT_DIR/${CAMERA}_control_spec.json"

  echo "--- $CAMERA ---"

  if [ ! -f "$RGB_VIDEO" ]; then
    echo "  SKIP: rgb.mp4 not found at $RGB_VIDEO"
    CAMERAS_FAILED=$((CAMERAS_FAILED + 1))
    continue
  fi

  mkdir -p "$CAM_OUTPUT_DIR"

  # Build control spec (edge=0.5, depth=0.5 — recipe recommendation for warehouse)
  DEPTH_CONTROL=""
  if [ -f "$DEPTH_VIDEO" ]; then
    DEPTH_CONTROL=", \"input_control\": \"${DEPTH_VIDEO}\""
  fi

  cat > "$CONTROL_SPEC" <<JSONEOF
{
  "prompt": "${WAREHOUSE_PROMPT}",
  "input_video_path": "${RGB_VIDEO}",
  "edge": {
    "control_weight": 0.5
  },
  "depth": {
    "control_weight": 0.5${DEPTH_CONTROL}
  }
}
JSONEOF

  CAM_START_NS=$(date +%s%N)

  PYTHONPATH="$(pwd)" torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --video_save_folder "$CAM_OUTPUT_DIR" \
    --controlnet_specs "$CONTROL_SPEC" \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus 1

  CAM_END_NS=$(date +%s%N)
  CAM_ELAPSED_MS=$(( (CAM_END_NS - CAM_START_NS) / 1000000 ))

  CAM_OUTPUT_VIDEO=$(find "$CAM_OUTPUT_DIR" -name "*.mp4" -newer "$CONTROL_SPEC" | head -1)
  CAM_FRAMES=0
  if [ -n "$CAM_OUTPUT_VIDEO" ] && command -v ffprobe &>/dev/null; then
    CAM_FRAMES=$(ffprobe -v error -select_streams v:0 \
      -count_packets -show_entries stream=nb_read_packets \
      -of csv=p=0 "$CAM_OUTPUT_VIDEO" 2>/dev/null || echo "0")
  fi

  echo "  Done: ${CAM_ELAPSED_MS} ms, ${CAM_FRAMES} frames -> $CAM_OUTPUT_VIDEO"

  TOTAL_FRAMES=$((TOTAL_FRAMES + CAM_FRAMES))
  TOTAL_ELAPSED_MS=$((TOTAL_ELAPSED_MS + CAM_ELAPSED_MS))
  CAMERAS_PROCESSED=$((CAMERAS_PROCESSED + 1))

  CAMERA_RESULTS="${CAMERA_RESULTS}    {\"camera\": \"${CAMERA}\", \"elapsed_ms\": ${CAM_ELAPSED_MS}, \"frames\": ${CAM_FRAMES}, \"output\": \"${CAM_OUTPUT_VIDEO}\"},"
done

# ── Step 9: Aggregate metrics ─────────────────────────────────────────────────

echo ""
echo "=== Step 9: Aggregate metrics ==="

if [ "$TOTAL_FRAMES" -gt 0 ] 2>/dev/null; then
  TIME_PER_FRAME=$(awk "BEGIN {printf \"%.2f\", $TOTAL_ELAPSED_MS / $TOTAL_FRAMES}")
  THROUGHPUT_FPS=$(awk "BEGIN {printf \"%.3f\", $TOTAL_FRAMES / ($TOTAL_ELAPSED_MS / 1000)}")
else
  TIME_PER_FRAME="N/A"
  THROUGHPUT_FPS="N/A"
fi

STATUS="success"
if [ "$CAMERAS_PROCESSED" -eq 0 ]; then
  STATUS="no_cameras_processed"
elif [ "$CAMERAS_FAILED" -gt 0 ]; then
  STATUS="partial_success_${CAMERAS_FAILED}_cameras_failed"
fi

# Remove trailing comma from camera results array
CAMERA_RESULTS="${CAMERA_RESULTS%,}"

cat > "$RESULTS_JSON" <<JSONEOF
{
  "recipe": "inference-warehouse-mv",
  "model": "https://github.com/nvidia-cosmos/cosmos-transfer1",
  "model_family": "transfer1",
  "gpu": "${GPU_NAME}",
  "dataset": "SURF_Booth_030825 (6-camera warehouse sample)",
  "cameras_processed": ${CAMERAS_PROCESSED},
  "cameras_failed": ${CAMERAS_FAILED},
  "controls_used": ["edge", "depth"],
  "edge_control_weight": 0.5,
  "depth_control_weight": 0.5,
  "wall_time_ms": ${TOTAL_ELAPSED_MS},
  "samples_processed": ${CAMERAS_PROCESSED},
  "frames_generated_total": ${TOTAL_FRAMES},
  "time_per_frame_ms": "${TIME_PER_FRAME}",
  "throughput_fps": "${THROUGHPUT_FPS}",
  "output_dir": "${OUTPUT_DIR}/",
  "per_camera": [
${CAMERA_RESULTS}
  ],
  "status": "${STATUS}"
}
JSONEOF

echo "Cameras processed: ${CAMERAS_PROCESSED} / ${#CAMERAS[@]}"
echo "Total frames:      ${TOTAL_FRAMES}"
echo "Total wall time:   ${TOTAL_ELAPSED_MS} ms"
echo "Time/frame:        ${TIME_PER_FRAME} ms"
echo "Throughput:        ${THROUGHPUT_FPS} fps"
echo ""
echo "=== Done ==="
echo "Output directory: $OUTPUT_DIR"
echo "Results JSON:     $RESULTS_JSON"
echo ""
echo "For extended datasets:"
echo "  huggingface-cli download nvidia/PhysicalAI-SmartSpaces --repo-type dataset"
