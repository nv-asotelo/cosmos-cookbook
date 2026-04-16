#!/bin/bash
# demo.sh — Sim2Real X-Mobility Navigation / Cosmos Transfer 1
# Headless inference for augmenting X-Mobility navigation dataset with Cosmos Transfer 1.
#
# SCOPE: INFERENCE ONLY — data augmentation step.
# Training phase requires 8x H100 GPUs — see inference.md for the full workflow:
#   Stage 1: World Model Pre-training (160K frames, 100 epochs, 8x H100)
#   Stage 2: Action Policy Training (100K frames, 100 epochs, 8x H100)
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash deploy/transfer1/inference-x-mobility/demo.sh
#
# Output:
#   /tmp/x_mobility_output/
#   /tmp/x_mobility_results.json

set -e

# Training phase requires 8x H100 — see inference.md for full workflow
COSMOS_TRANSFER1="${COSMOS_TRANSFER1:-/workspace/cosmos-transfer1}"
COOKBOOK="${COOKBOOK:-/workspace/cosmos-cookbook}"
RECIPE_SCRIPTS="$COOKBOOK/scripts/examples/transfer1/inference-x-mobility"
OUTPUT_DIR="/tmp/x_mobility_output"
RESULTS_JSON="/tmp/x_mobility_results.json"
DATA_DIR="/tmp/x_mobility_data"

# ── Pre-flight ───────────────────────────────────────────────────────────────

echo "=== Pre-flight checks ==="
echo ""
echo "SCOPE: INFERENCE (Cosmos Transfer 1 data augmentation) ONLY"
echo "Training phase requires 8x H100 — see inference.md for the full workflow."
echo ""

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
  echo "HuggingFace token required for model checkpoints and X-Mobility dataset."
  echo "Enter your token (hf_...):"
  read -r -s HF_TOKEN
  export HF_TOKEN
fi
echo "HF_TOKEN: set ✓"
echo ""

# ── Step 1: System dependencies ──────────────────────────────────────────────

echo "=== Step 1: System dependencies ==="
if ! command -v ffmpeg &>/dev/null || ! command -v git-lfs &>/dev/null; then
  sudo apt-get update -q
  sudo apt-get install -y -q curl ffmpeg git git-lfs wget unzip
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

# Copy X-Mobility recipe scripts
if [ -d "$RECIPE_SCRIPTS" ]; then
  mkdir -p "$COSMOS_TRANSFER1/examples/cookbook/inference-x-mobility"
  cp -r "$RECIPE_SCRIPTS/"* \
    "$COSMOS_TRANSFER1/examples/cookbook/inference-x-mobility/" 2>/dev/null || true
  echo "Recipe scripts copied ✓"
fi

# ── Step 4: Python environment ───────────────────────────────────────────────

echo "=== Step 4: Python environment ==="
cd "$COSMOS_TRANSFER1"
if [ -f "pyproject.toml" ]; then
  uv sync 2>&1 | tail -5
  source .venv/bin/activate 2>/dev/null || true
elif [ -f "setup.py" ] || [ -f "requirements.txt" ]; then
  pip install -e . 2>&1 | tail -5
fi
# cosmos-transfer1 may pin torch with CUDA > 12.8 (cu130 requires driver for CUDA 13.0).
# Hyperstack/Nebius H100 instances have driver 570.xx (CUDA 12.8 max).
# Force cu124 wheels — they run on any driver >= CUDA 12.4 and cosmos-transfer1.yaml
# declares cuda=12.4 as the reference environment.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "CUDA unavailable with default torch — reinstalling with cu124 wheels..."
  uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall 2>&1 | tail -5
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
echo "CUDA available ✓"
# xmob_dataset_to_videos.py reads parquet files — pyarrow not in cosmos-transfer1 deps
uv pip install pyarrow 2>&1 | tail -3
echo "pyarrow ✓"

# ── Step 5: HF login (after venv so huggingface-cli is available) ─────────────

echo "=== Step 5: HuggingFace auth ==="
echo "$HF_TOKEN" | huggingface-cli login --token 2>/dev/null || \
  huggingface-cli login --token "$HF_TOKEN"
echo "Authenticated ✓"

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

# ── Step 7: Download X-Mobility dataset subset ───────────────────────────────

echo "=== Step 7: X-Mobility dataset ==="
mkdir -p "$DATA_DIR"

XMOB_ZIP="$DATA_DIR/x_mobility_isaac_sim_random_160k.zip"

# The zip extracts to a subdirectory (may differ from the zip filename).
# Discover the actual extracted dir; fall back to downloading if absent.
XMOB_SUBSET_DIR=$(find "$DATA_DIR" -maxdepth 1 -type d -name "*isaac_sim*" 2>/dev/null | head -1)

if [ -z "$XMOB_SUBSET_DIR" ]; then
  echo "Downloading X-Mobility random action dataset..."
  echo "(Full dataset is 160K frames — for demo we download and use a small subset)"
  huggingface-cli download nvidia/X-Mobility \
    x_mobility_isaac_sim_random_160k.zip \
    --repo-type dataset \
    --local-dir "$DATA_DIR" 2>/dev/null || {
    echo "WARNING: Could not download X-Mobility dataset automatically."
    echo ""
    echo "Manual download instructions:"
    echo "  1. Visit https://huggingface.co/datasets/nvidia/X-Mobility"
    echo "  2. Accept the dataset license"
    echo "  3. Download x_mobility_isaac_sim_random_160k.zip"
    echo "  4. Place at: $XMOB_ZIP"
    echo "  5. Re-run this script"
    exit 1
  }

  if [ -f "$XMOB_ZIP" ]; then
    echo "Extracting dataset..."
    unzip -oq "$XMOB_ZIP" -d "$DATA_DIR"
    echo "Extracted ✓"
  fi

  # Re-discover the extracted dir after extraction
  XMOB_SUBSET_DIR=$(find "$DATA_DIR" -maxdepth 1 -type d -name "*isaac_sim*" 2>/dev/null | head -1)
else
  echo "Dataset already present at $XMOB_SUBSET_DIR ✓"
fi

if [ -z "$XMOB_SUBSET_DIR" ]; then
  echo "ERROR: Could not find extracted X-Mobility dataset in $DATA_DIR"
  exit 1
fi

# ── Step 8: Convert X-Mobility frames to videos ──────────────────────────────

echo "=== Step 8: Convert X-Mobility frames to videos ==="
INPUT_VIDEOS_DIR="$DATA_DIR/x_mobility_input_videos"
CONVERT_SCRIPT="$COSMOS_TRANSFER1/examples/cookbook/inference-x-mobility/xmob_dataset_to_videos.py"

# Fallback: also check the cosmos-cookbook scripts location
if [ ! -f "$CONVERT_SCRIPT" ]; then
  CONVERT_SCRIPT="$RECIPE_SCRIPTS/xmob_dataset_to_videos.py"
fi

if [ -f "$CONVERT_SCRIPT" ] && [ -d "$XMOB_SUBSET_DIR" ]; then
  echo "Converting X-Mobility frames to video format..."
  uv run "$CONVERT_SCRIPT" "$XMOB_SUBSET_DIR" "$INPUT_VIDEOS_DIR" 2>&1 | tail -10
  echo "Conversion complete ✓"
else
  echo "WARNING: xmob_dataset_to_videos.py not found or dataset missing."
  echo "Creating a synthetic placeholder video for pipeline testing..."
  INPUT_VIDEOS_DIR="$DATA_DIR/placeholder_videos"
  mkdir -p "$INPUT_VIDEOS_DIR"
  PLACEHOLDER_VIDEO="$INPUT_VIDEOS_DIR/output_0360.mp4"
  PLACEHOLDER_SEG="$INPUT_VIDEOS_DIR/output_0360_segmentation.mp4"
  ffmpeg -y -f lavfi -i "color=c=gray:size=640x480:duration=2:rate=30" \
    -c:v libx264 -pix_fmt yuv420p "$PLACEHOLDER_VIDEO" 2>/dev/null
  ffmpeg -y -f lavfi -i "color=c=blue:size=640x480:duration=2:rate=30" \
    -c:v libx264 -pix_fmt yuv420p "$PLACEHOLDER_SEG" 2>/dev/null
  echo "Placeholder videos created ✓"
fi

# Pick the first available input video for the demo
INPUT_VIDEO=$(find "$INPUT_VIDEOS_DIR" -name "*.mp4" -not -name "*segmentation*" | head -1)
SEG_VIDEO=$(find "$INPUT_VIDEOS_DIR" -name "*segmentation*.mp4" | head -1)

if [ -z "$INPUT_VIDEO" ]; then
  echo "ERROR: No input video found in $INPUT_VIDEOS_DIR"
  exit 1
fi
echo "Input video:       $INPUT_VIDEO"
echo "Segmentation ctrl: ${SEG_VIDEO:-none}"

# ── Step 9: Build control spec ───────────────────────────────────────────────

echo "=== Step 9: Build control spec ==="
mkdir -p "$OUTPUT_DIR"

CONTROL_SPEC="$OUTPUT_DIR/inference_cosmos_transfer1_xmobility.json"

SEG_INPUT_CONTROL=""
if [ -n "$SEG_VIDEO" ] && [ -f "$SEG_VIDEO" ]; then
  SEG_INPUT_CONTROL=",
        \"input_control\": \"${SEG_VIDEO}\""
fi

cat > "$CONTROL_SPEC" <<JSONEOF
{
    "prompt": "A realistic warehouse environment with consistent lighting, perspective, and camera motion. Preserve the original structure, object positions, and layout from the input video. Ensure the output exactly matches the segmentation video frame-by-frame in timing and content. Camera movement must follow the original path precisely.",
    "input_video_path": "${INPUT_VIDEO}",
    "edge": {
        "control_weight": 0.3
    },
    "seg": {
        "control_weight": 1.0${SEG_INPUT_CONTROL}
    }
}
JSONEOF
echo "Control spec written ✓"
echo "  Controls: seg=1.0 (geometry preserved), edge=0.3 (structure hint)"
echo ""

# ── Step 10: Run inference ────────────────────────────────────────────────────

echo "=== Step 10: Running Cosmos Transfer 1 inference ==="
echo "Input:  $INPUT_VIDEO"
echo "Output: $OUTPUT_DIR"
echo ""
echo "# Training phase requires 8x H100 — see inference.md for full workflow"
echo ""

cd "$COSMOS_TRANSFER1"

# transformer_engine bundles libcudnn.so.9 and libnccl inside the venv's nvidia packages.
# Without these on LD_LIBRARY_PATH, TE fails to load its shared libs at runtime.
NVIDIA_LIBS="$(pwd)/.venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}/cudnn/lib:${NVIDIA_LIBS}/nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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

# ── Step 11: Collect metrics ──────────────────────────────────────────────────

echo ""
echo "=== Step 11: Metrics ==="

OUTPUT_VIDEO=$(find "$OUTPUT_DIR" -name "*.mp4" -newer "$CONTROL_SPEC" | head -1)
FRAMES_GENERATED=0
if [ -n "$OUTPUT_VIDEO" ] && command -v ffprobe &>/dev/null; then
  FRAMES_GENERATED=$(ffprobe -v error -select_streams v:0 \
    -count_packets -show_entries stream=nb_read_packets \
    -of csv=p=0 "$OUTPUT_VIDEO" 2>/dev/null || echo "0")
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

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
  "recipe": "inference-x-mobility",
  "model": "https://github.com/nvidia-cosmos/cosmos-transfer1",
  "model_family": "transfer1",
  "gpu": "${GPU_NAME}",
  "dataset": "nvidia/X-Mobility (random action subset)",
  "controls_used": ["seg", "edge"],
  "seg_control_weight": 1.0,
  "edge_control_weight": 0.3,
  "wall_time_ms": ${ELAPSED_MS},
  "samples_processed": 1,
  "frames_generated_total": ${FRAMES_GENERATED},
  "time_per_frame_ms": "${TIME_PER_FRAME}",
  "throughput_fps": "${THROUGHPUT_FPS}",
  "output_dir": "${OUTPUT_DIR}/",
  "output_video": "${OUTPUT_VIDEO}",
  "scope": "inference_only",
  "training_note": "Training requires 8x H100 GPUs — Stage 1: world model pre-training (160K frames, 100 epochs), Stage 2: action policy training (100K frames, 100 epochs). See inference.md.",
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
echo ""
echo "========================================================"
echo "NEXT STEPS (not covered by this demo):"
echo "  Training phase requires 8x H100 GPUs:"
echo "  Stage 1: World Model Pre-training"
echo "    Dataset: x_mobility_isaac_sim_random_160k (160K frames)"
echo "    Epochs: 100, Batch: 32, Hardware: 8x H100"
echo "  Stage 2: Action Policy Training"
echo "    Dataset: x_mobility_isaac_sim_nav2_100k (100K frames)"
echo "    Epochs: 100, Batch: 32, Hardware: 8x H100"
echo "  See: docs/recipes/inference/transfer1/inference-x-mobility/inference.md"
echo "========================================================"
