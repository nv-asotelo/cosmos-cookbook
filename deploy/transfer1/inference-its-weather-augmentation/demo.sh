#!/bin/bash
# demo.sh — ITS Weather Augmentation / Cosmos Transfer 1
# Headless video generation for weather-augmenting ITS (Intelligent Transportation System) images.
#
# Transforms clear-weather highway scenes into adverse weather conditions
# (rainy night, fog, snow) using Cosmos Transfer 1 with depth + segmentation controls.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash deploy/transfer1/inference-its-weather-augmentation/demo.sh
#
# Optional — bring your own ITS video:
#   export BYO_VIDEO=/path/to/your/clear_weather_its_video.mp4
#
# Output:
#   /tmp/its_weather_output/
#   /tmp/its_weather_results.json

set -e

COSMOS_TRANSFER1="${COSMOS_TRANSFER1:-/workspace/cosmos-transfer1}"
COOKBOOK="${COOKBOOK:-/workspace/cosmos-cookbook}"
RECIPE_ASSETS="$COOKBOOK/docs/recipes/inference/transfer1/inference-its-weather-augmentation/assets"
OUTPUT_DIR="/tmp/its_weather_output"
RESULTS_JSON="/tmp/its_weather_results.json"

# BYO_VIDEO: override with your own clear-weather ITS video if desired
BYO_VIDEO="${BYO_VIDEO:-}"

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

# ── Step 4: Python environment ───────────────────────────────────────────────
# (must come before HF login — huggingface-cli lives in the venv)

echo "=== Step 4: Python environment ==="
cd "$COSMOS_TRANSFER1"
if [ -f "pyproject.toml" ]; then
  uv sync 2>&1 | tail -5
  source .venv/bin/activate 2>/dev/null || true
elif [ -f "setup.py" ] || [ -f "requirements.txt" ]; then
  pip install -e . 2>&1 | tail -5
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
echo "CUDA available ✓"

# ── Step 5: HF login ─────────────────────────────────────────────────────────

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

# ── Step 7: Prepare input video ──────────────────────────────────────────────

echo "=== Step 7: Prepare input video ==="
mkdir -p "$OUTPUT_DIR"

if [ -n "$BYO_VIDEO" ] && [ -f "$BYO_VIDEO" ]; then
  INPUT_VIDEO="$BYO_VIDEO"
  echo "Using BYO video: $INPUT_VIDEO ✓"
elif [ -f "$RECIPE_ASSETS/input.jpg" ]; then
  # Convert the recipe's sample ITS image into a short looping video for Transfer 1
  INPUT_VIDEO="$OUTPUT_DIR/input_its_video.mp4"
  echo "Converting sample ITS image to video (30 frames @ 30fps = 1 second)..."
  ffmpeg -y -loop 1 -i "$RECIPE_ASSETS/input.jpg" \
    -t 1 -r 30 -vf "scale=1280:720" \
    -c:v libx264 -pix_fmt yuv420p \
    "$INPUT_VIDEO" 2>/dev/null
  echo "Sample ITS video created: $INPUT_VIDEO ✓"
else
  echo "WARNING: No input image found at $RECIPE_ASSETS/input.jpg and no BYO_VIDEO set."
  echo "Creating a synthetic gray placeholder video for pipeline testing..."
  INPUT_VIDEO="$OUTPUT_DIR/placeholder_its_video.mp4"
  ffmpeg -y -f lavfi -i "color=c=gray:size=1280x720:duration=1:rate=30" \
    -c:v libx264 -pix_fmt yuv420p \
    "$INPUT_VIDEO" 2>/dev/null
  echo "Placeholder video created: $INPUT_VIDEO ✓"
  echo ""
  echo "NOTE: For real weather augmentation, provide an ITS image/video:"
  echo "  - Place a clear-weather highway image at: $RECIPE_ASSETS/input.jpg"
  echo "  - Or set: export BYO_VIDEO=/path/to/your/video.mp4"
fi

# Build controlnet spec JSON — depth + seg only (recipe recommendation)
CONTROL_SPEC="$OUTPUT_DIR/control_spec_its_weather.json"
cat > "$CONTROL_SPEC" <<JSONEOF
{
    "input_video_path": "${INPUT_VIDEO}",
    "prompt": "The video depicts a busy highway scene during a rainy night, shrouded in deep darkness. The sky is obscured, and the only visible light comes from scattered headlights and dim streetlights, casting faint reflections on the rain-soaked road. The highway is multi-lane, with traffic flowing in both directions, and the surrounding landscape is barely visible. There are several cars on the road, their headlights illuminating the rain-soaked road, where puddles and thin streams of water shimmer faintly. The overall atmosphere is calm yet active, typical of a nighttime commute.",
    "negative_prompt": "The video shows an unrealistic traffic scene with floating or jittery cars that ignore physics, sliding without friction, turning sharply without steering, or vanishing mid-motion. Vehicles overlap unnaturally, lack weight or inertia, and do not align with the road. Road markings are inconsistent or missing, and lanes appear distorted. Lighting is flickering and fake, while backgrounds look melted or warped. Pedestrians and traffic signals are misshapen, duplicated, or misplaced. Overall, the scene feels chaotic, lacks depth, structure, and visual coherence.",
    "guidance": 8.0,
    "sigma_max": 90.0,
    "depth": {
        "control_weight": 0.9
    },
    "seg": {
        "control_weight": 0.9
    }
}
JSONEOF
echo "Control spec written ✓"
echo "  Controls: depth=0.9, seg=0.9 (vis/edge excluded for night scene generation)"

# ── Step 8: Run inference ─────────────────────────────────────────────────────

echo ""
echo "=== Step 8: Running Cosmos Transfer 1 weather augmentation ==="
echo "Input:     $INPUT_VIDEO"
echo "Output:    $OUTPUT_DIR"
echo "Weather:   Rainy night"
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
  "recipe": "inference-its-weather-augmentation",
  "model": "https://github.com/nvidia-cosmos/cosmos-transfer1",
  "model_family": "transfer1",
  "gpu": "${GPU_NAME}",
  "dataset": "ITS sample (recipe assets) — ACDC/SUTD/DAWN for full evaluation",
  "weather_condition": "rainy_night",
  "controls_used": ["depth", "seg"],
  "control_weight": 0.9,
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
echo ""
echo "For full benchmark evaluation, download ACDC/SUTD/DAWN datasets:"
echo "  ACDC: https://acdc.vision.ee.ethz.ch/"
echo "  SUTD: https://sutdcv.github.io/SUTD-TrafficQA/#/download"
echo "  DAWN: https://www.kaggle.com/datasets/shuvoalok/dawn-dataset"
