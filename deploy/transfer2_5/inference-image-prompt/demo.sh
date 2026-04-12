#!/bin/bash
# demo.sh — Style-Guided Video Generation with Cosmos Transfer 2.5
# Headless inference for Brev or any Linux GPU machine.
# No browser or JupyterLab required.
#
# BYO data — set BYO_VIDEO and BYO_STYLE_IMAGE before running:
#   export BYO_VIDEO=/path/to/your/control_video.mp4
#   export BYO_STYLE_IMAGE=/path/to/your/style_reference.jpg
#
# BYO_VIDEO should be an edge, depth, or segmentation map video.
# BYO_STYLE_IMAGE defines the target visual style (color, lighting, mood).
# NOTE: Blur control does NOT support image prompts.
#
# Usage:
#   export HF_TOKEN=hf_...
#   export BYO_VIDEO=/path/to/control_video.mp4
#   export BYO_STYLE_IMAGE=/path/to/style_reference.jpg
#   bash deploy/transfer2_5/inference-image-prompt/demo.sh
#
# Output:
#   /tmp/image_prompt_output/   — style-guided generated video
#   /tmp/image_prompt_results.json — timing and throughput metrics

set -e

RECIPE="image_prompt"
# Prefer /workspace (Brev default), fall back to $HOME (local dev)
COSMOS_DIR="${COSMOS_DIR:-/workspace/cosmos-transfer2_5}"
if [ ! -d "$COSMOS_DIR" ]; then
  COSMOS_DIR="$HOME/cosmos-transfer2.5"
fi
OUTPUT_DIR="/tmp/${RECIPE}_output"
RESULTS_JSON="/tmp/${RECIPE}_results.json"

# ── BYO data gate ────────────────────────────────────────────────────────────

# BYO data — set BYO_VIDEO to your control video path before running
BYO_VIDEO="${BYO_VIDEO:-}"
if [ -z "$BYO_VIDEO" ]; then
  echo "ERROR: BYO_VIDEO not set. Provide a control video (edge/depth/segmentation):"
  echo "  export BYO_VIDEO=/path/to/your/control_video.mp4"
  echo ""
  echo "BYO_VIDEO should be an edge, depth, or segmentation map video (MP4)."
  echo "Generate edge/depth/seg from your source video using cosmos-transfer2.5 preprocessing tools."
  exit 1
fi

if [ ! -f "$BYO_VIDEO" ]; then
  echo "ERROR: BYO_VIDEO file not found: ${BYO_VIDEO}"
  exit 1
fi

# BYO style image
BYO_STYLE_IMAGE="${BYO_STYLE_IMAGE:-}"
if [ -z "$BYO_STYLE_IMAGE" ]; then
  echo "ERROR: BYO_STYLE_IMAGE not set. Provide a style reference image:"
  echo "  export BYO_STYLE_IMAGE=/path/to/your/style_reference.jpg"
  echo ""
  echo "Use any JPEG or PNG image to define target visual style (color, lighting, mood)."
  echo "High-resolution reference images produce better style transfer results."
  exit 1
fi

if [ ! -f "$BYO_STYLE_IMAGE" ]; then
  echo "ERROR: BYO_STYLE_IMAGE file not found: ${BYO_STYLE_IMAGE}"
  exit 1
fi

echo "Control video:  ${BYO_VIDEO} ✓"
echo "Style image:    ${BYO_STYLE_IMAGE} ✓"

# ── Pre-flight ──────────────────────────────────────────────────────────────

echo "=== Pre-flight checks ==="

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader
echo ""

# VRAM check (require >= 70000 MiB free for Transfer 2.5)
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$VRAM_FREE" -lt 70000 ]; then
  echo "ERROR: Only ${VRAM_FREE} MiB VRAM free. Cosmos Transfer 2.5 requires >= 70000 MiB."
  exit 1
fi
echo "VRAM: ${VRAM_FREE} MiB free ✓"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')

# HF token check
if [ -z "$HF_TOKEN" ]; then
  echo ""
  echo "HuggingFace token required for model weights. Enter your token (hf_...):"
  read -r -s HF_TOKEN
  export HF_TOKEN
fi
echo "HF_TOKEN: set ✓"
echo ""

# ── Step 1: System dependencies ─────────────────────────────────────────────

echo "=== Step 1: System dependencies ==="
if ! command -v ffmpeg &>/dev/null || ! command -v git-lfs &>/dev/null; then
  sudo apt-get update -q
  sudo apt-get install -y -q curl ffmpeg git git-lfs
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

# ── Step 3: Clone cosmos-transfer2.5 ─────────────────────────────────────────

echo "=== Step 3: cosmos-transfer2.5 ==="
if [ ! -d "$COSMOS_DIR" ]; then
  git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$COSMOS_DIR"
  git -C "$COSMOS_DIR" lfs pull
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

echo "=== Step 5: Python environment (cu128) ==="
cd "$COSMOS_DIR"
uv sync --extra cu128 2>&1 | tail -5
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
echo "CUDA available ✓"

# ── Step 6: Build JSON spec file ─────────────────────────────────────────────

echo "=== Step 6: Build JSON spec file ==="
mkdir -p "$OUTPUT_DIR"
SPEC_FILE="/tmp/${RECIPE}_spec.json"

python - <<PYEOF
import json

spec = {
    "name": "image_style",
    "prompt": (
        "The camera moves steadily forward, simulating the perspective of a vehicle "
        "driving down the street. This forward motion is smooth, without any noticeable "
        "shaking or abrupt changes in direction, providing a continuous view of the urban "
        "landscape. The video maintains a consistent focus on the road ahead, with the "
        "buildings gradually receding into the distance as the camera progresses. "
        "The overall atmosphere is calm and quiet, with no pedestrians or vehicles in sight, "
        "emphasizing the emptiness of the street."
    ),
    "video_path": "${BYO_VIDEO}",
    "image_context_path": "${BYO_STYLE_IMAGE}",
    "seed": 1,
    "edge": {
        "control_weight": 1.0,
    },
}

with open("${SPEC_FILE}", "w") as f:
    json.dump(spec, f, indent=2)

print(f"Spec file written: ${SPEC_FILE}")
print(f"  Control video: ${BYO_VIDEO}")
print(f"  Style image:   ${BYO_STYLE_IMAGE}")
PYEOF

echo "Spec file ✓"

# ── Step 7: Run inference ────────────────────────────────────────────────────

echo ""
echo "=== Step 7: Running Cosmos Transfer 2.5 image-guided inference ==="
echo ""

INFER_SCRIPT="$COSMOS_DIR/examples/inference.py"
if [ ! -f "$INFER_SCRIPT" ]; then
  INFER_SCRIPT=$(find "$COSMOS_DIR" -name "inference.py" | head -1)
fi

# Get frame count from input video
FRAME_COUNT=$(python -c "
import cv2
cap = cv2.VideoCapture('${BYO_VIDEO}')
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(n if n > 0 else 24)
" 2>/dev/null || echo "24")

START_NS=$(date +%s%N)

python "$INFER_SCRIPT" -i "$SPEC_FILE" -o "$OUTPUT_DIR" 2>&1 | tail -5

END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

echo ""
echo "Inference complete"

# ── Step 8: Write results JSON ──────────────────────────────────────────────

echo "=== Step 8: Writing results ==="

python - <<PYEOF
import json
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
generated = list(output_dir.glob("*.mp4"))

elapsed_ms = ${ELAPSED_MS}
frames_total = ${FRAME_COUNT}

time_per_frame = round(elapsed_ms / frames_total, 2) if frames_total > 0 else 0
throughput_fps = round(frames_total / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else 0

result = {
    "recipe": "inference_image_prompt",
    "model": "cosmos-transfer2.5",
    "model_family": "transfer2_5",
    "gpu": "${GPU_NAME}",
    "dataset": "BYO control video + style reference image",
    "control_video": "${BYO_VIDEO}",
    "style_image": "${BYO_STYLE_IMAGE}",
    "wall_time_ms": elapsed_ms,
    "samples_processed": 1,
    "frames_generated_total": frames_total,
    "time_per_frame_ms": time_per_frame,
    "throughput_fps": throughput_fps,
    "output_dir": str(output_dir),
    "status": "success" if len(generated) > 0 else "no_outputs",
}

with open("${RESULTS_JSON}", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
PYEOF

echo ""
echo "=== Done ==="
echo "Output video: ${OUTPUT_DIR}"
echo "Results JSON: ${RESULTS_JSON}"
echo ""
echo "Try different style images or control modalities (depth/seg) — see:"
echo "docs/recipes/inference/transfer2_5/inference-image-prompt/inference.md"
