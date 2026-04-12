#!/bin/bash
# demo.sh — CARLA Sim2Real Augmentation with Cosmos Transfer 2.5
# Headless inference for Brev or any Linux GPU machine.
# No browser or JupyterLab required.
#
# BYO data — set BYO_VIDEO to your simulator RGB video path before running:
#   export BYO_VIDEO=/path/to/your/simulator_rgb_video.mp4
#
# Runs 2 representative augmentations (edge control: snow + night) from the
# full 18-condition pipeline described in inference.md.
#
# Usage:
#   export HF_TOKEN=hf_...
#   export BYO_VIDEO=/path/to/simulator_rgb_video.mp4
#   bash deploy/transfer2_5/inference-carla-sdg-augmentation/demo.sh
#
# Output:
#   /tmp/carla_sdg_output/   — photorealistic augmented driving videos
#   /tmp/carla_sdg_results.json — timing and throughput metrics

set -e

RECIPE="carla_sdg"
COSMOS_DIR="$HOME/cosmos-transfer2.5"
OUTPUT_DIR="/tmp/${RECIPE}_output"
RESULTS_JSON="/tmp/${RECIPE}_results.json"

# ── BYO data gate ────────────────────────────────────────────────────────────

# BYO data — set BYO_VIDEO to your control video path before running
BYO_VIDEO="${BYO_VIDEO:-}"
if [ -z "$BYO_VIDEO" ]; then
  echo "ERROR: BYO_VIDEO not set. Provide a simulator RGB video:"
  echo "  export BYO_VIDEO=/path/to/your/simulator_rgb_video.mp4"
  echo ""
  echo "Generate a CARLA video at: https://carla.org/"
  echo "Any simulator-generated driving video (MP4) will work."
  exit 1
fi

if [ ! -f "$BYO_VIDEO" ]; then
  echo "ERROR: BYO_VIDEO file not found: ${BYO_VIDEO}"
  exit 1
fi

echo "Input video: ${BYO_VIDEO} ✓"

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
  sudo apt-get install -y -q curl ffmpeg git git-lfs libgl1 libglib2.0-0
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

# ── Step 6: Install opencv for edge generation ───────────────────────────────

echo "=== Step 6: Recipe dependencies ==="
pip install -q opencv-python
echo "OpenCV ✓"

# ── Step 7: Generate Canny edge map from input ───────────────────────────────

echo "=== Step 7: Generate Canny edge map ==="
mkdir -p "$OUTPUT_DIR"
EDGE_VIDEO="/tmp/${RECIPE}_edge.mp4"

python - <<PYEOF
import cv2, sys
from pathlib import Path

input_video = "${BYO_VIDEO}"
output_video = "${EDGE_VIDEO}"

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"ERROR: Cannot open video: {input_video}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 24
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

out = cv2.VideoWriter(output_video,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (w, h), isColor=False)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    out.write(edges)

cap.release()
out.release()
print(f"Edge map written: {output_video}")
PYEOF

echo "Edge map ✓"

# ── Step 8: Build JSON spec files (2 augmentations: snow + night) ────────────

echo "=== Step 8: Build augmentation spec files ==="
SPECS_DIR="/tmp/${RECIPE}_specs"
mkdir -p "$SPECS_DIR"

python - <<PYEOF
import json
from pathlib import Path

video_path = "${BYO_VIDEO}"
edge_path = "${EDGE_VIDEO}"
specs_dir = Path("${SPECS_DIR}")

NEG_PROMPT = (
    "cartoon, CG rendering, video game, bad graphics, low quality textures, "
    "pixelated, unrealistic, fake lighting, subtitles."
)

augmentations = [
    {
        "name": "snow_falling",
        "prompt": (
            "The video depicts an urban intersection during daytime with clear skies, "
            "heavy snow falling from above, snow-covered ground with visible tire tracks, "
            "vehicles navigating carefully through the snow, traffic lights overhead, "
            "modern city buildings in the background. Photorealistic winter conditions."
        ),
    },
    {
        "name": "night",
        "prompt": (
            "The video depicts an urban intersection at night, low ambient light, "
            "vehicle headlights and streetlamps illuminating the road, high contrast "
            "shadows, wet road surface reflecting light, realistic nighttime city atmosphere, "
            "traffic moving through the intersection."
        ),
    },
]

for aug in augmentations:
    spec = {
        "name": aug["name"],
        "prompt": aug["prompt"],
        "negative_prompt": NEG_PROMPT,
        "video_path": video_path,
        "control_weight": 1.0,
        "edge": {
            "control_path": edge_path,
        },
    }
    spec_path = specs_dir / f"{aug['name']}.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"Created spec: {spec_path}")
PYEOF

echo "Spec files ✓"

# ── Step 9: Run inference ────────────────────────────────────────────────────

echo ""
echo "=== Step 9: Running Cosmos Transfer 2.5 inference (2 augmentations) ==="
echo ""

INFER_SCRIPT="$COSMOS_DIR/examples/inference.py"
if [ ! -f "$INFER_SCRIPT" ]; then
  INFER_SCRIPT=$(find "$COSMOS_DIR" -name "inference.py" | head -1)
fi

START_NS=$(date +%s%N)
FRAMES_TOTAL=0
PROCESSED=0

# Get frame count from input video
FRAME_COUNT=$(python -c "
import cv2
cap = cv2.VideoCapture('${BYO_VIDEO}')
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cap.release()
" 2>/dev/null || echo "24")

for SPEC_FILE in "$SPECS_DIR"/*.json; do
  STEM=$(basename "$SPEC_FILE" .json)
  echo "  Augmenting: $STEM ..."
  python "$INFER_SCRIPT" -i "$SPEC_FILE" -o "$OUTPUT_DIR" 2>&1 | tail -3
  PROCESSED=$((PROCESSED + 1))
  FRAMES_TOTAL=$((FRAMES_TOTAL + FRAME_COUNT))
done

END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

echo ""
echo "Inference complete: ${PROCESSED} augmentations generated"

# ── Step 10: Write results JSON ──────────────────────────────────────────────

echo "=== Step 10: Writing results ==="

python - <<PYEOF
import json
from pathlib import Path

elapsed_ms = ${ELAPSED_MS}
frames_total = ${FRAMES_TOTAL}
processed = ${PROCESSED}

time_per_frame = round(elapsed_ms / frames_total, 2) if frames_total > 0 else 0
throughput_fps = round(frames_total / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else 0

result = {
    "recipe": "carla_sdg_augmentation",
    "model": "cosmos-transfer2.5",
    "model_family": "transfer2_5",
    "gpu": "${GPU_NAME}",
    "dataset": "BYO CARLA simulator video",
    "input_video": "${BYO_VIDEO}",
    "augmentations_run": ["snow_falling", "night"],
    "wall_time_ms": elapsed_ms,
    "samples_processed": processed,
    "frames_generated_total": frames_total,
    "time_per_frame_ms": time_per_frame,
    "throughput_fps": throughput_fps,
    "output_dir": "${OUTPUT_DIR}",
    "status": "success" if processed > 0 else "no_outputs",
}

with open("${RESULTS_JSON}", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
PYEOF

echo ""
echo "=== Done ==="
echo "Output videos: ${OUTPUT_DIR}"
echo "Results JSON:  ${RESULTS_JSON}"
echo ""
echo "Full 18-augmentation pipeline: docs/recipes/inference/transfer2_5/inference-carla-sdg-augmentation/inference.md"
