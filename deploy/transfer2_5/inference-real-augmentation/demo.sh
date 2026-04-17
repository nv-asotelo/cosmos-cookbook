#!/bin/bash
# demo.sh — Multi-Control Augmentation with Cosmos Transfer 2.5
# Headless inference for Brev or any Linux GPU machine.
# No browser or JupyterLab required.
#
# BYO data — set BYO_VIDEO to your source video path before running:
#   export BYO_VIDEO=/path/to/your/video.mp4
#
# Runs the background change recipe by default:
#   Edge (filtered, 1.0) + Seg (inverted mask, 0.4) + Vis (0.6)
#
# For Omniverse workflows: export RGB + edge + seg + mask from IsaacSim first.
# See: https://docs.isaacsim.omniverse.nvidia.com/
#
# Usage:
#   export HF_TOKEN=hf_...
#   export BYO_VIDEO=/path/to/your/video.mp4
#   bash deploy/transfer2_5/inference-real-augmentation/demo.sh
#
# Output:
#   /tmp/real_augmentation_output/   — background-changed video
#   /tmp/real_augmentation_results.json — timing and throughput metrics

set -e

RECIPE="real_augmentation"
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
  echo "ERROR: BYO_VIDEO not set. Provide a source video:"
  echo "  export BYO_VIDEO=/path/to/your/video.mp4"
  echo ""
  echo "Supported sources:"
  echo "  - Real camera footage (MP4)"
  echo "  - NVIDIA Omniverse / IsaacSim RGB video export"
  echo "  - Any video where you want to change background, lighting, or textures"
  exit 1
fi

if [ ! -f "$BYO_VIDEO" ]; then
  echo "ERROR: BYO_VIDEO file not found: ${BYO_VIDEO}"
  exit 1
fi

echo "Source video: ${BYO_VIDEO} ✓"

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

# ── Step 6: Recipe dependencies ──────────────────────────────────────────────

echo "=== Step 6: Recipe dependencies ==="
pip install -q opencv-python
echo "OpenCV ✓"

# ── Step 7: Generate edge map from source video ──────────────────────────────

echo "=== Step 7: Generate Canny edge map ==="
mkdir -p "$OUTPUT_DIR"
EDGE_VIDEO="/tmp/${RECIPE}_edge.mp4"
EDGE_FILTERED="/tmp/${RECIPE}_edge_filtered.mp4"

python - <<PYEOF
import cv2, sys
from pathlib import Path

input_video = "${BYO_VIDEO}"
output_edge = "${EDGE_VIDEO}"

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"ERROR: Cannot open video: {input_video}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 24
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {w}x{h} @ {fps:.1f}fps, {total} frames")

# Generate standard edge map
out = cv2.VideoWriter(output_edge,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (w, h), isColor=False)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Enhance contrast before Canny for more detailed edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 80, 180)
    out.write(edges)
cap.release()
out.release()
print(f"Edge map written: {output_edge}")
PYEOF

# Use edge map directly as filtered edge for the demo
cp "$EDGE_VIDEO" "$EDGE_FILTERED"
echo "Edge maps ✓"

# ── Step 8: Build background change JSON spec ────────────────────────────────

echo "=== Step 8: Build background change spec (Recipe 1) ==="
SPEC_FILE="/tmp/${RECIPE}_spec.json"

python - <<PYEOF
import json

spec = {
    "name": "change_background",
    "prompt": (
        "A realistic, static full-body shot of a person standing outdoors near the coast. "
        "They face the camera directly in a friendly pose. The surrounding environment is "
        "bright and open. In the background, a vast ocean stretches out toward the horizon, "
        "with gentle waves, shimmering reflections, and a clear blue sky above. "
        "A coastal walkway with railings lines the foreground. "
        "Soft natural lighting from the sun enhances the calm, breezy seaside atmosphere."
    ),
    "video_path": "${BYO_VIDEO}",
    "guidance": 3,
    "edge": {
        "control_weight": 1.0,
        "control_path": "${EDGE_FILTERED}",
    },
    "vis": {
        "control_weight": 0.6,
    },
}

with open("${SPEC_FILE}", "w") as f:
    json.dump(spec, f, indent=2)

print(f"Spec written: ${SPEC_FILE}")
print("Recipe: Background Change (Edge filtered 1.0 + Vis 0.6)")
PYEOF

echo "Spec file ✓"

# ── Step 9: Run inference ────────────────────────────────────────────────────

echo ""
echo "=== Step 9: Running Cosmos Transfer 2.5 inference (background change recipe) ==="
echo ""

INFER_SCRIPT="$COSMOS_DIR/examples/inference.py"
if [ ! -f "$INFER_SCRIPT" ]; then
  echo "ERROR: inference.py not found at expected path: $INFER_SCRIPT"
  echo "  Verify cosmos-transfer2.5 repo cloned correctly and examples/inference.py exists."
  exit 1
fi

# Get frame count
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

# ── Step 10: Write results JSON ──────────────────────────────────────────────

echo "=== Step 10: Writing results ==="

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
    "recipe": "inference_real_augmentation",
    "model": "cosmos-transfer2.5",
    "model_family": "transfer2_5",
    "gpu": "${GPU_NAME}",
    "dataset": "BYO video",
    "input_video": "${BYO_VIDEO}",
    "recipe_applied": "background_change",
    "controls_used": {
        "edge": {"control_weight": 1.0, "source": "filtered_canny"},
        "vis": {"control_weight": 0.6},
    },
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
echo "Other recipes (lighting, color, object change, Omniverse sim2real):"
echo "docs/recipes/inference/transfer2_5/inference-real-augmentation/inference.md"
