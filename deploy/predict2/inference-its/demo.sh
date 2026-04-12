#!/bin/bash
# demo.sh — ITS Image Generation with Cosmos Predict 2
# Headless inference for Brev or any Linux GPU machine.
# No browser or JupyterLab required.
#
# Generates 5 representative ITS (Intelligent Transportation System) images
# from hardcoded prompts covering diverse camera angles, weather conditions,
# and traffic scenarios. No input data required.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash deploy/predict2/inference-its/demo.sh
#
# Output:
#   /tmp/its_output/          — 5 generated ITS images (PNG)
#   /tmp/its_results.json     — timing and throughput metrics

set -e

RECIPE="its"
COSMOS_DIR="$HOME/cosmos-predict2"
OUTPUT_DIR="/tmp/${RECIPE}_output"
RESULTS_JSON="/tmp/${RECIPE}_results.json"

# ── Pre-flight ──────────────────────────────────────────────────────────────

echo "=== Pre-flight checks ==="

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader
echo ""

# VRAM check (require >= 70000 MiB for Cosmos Predict 2)
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$VRAM_FREE" -lt 70000 ]; then
  echo "ERROR: Only ${VRAM_FREE} MiB VRAM free. Cosmos Predict 2 requires >= 70000 MiB."
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
if ! command -v git-lfs &>/dev/null; then
  sudo apt-get update -q
  sudo apt-get install -y -q curl git git-lfs
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

# ── Step 3: Clone cosmos-predict2 ────────────────────────────────────────────

echo "=== Step 3: cosmos-predict2 ==="
if [ ! -d "$COSMOS_DIR" ]; then
  git clone https://github.com/nvidia-cosmos/cosmos-predict2.git "$COSMOS_DIR"
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

# ── Step 6: Prepare output directory ────────────────────────────────────────

echo "=== Step 6: Prepare output directory ==="
mkdir -p "$OUTPUT_DIR"
echo "Output directory: ${OUTPUT_DIR} ✓"

# ── Step 7: Define ITS prompts ───────────────────────────────────────────────
# 5 representative prompts covering diverse ITS scenarios:
# camera angles: front dashboard, roadside, top-down/overhead
# weather: night, snow, fog, rain, clear daylight
# objects: motorcyclists, bicycles, buses, pedestrians, cars

echo "=== Step 7: ITS prompts defined ==="
echo "5 prompts: night urban intersection, snowy highway, foggy crossroads,"
echo "           rainy mixed traffic, clear suburban intersection"
echo ""

# ── Step 8: Run text-to-image inference ──────────────────────────────────────

echo "=== Step 8: Running Cosmos Predict 2 text-to-image inference ==="
echo "Generating 5 ITS images. Each may take 2-5 minutes on A100/H100."
echo ""

TOTAL_START_NS=$(date +%s%N)
IMAGES_GENERATED=0

# Discover the inference entry point
T2I_SCRIPT=$(find "$COSMOS_DIR" -name "inference_text2image.py" | head -1)
if [ -z "$T2I_SCRIPT" ]; then
  # Fallback: look for any text2image script
  T2I_SCRIPT=$(find "$COSMOS_DIR" -name "*text2image*" -name "*.py" | head -1)
fi
if [ -z "$T2I_SCRIPT" ]; then
  T2I_SCRIPT=$(find "$COSMOS_DIR" -name "inference.py" | head -1)
fi

echo "Inference script: ${T2I_SCRIPT}"
echo ""

# ── Prompt 1: Night urban intersection (front dashboard) ────────────────────

PROMPT_1="A nighttime street view from inside a vehicle shows a motorcyclist navigating through traffic at a busy urban intersection, with cars and streetlights illuminating the wet road. A bicycle leans against a lamppost near the sidewalk. Traffic lights glow red and green overhead, casting colored reflections on the wet asphalt. Pedestrians wait at the crosswalk. The scene captures the complexity of nighttime urban ITS conditions."

echo "  [1/5] Night urban intersection (dashboard view) ..."
IMG_1="${OUTPUT_DIR}/its_01_night_dashboard.png"
IMG_START_NS=$(date +%s%N)

python "$T2I_SCRIPT" \
  --prompt "$PROMPT_1" \
  --output_path "$IMG_1" \
  --seed 42 2>&1 | tail -3

IMG_END_NS=$(date +%s%N)
IMG_1_MS=$(( (IMG_END_NS - IMG_START_NS) / 1000000 ))
echo "  Generated in ${IMG_1_MS}ms ✓"
IMAGES_GENERATED=$((IMAGES_GENERATED + 1))

# ── Prompt 2: Snowy highway (roadside camera) ────────────────────────────────

PROMPT_2="A static roadside camera captures a snowy highway scene during daytime with heavy snow falling. Multiple vehicles including cars and a bus navigate carefully through snow-covered lanes with visible tire tracks. A cyclist rides cautiously on the right shoulder wearing reflective gear. Road markings are partially obscured by snow. Overcast sky with flat diffused lighting typical of winter ITS data collection conditions."

echo "  [2/5] Snowy highway (roadside camera) ..."
IMG_2="${OUTPUT_DIR}/its_02_snow_roadside.png"
IMG_START_NS=$(date +%s%N)

python "$T2I_SCRIPT" \
  --prompt "$PROMPT_2" \
  --output_path "$IMG_2" \
  --seed 123 2>&1 | tail -3

IMG_END_NS=$(date +%s%N)
IMG_2_MS=$(( (IMG_END_NS - IMG_START_NS) / 1000000 ))
echo "  Generated in ${IMG_2_MS}ms ✓"
IMAGES_GENERATED=$((IMAGES_GENERATED + 1))

# ── Prompt 3: Foggy intersection (top-down view) ─────────────────────────────

PROMPT_3="A fixed overhead camera records a foggy urban crossroads with severely reduced visibility. Depth-based haze progressively obscures distant vehicles and pedestrians. Buses and cars with headlights on navigate the intersection slowly. Pedestrians in dark clothing cross a marked crosswalk, their forms partially diffused by the fog. Traffic signals visible overhead. Classic adverse weather ITS challenge scenario."

echo "  [3/5] Foggy intersection (top-down view) ..."
IMG_3="${OUTPUT_DIR}/its_03_fog_overhead.png"
IMG_START_NS=$(date +%s%N)

python "$T2I_SCRIPT" \
  --prompt "$PROMPT_3" \
  --output_path "$IMG_3" \
  --seed 256 2>&1 | tail -3

IMG_END_NS=$(date +%s%N)
IMG_3_MS=$(( (IMG_END_NS - IMG_START_NS) / 1000000 ))
echo "  Generated in ${IMG_3_MS}ms ✓"
IMAGES_GENERATED=$((IMAGES_GENERATED + 1))

# ── Prompt 4: Rainy mixed traffic (front dashboard) ──────────────────────────

PROMPT_4="A front dashboard camera view shows heavy rain streaking across the windshield of a moving vehicle on a busy urban road. Rain streaks overlay the scene, puddles on the road reflect streetlights and traffic signals. Multiple cyclists in rain gear and pedestrians with umbrellas share the road with cars and motorcycles. Wet road sheen creates mirror-like reflections. Challenging rain ITS conditions with mixed road users."

echo "  [4/5] Rainy mixed traffic (dashboard view) ..."
IMG_4="${OUTPUT_DIR}/its_04_rain_dashboard.png"
IMG_START_NS=$(date +%s%N)

python "$T2I_SCRIPT" \
  --prompt "$PROMPT_4" \
  --output_path "$IMG_4" \
  --seed 512 2>&1 | tail -3

IMG_END_NS=$(date +%s%N)
IMG_4_MS=$(( (IMG_END_NS - IMG_START_NS) / 1000000 ))
echo "  Generated in ${IMG_4_MS}ms ✓"
IMAGES_GENERATED=$((IMAGES_GENERATED + 1))

# ── Prompt 5: Clear daylight suburban intersection (overhead) ────────────────

PROMPT_5="A fixed overhead camera records a clear daylight suburban intersection during mid-morning with balanced natural lighting and crisp shadows. Cars, a delivery truck, and multiple bicycles navigate an organized multi-lane intersection. Pedestrians cross on marked crosswalks. Clear lane markings and traffic signals are fully visible. Stop lines and road signage clearly legible. Ideal clear-weather ITS baseline conditions."

echo "  [5/5] Clear daylight suburban intersection (overhead camera) ..."
IMG_5="${OUTPUT_DIR}/its_05_clear_overhead.png"
IMG_START_NS=$(date +%s%N)

python "$T2I_SCRIPT" \
  --prompt "$PROMPT_5" \
  --output_path "$IMG_5" \
  --seed 1024 2>&1 | tail -3

IMG_END_NS=$(date +%s%N)
IMG_5_MS=$(( (IMG_END_NS - IMG_START_NS) / 1000000 ))
echo "  Generated in ${IMG_5_MS}ms ✓"
IMAGES_GENERATED=$((IMAGES_GENERATED + 1))

TOTAL_END_NS=$(date +%s%N)
TOTAL_ELAPSED_MS=$(( (TOTAL_END_NS - TOTAL_START_NS) / 1000000 ))

echo ""
echo "All ${IMAGES_GENERATED} images generated"

# ── Step 9: Write results JSON ───────────────────────────────────────────────

echo "=== Step 9: Writing results ==="

PROMPTS_JSON="[\"Night urban intersection (dashboard view)\", \"Snowy highway (roadside camera)\", \"Foggy intersection (top-down view)\", \"Rainy mixed traffic (dashboard view)\", \"Clear daylight suburban intersection (overhead camera)\"]"

python - <<PYEOF
import json, os
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
images = list(output_dir.glob("*.png"))

elapsed_ms = ${TOTAL_ELAPSED_MS}
images_generated = ${IMAGES_GENERATED}

time_per_image = round(elapsed_ms / images_generated, 2) if images_generated > 0 else 0
throughput_per_min = round((images_generated / elapsed_ms) * 60000, 2) if elapsed_ms > 0 else 0

result = {
    "recipe": "inference_its",
    "model": "cosmos-predict2",
    "model_family": "predict2",
    "gpu": "${GPU_NAME}",
    "prompts_used": [
        "Night urban intersection — motorcyclist, bicycles, wet road (front dashboard view)",
        "Snowy highway — cyclist on shoulder, bus, cars in snow (roadside camera)",
        "Foggy crossroads — buses, pedestrians, severe visibility reduction (top-down view)",
        "Rainy mixed traffic — cyclists, pedestrians, rain streaks (dashboard view)",
        "Clear daylight suburban intersection — bicycles, delivery truck, crosswalks (overhead camera)",
    ],
    "wall_time_ms": elapsed_ms,
    "images_generated": images_generated,
    "time_per_image_ms": time_per_image,
    "throughput_images_per_min": throughput_per_min,
    "output_dir": str(output_dir),
    "output_files": [f.name for f in sorted(images)],
    "status": "success" if images_generated == 5 else "partial",
}

with open("${RESULTS_JSON}", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
PYEOF

echo ""
echo "=== Done ==="
echo "Generated images: ${OUTPUT_DIR}"
echo "Results JSON:     ${RESULTS_JSON}"
echo ""
echo "Recipe: docs/recipes/inference/predict2/inference-its/inference.md"
echo "Evaluation datasets: ACDC, SUTD, DAWN (see inference.md for links)"
