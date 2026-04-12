#!/bin/bash
# demo.sh — BioTrove Moth Augmentation with Cosmos Transfer 2.5
# Headless inference for Brev or any Linux GPU machine.
# No browser or JupyterLab required.
#
# Downloads 20 moth images from pjramg/moth_biotrove (public HF dataset),
# converts them to videos, generates Canny edge maps, runs Cosmos Transfer 2.5
# inference, and writes timing metrics.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash deploy/transfer2_5/biotrove_augmentation/demo.sh
#
# Output:
#   /tmp/biotrove_augmentation_output/   — generated agricultural moth scene videos
#   /tmp/biotrove_augmentation_results.json — timing and throughput metrics

set -e

RECIPE="biotrove_augmentation"
# Prefer /workspace (Brev default), fall back to $HOME (local dev)
COSMOS_DIR="${COSMOS_DIR:-/workspace/cosmos-transfer2_5}"
if [ ! -d "$COSMOS_DIR" ]; then
  COSMOS_DIR="$HOME/cosmos-transfer2.5"
fi
OUTPUT_DIR="/tmp/${RECIPE}_output"
RESULTS_JSON="/tmp/${RECIPE}_results.json"
MAX_SAMPLES=20

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
pip install -q fiftyone opencv-python
echo "FiftyOne + OpenCV ✓"

# ── Step 7: Download BioTrove moth subset ────────────────────────────────────

echo "=== Step 7: Download BioTrove moth subset (${MAX_SAMPLES} images, public dataset) ==="
IMAGES_DIR="/tmp/${RECIPE}_images"
mkdir -p "$IMAGES_DIR"
mkdir -p "$OUTPUT_DIR"

python - <<PYEOF
import os, sys

try:
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh

    print(f"Downloading up to ${MAX_SAMPLES} moth images from pjramg/moth_biotrove ...")
    dataset = fouh.load_from_hub(
        "pjramg/moth_biotrove",
        persistent=True,
        overwrite=True,
        max_samples=${MAX_SAMPLES},
    )
    print(f"Downloaded {len(dataset)} samples")

    # Copy images to working directory
    import shutil
    from pathlib import Path
    images_dir = Path("${IMAGES_DIR}")
    count = 0
    for sample in dataset:
        src = Path(sample.filepath)
        if src.exists() and src.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dst = images_dir / f"moth_{count:04d}{src.suffix}"
            shutil.copy2(src, dst)
            count += 1
    print(f"Copied {count} images to ${IMAGES_DIR}")
except Exception as e:
    print(f"ERROR downloading dataset: {e}")
    sys.exit(1)
PYEOF

echo "Dataset ready ✓"

# ── Step 8: Convert images to videos ────────────────────────────────────────

echo "=== Step 8: Convert images to short videos ==="
VIDEOS_DIR="/tmp/${RECIPE}_videos"
EDGES_DIR="/tmp/${RECIPE}_edges"
mkdir -p "$VIDEOS_DIR" "$EDGES_DIR"

python - <<PYEOF
import subprocess, sys
from pathlib import Path

images_dir = Path("${IMAGES_DIR}")
videos_dir = Path("${VIDEOS_DIR}")

imgs = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png")))
print(f"Converting {len(imgs)} images to video clips ...")

for img in imgs:
    out = videos_dir / f"{img.stem}.mp4"
    cmd = [
        "ffmpeg", "-y", "-loop", "1",
        "-i", str(img), "-t", "1",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  WARNING: Failed to convert {img.name}")

done = list(videos_dir.glob("*.mp4"))
print(f"Created {len(done)} video clips")
if len(done) == 0:
    print("ERROR: No videos created.")
    sys.exit(1)
PYEOF

echo "Video conversion ✓"

# ── Step 9: Generate Canny edge maps ────────────────────────────────────────

echo "=== Step 9: Generate Canny edge maps ==="

python - <<PYEOF
import cv2, sys
from pathlib import Path

videos_dir = Path("${VIDEOS_DIR}")
edges_dir = Path("${EDGES_DIR}")

vids = list(videos_dir.glob("*.mp4"))
print(f"Generating edge maps for {len(vids)} videos ...")

def make_edge_video(input_video, output_video):
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_video),
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

for vid in vids:
    edge_out = edges_dir / vid.name
    make_edge_video(vid, edge_out)

done = list(edges_dir.glob("*.mp4"))
print(f"Generated {len(done)} edge map videos")
PYEOF

echo "Edge maps ✓"

# ── Step 10: Build JSON spec files ──────────────────────────────────────────

echo "=== Step 10: Build JSON spec files ==="
SPECS_DIR="/tmp/${RECIPE}_specs"
mkdir -p "$SPECS_DIR"

python - <<PYEOF
import json, sys
from pathlib import Path

videos_dir = Path("${VIDEOS_DIR}")
edges_dir = Path("${EDGES_DIR}")
specs_dir = Path("${SPECS_DIR}")

MOTH_PROMPT = (
    "A moth resting on green agricultural foliage in a sunlit field. "
    "Photorealistic outdoor scene with natural lighting, realistic plant textures, "
    "and authentic field background. The moth's wing structure and markings are clearly visible. "
    "High-quality nature photography."
)
NEG_PROMPT = (
    "cartoon, illustration, synthetic, CG, low quality, blurry, overexposed, "
    "indoor, laboratory, white background, studio."
)

vids = sorted(videos_dir.glob("*.mp4"))
count = 0
for vid in vids:
    edge = edges_dir / vid.name
    if not edge.exists():
        continue
    spec = {
        "name": vid.stem,
        "prompt": MOTH_PROMPT,
        "negative_prompt": NEG_PROMPT,
        "video_path": str(vid.resolve()),
        "guidance": 7.0,
        "resolution": [704, 1280],
        "num_steps": 35,
        "edge": {
            "control_weight": 1.0,
            "control_path": str(edge.resolve()),
        },
    }
    spec_path = specs_dir / f"{vid.stem}.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    count += 1

print(f"Created {count} JSON spec files")
PYEOF

echo "JSON specs ✓"

# ── Step 11: Run inference ───────────────────────────────────────────────────

echo ""
echo "=== Step 11: Running Cosmos Transfer 2.5 inference ==="
echo "Processing up to ${MAX_SAMPLES} videos. This may take 30-60 minutes."
echo ""

INFER_SCRIPT="$COSMOS_DIR/examples/inference.py"
if [ ! -f "$INFER_SCRIPT" ]; then
  # Fallback path
  INFER_SCRIPT=$(find "$COSMOS_DIR" -name "inference.py" | head -1)
fi

START_NS=$(date +%s%N)

FRAMES_TOTAL=0
PROCESSED=0

for SPEC_FILE in "$SPECS_DIR"/*.json; do
  STEM=$(basename "$SPEC_FILE" .json)
  echo "  Generating: $STEM"
  python "$INFER_SCRIPT" -i "$SPEC_FILE" -o "$OUTPUT_DIR" 2>&1 | tail -3
  PROCESSED=$((PROCESSED + 1))
  # Each 1-second clip at 24fps = 24 frames
  FRAMES_TOTAL=$((FRAMES_TOTAL + 24))
done

END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

echo ""
echo "Inference complete: ${PROCESSED} videos generated"

# ── Step 12: Write results JSON ──────────────────────────────────────────────

echo "=== Step 12: Writing results ==="

python - <<PYEOF
import json, os
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
generated = list(output_dir.glob("*.mp4"))

elapsed_ms = ${ELAPSED_MS}
frames_total = ${FRAMES_TOTAL}
processed = ${PROCESSED}

time_per_frame = round(elapsed_ms / frames_total, 2) if frames_total > 0 else 0
throughput_fps = round(frames_total / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else 0

result = {
    "recipe": "biotrove_augmentation",
    "model": "cosmos-transfer2.5",
    "model_family": "transfer2_5",
    "gpu": "${GPU_NAME}",
    "dataset": "pjramg/moth_biotrove",
    "wall_time_ms": elapsed_ms,
    "samples_processed": processed,
    "frames_generated_total": frames_total,
    "time_per_frame_ms": time_per_frame,
    "throughput_fps": throughput_fps,
    "output_dir": str(output_dir),
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
