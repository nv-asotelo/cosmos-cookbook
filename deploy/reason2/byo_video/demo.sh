#!/bin/bash
# demo.sh — BYO-Video Inference with Cosmos Reason2
# Headless inference for Brev or any Linux GPU machine.
# Runs Cosmos-Reason2 VLM against a single user-provided MP4 file.
#
# Supported models (auto-selected by VRAM):
#   nvidia/Cosmos-Reason2-8B  — requires >= 80 GB VRAM free
#   nvidia/Cosmos-Reason2-2B  — requires >= 40 GB VRAM free
#
# Usage:
#   export HF_TOKEN=hf_...
#   export BYO_VIDEO=/path/to/your/video.mp4
#   bash deploy/reason2/byo_video/demo.sh
#
# Optional overrides:
#   export MODEL_NAME=nvidia/Cosmos-Reason2-2B  # force a specific model
#
# Output: /tmp/byo_video_reason2_results.json
#
# Environment setup is handled by deploy/shared/brev-env.sh — no manual
# dependency management needed regardless of Brev provider (Nebius, Hyperstack).

set -euo pipefail

# Resolve COOKBOOK root relative to this script so it works regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK="${COOKBOOK:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

RESULTS_FILE="/tmp/byo_video_reason2_results.json"

# ── BYO video gate ────────────────────────────────────────────────────────────

if [ -z "${BYO_VIDEO:-}" ]; then
  echo "ERROR: BYO_VIDEO is not set."
  echo ""
  echo "  Provide the path to your MP4 file before running:"
  echo "    export BYO_VIDEO=/path/to/your/video.mp4"
  exit 1
fi

if [ ! -f "$BYO_VIDEO" ]; then
  echo "ERROR: BYO_VIDEO file not found: ${BYO_VIDEO}"
  exit 1
fi

echo "BYO video: ${BYO_VIDEO} ✓"

# ── Pre-flight: GPU check (fast-fail before any setup) ───────────────────────

echo ""
echo "=== Pre-flight checks ==="

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader
echo ""

VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "VRAM free: ${VRAM_FREE} MiB"

# ── VRAM-based model selection ────────────────────────────────────────────────
# Allow caller to override by pre-setting MODEL_NAME.

if [ -z "${MODEL_NAME:-}" ]; then
  if [ "$VRAM_FREE" -ge 80000 ]; then
    MODEL_NAME="nvidia/Cosmos-Reason2-8B"
    MODEL_DIR_NAME="Cosmos-Reason2-8B"
    echo "VRAM >= 80 GB → selecting Cosmos-Reason2-8B ✓"
  elif [ "$VRAM_FREE" -ge 40000 ]; then
    MODEL_NAME="nvidia/Cosmos-Reason2-2B"
    MODEL_DIR_NAME="Cosmos-Reason2-2B"
    echo "VRAM >= 40 GB → selecting Cosmos-Reason2-2B ✓"
  else
    MODEL_NAME="nvidia/Cosmos-Reason2-2B"
    MODEL_DIR_NAME="Cosmos-Reason2-2B"
    echo "WARNING: VRAM below minimum (40GB). Attempting CR2-2B — may OOM."
    echo "  Free VRAM: ${VRAM_FREE} MiB. Recommended: >= 40000 MiB."
  fi
else
  # Derive MODEL_DIR_NAME from the caller-supplied MODEL_NAME
  MODEL_DIR_NAME="$(basename "$MODEL_NAME")"
  echo "MODEL_NAME override: ${MODEL_NAME}"
fi

echo ""

# ── Environment setup ─────────────────────────────────────────────────────────
# brev-env.sh handles: HOME detection, git-lfs, uv, repo clone, Python venv,
# uv pip installs, and HF auth — consistently across all Brev providers.

export COSMOS_REPO_URL="https://github.com/nvidia-cosmos/cosmos-reason2.git"
export COSMOS_DIR="cosmos-reason2"
export COSMOS_UV_EXTRA="cu128"
# No extra deps needed for BYO-video (no FiftyOne)

source "$COOKBOOK/deploy/shared/brev-env.sh"

# brev-env.sh exports BREV_COSMOS_DIR — use it from here on
COSMOS_REASON2="$BREV_COSMOS_DIR"

# ── Model download ────────────────────────────────────────────────────────────

echo "=== Model download ==="
MODEL_DIR="$COSMOS_REASON2/models/$MODEL_DIR_NAME"
if [ ! -d "$MODEL_DIR" ]; then
  huggingface-cli download "$MODEL_NAME" \
    --repo-type model \
    --local-dir "$MODEL_DIR"
  echo "Downloaded ✓"
else
  echo "Already present at $MODEL_DIR ✓"
fi

# ── Run inference (headless) ──────────────────────────────────────────────────

echo ""
echo "=== Running Cosmos Reason2 inference ==="
echo "Model:  $MODEL_NAME"
echo "Video:  $BYO_VIDEO"
echo "Output: $RESULTS_FILE"
echo ""

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
START=$(date +%s%N)

export MODEL_DIR="$COSMOS_REASON2/models/$MODEL_DIR_NAME"
export REASON2_PROMPT="${REASON2_PROMPT:-Describe this video in detail — what is happening, who or what is present, the setting, and any notable events.}"

python - <<'PYEOF'
import os, torch, transformers, json

# CRITICAL: PyAV backend patch (torchcodec fails — FFmpeg not in system PATH on Hyperstack)
from transformers import video_processing_utils
from transformers.video_utils import load_video as _load_video

def _patched_fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
    if isinstance(video_url_or_urls, list):
        return list(zip(*[
            _patched_fetch_videos(self, x, sample_indices_fn=sample_indices_fn)
            for x in video_url_or_urls
        ]))
    return _load_video(video_url_or_urls, backend="pyav", sample_indices_fn=sample_indices_fn)

video_processing_utils.BaseVideoProcessor.fetch_videos = _patched_fetch_videos

BYO_VIDEO = os.environ["BYO_VIDEO"]
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/Cosmos-Reason2-2B")
MODEL_DIR = os.environ.get("MODEL_DIR", MODEL_NAME)  # local path or HF id

PIXELS_PER_TOKEN = 32 ** 2

model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
)
processor = transformers.Qwen3VLProcessor.from_pretrained(MODEL_DIR)
processor.image_processor.size = {"shortest_edge": 256 * PIXELS_PER_TOKEN, "longest_edge": 4096 * PIXELS_PER_TOKEN}
processor.video_processor.size = {"shortest_edge": 256 * PIXELS_PER_TOKEN, "longest_edge": 4096 * PIXELS_PER_TOKEN}

PROMPT = os.environ.get("REASON2_PROMPT", "Describe this video in detail — what is happening, who or what is present, the setting, and any notable events.")

conversation = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that analyzes videos with precision and detail."}]},
    {"role": "user", "content": [
        {"type": "video", "video": BYO_VIDEO},
        {"type": "text", "text": PROMPT},
    ]},
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt", fps=4,
)
inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
n = inputs["input_ids"].shape[1]
response = processor.decode(out_ids[0][n:], skip_special_tokens=True)

result = {
    "recipe": "reason2/byo_video",
    "model": MODEL_NAME,
    "byo_video": BYO_VIDEO,
    "prompt": PROMPT,
    "response": response,
    "status": "success",
}
with open("/tmp/byo_video_reason2_raw.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
PYEOF

END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))

# ── Assemble structured results JSON ─────────────────────────────────────────

echo ""
echo "=== Assembling results ==="

python - <<PYEOF
import json, sys

ELAPSED = $ELAPSED
GPU_NAME = "$GPU_NAME"
MODEL_NAME = "$MODEL_NAME"
BYO_VIDEO = "$BYO_VIDEO"
VRAM_FREE = $VRAM_FREE
RAW_FILE = "/tmp/byo_video_reason2_raw.json"
OUT_FILE = "$RESULTS_FILE"

try:
    with open(RAW_FILE) as f:
        raw = json.load(f)
except Exception as e:
    print(f"WARNING: Could not read raw results: {e}")
    raw = {}

output = {
    "recipe": "reason2/byo_video",
    "model": MODEL_NAME,
    "gpu": GPU_NAME,
    "vram_free_mib": VRAM_FREE,
    "byo_video": BYO_VIDEO,
    "wall_time_ms": ELAPSED,
    "raw_output": raw,
    "status": "success",
}

with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
PYEOF

echo ""
echo "=== Done ==="
echo ""
echo "Summary:"
echo "  Model:       $MODEL_NAME"
echo "  VRAM free:   ${VRAM_FREE} MiB"
echo "  BYO video:   $BYO_VIDEO"
echo "  Wall time:   ${ELAPSED} ms"
echo "  Output JSON: $RESULTS_FILE"
