#!/bin/bash
# demo.sh — VSS (Video Search and Summarization) Cosmos Reason 2
# SCAFFOLD SCRIPT — validates environment and documents deployment steps.
#
# VSS requires multi-service stack — see setup.md for full deployment.
# https://docs.nvidia.com/vss/latest/content/vss_dep_docker_compose_x86.html
#
# Usage:
#   export NGC_API_KEY=nvapi-...
#   bash deploy/reason2/vss/demo.sh
#
# Output: /tmp/vss_results.json with environment validation and instructions

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_FILE="/tmp/vss_results.json"

# ── Pre-flight ──────────────────────────────────────────────────────────────

echo "=== Pre-flight checks ==="

# GPU check
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader
echo ""

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "GPU count: ${GPU_COUNT}"
echo "VRAM (GPU 0): ${VRAM_FREE} MiB free"

# VSS recommends 8x H100 for full local deployment; 1x H100 for single-GPU profile
if [ "$VRAM_FREE" -lt 75000 ]; then
  echo "WARNING: Only ${VRAM_FREE} MiB VRAM free on GPU 0."
  echo "  Full VSS stack (Cosmos-Reason2-8B) requires 8x H100-80GB."
  echo "  Single-GPU profile requires 1x H100-80GB."
  echo "  Continuing in scaffold mode — see setup.md for hardware requirements."
fi

# NGC API key check
if [ -z "$NGC_API_KEY" ]; then
  echo ""
  echo "WARNING: NGC_API_KEY is not set."
  echo "  VSS requires an NGC API key to pull Docker images from nvcr.io."
  echo "  Set it with: export NGC_API_KEY=nvapi-..."
  echo ""
  NGC_READY="false"
else
  echo "NGC_API_KEY: set ✓"
  NGC_READY="true"
fi

# Docker check
if ! command -v docker &>/dev/null; then
  echo "WARNING: docker not found. VSS deployment requires Docker."
  DOCKER_READY="false"
else
  echo "Docker: $(docker --version) ✓"
  DOCKER_READY="true"
fi

echo ""
echo "=== VSS Deployment Scaffold ==="
echo ""
echo "VSS requires multi-service stack — see setup.md for full deployment."
echo ""
echo "────────────────────────────────────────────────────────────"
echo "OPTION A: Official VSS Brev Launchable (recommended)"
echo "────────────────────────────────────────────────────────────"
echo "  https://docs.nvidia.com/vss/latest/content/cloud_brev.html"
echo ""
echo "────────────────────────────────────────────────────────────"
echo "OPTION B: Multi-GPU local deployment (8x H100)"
echo "────────────────────────────────────────────────────────────"
echo "  1. Set NGC key:"
echo "     export NGC_API_KEY=nvapi-..."
echo ""
echo "  2. Login to NGC registry:"
echo "     echo \$NGC_API_KEY | docker login nvcr.io -u '\$oauthtoken' --password-stdin"
echo ""
echo "  3. Clone the VSS repository:"
echo "     git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git"
echo ""
echo "  4. Follow local deployment guide:"
echo "     https://docs.nvidia.com/vss/latest/content/vss_dep_docker_compose_x86.html#local-deployment"
echo ""
echo "────────────────────────────────────────────────────────────"
echo "OPTION C: Single-GPU local deployment (1x H100)"
echo "────────────────────────────────────────────────────────────"
echo "  Same steps as Option B but use the single-GPU Docker Compose profile:"
echo "  https://docs.nvidia.com/vss/latest/content/vss_dep_docker_compose_x86.html#fully-local-deployment-single-gpu"
echo ""
echo "────────────────────────────────────────────────────────────"
echo "After deployment: REST API usage example"
echo "────────────────────────────────────────────────────────────"
echo ""
cat <<'USAGE'
  import requests

  vss_host = "http://localhost:8100"

  # Upload a video
  with open("/path/to/video.mp4", "rb") as f:
      r = requests.post(vss_host + "/files",
          data={"purpose": "vision", "media_type": "video"},
          files={"file": ("video_file", f)})
  video_id = r.json()["id"]

  # Summarize
  r = requests.post(vss_host + "/summarize", json={
      "id": video_id,
      "prompt": "Write a detailed caption based on the video clip.",
      "model": "cosmos-reason2",
      "max_tokens": 1024,
      "chunk_duration": 20,
  })
  print(r.json()["choices"][0]["message"]["content"])
USAGE

echo ""

# ── Write results JSON ────────────────────────────────────────────────────────

START=$(date +%s%N)
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))

python3 - <<PYEOF
import json

output = {
    "recipe": "reason2/vss",
    "model": "nvidia/Cosmos-Reason2-8B",
    "gpu": "$GPU_NAME",
    "gpu_count": int("$GPU_COUNT"),
    "wall_time_ms": $ELAPSED,
    "samples_total": 0,
    "samples_success": 0,
    "mean_latency_ms": 0,
    "throughput_queries_per_sec": 0,
    "scaffold_mode": True,
    "ngc_api_key_set": "$NGC_READY" == "true",
    "docker_available": "$DOCKER_READY" == "true",
    "deployment_docs": "https://docs.nvidia.com/vss/latest/content/vss_dep_docker_compose_x86.html",
    "brev_launchable": "https://docs.nvidia.com/vss/latest/content/cloud_brev.html",
    "sample_results": [],
    "note": "VSS requires multi-service stack — see setup.md for full deployment.",
}

with open("$RESULTS_FILE", "w") as f:
    json.dump(output, f, indent=2)

print(f"Environment scaffold written to: $RESULTS_FILE")
PYEOF

echo ""
echo "=== Done (scaffold mode) ==="
echo "Results: $RESULTS_FILE"
echo ""
echo "VSS requires multi-service stack — see setup.md for full deployment."
echo "Recipe: $REPO_ROOT/docs/recipes/inference/reason2/vss/inference.md"
