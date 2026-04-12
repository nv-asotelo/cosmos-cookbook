#!/bin/bash
# demo.sh — Worker Safety Cosmos Reason 2
# Headless inference for Brev or any Linux GPU machine.
# Runs worker_safety.py against pjramg/Safe_Unsafe_Test (loaded via FiftyOne).
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash deploy/reason2/worker_safety/demo.sh
#
# Output: /tmp/worker_safety_results.json

set -e

# Prefer /workspace (Brev default), fall back to BASH_SOURCE-relative path (local dev)
COSMOS_REASON2="${COSMOS_REASON2:-/workspace/cosmos-reason2}"
if [ ! -d "$COSMOS_REASON2" ]; then
  COSMOS_REASON2="$HOME/cosmos-reason2"
fi
COOKBOOK="${COOKBOOK:-/workspace/cosmos-cookbook}"
if [ ! -d "$COOKBOOK" ]; then
  COOKBOOK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
REPO_ROOT="$COOKBOOK"
RESULTS_FILE="/tmp/worker_safety_results.json"

# ── Pre-flight ──────────────────────────────────────────────────────────────

echo "=== Pre-flight checks ==="

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader
echo ""

# VRAM check (require >= 40000 MiB free for Cosmos-Reason2-2B)
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$VRAM_FREE" -lt 40000 ]; then
  echo "ERROR: Only ${VRAM_FREE} MiB VRAM free. Cosmos-Reason2-2B requires >= 40000 MiB."
  exit 1
fi
echo "VRAM: ${VRAM_FREE} MiB free ✓"

if [ -z "$HF_TOKEN" ]; then
  echo ""
  echo "HuggingFace token required. Enter your token (hf_...):"
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

# ── Step 3: Clone cosmos-reason2 ─────────────────────────────────────────────

echo "=== Step 3: cosmos-reason2 ==="
if [ ! -d "$COSMOS_REASON2" ]; then
  git clone https://github.com/nvidia-cosmos/cosmos-reason2.git "$COSMOS_REASON2"
  git -C "$COSMOS_REASON2" lfs pull
  echo "Cloned ✓"
else
  echo "Already present ✓"
fi

# ── Step 4: Python environment ───────────────────────────────────────────────
# (must come before HF login — huggingface-cli lives in the venv)

echo "=== Step 4: Python environment (cu128) ==="
cd "$COSMOS_REASON2"
uv sync --extra cu128 2>&1 | tail -5
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
echo "CUDA available ✓"

# ── Step 5: HF login ─────────────────────────────────────────────────────────

echo "=== Step 5: HuggingFace auth ==="
echo "$HF_TOKEN" | huggingface-cli login --token 2>/dev/null || \
  huggingface-cli login --token "$HF_TOKEN"
echo "Authenticated ✓"

# ── Step 6: Recipe dependencies ──────────────────────────────────────────────

echo "=== Step 6: Recipe dependencies ==="
uv pip install -q -U fiftyone
echo "FiftyOne ✓"

# ── Step 7: Model download ───────────────────────────────────────────────────

echo "=== Step 7: Model download (~4 GB) ==="
MODEL_DIR="$COSMOS_REASON2/models/Cosmos-Reason2-2B"
if [ ! -d "$MODEL_DIR" ]; then
  huggingface-cli download nvidia/Cosmos-Reason2-2B \
    --repo-type model \
    --local-dir "$MODEL_DIR"
  echo "Downloaded ✓"
else
  echo "Already present ✓"
fi

# ── Step 8: Copy recipe script ───────────────────────────────────────────────

echo "=== Step 8: Recipe script ==="
RECIPE="$COOKBOOK/docs/recipes/inference/reason2/worker_safety"
cp "$RECIPE/worker_safety.py" "$COSMOS_REASON2/worker_safety.py"
echo "worker_safety.py in place ✓"

# ── Step 9: Run inference (headless) ─────────────────────────────────────────

echo ""
echo "=== Step 9: Running inference (headless) ==="
echo "Dataset: pjramg/Safe_Unsafe_Test (downloaded via FiftyOne)"
echo "Model: Cosmos-Reason2-2B"
echo ""

cd "$COSMOS_REASON2"

START=$(date +%s%N)

python - <<'PYEOF'
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Patch FiftyOne app launch so it doesn't block headlessly
import fiftyone as fo
_noop = type("S", (), {"wait": lambda self: None})()
fo.launch_app = lambda *a, **kw: _noop

# Execute the recipe script (jupytext-converted notebook)
# worker_safety.py uses exec-style cell execution
exec(open("worker_safety.py").read())

# Export results to JSON
try:
    dataset = fo.load_dataset("pjramg/Safe_Unsafe_Test")
    results = []
    for sample in dataset.iter_samples():
        cosmos_analysis = sample.get_field("cosmos_analysis")
        safety_label = sample.get_field("safety_label")
        results.append({
            "filepath": sample.filepath,
            "ground_truth": sample.get_field("ground_truth").label if sample.get_field("ground_truth") else None,
            "safety_label": safety_label.label if safety_label else None,
            "cosmos_analysis": cosmos_analysis,
            "error": sample.get_field("cosmos_error"),
        })
    with open("/tmp/worker_safety_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results: {len(results)} samples written to /tmp/worker_safety_raw.json")
except Exception as e:
    print(f"WARNING: Could not export FiftyOne results: {e}")
    with open("/tmp/worker_safety_raw.json", "w") as f:
        json.dump([], f)
PYEOF

END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))

# ── Step 10: Assemble structured results JSON ────────────────────────────────

echo ""
echo "=== Step 10: Assembling results ==="

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')

python - <<PYEOF
import json, os, sys

ELAPSED = $ELAPSED
GPU_NAME = "$GPU_NAME"
RAW_FILE = "/tmp/worker_safety_raw.json"
OUT_FILE = "$RESULTS_FILE"

try:
    with open(RAW_FILE) as f:
        raw = json.load(f)
except Exception:
    raw = []

total = len(raw)
success = [r for r in raw if r.get("safety_label") and not r.get("error")]
errors  = [r for r in raw if r.get("error")]
n_success = len(success)
n_errors  = len(errors)

mean_latency = round(ELAPSED / total, 1) if total > 0 else 0
throughput   = round(n_success / (ELAPSED / 1000), 4) if ELAPSED > 0 else 0

# First 3 sample results
sample_results = []
for r in raw[:3]:
    sample_results.append({
        "filepath": r.get("filepath"),
        "ground_truth": r.get("ground_truth"),
        "safety_label": r.get("safety_label"),
        "status": "success" if r.get("safety_label") else "error",
    })

output = {
    "recipe": "reason2/worker_safety",
    "model": "nvidia/Cosmos-Reason2-2B",
    "gpu": GPU_NAME,
    "wall_time_ms": ELAPSED,
    "samples_total": total,
    "samples_success": n_success,
    "mean_latency_ms": mean_latency,
    "throughput_queries_per_sec": throughput,
    "sample_results": sample_results,
}

with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"Total samples:          {total}")
print(f"Successfully classified: {n_success}")
print(f"Errors:                 {n_errors}")
print(f"Wall time:              {ELAPSED} ms")
print(f"Throughput:             {throughput} queries/sec")

if n_success == 0:
    print("\nWARNING: No samples classified. Check /tmp/worker_safety_raw.json for errors.")
    sys.exit(1)
PYEOF

echo ""
echo "=== Done ==="
echo "Full results: $RESULTS_FILE"
