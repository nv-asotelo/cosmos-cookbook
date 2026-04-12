#!/bin/bash
# demo.sh — IntBot Showcase Cosmos Reason 2
# Runs representative egocentric reasoning tests from the IntBot recipe.
# Uses Cosmos-Reason2-8B on local recipe assets (no external dataset download).
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash deploy/reason2/intbot_showcase/demo.sh
#
# Output: /tmp/intbot_results.json

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COSMOS_REASON2="$HOME/cosmos-reason2"
ASSETS="$REPO_ROOT/docs/recipes/inference/reason2/intbot_showcase/assets"
RESULTS_FILE="/tmp/intbot_results.json"

# ── Pre-flight ──────────────────────────────────────────────────────────────

echo "=== Pre-flight checks ==="

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader
echo ""

VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$VRAM_FREE" -lt 75000 ]; then
  echo "ERROR: Only ${VRAM_FREE} MiB VRAM free. Cosmos-Reason2-8B requires >= 75000 MiB (H100-80GB)."
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

# ── Setup (skip if already done by brev.yaml) ────────────────────────────────

echo "=== Environment ==="
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env" 2>/dev/null || true

if [ ! -d "$COSMOS_REASON2" ]; then
  git clone https://github.com/nvidia-cosmos/cosmos-reason2.git "$COSMOS_REASON2"
  git -C "$COSMOS_REASON2" lfs pull
fi

cd "$COSMOS_REASON2"

if [ ! -d ".venv" ]; then
  uv sync --extra cu128 2>&1 | tail -5
fi
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — check driver version'"
echo "CUDA available ✓"

echo "$HF_TOKEN" | huggingface-cli login --token 2>/dev/null || \
  huggingface-cli login --token "$HF_TOKEN"
echo "HF auth ✓"

# ── Model download ────────────────────────────────────────────────────────────

echo ""
echo "=== Model download (Cosmos-Reason2-8B, ~16 GB) ==="
MODEL_DIR="$COSMOS_REASON2/models/Cosmos-Reason2-8B"
if [ ! -d "$MODEL_DIR" ]; then
  huggingface-cli download nvidia/Cosmos-Reason2-8B \
    --repo-type model \
    --local-dir "$MODEL_DIR"
  echo "Downloaded ✓"
else
  echo "Already present ✓"
fi

# ── Run inference ─────────────────────────────────────────────────────────────

echo ""
echo "=== Running egocentric reasoning tests ==="
echo "Tests: fist-bump gesture, hat trajectory, shared attention"
echo ""

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')

START=$(date +%s%N)

python - <<PYEOF
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, "$COSMOS_REASON2")
os.chdir("$COSMOS_REASON2")

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "$MODEL_DIR"
ASSETS = "$ASSETS"

print(f"Loading Cosmos-Reason2-8B from {MODEL_PATH} ...")
llm = LLM(
    model=MODEL_PATH,
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    limit_mm_per_prompt={"image": 1, "video": 1},
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

# Representative tests from the IntBot recipe
TESTS = [
    {
        "id": "fist-bump",
        "image": "Test1-fistbump.jpeg",
        "prompt": (
            "The camera view is my view as a robot. "
            "Is this person doing a fist bump at me? "
            "Can you predict what is the most likely position her fist will be in the next 2 seconds?"
        ),
        "expected_theme": "fist bump directed at robot",
    },
    {
        "id": "hat-side-throw",
        "image": "Test2-side-throw-away.jpeg",
        "prompt": (
            "The camera view is my view as a robot. "
            "Please estimate the landing location of the hat. "
            "Is the hat getting closer to me or further away from me? "
            "Is there danger for the hat to hit me?"
        ),
        "expected_theme": "hat moving away, no danger",
    },
    {
        "id": "shared-attention",
        "image": "Test4-Two-Person-Handshake.jpeg",
        "prompt": (
            "The camera view is my view as a robot. "
            "Are these two people talking to each other, or are they looking at me? "
            "Should I approach and engage, or wait?"
        ),
        "expected_theme": "social context assessment",
    },
]

results = []
latencies = []

for test in TESTS:
    img_path = os.path.join(ASSETS, test["image"])
    if not os.path.exists(img_path):
        print(f"  SKIP {test['id']}: asset not found at {img_path}")
        results.append({"id": test["id"], "status": "skipped", "reason": "asset not found"})
        continue

    print(f"  Running: {test['id']} ...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": test["prompt"]},
            ],
        }
    ]

    t0 = time.time()
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        mm_data = {}
        if image_inputs:
            mm_data["image"] = image_inputs

        outputs = llm.generate(
            [{"prompt": text, "multi_modal_data": mm_data}],
            sampling_params=sampling_params,
        )
        response = outputs[0].outputs[0].text.strip()
        latency_ms = int((time.time() - t0) * 1000)
        latencies.append(latency_ms)
        results.append({
            "id": test["id"],
            "status": "success",
            "prompt": test["prompt"],
            "response": response,
            "expected_theme": test["expected_theme"],
            "latency_ms": latency_ms,
        })
        print(f"    [{latency_ms} ms] {response[:120]}{'...' if len(response) > 120 else ''}")
    except Exception as e:
        results.append({"id": test["id"], "status": "error", "error": str(e)})
        print(f"    ERROR: {e}")

# Write raw sample results (first 3 already — all tests are sample results)
with open("/tmp/intbot_raw.json", "w") as f:
    json.dump({"results": results, "latencies": latencies}, f, indent=2)

success = sum(1 for r in results if r["status"] == "success")
print(f"\nDone. {success}/{len(TESTS)} tests completed.")
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

try:
    with open("/tmp/intbot_raw.json") as f:
        raw = json.load(f)
    results = raw["results"]
    latencies = raw.get("latencies", [])
except Exception:
    results = []
    latencies = []

total = len(results)
success_items = [r for r in results if r.get("status") == "success"]
n_success = len(success_items)

mean_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0
throughput   = round(n_success / (ELAPSED / 1000), 4) if ELAPSED > 0 and n_success > 0 else 0

# First 3 sample results (all tests qualify)
sample_results = []
for r in results[:3]:
    sample_results.append({
        "id": r.get("id"),
        "status": r.get("status"),
        "expected_theme": r.get("expected_theme"),
        "response_excerpt": r.get("response", "")[:200] if r.get("response") else None,
    })

output = {
    "recipe": "reason2/intbot_showcase",
    "model": "nvidia/Cosmos-Reason2-8B",
    "gpu": GPU_NAME,
    "wall_time_ms": ELAPSED,
    "samples_total": total,
    "samples_success": n_success,
    "mean_latency_ms": mean_latency,
    "throughput_queries_per_sec": throughput,
    "sample_results": sample_results,
}

with open("$RESULTS_FILE", "w") as f:
    json.dump(output, f, indent=2)

print(f"Total tests:     {total}")
print(f"Successful:      {n_success}")
print(f"Wall time:       {ELAPSED} ms")
print(f"Mean latency:    {mean_latency} ms")
print(f"Throughput:      {throughput} queries/sec")

if n_success == 0:
    print("\nWARNING: No tests completed. Check logs above.")
    sys.exit(1)
PYEOF

echo ""
echo "=== Done ==="
echo "Full results: $RESULTS_FILE"
