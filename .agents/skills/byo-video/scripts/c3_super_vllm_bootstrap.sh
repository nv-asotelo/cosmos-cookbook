#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-nvidia/Cosmos3-Super-Reasoner}"
MODEL_DIR="${MODEL_DIR:-/opt/checkpoints/cosmos3-super-reasoner-may17}"
SERVED_MODEL="${SERVED_MODEL:-nvidia/Cosmos3-Super-Reasoner-May17}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-c3-super-vllm}"
HF_CREDENTIAL_FILE="${HF_CREDENTIAL_FILE:-/tmp/hf_token.env}"
HF_HOME="${HF_HOME:-/tmp/hf-cache}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
LOG_JSON="${LOG_JSON:-/tmp/c3_super_vllm_status.json}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MM_LIMITS_JSON="${MM_LIMITS_JSON:-{\"image\":1}}"
export MODEL_REPO MODEL_DIR SERVED_MODEL PORT CONTAINER_NAME HF_HOME

if [ ! -f "$HF_CREDENTIAL_FILE" ]; then
  echo "ERROR: $HF_CREDENTIAL_FILE is missing."
  exit 1
fi
chmod 600 "$HF_CREDENTIAL_FILE"
set -a
# shellcheck disable=SC1090
. "$HF_CREDENTIAL_FILE"
set +a

if [ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]; then
  echo "ERROR: HF token env is missing."
  exit 1
fi
export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl start docker || true
else
  sudo service docker start || true
fi
DOCKER_BIN="${DOCKER_BIN:-docker}"
if ! $DOCKER_BIN info >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
  DOCKER_BIN="sudo docker"
fi
$DOCKER_BIN info >/dev/null

mkdir -p "$MODEL_DIR" "$HF_HOME" /tmp/c3-super-smoke
chmod 0777 "$HF_HOME" || true

echo "[c3-super] installing bootstrap dependencies"
if ! python3 - <<'PY' >/tmp/c3_super_import_check.log 2>&1
import huggingface_hub, PIL, requests
PY
then
  BOOTSTRAP_VENV="${BOOTSTRAP_VENV:-/tmp/c3-super-bootstrap-venv}"
  python3 -m venv "$BOOTSTRAP_VENV"
  "$BOOTSTRAP_VENV/bin/python" -m pip install -U pip >/tmp/c3_super_pip.log 2>&1
  "$BOOTSTRAP_VENV/bin/python" -m pip install -U "huggingface_hub[cli]" pillow requests >>/tmp/c3_super_pip.log 2>&1
  PYTHON_BIN="$BOOTSTRAP_VENV/bin/python"
else
  PYTHON_BIN=python3
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[c3-super] downloading $MODEL_REPO to $MODEL_DIR"
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    local_dir=os.environ["MODEL_DIR"],
    local_dir_use_symlinks=False,
    token=os.environ["HF_TOKEN"],
    resume_download=True,
)
PY
else
  echo "[c3-super] checkpoint already staged at $MODEL_DIR"
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
img = Image.new("RGB", (384, 256), "white")
draw = ImageDraw.Draw(img)
draw.rectangle((40, 60, 150, 190), outline="red", width=8)
draw.ellipse((220, 70, 330, 180), outline="blue", width=8)
draw.text((44, 28), "C3 Super smoke image", fill="black")
Path("/tmp/c3-super-smoke/probe.jpg").parent.mkdir(parents=True, exist_ok=True)
img.save("/tmp/c3-super-smoke/probe.jpg", quality=92)
PY

$DOCKER_BIN rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
$DOCKER_BIN pull "$VLLM_IMAGE"
$DOCKER_BIN run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$MODEL_DIR:$MODEL_DIR:ro" \
  -p "${PORT}:8000" \
  "$VLLM_IMAGE" \
    "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --limit-mm-per-prompt "$MM_LIMITS_JSON"

echo "[c3-super] waiting for /v1/models on port $PORT"
ready=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/tmp/c3_super_models.json; then
    ready=1
    break
  fi
  sleep 10
done
if [ "$ready" != "1" ]; then
  echo "ERROR: vLLM did not become ready."
  $DOCKER_BIN logs --tail 200 "$CONTAINER_NAME" || true
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import base64
import json
import os
import time
from pathlib import Path
import requests

port = os.environ.get("PORT", "8000")
model = os.environ.get("SERVED_MODEL", "nvidia/Cosmos3-Super-Reasoner-May17")
image = Path("/tmp/c3-super-smoke/probe.jpg")
b64 = base64.b64encode(image.read_bytes()).decode("ascii")
payload = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "Briefly describe the visible shapes and colors. Return one sentence."},
        ],
    }],
    "max_tokens": 80,
    "temperature": 0.0,
}
t0 = time.time()
resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=120)
elapsed = round(time.time() - t0, 2)
out = {
    "status": "success" if resp.ok else "failed",
    "http_status": resp.status_code,
    "elapsed_s": elapsed,
    "model": model,
}
try:
    data = resp.json()
    out["response_head"] = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")[:300]
except Exception:
    out["body_head"] = resp.text[:300]
Path("/tmp/c3_super_smoke_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
if not resp.ok:
    raise SystemExit(1)
PY

"$PYTHON_BIN" - <<'PY'
import json, os, subprocess, time, urllib.request
def sh(cmd):
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
port = os.environ.get("PORT", "8000")
meta = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_repo": os.environ.get("MODEL_REPO", ""),
    "model_dir": os.environ.get("MODEL_DIR", ""),
    "served_model": os.environ.get("SERVED_MODEL", ""),
    "models": json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10).read().decode()),
    "gpu": sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader"),
    "container": sh(os.environ.get("DOCKER_PS_CMD", "docker ps --filter name=\"$CONTAINER_NAME\" --format '{{.Names}} {{.Image}} {{.Status}}'")),
}
open("/tmp/c3_super_vllm_status.json", "w", encoding="utf-8").write(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
PY
