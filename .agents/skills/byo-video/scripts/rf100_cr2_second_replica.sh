#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-cosmos-nim-gpu1}"
PORT="${PORT:-8001}"
GPU_DEVICE="${GPU_DEVICE:-1}"
IMAGE="${IMAGE:-nvcr.io/nim/nvidia/cosmos-reason2-8b:latest}"
SERVED_MODEL="${SERVED_MODEL:-nvidia/cosmos-reason2-8b}"
CACHE_DIR="${CACHE_DIR:-/ephemeral/nim-cache}"

if [ -f /tmp/rf100.env ]; then
  set -a
  # shellcheck disable=SC1091
  . /tmp/rf100.env
  set +a
fi

if [ -z "${NGC_API_KEY:-}" ]; then
  echo "ERROR: NGC_API_KEY is required in /tmp/rf100.env or environment."
  exit 1
fi

mkdir -p "$CACHE_DIR"
chmod 0777 "$CACHE_DIR" || true
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  --ipc host \
  --shm-size=64GB \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NGC_API_KEY \
  -e NIM_SERVED_MODEL_NAME="$SERVED_MODEL" \
  -e NIM_KVCACHE_PERCENT=0.9 \
  -e NIM_ENABLE_KV_CACHE_REUSE=1 \
  -e NIM_DISABLE_MM_PREPROCESSOR_CACHE=0 \
  -e 'NIM_COMPILATION_CONFIG={"cudagraph_mode":"FULL_AND_PIECEWISE","compile_mm_encoder":true}' \
  -e NIM_MAX_NUM_BATCHED_TOKENS=8192 \
  -e NIM_MAX_NUM_SEQS=256 \
  -e NIM_STREAM_INTERVAL=10 \
  -e NIM_ENABLE_CHUNKED_PREFILL=1 \
  -e NIM_DISABLE_CHUNKED_MM_INPUT=0 \
  -e NIM_VIDEO_PRUNING_RATE=0 \
  -e NIM_CONN_FAST_HTTP=1 \
  -e NIM_CONN_READ_BUFSIZE=4194304 \
  -e NIM_CONN_TRUST_ENV=1 \
  -v "$CACHE_DIR:/opt/nim/.cache" \
  -p "${PORT}:8000" \
  "$IMAGE"

docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
