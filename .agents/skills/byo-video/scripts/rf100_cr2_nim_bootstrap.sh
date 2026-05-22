#!/usr/bin/env bash
set -euo pipefail

MODEL_SHORT="${MODEL_SHORT:-cosmos-reason2-8b}"
SERVED_MODEL="${SERVED_MODEL:-nvidia/cosmos-reason2-8b}"
IMAGE="${IMAGE:-nvcr.io/nim/nvidia/cosmos-reason2-8b:latest}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-cosmos-nim}"
FORCE_RESTART="${FORCE_RESTART:-1}"
MAX_WAIT="${MAX_WAIT:-3600}"
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
NIM_TENSOR_PARALLEL_SIZE="${NIM_TENSOR_PARALLEL_SIZE:-2}"

if [ -f /tmp/rf100.env ]; then
  install -m 600 /tmp/rf100.env /tmp/byo_video_nim_credentials.env
  set -a
  # shellcheck disable=SC1091
  . /tmp/byo_video_nim_credentials.env
  set +a
fi

if [ -z "${NGC_API_KEY:-}" ]; then
  echo "ERROR: NGC_API_KEY is required in /tmp/rf100.env or the process environment."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed on this instance."
  exit 1
fi

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl start docker || true
else
  sudo service docker start || true
fi
docker info >/dev/null

mkdir -p "$LOCAL_NIM_CACHE" /tmp/benchmark-runs
chmod 0777 "$LOCAL_NIM_CACHE" || true
chmod +x /tmp/nim_launch.sh /tmp/rf100_brev_launch.sh 2>/dev/null || true

export MODEL="$MODEL_SHORT"
export IMAGE
export PORT
export CONTAINER_NAME
export FORCE_RESTART
export MAX_WAIT
export NIM_CACHE_MODE=host
export LOCAL_NIM_CACHE
export SHM_SIZE="${SHM_SIZE:-64GB}"
export NIM_SERVED_MODEL_NAME="$SERVED_MODEL"
export NIM_TENSOR_PARALLEL_SIZE
export NIM_KVCACHE_PERCENT="${NIM_KVCACHE_PERCENT:-0.9}"
export NIM_ENABLE_KV_CACHE_REUSE="${NIM_ENABLE_KV_CACHE_REUSE:-1}"
export NIM_DISABLE_MM_PREPROCESSOR_CACHE="${NIM_DISABLE_MM_PREPROCESSOR_CACHE:-0}"
export NIM_COMPILATION_CONFIG="${NIM_COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"compile_mm_encoder\":true}}"
export NIM_MAX_NUM_BATCHED_TOKENS="${NIM_MAX_NUM_BATCHED_TOKENS:-8192}"
export NIM_MAX_NUM_SEQS="${NIM_MAX_NUM_SEQS:-256}"
export NIM_STREAM_INTERVAL="${NIM_STREAM_INTERVAL:-10}"
export NIM_ENABLE_CHUNKED_PREFILL="${NIM_ENABLE_CHUNKED_PREFILL:-1}"
export NIM_DISABLE_CHUNKED_MM_INPUT="${NIM_DISABLE_CHUNKED_MM_INPUT:-0}"
export NIM_VIDEO_PRUNING_RATE="${NIM_VIDEO_PRUNING_RATE:-0}"
export NIM_CONN_FAST_HTTP="${NIM_CONN_FAST_HTTP:-1}"
export NIM_CONN_READ_BUFSIZE="${NIM_CONN_READ_BUFSIZE:-4194304}"
export NIM_CONN_TRUST_ENV="${NIM_CONN_TRUST_ENV:-1}"

bash /tmp/nim_launch.sh

python3 - <<'PY'
import json
import subprocess
import time
import urllib.request

def sh(cmd):
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()

metadata = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "models": json.loads(urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10).read().decode()),
    "version": None,
    "docker_image": sh("docker inspect cosmos-nim --format '{{.Config.Image}}'"),
    "docker_status": sh("docker inspect cosmos-nim --format '{{.State.Status}} {{.State.StartedAt}}'"),
    "gpu": sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader"),
}
try:
    metadata["version"] = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/v1/version", timeout=10).read().decode())
except Exception as exc:
    metadata["version_error"] = str(exc)
with open("/tmp/rf100_cr2_nim_ready.json", "w", encoding="utf-8") as fh:
    json.dump(metadata, fh, indent=2, sort_keys=True)
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
