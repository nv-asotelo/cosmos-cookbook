#!/bin/bash
# NIM launch script for any BYO-video supported VLM NIM on any host.
# Run as: NGC_API_KEY=... bash nim_launch.sh
# Backward-compatible positional args are still accepted, but automation should
# pass credentials through env or a chmod-0600 NIM_CREDENTIAL_FILE.
# Env overrides:
#   MODEL            — short id from nim_catalog.py (default cosmos3-reasoner-super)
#   IMAGE            — full image override (otherwise resolved from nim_catalog.py)
#   PORT             — host port (default 8000)
#   CONTAINER_NAME   — docker container name (default cosmos-nim)
#   LOCAL_NIM_CACHE  — host cache dir (default $HOME/.cache/nim)
#   NIM_CACHE_MODE   — internal | host (default internal; host bind-mounts LOCAL_NIM_CACHE)
#   MAX_WAIT         — seconds to wait for /v1/models (default 1800; first-pull + first-load can be long)
#   SHM_SIZE         — --shm-size value (default 32GB)
#   NIM_EXTRA_ENV    — comma-separated KEY=VALUE pairs forwarded to docker run
#   FORCE_RESTART    — 1/true forces replacement even if existing container is healthy
#   NIM_CREDENTIAL_FILE — env file sourced before launch (default /tmp/byo_video_nim_credentials.env)
#   RESOLVE_ONLY     — 1/true prints resolved MODEL/IMAGE/env and exits without Docker
#
# Output: detached container on $PORT serving OpenAI-compatible API at http://localhost:$PORT/v1.
# Logs streamed to /tmp/nim_launch.log; container logs via `docker logs $CONTAINER_NAME`.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIM_CREDENTIAL_FILE="${NIM_CREDENTIAL_FILE:-/tmp/byo_video_nim_credentials.env}"
if [ -f "$NIM_CREDENTIAL_FILE" ]; then
    _cred_mode=""
    if stat -c %a "$NIM_CREDENTIAL_FILE" >/dev/null 2>&1; then
        _cred_mode="$(stat -c %a "$NIM_CREDENTIAL_FILE")"
    fi
    if [ -n "$_cred_mode" ] && [ "$_cred_mode" != "600" ] && [ "$_cred_mode" != "400" ]; then
        echo "ERROR: $NIM_CREDENTIAL_FILE must be chmod 0600 or 0400 before sourcing"
        exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    . "$NIM_CREDENTIAL_FILE"
    set +a
fi
NGC_API_KEY="${1:-${NGC_API_KEY:-}}"
HF_TOKEN="${2:-${HF_TOKEN:-}}"

MODEL="${MODEL:-cosmos3-reasoner-super}"
IMAGE="${IMAGE:-}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-cosmos-nim}"
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
NIM_CACHE_MODE="${NIM_CACHE_MODE:-internal}"
NIM_EXTRA_ENV="${NIM_EXTRA_ENV:-}"
MAX_WAIT="${MAX_WAIT:-1800}"
SHM_SIZE="${SHM_SIZE:-32GB}"
FORCE_RESTART="${FORCE_RESTART:-0}"

_truthy() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

if [ -z "$IMAGE" ]; then
    _resolved="$(
        MODEL_TO_RESOLVE="$MODEL" \
        NIM_LAUNCH_SCRIPT_DIR="$SCRIPT_DIR" \
        python3 - <<'PY' 2>/dev/null || true
import os
import re
import shlex
import sys

script_dir = os.environ.get("NIM_LAUNCH_SCRIPT_DIR", "")
for path in (script_dir, "/tmp", os.getcwd()):
    if path and path not in sys.path:
        sys.path.insert(0, path)

try:
    from nim_catalog import KNOWN_VLM_NIMS  # type: ignore
except Exception:
    sys.exit(0)

requested = os.environ.get("MODEL_TO_RESOLVE", "").lower()
if requested.startswith("nvcr.io/nim/"):
    requested = requested[len("nvcr.io/nim/"):]

def _strip_tag(value: str) -> str:
    return re.sub(r":[^/:]+$", "", value)

requested = _strip_tag(requested)
requested_leaf = requested.split("/")[-1]
requested_size = os.environ.get("NIM_MODEL_SIZE", "").lower()
if requested_leaf == "cosmos3-reasoner" and requested_size in {"nano", "super"}:
    requested = f"cosmos3-reasoner-{requested_size}"
    requested_leaf = requested

def _image_key(image: str) -> str:
    key = image.lower()
    if key.startswith("nvcr.io/nim/"):
        key = key[len("nvcr.io/nim/"):]
    return _strip_tag(key)

for nim in KNOWN_VLM_NIMS:
    image_key = _image_key(getattr(nim, "image", ""))
    tokens = {
        getattr(nim, "short_id", "").lower(),
        getattr(nim, "served_model_id", "").lower(),
        image_key,
        image_key.split("/")[-1],
    }
    if requested in tokens or requested_leaf in tokens:
        print("MODEL=" + shlex.quote(getattr(nim, "short_id", "")))
        print("IMAGE=" + shlex.quote(getattr(nim, "image", "")))
        if not os.environ.get("NIM_SERVED_MODEL_NAME"):
            print("NIM_SERVED_MODEL_NAME=" + shlex.quote(getattr(nim, "served_model_id", "")))
        for key, value in (getattr(nim, "env", {}) or {}).items():
            if re.match(r"^[A-Z_][A-Z0-9_]*$", key) and not os.environ.get(key):
                print(f"{key}=" + shlex.quote(str(value)))
        break
PY
    )"
    if [ -n "$_resolved" ]; then
        eval "$_resolved"
    fi
fi

IMAGE="${IMAGE:-nvcr.io/nim/nvidia/${MODEL}:latest}"

if _truthy "${RESOLVE_ONLY:-0}"; then
    echo "MODEL=$MODEL"
    echo "IMAGE=$IMAGE"
    if [ -n "${NIM_SERVED_MODEL_NAME:-}" ]; then
        echo "NIM_SERVED_MODEL_NAME=$NIM_SERVED_MODEL_NAME"
    fi
    for _env_name in NIM_MAX_MODEL_LEN NIM_ENGINE NIM_MODEL_SIZE NIM_MODEL_PROFILE NIM_MEDIA_IO_KWARGS NIM_MAX_IMAGES_PER_PROMPT NIM_NSPECT_ID; do
        if [ -n "${!_env_name:-}" ]; then
            echo "$_env_name=${!_env_name}"
        fi
    done
    exit 0
fi

LOG=/tmp/nim_launch.log
exec > >(tee -a "$LOG") 2>&1

echo "=== NIM launch $(date) ==="
echo "MODEL=$MODEL"
echo "IMAGE=$IMAGE"
echo "PORT=$PORT"
echo "CONTAINER_NAME=$CONTAINER_NAME"
echo "LOCAL_NIM_CACHE=$LOCAL_NIM_CACHE"
echo "NIM_CACHE_MODE=$NIM_CACHE_MODE"
echo "FORCE_RESTART=$FORCE_RESTART"
echo "HOME=$HOME"

if [ -z "$NGC_API_KEY" ]; then
    echo "ERROR: NGC_API_KEY required via env var or chmod-0600 $NIM_CREDENTIAL_FILE"
    echo "Usage: NGC_API_KEY=... bash nim_launch.sh"
    exit 1
fi
export NGC_API_KEY
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
fi

DOCKER_CONFIG="$(mktemp -d /tmp/nim-docker-config.XXXXXX)"
chmod 700 "$DOCKER_CONFIG"
export DOCKER_CONFIG
cleanup_docker_config() {
    rm -rf "$DOCKER_CONFIG"
}
trap cleanup_docker_config EXIT

# 0. Idempotency: if a container of this name is already running AND /v1/models
#    is healthy, reuse it. This protects in-progress first-run model downloads.
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    if _truthy "$FORCE_RESTART"; then
        echo "FORCE_RESTART requested - replacing existing $CONTAINER_NAME."
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    elif curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "Container $CONTAINER_NAME is already running and healthy on port $PORT — reusing."
        echo "API endpoint: http://localhost:${PORT}/v1"
        exit 0
    fi
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "Container $CONTAINER_NAME is running but /v1/models not yet ready — leaving it alone."
        echo "Set FORCE_RESTART=1 to replace it."
        # Fall through to wait loop on the existing container.
        SKIP_LAUNCH=1
    fi
fi

# 1. Authenticate with nvcr.io (idempotent)
if [ -z "${SKIP_LAUNCH:-}" ]; then
    echo "Logging into nvcr.io..."
    echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
fi

# 2. Pull image (resumes if already pulled)
if [ -z "${SKIP_LAUNCH:-}" ]; then
    echo "Pulling $IMAGE (first pull ~10-30 min depending on model size)..."
    docker pull "$IMAGE"
fi

# 3. Stop + remove any existing container with the same name (only if we are launching fresh)
if [ -z "${SKIP_LAUNCH:-}" ]; then
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
fi

if [ -z "${SKIP_LAUNCH:-}" ]; then
    # 5. Free any existing process on $PORT (best-effort, ignores failure)
    if command -v fuser &>/dev/null; then
        fuser -k "${PORT}/tcp" 2>/dev/null || true
    fi

    # 6. Launch container in detached mode (mirrors official docker run from build.nvidia.com)
    echo "Starting NIM container on port $PORT..."
    DOCKER_ARGS=(
        docker run -d
        --name "$CONTAINER_NAME"
        --gpus all
        --ipc host
        --shm-size="$SHM_SIZE"
        --ulimit memlock=-1
        --ulimit stack=67108864
        -e NGC_API_KEY
        -p "${PORT}:8000"
    )
    if [ -n "$HF_TOKEN" ]; then
        DOCKER_ARGS+=(-e HF_TOKEN)
    fi
    if [ "$NIM_CACHE_MODE" = "host" ]; then
        mkdir -p "$LOCAL_NIM_CACHE"
        DOCKER_ARGS+=(-v "$LOCAL_NIM_CACHE:/opt/nim/.cache")
    else
        echo "Using container-internal NIM cache (set NIM_CACHE_MODE=host to bind-mount LOCAL_NIM_CACHE)."
    fi
    for _env_name in \
        NIM_MAX_MODEL_LEN NIM_ENGINE NIM_MODEL_SIZE NIM_MODEL_PROFILE NIM_SERVED_MODEL_NAME \
        NIM_MODEL_NAME NIM_MEDIA_IO_KWARGS NIM_MAX_IMAGES_PER_PROMPT NIM_MAX_VIDEOS_PER_PROMPT \
        NIM_NSPECT_ID NIM_TENSOR_PARALLEL_SIZE NIM_PIPELINE_PARALLEL_SIZE \
        NIM_PASSTHROUGH_ARGS NIM_KVCACHE_PERCENT NIM_ENABLE_KV_CACHE_REUSE \
        NIM_DISABLE_MM_PREPROCESSOR_CACHE NIM_COMPILATION_CONFIG NIM_MAX_NUM_BATCHED_TOKENS \
        NIM_MAX_NUM_SEQS NIM_STREAM_INTERVAL NIM_ENABLE_CHUNKED_PREFILL \
        NIM_DISABLE_CHUNKED_MM_INPUT NIM_VIDEO_PRUNING_RATE NIM_CONN_FAST_HTTP \
        NIM_CONN_READ_BUFSIZE NIM_CONN_TRUST_ENV
    do
        if [ -n "${!_env_name:-}" ]; then
            DOCKER_ARGS+=(-e "$_env_name=${!_env_name}")
            echo "Forwarding $_env_name=${!_env_name}"
        fi
    done
    if [ -n "$NIM_EXTRA_ENV" ]; then
        IFS=',' read -ra _pairs <<< "$NIM_EXTRA_ENV"
        for _pair in "${_pairs[@]}"; do
            if [ -n "$_pair" ]; then
                DOCKER_ARGS+=(-e "$_pair")
                echo "Forwarding ${_pair%%=*}=<set>"
            fi
        done
    fi
    DOCKER_ARGS+=("$IMAGE")
    "${DOCKER_ARGS[@]}"
fi

# 7. Wait for /v1/models (model download + load can be lengthy on first run)
echo "Waiting for NIM /v1/models on http://localhost:${PORT} (max ${MAX_WAIT}s)..."
ELAPSED=0
INTERVAL=10
READY=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    BODY=$(curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null || echo "")
    if [ -n "$BODY" ]; then
        echo "NIM is ready!"
        echo "$BODY"
        READY=1
        break
    fi
    # Check container is still running
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "ERROR: container $CONTAINER_NAME exited prematurely. Last logs:"
        docker logs --tail 80 "$CONTAINER_NAME" 2>&1 || true
        exit 2
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "  ...still waiting (${ELAPSED}s elapsed)"
    fi
done

if [ $READY -eq 0 ]; then
    echo "ERROR: NIM did not respond within ${MAX_WAIT}s. Container logs:"
    docker logs --tail 100 "$CONTAINER_NAME" 2>&1 || true
    exit 3
fi

echo "=== NIM launch complete ==="
echo "API endpoint: http://localhost:${PORT}/v1"
echo "Test:  curl http://localhost:${PORT}/v1/models"
echo "Logs:  docker logs -f $CONTAINER_NAME"
echo ""
echo "To launch Gradio with NIM backend:"
echo "  INFERENCE_BACKEND=nim_local VLLM_BASE_URL=http://localhost:${PORT}/v1 python3 /tmp/gradio_cr2_byo.py"
