#!/usr/bin/env bash
# cosmos3_native_launch.sh — wrap NVIDIA/cosmos-framework for /byo-video.
#
# Invoked when INFERENCE_BACKEND=cosmos3_native (i.e. for the Cosmos3 OSS *Generator*
# checkpoints — Cosmos3-Nano, Cosmos3-Super). Starts Ray Serve on :8000 and the upstream
# framework Gradio frontend on :8080, then writes the URL files our skill polls for.
#
# Env:
#   COSMOS3_DIR          Local checkout of NVIDIA/cosmos-framework (default: ~/cosmos-framework)
#   COSMOS3_CHECKPOINT   --checkpoint-path arg ("Cosmos3-Nano" | "Cosmos3-Super"). Required.
#   COSMOS3_GRADIO_PORT  Gradio bind port (default: 8080)
#   COSMOS3_SERVE_PORT   Ray Serve bind port (default: 8000)
#   COSMOS3_HOST         Gradio bind host (default: 0.0.0.0 — required for off-box access)
#   COSMOS3_PARALLELISM  --parallelism-preset (default: latency)
#   COSMOS3_OUTPUT_DIR   --o <dir> (default: outputs/ray_serve)
#   COSMOS3_MAX_WAIT     Seconds to wait for Ray Serve /info (default: 2400)
#   COSMOS3_DEVICE_MEMORY_BYTES
#                        Fallback total memory for GB10/NVML NotSupported (default: 137438953472)
#
# Writes:
#   /tmp/gradio_url.txt              http://<host>:<port>  (the framework Gradio URL)
#   /tmp/cosmos3_framework_gradio_url.txt
#   /tmp/gradio_live.flag            sentinel: "live\n" once Gradio is reachable
#   /tmp/cosmos3_serve.log           Ray Serve stdout/stderr
#   /tmp/cosmos3_gradio.log          Gradio stdout/stderr
#
# Manual smoke (run from the GPU host):
#   COSMOS3_CHECKPOINT=Cosmos3-Nano bash /tmp/cosmos3_native_launch.sh
#
# Stopping: kill the PIDs in /tmp/cosmos3_serve.pid and /tmp/cosmos3_gradio.pid.

set -u

COSMOS3_DIR="${COSMOS3_DIR:-$HOME/cosmos-framework}"
COSMOS3_CHECKPOINT="${COSMOS3_CHECKPOINT:?COSMOS3_CHECKPOINT must be Cosmos3-Nano or Cosmos3-Super}"
COSMOS3_GRADIO_PORT="${COSMOS3_GRADIO_PORT:-8080}"
COSMOS3_SERVE_PORT="${COSMOS3_SERVE_PORT:-8000}"
COSMOS3_HOST="${COSMOS3_HOST:-0.0.0.0}"
COSMOS3_PARALLELISM="${COSMOS3_PARALLELISM:-latency}"
COSMOS3_OUTPUT_DIR="${COSMOS3_OUTPUT_DIR:-outputs/ray_serve}"
COSMOS3_MAX_WAIT="${COSMOS3_MAX_WAIT:-2400}"
COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"
COSMOS3_DEVICE_MEMORY_BYTES="${COSMOS3_DEVICE_MEMORY_BYTES:-137438953472}"

if [[ ! -d "$COSMOS3_DIR" ]]; then
  echo "✗ COSMOS3_DIR=$COSMOS3_DIR not found." >&2
  echo "  Install first:" >&2
  echo "    git clone https://github.com/NVIDIA/cosmos-framework.git \"$COSMOS3_DIR\"" >&2
  echo "    cd \"$COSMOS3_DIR\" && uv sync --all-extras --group=$COSMOS3_UV_GROUP" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "✗ uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
fi

cd "$COSMOS3_DIR"
export LD_LIBRARY_PATH=    # required per cosmos-framework setup guidance (PyTorch _C import fix)

# GB10 unified-memory hosts return NVMLError_NotSupported for NVML memory info.
# cosmos-framework currently queries NVML before it can build default
# parallelism, even though the value is not used by the current default
# shard-size path. Patch only this launched process via sitecustomize.
COSMOS3_SITECUSTOMIZE_DIR="/tmp/cosmos3_gb10_sitecustomize"
mkdir -p "$COSMOS3_SITECUSTOMIZE_DIR"
cat > "$COSMOS3_SITECUSTOMIZE_DIR/sitecustomize.py" <<'PY'
import os

try:
    import importlib.metadata as _metadata
    import pynvml

    _original_version = _metadata.version
    _original_get_memory_info = pynvml.nvmlDeviceGetMemoryInfo

    def _version_with_cosmos3_fallback(distribution_name):
        try:
            return _original_version(distribution_name)
        except _metadata.PackageNotFoundError:
            if distribution_name == "cosmos3":
                return os.environ.get("COSMOS3_VERSION_FALLBACK", "cosmos-framework")
            raise

    class _FallbackMemoryInfo:
        def __init__(self, total):
            self.total = total
            self.free = total
            self.used = 0

    def _get_memory_info_with_gb10_fallback(handle):
        try:
            return _original_get_memory_info(handle)
        except Exception as exc:
            if exc.__class__.__name__ != "NVMLError_NotSupported":
                raise
            total = int(os.environ.get("COSMOS3_DEVICE_MEMORY_BYTES", "137438953472"))
            return _FallbackMemoryInfo(total)

    _metadata.version = _version_with_cosmos3_fallback
    pynvml.nvmlDeviceGetMemoryInfo = _get_memory_info_with_gb10_fallback
except Exception:
    pass
PY
export PYTHONPATH="$COSMOS3_SITECUSTOMIZE_DIR:${PYTHONPATH:-}"
export COSMOS3_DEVICE_MEMORY_BYTES

mkdir -p "$COSMOS3_OUTPUT_DIR"
: > /tmp/cosmos3_serve.log
: > /tmp/cosmos3_gradio.log
rm -f /tmp/gradio_live.flag /tmp/gradio_url.txt /tmp/cosmos3_framework_gradio_url.txt

echo "→ Starting Ray Serve (cosmos_framework.inference.ray.serve --checkpoint-path $COSMOS3_CHECKPOINT) on :$COSMOS3_SERVE_PORT"
nohup uv run --no-sync python -m cosmos_framework.inference.ray.serve \
    --parallelism-preset="$COSMOS3_PARALLELISM" \
    --keep-going \
    -o "$COSMOS3_OUTPUT_DIR" \
    --checkpoint-path "$COSMOS3_CHECKPOINT" \
    > /tmp/cosmos3_serve.log 2>&1 &
echo $! > /tmp/cosmos3_serve.pid

echo "→ Waiting up to ${COSMOS3_MAX_WAIT}s for Ray Serve /info on :$COSMOS3_SERVE_PORT"
for i in $(seq 1 "$COSMOS3_MAX_WAIT"); do
  if curl -fsS "http://localhost:${COSMOS3_SERVE_PORT}/info" >/dev/null 2>&1; then
    echo "✓ Ray Serve ready on :$COSMOS3_SERVE_PORT (after ${i}s)"
    break
  fi
  sleep 1
done

if ! curl -fsS "http://localhost:${COSMOS3_SERVE_PORT}/info" >/dev/null 2>&1; then
  echo "✗ Ray Serve did not become ready on :$COSMOS3_SERVE_PORT within ${COSMOS3_MAX_WAIT}s. Last 30 log lines:" >&2
  tail -30 /tmp/cosmos3_serve.log >&2 || true
  exit 2
fi

echo "→ Starting Gradio (cosmos_framework.inference.ray.gradio) on $COSMOS3_HOST:$COSMOS3_GRADIO_PORT"
nohup uv run --no-sync python -m cosmos_framework.inference.ray.gradio \
    --host "$COSMOS3_HOST" \
    --port "$COSMOS3_GRADIO_PORT" \
    --server-host localhost \
    --server-port "$COSMOS3_SERVE_PORT" \
    --server-output-dir "$COSMOS3_OUTPUT_DIR" \
    > /tmp/cosmos3_gradio.log 2>&1 &
echo $! > /tmp/cosmos3_gradio.pid

echo "→ Waiting up to 60s for Gradio to bind :$COSMOS3_GRADIO_PORT"
for i in $(seq 1 30); do
  if ss -ltn 2>/dev/null | grep -q ":${COSMOS3_GRADIO_PORT}\b"; then
    PUBLIC_HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -z "$PUBLIC_HOST" ]] && PUBLIC_HOST="$COSMOS3_HOST"
    URL="http://${PUBLIC_HOST}:${COSMOS3_GRADIO_PORT}"
    echo "$URL" > /tmp/gradio_url.txt
    echo "$URL" > /tmp/cosmos3_framework_gradio_url.txt
    echo "live" > /tmp/gradio_live.flag
    echo "✓ Gradio live at $URL"
    exit 0
  fi
  sleep 2
done

echo "✗ Gradio did not bind :$COSMOS3_GRADIO_PORT within 60s. Last 30 log lines:" >&2
tail -30 /tmp/cosmos3_gradio.log >&2 || true
exit 3
