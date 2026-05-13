#!/usr/bin/env bash
# cosmos3_native_launch.sh — wrap the upstream nvidia-cosmos/cosmos3 stack for /byo-video.
#
# Invoked when INFERENCE_BACKEND=cosmos3_native (i.e. for the Cosmos3 OSS *Generator*
# checkpoints — Cosmos3-Nano, Cosmos3-Super). Starts Ray Serve on :8000 and the upstream
# Gradio frontend on :8080, then writes the URL files our skill polls for.
#
# Env:
#   COSMOS3_DIR          Local checkout of nvidia-cosmos/cosmos3 (default: ~/cosmos3)
#   COSMOS3_CHECKPOINT   --checkpoint-path arg ("Cosmos3-Nano" | "Cosmos3-Super"). Required.
#   COSMOS3_GRADIO_PORT  Gradio bind port (default: 8080)
#   COSMOS3_SERVE_PORT   Ray Serve bind port (default: 8000)
#   COSMOS3_HOST         Gradio bind host (default: 0.0.0.0 — required for off-box access)
#   COSMOS3_PARALLELISM  --parallelism-preset (default: latency)
#   COSMOS3_OUTPUT_DIR   --o <dir> (default: outputs/ray_serve)
#
# Writes:
#   /tmp/gradio_url.txt              http://<host>:<port>  (the upstream Gradio URL)
#   /tmp/gradio_live.flag            sentinel: "live\n" once /v1/models-style ready
#   /tmp/cosmos3_serve.log           Ray Serve stdout/stderr
#   /tmp/cosmos3_gradio.log          Gradio stdout/stderr
#
# Manual smoke (run from the GPU host):
#   COSMOS3_CHECKPOINT=Cosmos3-Nano bash /tmp/cosmos3_native_launch.sh
#
# Stopping: kill the PIDs in /tmp/cosmos3_serve.pid and /tmp/cosmos3_gradio.pid.

set -u

COSMOS3_DIR="${COSMOS3_DIR:-$HOME/cosmos3}"
COSMOS3_CHECKPOINT="${COSMOS3_CHECKPOINT:?COSMOS3_CHECKPOINT must be Cosmos3-Nano or Cosmos3-Super}"
COSMOS3_GRADIO_PORT="${COSMOS3_GRADIO_PORT:-8080}"
COSMOS3_SERVE_PORT="${COSMOS3_SERVE_PORT:-8000}"
COSMOS3_HOST="${COSMOS3_HOST:-0.0.0.0}"
COSMOS3_PARALLELISM="${COSMOS3_PARALLELISM:-latency}"
COSMOS3_OUTPUT_DIR="${COSMOS3_OUTPUT_DIR:-outputs/ray_serve}"

if [[ ! -d "$COSMOS3_DIR" ]]; then
  echo "✗ COSMOS3_DIR=$COSMOS3_DIR not found." >&2
  echo "  Install first:" >&2
  echo "    git clone https://github.com/nvidia-cosmos/cosmos3.git \"$COSMOS3_DIR\"" >&2
  echo "    cd \"$COSMOS3_DIR\" && uv sync --all-extras --group=cu130-train" >&2
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
export LD_LIBRARY_PATH=    # required per cosmos3 setup.md (PyTorch _C import fix)

mkdir -p "$COSMOS3_OUTPUT_DIR"
: > /tmp/cosmos3_serve.log
: > /tmp/cosmos3_gradio.log
rm -f /tmp/gradio_live.flag /tmp/gradio_url.txt

echo "→ Starting Ray Serve (cosmos3.ray.serve --checkpoint-path $COSMOS3_CHECKPOINT) on :$COSMOS3_SERVE_PORT"
nohup uv run --no-sync python -m cosmos3.ray.serve \
    --parallelism-preset="$COSMOS3_PARALLELISM" \
    --keep-going \
    -o "$COSMOS3_OUTPUT_DIR" \
    --checkpoint-path "$COSMOS3_CHECKPOINT" \
    > /tmp/cosmos3_serve.log 2>&1 &
echo $! > /tmp/cosmos3_serve.pid

echo "→ Waiting up to 180s for Ray Serve to bind :$COSMOS3_SERVE_PORT"
for i in $(seq 1 60); do
  if ss -ltn 2>/dev/null | grep -q ":${COSMOS3_SERVE_PORT}\b"; then
    echo "✓ Ray Serve listening on :$COSMOS3_SERVE_PORT (after ${i}×3s)"
    break
  fi
  sleep 3
done

if ! ss -ltn 2>/dev/null | grep -q ":${COSMOS3_SERVE_PORT}\b"; then
  echo "✗ Ray Serve did not bind :$COSMOS3_SERVE_PORT within 180s. Last 30 log lines:" >&2
  tail -30 /tmp/cosmos3_serve.log >&2 || true
  exit 2
fi

echo "→ Starting Gradio (cosmos3.ray.gradio) on $COSMOS3_HOST:$COSMOS3_GRADIO_PORT"
nohup uv run --no-sync python -m cosmos3.ray.gradio \
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
    echo "live" > /tmp/gradio_live.flag
    echo "✓ Gradio live at $URL"
    exit 0
  fi
  sleep 2
done

echo "✗ Gradio did not bind :$COSMOS3_GRADIO_PORT within 60s. Last 30 log lines:" >&2
tail -30 /tmp/cosmos3_gradio.log >&2 || true
exit 3
