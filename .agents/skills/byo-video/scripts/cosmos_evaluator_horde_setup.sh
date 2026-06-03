#!/usr/bin/env bash
set -euo pipefail

EVALUATOR_REPO="${EVALUATOR_REPO:-https://github.com/nv-asotelo/cosmos-evaluator.git}"
EVALUATOR_REF="${EVALUATOR_REF:-codex/cosmos3-super-nim}"
EVALUATOR_DIR="${EVALUATOR_DIR:-$HOME/cosmos-evaluator}"
COOKBOOK_REPO="${COOKBOOK_REPO:-https://github.com/nv-asotelo/cosmos-cookbook.git}"
COOKBOOK_REF="${COOKBOOK_REF:-codex/cosmos3-evaluator-byo-video}"
COOKBOOK_DIR="${COOKBOOK_DIR:-$HOME/cosmos-cookbook}"
NIM_CREDENTIAL_FILE="${NIM_CREDENTIAL_FILE:-$HOME/.cosmos_evaluator/nim.env}"
EVALUATOR_ENV_FILE="${EVALUATOR_ENV_FILE:-$HOME/.cosmos_evaluator/evaluator.env}"
NIM_MODEL_SIZE="${NIM_MODEL_SIZE:-super}"
LAUNCH_GRADIO="${LAUNCH_GRADIO:-1}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
GRADIO_SHARE="${GRADIO_SHARE:-false}"

if [ -f "$NIM_CREDENTIAL_FILE" ]; then
  mode="$(stat -c %a "$NIM_CREDENTIAL_FILE" 2>/dev/null || stat -f %Lp "$NIM_CREDENTIAL_FILE")"
  case "$mode" in
    600|400) ;;
    *) echo "ERROR: $NIM_CREDENTIAL_FILE must be chmod 0600 or 0400" >&2; exit 1 ;;
  esac
else
  echo "ERROR: $NIM_CREDENTIAL_FILE is missing. It must define NGC_API_KEY." >&2
  exit 1
fi

if [ ! -f "$EVALUATOR_ENV_FILE" ]; then
  mkdir -p "$(dirname "$EVALUATOR_ENV_FILE")"
  umask 077
  cat > "$EVALUATOR_ENV_FILE" <<'EOF'
COSMOS_EVALUATOR_ENV=local
COSMOS_EVALUATOR_STORAGE_TYPE=local
COSMOS3_NIM_API_KEY=not-used
MULTISTORAGECLIENT_CONFIGURATION=
EOF
fi

if [ -d "$EVALUATOR_DIR/.git" ]; then
  git -C "$EVALUATOR_DIR" fetch origin "$EVALUATOR_REF"
  git -C "$EVALUATOR_DIR" checkout "$EVALUATOR_REF"
  git -C "$EVALUATOR_DIR" pull --ff-only origin "$EVALUATOR_REF"
else
  git clone --branch "$EVALUATOR_REF" "$EVALUATOR_REPO" "$EVALUATOR_DIR"
fi

cd "$EVALUATOR_DIR"
export NIM_MODEL_SIZE
bash deploy/horde/build_cosmos3_evaluator_images.sh
bash deploy/horde/launch_cosmos3_evaluator.sh
NIM_MODEL_SIZE="$NIM_MODEL_SIZE" bash deploy/horde/smoke_cosmos3_evaluator.sh

case "$LAUNCH_GRADIO" in
  1|true|TRUE|yes|YES|on|ON)
    if [ -d "$COOKBOOK_DIR/.git" ]; then
      git -C "$COOKBOOK_DIR" fetch origin "$COOKBOOK_REF"
      git -C "$COOKBOOK_DIR" checkout "$COOKBOOK_REF"
      git -C "$COOKBOOK_DIR" pull --ff-only origin "$COOKBOOK_REF"
    else
      git clone --branch "$COOKBOOK_REF" "$COOKBOOK_REPO" "$COOKBOOK_DIR"
    fi

    cp "$COOKBOOK_DIR/.agents/skills/byo-video/scripts/gradio_cosmos_evaluator.py" /tmp/gradio_cosmos_evaluator.py
    cp "$COOKBOOK_DIR/.agents/skills/byo-video/scripts/byo_video_setup.py" /tmp/byo_video_setup.py
    chmod +x /tmp/gradio_cosmos_evaluator.py /tmp/byo_video_setup.py

    rm -f /tmp/gradio_url.txt /tmp/gradio_live.flag /tmp/gradio_demo.log
    fuser -k "${GRADIO_PORT}/tcp" >/dev/null 2>&1 || true
    nohup env \
      BYO_VIDEO_FRONTEND=cosmos_evaluator \
      GRADIO_PORT="$GRADIO_PORT" \
      GRADIO_SHARE="$GRADIO_SHARE" \
      COSMOS_EVALUATOR_DATA_DIR="$EVALUATOR_DIR/checks/sample_data/cosmos_public" \
      COSMOS_EVALUATOR_NIM_URL="${COSMOS_EVALUATOR_NIM_URL:-http://localhost:8000}" \
      COSMOS_EVALUATOR_VLM_URL="${COSMOS_EVALUATOR_VLM_URL:-http://localhost:8083}" \
      COSMOS_EVALUATOR_CONTROL_URL="${COSMOS_EVALUATOR_CONTROL_URL:-http://localhost:8090}" \
      COSMOS_EVALUATOR_ATTRIBUTE_URL="${COSMOS_EVALUATOR_ATTRIBUTE_URL:-http://localhost:8086}" \
      COSMOS_EVALUATOR_HALLUCINATION_URL="${COSMOS_EVALUATOR_HALLUCINATION_URL:-http://localhost:8085}" \
      COSMOS_EVALUATOR_OBSTACLE_URL="${COSMOS_EVALUATOR_OBSTACLE_URL:-http://localhost:8082}" \
      python3 /tmp/byo_video_setup.py > /tmp/byo_video_setup.log 2>&1 &

    for _ in $(seq 1 120); do
      if [ -s /tmp/gradio_live.flag ]; then
        echo "Cosmos Evaluator Gradio URL: $(cat /tmp/gradio_url.txt)"
        exit 0
      fi
      sleep 2
    done
    echo "ERROR: Cosmos Evaluator Gradio did not become ready. Tail follows:" >&2
    tail -100 /tmp/byo_video_setup.log >&2 || true
    exit 1
    ;;
  *)
    echo "Cosmos Evaluator services are live. Set LAUNCH_GRADIO=1 to launch the Gradio workbench."
    ;;
esac
