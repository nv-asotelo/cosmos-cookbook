#!/usr/bin/env bash
set -euo pipefail

EVALUATOR_REPO="${EVALUATOR_REPO:-https://github.com/nv-asotelo/cosmos-evaluator.git}"
EVALUATOR_REF="${EVALUATOR_REF:-codex/cosmos3-super-nim}"
EVALUATOR_DIR="${EVALUATOR_DIR:-$HOME/cosmos-evaluator}"
NIM_CREDENTIAL_FILE="${NIM_CREDENTIAL_FILE:-$HOME/.cosmos_evaluator/nim.env}"
EVALUATOR_ENV_FILE="${EVALUATOR_ENV_FILE:-$HOME/.cosmos_evaluator/evaluator.env}"

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
bash deploy/horde/build_cosmos3_evaluator_images.sh
bash deploy/horde/launch_cosmos3_evaluator.sh
bash deploy/horde/smoke_cosmos3_evaluator.sh
