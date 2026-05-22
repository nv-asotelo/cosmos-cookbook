#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${BREV_INSTANCE_NAME:-$(hostname)}"
SHARD_COUNT="${SHARD_COUNT:-4}"
SHARD_INDEX="${SHARD_INDEX:?SHARD_INDEX is required}"
CONCURRENCY="${CONCURRENCY:-32}"
RUN_SLUG="${RUN_SLUG:-cr2-dual-shard${SHARD_INDEX}}"
CACHE_DIR="${CACHE_DIR:-/ephemeral/nim-cache}"

if [ -f /tmp/rf100.env ]; then
  chmod 600 /tmp/rf100.env
  set -a
  # shellcheck disable=SC1091
  . /tmp/rf100.env
  set +a
fi

if [ -z "${NGC_API_KEY:-}" ]; then
  echo "ERROR: NGC_API_KEY is required in /tmp/rf100.env or environment."
  exit 1
fi

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl start docker || true
else
  sudo service docker start || true
fi
docker info >/dev/null

python3 -m venv /tmp/rf100-runner-venv
/tmp/rf100-runner-venv/bin/python -m pip install --upgrade pip wheel setuptools
/tmp/rf100-runner-venv/bin/python -m pip install \
  openai requests pillow numpy pandas pycocotools roboflow rf100vl python-pptx

mkdir -p /ephemeral/benchmark-runs /ephemeral/rf100-vl /ephemeral/tmp "$CACHE_DIR"
chmod +x /tmp/rf100_brev_launch.sh /tmp/rf100_brev_runner.py /tmp/rf100_brev_status.py \
  /tmp/rf100_cr2_second_replica.sh /tmp/openai_round_robin_proxy.py

CACHE_DIR="$CACHE_DIR" CONTAINER_NAME=cosmos-nim-gpu0 PORT=8000 GPU_DEVICE=0 bash /tmp/rf100_cr2_second_replica.sh
CACHE_DIR="$CACHE_DIR" CONTAINER_NAME=cosmos-nim-gpu1 PORT=8001 GPU_DEVICE=1 bash /tmp/rf100_cr2_second_replica.sh

for port in 8000 8001; do
  echo "waiting for CR2 NIM on ${port}"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "ready ${port}"
      break
    fi
    sleep 10
  done
  curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null
done

pkill -f openai_round_robin_proxy.py >/dev/null 2>&1 || true
OPENAI_PROXY_TARGETS="http://127.0.0.1:8000,http://127.0.0.1:8001" \
  OPENAI_PROXY_PORT=8010 \
  nohup python3 /tmp/openai_round_robin_proxy.py >/tmp/openai_round_robin_proxy.log 2>&1 &

sleep 3
curl -sf http://127.0.0.1:8010/v1/models >/dev/null

BENCHMARK_RUN_ROOT=/ephemeral/benchmark-runs \
RF100_VL_DATA_ROOT=/ephemeral/rf100-vl \
TMPDIR=/ephemeral/tmp \
BENCHMARK_NIM_BASE_URL=http://127.0.0.1:8010/v1 \
CONCURRENCY="$CONCURRENCY" \
SHARD_COUNT="$SHARD_COUNT" \
SHARD_INDEX="$SHARD_INDEX" \
RUN_SLUG="$RUN_SLUG" \
BREV_INSTANCE_NAME="$INSTANCE" \
bash /tmp/rf100_brev_launch.sh
