#!/usr/bin/env python3
"""Launch an RF100-VL benchmark run on a Brev instance.

This tiny wrapper keeps the long-running experiment out of the web server
process. It imports the shared BYO benchmark implementation from the copied
batch UI script and writes a small pointer file for heartbeat monitoring.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.request


SCRIPT_PATH = Path(os.environ.get("BYO_BATCH_SCRIPT", "/tmp/byo_video_batch_inference.py"))
RUN_ROOT = Path(os.environ.get("BENCHMARK_RUN_ROOT", "/tmp/benchmark-runs"))
POINTER_PATH = Path(os.environ.get("RF100_CURRENT_RUN_PATH", "/tmp/rf100_brev_current_run.json"))


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return int(value)


def _load_batch_module():
    spec = importlib.util.spec_from_file_location("byo_video_batch_inference", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detect_model(base_url: str) -> str:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models returned by {base_url}/models")
    return models[0].get("id") or models[0].get("root") or "unknown-model"


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get("BENCHMARK_NIM_BASE_URL") or os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1"
    model_id = os.environ.get("MODEL_ID") or _detect_model(base_url)
    run_id = os.environ.get("RUN_ID") or f"bench-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
    dataset_id = os.environ.get("DATASET_ID", "rf100-vl")
    sample_size = int(os.environ.get("SAMPLE_SIZE", "0"))
    concurrency = int(os.environ.get("CONCURRENCY", "16"))
    seed = int(os.environ.get("SEED", "42"))
    source_config = json.loads(os.environ.get("SOURCE_CONFIG_JSON", '{"gate_run": true, "mode": "full"}'))
    shard_count = int(os.environ.get("SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("SHARD_INDEX", "0"))
    if shard_count > 1:
        if shard_index < 0 or shard_index >= shard_count:
            raise RuntimeError(f"SHARD_INDEX must be in [0, {shard_count - 1}], got {shard_index}")
        source_config = {**source_config, "shard_count": shard_count, "shard_index": shard_index}

    lane_direction = _env_first("LANE_DIRECTION", "SHARD_DIRECTION") or str(source_config.get("lane_direction") or "")
    shard_order = _env_first("SHARD_ORDER", "SHARD_TRAVERSAL") or str(source_config.get("shard_order") or "")
    if not lane_direction and shard_order.lower() in {"reverse", "backward", "descending"}:
        lane_direction = "reverse"
    if not shard_order:
        shard_order = lane_direction or "forward"
    lane_role = _env_first("LANE_ROLE", "AIR_SUPPORT_LANE_ROLE") or str(source_config.get("lane_role") or "")
    air_support_group = _env_first("AIR_SUPPORT_GROUP", "BENCHMARK_FLEET") or str(source_config.get("air_support_group") or "")
    shard_start_index = _env_int("SHARD_START_INDEX")
    shard_end_index = _env_int("SHARD_END_INDEX")
    lane_metadata = {
        "lane_role": lane_role,
        "lane_direction": lane_direction,
        "shard_order": shard_order,
        "air_support_group": air_support_group,
        "shard_start_index": shard_start_index,
        "shard_end_index": shard_end_index,
    }
    source_config = {**source_config, **{k: v for k, v in lane_metadata.items() if v not in ("", None)}}

    POINTER_PATH.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": dataset_id,
                "model": model_id,
                "base_url": base_url,
                "sample_size": sample_size,
                "concurrency": concurrency,
                "seed": seed,
                "shard_count": shard_count,
                "shard_index": shard_index,
                **{k: v for k, v in lane_metadata.items() if v not in ("", None)},
                "started_epoch": time.time(),
                "run_root": str(RUN_ROOT),
                "script_path": str(SCRIPT_PATH),
                "source_config": source_config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    module = _load_batch_module()
    model_config = {
        "id": model_id,
        "endpoint": base_url,
        "headers": module.benchmark_nim_headers(),
        "host": os.environ.get("BREV_INSTANCE_NAME", ""),
        "status": "ready",
    }
    module.run_rf100vl_benchmark(
        run_id,
        dataset_id,
        sample_size,
        concurrency,
        seed,
        source_config,
        model_config,
    )


if __name__ == "__main__":
    main()
