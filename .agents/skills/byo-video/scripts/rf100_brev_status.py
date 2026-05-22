#!/usr/bin/env python3
"""Summarize RF100 Brev benchmark progress without exposing secrets."""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


RUN_ROOT = Path(os.environ.get("BENCHMARK_RUN_ROOT", "/tmp/benchmark-runs"))
POINTER_PATH = Path(os.environ.get("RF100_CURRENT_RUN_PATH", "/tmp/rf100_brev_current_run.json"))


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest_results_path(pointer: dict) -> Path | None:
    run_root = Path(pointer.get("run_root") or RUN_ROOT)
    run_id = pointer.get("run_id")
    if run_id:
        p = run_root / str(run_id) / "results.json"
        if p.exists():
            return p
        return None
    candidates = [Path(p) for p in glob.glob(str(run_root / "bench-*" / "results.json"))]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def nvidia_smi() -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"unavailable: {exc}"]


def process_lines() -> list[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,etime,pcpu,pmem,comm,args"], text=True, timeout=10)
    except Exception:
        return []
    needles = ("rf100_brev_runner.py", "byo_video_batch_inference", "VLLM::EngineCore", "start_server", "vllm", "triton")
    lines = []
    for line in out.splitlines():
        if any(n in line for n in needles):
            lines.append(redact(line[:500]))
    return lines


def disk_lines() -> list[str]:
    paths = [p for p in ("/", "/ephemeral", "/tmp") if Path(p).exists()]
    try:
        out = subprocess.check_output(["df", "-h", *paths], text=True, timeout=10, stderr=subprocess.DEVNULL)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"unavailable: {exc}"]


def detect_model(base_url: str) -> str:
    if not base_url:
        base_url = "http://127.0.0.1:8000/v1"
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("data") or []
        if models:
            return models[0].get("id") or models[0].get("root") or ""
    except Exception as exc:
        return f"model_probe_error: {exc}"
    return ""


def as_int(value, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except Exception:
        return default


def merged_source_config(pointer: dict, snap: dict) -> dict:
    merged: dict = {}
    for origin in (pointer, snap):
        if not isinstance(origin, dict):
            continue
        for key in ("source_config", "rf100_source_config", "source"):
            value = origin.get(key)
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    value = {}
            if isinstance(value, dict):
                merged.update(value)
    return merged


def first_value(*values) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def lane_info(pointer: dict, snap: dict, observed_model: str, instance_name: str) -> dict:
    source_config = merged_source_config(pointer, snap)
    shard_count = as_int(first_value(snap.get("shard_count"), pointer.get("shard_count"), source_config.get("shard_count")), 1)
    shard_index = as_int(first_value(snap.get("shard_index"), pointer.get("shard_index"), source_config.get("shard_index")), 0)
    raw_direction = first_value(
        snap.get("lane_direction"),
        pointer.get("lane_direction"),
        source_config.get("lane_direction"),
        snap.get("shard_direction"),
        pointer.get("shard_direction"),
        source_config.get("shard_direction"),
        snap.get("shard_order"),
        pointer.get("shard_order"),
        source_config.get("shard_order"),
    ).lower()
    evidence_text = " ".join(
        str(v)
        for v in (
            observed_model,
            snap.get("model"),
            pointer.get("model"),
            snap.get("run_id"),
            pointer.get("run_id"),
            source_config.get("mode"),
            source_config.get("air_support_group"),
            source_config.get("lane_role"),
            source_config.get("lane_direction"),
            source_config.get("shard_order"),
        )
        if v
    ).lower()
    if raw_direction in {"reverse", "backward", "descending"} or "reverse" in evidence_text:
        lane_direction = "reverse"
    elif raw_direction in {"forward", "ascending"}:
        lane_direction = "forward"
    elif shard_count > 1:
        lane_direction = "forward"
    else:
        lane_direction = "single"

    lane_role = first_value(snap.get("lane_role"), pointer.get("lane_role"), source_config.get("lane_role"))
    if not lane_role:
        if any(token in evidence_text for token in ("cosmos3", "cosmos-3", "c3-super", "super-reasoner")):
            lane_role = "C3-Super RF100 shard"
        elif any(token in evidence_text for token in ("cosmos reason 2", "reason-2", "cr2", "cosmos-reason2")):
            lane_role = "CR2 RF100 shard"
        elif shard_count > 1:
            lane_role = "RF100 shard"
        else:
            lane_role = "unclassified"
    if lane_direction == "reverse" and lane_role == "C3-Super RF100 shard":
        lane_role = "reverse C3-Super RF100 shard"
    elif lane_direction == "forward" and lane_role == "C3-Super RF100 shard":
        lane_role = "forward C3-Super RF100 shard"

    stale_name_warning = ""
    if "cr2" in instance_name.lower() and "c3" in lane_role.lower():
        stale_name_warning = (
            "Instance name contains CR2, but current run metadata/model classifies it as a C3 lane. "
            "Air support must use current-job metadata, not the Brev name."
        )

    current_job = f"{lane_role}; {lane_direction} traversal; shard {shard_index}/{shard_count}"
    if source_config.get("shard_start_index") not in (None, "") or source_config.get("shard_end_index") not in (None, ""):
        current_job += f"; index window {source_config.get('shard_start_index', '?')}->{source_config.get('shard_end_index', '?')}"

    return {
        "lane_role": lane_role,
        "lane_direction": lane_direction,
        "current_job": current_job,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_order": first_value(snap.get("shard_order"), pointer.get("shard_order"), source_config.get("shard_order"), lane_direction),
        "air_support_group": first_value(snap.get("air_support_group"), pointer.get("air_support_group"), source_config.get("air_support_group")),
        "stale_instance_name_warning": stale_name_warning,
        "source_config": source_config,
    }


def deletion_safety(instance_name: str) -> dict:
    for path in (Path("/tmp/rf100_deletion_safety.json"), Path("/tmp/rf100_deletion_safe.json")):
        if not path.exists():
            continue
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        row = data.get(instance_name) or data.get("default") or {}
        if isinstance(row, dict):
            return row
    return {
        "safe_to_delete_for_result_evidence": False,
        "reason": "No local deletion-safety marker found; freeze/copy/merge/verify results before deleting.",
    }


def health_classification(
    status: str,
    pointer: dict,
    runner_active: bool,
    processes: list[str],
    done: int,
    total: int,
    recent_15m: int,
    result_age_seconds: float | None,
    started: float | None,
) -> tuple[str, str]:
    status_norm = str(status or "").lower()
    has_model_proc = any(any(token in line.lower() for token in ("vllm", "enginecore", "start_server", "triton")) for line in processes)
    remaining = max(total - done, 0) if total else None
    if status_norm in {"completed", "complete", "done"}:
        return "completed", "No action needed except evidence freeze and report merge."
    if recent_15m > 0:
        return "healthy", "Keep monitoring; report recent throughput and ETA."
    if runner_active and done == 0 and started and time.time() - float(started) > 15 * 60:
        return "dataset_load_stall", "Check dataset/cache/network logs; do not add more Brevs until one lane clears loading."
    if runner_active and result_age_seconds is not None and result_age_seconds > 15 * 60 and remaining:
        return "stuck_no_fresh_results", "Freeze current results, inspect newest log error, then relaunch the same run/shard if safe."
    if not runner_active and has_model_proc and pointer and remaining:
        return "loaded_but_not_producing", "vLLM appears loaded but runner is absent; relaunch runner with the same pointer/shard after freezing evidence."
    if not runner_active and pointer and remaining:
        return "runner_stopped_mid_run", "Runner stopped before completion; preserve results and relaunch same shard if compute scope is unchanged."
    if not pointer and not has_model_proc:
        return "idle", "No active run metadata or model server detected."
    if runner_active:
        return "active_but_waiting", "Runner is alive but has no fresh completions in 15 minutes; watch the next interval and inspect logs if unchanged."
    return status or "unknown", "Inspect pointer, processes, logs, GPU, and disk before acting."


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def redact(text: str) -> str:
    text = re.sub(r"(key=)[A-Za-z0-9_-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|token|secret|password)=)[^ \t\n]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:ROBOFLOW_API_KEY|HF_TOKEN|HUGGINGFACE_HUB_TOKEN|NGC_API_KEY|NVIDIA_API_KEY|NIM_API_KEY|VLLM_API_KEY)=)[^ \t\n]+", r"\1[REDACTED]", text)
    text = re.sub(r"(Authorization: Bearer )[A-Za-z0-9._-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(bearer )[A-Za-z0-9._-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(nvapi-)[A-Za-z0-9_-]+", r"\1[REDACTED]", text)
    return text


def main() -> None:
    pointer = read_json(POINTER_PATH)
    results_path = latest_results_path(pointer)
    snap = read_json(results_path) if results_path else {}
    results = snap.get("results") or []
    progress = snap.get("progress") or {}
    done = len(results) or int(progress.get("done") or progress.get("completed") or 0)
    total = int(snap.get("total") or progress.get("total") or 0)
    processes = process_lines()
    runner_active = any("rf100_brev_runner.py" in line for line in processes)
    errors = sum(1 for r in results if r.get("error")) or int(progress.get("errors") or 0)
    started = snap.get("started_epoch") or pointer.get("started_epoch")
    finished = snap.get("finished_epoch")
    now = time.time()
    elapsed = ((finished or now) - float(started)) if started else None
    remaining = max(total - done, 0) if total else None
    percent = (100.0 * done / total) if total else 0.0
    throughput = (done / elapsed * 3600.0) if elapsed and elapsed > 0 and done else 0.0
    eta = (remaining / throughput * 3600.0) if remaining is not None and throughput > 0 else None
    completed_epochs = []
    for row in results:
        try:
            if row.get("completed_epoch") is not None:
                completed_epochs.append(float(row.get("completed_epoch")))
        except Exception:
            pass
    fresh_results = len(completed_epochs)
    recent_15m = sum(1 for ts in completed_epochs if now - ts <= 15 * 60)
    recent_60m = sum(1 for ts in completed_epochs if now - ts <= 60 * 60)
    recent_15m_iph = recent_15m * 4.0
    recent_60m_iph = recent_60m * 1.0
    result_age_seconds = None
    if results_path and results_path.exists():
        try:
            result_age_seconds = now - results_path.stat().st_mtime
        except Exception:
            result_age_seconds = None
    relaunch_delta = None
    relaunch_elapsed = None
    relaunch_iph = None
    relaunch_path = Path("/tmp/rf100_relaunch_last.json")
    if relaunch_path.exists():
        relaunch = read_json(relaunch_path)
        for key in ("backup_tmp", "backup_ephemeral", "backup", "backup_path"):
            backup = relaunch.get(key)
            if not backup:
                continue
            backup_path = Path(str(backup))
            if not backup_path.exists():
                continue
            backup_snap = read_json(backup_path)
            backup_results = backup_snap.get("results") or []
            relaunch_delta = len(results) - len(backup_results)
            try:
                relaunch_elapsed = now - float(relaunch.get("epoch") or backup_path.stat().st_mtime)
                if relaunch_elapsed > 0:
                    relaunch_iph = relaunch_delta / relaunch_elapsed * 3600.0
            except Exception:
                pass
            break
    base_url = snap.get("base_url") or pointer.get("base_url") or "http://127.0.0.1:8000/v1"
    observed_model = snap.get("model") or pointer.get("model") or detect_model(base_url)
    if not snap.get("model"):
        probed = detect_model(base_url)
        if probed:
            observed_model = probed
    last_errors = []
    for row in results[-50:]:
        if row.get("error"):
            last_errors.append(str(row.get("error"))[:240])
    log_errors = []
    for path in sorted(Path("/tmp").glob("rf100_brev_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
        try:
            tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        except Exception:
            continue
        for line in tail:
            lower = line.lower()
            if (
                any(k in lower for k in ("error", "exception", "traceback", "timeout", "rate limit"))
                or "http 429" in lower
                or " 429:" in lower
                or "status 429" in lower
                or "http 500" in lower
                or " 500:" in lower
                or "status 500" in lower
            ):
                log_errors.append(redact(f"{path.name}: {line[:240]}"))
    log_tail = []
    for path in sorted(Path("/tmp").glob("rf100_brev_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:1]:
        try:
            log_tail = [redact(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]]
        except Exception:
            log_tail = []
    if snap.get("status"):
        status = snap.get("status")
    elif runner_active:
        status = "loading_dataset"
    elif pointer:
        status = "stopped_or_failed_before_results"
    else:
        status = "idle"
    instance_name = os.environ.get("BREV_INSTANCE_NAME") or os.uname().nodename
    lane = lane_info(pointer, snap, observed_model, instance_name)
    health, action_hint = health_classification(
        status,
        pointer,
        runner_active,
        processes,
        done,
        total,
        recent_15m,
        result_age_seconds,
        float(started) if started else None,
    )
    if lane.get("stale_instance_name_warning"):
        action_hint = lane["stale_instance_name_warning"] + " " + action_hint
    out = {
        "brev": instance_name,
        "run_id": snap.get("run_id") or pointer.get("run_id"),
        "status": status,
        "health": health,
        "air_support_action_hint": action_hint,
        "lane_role": lane["lane_role"],
        "lane_direction": lane["lane_direction"],
        "current_job": lane["current_job"],
        "shard_index": lane["shard_index"],
        "shard_count": lane["shard_count"],
        "shard_order": lane["shard_order"],
        "air_support_group": lane["air_support_group"],
        "stale_instance_name_warning": lane["stale_instance_name_warning"],
        "observed_model": observed_model,
        "base_url": base_url,
        "concurrency": snap.get("concurrency") or pointer.get("concurrency"),
        "images_done": done,
        "total_images": total,
        "remaining_images": remaining,
        "percent": round(percent, 3),
        "elapsed": fmt_seconds(elapsed),
        "eta": fmt_seconds(eta),
        "throughput_images_per_hour": round(throughput, 1),
        "fresh_results_with_epoch": fresh_results,
        "recent_15m_images": recent_15m,
        "recent_15m_images_per_hour": round(recent_15m_iph, 1),
        "recent_60m_images": recent_60m,
        "recent_60m_images_per_hour": round(recent_60m_iph, 1),
        "result_age_seconds": round(result_age_seconds, 1) if result_age_seconds is not None else None,
        "relaunch_delta_images": relaunch_delta,
        "relaunch_delta_images_per_hour": round(relaunch_iph, 1) if relaunch_iph is not None else None,
        "relaunch_elapsed": fmt_seconds(relaunch_elapsed) if relaunch_elapsed is not None else "n/a",
        "errors": errors,
        "last_errors": last_errors[-5:],
        "log_errors": log_errors[-5:],
        "log_tail": log_tail,
        "gpu": nvidia_smi(),
        "disk": disk_lines(),
        "processes": processes,
        "results_path": str(results_path) if results_path else "",
        "deletion_safety": deletion_safety(instance_name),
        "pointer": pointer,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
