#!/usr/bin/env python3
"""Runtime-agent frontend for Cosmos BYO-video inference.

This is intentionally self-contained: it serves a small browser UI, loads video
samples from a public Hugging Face dataset through FiftyOne when available, and
runs selected videos concurrently against the OpenAI-compatible vLLM/NIM server
that the BYO-video setup script starts.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import html
import json
import mimetypes
import os
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import requests
except Exception as exc:  # pragma: no cover - surfaced in UI
    requests = None
    REQUESTS_IMPORT_ERROR = exc
else:
    REQUESTS_IMPORT_ERROR = None

WORKER_SAFETY_SYSTEM = """
You are an expert Industrial Safety Inspector monitoring a manufacturing facility.
Your goal is to classify the video into EXACTLY ONE of the 8 classes defined below.

CRITICAL NEGATIVE CONSTRAINTS (What to IGNORE):
1. IGNORE SITTING WORKERS:
   - If a person is SITTING at a machine board working, this is NOT an intervention class. Ignore them.
   - If a person is SITTING driving a forklift, the driver is NOT the class. Focus only on the LOAD carried.
2. IGNORE BACKGROUND:
   - The facility is old. Do not report hazards based on faded floor markings or unpainted areas.
3. SINGLE OUTPUT:
   - Even if multiple things happen, choose the MOST PROMINENT behavior.
   - Prioritize UNSAFE behaviors over SAFE behaviors if both are present.
""".strip()

WORKER_SAFETY_USER = """
Analyze the video and output a JSON object. You MUST select the class ID and Label EXACTLY from the table below.

STRICT CLASSIFICATION TABLE (Use these exact IDs and Labels):

| ID | Label | Definition (Ground Truth) | Hazard Status |
| :--- | :--- | :--- | :--- |
| 0 | Safe Walkway Violation | Worker walks OUTSIDE the designated Green Path. | TRUE (Unsafe) |
| 4 | Safe Walkway | Worker walks INSIDE the designated Green Path. | FALSE (Safe) |
| 1 | Unauthorized Intervention | Worker interacts with machine board WITHOUT a green vest. | TRUE (Unsafe) |
| 5 | Authorized Intervention | Worker interacts with machine board WITH a green vest. | FALSE (Safe) |
| 2 | Opened Panel Cover | Machine panel cover is left OPEN after intervention. | TRUE (Unsafe) |
| 6 | Closed Panel Cover | Machine panel cover is CLOSED after intervention. | FALSE (Safe) |
| 3 | Carrying Overload with Forklift | Forklift carries 3 OR MORE blocks. | TRUE (Unsafe) |
| 7 | Safe Carrying | Forklift carries 2 OR FEWER blocks. | FALSE (Safe) |

INSTRUCTIONS:
1. Identify the behavior in the video.
2. Match it to one row in the table above.
3. Output the exact "ID" and "Label" from that row. Do not invent new labels like "safe and compliant".

OUTPUT FORMAT:
{
  "prediction_class_id": [Integer from Table],
  "prediction_label": "[Exact String from Table]",
  "video_description": "[Concise description of the observed action]",
  "hazard_detection": {
    "is_hazardous": [true/false based on the Hazard Status column],
    "temporal_segment": "[Start Time - End Time] or null"
  }
}
""".strip()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
WORKER_SAFETY_CLASSES = {
    0: {"label": "Safe Walkway Violation", "slug": "safe_walkway_violation", "hazardous": True},
    1: {"label": "Unauthorized Intervention", "slug": "unauthorized_intervention", "hazardous": True},
    2: {"label": "Opened Panel Cover", "slug": "opened_panel_cover", "hazardous": True},
    3: {"label": "Carrying Overload with Forklift", "slug": "carrying_overload_with_forklift", "hazardous": True},
    4: {"label": "Safe Walkway", "slug": "safe_walkway", "hazardous": False},
    5: {"label": "Authorized Intervention", "slug": "authorized_intervention", "hazardous": False},
    6: {"label": "Closed Panel Cover", "slug": "closed_panel_cover", "hazardous": False},
    7: {"label": "Safe Carrying", "slug": "safe_carrying", "hazardous": False},
}
DEFAULT_DATASET = os.getenv("RUNTIME_AGENT_DATASET", "pjramg/Safe_Unsafe_Test")
RESULTS_FILE = Path(os.getenv("RUNTIME_AGENT_RESULTS", "/tmp/byo_video_runtime_agent_results.json"))
THUMBNAIL_DIR = Path(os.getenv("RUNTIME_AGENT_THUMBNAILS", "/tmp/byo_video_runtime_agent_thumbnails"))
THUMBNAIL_SIZE = (220, 124)
REQUEST_TIMEOUT_SECONDS = float(os.getenv("RUNTIME_AGENT_REQUEST_TIMEOUT_SECONDS", "180"))
TEXT_TOKENS = 50
EMPIRICAL_VISUAL_TOKENS_PER_FRAME = 128
BASELINE_PIXELS = 524288
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "vllm").lower()
MAX_PIXELS_MIN = 64 * (32 ** 2)
MAX_PIXELS_MAX = 4096 * (32 ** 2)
RECOMMENDED_MAX_PIXELS = int(os.getenv("RUNTIME_AGENT_RECOMMENDED_MAX_PIXELS", os.getenv("GRADIO_MAX_PIXELS", str(512 * (32 ** 2)))))
DEFAULT_MAX_PIXELS = int(os.getenv("RUNTIME_AGENT_MAX_PIXELS", str(RECOMMENDED_MAX_PIXELS)))
DEFAULT_FPS = float(os.getenv("RUNTIME_AGENT_FPS", os.getenv("GRADIO_FPS", "2")))
DEFAULT_MAX_TOKENS = int(os.getenv("RUNTIME_AGENT_MAX_TOKENS", os.getenv("GRADIO_MAX_TOKENS", "512")))
DEFAULT_MAX_FRAMES = int(os.getenv("RUNTIME_AGENT_MAX_FRAMES", "8"))
if INFERENCE_BACKEND == "nim_local":
    RECOMMENDED_TEMPERATURE = 0.6
    RECOMMENDED_TOP_P = 0.3
    RECOMMENDED_REPETITION_PENALTY = 1.2
else:
    RECOMMENDED_TEMPERATURE = 0.0
    RECOMMENDED_TOP_P = 1.0
    RECOMMENDED_REPETITION_PENALTY = 1.05
DEFAULT_TEMPERATURE = float(os.getenv("RUNTIME_AGENT_TEMPERATURE", str(RECOMMENDED_TEMPERATURE)))
DEFAULT_TOP_P = float(os.getenv("RUNTIME_AGENT_TOP_P", str(RECOMMENDED_TOP_P)))
DEFAULT_REPETITION_PENALTY = float(os.getenv("RUNTIME_AGENT_REPETITION_PENALTY", str(RECOMMENDED_REPETITION_PENALTY)))
BUILD_NVIDIA_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.3,
    "repetition_penalty": 1.2,
}
SLIDER_META = {
    "fps": {
        "min": 1,
        "max": 8,
        "step": 1,
        "recommended": DEFAULT_FPS,
        "unit": "fps",
        "note": "Higher samples more frames and increases visual tokens.",
    },
    "max_pixels": {
        "min": MAX_PIXELS_MIN,
        "max": MAX_PIXELS_MAX,
        "step": MAX_PIXELS_MIN,
        "recommended": RECOMMENDED_MAX_PIXELS,
        "unit": "pixels/frame",
        "note": "Recommended keeps prefill tractable; raise for fine visual detail.",
    },
    "max_tokens": {
        "min": 64,
        "max": 2048,
        "step": 64,
        "recommended": DEFAULT_MAX_TOKENS,
        "unit": "tokens",
        "note": "Maximum generated text tokens.",
    },
    "temperature": {
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "recommended": RECOMMENDED_TEMPERATURE,
        "unit": "",
        "note": "0 is deterministic; NIM-local recommends 0.6.",
    },
    "top_p": {
        "min": 0.01,
        "max": 1.0,
        "step": 0.01,
        "recommended": RECOMMENDED_TOP_P,
        "unit": "",
        "note": "Nucleus sampling threshold; 1.0 disables nucleus truncation.",
    },
    "repetition_penalty": {
        "min": 1.0,
        "max": 2.0,
        "step": 0.05,
        "recommended": RECOMMENDED_REPETITION_PENALTY,
        "unit": "",
        "note": "Higher discourages repeated phrases.",
    },
    "max_frames": {
        "min": 0,
        "max": 128,
        "step": 1,
        "recommended": DEFAULT_MAX_FRAMES,
        "unit": "frames",
        "note": "0 disables this cap; fps still controls sampling.",
    },
}
STATE_LOCK = threading.Lock()
FIFTYONE_SESSION = None

GENERIC_SYSTEM = "You are a helpful assistant."
WAREHOUSE_SYSTEM = "You are a helpful warehouse monitoring system."
DEFAULT_SYSTEM = "You are a helpful assistant that analyzes videos."
DEFAULT_PROMPT = "Describe what is happening in this video. What are the key actions, objects, and events?"
PROMPT_PRESETS = [
    {"label": "Worker safety classification", "user_prompt": WORKER_SAFETY_USER, "system_prompt": WORKER_SAFETY_SYSTEM, "reasoning": False},
    {"label": "General description", "user_prompt": DEFAULT_PROMPT, "system_prompt": DEFAULT_SYSTEM, "reasoning": False},
    {"label": "Safety analysis", "user_prompt": "Identify any safety hazards, risks, or unsafe behaviors visible in this video. Be specific.", "system_prompt": DEFAULT_SYSTEM, "reasoning": False},
    {"label": "Non-expert summary", "user_prompt": "Summarize this video in plain language for someone with no domain expertise.", "system_prompt": DEFAULT_SYSTEM, "reasoning": False},
    {"label": "Action recognition", "user_prompt": "List every distinct action or motion performed in this video, in the order they occur.", "system_prompt": DEFAULT_SYSTEM, "reasoning": False},
    {"label": "Object inventory", "user_prompt": "List all objects, equipment, and people visible. Note their state.", "system_prompt": DEFAULT_SYSTEM, "reasoning": False},
    {"label": "Anomaly detection", "user_prompt": "Identify anything unusual, unexpected, or out of place in this video.", "system_prompt": DEFAULT_SYSTEM, "reasoning": False},
    {"label": "Temporal summary", "user_prompt": "Break this video into time segments and describe what changes in each segment.", "system_prompt": DEFAULT_SYSTEM, "reasoning": False},
    {"label": "Race car: timestamps", "user_prompt": "Describe the video. Add timestamps in mm:ss format.\n\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag and include the timestamps.", "system_prompt": GENERIC_SYSTEM, "reasoning": True},
    {"label": "Forklift: load weight (JSON)", "user_prompt": "Locate the bounding box of the load and determine if its size and weight of load within the forklift's limits. Estimate weights. Return all as json. Include json location, estimated weight of the load, and if it's in the limit.", "system_prompt": GENERIC_SYSTEM, "reasoning": False},
    {"label": "Mail package: pickup allowed?", "user_prompt": "Is the person allowed to pick up the packages?\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag.", "system_prompt": GENERIC_SYSTEM, "reasoning": True},
    {"label": "Warehouse: who picked up the box?", "user_prompt": "Which worker picked up the dropped box?\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag.", "system_prompt": WAREHOUSE_SYSTEM, "reasoning": True},
    {"label": "AV: next ego action", "user_prompt": "What's the next immediate action for the Ego vehicle?\n\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag.", "system_prompt": GENERIC_SYSTEM, "reasoning": True},
    {"label": "Robot arm: 2D trajectory (JSON)", "user_prompt": "You are given the task \"Move the tape into the basket\". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {\"point_2d\": [x, y], \"label\": \"gripper trajectory\"}.\n\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag.", "system_prompt": GENERIC_SYSTEM, "reasoning": True},
    {"label": "SDG critic: approve / reject", "user_prompt": "Approve or reject this generated video for inclusion in a dataset for physical world model ai training. It must perfectly adhere to physics, object permanence, and have no anomalies. Any issue or concern causes rejection.\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag. Answer with Approve or Reject only.", "system_prompt": GENERIC_SYSTEM, "reasoning": True},
]

STATE: Dict[str, Any] = {
    "dataset_repo": DEFAULT_DATASET,
    "dataset_source": None,
    "fo_dataset_name": None,
    "videos": [],
    "results": [],
    "running": False,
    "progress": {"done": 0, "total": 0, "errors": 0},
    "batch_metrics": None,
    "batch_history": [],
    "run_history": [],
    "logs": [],
    "server": {},
    "defaults": {
        "system_prompt": WORKER_SAFETY_SYSTEM,
        "user_prompt": WORKER_SAFETY_USER,
        "concurrency": int(os.getenv("RUNTIME_AGENT_CONCURRENCY", "4")),
        "max_videos": int(os.getenv("RUNTIME_AGENT_MAX_VIDEOS", "20")),
        "fps": DEFAULT_FPS,
        "max_pixels": DEFAULT_MAX_PIXELS,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "max_frames": DEFAULT_MAX_FRAMES,
        "prompt_presets": PROMPT_PRESETS,
        "slider_meta": SLIDER_META,
        "build_defaults": BUILD_NVIDIA_DEFAULTS,
        "use_build_defaults": INFERENCE_BACKEND == "nim_local",
    },
}


def log(message: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + message
    with STATE_LOCK:
        STATE["logs"].append(line)
        STATE["logs"] = STATE["logs"][-200:]
    print("[runtime-agent]", message, flush=True)


def snapshot() -> Dict[str, Any]:
    with STATE_LOCK:
        return json.loads(json.dumps(STATE, default=str))


def update_state(**items: Any) -> None:
    with STATE_LOCK:
        STATE.update(items)


def video_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8", "ignore")).hexdigest()[:16]


def is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def get_video_meta(path: str) -> Dict[str, Any]:
    meta = {"width": 0, "height": 0, "fps": 0.0, "duration_s": 0.0, "total_frames": 0}
    try:
        import av

        with av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if not stream:
                return meta
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            duration_s = float(container.duration) / 1_000_000 if container.duration else 0.0
            total_frames = int(stream.frames or 0)
            if total_frames <= 0 and duration_s and fps:
                total_frames = max(1, int(round(duration_s * fps)))
            meta.update({
                "width": int(stream.width or 0),
                "height": int(stream.height or 0),
                "fps": fps,
                "duration_s": duration_s,
                "total_frames": total_frames,
            })
    except Exception as exc:
        meta["error"] = str(exc)
    return meta


def stats_summary(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    nums = sorted(float(v) for v in values if v is not None)
    if not nums:
        return {"count": 0, "min": None, "max": None, "median": None, "average": None}
    return {
        "count": len(nums),
        "min": nums[0],
        "max": nums[-1],
        "median": statistics.median(nums),
        "average": sum(nums) / len(nums),
    }


def estimate_output_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / 4)))


def json_safe(value: Any, depth: int = 4, max_text: int = 1200) -> Any:
    if depth <= 0:
        return str(value)[:max_text]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:max_text]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v, depth - 1, max_text) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v, depth - 1, max_text) for v in list(value)[:50]]
    if hasattr(value, "to_dict"):
        try:
            return json_safe(value.to_dict(), depth - 1, max_text)
        except Exception:
            pass
    if hasattr(value, "label"):
        return {"label": str(value.label)}
    return str(value)[:max_text]


def compact_dataset_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keep: Dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("runtime_agent_") or key in {"frames"}:
            continue
        if key.startswith("_") and key not in {"_id", "_media_type"}:
            continue
        keep[key] = json_safe(value)
    return keep


def sample_to_row(sample: Any) -> Dict[str, Any]:
    try:
        row = sample.to_dict()
    except Exception:
        row = {}
    for field in ("filepath", "tags", "label", "ground_truth", "classification", "class", "hazard", "safety_label", "cosmos_analysis", "hf_repo", "hf_path"):
        if field in row or not hasattr(sample, field):
            continue
        try:
            row[field] = getattr(sample, field)
        except Exception:
            pass
    return compact_dataset_row(row)


def class_from_label_text(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    text = str(value).strip()
    lower = text.lower().replace("-", "_").replace(" ", "_")
    base_lower = Path(text).name.lower().replace("-", "_").replace(" ", "_")
    for class_id, info in WORKER_SAFETY_CLASSES.items():
        if lower == str(class_id) or lower.startswith(f"{class_id}_") or base_lower.startswith(f"{class_id}_"):
            return {"class_id": class_id, "label": info["label"], "hazardous": info["hazardous"], "source_value": text}
        if lower == info["slug"] or info["slug"] in lower:
            return {"class_id": class_id, "label": info["label"], "hazardous": info["hazardous"], "source_value": text}
        if text == info["label"]:
            return {"class_id": class_id, "label": info["label"], "hazardous": info["hazardous"], "source_value": text}
    return None


def expected_from_video(video: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    row = video.get("dataset_row") or {}
    candidates = [
        ("dataset_row.label", row.get("label")),
        ("video.label", video.get("label")),
        ("dataset_row.hf_path", row.get("hf_path")),
        ("video.name", video.get("name")),
        ("filepath", video.get("filepath")),
    ]
    for source, value in candidates:
        expected = class_from_label_text(value)
        if expected:
            expected["source"] = source
            return expected
    return None


def ensure_thumbnail(path: str) -> Optional[Path]:
    target = THUMBNAIL_DIR / f"{video_id(path)}.jpg"
    if target.exists():
        return target
    try:
        import av

        THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
        with av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if not stream:
                return None
            rate = float(stream.average_rate or 30)
            total_frames = int(stream.frames or 0)
            target_index = int(max(0, min(total_frames - 1, round(rate)))) if total_frames > 0 else int(max(0, round(rate)))
            for index, frame in enumerate(container.decode(stream)):
                if index < target_index:
                    continue
                image = frame.to_image().convert("RGB")
                image.thumbnail(THUMBNAIL_SIZE)
                image.save(target, format="JPEG", quality=82)
                return target
    except Exception as exc:
        log(f"Could not extract thumbnail for {Path(path).name}: {exc}")
    return None


def estimate_plan(meta: Dict[str, Any], fps: float, max_pixels: int, max_frames: int, model: str = "") -> Dict[str, Any]:
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    duration_s = float(meta.get("duration_s") or 0)
    source_frames = int(meta.get("total_frames") or 0)
    if model_prefers_file_url(model) or model_prefers_video_data(model):
        return {
            "mode": "native_video_url",
            "frames_passed": source_frames,
            "visual_tokens_est": None,
            "effective_pixels": width * height,
            "note": "Video is passed as video_url; backend samples frames internally.",
        }
    requested = max(1, int(round(duration_s * max(fps, 0.1)))) if duration_s else max(1, source_frames)
    frames_passed = requested if max_frames <= 0 else min(requested, max_frames)
    native_pixels = width * height
    effective_pixels = min(native_pixels, max_pixels) if native_pixels else max_pixels
    visual_tokens = int(frames_passed * effective_pixels / BASELINE_PIXELS * EMPIRICAL_VISUAL_TOKENS_PER_FRAME)
    return {
        "mode": "image_frames",
        "frames_passed": frames_passed,
        "requested_frames": requested,
        "max_frames": max_frames,
        "visual_tokens_est": visual_tokens,
        "total_tokens_est": visual_tokens + TEXT_TOKENS,
        "effective_pixels": effective_pixels,
        "native_pixels": native_pixels,
        "note": "Set max frames to 0 to disable the cap; fps still controls sampling.",
    }


def attach_meta(video: Dict[str, Any]) -> Dict[str, Any]:
    meta = get_video_meta(video["filepath"])
    video["meta"] = meta
    video["plan"] = estimate_plan(
        meta,
        STATE["defaults"]["fps"],
        STATE["defaults"]["max_pixels"],
        STATE["defaults"]["max_frames"],
        os.getenv("MODEL_NAME", ""),
    )
    thumbnail = ensure_thumbnail(video["filepath"])
    if thumbnail:
        video["thumbnail_path"] = str(thumbnail)
        video["thumbnail_url"] = f"/api/thumb/{video['id']}.jpg"
    expected = expected_from_video(video)
    if expected:
        video["expected"] = expected
    return video


def load_with_fiftyone(repo_id: str, max_videos: int) -> List[Dict[str, Any]]:
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh

    name = repo_id.replace("/", "_")
    if name in fo.list_datasets():
        dataset = fo.load_dataset(name)
    else:
        try:
            if max_videos > 0:
                dataset = fouh.load_from_hub(repo_id, dataset_name=name, max_samples=max_videos, persistent=True)
            else:
                dataset = fouh.load_from_hub(repo_id, dataset_name=name, persistent=True)
        except TypeError:
            try:
                dataset = fouh.load_from_hub(repo_id, dataset_name=name, persistent=True)
            except TypeError:
                dataset = fouh.load_from_hub(repo_id, persistent=True)

    videos: List[Dict[str, Any]] = []
    for sample in dataset:
        path = str(sample.filepath)
        if not is_video_path(path):
            continue
        row = sample_to_row(sample)
        label = row.get("label") or guess_label(sample)
        videos.append(attach_meta({
            "id": video_id(path),
            "name": Path(path).name,
            "filepath": path,
            "sample_id": str(sample.id),
            "label": label,
            "dataset_row": row,
            "source": "fiftyone",
        }))
        if max_videos > 0 and len(videos) >= max_videos:
            break
    if not videos:
        raise RuntimeError(f"FiftyOne loaded {repo_id}, but no video samples were found")
    update_state(dataset_source="fiftyone", fo_dataset_name=name)
    return videos


def guess_label(sample: Any) -> Optional[str]:
    for field in ("ground_truth", "label", "classification", "class", "hazard"):
        if not hasattr(sample, field):
            continue
        value = getattr(sample, field)
        if hasattr(value, "label"):
            return str(value.label)
        if value is not None and not isinstance(value, (dict, list)):
            return str(value)
    return None


def load_hf_sidecar_rows(repo_id: str, files: List[str]) -> Dict[str, Dict[str, Any]]:
    if "samples.json" not in files:
        return {}
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="samples.json")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = data.get("samples") if isinstance(data, dict) else data
    except Exception as exc:
        log(f"Could not load HF sidecar samples.json: {exc}")
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        compact = compact_dataset_row(row)
        filepath = str(row.get("filepath") or "")
        if not filepath:
            continue
        index[filepath] = compact
        index[Path(filepath).name] = compact
    return index


def load_with_hf_hub(repo_id: str, max_videos: int) -> List[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download, list_repo_files

    repo_files = list_repo_files(repo_id, repo_type="dataset")
    sidecar_rows = load_hf_sidecar_rows(repo_id, repo_files)
    files = [f for f in repo_files if is_video_path(f)]
    if not files:
        raise RuntimeError(f"No video files found in Hugging Face dataset {repo_id}")
    local_root = Path("/tmp/byo_video_datasets") / repo_id.replace("/", "_")
    local_root.mkdir(parents=True, exist_ok=True)
    videos: List[Dict[str, Any]] = []
    selected_files = files if max_videos <= 0 else files[:max_videos]
    for file_name in selected_files:
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file_name, local_dir=str(local_root))
        row = dict(sidecar_rows.get(file_name) or sidecar_rows.get(Path(file_name).name) or {})
        row.setdefault("hf_repo", repo_id)
        row.setdefault("hf_path", file_name)
        label = row.get("label")
        videos.append(attach_meta({
            "id": video_id(local),
            "name": file_name,
            "filepath": local,
            "sample_id": None,
            "label": label,
            "dataset_row": row,
            "source": "hf_hub",
        }))
    try:
        import fiftyone as fo

        name = repo_id.replace("/", "_") + "_runtime"
        if name in fo.list_datasets():
            dataset = fo.load_dataset(name)
            try:
                dataset.delete_samples(dataset.values("id"))
            except Exception:
                pass
        else:
            dataset = fo.Dataset(name, persistent=True)

        for video in videos:
            sample = fo.Sample(filepath=video["filepath"])
            sample["hf_repo"] = repo_id
            sample["hf_path"] = video["name"]
            if video.get("label"):
                sample["hf_label"] = str(video["label"])
            expected = video.get("expected") or {}
            if expected:
                sample["expected_class_id"] = expected.get("class_id")
                sample["expected_label"] = expected.get("label")
                sample["expected_is_hazardous"] = expected.get("hazardous")
            dataset.add_sample(sample)
            video["sample_id"] = str(sample.id)
            video["source"] = "fiftyone"
        update_state(dataset_source="fiftyone", fo_dataset_name=name)
        log(f"Wrapped HF files in local FiftyOne dataset {name}")
    except Exception as exc:
        log(f"Could not wrap HF files in FiftyOne: {exc}")
        update_state(dataset_source="hf_hub", fo_dataset_name=None)
    return videos


def load_dataset(repo_id: str, max_videos: int) -> List[Dict[str, Any]]:
    limit = "all" if max_videos <= 0 else str(max_videos)
    log(f"Loading dataset {repo_id} (up to {limit} videos)")
    try:
        videos = load_with_fiftyone(repo_id, max_videos)
        log(f"Loaded {len(videos)} videos with FiftyOne")
    except Exception as exc:
        log(f"FiftyOne load failed; falling back to huggingface_hub: {exc}")
        videos = load_with_hf_hub(repo_id, max_videos)
        log(f"Loaded {len(videos)} videos with huggingface_hub")
    update_state(dataset_repo=repo_id, videos=videos, results=[], progress={"done": 0, "total": 0, "errors": 0}, batch_metrics=None)
    return videos


def detect_server() -> Dict[str, Any]:
    info = {
        "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        "model": os.getenv("MODEL_NAME", ""),
        "backend": os.getenv("INFERENCE_BACKEND", "vllm"),
        **detect_instance_resources(),
    }
    if requests is None:
        info["error"] = f"requests import failed: {REQUESTS_IMPORT_ERROR}"
        update_state(server=info)
        return info
    try:
        resp = requests.get(info["base_url"].rstrip("/") + "/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models and not info["model"]:
            info["model"] = models[0].get("id") or models[0].get("root") or ""
        info["models"] = [m.get("id") or m.get("root") for m in models]
    except Exception as exc:
        info["error"] = str(exc)
    update_state(server=info)
    return info


def detect_instance_resources() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "instance": socket.gethostname(),
        "host_ip": detect_host_ip(),
        "user": os.getenv("USER") or os.getenv("LOGNAME") or "",
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        ).strip().splitlines()[0]
        name, used, free, total = [part.strip() for part in out.split(",")]
        info["gpu"] = name
        info["vram_used_mib"] = int(used)
        info["vram_free_mib"] = int(free)
        info["vram_total_mib"] = int(total)
    except Exception as exc:
        info["gpu_error"] = str(exc)
    try:
        path = os.getenv("COSMOS_DIR") or str(Path.home())
        usage = shutil.disk_usage(path)
        info["storage_path"] = path
        info["ssd_free_gb"] = round(usage.free / (1024 ** 3), 1)
        info["ssd_used_gb"] = round(usage.used / (1024 ** 3), 1)
        info["ssd_total_gb"] = round(usage.total / (1024 ** 3), 1)
    except Exception as exc:
        info["storage_error"] = str(exc)
    return info


def resize_to_max_pixels(image: Any, max_pixels: int) -> Any:
    width, height = image.size
    pixels = width * height
    if not max_pixels or pixels <= max_pixels:
        return image
    scale = (max_pixels / pixels) ** 0.5
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size)


def extract_frames_b64(path: str, fps: float, max_frames: int = 8, max_pixels: int = DEFAULT_MAX_PIXELS) -> List[str]:
    import av

    frames: List[str] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 30)
        total_frames = int(stream.frames or 0)
        if total_frames <= 0:
            duration_s = float(container.duration) / 1_000_000 if container.duration else 0.0
            total_frames = max(1, int(round(duration_s * rate)))
        requested = max(1, int(round(total_frames * (max(fps, 0.1) / max(rate, 0.1)))))
        if max_frames > 0 and requested > max_frames:
            wanted = max_frames
            target_indices = {int(round(i * (total_frames - 1) / max(wanted - 1, 1))) for i in range(wanted)}
        else:
            step = max(1, int(round(rate / max(fps, 0.1))))
            target_indices = set(range(0, total_frames, step))
            wanted = requested
        for index, frame in enumerate(container.decode(stream)):
            if index not in target_indices:
                continue
            image = frame.to_image().convert("RGB")
            image = resize_to_max_pixels(image, max_pixels)
            from io import BytesIO

            buf = BytesIO()
            image.save(buf, format="JPEG", quality=85)
            frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            if len(frames) >= wanted:
                break
    return frames


def model_prefers_file_url(model: str) -> bool:
    lower = model.lower()
    return "qwen" in lower or "nemotron" in lower


def model_prefers_video_data(model: str) -> bool:
    lower = model.lower()
    backend = os.getenv("INFERENCE_BACKEND", "").lower()
    return backend == "nim" or ("cosmos-reason" in lower and "nim" in lower)


def content_for_video(video_path: str, prompt: str, model: str, fps: float, max_pixels: int, max_frames: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = get_video_meta(video_path)
    plan = estimate_plan(meta, fps, max_pixels, max_frames, model)
    if model_prefers_video_data(model):
        mime = mimetypes.guess_type(video_path)[0] or "video/mp4"
        data = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")
        return [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}},
        ], plan
    if model_prefers_file_url(model):
        tmp = Path("/tmp") / ("byo_agent_" + Path(video_path).name)
        if Path(video_path).resolve() != tmp.resolve():
            shutil.copy2(video_path, tmp)
        return [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": tmp.as_uri()}},
        ], plan
    frames = extract_frames_b64(video_path, fps=fps, max_frames=max_frames, max_pixels=max_pixels)
    if not frames:
        raise RuntimeError("No frames could be extracted from the video")
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    plan["frames_passed"] = len(frames)
    return content, plan


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def response_text_from_choice(choice: Dict[str, Any]) -> str:
    delta = choice.get("delta") or {}
    if isinstance(delta.get("content"), str):
        return delta["content"]
    if isinstance(delta.get("reasoning_content"), str):
        return delta["reasoning_content"]
    if isinstance(choice.get("text"), str):
        return choice["text"]
    return ""


def raise_for_status_with_body(resp: Any) -> None:
    if resp.status_code < 400:
        return
    body = (resp.text or "").strip()
    detail = f"{resp.status_code} Client Error: {resp.reason} for url: {resp.url}"
    if body:
        detail += f" :: {body[:800]}"
    raise requests.HTTPError(detail, response=resp)


def post_chat_completion(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    request_started = time.monotonic()
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    stream_payload["stream_options"] = {"include_usage": True}
    resp = requests.post(url, headers=headers, json=stream_payload, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
    if resp.status_code >= 400 and "stream_options" in (resp.text or ""):
        stream_payload.pop("stream_options", None)
        resp = requests.post(url, headers=headers, json=stream_payload, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
    raise_for_status_with_body(resp)

    chunks: List[str] = []
    usage: Dict[str, Any] = {}
    first_token_at: Optional[float] = None
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            piece = response_text_from_choice(choice)
            if not piece:
                continue
            if first_token_at is None:
                first_token_at = time.monotonic()
            chunks.append(piece)

    request_ended = time.monotonic()
    text = "".join(chunks)
    output_tokens = usage.get("completion_tokens") if usage else None
    tokens_estimated = output_tokens is None
    if output_tokens is None:
        output_tokens = estimate_output_tokens(text)
    ttft = first_token_at - request_started if first_token_at is not None else None
    decode_seconds = request_ended - first_token_at if first_token_at is not None else request_ended - request_started
    output_tps = float(output_tokens) / decode_seconds if decode_seconds > 0 and output_tokens else 0.0
    total_tps = float(output_tokens) / (request_ended - request_started) if request_ended > request_started and output_tokens else 0.0
    return {
        "text": text,
        "usage": usage,
        "metrics": {
            "ttft_seconds": ttft,
            "request_seconds": request_ended - request_started,
            "decode_seconds": decode_seconds,
            "output_tokens": int(output_tokens),
            "output_tokens_estimated": tokens_estimated,
            "output_tokens_per_second": output_tps,
            "total_tokens_per_second": total_tps,
        },
    }


def evaluate_result(video: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    expected = video.get("expected") or {}
    if not expected:
        return {"has_expected": False}
    parsed = result.get("json") or {}
    pred_id = parsed.get("prediction_class_id")
    try:
        pred_id = int(pred_id) if pred_id is not None else None
    except Exception:
        pred_id = None
    pred_label = str(parsed.get("prediction_label") or "")
    pred_hazard = (parsed.get("hazard_detection") or {}).get("is_hazardous")
    class_id_match = pred_id == expected.get("class_id")
    label_match = pred_label == expected.get("label")
    hazard_match = pred_hazard == expected.get("hazardous") if pred_hazard is not None else False
    return {
        "has_expected": True,
        "expected_class_id": expected.get("class_id"),
        "expected_label": expected.get("label"),
        "expected_is_hazardous": expected.get("hazardous"),
        "predicted_class_id": pred_id,
        "predicted_label": pred_label,
        "predicted_is_hazardous": pred_hazard,
        "class_id_match": class_id_match,
        "label_match": label_match,
        "hazard_match": hazard_match,
        "is_correct": bool(class_id_match and label_match),
    }


def run_one(video: Dict[str, Any], system_prompt: str, user_prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError(f"requests import failed: {REQUESTS_IMPORT_ERROR}")
    overall_started = time.monotonic()
    server = detect_server()
    base_url = server.get("base_url") or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = server.get("model") or os.getenv("MODEL_NAME") or "cosmos-reason"
    headers = {"Content-Type": "application/json"}
    if os.getenv("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('VLLM_API_KEY')}"
    content, plan = content_for_video(
        video["filepath"],
        user_prompt,
        model,
        float(params.get("fps") or STATE["defaults"]["fps"]),
        int(params.get("max_pixels") or STATE["defaults"]["max_pixels"]),
        int(params.get("max_frames") if params.get("max_frames") is not None else STATE["defaults"]["max_frames"]),
    )
    preprocessing_seconds = time.monotonic() - overall_started
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": int(params.get("max_tokens") or STATE["defaults"]["max_tokens"]),
        "temperature": float(params.get("temperature") if params.get("temperature") is not None else STATE["defaults"]["temperature"]),
        "top_p": float(params.get("top_p") if params.get("top_p") is not None else STATE["defaults"]["top_p"]),
        "repetition_penalty": float(params.get("repetition_penalty") if params.get("repetition_penalty") is not None else STATE["defaults"]["repetition_penalty"]),
        "stream": False,
    }
    completion = post_chat_completion(base_url, headers, payload)
    text = completion["text"]
    metrics = dict(completion["metrics"])
    metrics["preprocessing_seconds"] = preprocessing_seconds
    metrics["e2e_seconds"] = time.monotonic() - overall_started
    parsed = parse_json_from_text(text)
    result = {
        "id": video["id"],
        "name": video["name"],
        "source_label": video.get("label"),
        "response": text,
        "json": parsed,
        "plan": plan,
        "params": params,
        "usage": completion.get("usage") or {},
        "metrics": metrics,
        "expected": video.get("expected"),
        "error": None,
    }
    result["evaluation"] = evaluate_result(video, result)
    write_fiftyone_result(video, result)
    return result


def write_fiftyone_result(video: Dict[str, Any], result: Dict[str, Any]) -> None:
    snap = snapshot()
    if snap.get("dataset_source") != "fiftyone" or not video.get("sample_id") or not snap.get("fo_dataset_name"):
        return
    try:
        import fiftyone as fo

        dataset = fo.load_dataset(snap["fo_dataset_name"])
        sample = dataset[video["sample_id"]]
        sample["runtime_agent_response"] = result.get("response") or ""
        sample["runtime_agent_plan"] = result.get("plan") or {}
        sample["runtime_agent_params"] = result.get("params") or {}
        sample["runtime_agent_metrics"] = result.get("metrics") or {}
        sample["runtime_agent_expected"] = result.get("expected") or {}
        sample["runtime_agent_evaluation"] = result.get("evaluation") or {}
        if (result.get("evaluation") or {}).get("has_expected"):
            sample["runtime_agent_correct"] = bool((result.get("evaluation") or {}).get("is_correct"))
        if result.get("json") is not None:
            sample["runtime_agent_json"] = result["json"]
            label = result["json"].get("prediction_label")
            if label:
                sample["runtime_agent_prediction"] = fo.Classification(label=str(label))
        if result.get("error"):
            sample["runtime_agent_error"] = str(result["error"])
        sample.save()
    except Exception as exc:
        log(f"Could not write result back to FiftyOne: {exc}")


def batch_summary(
    dataset_repo: str,
    results: List[Dict[str, Any]],
    total: int,
    errors: int,
    concurrency: int,
    started_monotonic: float,
    status: str,
    run_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    elapsed = max(time.monotonic() - started_monotonic, 0.0)
    metrics = [r.get("metrics") or {} for r in results]
    evaluations = [r.get("evaluation") or {} for r in results if (r.get("evaluation") or {}).get("has_expected")]
    correct = sum(1 for e in evaluations if e.get("is_correct"))
    hazard_correct = sum(1 for e in evaluations if e.get("hazard_match"))
    by_class: Dict[str, Dict[str, Any]] = {}
    for evaluation in evaluations:
        class_id = str(evaluation.get("expected_class_id"))
        item = by_class.setdefault(class_id, {"total": 0, "correct": 0, "label": evaluation.get("expected_label")})
        item["total"] += 1
        if evaluation.get("is_correct"):
            item["correct"] += 1
    for item in by_class.values():
        item["accuracy"] = item["correct"] / item["total"] if item["total"] else None
    ok = max(0, len(results) - errors)
    return {
        **(run_context or {}),
        "dataset_repo": dataset_repo,
        "status": status,
        "total": total,
        "completed": len(results),
        "ok": ok,
        "errors": errors,
        "concurrency": concurrency,
        "batch_wall_seconds": elapsed,
        "video_requests_per_second": (len(results) / elapsed) if elapsed > 0 else 0.0,
        "e2e_seconds": stats_summary(m.get("e2e_seconds") for m in metrics),
        "ttft_seconds": stats_summary(m.get("ttft_seconds") for m in metrics),
        "output_tokens_per_second": stats_summary(m.get("output_tokens_per_second") for m in metrics),
        "evaluation": {
            "evaluated": len(evaluations),
            "correct": correct,
            "accuracy": correct / len(evaluations) if evaluations else None,
            "hazard_correct": hazard_correct,
            "hazard_accuracy": hazard_correct / len(evaluations) if evaluations else None,
            "by_expected_class": by_class,
        },
    }


def compact_run_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for result in results:
        evaluation = result.get("evaluation") or {}
        parsed = result.get("json") or {}
        compact.append({
            "name": result.get("name"),
            "expected_label": evaluation.get("expected_label"),
            "predicted_label": parsed.get("prediction_label"),
            "predicted_class_id": parsed.get("prediction_class_id"),
            "is_correct": evaluation.get("is_correct"),
            "error": result.get("error"),
            "metrics": result.get("metrics") or {},
        })
    return compact


def make_run_context(run_label: str, system_prompt: str, user_prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    prompt_hash = hashlib.sha1((system_prompt + "\n---\n" + user_prompt).encode("utf-8", "ignore")).hexdigest()[:10]
    return {
        "run_id": time.strftime("%Y%m%d-%H%M%S") + "-" + prompt_hash,
        "run_label": run_label or "Custom prompt",
        "prompt_hash": prompt_hash,
        "reasoning_prompt": "<think>" in user_prompt.lower() or "reasoning" in run_label.lower(),
        "params": params,
    }


def run_one_recorded(video: Dict[str, Any], system_prompt: str, user_prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        return run_one(video, system_prompt, user_prompt, params)
    except Exception as exc:
        result = {
            "id": video["id"],
            "name": video["name"],
            "source_label": video.get("label"),
            "response": "",
            "json": None,
            "plan": video.get("plan"),
            "params": params,
            "metrics": {"e2e_seconds": time.monotonic() - started},
            "expected": video.get("expected"),
            "error": str(exc),
        }
        result["evaluation"] = evaluate_result(video, result)
        write_fiftyone_result(video, result)
        return result


def run_batch(ids: Iterable[str], concurrency: int, system_prompt: str, user_prompt: str, params: Dict[str, Any], run_label: str = "") -> None:
    snap = snapshot()
    videos_by_id = {v["id"]: v for v in snap["videos"]}
    selected = [videos_by_id[i] for i in ids if i in videos_by_id] or list(videos_by_id.values())
    dataset_repo = str(snap.get("dataset_repo") or DEFAULT_DATASET)
    batch_started = time.monotonic()
    run_context = make_run_context(run_label, system_prompt, user_prompt, params)
    update_state(
        running=True,
        results=[],
        progress={"done": 0, "total": len(selected), "errors": 0},
        batch_metrics=batch_summary(dataset_repo, [], len(selected), 0, concurrency, batch_started, "running", run_context),
    )
    log(f"Running {len(selected)} videos with concurrency={concurrency} ({run_context['run_label']}, {run_context['prompt_hash']})")
    results: List[Dict[str, Any]] = []
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        future_map = {pool.submit(run_one_recorded, v, system_prompt, user_prompt, params): v for v in selected}
        for future in concurrent.futures.as_completed(future_map):
            video = future_map[future]
            try:
                result = future.result()
                if result.get("error"):
                    errors += 1
                    log(f"Failed {video['name']}: {result['error']}")
                else:
                    log(f"Completed {video['name']}")
            except Exception as exc:
                errors += 1
                result = {"id": video["id"], "name": video["name"], "source_label": video.get("label"), "response": "", "json": None, "plan": video.get("plan"), "params": params, "metrics": {}, "expected": video.get("expected"), "error": str(exc)}
                result["evaluation"] = evaluate_result(video, result)
                write_fiftyone_result(video, result)
                log(f"Failed {video['name']}: {exc}")
            results.append(result)
            with STATE_LOCK:
                STATE["results"] = results
                STATE["progress"] = {"done": len(results), "total": len(selected), "errors": errors}
                STATE["batch_metrics"] = batch_summary(dataset_repo, results, len(selected), errors, concurrency, batch_started, "running", run_context)
            RESULTS_FILE.write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")
    final_summary = batch_summary(dataset_repo, results, len(selected), errors, concurrency, batch_started, "complete", run_context)
    with STATE_LOCK:
        STATE["running"] = False
        STATE["batch_metrics"] = final_summary
        STATE["batch_history"] = (STATE.get("batch_history") or [])[-19:] + [final_summary]
        STATE["run_history"] = (STATE.get("run_history") or [])[-5:] + [{"summary": final_summary, "results": compact_run_results(results)}]
    RESULTS_FILE.write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")
    log(f"Batch complete: {len(results) - errors} ok, {errors} errors")


def launch_fiftyone(port: int) -> str:
    global FIFTYONE_SESSION
    snap = snapshot()
    if snap.get("dataset_source") != "fiftyone" or not snap.get("fo_dataset_name"):
        raise RuntimeError("Load the dataset with FiftyOne before launching the FiftyOne app")
    import fiftyone as fo

    dataset = fo.load_dataset(snap["fo_dataset_name"])
    FIFTYONE_SESSION = fo.launch_app(dataset, address="0.0.0.0", port=port, auto=False)
    url = f"http://{os.getenv('HOST_IP') or detect_host_ip() or 'localhost'}:{port}/"
    log(f"FiftyOne app is available at {url}")
    return url


def detect_host_ip() -> Optional[str]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        return None
    return None


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cosmos BYO Video Runtime Agent</title>
<style>
:root { color-scheme: light; --ink:#1f2937; --muted:#6b7280; --line:#d8dee8; --panel:#ffffff; --bg:#f5f7fb; --accent:#0f766e; --warn:#b45309; --bad:#b91c1c; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
header { padding:18px 28px; border-bottom:1px solid var(--line); background:#fff; display:flex; align-items:center; justify-content:space-between; gap:20px; }
h1 { font-size:20px; margin:0; letter-spacing:0; }
main { max-width:1280px; margin:0 auto; padding:20px; display:grid; grid-template-columns: 340px 1fr; gap:18px; }
section, aside { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
input, textarea, select { width:100%; border:1px solid var(--line); border-radius:6px; padding:9px 10px; font:inherit; background:#fff; }
input[type=range] { padding:0; }
input[type=checkbox] { width:auto; }
textarea { min-height:120px; resize:vertical; }
button { border:0; border-radius:6px; padding:9px 12px; font-weight:650; color:#fff; background:var(--accent); cursor:pointer; }
button.secondary { background:#334155; }
button.warn { background:var(--warn); }
button:disabled { opacity:.55; cursor:not-allowed; }
.actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.kv { display:grid; grid-template-columns:110px 1fr; gap:6px; font-size:13px; color:var(--muted); }
.params { display:grid; grid-template-columns:repeat(2, minmax(220px, 1fr)); gap:10px 18px; margin:12px 0; }
.param-value { color:var(--ink); font-weight:650; float:right; }
.range-note { display:block; color:var(--muted); font-size:11px; line-height:1.3; margin-top:4px; }
.thumb { width:96px; height:54px; object-fit:cover; border:1px solid var(--line); border-radius:6px; background:#e5e7eb; display:block; }
td.metric { white-space:nowrap; color:var(--muted); }
details.meta-details { max-width:260px; }
details.meta-details pre { max-height:160px; overflow:auto; white-space:pre-wrap; font-size:11px; color:var(--muted); }
.toggle-row { display:flex; align-items:flex-start; gap:8px; color:var(--ink); margin:4px 0 10px; }
.toggle-row span { display:block; color:var(--muted); font-size:11px; line-height:1.3; margin-top:2px; }
.hint { color:var(--muted); font-size:12px; margin:6px 0 10px; }
.guide { display:grid; gap:8px; }
.step { border-left:3px solid var(--line); padding-left:10px; color:var(--muted); font-size:13px; }
.step strong { color:var(--ink); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { border-bottom:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; }
th { color:var(--muted); font-size:12px; font-weight:650; }
.results { max-height:420px; overflow:auto; border:1px solid var(--line); border-radius:6px; }
.log { height:150px; overflow:auto; background:#0f172a; color:#d1fae5; padding:10px; border-radius:6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; white-space:pre-wrap; }
.pill { display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:3px 8px; font-size:12px; color:var(--muted); background:#fff; }
.progress { height:8px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
.progress div { height:100%; width:0; background:var(--accent); transition:width .2s ease; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; padding:14px; } header { padding:14px; } }
</style>
</head>
<body>
<header>
  <h1>Cosmos BYO Video Runtime Agent</h1>
  <span class="pill" id="serverPill">checking backend</span>
</header>
<main>
<aside>
  <div class="guide">
    <div class="step"><strong>1. Backend</strong><br/>The agent uses the local OpenAI-compatible vLLM/NIM endpoint.</div>
    <div class="step"><strong>2. Dataset</strong><br/>Load a public Hugging Face dataset, then choose any number of videos.</div>
    <div class="step"><strong>3. Prompt</strong><br/>The worker-safety smoke prompt is preloaded and editable.</div>
    <div class="step"><strong>4. Run</strong><br/>Selected videos are processed concurrently and results are written back to FiftyOne when available.</div>
  </div>
  <label>Hugging Face dataset</label>
  <input id="repo" value="pjramg/Safe_Unsafe_Test" />
  <label>Max videos to load (0 = all)</label>
  <input id="maxVideos" type="number" min="0" value="20" />
  <label>Concurrency</label>
  <input id="concurrency" type="number" min="1" value="4" />
  <div class="actions">
    <button id="loadBtn">Load dataset</button>
    <button class="warn" id="smokeBtn">Run smoke</button>
    <button class="secondary" id="foBtn">Open FiftyOne</button>
  </div>
  <h3>Backend</h3>
  <div class="kv" id="serverKv"></div>
  <p class="hint" id="framePolicy"></p>
</aside>
<section>
  <div class="progress"><div id="bar"></div></div>
  <p id="progressText" class="pill">idle</p>
  <h3>Batch metrics</h3>
  <div class="kv" id="batchKv"></div>
  <label>Demo prompt</label>
  <select id="promptPreset"></select>
  <label>System instructions</label>
  <textarea id="systemPrompt"></textarea>
  <label>User prompt</label>
  <textarea id="userPrompt"></textarea>
  <h3>Parameters</h3>
  <label class="toggle-row"><input id="buildDefaultsToggle" type="checkbox" /><span><strong>Use build.nvidia.com parameter settings</strong><br/>Sets Temperature=0.6, Top P=0.3, Repetition Penalty=1.2. Turning it off restores the runtime recommendations.</span></label>
  <div class="params">
    <label>Sampling fps <span class="param-value" id="fpsValue"></span><input id="fpsSlider" type="range" min="1" max="8" step="1" /><span class="range-note" id="fpsHint"></span></label>
    <label>Max pixels / frame <span class="param-value" id="maxPixelsValue"></span><input id="maxPixelsSlider" type="range" min="65536" max="4194304" step="65536" /><span class="range-note" id="maxPixelsHint"></span></label>
    <label>Max output tokens <span class="param-value" id="maxTokensValue"></span><input id="maxTokensSlider" type="range" min="64" max="2048" step="64" /><span class="range-note" id="maxTokensHint"></span></label>
    <label>Temperature <span class="param-value" id="temperatureValue"></span><input id="temperatureSlider" type="range" min="0" max="1" step="0.05" /><span class="range-note" id="temperatureHint"></span></label>
    <label>Top P <span class="param-value" id="topPValue"></span><input id="topPSlider" type="range" min="0.01" max="1" step="0.01" /><span class="range-note" id="topPHint"></span></label>
    <label>Repetition penalty <span class="param-value" id="repPenaltyValue"></span><input id="repPenaltySlider" type="range" min="1" max="2" step="0.05" /><span class="range-note" id="repPenaltyHint"></span></label>
    <label>Max input frames <span class="param-value" id="maxFramesValue"></span><input id="maxFramesSlider" type="range" min="0" max="128" step="1" /><span class="range-note" id="maxFramesHint"></span></label>
  </div>
  <p class="hint" id="paramSummary"></p>
  <div class="actions">
    <button id="runBtn">Run selected videos</button>
    <button class="secondary" id="allBtn">Select all</button>
    <button class="secondary" id="noneBtn">Select none</button>
  </div>
  <h3>Videos</h3>
  <div class="results"><table><thead><tr><th></th><th>Preview</th><th>Name</th><th>Expected</th><th>HF row</th><th>Resolution</th><th>Duration</th><th>Source frames</th><th>Frames to VLM</th><th>Est visual tokens</th><th>Path</th></tr></thead><tbody id="videoRows"></tbody></table></div>
  <h3>Results</h3>
  <div class="results"><table><thead><tr><th>Video</th><th>Expected</th><th>Prediction</th><th>Match</th><th>Hazard</th><th>TTFT</th><th>Output tok/s</th><th>E2E</th><th>Description / error</th></tr></thead><tbody id="resultRows"></tbody></table></div>
  <h3>Batch history</h3>
  <div class="results"><table><thead><tr><th>Dataset</th><th>Prompt</th><th>Status</th><th>Videos</th><th>Errors</th><th>Accuracy</th><th>Batch E2E</th><th>Video req/s</th><th>Median video E2E</th><th>Prompt hash</th></tr></thead><tbody id="batchRows"></tbody></table></div>
  <h3>Runtime log</h3>
  <div class="log" id="log"></div>
</section>
</main>
<script>
let state = null;
let initialized = false;
function esc(s){ return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function api(path, body){ const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); const j = await r.json(); if(!r.ok) throw new Error(j.error||r.statusText); return j; }
function setBusy(message){ document.getElementById('progressText').textContent = message; document.getElementById('bar').style.width = '12%'; ['loadBtn','runBtn','smokeBtn','foBtn'].forEach(id=>document.getElementById(id).disabled=true); }
function checkedIds(){ return [...document.querySelectorAll('.pick:checked')].map(x=>x.value); }
function el(id){ return document.getElementById(id); }
function num(id){ return Number(document.getElementById(id).value); }
function params(){ return {fps:num('fpsSlider'),max_pixels:num('maxPixelsSlider'),max_tokens:num('maxTokensSlider'),temperature:num('temperatureSlider'),top_p:num('topPSlider'),repetition_penalty:num('repPenaltySlider'),max_frames:num('maxFramesSlider')}; }
function nativeVideoMode(){ const srv=state?.server||{}; const model=String(srv.model||'').toLowerCase(); const backend=String(srv.backend||'').toLowerCase(); return backend.includes('nim') || model.includes('qwen') || model.includes('nemotron'); }
function fmt(n, digits=0){ if(n === null || n === undefined || Number.isNaN(Number(n))) return ''; return Number(n).toLocaleString(undefined,{maximumFractionDigits:digits}); }
function fmtMetaValue(v){ return typeof v === 'number' ? fmt(v, Number.isInteger(v) ? 0 : 2) : String(v ?? ''); }
function sec(v){ return v === null || v === undefined || Number.isNaN(Number(v)) ? '' : `${Number(v).toFixed(2)}s`; }
function rate(v){ return v === null || v === undefined || Number.isNaN(Number(v)) ? '' : Number(v).toFixed(2); }
function statText(s){ if(!s || !s.count) return 'no samples yet'; return `min ${sec(s.min)} · max ${sec(s.max)} · median ${sec(s.median)} · avg ${sec(s.average)}`; }
function percent(v){ return v === null || v === undefined || Number.isNaN(Number(v)) ? '' : `${(Number(v)*100).toFixed(1)}%`; }
function expectedText(x){ return x ? `${esc(x.class_id)} ${esc(x.label)}` : ''; }
function rowSummary(row){ if(!row) return ''; const parts=[]; if(row.label) parts.push(`label=${row.label}`); if(row.tags) parts.push(`tags=${Array.isArray(row.tags) ? row.tags.join(',') : row.tags}`); if(row.hf_path) parts.push(`path=${row.hf_path}`); return parts.join(' · '); }
function detailsJson(label,obj){ if(!obj || Object.keys(obj).length===0) return ''; return `<details class="meta-details"><summary>${esc(label)}</summary><pre>${esc(JSON.stringify(obj,null,2))}</pre></details>`; }
function promptLabel(){ const s=el('promptPreset'); const opt=s && s.options ? s.options[s.selectedIndex] : null; return opt ? opt.textContent : 'Custom prompt'; }
function videoPlan(v){ const m=v.meta||{}; if(nativeVideoMode()) return {frames:'server', tokens:'server', note:'server-decoded'}; const p=params(); const duration=Number(m.duration_s||0); const requested=Math.max(1, Math.round((duration || 1) * p.fps)); const frames=p.max_frames<=0 ? requested : Math.min(requested, p.max_frames); const nativePx=Number(m.width||0)*Number(m.height||0); const effectivePx=nativePx ? Math.min(nativePx, p.max_pixels) : p.max_pixels; const tokens=Math.round(frames * effectivePx / 524288 * 128); return {frames, tokens, note:''}; }
function sliderLabel(id, label, suffix=''){ document.getElementById(id+'Value').textContent = label + suffix; }
function sliderHint(domId, key){ const m=(state.defaults.slider_meta||{})[key]||{}; const unit=m.unit ? ' '+m.unit : ''; const recommended = key === 'max_frames' && m.recommended === 0 ? 'disabled' : fmtMetaValue(m.recommended); document.getElementById(domId+'Hint').textContent = `min ${fmtMetaValue(m.min)}${unit} · max ${fmtMetaValue(m.max)}${unit} · recommended ${recommended}${unit}. ${m.note||''}`; }
function renderParamLabels(){ const p=params(); sliderLabel('fps', p.fps, ' fps'); sliderLabel('maxPixels', fmt(p.max_pixels)); sliderLabel('maxTokens', fmt(p.max_tokens)); sliderLabel('temperature', p.temperature.toFixed(2)); sliderLabel('topP', p.top_p.toFixed(2)); sliderLabel('repPenalty', p.repetition_penalty.toFixed(2)); sliderLabel('maxFrames', p.max_frames === 0 ? 'disabled' : fmt(p.max_frames)); sliderHint('fps','fps'); sliderHint('maxPixels','max_pixels'); sliderHint('maxTokens','max_tokens'); sliderHint('temperature','temperature'); sliderHint('topP','top_p'); sliderHint('repPenalty','repetition_penalty'); sliderHint('maxFrames','max_frames'); const d=state.defaults; const buildOn=!!el('buildDefaultsToggle')?.checked; const mode=nativeVideoMode() ? 'native video_url; backend samples frames internally' : `image-frame mode; max input frames cap is ${p.max_frames === 0 ? 'disabled' : p.max_frames}`; const buildText=buildOn ? 'build.nvidia.com defaults are ON: temperature 0.6, top P 0.3, repetition 1.2. ' : ''; document.getElementById('paramSummary').textContent = `${buildText}Recommended: fps ${d.slider_meta.fps.recommended}, max pixels ${fmt(d.slider_meta.max_pixels.recommended)}, max tokens ${d.slider_meta.max_tokens.recommended}, temperature ${d.slider_meta.temperature.recommended}, top P ${d.slider_meta.top_p.recommended}, repetition ${d.slider_meta.repetition_penalty.recommended}, max frames ${d.slider_meta.max_frames.recommended}. Frame policy: ${mode}.`; }
function applySliderMeta(){ const map=[['fpsSlider','fps'],['maxPixelsSlider','max_pixels'],['maxTokensSlider','max_tokens'],['temperatureSlider','temperature'],['topPSlider','top_p'],['repPenaltySlider','repetition_penalty'],['maxFramesSlider','max_frames']]; for(const [id,key] of map){ const m=(state.defaults.slider_meta||{})[key]||{}; const el=document.getElementById(id); if(m.min !== undefined) el.min=m.min; if(m.max !== undefined) el.max=m.max; if(m.step !== undefined) el.step=m.step; } }
function applyBuildDefaults(checked){ const d=state.defaults; const b=d.build_defaults||{}; const m=d.slider_meta||{}; el('temperatureSlider').value = checked ? b.temperature : m.temperature.recommended; el('topPSlider').value = checked ? b.top_p : m.top_p.recommended; el('repPenaltySlider').value = checked ? b.repetition_penalty : m.repetition_penalty.recommended; render(); }
function initControls(){ if(initialized || !state) return; const d=state.defaults; applySliderMeta(); document.getElementById('repo').value = state.dataset_repo; document.getElementById('maxVideos').value = d.max_videos; document.getElementById('concurrency').value = d.concurrency; document.getElementById('systemPrompt').value = d.system_prompt; document.getElementById('userPrompt').value = d.user_prompt; document.getElementById('fpsSlider').value = d.fps; document.getElementById('maxPixelsSlider').value = d.max_pixels; document.getElementById('maxTokensSlider').value = d.max_tokens; document.getElementById('temperatureSlider').value = d.temperature; document.getElementById('topPSlider').value = d.top_p; document.getElementById('repPenaltySlider').value = d.repetition_penalty; document.getElementById('maxFramesSlider').value = d.max_frames; document.getElementById('buildDefaultsToggle').checked = !!d.use_build_defaults; const presets=d.prompt_presets||[]; document.getElementById('promptPreset').innerHTML = presets.map((p,i)=>`<option value="${i}">${esc(p.label)}${p.reasoning?' (reasoning)':''}</option>`).join(''); initialized = true; if(d.use_build_defaults) applyBuildDefaults(true); }
function render(){ if(!state) return; initControls(); renderParamLabels();
 const srv = state.server || {}; document.getElementById('serverPill').textContent = srv.error ? 'backend unavailable' : (srv.model ? 'model: '+srv.model : 'backend ready'); const rows=[['instance', srv.instance],['host_ip', srv.host_ip],['backend', srv.backend],['model', srv.model],['base_url', srv.base_url],['gpu', srv.gpu],['vram', srv.vram_total_mib ? `${fmt(srv.vram_free_mib)} MiB free / ${fmt(srv.vram_total_mib)} MiB total` : srv.gpu_error],['ssd', srv.ssd_total_gb ? `${fmt(srv.ssd_free_gb,1)} GB free / ${fmt(srv.ssd_total_gb,1)} GB total (${srv.storage_path})` : srv.storage_error]]; document.getElementById('serverKv').innerHTML = rows.map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v||'')}</div>`).join('');
 document.getElementById('framePolicy').textContent = nativeVideoMode() ? 'This backend receives a video_url; frame count and visual tokens are sampled by the model server.' : 'This backend receives sampled image frames; the runtime agent controls fps, max pixels, and max input frames.';
 const prog = state.progress || {done:0,total:0,errors:0}; const pct = prog.total ? Math.round(100*prog.done/prog.total) : 0; document.getElementById('bar').style.width = pct+'%'; document.getElementById('progressText').textContent = state.running ? `${prog.done}/${prog.total} running, ${prog.errors} errors` : `${prog.done}/${prog.total} complete, ${prog.errors} errors`;
 const bm=state.batch_metrics||{}; const ev=bm.evaluation||{}; const bmRows=bm.total ? [['dataset',bm.dataset_repo],['prompt',`${bm.run_label||''} (${bm.prompt_hash||''})`],['status',bm.status],['completed',`${bm.completed}/${bm.total} (${bm.errors} errors)`],['accuracy',ev.evaluated ? `${ev.correct}/${ev.evaluated} (${percent(ev.accuracy)})` : 'no expected labels'],['hazard accuracy',ev.evaluated ? `${ev.hazard_correct}/${ev.evaluated} (${percent(ev.hazard_accuracy)})` : 'no expected labels'],['batch E2E',sec(bm.batch_wall_seconds)],['video requests/sec',rate(bm.video_requests_per_second)],['video E2E stats',statText(bm.e2e_seconds)],['TTFT stats',statText(bm.ttft_seconds)],['output tok/s stats',statText(bm.output_tokens_per_second)]] : [['batch','No batch has run yet']]; document.getElementById('batchKv').innerHTML = bmRows.map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v||'')}</div>`).join('');
 const selected = new Set(checkedIds()); document.getElementById('videoRows').innerHTML = (state.videos||[]).map(v=>{ const m=v.meta||{}; const plan=videoPlan(v); const thumb=v.thumbnail_url ? `<img class="thumb" src="${esc(v.thumbnail_url)}" alt="">` : ''; const row=v.dataset_row||{}; return `<tr><td><input class="pick" type="checkbox" value="${esc(v.id)}" ${selected.size===0||selected.has(v.id)?'checked':''}></td><td>${thumb}</td><td>${esc(v.name)}</td><td>${expectedText(v.expected)}</td><td>${esc(rowSummary(row))}${detailsJson('row',row)}</td><td>${fmt(m.width)}x${fmt(m.height)}</td><td>${fmt(m.duration_s,1)}s</td><td>${fmt(m.total_frames)}</td><td>${esc(plan.frames)}</td><td>${esc(plan.tokens)}</td><td>${esc(v.filepath)}</td></tr>`; }).join('');
 document.getElementById('resultRows').innerHTML = (state.results||[]).map(r=>{ const j=r.json||{}; const hz=j.hazard_detection||{}; const ev=r.evaluation||{}; const pred=j.prediction_label ? `${esc(j.prediction_class_id)} ${esc(j.prediction_label)}` : ''; const expected=ev.has_expected ? `${esc(ev.expected_class_id)} ${esc(ev.expected_label)}` : ''; const match=ev.has_expected ? (ev.is_correct ? 'correct' : 'miss') : ''; const plan=r.plan||{}; const met=r.metrics||{}; const tokenSuffix=met.output_tokens_estimated ? ' est' : ''; const meta = plan.frames_passed ? `frames=${esc(plan.frames_passed)} tokens=${esc(plan.visual_tokens_est||'server')}` : ''; const desc=r.error ? `<span style="color:var(--bad)">${esc(r.error)}</span>` : esc((meta ? meta+' · ' : '') + (j.video_description||r.response||'')); return `<tr><td>${esc(r.name)}</td><td>${expected}</td><td>${pred}</td><td>${esc(match)}</td><td>${esc(hz.is_hazardous)}</td><td class="metric">${sec(met.ttft_seconds)}</td><td class="metric">${rate(met.output_tokens_per_second)}${tokenSuffix}</td><td class="metric">${sec(met.e2e_seconds)}</td><td>${desc}${detailsJson('output',{json:r.json, evaluation:r.evaluation, usage:r.usage})}</td></tr>`; }).join('');
 document.getElementById('batchRows').innerHTML = (state.batch_history||[]).slice().reverse().map(b=>{ const ev=b.evaluation||{}; return `<tr><td>${esc(b.dataset_repo)}</td><td>${esc(b.run_label||'')}</td><td>${esc(b.status)}</td><td>${esc(b.completed)}/${esc(b.total)}</td><td>${esc(b.errors)}</td><td>${ev.evaluated ? `${esc(ev.correct)}/${esc(ev.evaluated)} (${percent(ev.accuracy)})` : ''}</td><td class="metric">${sec(b.batch_wall_seconds)}</td><td class="metric">${rate(b.video_requests_per_second)}</td><td class="metric">${sec(b.e2e_seconds?.median)}</td><td>${esc(b.prompt_hash||'')}</td></tr>`; }).join('');
 document.getElementById('log').textContent = (state.logs||[]).join('\\n');
 document.getElementById('loadBtn').disabled = state.running; document.getElementById('runBtn').disabled = state.running; document.getElementById('smokeBtn').disabled = state.running; document.getElementById('foBtn').disabled = state.running; }
async function poll(){ const r = await fetch('/api/state'); state = await r.json(); render(); }
document.getElementById('promptPreset').onchange = ()=>{ const p=(state.defaults.prompt_presets||[])[Number(el('promptPreset').value)]; if(!p) return; el('systemPrompt').value=p.system_prompt; el('userPrompt').value=p.user_prompt; };
['fpsSlider','maxPixelsSlider','maxTokensSlider','temperatureSlider','topPSlider','repPenaltySlider','maxFramesSlider'].forEach(id=>document.getElementById(id).oninput=render);
document.getElementById('buildDefaultsToggle').onchange = ()=>applyBuildDefaults(el('buildDefaultsToggle').checked);
document.getElementById('loadBtn').onclick = async()=>{ try{ setBusy('Loading dataset from Hugging Face...'); await api('/api/load',{repo_id:el('repo').value,max_videos:Number(el('maxVideos').value)}); await poll(); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('runBtn').onclick = async()=>{ try{ setBusy('Starting selected-video batch...'); await api('/api/run',{ids:checkedIds(),concurrency:Number(el('concurrency').value),prompt_label:promptLabel(),system_prompt:el('systemPrompt').value,user_prompt:el('userPrompt').value,...params()}); await poll(); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('smokeBtn').onclick = async()=>{ try{ setBusy('Loading smoke dataset and starting batch...'); await api('/api/smoke',{max_videos:Number(el('maxVideos').value),concurrency:Number(el('concurrency').value),...params()}); await poll(); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('foBtn').onclick = async()=>{ try{ setBusy('Opening FiftyOne app...'); const j=await api('/api/fiftyone',{}); await poll(); alert('FiftyOne: '+j.url); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('allBtn').onclick = ()=>{ document.querySelectorAll('.pick').forEach(x=>x.checked=true); };
document.getElementById('noneBtn').onclick = ()=>{ document.querySelectorAll('.pick').forEach(x=>x.checked=false); };
poll(); setInterval(poll, 1500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self.send_text(INDEX_HTML, "text/html")
        elif self.path == "/api/state":
            detect_server()
            self.send_json(snapshot())
        elif self.path.startswith("/api/thumb/"):
            thumb_id = Path(urllib.parse.urlparse(self.path).path).name.rsplit(".", 1)[0]
            video = next((v for v in snapshot().get("videos", []) if v.get("id") == thumb_id), None)
            thumb_value = video.get("thumbnail_path") if video else None
            thumb = Path(thumb_value) if thumb_value else None
            if not thumb or not thumb.exists():
                self.send_error(404)
                return
            self.send_file(thumb, "image/jpeg")
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self.read_json()
            if self.path == "/api/load":
                videos = load_dataset(str(payload.get("repo_id") or DEFAULT_DATASET), int_payload(payload, "max_videos", 20))
                self.send_json({"videos": videos})
            elif self.path == "/api/run":
                if snapshot().get("running"):
                    raise RuntimeError("A batch is already running")
                thread = threading.Thread(
                    target=run_batch,
                    args=(
                        payload.get("ids") or [],
                        int(payload.get("concurrency") or 4),
                        str(payload.get("system_prompt") or WORKER_SAFETY_SYSTEM),
                        str(payload.get("user_prompt") or WORKER_SAFETY_USER),
                        params_from_payload(payload),
                        str(payload.get("prompt_label") or "Custom prompt"),
                    ),
                    daemon=True,
                )
                thread.start()
                self.send_json({"ok": True})
            elif self.path == "/api/smoke":
                if snapshot().get("running"):
                    raise RuntimeError("A batch is already running")
                max_videos = int_payload(payload, "max_videos", 2)
                concurrency = int_payload(payload, "concurrency", 2)
                load_dataset(DEFAULT_DATASET, max_videos)
                ids = [v["id"] for v in snapshot()["videos"]]
                params = params_from_payload(payload)
                thread = threading.Thread(target=run_batch, args=(ids, concurrency, WORKER_SAFETY_SYSTEM, WORKER_SAFETY_USER, params, "Worker safety smoke"), daemon=True)
                thread.start()
                self.send_json({"ok": True, "dataset": DEFAULT_DATASET, "videos": len(ids)})
            elif self.path == "/api/fiftyone":
                url = launch_fiftyone(int(payload.get("port") or os.getenv("FIFTYONE_PORT", "5151")))
                self.send_json({"url": url})
            else:
                self.send_error(404)
        except Exception as exc:
            log(f"API error: {exc}")
            self.send_json({"error": str(exc)}, status=500)

    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def int_payload(payload: Dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key)
    if value is None or value == "":
        return default
    return int(value)


def params_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = STATE["defaults"]
    return {
        "fps": float(payload.get("fps") if payload.get("fps") is not None else defaults["fps"]),
        "max_pixels": int(payload.get("max_pixels") if payload.get("max_pixels") is not None else defaults["max_pixels"]),
        "max_tokens": int(payload.get("max_tokens") if payload.get("max_tokens") is not None else defaults["max_tokens"]),
        "temperature": float(payload.get("temperature") if payload.get("temperature") is not None else defaults["temperature"]),
        "top_p": float(payload.get("top_p") if payload.get("top_p") is not None else defaults["top_p"]),
        "repetition_penalty": float(payload.get("repetition_penalty") if payload.get("repetition_penalty") is not None else defaults["repetition_penalty"]),
        "max_frames": int(payload.get("max_frames") if payload.get("max_frames") is not None else defaults["max_frames"]),
    }


def serve(host: str, port: int) -> None:
    detect_server()
    url = f"http://{host}:{port}/"
    Path(os.getenv("RUNTIME_AGENT_URL_FILE", "/tmp/byo_video_runtime_agent_url.txt")).write_text(url, encoding="utf-8")
    Path("/tmp/gradio_url.txt").write_text(url, encoding="utf-8")
    log(f"URL: {url}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def smoke(args: argparse.Namespace) -> int:
    load_dataset(args.dataset, args.max_videos)
    ids = [v["id"] for v in snapshot()["videos"]]
    run_batch(
        ids,
        args.concurrency,
        WORKER_SAFETY_SYSTEM,
        WORKER_SAFETY_USER,
        {
            "fps": args.fps,
            "max_pixels": DEFAULT_MAX_PIXELS,
            "max_tokens": args.max_tokens,
            "temperature": STATE["defaults"]["temperature"],
            "top_p": STATE["defaults"]["top_p"],
            "repetition_penalty": STATE["defaults"]["repetition_penalty"],
            "max_frames": STATE["defaults"]["max_frames"],
        },
    )
    snap = snapshot()
    print(json.dumps({"progress": snap["progress"], "results_file": str(RESULTS_FILE)}, indent=2))
    return 1 if snap["progress"].get("errors") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Cosmos BYO-video runtime-agent frontend")
    sub = parser.add_subparsers(dest="cmd")
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--host", default=os.getenv("RUNTIME_AGENT_HOST", "0.0.0.0"))
    serve_p.add_argument("--port", type=int, default=int(os.getenv("RUNTIME_AGENT_PORT", "7861")))
    smoke_p = sub.add_parser("smoke")
    smoke_p.add_argument("--dataset", default=DEFAULT_DATASET)
    smoke_p.add_argument("--max-videos", type=int, default=int(os.getenv("RUNTIME_AGENT_SMOKE_VIDEOS", "2")))
    smoke_p.add_argument("--concurrency", type=int, default=int(os.getenv("RUNTIME_AGENT_CONCURRENCY", "2")))
    smoke_p.add_argument("--fps", type=float, default=float(os.getenv("RUNTIME_AGENT_FPS", "1")))
    smoke_p.add_argument("--max-tokens", type=int, default=int(os.getenv("RUNTIME_AGENT_MAX_TOKENS", "1024")))
    args = parser.parse_args()
    if args.cmd == "smoke":
        return smoke(args)
    serve(args.host if hasattr(args, "host") else "0.0.0.0", args.port if hasattr(args, "port") else 7861)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
