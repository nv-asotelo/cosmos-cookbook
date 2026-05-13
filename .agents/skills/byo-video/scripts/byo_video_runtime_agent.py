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
import csv
import hashlib
import html
import json
import mimetypes
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
PRISM_CAPABILITY_DOMAINS = {
    "ER": "Embodied Reasoning",
    "CS": "Common Sense",
    "SP": "Spatial Perception",
    "IP": "Intuitive Physics",
    "MCQ": "Evaluation",
}
TEXT_SCORE_THRESHOLD = float(os.getenv("RUNTIME_AGENT_TEXT_SCORE_THRESHOLD", "0.75"))
DEFAULT_DATASET = os.getenv("RUNTIME_AGENT_DATASET", "pjramg/Safe_Unsafe_Test")
RESULTS_FILE = Path(os.getenv("RUNTIME_AGENT_RESULTS", "/tmp/byo_video_runtime_agent_results.json"))
THUMBNAIL_DIR = Path(os.getenv("RUNTIME_AGENT_THUMBNAILS", "/tmp/byo_video_runtime_agent_thumbnails"))
THUMBNAIL_SIZE = (220, 124)
REQUEST_TIMEOUT_SECONDS = float(os.getenv("RUNTIME_AGENT_REQUEST_TIMEOUT_SECONDS", "180"))
EXPORT_DIR = Path(os.getenv("RUNTIME_AGENT_EXPORTS", "/tmp/byo_video_runtime_agent_exports"))
CONTEXT_PATCH_PIXELS = int(os.getenv("RUNTIME_AGENT_CONTEXT_PATCH_PIXELS", str(14 * 14)))
CONTEXT_SAFETY_RESERVE = int(os.getenv("RUNTIME_AGENT_CONTEXT_SAFETY_RESERVE", "1024"))
DEFAULT_MODEL_MAX_LEN = int(os.getenv("RUNTIME_AGENT_MODEL_MAX_LEN", "32768"))
MODEL_FIT_VISUAL_TOKENS = int(os.getenv("RUNTIME_AGENT_MODEL_FIT_VISUAL_TOKENS", "6144"))
MODEL_FIT_TARGET_FRAMES = int(os.getenv("RUNTIME_AGENT_MODEL_FIT_TARGET_FRAMES", "4"))
TEXT_TOKENS = 50
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
HTTP_TIMEOUT_SECONDS = float(os.getenv("RUNTIME_AGENT_HTTP_TIMEOUT_SECONDS", "20"))
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
        "note": "NIM-local ignores this; server max_model_len governs." if INFERENCE_BACKEND == "nim_local" else "Maximum generated text tokens.",
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
PROMPT_SOURCE_CACHE_DIR = Path("/tmp/byo_video_prompt_sources")
PROMPT_SCAN_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".py", ".json", ".jsonl", ".ipynb", ".yaml", ".yml"}
PROMPT_SCAN_PDF_EXTENSIONS = {".pdf"}
PROMPT_SCAN_DIR_LIMIT = int(os.getenv("RUNTIME_AGENT_PROMPT_SCAN_DIR_LIMIT", "80"))
PROMPT_SCAN_MAX_TEXT_BYTES = int(os.getenv("RUNTIME_AGENT_PROMPT_SCAN_MAX_TEXT_BYTES", str(2 * 1024 * 1024)))
REASONING_PROFILES = {
    "cosmos_think_answer": {
        "id": "cosmos_think_answer",
        "label": "Cosmos Reason / Cosmos3",
        "prefix": "<think>\nyour reasoning\n</think>\n<answer>\n",
        "suffix": "\n</answer>",
        "note": "Cosmos Reason/Cosmos3 prompt scaffold.",
    },
    "qwen_think_answer": {
        "id": "qwen_think_answer",
        "label": "Qwen3-VL",
        "prefix": "<think>\nyour reasoning\n</think>\n<answer>\n",
        "suffix": "\n</answer>",
        "note": "Qwen3-VL thinking-compatible prompt scaffold.",
    },
    "nemotron_think_answer": {
        "id": "nemotron_think_answer",
        "label": "Nemotron VL / Omni",
        "prefix": "<think>\nyour reasoning\n</think>\n<answer>\n",
        "suffix": "\n</answer>",
        "note": "Nemotron reasoning prompt scaffold.",
    },
    "gemma_think_answer": {
        "id": "gemma_think_answer",
        "label": "Gemma",
        "prefix": "<think>\nyour reasoning\n</think>\n<answer>\n",
        "suffix": "\n</answer>",
        "note": "Generic Gemma-compatible reasoning scaffold.",
    },
    "generic_think_answer": {
        "id": "generic_think_answer",
        "label": "Generic VLM",
        "prefix": "<think>\nyour reasoning\n</think>\n<answer>\n",
        "suffix": "\n</answer>",
        "note": "Fallback reasoning scaffold for unrecognized OpenAI-compatible VLMs.",
    },
}
PAPER_FALLBACKS = {
    "2603.29281": {
        "title": "PRISM: A Multi-View Multi-Capability Retail Video Dataset for Embodied Vision-Language Models",
        "datasets": ["DreamVu/PRISM-100K"],
        "models": ["DreamVu/Cosmos-Reason2-2B-Retail-Grocery-EgoExo"],
        "prompt_presets": [
            {
                "label": "PRISM ER-1: next subtask",
                "system_prompt": GENERIC_SYSTEM,
                "user_prompt": "What is the next subtask the person will perform?",
                "reasoning": False,
                "source": "DreamVu/PRISM-100K dataset card sample",
            },
            {
                "label": "PRISM retail: person action",
                "system_prompt": GENERIC_SYSTEM,
                "user_prompt": "What is the person doing in this video?",
                "reasoning": False,
                "source": "DreamVu/Cosmos-Reason2-2B-Retail-Grocery-EgoExo model card",
            },
        ],
    }
}
EXPORT_SECTIONS = [
    {"id": "overview", "label": "Executive overview", "default": True},
    {"id": "run_metrics", "label": "Run metrics", "default": True},
    {"id": "evaluation", "label": "Evaluation summary", "default": True},
    {"id": "class_breakdown", "label": "Per-class breakdown", "default": True},
    {"id": "error_analysis", "label": "Errors and misses", "default": True},
    {"id": "result_table", "label": "Per-video table", "default": True},
    {"id": "samples", "label": "Representative samples", "default": True},
    {"id": "prompt_params", "label": "Prompt and parameters", "default": False},
    {"id": "infrastructure", "label": "Backend and instance", "default": False},
    {"id": "recommendations", "label": "Recommendations", "default": True},
]
EXPORT_CONTENT_TYPES = {
    "html": "text/html; charset=utf-8",
    "json": "application/json",
    "csv": "text/csv; charset=utf-8",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

STATE: Dict[str, Any] = {
    "dataset_repo": DEFAULT_DATASET,
    "dataset_source": None,
    "fo_dataset_name": None,
    "videos": [],
    "results": [],
    "running": False,
    "loading_dataset": False,
    "active_requests": {},
    "progress": {"done": 0, "total": 0, "errors": 0},
    "batch_metrics": None,
    "batch_history": [],
    "run_history": [],
    "paper_import": None,
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
        "context_patch_pixels": CONTEXT_PATCH_PIXELS,
        "context_safety_reserve": CONTEXT_SAFETY_RESERVE,
        "context_text_tokens": TEXT_TOKENS,
        "default_model_max_len": DEFAULT_MODEL_MAX_LEN,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "model_fit_visual_tokens": MODEL_FIT_VISUAL_TOKENS,
        "model_fit_target_frames": MODEL_FIT_TARGET_FRAMES,
        "reasoning_profiles": REASONING_PROFILES,
        "export_sections": EXPORT_SECTIONS,
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
        data = json.loads(json.dumps(STATE, default=str))
    data["server_epoch"] = time.time()
    return data


def update_state(**items: Any) -> None:
    with STATE_LOCK:
        STATE.update(items)


def update_progress(**items: Any) -> None:
    with STATE_LOCK:
        progress = dict(STATE.get("progress") or {})
        progress.update(items)
        progress["updated_epoch"] = items.get("updated_epoch", time.time())
        STATE["progress"] = progress


def update_load_progress(
    phase: str,
    event: str,
    *,
    done: Optional[int] = None,
    total: Optional[int] = None,
    current_file: Optional[str] = None,
    videos: Optional[List[Dict[str, Any]]] = None,
    last_error: Optional[str] = None,
) -> None:
    now = time.time()
    with STATE_LOCK:
        progress = dict(STATE.get("progress") or {})
        progress.update({
            "mode": "dataset_load",
            "phase": phase,
            "updated_epoch": now,
            "last_event": event,
        })
        if done is not None:
            progress["done"] = done
        if total is not None:
            progress["total"] = total
        if current_file is not None:
            progress["current_file"] = current_file
        if last_error is not None:
            progress["last_error"] = last_error
        STATE["progress"] = progress
        if videos is not None:
            STATE["videos"] = videos


def update_active_request(video: Dict[str, Any], stage: str, event: str, **items: Any) -> None:
    now = time.time()
    key = str(video.get("id") or video.get("filepath") or video.get("name") or f"request-{now}")
    with STATE_LOCK:
        active = dict(STATE.get("active_requests") or {})
        current = dict(active.get(key) or {})
        current.update({
            "id": key,
            "name": video.get("name") or Path(str(video.get("filepath") or key)).name,
            "stage": stage,
            "started_epoch": current.get("started_epoch") or now,
            "updated_epoch": now,
            "last_event": event,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        })
        current.update({k: v for k, v in items.items() if v is not None})
        active[key] = current
        progress = dict(STATE.get("progress") or {})
        progress.update({
            "mode": "inference",
            "phase": stage,
            "updated_epoch": now,
            "current_file": current["name"],
            "last_event": event,
            "active_count": len(active),
            "active_requests": list(active.values()),
        })
        STATE["active_requests"] = active
        STATE["progress"] = progress


def clear_active_request(video: Dict[str, Any], event: Optional[str] = None) -> None:
    now = time.time()
    key = str(video.get("id") or video.get("filepath") or video.get("name") or "")
    with STATE_LOCK:
        active = dict(STATE.get("active_requests") or {})
        active.pop(key, None)
        progress = dict(STATE.get("progress") or {})
        progress.update({
            "updated_epoch": now,
            "active_count": len(active),
            "active_requests": list(active.values()),
        })
        if event:
            progress["last_event"] = event
        STATE["active_requests"] = active
        STATE["progress"] = progress


class ClientInputError(RuntimeError):
    """Raised when the browser submitted settings that should be corrected."""


def hf_auth_headers() -> Dict[str, str]:
    headers = {"User-Agent": "cosmos-byo-video-runtime-agent"}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def http_get_text(url: str, timeout: float = HTTP_TIMEOUT_SECONDS) -> str:
    headers = hf_auth_headers()
    if requests is not None:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            body = (resp.text or "").strip()
            raise RuntimeError(f"HTTP {resp.status_code} for {url}: {body[:400]}")
        return resp.text
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as handle:
            return handle.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:400]}")


def http_get_bytes(url: str, timeout: float = HTTP_TIMEOUT_SECONDS) -> bytes:
    headers = hf_auth_headers()
    if requests is not None:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            body = (resp.text or "").strip()
            raise RuntimeError(f"HTTP {resp.status_code} for {url}: {body[:400]}")
        return resp.content
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as handle:
            return handle.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:400]}")


def http_get_bytes_prefix(url: str, max_bytes: int, timeout: float = HTTP_TIMEOUT_SECONDS) -> bytes:
    headers = hf_auth_headers()
    headers["Range"] = f"bytes=0-{max(0, max_bytes - 1)}"
    if requests is not None:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            body = (resp.text or "").strip()
            raise RuntimeError(f"HTTP {resp.status_code} for {url}: {body[:400]}")
        return resp.content[:max_bytes]
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as handle:
            return handle.read(max_bytes)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:400]}")


def http_get_json(url: str, timeout: float = HTTP_TIMEOUT_SECONDS) -> Any:
    return json.loads(http_get_text(url, timeout=timeout))


def plain_text_from_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", text)


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        clean = str(value or "").strip().strip("/")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def extract_arxiv_id(source: str) -> Optional[str]:
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", source or "")
    return match.group(1) if match else None


def extract_hf_repo_from_url(source: str, repo_type: str) -> Optional[str]:
    prefix = "datasets" if repo_type == "dataset" else ""
    if prefix:
        pattern = r"huggingface\.co/datasets/([^/?#]+/[^/?#]+)"
    else:
        pattern = r"huggingface\.co/(?!datasets/|papers/|spaces/)([^/?#]+/[^/?#]+)"
    match = re.search(pattern, source or "")
    return urllib.parse.unquote(match.group(1)).strip("/") if match else None


def clean_imported_prompt(text: Any) -> str:
    prompt = html.unescape(str(text or ""))
    prompt = prompt.replace("\\n", "\n")
    prompt = prompt.replace("<video>", "").replace("<image>", "")
    prompt = prompt.replace("⟨think⟩", "<think>").replace("⟨/think⟩", "</think>")
    prompt = re.sub(r"[ \t]+\n", "\n", prompt)
    return prompt.strip()


def prompt_source_label(source: str) -> str:
    value = str(source or "").strip()
    if not value:
        return "Prompt source"
    if value.startswith("http"):
        arxiv_id = extract_arxiv_id(value)
        if arxiv_id:
            return f"arXiv {arxiv_id}"
        repo = extract_hf_repo_from_url(value, "dataset") or extract_hf_repo_from_url(value, "model")
        if repo:
            return Path(repo).name
        return urllib.parse.urlparse(value).netloc or value
    return Path(value).stem or Path(value).name or value


def prompt_text_excerpt(text: str, limit: int = 64) -> str:
    clean = re.sub(r"\s+", " ", clean_imported_prompt(text)).strip(" \"'")
    return clean[: limit - 1] + "..." if len(clean) > limit else clean


def normalize_source_text(text: str) -> str:
    value = str(text or "").replace("\r", "\n").replace("\f", "\n")
    value = value.replace("⟨think⟩", "<think>").replace("⟨/think⟩", "</think>")
    value = value.replace("“", '"').replace("”", '"').replace("’", "'")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def source_kind_from_name(source: str) -> str:
    lower = str(source or "").lower()
    if "huggingface.co/datasets" in lower or "dataset" in lower:
        return "hf"
    if "arxiv" in lower or lower.endswith(".pdf") or "paper" in lower:
        return "paper"
    if any(part in lower for part in ("/recipes/", ".agents/skills", ".claude/skills", "skill")):
        return "recipe"
    return "import"


def looks_like_prompt_heading(line: str) -> bool:
    clean = prompt_text_excerpt(line, 120)
    if not clean:
        return False
    return bool(
        re.match(r"^(?:[A-Z]\.\d+|[A-Z]\s{1,3}\b|Appendix\b|Table\s+\d+|Figure\s+\d+)", clean)
        or re.search(r"\b(?:[A-Z]{2}(?:-[A-Z]+)?-\d+|ER-\d+|CS-[A-Z]-\d+|SP-\d+|IP-\d+)\b|\bMCQ\b|Video QA|Event Verification|Temporal Grounding|Dense Video Captioning|Object Detection|Pointing|Referring Exp|Single[- ]Object Tracking", clean, re.I)
    )


def qa_evaluation_type(question: str, answer: str) -> str:
    if re.search(r"(?m)^\s*[A-D][\.\)]\s+\S", question) and re.match(r"^\s*[A-D]\b", answer.strip(), re.I):
        return "mcq"
    if "<think>" in answer.lower():
        return "reasoning_text"
    if re.search(r'\{\s*"(?:start|end|bbox|point_2d)', answer, re.I):
        return "structured_json"
    if re.match(r"^\s*(yes|no)\b", answer.strip(), re.I):
        return "yes_no"
    return "open_text"


def qa_pair_looks_garbled(question: str, answer: str) -> bool:
    joined = f"{question}\n{answer}"
    if "Timestamp:" in joined or "Event Description:" in joined:
        return True
    if re.search(r"\b[A-D][\)\.]\s+", joined) and len(answer) > 160 and re.search(r"\d{2}:\d{2}|–|-", answer):
        return True
    if len(re.findall(r'["]?\s*[QA]:', joined)) > 2:
        return True
    return False


def prompt_presets_from_data(value: Any, label_prefix: str, source: str, limit: int = 24) -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []

    def visit(current: Any, path: str = "") -> None:
        if len(presets) >= limit:
            return
        if isinstance(current, dict):
            if isinstance(current.get("conversations") or current.get("messages"), list):
                preset = conversation_prompt_preset(current, f"{label_prefix}: conversation prompt {len(presets) + 1}", source)
                if preset:
                    presets.append(preset)
                return
            user_prompt = current.get("user_prompt") or current.get("prompt") or current.get("question") or current.get("query")
            system_prompt = current.get("system_prompt") or current.get("system") or current.get("instruction") or GENERIC_SYSTEM
            expected_answer = current.get("expected_answer") or current.get("answer") or current.get("label")
            if user_prompt and not isinstance(user_prompt, (dict, list)):
                label = current.get("label") or current.get("task") or current.get("name") or f"{label_prefix}: prompt {len(presets) + 1}"
                item = {
                    "label": prompt_text_excerpt(label, 92),
                    "system_prompt": clean_imported_prompt(system_prompt) or GENERIC_SYSTEM,
                    "user_prompt": clean_imported_prompt(user_prompt),
                    "reasoning": "<think>" in str(user_prompt).lower() or "<think>" in str(system_prompt).lower(),
                    "source": source,
                    "source_type": source_kind_from_name(source),
                    "paper_import": True,
                }
                if expected_answer and not isinstance(expected_answer, (dict, list)):
                    item.update({
                        "qa_pair": True,
                        "expected_answer": clean_imported_prompt(expected_answer),
                        "evaluation_type": qa_evaluation_type(str(user_prompt), str(expected_answer)),
                    })
                presets.append(item)
                return
            for child in current.values():
                visit(child, path)
        elif isinstance(current, list):
            for child in current[:200]:
                visit(child, path)
                if len(presets) >= limit:
                    break

    visit(value)
    return presets


def role_prompt_presets_from_text(text: str, label_prefix: str, source: str) -> List[Dict[str, Any]]:
    readable = normalize_source_text(plain_text_from_html(text) if "<" in text and ">" in text else text)
    patterns = [
        r'"role"\s*:\s*"system"\s*,\s*"content"\s*:\s*"((?:\\"|[^"])*)"',
        r"'role'\s*:\s*'system'\s*,\s*'content'\s*:\s*'((?:\\'|[^'])*)'",
    ]
    systems: List[str] = []
    for pattern in patterns:
        systems.extend(re.findall(pattern, readable, flags=re.DOTALL))
    user_patterns = [
        r'"role"\s*:\s*"user"\s*,\s*"content"\s*:\s*"((?:\\"|[^"])*)"',
        r"'role'\s*:\s*'user'\s*,\s*'content'\s*:\s*'((?:\\'|[^'])*)'",
        r'"user_prompt"\s*:\s*"((?:\\"|[^"])*)"',
        r'"prompt"\s*:\s*"((?:\\"|[^"])*)"',
    ]
    users: List[str] = []
    for pattern in user_patterns:
        users.extend(re.findall(pattern, readable, flags=re.DOTALL))
    presets = []
    for index, user_prompt in enumerate(users[:8]):
        system_prompt = systems[min(index, len(systems) - 1)] if systems else GENERIC_SYSTEM
        user_prompt = clean_imported_prompt(user_prompt)
        system_prompt = clean_imported_prompt(system_prompt) or GENERIC_SYSTEM
        if not user_prompt:
            continue
        presets.append({
            "label": f"{label_prefix}: sample prompt {index + 1}",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "reasoning": "<think>" in user_prompt.lower() or "<think>" in system_prompt.lower(),
            "source": source,
            "source_type": source_kind_from_name(source),
            "paper_import": True,
        })
    return presets


def qa_prompt_presets_from_text(text: str, label_prefix: str, source: str, limit: int = 24) -> List[Dict[str, Any]]:
    readable = normalize_source_text(plain_text_from_html(text) if "<" in text and ">" in text else text)
    readable = re.sub(r'\s+(["]?\s*Q:)', r"\n\1", readable)
    readable = re.sub(r'\s+(["]?\s*A:)', r"\n\1", readable)
    presets: List[Dict[str, Any]] = []
    current_heading = ""
    current_example = ""
    mode: Optional[str] = None
    question_lines: List[str] = []
    answer_lines: List[str] = []
    question_heading = ""
    question_example = ""

    def flush() -> None:
        nonlocal mode, question_lines, answer_lines, question_heading, question_example
        if len(presets) >= limit:
            mode = None
            question_lines = []
            answer_lines = []
            return
        question = clean_imported_prompt("\n".join(question_lines))
        answer = clean_imported_prompt("\n".join(answer_lines))
        if not question or not answer:
            mode = None
            question_lines = []
            answer_lines = []
            return
        if len(question) > 3000 or len(answer) > 3000:
            mode = None
            question_lines = []
            answer_lines = []
            return
        if qa_pair_looks_garbled(question, answer):
            mode = None
            question_lines = []
            answer_lines = []
            return
        heading_label = prompt_text_excerpt(question_heading, 48)
        example = prompt_text_excerpt(question_example or "", 36)
        question_label = prompt_text_excerpt(question, 50)
        if heading_label and example:
            label = f"{label_prefix}: {heading_label} {example}"
        elif heading_label:
            label = f"{label_prefix}: {heading_label}"
        else:
            label = f"{label_prefix}: QA {len(presets) + 1}"
        if question_label and question_label.lower() not in label.lower():
            label = f"{label} - {question_label}"
        presets.append({
            "label": prompt_text_excerpt(label, 110),
            "system_prompt": GENERIC_SYSTEM,
            "user_prompt": question,
            "reasoning": "<think>" in question.lower() or "<think>" in answer.lower(),
            "source": source,
            "source_type": source_kind_from_name(source),
            "paper_import": True,
            "qa_pair": True,
            "expected_answer": answer,
            "evaluation_type": qa_evaluation_type(question, answer),
        })
        mode = None
        question_lines = []
        answer_lines = []
        question_heading = ""
        question_example = ""

    for raw_line in readable.splitlines():
        line = clean_imported_prompt(raw_line)
        if not line:
            continue
        if re.match(r"^(?:Figure|Table)\s+\d+\b", line) or re.match(r"^(?:Appendix|References)\b", line):
            if mode == "answer":
                flush()
            current_heading = line if looks_like_prompt_heading(line) else current_heading
            continue
        example_match = re.match(r"^Example\s+(.+)$", line, flags=re.I)
        if example_match:
            if mode == "answer":
                flush()
            current_example = example_match.group(1)
            continue
        question_match = re.match(r'^"?\s*Q:\s*(.*)$', line, flags=re.I)
        if question_match:
            if mode == "answer":
                flush()
            mode = "question"
            question_lines = [question_match.group(1).strip()]
            answer_lines = []
            question_heading = current_heading
            question_example = current_example
            continue
        answer_match = re.match(r'^"?\s*A:\s*(.*)$', line, flags=re.I)
        if answer_match and mode in {"question", "answer"}:
            mode = "answer"
            answer_lines.append(answer_match.group(1).strip())
            continue
        if mode == "question":
            question_lines.append(line)
            continue
        if mode == "answer":
            if looks_like_prompt_heading(line):
                flush()
                current_heading = line
                current_example = ""
            else:
                answer_lines.append(line)
            continue
        if looks_like_prompt_heading(line):
            current_heading = line
            current_example = ""
        if len(presets) >= limit:
            break
    if mode == "answer":
        flush()
    return presets


def benchmark_task_prompt_presets_from_text(text: str, label_prefix: str, source: str) -> List[Dict[str, Any]]:
    readable = normalize_source_text(text)
    lower = readable.lower()
    presets: List[Dict[str, Any]] = []
    if "vantage-bench" in lower or "vantage bench" in lower:
        specs = [
            ("VANTAGE Event Verification", "Verify the operational hypothesis in the video. Answer Yes or No, then give one concise sentence of evidence.\n\nHypothesis: <replace with event hypothesis>", "yes_no"),
            ("VANTAGE Video QA (MCQ)", "Answer the multiple-choice video question. Return the option letter first, then a concise explanation.\n\nQuestion: <replace with question>\nA. <option A>\nB. <option B>\nC. <option C>\nD. <option D>", "mcq"),
            ("VANTAGE Temporal Grounding", "Locate when the queried event occurs in the video. Return JSON only with this schema: {\"start\": \"MM:SS.ss\", \"end\": \"MM:SS.ss\", \"evidence\": \"short visual cue\"}.\n\nQuery: <replace with temporal query>", "structured_json"),
            ("VANTAGE Dense Video Captioning", "Segment the video into chronological events. Return a JSON array of objects with start, end, and caption fields.", "structured_json"),
            ("VANTAGE Referring Expression", "Locate every instance described by the expression. Return JSON only as a list of bounding boxes [x1, y1, x2, y2].\n\nExpression: <replace with referring expression>", "structured_json"),
            ("VANTAGE Spatial Pointing", "Answer the spatial pointing multiple-choice question. Return only the option letter and the selected coordinate.\n\nQuestion: <replace with question and coordinate options>", "mcq"),
            ("VANTAGE Object Localization", "Locate every instance that belongs to the requested category. Return JSON only with class names and bbox coordinates [x1, y1, x2, y2].\n\nCategory: <replace with category>", "structured_json"),
            ("VANTAGE Single Object Tracking", "Given the initial object anchor, identify and track the object across the video. Return JSON with frame or timestamp keys and bbox coordinates [x1, y1, x2, y2].", "structured_json"),
        ]
        for label, user_prompt, eval_type in specs:
            presets.append({
                "label": f"{label_prefix}: {label}",
                "system_prompt": "You are an expert infrastructure video evaluator. Follow the requested output format exactly.",
                "user_prompt": user_prompt,
                "reasoning": False,
                "source": source,
                "source_type": "paper",
                "paper_import": True,
                "evaluation_type": eval_type,
            })
    return presets


def prompt_presets_from_text(text: str, label_prefix: str, source: str) -> List[Dict[str, Any]]:
    raw = str(text or "")
    presets: List[Dict[str, Any]] = []
    stripped = raw.strip()
    if stripped and stripped[0] in "[{":
        try:
            presets.extend(prompt_presets_from_data(json.loads(stripped), label_prefix, source))
        except Exception:
            pass
    presets.extend(role_prompt_presets_from_text(raw, label_prefix, source))
    presets.extend(qa_prompt_presets_from_text(raw, label_prefix, source))
    presets.extend(benchmark_task_prompt_presets_from_text(raw, label_prefix, source))
    return dedupe_prompt_presets(presets)


def conversation_prompt_preset(row: Dict[str, Any], label: str, source: str) -> Optional[Dict[str, Any]]:
    conversations = row.get("conversations") or row.get("messages")
    if not isinstance(conversations, list):
        return None
    system_prompt = GENERIC_SYSTEM
    user_prompt = ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in content)
        if role == "system" and content:
            system_prompt = clean_imported_prompt(content)
        elif role == "user" and content and not user_prompt:
            user_prompt = clean_imported_prompt(content)
    if not user_prompt:
        return None
    return {
        "label": label,
        "system_prompt": system_prompt or GENERIC_SYSTEM,
        "user_prompt": user_prompt,
        "reasoning": "<think>" in user_prompt.lower(),
        "source": source,
        "paper_import": True,
    }


def iter_conversation_rows(value: Any, limit: int = 8) -> Iterable[Dict[str, Any]]:
    found = 0
    stack = [value]
    while stack and found < limit:
        current = stack.pop(0)
        if isinstance(current, dict):
            if isinstance(current.get("conversations") or current.get("messages"), list):
                found += 1
                yield current
                continue
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current[:50])


def iter_json_array_prefix(raw: str, limit: int = 8) -> Iterable[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    index = 0
    length = len(raw)
    while index < length and raw[index].isspace():
        index += 1
    if index >= length or raw[index] != "[":
        return
    index += 1
    yielded = 0
    while index < length and yielded < limit:
        while index < length and raw[index].isspace():
            index += 1
        if index < length and raw[index] == ",":
            index += 1
            continue
        if index < length and raw[index] == "]":
            break
        try:
            value, index = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            yielded += 1
            yield value


def prompt_presets_from_hf_annotation_prefix(repo_id: str, annotation_path: str, limit: int = 8) -> List[Dict[str, Any]]:
    taxonomy: Dict[str, Any] = {}
    try:
        from huggingface_hub import hf_hub_download
        taxonomy_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="annotations/task_taxonomy.json")
        taxonomy = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
    except Exception:
        taxonomy = {}

    prefix_bytes = int(os.getenv("RUNTIME_AGENT_PROMPT_SCAN_ANNOTATION_PREFIX_BYTES", str(12 * 1024 * 1024)))
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=annotation_path)
        raw = Path(local).read_bytes()[:prefix_bytes].decode("utf-8", "replace")
    except Exception:
        encoded_repo = urllib.parse.quote(repo_id, safe="/")
        encoded_path = urllib.parse.quote(annotation_path, safe="/")
        raw = http_get_bytes_prefix(
            f"https://huggingface.co/datasets/{encoded_repo}/resolve/main/{encoded_path}",
            prefix_bytes,
            timeout=max(HTTP_TIMEOUT_SECONDS, 60),
        ).decode("utf-8", "replace")
    presets: List[Dict[str, Any]] = []
    for row in iter_json_array_prefix(raw, limit=limit * 10):
        try:
            compact = compact_annotation_row(row, taxonomy)
        except Exception:
            continue
        user_prompt = clean_imported_prompt(compact.get("user_prompt") or "")
        expected_answer = clean_imported_prompt(compact.get("expected_final_answer") or compact.get("expected_answer") or "")
        if not user_prompt or not expected_answer:
            continue
        task = compact.get("task") or f"annotation {len(presets) + 1}"
        presets.append({
            "label": f"{Path(repo_id).name}: {task} QA sample {len(presets) + 1}",
            "system_prompt": clean_imported_prompt(compact.get("system_prompt") or GENERIC_SYSTEM) or GENERIC_SYSTEM,
            "user_prompt": user_prompt,
            "reasoning": "<think>" in user_prompt.lower() or "<think>" in str(compact.get("expected_answer") or "").lower(),
            "source": f"{repo_id}/{annotation_path}",
            "source_type": "hf",
            "paper_import": True,
            "qa_pair": True,
            "expected_answer": expected_answer,
            "evaluation_type": compact.get("evaluation_type") or qa_evaluation_type(user_prompt, expected_answer),
        })
        if len(presets) >= limit:
            break
    return presets


def dedupe_prompt_presets(presets: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for preset in presets:
        system_prompt = clean_imported_prompt(preset.get("system_prompt") or GENERIC_SYSTEM) or GENERIC_SYSTEM
        user_prompt = clean_imported_prompt(preset.get("user_prompt") or "")
        if not user_prompt:
            continue
        key = (system_prompt, user_prompt)
        if key in seen:
            continue
        seen.add(key)
        item = dict(preset)
        item["system_prompt"] = system_prompt
        item["user_prompt"] = user_prompt
        item["reasoning"] = bool(item.get("reasoning") or "<think>" in user_prompt.lower())
        item["paper_import"] = True
        out.append(item)
    return out


def inspect_hf_dataset_for_prompts(repo_id: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"repo_id": repo_id, "prompt_presets": [], "warnings": []}
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    try:
        api_data = http_get_json(f"https://huggingface.co/api/datasets/{encoded_repo}")
        info["gated"] = api_data.get("gated")
        info["private"] = api_data.get("private")
        info["tags"] = api_data.get("tags") or []
    except Exception as exc:
        info["warnings"].append(f"Could not read HF dataset API metadata: {exc}")

    try:
        page = http_get_text(f"https://huggingface.co/datasets/{repo_id}")
        info["prompt_presets"].extend(prompt_presets_from_text(page, Path(repo_id).name, f"{repo_id} dataset card"))
    except Exception as exc:
        info["warnings"].append(f"Could not scrape dataset card prompts: {exc}")

    max_bytes = int(os.getenv("RUNTIME_AGENT_PROMPT_SCAN_MAX_BYTES", str(5 * 1024 * 1024)))
    try:
        tree = http_get_json(f"https://huggingface.co/api/datasets/{encoded_repo}/tree/main/annotations?expand=1")
        for item in tree if isinstance(tree, list) else []:
            path = str(item.get("path") or "")
            size = int(item.get("size") or 0)
            if not path.endswith((".json", ".jsonl")):
                continue
            if size > max_bytes:
                if path.endswith(".json") and path.startswith("annotations/") and "train" in Path(path).name:
                    try:
                        sampled = prompt_presets_from_hf_annotation_prefix(repo_id, path, limit=8)
                        info["prompt_presets"].extend(sampled)
                        info["warnings"].append(f"Sampled {len(sampled)} prompt/answer rows from large {path} instead of scanning all {size:,} bytes.")
                    except Exception as exc:
                        info["warnings"].append(f"Skipped {path} prompt scan because it is {size:,} bytes; limit is {max_bytes:,}; large annotation sampling failed: {exc}")
                else:
                    info["warnings"].append(f"Skipped {path} prompt scan because it is {size:,} bytes; limit is {max_bytes:,}.")
                continue
            try:
                raw = http_get_text(f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}")
                if path.endswith(".jsonl"):
                    rows = [json.loads(line) for line in raw.splitlines() if line.strip()][:20]
                else:
                    rows = json.loads(raw)
                for index, row in enumerate(iter_conversation_rows(rows, limit=6), 1):
                    preset = conversation_prompt_preset(row, f"{Path(repo_id).name}: annotation prompt {index}", path)
                    if preset:
                        info["prompt_presets"].append(preset)
            except Exception as exc:
                info["warnings"].append(f"Could not scan {path}: {exc}")
    except Exception as exc:
        info["warnings"].append(f"Could not inspect annotation files: {exc}")

    info["prompt_presets"] = dedupe_prompt_presets(info["prompt_presets"])
    return info


def hf_links_from_paper_page(arxiv_id: str) -> Tuple[List[str], List[str]]:
    try:
        page = http_get_text(f"https://huggingface.co/papers/{arxiv_id}")
    except Exception:
        return [], []
    dataset_links = re.findall(r'href=["\']/datasets/([^"\']+)["\']', page)
    model_links = []
    for repo in re.findall(r'href=["\']/([^"\'?#]+/[^"\'?#]+)["\']', page):
        first = repo.split("/", 1)[0]
        if first in {"datasets", "papers", "spaces", "models", "docs", "api", "settings", "front", "collections", "new"}:
            continue
        model_links.append(repo)
    return ordered_unique(dataset_links), ordered_unique(model_links)


def title_from_paper_markdown(arxiv_id: str) -> str:
    try:
        markdown = http_get_text(f"https://huggingface.co/papers/{arxiv_id}.md")
    except Exception:
        markdown = ""
    match = re.search(r"^Title:\s*(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()
    try:
        abs_page = http_get_text(f"https://arxiv.org/abs/{arxiv_id}")
        text = plain_text_from_html(abs_page)
        match = re.search(r"Title:\s*(.+?)\s+Authors:", text)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return ""


def resolve_local_prompt_source(source: str) -> Optional[Path]:
    value = os.path.expanduser(str(source or "").strip())
    if not value or re.match(r"^https?://", value):
        return None
    path = Path(value)
    if path.exists():
        return path
    cwd_path = Path.cwd() / value
    if cwd_path.exists():
        return cwd_path
    return None


def read_text_file_limited(path: Path) -> str:
    data = path.read_bytes()[:PROMPT_SCAN_MAX_TEXT_BYTES]
    return data.decode("utf-8", "replace")


def pdf_text_from_path(path: Path) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        raise RuntimeError("pdftotext is not installed, so PDF prompt extraction is unavailable on this host.")
    PROMPT_SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROMPT_SOURCE_CACHE_DIR / f"{video_id(str(path))}.txt"
    subprocess.run([exe, "-layout", str(path), str(out_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return read_text_file_limited(out_path)


def download_prompt_source(url: str) -> Path:
    PROMPT_SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".bin"
    target = PROMPT_SOURCE_CACHE_DIR / f"{video_id(url)}{suffix}"
    if not target.exists():
        target.write_bytes(http_get_bytes(url, timeout=max(HTTP_TIMEOUT_SECONDS, 60)))
    return target


def iter_prompt_scan_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    skipped = {".git", ".venv", "venv", "node_modules", "__pycache__", "outputs", "runs", ".cache"}
    count = 0
    for child in path.rglob("*"):
        if count >= PROMPT_SCAN_DIR_LIMIT:
            break
        if not child.is_file():
            continue
        if any(part in skipped for part in child.parts):
            continue
        suffix = child.suffix.lower()
        if suffix not in PROMPT_SCAN_TEXT_EXTENSIONS and suffix not in PROMPT_SCAN_PDF_EXTENSIONS:
            continue
        if suffix in PROMPT_SCAN_TEXT_EXTENSIONS:
            try:
                if child.stat().st_size > PROMPT_SCAN_MAX_TEXT_BYTES:
                    continue
            except Exception:
                continue
        count += 1
        yield child


def read_prompt_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in PROMPT_SCAN_PDF_EXTENSIONS:
        return pdf_text_from_path(path)
    return read_text_file_limited(path)


def default_recipe_prompt_paths() -> List[Path]:
    root = Path.cwd()
    return [
        root / ".agents" / "skills" / "byo-video",
        root / ".claude" / "skills",
        root / "docs" / "recipes",
        root / "docs" / "gallery" / "assets",
    ]


def prompt_text_sources(source: str, arxiv_id: Optional[str] = None) -> Tuple[List[Dict[str, str]], List[str]]:
    texts: List[Dict[str, str]] = []
    warnings: List[str] = []
    value = str(source or "").strip()
    local = resolve_local_prompt_source(value)
    if local:
        for path in iter_prompt_scan_files(local):
            try:
                texts.append({"source": str(path), "label": prompt_source_label(str(path)), "text": read_prompt_file(path)})
            except Exception as exc:
                warnings.append(f"Could not read {path}: {exc}")
        return texts, warnings

    if value.lower() in {"recipes", "recipe", "cookbook recipes"}:
        for path in default_recipe_prompt_paths():
            if not path.exists():
                continue
            for file_path in iter_prompt_scan_files(path):
                try:
                    texts.append({"source": str(file_path), "label": prompt_source_label(str(file_path)), "text": read_prompt_file(file_path)})
                except Exception as exc:
                    warnings.append(f"Could not read {file_path}: {exc}")
        return texts, warnings

    if arxiv_id:
        try:
            markdown = http_get_text(f"https://huggingface.co/papers/{arxiv_id}.md")
            texts.append({"source": f"https://huggingface.co/papers/{arxiv_id}", "label": f"arXiv {arxiv_id}", "text": markdown})
        except Exception as exc:
            warnings.append(f"Could not read Hugging Face paper markdown: {exc}")
        try:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            pdf_path = download_prompt_source(pdf_url)
            texts.append({"source": pdf_url, "label": f"PRISM paper {arxiv_id}" if arxiv_id == "2603.29281" else f"arXiv {arxiv_id}", "text": pdf_text_from_path(pdf_path)})
        except Exception as exc:
            warnings.append(f"Could not extract arXiv PDF prompts: {exc}")
        return texts, warnings

    if re.match(r"^https?://", value):
        try:
            if urllib.parse.urlparse(value).path.lower().endswith(".pdf"):
                path = download_prompt_source(value)
                texts.append({"source": value, "label": prompt_source_label(value), "text": pdf_text_from_path(path)})
            else:
                page = http_get_text(value)
                texts.append({"source": value, "label": prompt_source_label(value), "text": page})
        except Exception as exc:
            warnings.append(f"Could not read prompt source {value}: {exc}")
    return texts, warnings


def discover_paper_source(source: str, require_dataset: bool = True) -> Dict[str, Any]:
    source = str(source or "").strip()
    if not source:
        raise ClientInputError("Enter an arXiv ID, arXiv URL, Hugging Face source, local PDF, or recipe path.")
    arxiv_id = extract_arxiv_id(source)
    datasets = []
    models = []
    warnings: List[str] = []
    prompt_presets: List[Dict[str, Any]] = []
    title = ""

    dataset_repo = extract_hf_repo_from_url(source, "dataset")
    model_repo = extract_hf_repo_from_url(source, "model")
    if not dataset_repo and re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", source):
        dataset_repo = source
    if dataset_repo:
        datasets.append(dataset_repo)
    if model_repo and model_repo != dataset_repo:
        models.append(model_repo)

    if arxiv_id:
        title = title_from_paper_markdown(arxiv_id)
        paper_datasets, paper_models = hf_links_from_paper_page(arxiv_id)
        datasets.extend(paper_datasets)
        models.extend(paper_models)
        fallback = PAPER_FALLBACKS.get(arxiv_id)
        if fallback:
            title = title or fallback.get("title", "")
            datasets.extend(fallback.get("datasets", []))
            models.extend(fallback.get("models", []))
            prompt_presets.extend(fallback.get("prompt_presets", []))

    text_sources, text_warnings = prompt_text_sources(source, arxiv_id)
    warnings.extend(text_warnings)
    for text_source in text_sources:
        text = text_source.get("text") or ""
        label = text_source.get("label") or prompt_source_label(text_source.get("source") or source)
        extracted = prompt_presets_from_text(text, label, text_source.get("source") or source)
        prompt_presets.extend(extracted)

    datasets = ordered_unique(datasets)
    models = ordered_unique(models)
    dataset_details = []
    for repo_id in datasets[:5]:
        details = inspect_hf_dataset_for_prompts(repo_id)
        dataset_details.append(details)
        prompt_presets.extend(details.get("prompt_presets") or [])
        warnings.extend(details.get("warnings") or [])

    prompt_presets = dedupe_prompt_presets(prompt_presets)
    if not prompt_presets:
        prompt_presets = [{
            "label": "Paper import: general video QA",
            "system_prompt": GENERIC_SYSTEM,
            "user_prompt": DEFAULT_PROMPT,
            "reasoning": False,
            "source": "fallback",
            "paper_import": True,
        }]
        warnings.append("No exact prompt examples were found; using the generic video QA prompt.")
    if require_dataset and not datasets:
        raise ClientInputError("No Hugging Face dataset link was found for this paper/source. Use Import prompts to load prompt examples without loading a dataset.")

    return {
        "source": source,
        "arxiv_id": arxiv_id,
        "title": title,
        "datasets": datasets,
        "selected_dataset": datasets[0] if datasets else None,
        "models": models,
        "prompt_presets": prompt_presets,
        "dataset_details": dataset_details,
        "warnings": ordered_unique(warnings),
    }


def apply_paper_import(discovery: Dict[str, Any]) -> None:
    imported = dedupe_prompt_presets(discovery.get("prompt_presets") or [])
    primary = imported[0] if imported else {"system_prompt": GENERIC_SYSTEM, "user_prompt": DEFAULT_PROMPT}
    with STATE_LOCK:
        current_presets = [dict(p) for p in (STATE.get("defaults", {}).get("prompt_presets") or [])]
        saved_custom = [dict(p) for p in current_presets if p.get("custom")]
        imported_sources = {str(p.get("source") or "") for p in imported}
        previous_imports = [
            dict(p) for p in current_presets
            if p.get("paper_import") and str(p.get("source") or "") not in imported_sources
        ]
    merged = list(PROMPT_PRESETS) + saved_custom
    existing_labels = {preset["label"] for preset in merged}
    for preset in list(imported) + previous_imports:
        item = dict(preset)
        label = str(item.get("label") or "Paper import")
        if label in existing_labels:
            label = f"{label} ({len(existing_labels) + 1})"
        item["label"] = label
        existing_labels.add(label)
        merged.append(item)
    with STATE_LOCK:
        defaults = dict(STATE["defaults"])
        defaults.update({
            "system_prompt": primary.get("system_prompt") or GENERIC_SYSTEM,
            "user_prompt": primary.get("user_prompt") or DEFAULT_PROMPT,
            "prompt_presets": merged,
        })
        STATE["defaults"] = defaults
        if discovery.get("selected_dataset"):
            STATE["dataset_repo"] = discovery.get("selected_dataset") or STATE["dataset_repo"]
        STATE["paper_import"] = discovery


def add_prompt_preset(label: str, system_prompt: str, user_prompt: str, source: str = "runtime") -> Dict[str, Any]:
    system_prompt = clean_imported_prompt(system_prompt) or GENERIC_SYSTEM
    user_prompt = clean_imported_prompt(user_prompt)
    if not user_prompt:
        raise ClientInputError("Enter a user prompt before saving it.")
    label = clean_imported_prompt(label) or "Custom prompt"
    item = {
        "label": label,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "reasoning": "<think>" in user_prompt.lower() or "<think>" in system_prompt.lower(),
        "source": source,
        "custom": True,
    }
    with STATE_LOCK:
        defaults = dict(STATE["defaults"])
        presets = list(defaults.get("prompt_presets") or [])
        for existing in presets:
            if (
                clean_imported_prompt(existing.get("system_prompt") or GENERIC_SYSTEM) == system_prompt
                and clean_imported_prompt(existing.get("user_prompt") or "") == user_prompt
            ):
                return {"preset": existing, "duplicate": True, "count": len(presets)}
        labels = {str(p.get("label") or "") for p in presets}
        base_label = label
        suffix = 2
        while label in labels:
            label = f"{base_label} {suffix}"
            suffix += 1
        item["label"] = label
        presets.append(item)
        defaults["prompt_presets"] = presets
        defaults["system_prompt"] = system_prompt
        defaults["user_prompt"] = user_prompt
        STATE["defaults"] = defaults
    log(f"Saved prompt preset {label}")
    return {"preset": item, "duplicate": False, "count": len(presets)}


def import_paper_source(source: str, max_videos: int = 0, load_now: bool = True) -> Dict[str, Any]:
    discovery = discover_paper_source(source, require_dataset=load_now)
    apply_paper_import(discovery)
    selected = discovery.get("selected_dataset") or "no dataset selected"
    log(f"Imported prompt metadata for {discovery.get('arxiv_id') or source}: {selected}, {len(discovery.get('prompt_presets') or [])} prompt presets")
    with STATE_LOCK:
        progress = dict(STATE.get("progress") or {})
        if progress.get("mode") == "paper_import":
            progress.update({
                "phase": "dataset_selected" if discovery.get("selected_dataset") else "metadata_imported",
                "updated_epoch": time.time(),
                "last_event": f"Selected dataset {discovery.get('selected_dataset')}" if discovery.get("selected_dataset") else "Paper metadata imported; no dataset selected",
            })
            STATE["progress"] = progress
    if load_now and discovery.get("selected_dataset"):
        try:
            videos = load_dataset(str(discovery["selected_dataset"]), max_videos)
            discovery["loaded_videos"] = len(videos)
        except Exception as exc:
            discovery["load_error"] = str(exc)
            log(f"Paper dataset load failed: {exc}")
            with STATE_LOCK:
                progress = dict(STATE.get("progress") or {})
                progress.update({
                    "updated_epoch": time.time(),
                    "last_event": "Paper dataset load failed",
                    "last_error": str(exc),
                })
                STATE["progress"] = progress
    return discovery


def import_paper_worker(source: str, max_videos: int, load_now: bool = True) -> None:
    started_epoch = time.time()
    import_label = "paper metadata and dataset links" if load_now else "prompt examples"
    update_state(
        loading_dataset=True,
        progress={
            "mode": "paper_import",
            "phase": "paper_fetch",
            "done": 0,
            "total": 0,
            "errors": 0,
            "started_epoch": started_epoch,
            "updated_epoch": started_epoch,
            "last_result_epoch": None,
            "current_file": None,
            "last_event": f"Importing {import_label} from {source}",
            "last_error": None,
        },
    )
    log(f"Importing {import_label} from {source}")
    try:
        discovery = import_paper_source(source, max_videos, load_now)
        now = time.time()
        with STATE_LOCK:
            progress = dict(STATE.get("progress") or {})
            if progress.get("mode") == "paper_import":
                progress.update({
                    "phase": "complete",
                    "done": 1,
                    "total": 1,
                    "updated_epoch": now,
                    "finished_epoch": now,
                    "last_event": "Paper metadata imported",
                    "last_error": None,
                })
                STATE["progress"] = progress
                STATE["loading_dataset"] = False
            STATE["paper_import"] = discovery
    except Exception as exc:
        now = time.time()
        log(f"Paper import failed: {exc}")
        update_state(
            loading_dataset=False,
            progress={
                "mode": "paper_import",
                "phase": "failed",
                "done": 0,
                "total": 0,
                "errors": 1,
                "started_epoch": started_epoch,
                "updated_epoch": now,
                "finished_epoch": now,
                "last_event": "Paper import failed",
                "last_error": str(exc),
            },
        )
    finally:
        try:
            RESULTS_FILE.write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")
        except Exception:
            pass


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


def estimate_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    return estimate_output_tokens(system_prompt) + estimate_output_tokens(user_prompt) + TEXT_TOKENS


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


def capability_domain_for_task(task: Any) -> str:
    text = str(task or "").upper()
    if text.startswith("CS-"):
        return PRISM_CAPABILITY_DOMAINS["CS"]
    if text.startswith("ER-"):
        return PRISM_CAPABILITY_DOMAINS["ER"]
    if text.startswith("SP-"):
        return PRISM_CAPABILITY_DOMAINS["SP"]
    if text.startswith("IP-"):
        return PRISM_CAPABILITY_DOMAINS["IP"]
    if text.startswith("MCQ"):
        return PRISM_CAPABILITY_DOMAINS["MCQ"]
    return ""


def strip_video_placeholder(text: str) -> str:
    return re.sub(r"^\s*<video>\s*", "", str(text or ""), flags=re.IGNORECASE).strip()


def strip_thinking(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL).strip()
    if "</think>" in value.lower():
        value = re.split(r"</think>", value, flags=re.IGNORECASE)[-1].strip()
    return value


def conversation_content(row: Dict[str, Any], role: str) -> str:
    for message in row.get("conversations") or []:
        if isinstance(message, dict) and str(message.get("role") or "").lower() == role:
            return str(message.get("content") or "")
    return ""


def annotation_eval_type(row: Dict[str, Any], expected_answer: str) -> str:
    sft_type = str(row.get("sft_type") or "").lower()
    task = str(row.get("task") or "")
    if sft_type == "mcq" or task.upper().startswith("MCQ"):
        return "mcq"
    if "<think>" in expected_answer.lower() or sft_type == "reasoning":
        return "reasoning_text"
    return "open_text"


def compact_annotation_row(row: Dict[str, Any], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    video_path = str(row.get("video") or row.get("filepath") or "")
    task = str(row.get("task") or "")
    tax = taxonomy.get(task) if isinstance(taxonomy.get(task), dict) else {}
    system_prompt = conversation_content(row, "system") or GENERIC_SYSTEM
    user_prompt = strip_video_placeholder(conversation_content(row, "user"))
    expected_answer = conversation_content(row, "assistant")
    expected_final = strip_thinking(expected_answer)
    compact = {
        "annotation_id": row.get("id"),
        "task": task,
        "capability_domain": capability_domain_for_task(task),
        "domain": row.get("domain") or tax.get("domain") or (row.get("metadata") or {}).get("domain"),
        "sft_type": row.get("sft_type") or tax.get("sft_type"),
        "internal_name": tax.get("internal_name"),
        "hf_path": video_path,
        "video": video_path,
        "fps": row.get("fps"),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "expected_answer": expected_answer,
        "expected_final_answer": expected_final,
        "evaluation_type": annotation_eval_type(row, expected_answer),
        "metadata": json_safe(row.get("metadata") or {}),
    }
    return {key: value for key, value in compact.items() if value not in (None, "")}


def resolve_hf_video_path(video_path: str, files: Iterable[str]) -> Optional[str]:
    file_set = files if isinstance(files, set) else set(files)
    path = str(video_path or "").lstrip("/")
    candidates = [
        path,
        f"videos/{path}",
        str(Path("videos") / Path(path).name),
    ]
    parts = Path(path).parts
    if len(parts) >= 2:
        candidates.append(str(Path("videos") / parts[0] / parts[-1]))
    for candidate in candidates:
        if candidate in file_set:
            return candidate
    return None


def load_hf_annotation_samples(repo_id: str, files: List[str]) -> Dict[str, Any]:
    if "annotations/train.json" not in files:
        return {"samples": [], "by_path": {}, "taxonomy": {}}
    try:
        from huggingface_hub import hf_hub_download

        taxonomy: Dict[str, Any] = {}
        if "annotations/task_taxonomy.json" in files:
            update_load_progress("hf_metadata", "Loading PRISM task taxonomy")
            taxonomy_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="annotations/task_taxonomy.json")
            taxonomy = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
        update_load_progress("hf_metadata", "Loading PRISM annotation samples from annotations/train.json")
        annotations_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="annotations/train.json")
        rows = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"Could not load HF annotations/train.json: {exc}")
        return {"samples": [], "by_path": {}, "taxonomy": {}}
    samples: List[Dict[str, Any]] = []
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    total = len(rows) if isinstance(rows, list) else 0
    file_set = set(files)
    update_load_progress("hf_metadata", f"Indexing {total:,} PRISM annotation rows", done=0, total=total)
    for index, row in enumerate(rows if isinstance(rows, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        compact = compact_annotation_row(row, taxonomy)
        video_path = str(compact.get("hf_path") or "")
        resolved_path = resolve_hf_video_path(video_path, file_set)
        if not resolved_path or not is_video_path(resolved_path):
            continue
        compact["annotation_video"] = video_path
        compact["hf_path"] = resolved_path
        samples.append(compact)
        by_path.setdefault(resolved_path, []).append(compact)
        by_path.setdefault(video_path, []).append(compact)
        by_path.setdefault(Path(resolved_path).name, []).append(compact)
        if index % 10000 == 0:
            update_load_progress("hf_metadata", f"Indexed {index:,}/{total:,} PRISM annotation rows", done=index, total=total)
    update_load_progress("hf_metadata", f"Indexed {total:,}/{total:,} PRISM annotation rows", done=total, total=total)
    log(f"Loaded {len(samples):,} PRISM annotation samples from annotations/train.json")
    return {"samples": samples, "by_path": by_path, "taxonomy": taxonomy}


def expected_from_video(video: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    row = video.get("dataset_row") or {}
    if row.get("expected_answer"):
        return {
            "kind": "dataset_answer",
            "answer": row.get("expected_answer"),
            "final_answer": row.get("expected_final_answer") or strip_thinking(str(row.get("expected_answer") or "")),
            "label": row.get("expected_final_answer") or strip_thinking(str(row.get("expected_answer") or "")),
            "task": row.get("task"),
            "domain": row.get("domain"),
            "capability_domain": row.get("capability_domain"),
            "sft_type": row.get("sft_type"),
            "evaluation_type": row.get("evaluation_type") or "open_text",
            "source": "dataset_row.conversations",
        }
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
            expected["kind"] = "worker_safety"
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


def estimate_plan(meta: Dict[str, Any], fps: float, max_pixels: int, max_frames: int, model: str = "", backend: str = "") -> Dict[str, Any]:
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    duration_s = float(meta.get("duration_s") or 0)
    source_frames = int(meta.get("total_frames") or 0)
    if model_uses_native_video(model, backend):
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
    visual_tokens = int(frames_passed * effective_pixels / CONTEXT_PATCH_PIXELS)
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
        os.getenv("INFERENCE_BACKEND", ""),
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

    requested_name = repo_id.replace("/", "_")
    update_load_progress("fiftyone", f"Checking local FiftyOne datasets for {repo_id}")
    existing_names = set(fo.list_datasets())
    if requested_name in existing_names:
        dataset = fo.load_dataset(requested_name)
    elif repo_id in existing_names:
        dataset = fo.load_dataset(repo_id)
    else:
        update_load_progress("fiftyone", f"Loading {repo_id} from the FiftyOne Hugging Face integration")
        try:
            if max_videos > 0:
                dataset = fouh.load_from_hub(repo_id, dataset_name=requested_name, max_samples=max_videos, persistent=True)
            else:
                dataset = fouh.load_from_hub(repo_id, dataset_name=requested_name, persistent=True)
        except TypeError:
            try:
                dataset = fouh.load_from_hub(repo_id, dataset_name=requested_name, persistent=True)
            except TypeError:
                dataset = fouh.load_from_hub(repo_id, persistent=True)

    videos: List[Dict[str, Any]] = []
    expected_total = max_videos if max_videos > 0 else 0
    update_load_progress("fiftyone", f"Reading video samples from FiftyOne dataset {dataset.name}", done=0, total=expected_total)
    for sample in dataset:
        path = str(sample.filepath)
        if not is_video_path(path):
            continue
        row = sample_to_row(sample)
        label = row.get("label") or guess_label(sample)
        video = attach_meta({
            "id": video_id(path),
            "name": Path(path).name,
            "filepath": path,
            "sample_id": str(sample.id),
            "label": label,
            "dataset_row": row,
            "source": "fiftyone",
        })
        videos.append(video)
        update_load_progress(
            "fiftyone",
            f"Prepared {len(videos)} video samples from FiftyOne",
            done=len(videos),
            total=expected_total,
            current_file=Path(path).name,
            videos=list(videos),
        )
        if max_videos > 0 and len(videos) >= max_videos:
            break
    if not videos:
        raise RuntimeError(f"FiftyOne loaded {repo_id}, but no video samples were found")
    update_state(dataset_source="fiftyone", fo_dataset_name=dataset.name)
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

    update_load_progress("hf_listing", f"Listing files in Hugging Face dataset {repo_id}")
    repo_files = list_repo_files(repo_id, repo_type="dataset")
    update_load_progress("hf_metadata", f"Found {len(repo_files)} files; checking sidecar metadata")
    sidecar_rows = load_hf_sidecar_rows(repo_id, repo_files)
    annotation_bundle = load_hf_annotation_samples(repo_id, repo_files)
    annotation_samples = annotation_bundle.get("samples") or []
    files = [f for f in repo_files if is_video_path(f)]
    if not files:
        raise RuntimeError(f"No video files found in Hugging Face dataset {repo_id}")
    local_root = Path("/tmp/byo_video_datasets") / repo_id.replace("/", "_")
    local_root.mkdir(parents=True, exist_ok=True)
    videos: List[Dict[str, Any]] = []
    if annotation_samples:
        selected_items = annotation_samples if max_videos <= 0 else annotation_samples[:max_videos]
        update_load_progress(
            "hf_download",
            f"Selected {len(selected_items)} annotated PRISM samples from {len(annotation_samples):,} available samples",
            done=0,
            total=len(selected_items),
        )
    else:
        selected_items = [{"hf_path": file_name} for file_name in (files if max_videos <= 0 else files[:max_videos])]
        update_load_progress(
            "hf_download",
            f"Selected {len(selected_items)} videos from {len(files)} available video files",
            done=0,
            total=len(selected_items),
        )
    total = len(selected_items)
    for index, item in enumerate(selected_items, start=1):
        file_name = str(item.get("hf_path") or item.get("video") or "")
        if not file_name:
            continue
        log(f"Downloading {index}/{total} from Hugging Face: {file_name}")
        update_load_progress(
            "hf_download",
            f"Downloading {index}/{total}: {file_name}",
            done=index - 1,
            total=total,
            current_file=file_name,
        )
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file_name, local_dir=str(local_root))
        if annotation_samples:
            row = dict(item)
        else:
            row = dict(sidecar_rows.get(file_name) or sidecar_rows.get(Path(file_name).name) or {})
        row.setdefault("hf_repo", repo_id)
        row.setdefault("hf_path", file_name)
        label = row.get("label")
        update_load_progress(
            "metadata",
            f"Extracting metadata and thumbnail for {index}/{total}: {file_name}",
            done=index - 1,
            total=total,
            current_file=file_name,
        )
        sample_key = str(row.get("annotation_id") or file_name)
        video = attach_meta({
            "id": video_id(f"{local}::{sample_key}"),
            "name": file_name,
            "filepath": local,
            "sample_id": None,
            "label": label,
            "dataset_row": row,
            "source": "hf_hub",
        })
        videos.append(video)
        update_load_progress(
            "metadata",
            f"Prepared {index}/{total}: {file_name}",
            done=index,
            total=total,
            current_file=file_name,
            videos=list(videos),
        )
    try:
        import fiftyone as fo

        name = repo_id.replace("/", "_") + "_runtime"
        update_load_progress("fiftyone_wrap", f"Wrapping {len(videos)} downloaded videos in local FiftyOne dataset {name}", done=0, total=len(videos))
        if name in fo.list_datasets():
            dataset = fo.load_dataset(name)
            try:
                dataset.delete_samples(dataset.values("id"))
            except Exception:
                pass
        else:
            dataset = fo.Dataset(name, persistent=True)

        for index, video in enumerate(videos, start=1):
            sample = fo.Sample(filepath=video["filepath"])
            sample["hf_repo"] = repo_id
            sample["hf_path"] = video["name"]
            row = video.get("dataset_row") or {}
            for field in ("annotation_id", "task", "capability_domain", "domain", "sft_type", "evaluation_type"):
                if row.get(field) is not None:
                    sample[f"hf_{field}"] = str(row.get(field))
            if row.get("user_prompt"):
                sample["hf_user_prompt"] = str(row.get("user_prompt"))
            if row.get("expected_final_answer"):
                sample["hf_expected_answer"] = str(row.get("expected_final_answer"))
            if video.get("label"):
                sample["hf_label"] = str(video["label"])
            expected = video.get("expected") or {}
            if expected:
                sample["expected_class_id"] = expected.get("class_id")
                sample["expected_label"] = expected.get("label")
                sample["expected_is_hazardous"] = expected.get("hazardous")
                if expected.get("final_answer"):
                    sample["expected_answer"] = expected.get("final_answer")
            dataset.add_sample(sample)
            video["sample_id"] = str(sample.id)
            video["source"] = "fiftyone"
            update_load_progress(
                "fiftyone_wrap",
                f"Added {index}/{len(videos)} videos to FiftyOne",
                done=index,
                total=len(videos),
                current_file=video["name"],
                videos=list(videos),
            )
        update_state(dataset_source="fiftyone", fo_dataset_name=name)
        log(f"Wrapped HF files in local FiftyOne dataset {name}")
    except Exception as exc:
        log(f"Could not wrap HF files in FiftyOne: {exc}")
        update_state(dataset_source="hf_hub", fo_dataset_name=None)
    return videos


def load_dataset(repo_id: str, max_videos: int) -> List[Dict[str, Any]]:
    limit = "all" if max_videos <= 0 else str(max_videos)
    started_epoch = time.time()
    update_state(
        loading_dataset=True,
        dataset_repo=repo_id,
        dataset_source=None,
        fo_dataset_name=None,
        videos=[],
        results=[],
        active_requests={},
        progress={
            "mode": "dataset_load",
            "phase": "starting",
            "done": 0,
            "total": 0,
            "errors": 0,
            "started_epoch": started_epoch,
            "updated_epoch": started_epoch,
            "last_result_epoch": None,
            "current_file": None,
            "last_event": f"Loading dataset {repo_id} (up to {limit} videos)",
            "last_error": None,
        },
        batch_metrics=None,
    )
    log(f"Loading dataset {repo_id} (up to {limit} videos)")
    try:
        try:
            videos = load_with_fiftyone(repo_id, max_videos)
            log(f"Loaded {len(videos)} videos with FiftyOne")
        except Exception as exc:
            log(f"FiftyOne load failed; falling back to huggingface_hub: {exc}")
            update_load_progress(
                "hf_fallback",
                f"FiftyOne load failed; falling back to Hugging Face Hub: {exc}",
                done=0,
                total=0,
                current_file=None,
            )
            videos = load_with_hf_hub(repo_id, max_videos)
            log(f"Loaded {len(videos)} videos with huggingface_hub")
    except Exception as exc:
        now = time.time()
        update_state(
            loading_dataset=False,
            progress={
                "mode": "dataset_load",
                "phase": "failed",
                "done": 0,
                "total": 0,
                "errors": 1,
                "started_epoch": started_epoch,
                "updated_epoch": now,
                "finished_epoch": now,
                "last_event": "Dataset load failed",
                "last_error": str(exc),
            },
        )
        raise
    now = time.time()
    update_state(
        dataset_repo=repo_id,
        videos=videos,
        results=[],
        loading_dataset=False,
        progress={
            "mode": "dataset_load",
            "phase": "complete",
            "done": len(videos),
            "total": len(videos),
            "errors": 0,
            "started_epoch": started_epoch,
            "updated_epoch": now,
            "finished_epoch": now,
            "last_event": f"Loaded {len(videos)} videos",
            "last_error": None,
        },
        batch_metrics=None,
    )
    return videos


def load_dataset_worker(repo_id: str, max_videos: int) -> None:
    try:
        load_dataset(repo_id, max_videos)
    except Exception as exc:
        log(f"Dataset load failed: {exc}")
    finally:
        try:
            RESULTS_FILE.write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")
        except Exception:
            pass


def detect_server() -> Dict[str, Any]:
    info = {
        "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        "model": os.getenv("MODEL_NAME", ""),
        "backend": os.getenv("INFERENCE_BACKEND", "vllm"),
        "model_max_len": DEFAULT_MODEL_MAX_LEN,
        "model_max_len_source": "default",
        "context_safety_reserve": CONTEXT_SAFETY_RESERVE,
        "context_patch_pixels": CONTEXT_PATCH_PIXELS,
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
        if models:
            configured_model = info["model"]
            chosen_model = None
            if configured_model:
                chosen_model = next((m for m in models if configured_model in {m.get("id"), m.get("root")}), None)
            chosen_model = chosen_model or models[0]
            if not info["model"]:
                info["model"] = chosen_model.get("id") or chosen_model.get("root") or ""
            info["models"] = [m.get("id") or m.get("root") for m in models]
            for key in ("max_model_len", "max_context_len", "context_length", "max_sequence_length"):
                raw_value = chosen_model.get(key)
                if raw_value is None:
                    continue
                try:
                    info["model_max_len"] = int(float(raw_value))
                    info["model_max_len_source"] = key
                    break
                except Exception:
                    continue
            info["model_info"] = json_safe(chosen_model, depth=2, max_text=300)
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


def model_prefers_video_data(model: str, backend: str = "") -> bool:
    lower = model.lower()
    backend_lower = (backend or os.getenv("INFERENCE_BACKEND", "")).lower()
    return "nim" in backend_lower or ("cosmos-reason" in lower and "nim" in lower)


def model_uses_native_video(model: str, backend: str = "") -> bool:
    backend_lower = (backend or os.getenv("INFERENCE_BACKEND", "")).lower()
    return "nim" in backend_lower or model_prefers_file_url(model) or model_prefers_video_data(model, backend)


def reasoning_profile_for_model(model: str, backend: str = "", requested: str = "auto") -> Dict[str, str]:
    requested = str(requested or "auto")
    if requested != "auto" and requested in REASONING_PROFILES:
        return dict(REASONING_PROFILES[requested])
    lower = f"{model or ''} {backend or ''}".lower()
    if "cosmos" in lower or re.search(r"\bcr[123]|\bc3", lower):
        return dict(REASONING_PROFILES["cosmos_think_answer"])
    if "qwen" in lower or "qw3" in lower:
        return dict(REASONING_PROFILES["qwen_think_answer"])
    if "nemotron" in lower or "omni" in lower:
        return dict(REASONING_PROFILES["nemotron_think_answer"])
    if "gemma" in lower:
        return dict(REASONING_PROFILES["gemma_think_answer"])
    return dict(REASONING_PROFILES["generic_think_answer"])


def strip_reasoning_wrapper(prompt: str) -> str:
    text = str(prompt or "").strip()
    pattern = re.compile(
        r"^\s*(?:/think\s*)?<think>\s*your reasoning\.?\s*</think>\s*<answer>\s*(.*?)\s*</answer>\s*$",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.match(text)
    if match:
        return match.group(1).strip()
    return text


def apply_reasoning_wrapper(prompt: str, profile: Dict[str, str]) -> str:
    base = strip_reasoning_wrapper(prompt)
    return f"{profile.get('prefix', '')}{base}{profile.get('suffix', '')}".strip()


def bool_param(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def content_for_video(video_path: str, prompt: str, model: str, fps: float, max_pixels: int, max_frames: int, backend: str = "", force_frames: bool = False) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = get_video_meta(video_path)
    plan = estimate_plan(meta, fps, max_pixels, max_frames, model if not force_frames else "", backend if not force_frames else "")
    if not force_frames and model_uses_native_video(model, backend):
        mime = mimetypes.guess_type(video_path)[0] or "video/mp4"
        data = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")
        return [
            {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}},
            {"type": "text", "text": prompt},
        ], plan
    frames = extract_frames_b64(video_path, fps=fps, max_frames=max_frames, max_pixels=max_pixels)
    if not frames:
        raise RuntimeError("No frames could be extracted from the video")
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    plan["frames_passed"] = len(frames)
    return content, plan


def prompts_for_video(video: Dict[str, Any], system_prompt: str, user_prompt: str, params: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str]:
    params = params or {}
    prompt_mode = str(params.get("prompt_mode") or "auto")
    row = video.get("dataset_row") or {}
    row_system = str(row.get("system_prompt") or "").strip()
    row_user = str(row.get("user_prompt") or "").strip()
    if prompt_mode in {"auto", "dataset_row"} and row_system and row_user:
        prompt_source = "dataset_row"
        effective_system = row_system
        effective_user = row_user
    else:
        prompt_source = "runtime_form"
        effective_system = system_prompt
        effective_user = user_prompt
    if bool_param(params.get("reasoning_enabled")):
        profile = reasoning_profile_for_model(
            str(params.get("reasoning_model") or ""),
            str(params.get("reasoning_backend") or ""),
            str(params.get("reasoning_format") or "auto"),
        )
        if not params.get("reasoning_model"):
            server = detect_server()
            profile = reasoning_profile_for_model(
                str(server.get("model") or ""),
                str(server.get("backend") or ""),
                str(params.get("reasoning_format") or "auto"),
            )
        effective_user = apply_reasoning_wrapper(effective_user, profile)
        prompt_source = f"{prompt_source}+reasoning:{profile['id']}"
    return effective_system, effective_user, prompt_source


def selected_videos_for_ids(ids: Iterable[str]) -> List[Dict[str, Any]]:
    snap = snapshot()
    videos_by_id = {v["id"]: v for v in snap["videos"]}
    selected = [videos_by_id[i] for i in ids if i in videos_by_id]
    return selected or list(videos_by_id.values())


def context_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fps": float(params.get("fps") if params.get("fps") is not None else STATE["defaults"]["fps"]),
        "max_pixels": int(params.get("max_pixels") if params.get("max_pixels") is not None else STATE["defaults"]["max_pixels"]),
        "max_tokens": int(params.get("max_tokens") if params.get("max_tokens") is not None else STATE["defaults"]["max_tokens"]),
        "max_frames": int(params.get("max_frames") if params.get("max_frames") is not None else STATE["defaults"]["max_frames"]),
        "prompt_mode": str(params.get("prompt_mode") or "auto"),
    }


def context_budget_report(
    videos: List[Dict[str, Any]],
    params: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    server = detect_server()
    model = str(server.get("model") or os.getenv("MODEL_NAME") or "")
    backend = str(server.get("backend") or INFERENCE_BACKEND)
    report: Dict[str, Any] = {
        "enabled": False,
        "reason": "native_video",
        "model": model,
        "backend": backend,
        "model_max_len": int(server.get("model_max_len") or DEFAULT_MODEL_MAX_LEN),
        "reserve_tokens": CONTEXT_SAFETY_RESERVE,
        "patch_pixels": CONTEXT_PATCH_PIXELS,
        "videos": [],
        "violations": [],
        "warnings": [],
        "worst_video": None,
    }
    if model_uses_native_video(model, backend):
        report["message"] = "Native video/NIM mode: context enforcement is delegated to the model service."
        return report

    p = context_params(params)
    output_tokens = p["max_tokens"]
    allowed_input_tokens = max(1, report["model_max_len"] - output_tokens - CONTEXT_SAFETY_RESERVE)
    report.update({
        "enabled": True,
        "reason": "oss_image_frames",
        "prompt_tokens_est": estimate_prompt_tokens(system_prompt, user_prompt),
        "max_output_tokens": output_tokens,
        "allowed_input_tokens": allowed_input_tokens,
        "budget_warning_ratio": 0.85,
    })

    for video in videos:
        meta = video.get("meta") or get_video_meta(video["filepath"])
        plan = estimate_plan(meta, p["fps"], p["max_pixels"], p["max_frames"], model)
        context_prompt_params = dict(params)
        context_prompt_params.setdefault("reasoning_model", model)
        context_prompt_params.setdefault("reasoning_backend", backend)
        effective_system, effective_user, prompt_source = prompts_for_video(video, system_prompt, user_prompt, context_prompt_params)
        prompt_tokens = estimate_prompt_tokens(effective_system, effective_user)
        visual_tokens = int(plan.get("visual_tokens_est") or 0)
        input_tokens = visual_tokens + prompt_tokens
        ratio = input_tokens / allowed_input_tokens if allowed_input_tokens else 999.0
        item = {
            "id": video.get("id"),
            "name": video.get("name"),
            "frames_passed": plan.get("frames_passed"),
            "effective_pixels": plan.get("effective_pixels"),
            "visual_tokens_est": visual_tokens,
            "prompt_tokens_est": prompt_tokens,
            "prompt_source": prompt_source,
            "estimated_input_tokens": input_tokens,
            "allowed_input_tokens": allowed_input_tokens,
            "ratio": ratio,
            "over": input_tokens > allowed_input_tokens,
        }
        report["videos"].append(item)

    if report["videos"]:
        report["worst_video"] = max(report["videos"], key=lambda item: item["estimated_input_tokens"])
        report["violations"] = [item for item in report["videos"] if item["over"]]
        report["warnings"] = [item for item in report["videos"] if not item["over"] and item["ratio"] >= report["budget_warning_ratio"]]
    if report["violations"]:
        worst = report["worst_video"] or report["violations"][0]
        report["message"] = (
            f"Estimated input context is over budget for {len(report['violations'])}/{len(report['videos'])} videos. "
            f"Worst: {worst['name']} uses about {worst['estimated_input_tokens']:,} input tokens; "
            f"budget is {allowed_input_tokens:,} after reserving output and safety tokens."
        )
    elif report["warnings"]:
        report["message"] = "Settings are close to the OSS context limit; expect slower prefill and consider lowering frames or pixels."
    else:
        report["message"] = "Settings are inside the estimated OSS context budget."
    return report


def validate_context_budget(
    ids: Iterable[str],
    params: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    allow_over_context: bool = False,
) -> Dict[str, Any]:
    report = context_budget_report(selected_videos_for_ids(ids), params, system_prompt, user_prompt)
    if allow_over_context:
        if report.get("violations"):
            log("Context guard override enabled; running over-budget OSS settings")
        return report
    if report.get("enabled") and report.get("violations"):
        raise ClientInputError(
            f"{report['message']} Use Fit to model, reduce max input frames/fps/max pixels, "
            "or explicitly enable Allow over-budget OSS run for stress testing."
        )
    return report


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


def normalize_eval_text(text: Any) -> str:
    value = strip_thinking(str(text or "")).lower()
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def token_f1(expected: str, predicted: str) -> float:
    expected_tokens = normalize_eval_text(expected).split()
    predicted_tokens = normalize_eval_text(predicted).split()
    if not expected_tokens or not predicted_tokens:
        return 0.0
    expected_counts = Counter(expected_tokens)
    predicted_counts = Counter(predicted_tokens)
    overlap = sum((expected_counts & predicted_counts).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return (2 * precision * recall) / (precision + recall) if precision + recall else 0.0


def extract_mcq_choice(text: str) -> Optional[str]:
    value = strip_thinking(text).strip()
    match = re.search(r"\b([ABCD])\b", value, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def evaluate_text_result(expected: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    expected_answer = str(expected.get("final_answer") or expected.get("answer") or "")
    predicted_answer = strip_thinking(result.get("response") or "")
    eval_type = str(expected.get("evaluation_type") or "open_text")
    expected_norm = normalize_eval_text(expected_answer)
    predicted_norm = normalize_eval_text(predicted_answer)
    exact_match = bool(expected_norm and predicted_norm and expected_norm == predicted_norm)
    contains_match = bool(expected_norm and predicted_norm and (expected_norm in predicted_norm or predicted_norm in expected_norm))
    if eval_type == "mcq":
        expected_choice = extract_mcq_choice(expected_answer)
        predicted_choice = extract_mcq_choice(predicted_answer)
        is_correct = bool(expected_choice and predicted_choice and expected_choice == predicted_choice)
        score = 1.0 if is_correct else 0.0
        metric = "mcq_accuracy"
    else:
        score = token_f1(expected_answer, predicted_answer)
        is_correct = bool(exact_match or contains_match or score >= TEXT_SCORE_THRESHOLD)
        metric = "answer_token_f1"
    return {
        "has_expected": True,
        "kind": "dataset_answer",
        "metric": metric,
        "evaluation_type": eval_type,
        "task": expected.get("task"),
        "domain": expected.get("domain"),
        "capability_domain": expected.get("capability_domain"),
        "sft_type": expected.get("sft_type"),
        "expected_answer": expected_answer,
        "predicted_answer": predicted_answer,
        "expected_label": expected_answer,
        "predicted_label": predicted_answer,
        "answer_score": score,
        "score_threshold": TEXT_SCORE_THRESHOLD if metric == "answer_token_f1" else 1.0,
        "exact_match": exact_match,
        "contains_match": contains_match,
        "expected_choice": extract_mcq_choice(expected_answer),
        "predicted_choice": extract_mcq_choice(predicted_answer),
        "is_correct": is_correct,
    }


def evaluate_result(video: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    expected = video.get("expected") or {}
    if not expected:
        return {"has_expected": False}
    if expected.get("kind") == "dataset_answer":
        return evaluate_text_result(expected, result)
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
        "kind": "worker_safety",
        "metric": "class_label_accuracy",
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
    backend = str(server.get("backend") or os.getenv("INFERENCE_BACKEND", "vllm")).lower()
    headers = {"Content-Type": "application/json"}
    if os.getenv("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('VLLM_API_KEY')}"
    fps = float(params.get("fps") or STATE["defaults"]["fps"])
    max_pixels = int(params.get("max_pixels") or STATE["defaults"]["max_pixels"])
    max_frames = int(params.get("max_frames") if params.get("max_frames") is not None else STATE["defaults"]["max_frames"])
    prompt_params = dict(params)
    prompt_params.setdefault("reasoning_model", model)
    prompt_params.setdefault("reasoning_backend", backend)
    effective_system_prompt, effective_user_prompt, prompt_source = prompts_for_video(
        video,
        system_prompt,
        user_prompt,
        prompt_params,
    )
    update_active_request(
        video,
        "preprocessing",
        f"Preparing {video['name']} for the model server",
        model=model,
        backend=backend,
        prompt_source=prompt_source,
    )
    content, plan = content_for_video(
        video["filepath"],
        effective_user_prompt,
        model,
        fps,
        max_pixels,
        max_frames,
        backend=backend,
    )
    preprocessing_seconds = time.monotonic() - overall_started
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": float(params.get("temperature") if params.get("temperature") is not None else STATE["defaults"]["temperature"]),
        "top_p": float(params.get("top_p") if params.get("top_p") is not None else STATE["defaults"]["top_p"]),
        "stream": False,
    }
    rep_penalty = float(params.get("repetition_penalty") if params.get("repetition_penalty") is not None else STATE["defaults"]["repetition_penalty"])
    if "nim" in backend:
        payload["nvext"] = {"repetition_penalty": rep_penalty}
    else:
        payload["max_tokens"] = int(params.get("max_tokens") or STATE["defaults"]["max_tokens"])
        payload["repetition_penalty"] = rep_penalty
    try:
        update_active_request(
            video,
            "model_wait",
            f"Waiting for model server response: {video['name']}",
            model=model,
            backend=backend,
            prompt_source=prompt_source,
            frames=plan.get("frames_passed"),
            visual_tokens=plan.get("visual_tokens_est"),
        )
        completion = post_chat_completion(base_url, headers, payload)
    except requests.HTTPError as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if "nim" in backend and status_code in (400, 422) and plan.get("mode") == "native_video_url":
            update_active_request(
                video,
                "model_retry",
                f"Retrying with frame fallback after HTTP {status_code}: {video['name']}",
                model=model,
                backend=backend,
                prompt_source=prompt_source,
            )
            fallback_content, fallback_plan = content_for_video(
                video["filepath"],
                effective_user_prompt,
                model,
                fps,
                max_pixels,
                0,
                backend=backend,
                force_frames=True,
            )
            fallback_plan["fallback_from_video_url"] = status_code
            payload["messages"][1]["content"] = fallback_content
            update_active_request(
                video,
                "model_wait",
                f"Waiting for fallback model response: {video['name']}",
                model=model,
                backend=backend,
                prompt_source=prompt_source,
                frames=fallback_plan.get("frames_passed"),
                visual_tokens=fallback_plan.get("visual_tokens_est"),
            )
            completion = post_chat_completion(base_url, headers, payload)
            plan = fallback_plan
        else:
            raise
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
        "prompt_source": prompt_source,
        "system_prompt_used": effective_system_prompt,
        "user_prompt_used": effective_user_prompt,
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
        schema = dataset.get_field_schema()
        if "runtime_agent_correct" not in schema:
            dataset.add_sample_field("runtime_agent_correct", fo.BooleanField)
        if "runtime_agent_error" not in schema:
            dataset.add_sample_field("runtime_agent_error", fo.StringField)
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
    answer_scores = [float(e.get("answer_score")) for e in evaluations if e.get("answer_score") is not None]
    by_class: Dict[str, Dict[str, Any]] = {}
    by_task: Dict[str, Dict[str, Any]] = {}
    by_domain: Dict[str, Dict[str, Any]] = {}
    by_capability_domain: Dict[str, Dict[str, Any]] = {}

    def add_group(grouped: Dict[str, Dict[str, Any]], key: str, evaluation: Dict[str, Any]) -> None:
        if not key:
            return
        item = grouped.setdefault(key, {"total": 0, "correct": 0, "score_sum": 0.0, "score_count": 0})
        item["total"] += 1
        if evaluation.get("is_correct"):
            item["correct"] += 1
        if evaluation.get("answer_score") is not None:
            item["score_sum"] += float(evaluation.get("answer_score") or 0.0)
            item["score_count"] += 1

    for evaluation in evaluations:
        if evaluation.get("kind") == "worker_safety":
            class_id = str(evaluation.get("expected_class_id"))
            item = by_class.setdefault(class_id, {"total": 0, "correct": 0, "label": evaluation.get("expected_label")})
            item["total"] += 1
            if evaluation.get("is_correct"):
                item["correct"] += 1
        add_group(by_task, str(evaluation.get("task") or ""), evaluation)
        add_group(by_domain, str(evaluation.get("domain") or ""), evaluation)
        add_group(by_capability_domain, str(evaluation.get("capability_domain") or ""), evaluation)
    for item in by_class.values():
        item["accuracy"] = item["correct"] / item["total"] if item["total"] else None
    for grouped in (by_task, by_domain, by_capability_domain):
        for item in grouped.values():
            item["accuracy"] = item["correct"] / item["total"] if item["total"] else None
            item["average_score"] = item["score_sum"] / item["score_count"] if item["score_count"] else None
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
            "answer_score_average": sum(answer_scores) / len(answer_scores) if answer_scores else None,
            "hazard_correct": hazard_correct,
            "hazard_accuracy": hazard_correct / len(evaluations) if evaluations else None,
            "by_expected_class": by_class,
            "by_task": by_task,
            "by_domain": by_domain,
            "by_capability_domain": by_capability_domain,
        },
    }


def export_section_ids(sections: Optional[Iterable[str]]) -> List[str]:
    allowed = {section["id"] for section in EXPORT_SECTIONS}
    selected = [str(section) for section in (sections or []) if str(section) in allowed]
    if selected:
        return selected
    return [section["id"] for section in EXPORT_SECTIONS if section.get("default")]


def export_stem(fmt: str) -> str:
    return f"byo-video-runtime-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.{fmt}"


def safe_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)


def percent_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def seconds_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}s"


def result_rows(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    videos_by_id = {v.get("id"): v for v in snap.get("videos") or []}
    rows: List[Dict[str, Any]] = []
    for result in snap.get("results") or []:
        video = videos_by_id.get(result.get("id")) or {}
        meta = video.get("meta") or {}
        parsed = result.get("json") or {}
        evaluation = result.get("evaluation") or {}
        metrics = result.get("metrics") or {}
        hazard = parsed.get("hazard_detection") or {}
        plan = result.get("plan") or video.get("plan") or {}
        row = video.get("dataset_row") or {}
        expected_text = evaluation.get("expected_answer") or evaluation.get("expected_label")
        prediction_text = evaluation.get("predicted_answer") or evaluation.get("predicted_label") or parsed.get("prediction_label")
        rows.append({
            "video": result.get("name") or video.get("name") or "",
            "task": evaluation.get("task") or row.get("task") or "",
            "domain": evaluation.get("domain") or row.get("domain") or "",
            "capability_domain": evaluation.get("capability_domain") or row.get("capability_domain") or "",
            "sft_type": evaluation.get("sft_type") or row.get("sft_type") or "",
            "evaluation_type": evaluation.get("evaluation_type") or row.get("evaluation_type") or "",
            "metric": evaluation.get("metric") or "",
            "answer_score": evaluation.get("answer_score"),
            "expected_answer": evaluation.get("expected_answer") or "",
            "predicted_answer": evaluation.get("predicted_answer") or result.get("response") or "",
            "prompt_source": result.get("prompt_source") or "",
            "user_prompt": result.get("user_prompt_used") or row.get("user_prompt") or "",
            "expected_class_id": evaluation.get("expected_class_id"),
            "expected_label": expected_text,
            "expected_hazard": evaluation.get("expected_is_hazardous"),
            "prediction_class_id": parsed.get("prediction_class_id"),
            "prediction_label": prediction_text,
            "prediction_hazard": hazard.get("is_hazardous"),
            "match": "correct" if evaluation.get("is_correct") else ("miss" if evaluation.get("has_expected") else ""),
            "hazard_match": evaluation.get("hazard_match"),
            "description": parsed.get("video_description") or result.get("response") or "",
            "error": result.get("error") or "",
            "ttft_seconds": metrics.get("ttft_seconds"),
            "output_tokens_per_second": metrics.get("output_tokens_per_second"),
            "e2e_seconds": metrics.get("e2e_seconds"),
            "frames_to_vlm": plan.get("frames_passed"),
            "visual_tokens_est": plan.get("visual_tokens_est"),
            "resolution": f"{meta.get('width') or ''}x{meta.get('height') or ''}",
            "duration_s": meta.get("duration_s"),
            "source_frames": meta.get("total_frames"),
            "filepath": video.get("filepath") or result.get("filepath") or "",
            "hf_label": video.get("label") or result.get("source_label") or "",
        })
    if rows:
        return rows
    for video in snap.get("videos") or []:
        meta = video.get("meta") or {}
        expected = video.get("expected") or {}
        plan = video.get("plan") or {}
        row = video.get("dataset_row") or {}
        rows.append({
            "video": video.get("name") or "",
            "task": expected.get("task") or row.get("task") or "",
            "domain": expected.get("domain") or row.get("domain") or "",
            "capability_domain": expected.get("capability_domain") or row.get("capability_domain") or "",
            "sft_type": expected.get("sft_type") or row.get("sft_type") or "",
            "evaluation_type": expected.get("evaluation_type") or row.get("evaluation_type") or "",
            "metric": "",
            "answer_score": None,
            "expected_answer": expected.get("final_answer") or expected.get("label") or "",
            "predicted_answer": "",
            "prompt_source": "dataset_row" if row.get("user_prompt") else "",
            "user_prompt": row.get("user_prompt") or "",
            "expected_class_id": expected.get("class_id"),
            "expected_label": expected.get("final_answer") or expected.get("label"),
            "expected_hazard": expected.get("hazardous"),
            "prediction_class_id": "",
            "prediction_label": "",
            "prediction_hazard": "",
            "match": "",
            "hazard_match": "",
            "description": "",
            "error": "",
            "ttft_seconds": None,
            "output_tokens_per_second": None,
            "e2e_seconds": None,
            "frames_to_vlm": plan.get("frames_passed"),
            "visual_tokens_est": plan.get("visual_tokens_est"),
            "resolution": f"{meta.get('width') or ''}x{meta.get('height') or ''}",
            "duration_s": meta.get("duration_s"),
            "source_frames": meta.get("total_frames"),
            "filepath": video.get("filepath") or "",
            "hf_label": video.get("label") or "",
        })
    return rows


def class_breakdown(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        label = row.get("task") or row.get("expected_label") or row.get("hf_label") or "Unknown"
        item = grouped.setdefault(label, {"label": label, "total": 0, "correct": 0, "errors": 0, "predictions": Counter()})
        item["total"] += 1
        if row.get("match") == "correct":
            item["correct"] += 1
        if row.get("error"):
            item["errors"] += 1
        pred = row.get("prediction_label") or ("ERROR" if row.get("error") else "unrun")
        item["predictions"][str(pred)] += 1
    out = []
    for item in grouped.values():
        total = item["total"] or 1
        most_common = item["predictions"].most_common(1)
        out.append({
            "label": item["label"],
            "total": item["total"],
            "correct": item["correct"],
            "errors": item["errors"],
            "accuracy": item["correct"] / total,
            "most_predicted": f"{most_common[0][0]} ({most_common[0][1]}/{item['total']})" if most_common else "",
        })
    return sorted(out, key=lambda item: (item["accuracy"], item["label"]))


def export_summary(snap: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    bm = snap.get("batch_metrics") or {}
    evaluation = bm.get("evaluation") or {}
    evaluated_rows = [row for row in rows if row.get("match")]
    correct = sum(1 for row in evaluated_rows if row.get("match") == "correct")
    errors = sum(1 for row in rows if row.get("error"))
    total = int(bm.get("total") or len(rows))
    completed = int(bm.get("completed") or len(snap.get("results") or []))
    accuracy = evaluation.get("accuracy")
    if accuracy is None and evaluated_rows:
        accuracy = correct / len(evaluated_rows)
    return {
        "dataset": snap.get("dataset_repo") or DEFAULT_DATASET,
        "model": (snap.get("server") or {}).get("model") or "",
        "backend": (snap.get("server") or {}).get("backend") or "",
        "instance": (snap.get("server") or {}).get("instance") or "",
        "total": total,
        "completed": completed,
        "correct": correct,
        "errors": int(bm.get("errors") if bm.get("errors") is not None else errors),
        "accuracy": accuracy,
        "answer_score_average": evaluation.get("answer_score_average"),
        "hazard_accuracy": evaluation.get("hazard_accuracy"),
        "batch_e2e_seconds": bm.get("batch_wall_seconds"),
        "video_requests_per_second": bm.get("video_requests_per_second"),
        "median_video_e2e": (bm.get("e2e_seconds") or {}).get("median"),
        "avg_video_e2e": (bm.get("e2e_seconds") or {}).get("average"),
        "run_label": bm.get("run_label") or "",
        "prompt_hash": bm.get("prompt_hash") or "",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def export_recommendations(summary: Dict[str, Any], rows: List[Dict[str, Any]], classes: List[Dict[str, Any]]) -> List[str]:
    recs: List[str] = []
    if not rows or not any(row.get("prediction_label") or row.get("error") for row in rows):
        recs.append("Run a batch before using the report for model quality decisions.")
    if summary.get("errors"):
        recs.append("Resolve runtime errors first; error rows can dominate accuracy and throughput interpretation.")
    if summary.get("accuracy") is not None and summary["accuracy"] < 0.7:
        recs.append("Use the per-class breakdown to target prompt or sampling changes before treating the run as production-ready.")
    weak = [item["label"] for item in classes if item.get("total") and item.get("accuracy", 0) < 0.5][:3]
    if weak:
        recs.append("Prioritize review of weak classes: " + ", ".join(weak) + ".")
    if not recs:
        recs.append("Promote this configuration to a larger regression run and compare against alternate prompts or reasoning modes.")
    return recs


def export_payload(sections: List[str]) -> Dict[str, Any]:
    snap = snapshot()
    rows = result_rows(snap)
    classes = class_breakdown(rows)
    summary = export_summary(snap, rows)
    recommendations = export_recommendations(summary, rows, classes)
    return {
        "snapshot": snap,
        "sections": sections,
        "summary": summary,
        "rows": rows,
        "class_breakdown": classes,
        "recommendations": recommendations,
    }


def write_json_export(payload: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def csv_headers() -> List[str]:
    return [
        "video",
        "task",
        "capability_domain",
        "domain",
        "sft_type",
        "evaluation_type",
        "metric",
        "answer_score",
        "prompt_source",
        "user_prompt",
        "expected_answer",
        "predicted_answer",
        "expected_class_id",
        "expected_label",
        "prediction_class_id",
        "prediction_label",
        "match",
        "expected_hazard",
        "prediction_hazard",
        "hazard_match",
        "ttft_seconds",
        "output_tokens_per_second",
        "e2e_seconds",
        "frames_to_vlm",
        "visual_tokens_est",
        "resolution",
        "duration_s",
        "source_frames",
        "error",
        "description",
        "filepath",
    ]


def write_csv_export(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_headers(), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def xlsx_cell(value: Any, row: int, col: int) -> str:
    ref = f"{col_name(col)}{row}"
    if value is None or value == "":
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{html.escape(str(value), quote=True)}</t></is></c>'


def xlsx_sheet(rows: List[List[Any]]) -> str:
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>']
    out.append('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>')
    for ri, row in enumerate(rows, 1):
        out.append(f'<row r="{ri}">')
        for ci, value in enumerate(row, 1):
            out.append(xlsx_cell(value, ri, ci))
        out.append("</row>")
    out.append("</sheetData></worksheet>")
    return "".join(out)


def write_xlsx_export(payload: Dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    classes = payload["class_breakdown"]
    errors = [row for row in rows if row.get("error") or row.get("match") == "miss"]
    sheets: List[Tuple[str, List[List[Any]]]] = [
        ("Summary", [["Metric", "Value"]] + [[key, value] for key, value in summary.items()]),
        ("Results", [csv_headers()] + [[row.get(header) for header in csv_headers()] for row in rows]),
        ("Class Breakdown", [["Class", "Correct", "Total", "Accuracy", "Errors", "Most Predicted"]] + [[item["label"], item["correct"], item["total"], percent_value(item["accuracy"]), item["errors"], item["most_predicted"]] for item in classes]),
        ("Errors Misses", [csv_headers()] + [[row.get(header) for header in csv_headers()] for row in errors]),
    ]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        overrides = [
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        ]
        for i, _ in enumerate(sheets, 1):
            overrides.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        zf.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>' + "".join(overrides) + "</Types>")
        zf.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        zf.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join(f'<sheet name="{html.escape(name[:31], quote=True)}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _) in enumerate(sheets, 1)) + "</sheets></workbook>")
        zf.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1)) + "</Relationships>")
        for i, (_, sheet_rows) in enumerate(sheets, 1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", xlsx_sheet(sheet_rows))


def html_table(headers: List[str], rows: List[List[Any]]) -> str:
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(safe_cell(cell)))}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html_report(payload: Dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    classes = payload["class_breakdown"]
    sections = set(payload["sections"])
    misses = [row for row in rows if row.get("error") or row.get("match") == "miss"]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>BYO Video Runtime Report</title><style>:root{{--green:#76B900;--dark:#1A1A1A;--gray:#f5f5f5;--border:#ddd;--bad:#b91c1c;}}*{{box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;color:var(--dark);line-height:1.5}}header{{background:var(--dark);color:#fff;padding:32px 40px}}header h1{{margin:0 0 6px;font-size:28px}}header .sub{{color:#aaa}}main{{max-width:1200px;margin:0 auto;padding:32px 40px}}section{{margin-bottom:42px}}h2{{border-bottom:2px solid var(--green);padding-bottom:6px;text-transform:uppercase;font-size:16px;letter-spacing:.06em}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.stat{{background:var(--gray);border-radius:8px;padding:18px;text-align:center}}.val{{font-size:30px;font-weight:800;color:var(--green)}}.lbl{{font-size:12px;color:#666;text-transform:uppercase}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{background:var(--dark);color:#fff;text-align:left;padding:9px}}td{{border-bottom:1px solid var(--border);padding:8px;vertical-align:top}}tr:nth-child(even) td{{background:var(--gray)}}.bad{{color:var(--bad);font-weight:700}}.note{{background:#fffbef;border-left:4px solid var(--green);padding:14px 18px;border-radius:0 6px 6px 0}}</style></head><body>",
        "<header><h1>BYO Video Runtime Report</h1>",
        f"<div class='sub'>{html.escape(summary['dataset'])} | {html.escape(summary['model'])} | generated {html.escape(summary['generated_at'])}</div></header><main>",
    ]
    if "overview" in sections:
        parts.append("<section><h2>Results Summary</h2><div class='stats'>")
        for value, label in [
            (summary["total"], "Total samples"),
            (summary["completed"], "Completed"),
            (percent_value(summary.get("accuracy")), "Accuracy"),
            (summary["errors"], "Errors"),
        ]:
            parts.append(f"<div class='stat'><div class='val'>{html.escape(str(value))}</div><div class='lbl'>{html.escape(label)}</div></div>")
        parts.append("</div></section>")
    if "run_metrics" in sections:
        parts.append("<section><h2>Run Metrics</h2>")
        parts.append(html_table(["Metric", "Value"], [["Batch E2E", seconds_value(summary.get("batch_e2e_seconds"))], ["Video req/s", summary.get("video_requests_per_second")], ["Median video E2E", seconds_value(summary.get("median_video_e2e"))], ["Average video E2E", seconds_value(summary.get("avg_video_e2e"))], ["Prompt hash", summary.get("prompt_hash")]]))
        parts.append("</section>")
    if "evaluation" in sections:
        parts.append("<section><h2>Evaluation</h2>")
        parts.append(html_table(["Metric", "Value"], [["Correct", summary["correct"]], ["Accuracy", percent_value(summary.get("accuracy"))], ["Average answer score", percent_value(summary.get("answer_score_average"))], ["Hazard accuracy", percent_value(summary.get("hazard_accuracy"))], ["Evaluated rows", len([row for row in rows if row.get("match")])]]))
        parts.append("</section>")
    if "class_breakdown" in sections:
        parts.append("<section><h2>Per-Class Breakdown</h2>")
        parts.append(html_table(["Class", "Correct / Total", "Accuracy", "Errors", "Most Predicted"], [[item["label"], f"{item['correct']} / {item['total']}", percent_value(item["accuracy"]), item["errors"], item["most_predicted"]] for item in classes]))
        parts.append("</section>")
    if "error_analysis" in sections:
        parts.append("<section><h2>Errors and Misses</h2>")
        parts.append(html_table(["Video", "Expected", "Prediction", "Match", "Error"], [[row["video"], row["expected_label"], row["prediction_label"], row["match"], row["error"]] for row in misses[:80]]))
        parts.append("</section>")
    if "result_table" in sections:
        parts.append("<section><h2>Per-Video Results</h2>")
        parts.append(html_table(["Video", "Task", "Domain", "Expected", "Model output", "Metric", "Score", "Match", "E2E", "Description / error"], [[row["video"], row.get("task"), row.get("capability_domain") or row.get("domain"), row["expected_label"], row["prediction_label"], row.get("metric"), percent_value(row.get("answer_score")) if row.get("answer_score") is not None else "", row["match"], seconds_value(row.get("e2e_seconds")), row["error"] or row["description"]] for row in rows[:200]]))
        parts.append("</section>")
    if "samples" in sections:
        sample_rows = (misses[:5] or rows[:5])
        parts.append("<section><h2>Representative Samples</h2>")
        parts.append(html_table(["Video", "Expected", "Prediction", "Observation"], [[row["video"], row["expected_label"], row["prediction_label"], row["error"] or row["description"]] for row in sample_rows]))
        parts.append("</section>")
    if "prompt_params" in sections:
        bm = payload["snapshot"].get("batch_metrics") or {}
        params = (bm.get("params") or payload["snapshot"].get("defaults") or {})
        parts.append("<section><h2>Prompt and Parameters</h2>")
        parts.append(f"<pre>{html.escape(json.dumps(params, indent=2, default=str))}</pre></section>")
    if "infrastructure" in sections:
        server = payload["snapshot"].get("server") or {}
        parts.append("<section><h2>Backend and Instance</h2>")
        parts.append(html_table(["Field", "Value"], [[key, server.get(key)] for key in ["instance", "host_ip", "backend", "model", "base_url", "gpu", "vram_free_mib", "vram_total_mib", "ssd_free_gb", "ssd_total_gb"]]))
        parts.append("</section>")
    if "recommendations" in sections:
        parts.append("<section><h2>Recommendations</h2><div class='note'><ul>")
        parts.extend(f"<li>{html.escape(rec)}</li>" for rec in payload["recommendations"])
        parts.append("</ul></div></section>")
    parts.append("</main></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def ppt_escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def emu(inches: float) -> int:
    return int(inches * 914400)


def ppt_text_shape(shape_id: int, name: str, x: float, y: float, w: float, h: float, lines: Iterable[str], size: int = 22, color: str = "1A1A1A", bold: bool = False, fill: Optional[str] = None) -> str:
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else ""
    paragraphs = []
    for line in lines:
        paragraphs.append(f'<a:p><a:r><a:rPr lang="en-US" sz="{size * 100}" b="{1 if bold else 0}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr><a:t>{ppt_escape(line)}</a:t></a:r></a:p>')
    return f'''<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{ppt_escape(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_xml}</p:spPr><p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="91440" rIns="91440" bIns="91440"/><a:lstStyle/>{"".join(paragraphs)}</p:txBody></p:sp>'''


def ppt_slide_xml(shapes: List[str]) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{"".join(shapes)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'''


def ppt_metric_cards(summary: Dict[str, Any]) -> List[str]:
    cards = [
        (summary["total"], "samples"),
        (summary["completed"], "completed"),
        (percent_value(summary.get("accuracy")) or "n/a", "accuracy"),
        (summary["errors"], "errors"),
    ]
    shapes = []
    for i, (value, label) in enumerate(cards):
        x = 0.7 + i * 3.05
        shapes.append(ppt_text_shape(20 + i, label, x, 4.8, 2.65, 1.1, [str(value), label.upper()], 18, "FFFFFF", True, "1A1A1A" if i != 3 else "8B1A1A"))
    return shapes


def write_pptx_export(payload: Dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    classes = payload["class_breakdown"]
    sections = set(payload["sections"])
    misses = [row for row in rows if row.get("error") or row.get("match") == "miss"]
    slides: List[List[str]] = []
    title_shapes = [
        ppt_text_shape(2, "Title", 0.65, 0.55, 11.8, 1.1, ["BYO Video Runtime Report"], 34, "76B900", True),
        ppt_text_shape(3, "Subtitle", 0.7, 1.55, 11.6, 0.8, [f"{summary['dataset']} | {summary['model']} | {summary['generated_at']}"], 16, "666666"),
        ppt_text_shape(4, "Claim", 0.7, 2.45, 11.5, 1.3, [f"{summary['completed']} of {summary['total']} videos completed; accuracy {percent_value(summary.get('accuracy')) or 'not yet evaluated'}; errors {summary['errors']}."], 24, "1A1A1A", True, "F0F7E6"),
    ] + ppt_metric_cards(summary)
    slides.append(title_shapes)
    if "run_metrics" in sections:
        slides.append([
            ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Runtime metrics"], 30, "1A1A1A", True),
            ppt_text_shape(3, "Metrics", 0.8, 1.4, 5.7, 4.9, [
                f"Batch E2E: {seconds_value(summary.get('batch_e2e_seconds')) or 'n/a'}",
                f"Video req/s: {summary.get('video_requests_per_second') or 'n/a'}",
                f"Median video E2E: {seconds_value(summary.get('median_video_e2e')) or 'n/a'}",
                f"Average video E2E: {seconds_value(summary.get('avg_video_e2e')) or 'n/a'}",
                f"Prompt hash: {summary.get('prompt_hash') or 'n/a'}",
            ], 20),
            ppt_text_shape(4, "Frame", 7.0, 1.4, 5.2, 4.9, ["Performance readout should be interpreted with the selected frame budget, max pixels, concurrency, and backend context guard in mind."], 22, "FFFFFF", True, "1A1A1A"),
        ])
    if "evaluation" in sections:
        evaluated = len([row for row in rows if row.get("match")])
        lines = [
            f"Correct: {summary['correct']} / {evaluated or 'not evaluated'}",
            f"Accuracy: {percent_value(summary.get('accuracy')) or 'n/a'}",
            f"Average answer score: {percent_value(summary.get('answer_score_average')) or 'n/a'}",
            f"Hazard accuracy: {percent_value(summary.get('hazard_accuracy')) or 'n/a'}",
            f"Runtime errors: {summary['errors']}",
        ]
        if classes:
            weakest = classes[:3]
            lines.append("Lowest-scoring classes: " + "; ".join(f"{item['label']} {percent_value(item['accuracy'])}" for item in weakest))
        slides.append([
            ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Evaluation summary"], 30, "1A1A1A", True),
            ppt_text_shape(3, "Evaluation", 0.8, 1.35, 11.7, 5.4, lines, 22, "1A1A1A", False, "F0F7E6"),
        ])
    if "class_breakdown" in sections:
        lines = [f"{item['label']}: {item['correct']}/{item['total']} ({percent_value(item['accuracy'])}) - most predicted {item['most_predicted']}" for item in classes[:8]]
        slides.append([
            ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Per-class performance"], 30, "1A1A1A", True),
            ppt_text_shape(3, "Classes", 0.8, 1.35, 11.7, 5.4, lines or ["No class-level results yet."], 16),
        ])
    if "error_analysis" in sections:
        lines = [f"{row['video']}: {row['expected_label']} -> {row['prediction_label'] or 'ERROR'}" for row in misses[:10]]
        slides.append([
            ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Errors and misses"], 30, "1A1A1A", True),
            ppt_text_shape(3, "Errors", 0.8, 1.35, 11.7, 5.4, lines or ["No errors or misses in the current exported result set."], 16),
        ])
    if "result_table" in sections:
        lines = [
            f"{row['video']} [{row.get('task') or 'task n/a'}]: score {percent_value(row.get('answer_score')) or row.get('match') or 'n/a'}; E2E {seconds_value(row.get('e2e_seconds')) or 'n/a'}"
            for row in rows[:9]
        ]
        slides.append([
            ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Per-video result sample"], 30, "1A1A1A", True),
            ppt_text_shape(3, "Results", 0.8, 1.35, 11.7, 5.4, lines or ["No video rows are loaded yet."], 15),
        ])
    if "samples" in sections:
        sample_rows = (misses[:5] or rows[:5])
        lines = [f"{row['video']}: {row['prediction_label'] or 'unrun'} - {str(row['description'] or row['error'])[:110]}" for row in sample_rows]
        slides.append([
            ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Representative evidence"], 30, "1A1A1A", True),
            ppt_text_shape(3, "Samples", 0.8, 1.35, 11.7, 5.4, lines or ["No sample results yet."], 16),
        ])
    if "prompt_params" in sections:
        bm = payload["snapshot"].get("batch_metrics") or {}
        params = bm.get("params") or payload["snapshot"].get("defaults") or {}
        param_lines = [
            f"Run label: {summary.get('run_label') or 'Custom prompt'}",
            f"Prompt hash: {summary.get('prompt_hash') or 'n/a'}",
            f"FPS: {params.get('fps')}",
            f"Max pixels/frame: {params.get('max_pixels')}",
            f"Max output tokens: {params.get('max_tokens')}",
            f"Temperature / top P / repetition: {params.get('temperature')} / {params.get('top_p')} / {params.get('repetition_penalty')}",
            f"Max input frames: {params.get('max_frames')}",
        ]
        slides.append([
            ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Prompt and parameters"], 30, "1A1A1A", True),
            ppt_text_shape(3, "Params", 0.8, 1.35, 11.7, 5.4, param_lines, 18),
        ])
    if "recommendations" in sections:
        slides.append([
            ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Recommended next actions"], 30, "1A1A1A", True),
            ppt_text_shape(3, "Recommendations", 0.8, 1.35, 11.7, 5.4, payload["recommendations"], 21, "1A1A1A", False, "FFFBED"),
        ])
    if "infrastructure" in sections:
        server = payload["snapshot"].get("server") or {}
        lines = [f"{key}: {server.get(key)}" for key in ["instance", "host_ip", "backend", "model", "gpu", "vram_free_mib", "vram_total_mib", "ssd_free_gb"]]
        slides.append([
            ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Backend and instance"], 30, "1A1A1A", True),
            ppt_text_shape(3, "Infra", 0.8, 1.35, 11.7, 5.4, lines, 18),
        ])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        slide_overrides = "".join(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, len(slides) + 1))
        zf.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>' + slide_overrides + "</Types>")
        zf.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
        sld_ids = "".join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, len(slides) + 1))
        zf.writestr("ppt/presentation.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldIdLst>{sld_ids}</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="wide"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
        rels = "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, len(slides) + 1))
        zf.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + rels + "</Relationships>")
        for i, shapes in enumerate(slides, 1):
            zf.writestr(f"ppt/slides/slide{i}.xml", ppt_slide_xml(shapes))


def create_export(fmt: str, sections: Optional[Iterable[str]]) -> Path:
    fmt = (fmt or "html").lower().strip()
    if fmt == "report":
        fmt = "html"
    if fmt not in {"html", "json", "csv", "xlsx", "pptx"}:
        raise ClientInputError(f"Unsupported export format: {fmt}")
    section_ids = export_section_ids(sections)
    payload = export_payload(section_ids)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / export_stem(fmt)
    if fmt == "html":
        write_html_report(payload, path)
    elif fmt == "json":
        write_json_export(payload, path)
    elif fmt == "csv":
        write_csv_export(payload["rows"], path)
    elif fmt == "xlsx":
        write_xlsx_export(payload, path)
    elif fmt == "pptx":
        write_pptx_export(payload, path)
    log(f"Exported {fmt.upper()} report to {path}")
    return path


def resolve_export_download(filename: str) -> Path:
    safe_name = Path(urllib.parse.unquote(filename)).name
    if not safe_name:
        raise ClientInputError("Missing export filename")
    path = EXPORT_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(safe_name)
    return path


def compact_run_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for result in results:
        evaluation = result.get("evaluation") or {}
        parsed = result.get("json") or {}
        compact.append({
            "name": result.get("name"),
            "task": evaluation.get("task"),
            "domain": evaluation.get("capability_domain") or evaluation.get("domain"),
            "metric": evaluation.get("metric"),
            "answer_score": evaluation.get("answer_score"),
            "expected_label": evaluation.get("expected_answer") or evaluation.get("expected_label"),
            "predicted_label": evaluation.get("predicted_answer") or parsed.get("prediction_label"),
            "predicted_class_id": parsed.get("prediction_class_id"),
            "is_correct": evaluation.get("is_correct"),
            "error": result.get("error"),
            "metrics": result.get("metrics") or {},
        })
    return compact


def make_run_context(run_label: str, system_prompt: str, user_prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    prompt_mode = str(params.get("prompt_mode") or "auto")
    reasoning_enabled = bool_param(params.get("reasoning_enabled"))
    server = detect_server()
    profile = reasoning_profile_for_model(
        str(params.get("reasoning_model") or server.get("model") or ""),
        str(params.get("reasoning_backend") or server.get("backend") or ""),
        str(params.get("reasoning_format") or "auto"),
    )
    hash_parts = [
        system_prompt,
        user_prompt,
        prompt_mode,
        "reasoning" if reasoning_enabled else "plain",
        profile["id"] if reasoning_enabled else "",
    ]
    prompt_hash = hashlib.sha1("\n---\n".join(hash_parts).encode("utf-8", "ignore")).hexdigest()[:10]
    return {
        "run_id": time.strftime("%Y%m%d-%H%M%S") + "-" + prompt_hash,
        "run_label": run_label or "Custom prompt",
        "prompt_hash": prompt_hash,
        "prompt_mode": prompt_mode,
        "reasoning_prompt": reasoning_enabled or "<think>" in user_prompt.lower() or "reasoning" in run_label.lower(),
        "reasoning_format": profile if reasoning_enabled else None,
        "params": params,
    }


def run_one_recorded(video: Dict[str, Any], system_prompt: str, user_prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    result: Optional[Dict[str, Any]] = None
    try:
        result = run_one(video, system_prompt, user_prompt, params)
    except Exception as exc:
        effective_system_prompt, effective_user_prompt, prompt_source = prompts_for_video(
            video,
            system_prompt,
            user_prompt,
            params,
        )
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
            "prompt_source": prompt_source,
            "system_prompt_used": effective_system_prompt,
            "user_prompt_used": effective_user_prompt,
            "error": str(exc),
        }
        result["evaluation"] = evaluate_result(video, result)
        write_fiftyone_result(video, result)
    finally:
        clear_active_request(video, f"Finished {video.get('name') or 'video'}")
    return result or {}


def run_batch(ids: Iterable[str], concurrency: int, system_prompt: str, user_prompt: str, params: Dict[str, Any], run_label: str = "") -> None:
    snap = snapshot()
    videos_by_id = {v["id"]: v for v in snap["videos"]}
    selected = [videos_by_id[i] for i in ids if i in videos_by_id] or list(videos_by_id.values())
    dataset_repo = str(snap.get("dataset_repo") or DEFAULT_DATASET)
    batch_started = time.monotonic()
    started_epoch = time.time()
    last_result_epoch: Optional[float] = None
    run_context = make_run_context(run_label, system_prompt, user_prompt, params)
    update_state(
        running=True,
        active_requests={},
        results=[],
        progress={
            "mode": "inference",
            "phase": "starting",
            "done": 0,
            "total": len(selected),
            "errors": 0,
            "started_epoch": started_epoch,
            "updated_epoch": started_epoch,
            "last_result_epoch": None,
            "concurrency": concurrency,
            "run_label": run_context["run_label"],
            "prompt_hash": run_context["prompt_hash"],
            "last_event": f"Running {len(selected)} videos with concurrency={concurrency}",
            "last_error": None,
            "active_count": 0,
            "active_requests": [],
        },
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
            last_result_epoch = time.time()
            last_event = f"Failed {video['name']}" if result.get("error") else f"Completed {video['name']}"
            with STATE_LOCK:
                active = dict(STATE.get("active_requests") or {})
                STATE["results"] = results
                STATE["progress"] = {
                    "mode": "inference",
                    "phase": "collecting",
                    "done": len(results),
                    "total": len(selected),
                    "errors": errors,
                    "started_epoch": started_epoch,
                    "updated_epoch": last_result_epoch,
                    "last_result_epoch": last_result_epoch,
                    "concurrency": concurrency,
                    "run_label": run_context["run_label"],
                    "prompt_hash": run_context["prompt_hash"],
                    "last_event": last_event,
                    "last_error": str(result.get("error") or "") or None,
                    "active_count": len(active),
                    "active_requests": list(active.values()),
                }
                STATE["batch_metrics"] = batch_summary(dataset_repo, results, len(selected), errors, concurrency, batch_started, "running", run_context)
            RESULTS_FILE.write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")
    final_summary = batch_summary(dataset_repo, results, len(selected), errors, concurrency, batch_started, "complete", run_context)
    finished_epoch = time.time()
    with STATE_LOCK:
        STATE["running"] = False
        STATE["active_requests"] = {}
        STATE["progress"] = {
            "mode": "inference",
            "phase": "complete",
            "done": len(results),
            "total": len(selected),
            "errors": errors,
            "started_epoch": started_epoch,
            "updated_epoch": finished_epoch,
            "finished_epoch": finished_epoch,
            "last_result_epoch": last_result_epoch,
            "concurrency": concurrency,
            "run_label": run_context["run_label"],
            "prompt_hash": run_context["prompt_hash"],
            "last_event": f"Batch complete: {len(results) - errors} ok, {errors} errors",
            "last_error": None,
            "active_count": 0,
            "active_requests": [],
        }
        STATE["batch_metrics"] = final_summary
        STATE["batch_history"] = (STATE.get("batch_history") or [])[-19:] + [final_summary]
        STATE["run_history"] = (STATE.get("run_history") or [])[-5:] + [{"summary": final_summary, "results": compact_run_results(results)}]
    RESULTS_FILE.write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")
    log(f"Batch complete: {len(results) - errors} ok, {errors} errors")


def launch_fiftyone(port: int) -> str:
    global FIFTYONE_SESSION
    snap = snapshot()
    import fiftyone as fo

    dataset_name = snap.get("fo_dataset_name")
    if not dataset_name:
        names = set(fo.list_datasets())
        repo_name = str(snap.get("dataset_repo") or DEFAULT_DATASET).replace("/", "_")
        candidates = [
            f"{repo_name}_runtime",
            repo_name,
            DEFAULT_DATASET.replace("/", "_") + "_runtime",
            DEFAULT_DATASET.replace("/", "_"),
        ]
        dataset_name = next((name for name in candidates if name in names), None)
        if dataset_name:
            update_state(dataset_source="fiftyone", fo_dataset_name=dataset_name)
            log(f"Reattached to existing FiftyOne dataset {dataset_name}")
    if not dataset_name:
        raise RuntimeError("Load a dataset before launching the FiftyOne app; no existing FiftyOne runtime dataset was found")
    dataset = fo.load_dataset(dataset_name)
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
<title>Cosmos BYO Video Batch Inference</title>
<style>
:root { color-scheme: light; --ink:#1f2937; --muted:#6b7280; --line:#d8dee8; --panel:#ffffff; --bg:#f5f7fb; --accent:#0f766e; --warn:#b45309; --bad:#b91c1c; }
* { box-sizing: border-box; }
html, body { max-width:100%; overflow-x:hidden; }
body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
header { padding:18px 28px; border-bottom:1px solid var(--line); background:#fff; display:flex; align-items:center; justify-content:space-between; gap:20px; min-width:0; }
h1 { font-size:20px; margin:0; letter-spacing:0; }
main { width:min(100% - 32px, 1680px); margin:0 auto; padding:20px 0; display:grid; grid-template-columns:minmax(280px, clamp(300px, 24vw, 420px)) minmax(0, 1fr); gap:18px; align-items:start; }
section, aside { min-width:0; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
aside { position:sticky; top:16px; max-height:calc(100vh - 104px); overflow:auto; }
section { overflow:hidden; }
label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
input, textarea, select { width:100%; min-width:0; border:1px solid var(--line); border-radius:6px; padding:9px 10px; font:inherit; background:#fff; }
select { overflow:hidden; text-overflow:ellipsis; }
input[type=range] { padding:0; }
input[type=checkbox] { width:auto; }
textarea { min-height:96px; max-height:min(42vh, 520px); resize:vertical; overflow-y:auto; line-height:1.45; }
#systemPrompt { min-height:88px; }
#userPrompt { min-height:120px; }
button { border:0; border-radius:6px; padding:9px 12px; font-weight:650; color:#fff; background:var(--accent); cursor:pointer; }
button.secondary { background:#334155; }
button.warn { background:var(--warn); }
button:disabled { opacity:.55; cursor:not-allowed; }
.actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.kv { display:grid; grid-template-columns:minmax(86px, max-content) minmax(0, 1fr); gap:6px 10px; font-size:13px; color:var(--muted); }
.kv > div { min-width:0; overflow-wrap:anywhere; word-break:break-word; }
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
.budget-ok { color:var(--accent); }
.budget-warn { color:var(--warn); }
.budget-bad { color:var(--bad); }
.status-panel { margin-top:14px; border:1px solid var(--line); border-radius:8px; padding:12px; background:#f8fafc; }
.status-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:8px; }
.status-title { font-weight:700; color:var(--ink); }
.status-pct { font-variant-numeric:tabular-nums; font-weight:700; color:var(--accent); }
.status-panel .kv { grid-template-columns:96px 1fr; margin-top:10px; }
.status-note { border-left:3px solid var(--line); padding-left:9px; color:var(--muted); font-size:12px; line-height:1.35; margin:10px 0 0; }
.status-note.warn { border-left-color:var(--warn); color:var(--warn); }
.status-note.bad { border-left-color:var(--bad); color:var(--bad); }
.export-panel { margin-top:10px; border:1px solid var(--line); border-radius:8px; padding:12px; background:#f8fafc; }
.export-sections { display:grid; grid-template-columns:repeat(2, minmax(180px, 1fr)); gap:2px 14px; margin:8px 0 10px; }
.export-sections .toggle-row { margin:3px 0; }
.export-links { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.export-link { display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:5px 9px; font-size:12px; color:var(--accent); background:#fff; text-decoration:none; }
.paper-panel { margin-top:12px; border:1px solid var(--line); border-radius:8px; padding:12px; background:#f8fafc; }
.paper-panel .kv { grid-template-columns:minmax(72px, max-content) minmax(0, 1fr); margin-top:8px; max-height:260px; overflow:auto; padding-right:4px; }
.guide { display:grid; gap:8px; }
.step { border-left:3px solid var(--line); padding-left:10px; color:var(--muted); font-size:13px; overflow-wrap:anywhere; }
.step strong { color:var(--ink); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { border-bottom:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; overflow-wrap:anywhere; }
th { color:var(--muted); font-size:12px; font-weight:650; }
.results { max-height:420px; overflow:auto; border:1px solid var(--line); border-radius:6px; }
.log { height:150px; overflow:auto; background:#0f172a; color:#d1fae5; padding:10px; border-radius:6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; white-space:pre-wrap; }
.pill { display:inline-flex; align-items:center; min-width:0; border:1px solid var(--line); border-radius:999px; padding:3px 8px; font-size:12px; color:var(--muted); background:#fff; }
#serverPill { max-width:min(52vw, 560px); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.progress { height:8px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
.progress div { height:100%; width:0; background:var(--accent); transition:width .2s ease; }
@media (min-width: 1440px) { main { width:calc(100% - 40px); } }
@media (max-width: 900px) {
  header { padding:14px; align-items:flex-start; flex-direction:column; }
  #serverPill { max-width:100%; }
  main { display:block; width:calc(100% - 28px); max-width:none; margin:0 14px; padding:14px 0; }
  section, aside { width:auto; max-width:100%; margin-right:14px; }
  section { margin-top:14px; }
  aside { position:static; max-height:none; overflow:visible; }
  .actions button { flex:1 1 100%; }
  .params, .export-sections { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Cosmos BYO Video Batch Inference</h1>
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
  <div class="paper-panel">
    <label>Prompt / dataset source</label>
    <input id="paperSource" value="https://huggingface.co/papers/2603.29281" />
    <p class="hint">Use an arXiv/HF URL, local PDF, recipe path, or HF dataset. Import prompts updates the dropdown; import + load also loads the linked dataset when one is discovered.</p>
    <div class="actions">
      <button class="secondary" id="promptImportBtn">Import prompts</button>
      <button class="secondary" id="paperBtn">Import paper + load</button>
    </div>
    <div class="kv" id="paperKv"></div>
  </div>
  <label>Max videos to load (0 = all)</label>
  <input id="maxVideos" type="number" min="0" value="20" />
  <label>Concurrency</label>
  <input id="concurrency" type="number" min="1" value="4" />
  <div class="actions">
    <button id="loadBtn">Load dataset</button>
    <button class="warn" id="smokeBtn">Run smoke</button>
    <button class="secondary" id="foBtn">Open FiftyOne</button>
  </div>
  <h3>Run status</h3>
  <div class="status-panel">
    <div class="status-head"><span class="status-title" id="statusTitle">Idle</span><span class="status-pct" id="statusPct">0%</span></div>
    <div class="progress"><div id="sideBar"></div></div>
    <div class="kv" id="runtimeKv"></div>
    <p class="status-note" id="runtimeNotice">No batch is running.</p>
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
  <label>Prompt preset</label>
  <select id="promptPreset"></select>
  <div class="actions prompt-actions">
    <button class="secondary" id="savePromptBtn">Save current prompt</button>
    <button class="secondary" id="reloadPromptBtn">Reload selected prompt</button>
  </div>
  <label class="toggle-row"><input id="reasoningToggle" type="checkbox" /><span><strong>Use model reasoning format</strong><br/><span id="reasoningFormatText">Auto-detects the loaded model and wraps prompts with its think/answer scaffold.</span></span></label>
  <p class="hint" id="promptStatus"></p>
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
  <h3>Context guard</h3>
  <div class="kv" id="contextKv"></div>
  <p class="hint" id="contextHint"></p>
  <div class="actions">
    <button class="secondary" id="fitBudgetBtn">Fit to model</button>
    <label class="toggle-row"><input id="allowOverContext" type="checkbox" /><span><strong>Allow over-budget OSS run</strong><br/>Only for stress testing. Over-budget OSS image-frame runs may produce model 400 errors; NIM/native-video backends are guarded by the service.</span></label>
  </div>
  <div class="actions">
    <button id="runBtn">Run selected videos</button>
    <button class="secondary" id="allBtn">Select all</button>
    <button class="secondary" id="noneBtn">Select none</button>
  </div>
  <h3>Videos</h3>
  <div class="results"><table><thead><tr><th></th><th>Preview</th><th>Name</th><th>Expected</th><th>HF row</th><th>Resolution</th><th>Duration</th><th>Source frames</th><th>Frames to VLM</th><th>Est visual tokens</th><th>Path</th></tr></thead><tbody id="videoRows"></tbody></table></div>
  <h3>Results</h3>
  <div class="results"><table><thead><tr><th>Video</th><th>Task</th><th>Domain</th><th>Expected</th><th>Model output</th><th>Metric</th><th>Score / match</th><th>TTFT</th><th>Output tok/s</th><th>E2E</th><th>Description / error</th></tr></thead><tbody id="resultRows"></tbody></table></div>
  <h3>Batch history</h3>
  <div class="results"><table><thead><tr><th>Dataset</th><th>Prompt</th><th>Status</th><th>Videos</th><th>Errors</th><th>Accuracy</th><th>Batch E2E</th><th>Video req/s</th><th>Median video E2E</th><th>Prompt hash</th></tr></thead><tbody id="batchRows"></tbody></table></div>
  <h3>Runtime log</h3>
  <div class="log" id="log"></div>
  <h3>Exports</h3>
  <div class="export-panel">
    <p class="hint">Choose the sections to include in the narrative report and PowerPoint. Raw JSON, CSV, and Excel exports include the current run data for downstream analysis.</p>
    <div class="export-sections" id="exportSections"></div>
    <div class="actions">
      <button class="secondary exportBtn" data-format="html">Report</button>
      <button class="secondary exportBtn" data-format="json">JSON</button>
      <button class="secondary exportBtn" data-format="csv">CSV</button>
      <button class="secondary exportBtn" data-format="xlsx">Excel</button>
      <button class="secondary exportBtn" data-format="pptx">PowerPoint</button>
    </div>
    <div class="export-links" id="exportLinks"></div>
  </div>
</section>
</main>
<script>
let state = null;
let initialized = false;
let contextBlocked = false;
let promptPresetSig = '';
let promptDirty = false;
let lastAppliedPromptKey = '';
let promptSyncing = false;
let promptUserSelected = false;
let reasoningProfileId = '';
const CONTEXT_WARNING_RATIO = 0.85;
const REASONING_PROFILES = {
  cosmos_think_answer: {id:'cosmos_think_answer', label:'Cosmos Reason / Cosmos3', prefix:'<think>\\nyour reasoning\\n</think>\\n<answer>\\n', suffix:'\\n</answer>', note:'Cosmos Reason/Cosmos3 prompt scaffold'},
  qwen_think_answer: {id:'qwen_think_answer', label:'Qwen3-VL', prefix:'<think>\\nyour reasoning\\n</think>\\n<answer>\\n', suffix:'\\n</answer>', note:'Qwen3-VL thinking-compatible prompt scaffold'},
  nemotron_think_answer: {id:'nemotron_think_answer', label:'Nemotron VL / Omni', prefix:'<think>\\nyour reasoning\\n</think>\\n<answer>\\n', suffix:'\\n</answer>', note:'Nemotron reasoning prompt scaffold'},
  gemma_think_answer: {id:'gemma_think_answer', label:'Gemma', prefix:'<think>\\nyour reasoning\\n</think>\\n<answer>\\n', suffix:'\\n</answer>', note:'Generic Gemma-compatible reasoning scaffold'},
  generic_think_answer: {id:'generic_think_answer', label:'Generic VLM', prefix:'<think>\\nyour reasoning\\n</think>\\n<answer>\\n', suffix:'\\n</answer>', note:'Fallback reasoning scaffold'}
};
function esc(s){ return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function api(path, body){ const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); const j = await r.json(); if(!r.ok) throw new Error(j.error||r.statusText); return j; }
function setBusy(message){ document.getElementById('progressText').textContent = message; document.getElementById('bar').style.width = '12%'; if(el('sideBar')){ el('sideBar').style.width='12%'; el('statusTitle').textContent='Starting'; el('statusPct').textContent='12%'; el('runtimeNotice').className='status-note'; el('runtimeNotice').textContent=message; } ['loadBtn','runBtn','smokeBtn','foBtn','fitBudgetBtn','paperBtn','promptImportBtn','reasoningToggle'].forEach(id=>document.getElementById(id).disabled=true); }
function checkedIds(){ return [...document.querySelectorAll('.pick:checked')].map(x=>x.value); }
function el(id){ return document.getElementById(id); }
function num(id){ return Number(document.getElementById(id).value); }
function params(){ return {fps:num('fpsSlider'),max_pixels:num('maxPixelsSlider'),max_tokens:num('maxTokensSlider'),temperature:num('temperatureSlider'),top_p:num('topPSlider'),repetition_penalty:num('repPenaltySlider'),max_frames:num('maxFramesSlider')}; }
function nativeVideoMode(){ const srv=state?.server||{}; const model=String(srv.model||'').toLowerCase(); const backend=String(srv.backend||'').toLowerCase(); return backend.includes('nim') || model.includes('qwen') || model.includes('nemotron'); }
function reasoningProfiles(){ return state?.defaults?.reasoning_profiles || REASONING_PROFILES; }
function reasoningProfile(){
 const profiles=reasoningProfiles();
 const srv=state?.server||{}; const lower=`${srv.model||''} ${srv.backend||''}`.toLowerCase();
 if(lower.includes('cosmos') || /\\b(cr[123]|c3)\\b/.test(lower)) return profiles.cosmos_think_answer || REASONING_PROFILES.cosmos_think_answer;
 if(lower.includes('qwen') || lower.includes('qw3')) return profiles.qwen_think_answer || REASONING_PROFILES.qwen_think_answer;
 if(lower.includes('nemotron') || lower.includes('omni')) return profiles.nemotron_think_answer || REASONING_PROFILES.nemotron_think_answer;
 if(lower.includes('gemma')) return profiles.gemma_think_answer || REASONING_PROFILES.gemma_think_answer;
 return profiles.generic_think_answer || REASONING_PROFILES.generic_think_answer;
}
function reasoningEnabled(){ return !!el('reasoningToggle')?.checked; }
function stripReasoningWrapper(text){ const s=String(text||'').trim(); const m=s.match(/^\\s*(?:\\/think\\s*)?<think>\\s*your reasoning\\.?\\s*<\\/think>\\s*<answer>\\s*([\\s\\S]*?)\\s*<\\/answer>\\s*$/i); return m ? m[1].trim() : s; }
function applyReasoningWrapper(text, profile=reasoningProfile()){ return `${profile.prefix}${stripReasoningWrapper(text)}${profile.suffix}`.trim(); }
function effectiveUserPrompt(text){ return reasoningEnabled() ? applyReasoningWrapper(text) : text; }
function baseUserPromptForPreset(){ const text=el('userPrompt')?.value||''; return reasoningEnabled() ? stripReasoningWrapper(text) : text; }
function reasoningPayload(){ const srv=state?.server||{}; return {reasoning_enabled:reasoningEnabled(),reasoning_format:'auto',reasoning_model:srv.model||'',reasoning_backend:srv.backend||''}; }
function syncReasoningFormatLabel(){
 const profile=reasoningProfile(); const target=el('reasoningFormatText'); const model=state?.server?.model||'loaded model';
 if(reasoningEnabled() && reasoningProfileId && reasoningProfileId !== profile.id && el('userPrompt')){
  promptSyncing=true; el('userPrompt').value=applyReasoningWrapper(el('userPrompt').value, profile); promptSyncing=false; autosizeTextarea(el('userPrompt'));
 }
 reasoningProfileId=profile.id;
 if(target) target.textContent=reasoningEnabled() ? `Using ${profile.label} for ${model}; the user prompt is wrapped in <think>...</think><answer>...</answer> before inference.` : `Auto-detects the loaded model; current match is ${profile.label}.`;
}
function applyReasoningToPrompt(){
 const prompt=el('userPrompt'); if(!prompt) return; const profile=reasoningProfile(); reasoningProfileId=profile.id;
 promptSyncing=true; prompt.value=reasoningEnabled() ? applyReasoningWrapper(prompt.value, profile) : stripReasoningWrapper(prompt.value); promptSyncing=false;
 autosizeTextarea(prompt); syncPromptPresets(); updatePromptStatus(); render();
}
function fmt(n, digits=0){ if(n === null || n === undefined || Number.isNaN(Number(n))) return ''; return Number(n).toLocaleString(undefined,{maximumFractionDigits:digits}); }
function fmtMetaValue(v){ return typeof v === 'number' ? fmt(v, Number.isInteger(v) ? 0 : 2) : String(v ?? ''); }
function sec(v){ return v === null || v === undefined || Number.isNaN(Number(v)) ? '' : `${Number(v).toFixed(2)}s`; }
function rate(v){ return v === null || v === undefined || Number.isNaN(Number(v)) ? '' : Number(v).toFixed(2); }
function serverNow(){ const received=Number(state?._receivedAt||Date.now()/1000); const server=Number(state?.server_epoch||received); return server + (Date.now()/1000 - received); }
function span(v){ if(v === null || v === undefined || Number.isNaN(Number(v))) return ''; const s=Math.max(0, Math.round(Number(v))); if(s < 60) return `${s}s`; const m=Math.floor(s/60); const r=s%60; if(m < 60) return `${m}m ${r}s`; const h=Math.floor(m/60); return `${h}h ${m%60}m`; }
function statText(s){ if(!s || !s.count) return 'no samples yet'; return `min ${sec(s.min)} | max ${sec(s.max)} | median ${sec(s.median)} | avg ${sec(s.average)}`; }
function percent(v){ return v === null || v === undefined || Number.isNaN(Number(v)) ? '' : `${(Number(v)*100).toFixed(1)}%`; }
function expectedText(x){ if(!x) return ''; if(x.kind==='dataset_answer') return esc(clipText(x.final_answer||x.label||'',120)); return `${esc(x.class_id ?? '')} ${esc(x.label ?? '')}`; }
function rowSummary(row){ if(!row) return ''; const parts=[]; if(row.task) parts.push(`task=${row.task}`); if(row.capability_domain) parts.push(`domain=${row.capability_domain}`); else if(row.domain) parts.push(`domain=${row.domain}`); if(row.sft_type) parts.push(`format=${row.sft_type}`); if(row.evaluation_type) parts.push(`metric=${row.evaluation_type}`); if(row.user_prompt) parts.push(`prompt=${clipText(row.user_prompt,80)}`); if(row.label) parts.push(`label=${row.label}`); if(row.tags) parts.push(`tags=${Array.isArray(row.tags) ? row.tags.join(',') : row.tags}`); if(row.hf_path) parts.push(`path=${row.hf_path}`); return parts.join(' | '); }
function detailsJson(label,obj){ if(!obj || Object.keys(obj).length===0) return ''; return `<details class="meta-details"><summary>${esc(label)}</summary><pre>${esc(JSON.stringify(obj,null,2))}</pre></details>`; }
function simpleHash(text){ let h=0; const s=String(text||''); for(let i=0;i<s.length;i++){ h=((h<<5)-h+s.charCodeAt(i))|0; } return Math.abs(h).toString(36); }
function promptFingerprint(system,user){ return simpleHash(`${system||''}\\n---\\n${user||''}`); }
function promptKey(p, source='preset'){ if(p.key) return p.key; return `${source}:${promptFingerprint(p.system_prompt,p.user_prompt)}`; }
function selectedDatasetPromptPreset(){ const prompted=currentSelectionVideos().filter(v=>v?.dataset_row?.system_prompt && v?.dataset_row?.user_prompt); if(!prompted.length) return null; const row=prompted[0].dataset_row||{}; const uniq=new Set(prompted.map(v=>`${v.dataset_row.system_prompt}\\n---\\n${v.dataset_row.user_prompt}`)); const label=uniq.size>1 ? `Dataset row prompts (${prompted.length} videos)` : `Dataset row prompt: ${clipText(row.task||row.user_prompt,54)}`; return {key:'dataset-row', label, system_prompt:row.system_prompt, user_prompt:row.user_prompt, mode:'dataset_row', dataset_row:true, source:'Hugging Face dataset rows'}; }
function allPromptPresets(includeUnsaved=true){ const out=[]; const datasetPreset=selectedDatasetPromptPreset(); if(datasetPreset) out.push(datasetPreset); for(const p of (state?.defaults?.prompt_presets||[])){ out.push({...p, key:promptKey(p,'preset'), mode:'runtime_form'}); } const current={system_prompt:el('systemPrompt')?.value||'', user_prompt:baseUserPromptForPreset()}; if(includeUnsaved && current.user_prompt.trim()){ const key=`custom:${promptFingerprint(current.system_prompt,current.user_prompt)}`; const exists=out.some(p=>promptFingerprint(p.system_prompt,p.user_prompt)===promptFingerprint(current.system_prompt,current.user_prompt)); if(!exists) out.push({key, label:'Custom prompt (unsaved)', ...current, mode:'runtime_form', custom:true, source:'Current form'}); } return out; }
function findPromptByKey(key){ return allPromptPresets(true).find(p=>p.key===key) || null; }
function promptLabel(){ const p=findPromptByKey(el('promptPreset')?.value); const label=p ? p.label : 'Custom prompt'; return reasoningEnabled() ? `${label} + ${reasoningProfile().label} reasoning` : label; }
function promptMode(){ const p=findPromptByKey(el('promptPreset')?.value); return p?.mode || 'runtime_form'; }
function promptSuffix(p){ const tags=[]; if(p.dataset_row) tags.push('dataset rows'); if(p.qa_pair) tags.push('QA'); if(p.source_type==='recipe') tags.push('recipe'); else if(p.source_type==='hf') tags.push('HF'); else if(p.paper_import) tags.push('import'); if(p.custom) tags.push('custom'); if(p.reasoning) tags.push('reasoning'); return tags.length ? ` (${tags.join(', ')})` : ''; }
function updatePromptStatus(){ const p=findPromptByKey(el('promptPreset')?.value); const status=el('promptStatus'); if(!status) return; const reasoning=reasoningEnabled() ? ` Reasoning format: ${reasoningProfile().label}.` : ''; if(promptDirty){ status.textContent='Custom edits are active. Save them to keep this prompt in the dropdown, or run them directly.' + reasoning; return; } if(p?.dataset_row){ status.textContent='Using Hugging Face dataset row prompts. Each selected video keeps its own benchmark question and expected answer when available.' + (reasoningEnabled() ? ' The backend wraps each row prompt with the selected reasoning format.' : ''); return; } if(p){ const flags=[p.qa_pair?'QA example':null,p.evaluation_type?`metric: ${p.evaluation_type}`:null,p.paper_import?'imported':null,p.custom?'saved custom':null,p.source?`source: ${p.source}`:null,reasoningEnabled()?`reasoning: ${reasoningProfile().label}`:null].filter(Boolean); const expected=p.expected_answer ? ` | expected: ${clipText(p.expected_answer,160)}` : ''; status.textContent=(flags.join(' | ') || 'Using a built-in prompt preset.') + expected; return; } status.textContent='Choose a preset, import prompts from a source, or edit and save a custom prompt.' + reasoning; }
function setPromptFields(p){ if(!p) return; promptSyncing=true; el('systemPrompt').value=p.system_prompt||''; el('userPrompt').value=reasoningEnabled() ? applyReasoningWrapper(p.user_prompt||'') : (p.user_prompt||''); promptSyncing=false; promptDirty=false; lastAppliedPromptKey=p.key; autosizePrompts(); updatePromptStatus(); }
function syncPromptPresets(){ const prompts=allPromptPresets(true); const sig=prompts.map(p=>`${p.key}|${p.label}|${p.user_prompt}|${p.expected_answer||''}`).join('||'); const select=el('promptPreset'); const old=select.value || lastAppliedPromptKey; if(sig!==promptPresetSig){ select.innerHTML = prompts.map(p=>`<option value="${esc(p.key)}">${esc(p.label)}${promptSuffix(p)}</option>`).join(''); promptPresetSig=sig; } let next=old; const dataset=prompts.find(p=>p.dataset_row); const imported=prompts.find(p=>p.paper_import); if(promptDirty){ const currentKey=`custom:${promptFingerprint(el('systemPrompt').value,baseUserPromptForPreset())}`; next=prompts.some(p=>p.key===currentKey) ? currentKey : next; } else if(!promptUserSelected && (dataset || imported)){ next=(dataset||imported).key; } else if(!next || !prompts.some(p=>p.key===next)){ next=(dataset||imported||prompts[0]||{}).key || ''; } select.value=next; const selected=findPromptByKey(next); if(selected && !promptDirty && lastAppliedPromptKey!==next) setPromptFields(selected); updatePromptStatus(); }
function applySelectedPrompt(){ promptUserSelected=true; const selected=findPromptByKey(el('promptPreset').value); if(selected){ setPromptFields(selected); render(); } }
function markPromptDirty(){ if(promptSyncing) return; promptUserSelected=true; promptDirty=true; syncPromptPresets(); render(); }
async function saveCurrentPrompt(){ const system_prompt=el('systemPrompt').value; const user_prompt=baseUserPromptForPreset(); if(!user_prompt.trim()){ alert('Enter a user prompt before saving.'); return; } const fallback=`Custom prompt ${new Date().toLocaleString()}`; const label=window.prompt('Name this prompt', fallback) || fallback; const saved=await api('/api/prompt',{label,system_prompt,user_prompt,source:'runtime form'}); lastAppliedPromptKey=promptKey(saved.preset,'preset'); promptUserSelected=true; promptDirty=false; await poll(); el('promptPreset').value=lastAppliedPromptKey; updatePromptStatus(); }
function autosizeTextarea(textarea){ if(!textarea) return; textarea.style.height='auto'; const styles=getComputedStyle(textarea); const min=parseFloat(styles.minHeight)||0; const max=parseFloat(styles.maxHeight)||window.innerHeight*.42; const next=Math.max(min, Math.min(textarea.scrollHeight + 2, max)); textarea.style.height=next+'px'; textarea.style.overflowY=textarea.scrollHeight > max ? 'auto' : 'hidden'; }
function autosizePrompts(){ ['systemPrompt','userPrompt'].forEach(id=>autosizeTextarea(el(id))); }
function renderPaperImport(){ const p=state?.paper_import||{}; const rows=[]; if(p.source) rows.push(['source',clipText(p.source,90)]); if(p.arxiv_id) rows.push(['arXiv',p.arxiv_id]); if(p.title) rows.push(['title',clipText(p.title,90)]); if(p.selected_dataset) rows.push(['dataset',p.selected_dataset]); else if(p.source) rows.push(['dataset','none selected']); if(p.models?.length) rows.push(['model',p.models[0]]); if(p.prompt_presets?.length){ const qa=(p.prompt_presets||[]).filter(x=>x.qa_pair).length; rows.push(['prompts',`${p.prompt_presets.length}${qa ? ` (${qa} QA examples)` : ''}`]); } if(p.load_error) rows.push(['load error',clipText(p.load_error,160)]); if(p.warnings?.length) rows.push(['warnings',clipText(p.warnings.join(' | '),180)]); el('paperKv').innerHTML = rows.length ? rows.map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v)}</div>`).join('') : '<div>source</div><div>No prompts imported yet.</div>'; }
function checkedExportSections(){ return [...document.querySelectorAll('.exportSection:checked')].map(x=>x.value); }
function renderExportSections(){ const sections=state?.defaults?.export_sections||[]; const target=el('exportSections'); if(!target || target.dataset.ready) return; target.innerHTML = sections.map(s=>`<label class="toggle-row"><input class="exportSection" type="checkbox" value="${esc(s.id)}" ${s.default?'checked':''}/><span><strong>${esc(s.label)}</strong></span></label>`).join(''); target.dataset.ready='1'; }
function addExportLink(item){ const box=el('exportLinks'); const a=document.createElement('a'); a.className='export-link'; a.href=item.url; a.target='_blank'; a.textContent=`${item.format.toUpperCase()} | ${item.filename}`; box.prepend(a); }
function parseDimensionText(value){ const m=String(value||'').match(/(\\d{2,5})\\s*[xX×]\\s*(\\d{2,5})/); return m ? {width:Number(m[1]), height:Number(m[2])} : null; }
function videoNativeSize(v){ const m=v?.meta||{}; let width=Number(m.width||0); let height=Number(m.height||0); let source=width&&height?'decoded video':''; const row=v?.dataset_row||{}; const meta=row.metadata&&typeof row.metadata==='object'?row.metadata:{}; if(!(width&&height)){ for(const obj of [row,meta]){ width=Number(obj.width||obj.video_width||obj.w||0); height=Number(obj.height||obj.video_height||obj.h||0); if(width&&height){ source='dataset metadata'; break; } const parsed=parseDimensionText(obj.resolution||obj.dimensions||obj.size||obj.frame_size||''); if(parsed){ width=parsed.width; height=parsed.height; source='dataset resolution'; break; } } } return {width,height,pixels:width&&height?width*height:0,source}; }
function modelFitVisualTokens(){ return Number(state?.defaults?.model_fit_visual_tokens || 6144); }
function modelFitTargetFrames(){ return Number(state?.defaults?.model_fit_target_frames || 4); }
function sliderStep(id){ return Math.max(1, Number(el(id)?.step || 1)); }
function floorToStep(value, step, min){ return Math.max(min, Math.floor(Number(value||0)/step)*step); }
function videoPlan(v, override={}){ const m=v.meta||{}; if(nativeVideoMode()) return {frames:'server', tokens:'server', note:'server-decoded'}; const p={...params(),...override}; const duration=Number(m.duration_s||0); const requested=Math.max(1, Math.round((duration || 1) * p.fps)); const frames=p.max_frames<=0 ? requested : Math.min(requested, p.max_frames); const size=videoNativeSize(v); const nativePx=size.pixels; const effectivePx=nativePx ? Math.min(nativePx, p.max_pixels) : p.max_pixels; const patchPixels=Number(state?.defaults?.context_patch_pixels||196); const tokens=Math.ceil(frames * effectivePx / patchPixels); return {frames, requested, tokens, effectivePx, nativePx, width:size.width, height:size.height, sizeSource:size.source, note:''}; }
function sliderLabel(id, label, suffix=''){ document.getElementById(id+'Value').textContent = label + suffix; }
function sliderHint(domId, key){ const m=(state.defaults.slider_meta||{})[key]||{}; const unit=m.unit ? ' '+m.unit : ''; const recommended = key === 'max_frames' && m.recommended === 0 ? 'disabled' : fmtMetaValue(m.recommended); document.getElementById(domId+'Hint').textContent = `min ${fmtMetaValue(m.min)}${unit} | max ${fmtMetaValue(m.max)}${unit} | recommended ${recommended}${unit}. ${m.note||''}`; }
function renderParamLabels(){ const p=params(); sliderLabel('fps', p.fps, ' fps'); sliderLabel('maxPixels', fmt(p.max_pixels)); sliderLabel('maxTokens', fmt(p.max_tokens)); sliderLabel('temperature', p.temperature.toFixed(2)); sliderLabel('topP', p.top_p.toFixed(2)); sliderLabel('repPenalty', p.repetition_penalty.toFixed(2)); sliderLabel('maxFrames', p.max_frames === 0 ? 'disabled' : fmt(p.max_frames)); sliderHint('fps','fps'); sliderHint('maxPixels','max_pixels'); sliderHint('maxTokens','max_tokens'); sliderHint('temperature','temperature'); sliderHint('topP','top_p'); sliderHint('repPenalty','repetition_penalty'); sliderHint('maxFrames','max_frames'); const d=state.defaults; const buildOn=!!el('buildDefaultsToggle')?.checked; const mode=nativeVideoMode() ? 'native video_url; backend samples frames internally' : `image-frame mode; max input frames cap is ${p.max_frames === 0 ? 'disabled' : p.max_frames}`; const buildText=buildOn ? 'build.nvidia.com defaults are ON: temperature 0.6, top P 0.3, repetition 1.2. ' : ''; document.getElementById('paramSummary').textContent = `${buildText}Recommended: fps ${d.slider_meta.fps.recommended}, max pixels ${fmt(d.slider_meta.max_pixels.recommended)}, max tokens ${d.slider_meta.max_tokens.recommended}, temperature ${d.slider_meta.temperature.recommended}, top P ${d.slider_meta.top_p.recommended}, repetition ${d.slider_meta.repetition_penalty.recommended}, max frames ${d.slider_meta.max_frames.recommended}. Frame policy: ${mode}.`; }
function modelMaxLen(){ return Number(state?.server?.model_max_len || state?.defaults?.default_model_max_len || 32768); }
function reserveTokens(){ return Number(state?.server?.context_safety_reserve || state?.defaults?.context_safety_reserve || 1024); }
function patchPixels(){ return Number(state?.server?.context_patch_pixels || state?.defaults?.context_patch_pixels || 196); }
function promptTokensEst(){ const text=(el('systemPrompt').value||'') + '\\n' + effectiveUserPrompt(el('userPrompt').value||''); return Math.max(1, Math.round(text.length / 4)) + Number(state?.defaults?.context_text_tokens||50); }
function promptTokensForVideo(v){ const row=v?.dataset_row||{}; const useDataset=promptMode()==='dataset_row' && row.system_prompt && row.user_prompt; const user=useDataset ? row.user_prompt : (el('userPrompt').value||''); const text=useDataset ? `${row.system_prompt}\\n${effectiveUserPrompt(user)}` : ((el('systemPrompt').value||'') + '\\n' + effectiveUserPrompt(user)); return Math.max(1, Math.round(String(text).length / 4)) + Number(state?.defaults?.context_text_tokens||50); }
function currentSelectionVideos(){ const videos=state?.videos||[]; const ids=new Set(checkedIds()); return ids.size ? videos.filter(v=>ids.has(v.id)) : videos; }
function contextReport(){ const p=params(); const videos=currentSelectionVideos(); const maxLen=modelMaxLen(); const reserve=reserveTokens(); const allowed=Math.max(1, maxLen - p.max_tokens - reserve); const visualBudget=modelFitVisualTokens(); if(nativeVideoMode()) return {enabled:false, reason:'native', videos, maxLen, reserve, allowed, visualBudget}; const prompt=promptTokensEst(); const rows=videos.map(v=>{ const plan=videoPlan(v); const rowPrompt=promptTokensForVideo(v); const input=Number(plan.tokens||0)+rowPrompt; const ratio=input/allowed; const workloadRatio=Number(plan.tokens||0)/visualBudget; return {video:v, plan, input, prompt:rowPrompt, ratio, workloadRatio, over:input>allowed, warn:input<=allowed && ratio>=CONTEXT_WARNING_RATIO, timeoutRisk:Number(plan.tokens||0)>visualBudget}; }); const worst=rows.length ? rows.reduce((a,b)=>b.input>a.input?b:a, rows[0]) : null; const worstWorkload=rows.length ? rows.reduce((a,b)=>(b.plan.tokens||0)>(a.plan.tokens||0)?b:a, rows[0]) : null; return {enabled:true, videos, rows, worst, worstWorkload, maxLen, reserve, allowed, visualBudget, prompt, over:rows.some(r=>r.over), warn:rows.some(r=>r.warn), timeoutRisk:rows.some(r=>r.timeoutRisk)}; }
function renderContextGuard(){ const kv=el('contextKv'); const hint=el('contextHint'); const fit=el('fitBudgetBtn'); const allow=el('allowOverContext'); const busy=!!(state.running||state.loading_dataset); const r=contextReport(); if(!r.enabled){ kv.innerHTML = [['mode','native video / NIM'],['model context',fmt(r.maxLen)+' tokens'],['guard','delegated to model service']].map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v)}</div>`).join(''); hint.className='hint'; hint.textContent='This backend receives video_url/native video input, so the microservice owns frame sampling and context validation.'; fit.disabled=true; allow.disabled=true; return false; } fit.disabled=busy || !r.videos.length; allow.disabled=busy; const worst=r.worst; const load=r.worstWorkload; const size=load?.plan?.width&&load?.plan?.height ? `${fmt(load.plan.width)}x${fmt(load.plan.height)} (${load.plan.sizeSource||'video'})` : ''; kv.innerHTML = [['mode','OSS image frames'],['model context',fmt(r.maxLen)+' tokens'],['input budget',fmt(r.allowed)+' tokens'],['visual fit target',fmt(r.visualBudget)+' visual tokens/request'],['selected videos',fmt(r.videos.length)],['worst context',worst ? `${worst.video.name}: ${fmt(worst.input)} input tokens` : ''],['worst visual workload',load ? `${load.video.name}: ${fmt(load.plan.tokens)} visual tokens (${fmt(load.plan.frames)} frames, ${fmt(load.plan.effectivePx)} px/frame)` : ''],['dataset resolution',size]].map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v)}</div>`).join(''); if(!r.videos.length){ hint.className='hint'; hint.textContent='Load a dataset to estimate both model context and visual prefill workload.'; return false; } if(r.over){ hint.className='hint budget-bad'; hint.textContent = allow.checked ? 'Over-budget override is enabled. This is useful for stress testing, but the OSS backend may still return 400 errors.' : 'These settings are likely to exceed the OSS model context and cause a 400. Use Fit to model, lower fps/max pixels/max frames, or explicitly allow an over-budget stress test.'; return !allow.checked; } if(r.timeoutRisk){ hint.className='hint budget-warn'; hint.textContent='These settings fit the context window, but the selected dataset videos exceed the visual-workload target and may timeout. Fit to model lowers max pixels and max frames using the actual dataset resolution/duration.'; return false; } if(r.warn){ hint.className='hint budget-warn'; hint.textContent='These settings are close to the context limit; they should run, but prefill may be slow.'; return false; } hint.className='hint budget-ok'; hint.textContent='These settings fit within both the estimated OSS context budget and the dataset-aware visual workload target.'; return false; }
function fitToModel(){ const r=contextReport(); if(!r.enabled || !r.videos.length) return; const p=params(); const minPx=Number(el('maxPixelsSlider').min||65536); const maxPxStep=sliderStep('maxPixelsSlider'); const maxFramesMax=Number(el('maxFramesSlider').max||128); let safeFrames=Math.max(1, Math.min(maxFramesMax, p.max_frames>0?p.max_frames:maxFramesMax, modelFitTargetFrames())); let safePixels=Math.max(minPx, Math.min(p.max_pixels, Number(state?.defaults?.slider_meta?.max_pixels?.recommended||p.max_pixels))); const perVideoBudgets=[]; for(const v of r.videos){ const prompt=promptTokensForVideo(v); const contextVisual=Math.max(1, Math.floor((modelMaxLen() - p.max_tokens - reserveTokens() - prompt) * 0.85)); const budget=Math.max(1, Math.min(modelFitVisualTokens(), contextVisual)); const plan=videoPlan(v,{max_pixels:safePixels,max_frames:safeFrames}); const frames=Math.max(1, Math.min(safeFrames, Number(plan.requested||safeFrames))); const allowedPixels=Math.floor(budget * patchPixels() / frames); safePixels=Math.min(safePixels, allowedPixels); perVideoBudgets.push({v,budget}); } safePixels=floorToStep(safePixels,maxPxStep,minPx); for(const item of perVideoBudgets){ const plan=videoPlan(item.v,{max_pixels:safePixels,max_frames:safeFrames}); const effectivePx=Math.max(1, Number(plan.effectivePx||safePixels)); const framesAllowed=Math.max(1, Math.floor(item.budget * patchPixels() / effectivePx)); safeFrames=Math.min(safeFrames, framesAllowed, Number(plan.requested||safeFrames)); } safeFrames=Math.max(1, Math.min(maxFramesMax, Math.floor(safeFrames))); el('maxPixelsSlider').value=safePixels; el('maxFramesSlider').value=safeFrames; render(); }
function latestRuntimeError(){ const p=state?.progress||{}; if(p.last_error) return String(p.last_error); const results=[...(state?.results||[])].reverse(); const failed=results.find(r=>r && r.error); return failed ? `${failed.name||'video'}: ${failed.error}` : ''; }
function clipText(text, max=260){ const s=String(text||''); return s.length > max ? s.slice(0, max-1)+'...' : s; }
function activeRequestList(){ const top=state?.active_requests||{}; if(Array.isArray(top)) return top; const vals=Object.values(top); if(vals.length) return vals; const p=state?.progress||{}; const nested=p.active_requests||{}; return Array.isArray(nested) ? nested : Object.values(nested); }
function renderRuntimeStatus(){
 const p=state.progress||{}; const bm=state.batch_metrics||{}; const activeReqs=activeRequestList(); const isPaper=!!(state.loading_dataset && p.mode==='paper_import'); const isLoad=!!(state.loading_dataset || p.mode==='dataset_load' || p.mode==='paper_import'); const active=!!(state.running || state.loading_dataset);
 const modelReqs=activeReqs.filter(r=>['model_wait','model_retry'].includes(String(r.stage||'')));
 const modelWait=!!(state.running && modelReqs.length);
 const total=Number(p.total ?? bm.total ?? 0); const done=Number(p.done ?? bm.completed ?? 0); const errors=Number(p.errors ?? bm.errors ?? 0); const pct=total ? Math.min(100, Math.round(100*done/total)) : (active ? (isPaper ? 8 : 5) : 0);
 const now=serverNow(); const started=Number(p.started_epoch||0); const finished=Number(p.finished_epoch||0); const elapsed=started ? ((active ? now : (finished || Number(p.updated_epoch||now))) - started) : Number(bm.batch_wall_seconds||0);
 const lastActivity=Number(p.updated_epoch||p.last_result_epoch||0); const waitSince=active ? now - (lastActivity || started || now) : null;
 const activePool=modelReqs.length ? modelReqs : activeReqs;
 const oldestActive=activePool.length ? activePool.reduce((a,b)=>Number(b.started_epoch||now)<Number(a.started_epoch||now)?b:a, activePool[0]) : null;
 const requestElapsed=oldestActive ? now - Number(oldestActive.started_epoch||now) : null;
 const requestTimeout=Number(oldestActive?.timeout_seconds || state?.defaults?.request_timeout_seconds || 0);
 const requestRemaining=requestTimeout && requestElapsed !== null ? Math.max(0, requestTimeout-requestElapsed) : null;
 const modelSlowThreshold=requestTimeout ? Math.min(60, Math.max(30, requestTimeout*0.33)) : 45;
 const modelNearTimeout=modelWait && requestRemaining !== null && requestRemaining <= Math.min(45, requestTimeout*0.25);
 const modelSlow=modelWait && requestElapsed !== null && requestElapsed >= modelSlowThreshold;
 let eta=null;
 if(active && total && done > 0 && elapsed > 0) eta=((total-done) / (done / elapsed));
 else if(state.running && total && activeReqs.length && requestTimeout){ const remaining=Math.max(0,total-done-activeReqs.length); const workers=Math.max(1, Number(p.concurrency||activeReqs.length||1)); eta=requestRemaining + (remaining * requestTimeout / workers); }
 const latestError=latestRuntimeError(); const delayed=active && waitSince !== null && waitSince > 90; const stalled=active && waitSince !== null && waitSince > 300;
 const title=latestError ? (active ? (isLoad ? 'Loading with errors' : 'Running with errors') : 'Attention') : modelNearTimeout ? 'Model request near timeout' : modelSlow ? 'Model response slow' : modelWait ? 'Waiting for model' : active ? (isLoad ? (isPaper ? 'Importing source' : (stalled ? 'Dataset load stalled' : delayed ? 'Dataset load slow' : 'Loading dataset')) : (stalled ? 'Possibly stalled' : delayed ? 'Running slowly' : 'Running')) : total ? (errors ? 'Complete with errors' : (isLoad ? 'Dataset loaded' : 'Complete')) : 'Idle';
 const bar=el('sideBar'); bar.style.width=pct+'%'; bar.style.background=latestError||stalled||modelNearTimeout ? 'var(--bad)' : delayed||errors||modelSlow ? 'var(--warn)' : 'var(--accent)'; el('statusTitle').textContent=title; el('statusPct').textContent=`${pct}%`;
 const etaText=active ? (eta === null ? (done ? 'calculating' : (state.running ? 'waiting for model' : 'waiting for first item')) : span(eta)) : '';
 const activeNames=activePool.map(r=>r.name).filter(Boolean).slice(0,3).join(', ');
 const inputShape=oldestActive ? [oldestActive.frames ? `${oldestActive.frames} frames` : '', oldestActive.visual_tokens ? `${fmt(oldestActive.visual_tokens)} visual tokens` : ''].filter(Boolean).join(' / ') : '';
 const rows=[['task', isPaper ? 'paper import' : (isLoad ? 'dataset load' : (state.running ? 'inference batch' : ''))],['phase', p.phase||''],['completed', total ? `${fmt(done)}/${fmt(total)}` : (done ? fmt(done) : 'pending')],['progress', `${pct}%`],['elapsed', elapsed ? span(elapsed) : '0s'],['remaining', etaText],['active', activeReqs.length ? `${activeReqs.length} request${activeReqs.length===1?'':'s'}` : ''],['model wait', oldestActive ? `${oldestActive.stage||''} ${activeNames ? 'for '+activeNames : ''}` : ''],['request elapsed', requestElapsed !== null ? span(requestElapsed) : ''],['timeout left', requestRemaining !== null ? span(requestRemaining) : ''],['input', inputShape],['backend', oldestActive?.backend||''],['model', oldestActive?.model||''],['prompt source', oldestActive?.prompt_source||''],['quiet for', active && waitSince !== null ? span(waitSince) : ''],['current', p.current_file||''],['errors', fmt(errors)],['event', p.last_event||'']];
 el('runtimeKv').innerHTML=rows.filter(([_,v])=>v!==''&&v!==null&&v!==undefined).map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v)}</div>`).join('');
 const notice=el('runtimeNotice');
 if(latestError){ const timedOut=/read timed out|timeout/i.test(latestError); notice.className='status-note bad'; notice.textContent=timedOut ? 'The model backend did not return before the HTTP read timeout. Try Fit to model, reduce fps/max pixels/max frames or concurrency, then rerun. Raw issue: '+clipText(latestError) : 'Latest runtime issue: '+clipText(latestError); }
 else if(modelWait && oldestActive && modelNearTimeout){ notice.className='status-note bad'; notice.textContent=`The model request for ${oldestActive.name||'a video'} is close to the HTTP read timeout. It has been open ${span(requestElapsed)}${requestRemaining!==null ? `; timeout in ${span(requestRemaining)}` : ''}. The backend may be overloaded or spending too long in visual prefill/generation.`; }
 else if(modelWait && oldestActive && modelSlow){ notice.className='status-note warn'; notice.textContent=`The model request for ${oldestActive.name||'a video'} is taking longer than expected. The runtime is still waiting on localhost:8000, not silently failing. Request open ${span(requestElapsed)}${requestRemaining!==null ? `; timeout in ${span(requestRemaining)}` : ''}.`; }
 else if(modelWait && oldestActive){ notice.className='status-note'; notice.textContent=`Waiting for the terminal/model server response for ${oldestActive.name||'a video'}. Request open ${span(requestElapsed)}${requestRemaining!==null ? `; timeout in ${span(requestRemaining)}` : ''}.`; }
 else if(stalled){ notice.className='status-note bad'; notice.textContent=isLoad ? `No dataset-load update for ${span(waitSince)}. Hugging Face may still be transferring a large file, but this is unusually quiet.` : `No video has completed for ${span(waitSince)}. The backend may still be in long prefill/generation, but this is now unusually quiet.`; }
 else if(delayed){ notice.className='status-note warn'; notice.textContent=isLoad ? `No dataset-load update for ${span(waitSince)}. Still waiting for Hugging Face or local metadata extraction.` : `No video has completed for ${span(waitSince)}. Still waiting for the VLM backend to return a result.`; }
 else if(isPaper && active){ notice.className='status-note'; notice.textContent='Source import is active. The runtime is fetching prompt examples and dataset links; dataset loading starts only when import + load is selected and a dataset is found.'; }
 else if(isLoad && active && total){ notice.className='status-note'; notice.textContent='Dataset retrieval is active. Rows and thumbnails will appear as each video finishes downloading and metadata extraction completes.'; }
 else if(isLoad && active){ notice.className='status-note'; notice.textContent='Dataset retrieval is active. Waiting for Hugging Face file listing or the first selected video.'; }
 else if(state.running && done === 0){ notice.className='status-note'; notice.textContent='Batch accepted; preparing videos and waiting for the first model response.'; }
 else if(state.running){ notice.className='status-note'; notice.textContent='Batch is making progress.'; }
 else if(total){ notice.className=errors ? 'status-note warn' : 'status-note'; notice.textContent=errors ? 'Finished with errors. See Results and Runtime log for details.' : (isLoad ? 'Dataset is ready for selection and inference.' : 'Batch finished successfully.'); }
 else { notice.className='status-note'; notice.textContent='No batch is running.'; }
}
function applySliderMeta(){ const map=[['fpsSlider','fps'],['maxPixelsSlider','max_pixels'],['maxTokensSlider','max_tokens'],['temperatureSlider','temperature'],['topPSlider','top_p'],['repPenaltySlider','repetition_penalty'],['maxFramesSlider','max_frames']]; for(const [id,key] of map){ const m=(state.defaults.slider_meta||{})[key]||{}; const el=document.getElementById(id); if(m.min !== undefined) el.min=m.min; if(m.max !== undefined) el.max=m.max; if(m.step !== undefined) el.step=m.step; } }
function applyBuildDefaults(checked){ const d=state.defaults; const b=d.build_defaults||{}; const m=d.slider_meta||{}; el('temperatureSlider').value = checked ? b.temperature : m.temperature.recommended; el('topPSlider').value = checked ? b.top_p : m.top_p.recommended; el('repPenaltySlider').value = checked ? b.repetition_penalty : m.repetition_penalty.recommended; render(); }
function initControls(){ if(initialized || !state) return; const d=state.defaults; applySliderMeta(); document.getElementById('repo').value = state.dataset_repo; document.getElementById('maxVideos').value = d.max_videos; document.getElementById('concurrency').value = d.concurrency; document.getElementById('systemPrompt').value = d.system_prompt; document.getElementById('userPrompt').value = d.user_prompt; autosizePrompts(); document.getElementById('fpsSlider').value = d.fps; document.getElementById('maxPixelsSlider').value = d.max_pixels; document.getElementById('maxTokensSlider').value = d.max_tokens; document.getElementById('temperatureSlider').value = d.temperature; document.getElementById('topPSlider').value = d.top_p; document.getElementById('repPenaltySlider').value = d.repetition_penalty; document.getElementById('maxFramesSlider').value = d.max_frames; document.getElementById('buildDefaultsToggle').checked = !!d.use_build_defaults; syncPromptPresets(); renderExportSections(); initialized = true; if(d.use_build_defaults) applyBuildDefaults(true); }
function render(){ if(!state) return; initControls(); syncReasoningFormatLabel(); renderParamLabels();
 syncPromptPresets(); renderPaperImport();
 const srv = state.server || {}; document.getElementById('serverPill').textContent = srv.error ? 'backend unavailable' : (srv.model ? 'model: '+srv.model : 'backend ready'); const rows=[['instance', srv.instance],['host_ip', srv.host_ip],['backend', srv.backend],['model', srv.model],['model context', srv.model_max_len ? `${fmt(srv.model_max_len)} tokens (${srv.model_max_len_source||'default'})` : ''],['base_url', srv.base_url],['gpu', srv.gpu],['vram', srv.vram_total_mib ? `${fmt(srv.vram_free_mib)} MiB free / ${fmt(srv.vram_total_mib)} MiB total` : srv.gpu_error],['ssd', srv.ssd_total_gb ? `${fmt(srv.ssd_free_gb,1)} GB free / ${fmt(srv.ssd_total_gb,1)} GB total (${srv.storage_path})` : srv.storage_error]]; document.getElementById('serverKv').innerHTML = rows.map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v||'')}</div>`).join('');
 document.getElementById('framePolicy').textContent = nativeVideoMode() ? 'This backend receives a video_url; frame count and visual tokens are sampled by the model server.' : 'This backend receives sampled image frames; the batch inference UI controls fps, max pixels, and max input frames.';
 const prog = state.progress || {done:0,total:0,errors:0}; const active=!!(state.running||state.loading_dataset); const activeReqs=activeRequestList(); const pct = prog.total ? Math.round(100*prog.done/prog.total) : (active ? (prog.mode==='paper_import' ? 8 : 5) : 0); document.getElementById('bar').style.width = pct+'%'; const modelReqs=activeReqs.filter(r=>['model_wait','model_retry'].includes(String(r.stage||''))); const oldestModelReq=modelReqs.length ? modelReqs.reduce((a,b)=>Number(b.started_epoch||serverNow())<Number(a.started_epoch||serverNow())?b:a, modelReqs[0]) : null; const modelElapsed=oldestModelReq ? serverNow()-Number(oldestModelReq.started_epoch||serverNow()) : null; const timeoutLeft=oldestModelReq?.timeout_seconds ? Math.max(0, Number(oldestModelReq.timeout_seconds)-modelElapsed) : null; const activeName=oldestModelReq?.name ? `: ${oldestModelReq.name}` : (activeReqs[0]?.name ? `: ${activeReqs[0].name}` : ''); const modelWaitPhrase=oldestModelReq && modelElapsed>45 ? `model response taking longer than expected${activeName} (open ${span(modelElapsed)}${timeoutLeft!==null ? `, timeout in ${span(timeoutLeft)}` : ''})` : `${activeReqs.length} waiting on model${activeName}`; document.getElementById('progressText').textContent = state.loading_dataset ? `${prog.last_event||'Loading dataset'}${prog.total ? ` (${prog.done}/${prog.total})` : ''}` : state.running && activeReqs.length ? `${prog.done||0}/${prog.total||0} complete, ${modelWaitPhrase}, ${prog.errors||0} errors` : state.running ? `${prog.done}/${prog.total} running, ${prog.errors} errors` : `${prog.done}/${prog.total} complete, ${prog.errors} errors`; renderRuntimeStatus();
 const bm=state.batch_metrics||{}; const ev=bm.evaluation||{}; const domainText=ev.by_capability_domain ? Object.entries(ev.by_capability_domain).map(([k,v])=>`${k}: ${v.correct}/${v.total}${v.average_score!==null&&v.average_score!==undefined ? ' score '+percent(v.average_score) : ''}`).join(' | ') : ''; const bmRows=bm.total ? [['dataset',bm.dataset_repo],['prompt',`${bm.run_label||''} (${bm.prompt_hash||''})`],['status',bm.status],['completed',`${bm.completed}/${bm.total} (${bm.errors} errors)`],['accuracy',ev.evaluated ? `${ev.correct}/${ev.evaluated} (${percent(ev.accuracy)})` : 'no expected labels'],['avg answer score',ev.answer_score_average!==null&&ev.answer_score_average!==undefined ? percent(ev.answer_score_average) : 'n/a'],['by capability',domainText],['batch E2E',sec(bm.batch_wall_seconds)],['video requests/sec',rate(bm.video_requests_per_second)],['video E2E stats',statText(bm.e2e_seconds)],['TTFT stats',statText(bm.ttft_seconds)],['output tok/s stats',statText(bm.output_tokens_per_second)]] : [['batch','No batch has run yet']]; document.getElementById('batchKv').innerHTML = bmRows.map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v||'')}</div>`).join('');
 const selected = new Set(checkedIds()); document.getElementById('videoRows').innerHTML = (state.videos||[]).map(v=>{ const m=v.meta||{}; const plan=videoPlan(v); const thumb=v.thumbnail_url ? `<img class="thumb" src="${esc(v.thumbnail_url)}" alt="">` : ''; const row=v.dataset_row||{}; return `<tr><td><input class="pick" type="checkbox" value="${esc(v.id)}" ${selected.size===0||selected.has(v.id)?'checked':''}></td><td>${thumb}</td><td>${esc(v.name)}</td><td>${expectedText(v.expected)}</td><td>${esc(rowSummary(row))}${detailsJson('row',row)}</td><td>${fmt(m.width)}x${fmt(m.height)}</td><td>${fmt(m.duration_s,1)}s</td><td>${fmt(m.total_frames)}</td><td>${esc(plan.frames)}</td><td>${esc(plan.tokens)}</td><td>${esc(v.filepath)}</td></tr>`; }).join('');
 contextBlocked = renderContextGuard();
 document.getElementById('resultRows').innerHTML = (state.results||[]).map(r=>{ const j=r.json||{}; const ev=r.evaluation||{}; const expected=ev.has_expected ? clipText(ev.expected_answer||ev.expected_label||'',180) : ''; const pred=clipText(ev.predicted_answer||ev.predicted_label||(j.prediction_label ? `${j.prediction_class_id} ${j.prediction_label}` : r.response||''),180); const match=ev.has_expected ? (ev.is_correct ? 'correct' : 'miss') : ''; const score=ev.answer_score!==null&&ev.answer_score!==undefined ? percent(ev.answer_score) : match; const plan=r.plan||{}; const met=r.metrics||{}; const tokenSuffix=met.output_tokens_estimated ? ' est' : ''; const meta = plan.frames_passed ? `frames=${esc(plan.frames_passed)} tokens=${esc(plan.visual_tokens_est||'server')}` : ''; const desc=r.error ? `<span style="color:var(--bad)">${esc(r.error)}</span>` : esc((meta ? meta+' | ' : '') + (j.video_description||clipText(r.response||'',260))); const domain=ev.capability_domain||ev.domain||''; return `<tr><td>${esc(r.name)}</td><td>${esc(ev.task||'')}</td><td>${esc(domain)}<br>${esc(ev.sft_type||'')}</td><td>${esc(expected)}</td><td>${esc(pred)}</td><td>${esc(ev.metric||'')}</td><td>${esc(score)}</td><td class="metric">${sec(met.ttft_seconds)}</td><td class="metric">${rate(met.output_tokens_per_second)}${tokenSuffix}</td><td class="metric">${sec(met.e2e_seconds)}</td><td>${desc}${detailsJson('output',{json:r.json, evaluation:r.evaluation, usage:r.usage, prompt_source:r.prompt_source, user_prompt:r.user_prompt_used})}</td></tr>`; }).join('');
 document.getElementById('batchRows').innerHTML = (state.batch_history||[]).slice().reverse().map(b=>{ const ev=b.evaluation||{}; return `<tr><td>${esc(b.dataset_repo)}</td><td>${esc(b.run_label||'')}</td><td>${esc(b.status)}</td><td>${esc(b.completed)}/${esc(b.total)}</td><td>${esc(b.errors)}</td><td>${ev.evaluated ? `${esc(ev.correct)}/${esc(ev.evaluated)} (${percent(ev.accuracy)})` : ''}</td><td class="metric">${sec(b.batch_wall_seconds)}</td><td class="metric">${rate(b.video_requests_per_second)}</td><td class="metric">${sec(b.e2e_seconds?.median)}</td><td>${esc(b.prompt_hash||'')}</td></tr>`; }).join('');
 document.getElementById('log').textContent = (state.logs||[]).join('\\n');
 const busy=!!(state.running||state.loading_dataset); document.getElementById('loadBtn').disabled = busy; document.getElementById('paperBtn').disabled = busy; document.getElementById('promptImportBtn').disabled = busy; document.getElementById('runBtn').disabled = busy || contextBlocked; document.getElementById('smokeBtn').disabled = busy || contextBlocked; document.getElementById('foBtn').disabled = busy; document.getElementById('reasoningToggle').disabled = busy; requestAnimationFrame(autosizePrompts); }
async function poll(){ const r = await fetch('/api/state'); state = await r.json(); state._receivedAt = Date.now()/1000; render(); }
document.getElementById('promptPreset').onchange = applySelectedPrompt;
document.getElementById('savePromptBtn').onclick = async()=>{ try{ await saveCurrentPrompt(); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('reloadPromptBtn').onclick = applySelectedPrompt;
['systemPrompt','userPrompt'].forEach(id=>document.getElementById(id).oninput=()=>{ autosizeTextarea(el(id)); markPromptDirty(); });
document.getElementById('reasoningToggle').onchange = applyReasoningToPrompt;
['fpsSlider','maxPixelsSlider','maxTokensSlider','temperatureSlider','topPSlider','repPenaltySlider','maxFramesSlider'].forEach(id=>document.getElementById(id).oninput=render);
document.getElementById('buildDefaultsToggle').onchange = ()=>applyBuildDefaults(el('buildDefaultsToggle').checked);
document.getElementById('allowOverContext').onchange = render;
document.getElementById('fitBudgetBtn').onclick = fitToModel;
document.getElementById('videoRows').addEventListener('change', e=>{ if(e.target.classList.contains('pick')) render(); });
document.getElementById('promptImportBtn').onclick = async()=>{ try{ if(!promptDirty) promptUserSelected=false; setBusy('Importing prompt examples...'); await api('/api/prompts/import',{source:el('paperSource').value}); await poll(); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('paperBtn').onclick = async()=>{ try{ if(!promptDirty) promptUserSelected=false; setBusy('Importing paper metadata and prompts...'); const j=await api('/api/paper',{source:el('paperSource').value,max_videos:Number(el('maxVideos').value),load_dataset:true}); if(j.selected_dataset) el('repo').value=j.selected_dataset; const p=(j.prompt_presets||[])[0]; if(p && !promptDirty){ setPromptFields({...p,key:promptKey(p,'imported')}); } await poll(); if(j.status==='importing') return; if(j.load_error) alert('Imported prompts, but dataset load failed: '+j.load_error); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('loadBtn').onclick = async()=>{ try{ setBusy('Loading dataset from Hugging Face...'); await api('/api/load',{repo_id:el('repo').value,max_videos:Number(el('maxVideos').value)}); await poll(); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('runBtn').onclick = async()=>{ if(contextBlocked){ alert('Current OSS image-frame settings are over the estimated model context. Use Fit to model or enable Allow over-budget OSS run.'); return; } try{ setBusy('Starting selected-video batch...'); await api('/api/run',{ids:checkedIds(),concurrency:Number(el('concurrency').value),prompt_label:promptLabel(),prompt_mode:promptMode(),system_prompt:el('systemPrompt').value,user_prompt:el('userPrompt').value,allow_over_context:el('allowOverContext').checked,...params(),...reasoningPayload()}); await poll(); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('smokeBtn').onclick = async()=>{ if(contextBlocked){ alert('Current OSS image-frame settings are over the estimated model context. Use Fit to model or enable Allow over-budget OSS run.'); return; } try{ setBusy('Loading smoke dataset and starting batch...'); await api('/api/smoke',{max_videos:Number(el('maxVideos').value),concurrency:Number(el('concurrency').value),prompt_mode:'runtime_form',allow_over_context:el('allowOverContext').checked,...params(),...reasoningPayload()}); await poll(); }catch(e){ alert(e.message); await poll(); } };
document.getElementById('foBtn').onclick = async()=>{ try{ setBusy('Opening FiftyOne app...'); const j=await api('/api/fiftyone',{}); await poll(); alert('FiftyOne: '+j.url); }catch(e){ alert(e.message); await poll(); } };
async function exportArtifact(format){ const buttons=[...document.querySelectorAll('.exportBtn')]; let status=null; try{ buttons.forEach(b=>b.disabled=true); status=document.createElement('span'); status.className='export-link'; status.textContent=`Creating ${format.toUpperCase()}...`; el('exportLinks').prepend(status); const j=await api('/api/export',{format,sections:checkedExportSections()}); status.remove(); status=null; addExportLink(j); await poll(); }catch(e){ if(status) status.remove(); alert(e.message); await poll(); } finally{ buttons.forEach(b=>b.disabled=false); } }
document.querySelectorAll('.exportBtn').forEach(btn=>{ btn.onclick = ()=>exportArtifact(btn.dataset.format); });
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
        elif self.path.startswith("/api/export/"):
            filename = Path(urllib.parse.urlparse(self.path).path).name
            try:
                export_path = resolve_export_download(filename)
            except FileNotFoundError:
                self.send_error(404)
                return
            content_type = EXPORT_CONTENT_TYPES.get(export_path.suffix.lstrip(".").lower(), mimetypes.guess_type(str(export_path))[0] or "application/octet-stream")
            self.send_file(export_path, content_type, download_name=export_path.name)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self.read_json()
            if self.path == "/api/load":
                snap = snapshot()
                if snap.get("running") or snap.get("loading_dataset"):
                    raise RuntimeError("A batch or dataset load is already running")
                repo_id = str(payload.get("repo_id") or DEFAULT_DATASET)
                max_videos = int_payload(payload, "max_videos", 20)
                thread = threading.Thread(target=load_dataset_worker, args=(repo_id, max_videos), daemon=True)
                thread.start()
                self.send_json({"ok": True, "status": "loading", "repo_id": repo_id, "max_videos": max_videos})
            elif self.path == "/api/paper":
                load_now = bool(payload.get("load_dataset", True))
                source = str(payload.get("source") or "")
                max_videos = int_payload(payload, "max_videos", 20)
                if load_now:
                    snap = snapshot()
                    if snap.get("running") or snap.get("loading_dataset"):
                        raise RuntimeError("A batch or dataset load is already running")
                    thread = threading.Thread(target=import_paper_worker, args=(source, max_videos, load_now), daemon=True)
                    thread.start()
                    self.send_json({"ok": True, "status": "importing", "source": source, "max_videos": max_videos})
                else:
                    discovery = import_paper_source(source, max_videos, load_now)
                    self.send_json(discovery)
            elif self.path == "/api/prompts/import":
                snap = snapshot()
                if snap.get("running") or snap.get("loading_dataset"):
                    raise RuntimeError("A batch, dataset load, or prompt import is already running")
                source = str(payload.get("source") or "")
                thread = threading.Thread(target=import_paper_worker, args=(source, 0, False), daemon=True)
                thread.start()
                self.send_json({"ok": True, "status": "importing", "source": source})
            elif self.path == "/api/prompt":
                saved = add_prompt_preset(
                    str(payload.get("label") or ""),
                    str(payload.get("system_prompt") or ""),
                    str(payload.get("user_prompt") or ""),
                    str(payload.get("source") or "runtime"),
                )
                self.send_json({"ok": True, **saved})
            elif self.path == "/api/run":
                snap = snapshot()
                if snap.get("running") or snap.get("loading_dataset"):
                    raise RuntimeError("A batch or dataset load is already running")
                ids = payload.get("ids") or []
                system_prompt = str(payload.get("system_prompt") or WORKER_SAFETY_SYSTEM)
                user_prompt = str(payload.get("user_prompt") or WORKER_SAFETY_USER)
                params = params_from_payload(payload)
                context_report = validate_context_budget(ids, params, system_prompt, user_prompt, bool(payload.get("allow_over_context")))
                thread = threading.Thread(
                    target=run_batch,
                    args=(
                        ids,
                        int(payload.get("concurrency") or 4),
                        system_prompt,
                        user_prompt,
                        params,
                        str(payload.get("prompt_label") or "Custom prompt"),
                    ),
                    daemon=True,
                )
                thread.start()
                self.send_json({"ok": True, "context_budget": context_report})
            elif self.path == "/api/smoke":
                snap = snapshot()
                if snap.get("running") or snap.get("loading_dataset"):
                    raise RuntimeError("A batch or dataset load is already running")
                max_videos = int_payload(payload, "max_videos", 2)
                concurrency = int_payload(payload, "concurrency", 2)
                load_dataset(DEFAULT_DATASET, max_videos)
                ids = [v["id"] for v in snapshot()["videos"]]
                params = params_from_payload(payload)
                context_report = validate_context_budget(ids, params, WORKER_SAFETY_SYSTEM, WORKER_SAFETY_USER, bool(payload.get("allow_over_context")))
                thread = threading.Thread(target=run_batch, args=(ids, concurrency, WORKER_SAFETY_SYSTEM, WORKER_SAFETY_USER, params, "Worker safety smoke"), daemon=True)
                thread.start()
                self.send_json({"ok": True, "dataset": DEFAULT_DATASET, "videos": len(ids), "context_budget": context_report})
            elif self.path == "/api/fiftyone":
                url = launch_fiftyone(int(payload.get("port") or os.getenv("FIFTYONE_PORT", "5151")))
                self.send_json({"url": url})
            elif self.path == "/api/export":
                fmt = str(payload.get("format") or "html")
                path = create_export(fmt, payload.get("sections") or [])
                exported_fmt = path.suffix.lstrip(".").lower()
                self.send_json({
                    "ok": True,
                    "format": exported_fmt,
                    "filename": path.name,
                    "url": "/api/export/" + urllib.parse.quote(path.name),
                    "bytes": path.stat().st_size,
                })
            else:
                self.send_error(404)
        except Exception as exc:
            log(f"API error: {exc}")
            with STATE_LOCK:
                progress = dict(STATE.get("progress") or {})
                progress.update({
                    "updated_epoch": time.time(),
                    "last_event": "API error",
                    "last_error": str(exc),
                })
                STATE["progress"] = progress
            self.send_json({"error": str(exc)}, status=400 if isinstance(exc, ClientInputError) else 500)

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

    def send_file(self, path: Path, content_type: str, download_name: Optional[str] = None) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
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
        "prompt_mode": str(payload.get("prompt_mode") or "auto"),
        "reasoning_enabled": bool_param(payload.get("reasoning_enabled")),
        "reasoning_format": str(payload.get("reasoning_format") or "auto"),
        "reasoning_model": str(payload.get("reasoning_model") or ""),
        "reasoning_backend": str(payload.get("reasoning_backend") or ""),
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
