#!/usr/bin/env python3
"""Batch-inference frontend for Cosmos BYO-video inference.

This is intentionally self-contained: it serves a small browser UI, loads video
samples from a public Hugging Face dataset through FiftyOne when available, and
runs selected videos concurrently against the OpenAI-compatible vLLM/NIM server
that the BYO-video setup script starts.

Two views are exposed in the UI:
- Basic View: per-video inference against a Hugging Face dataset with prompt
  presets, parameter sliders, and FiftyOne integration (preserved from the
  original tool).
- Benchmark View: 1000-row LingoQA Q/A benchmark against a reasoning-capable
  NIM (e.g. Cosmos3-Super-Reasoner / cosmos-reason2-*), scored by the official
  wayveai/Lingo-Judge (DeBERTa-v3-base) and rendered in an arxiv-2312.14115
  style report. Implements the NIM Message-Shape standing order verbatim:
  base64 data: URL image_url[] content array, auto-detected served model name,
  no max_tokens / max_frames / max_pixels client-side caps in the benchmark
  path. Lingo-Judge runs in-process; weights are pulled lazily on first use.
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
TEXT_SCORE_THRESHOLD = float(os.getenv("BATCH_INFERENCE_TEXT_SCORE_THRESHOLD", "0.75"))
DEFAULT_DATASET = os.getenv("BATCH_INFERENCE_DATASET", "pjramg/Safe_Unsafe_Test")
RESULTS_FILE = Path(os.getenv("BATCH_INFERENCE_RESULTS", "/tmp/byo_video_batch_inference_results.json"))
THUMBNAIL_DIR = Path(os.getenv("BATCH_INFERENCE_THUMBNAILS", "/tmp/byo_video_batch_inference_thumbnails"))
THUMBNAIL_SIZE = (220, 124)
REQUEST_TIMEOUT_SECONDS = float(os.getenv("BATCH_INFERENCE_REQUEST_TIMEOUT_SECONDS", "180"))
EXPORT_DIR = Path(os.getenv("BATCH_INFERENCE_EXPORTS", "/tmp/byo_video_batch_inference_exports"))
CONTEXT_PATCH_PIXELS = int(os.getenv("BATCH_INFERENCE_CONTEXT_PATCH_PIXELS", str(14 * 14)))
CONTEXT_SAFETY_RESERVE = int(os.getenv("BATCH_INFERENCE_CONTEXT_SAFETY_RESERVE", "1024"))
DEFAULT_MODEL_MAX_LEN = int(os.getenv("BATCH_INFERENCE_MODEL_MAX_LEN", "32768"))
MODEL_FIT_VISUAL_TOKENS = int(os.getenv("BATCH_INFERENCE_MODEL_FIT_VISUAL_TOKENS", "6144"))
MODEL_FIT_TARGET_FRAMES = int(os.getenv("BATCH_INFERENCE_MODEL_FIT_TARGET_FRAMES", "4"))
TEXT_TOKENS = 50
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "vllm").lower()
MAX_PIXELS_MIN = 64 * (32 ** 2)
MAX_PIXELS_MAX = 4096 * (32 ** 2)
RECOMMENDED_MAX_PIXELS = int(os.getenv("BATCH_INFERENCE_RECOMMENDED_MAX_PIXELS", os.getenv("GRADIO_MAX_PIXELS", str(512 * (32 ** 2)))))
DEFAULT_MAX_PIXELS = int(os.getenv("BATCH_INFERENCE_MAX_PIXELS", str(RECOMMENDED_MAX_PIXELS)))
DEFAULT_FPS = float(os.getenv("BATCH_INFERENCE_FPS", os.getenv("GRADIO_FPS", "2")))
DEFAULT_MAX_TOKENS = int(os.getenv("BATCH_INFERENCE_MAX_TOKENS", os.getenv("GRADIO_MAX_TOKENS", "512")))
DEFAULT_MAX_FRAMES = int(os.getenv("BATCH_INFERENCE_MAX_FRAMES", "8"))
if INFERENCE_BACKEND == "nim_local":
    RECOMMENDED_TEMPERATURE = 0.6
    RECOMMENDED_TOP_P = 0.3
    RECOMMENDED_REPETITION_PENALTY = 1.2
else:
    RECOMMENDED_TEMPERATURE = 0.0
    RECOMMENDED_TOP_P = 1.0
    RECOMMENDED_REPETITION_PENALTY = 1.05
DEFAULT_TEMPERATURE = float(os.getenv("BATCH_INFERENCE_TEMPERATURE", str(RECOMMENDED_TEMPERATURE)))
DEFAULT_TOP_P = float(os.getenv("BATCH_INFERENCE_TOP_P", str(RECOMMENDED_TOP_P)))
DEFAULT_REPETITION_PENALTY = float(os.getenv("BATCH_INFERENCE_REPETITION_PENALTY", str(RECOMMENDED_REPETITION_PENALTY)))
HTTP_TIMEOUT_SECONDS = float(os.getenv("BATCH_INFERENCE_HTTP_TIMEOUT_SECONDS", "20"))
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
PROMPT_SCAN_DIR_LIMIT = int(os.getenv("BATCH_INFERENCE_PROMPT_SCAN_DIR_LIMIT", "80"))
PROMPT_SCAN_MAX_TEXT_BYTES = int(os.getenv("BATCH_INFERENCE_PROMPT_SCAN_MAX_TEXT_BYTES", str(2 * 1024 * 1024)))
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
        "concurrency": int(os.getenv("BATCH_INFERENCE_CONCURRENCY", "4")),
        "max_videos": int(os.getenv("BATCH_INFERENCE_MAX_VIDEOS", "20")),
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
    print("[batch-inference]", message, flush=True)


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
    headers = {"User-Agent": "cosmos-byo-video-batch-inference"}
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

    prefix_bytes = int(os.getenv("BATCH_INFERENCE_PROMPT_SCAN_ANNOTATION_PREFIX_BYTES", str(12 * 1024 * 1024)))
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

    max_bytes = int(os.getenv("BATCH_INFERENCE_PROMPT_SCAN_MAX_BYTES", str(5 * 1024 * 1024)))
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
        if key.startswith("batch_inference_") or key in {"frames"}:
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
    configured_backend = os.getenv("INFERENCE_BACKEND", "vllm").lower()
    is_cosmos3 = configured_backend == "cosmos3_native" or configured_backend.startswith("cosmos3")
    if is_cosmos3:
        # Cosmos3 Ray Serve does not expose /v1/models; it serves /generate
        # at the root. Do not append /v1 and probe the active-model endpoint
        # for a friendlier display name.
        default_base = (
            os.getenv("COSMOS3_BASE_URL")
            or os.getenv("RAY_SERVE_BASE_URL")
            or "http://localhost:8000"
        )
        default_model = os.getenv("MODEL_NAME") or "Cosmos3-Nano"
    else:
        default_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        default_model = os.getenv("MODEL_NAME", "")
    info = {
        "base_url": default_base,
        "model": default_model,
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
    if is_cosmos3:
        # Best-effort probe of the Cosmos3 active-model sidecar; fall back to
        # the env-provided MODEL_NAME if it isn't reachable.
        try:
            probe = requests.get("http://localhost:8088/active-model", timeout=2)
            if probe.status_code < 400:
                probe_json = probe.json() or {}
                checkpoint = probe_json.get("checkpoint") or probe_json.get("model")
                if checkpoint:
                    info["model"] = str(checkpoint)
                info["model_info"] = json_safe(probe_json, depth=2, max_text=300)
        except Exception as exc:
            info["model_probe_error"] = str(exc)
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
    return (
        "nim" in backend_lower
        or ("cosmos-reason" in lower and "nim" in lower)
        or "cosmos3" in lower
        or "cosmos-3" in lower
    )


def model_uses_native_video(model: str, backend: str = "") -> bool:
    backend_lower = (backend or os.getenv("INFERENCE_BACKEND", "")).lower()
    return "nim" in backend_lower or model_prefers_file_url(model) or model_prefers_video_data(model, backend)


def nim_frame_fallback_limit() -> int:
    raw = os.getenv("REASONER_FRAME_FALLBACK_MAX_IMAGES") or os.getenv("NIM_MAX_IMAGES_PER_PROMPT") or "5"
    try:
        value = int(float(raw))
    except Exception:
        value = 5
    return max(1, value)


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
    effective_max_frames = int(max_frames)
    if force_frames and "nim" in (backend or os.getenv("INFERENCE_BACKEND", "")).lower():
        fallback_limit = nim_frame_fallback_limit()
        if effective_max_frames <= 0 or effective_max_frames > fallback_limit:
            effective_max_frames = fallback_limit
    plan = estimate_plan(meta, fps, max_pixels, effective_max_frames, model if not force_frames else "", backend if not force_frames else "")
    if not force_frames and model_uses_native_video(model, backend):
        mime = mimetypes.guess_type(video_path)[0] or "video/mp4"
        data = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")
        return [
            {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}},
            {"type": "text", "text": prompt},
        ], plan
    frames = extract_frames_b64(video_path, fps=fps, max_frames=effective_max_frames, max_pixels=max_pixels)
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


def post_cosmos3_generate(
    base_url: str,
    headers: Dict[str, str],
    vision_path: str,
    prompt: str,
    params: Dict[str, Any],
    run_name: str,
) -> Dict[str, Any]:
    """POST to Cosmos3 Ray Serve /generate (diffusion path).

    Returns a dict shaped like post_chat_completion's return so downstream
    code in run_one() does not have to special-case the diffusion branch.
    """
    url = base_url.rstrip("/") + "/generate"
    body = {
        "name": run_name,
        "model": "",
        "prompt": prompt or "",
        "negative_prompt": str(params.get("negative_prompt") or ""),
        "vision_path": str(vision_path),
        "num_frames": int(params.get("num_frames") if params.get("num_frames") is not None else 17),
        # CRITICAL: resolution must be a string literal ('256'|'480'|'720'|'1080').
        # Integer 256 -> 422. Always coerce via str().
        "resolution": str(params.get("resolution") if params.get("resolution") is not None else "256"),
        "aspect_ratio": str(params.get("aspect_ratio") or "1,1"),
        "fps": int(params.get("fps") if params.get("fps") is not None else 12),
        "num_steps": int(params.get("num_steps") if params.get("num_steps") is not None else 20),
        "guidance": float(params.get("guidance") if params.get("guidance") is not None else 6.0),
        "seed": int(params.get("seed") if params.get("seed") is not None else 42),
    }
    request_started = time.monotonic()
    resp = requests.post(url, headers=headers, json=body, timeout=600)
    raise_for_status_with_body(resp)
    elapsed_seconds = time.monotonic() - request_started
    try:
        response_json = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Cosmos3 /generate returned non-JSON: {exc}: {resp.text[:400]}") from exc

    status = str(response_json.get("status") or "").lower()
    if status != "success":
        message = response_json.get("message") or "unknown error"
        stack_trace = response_json.get("stack_trace") or ""
        detail = f"Cosmos3 /generate status={status}: {message}"
        if stack_trace:
            detail += f"\n{stack_trace}"
        raise RuntimeError(detail)

    outputs = response_json.get("outputs") or []
    source_files: List[str] = []
    for out in outputs:
        for fp in (out.get("files") or []):
            if fp:
                source_files.append(str(fp))

    # Copy each file from Ray Serve's output dir into our exports area so
    # downstream UI code can serve them from a stable location.
    copied_files: List[str] = []
    dest_dir = EXPORT_DIR / "runs" / run_name
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    for src in source_files:
        try:
            src_path = Path(src)
            if src_path.is_file():
                dest_path = dest_dir / src_path.name
                shutil.copy2(str(src_path), str(dest_path))
                copied_files.append(str(dest_path))
            else:
                copied_files.append(src)
        except Exception:
            copied_files.append(src)

    files_for_metrics = copied_files or source_files
    text_payload = json.dumps({
        "output_files": files_for_metrics,
        "status": "success",
        "model_mode": "image2video",
    })
    return {
        "text": text_payload,
        "metrics": {
            "ttft_seconds": None,
            "decode_seconds": elapsed_seconds,
            "request_total_seconds": elapsed_seconds,
            "output_tokens": 0,
            "output_tokens_estimated": True,
            "output_tps": 0.0,
            "total_tps": 0.0,
            "files": files_for_metrics,
            "frames": int(body["num_frames"]),
            "resolution": str(body["resolution"]),
        },
    }


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
    if backend == "cosmos3_native" or backend.startswith("cosmos3"):
        # Diffusion path: bypass chat/completions and call Ray Serve /generate
        # directly. Cosmos3 expects an OmniSampleOverrides body keyed on
        # `vision_path` (an absolute path readable by the Ray Serve worker
        # which co-runs on this host).
        cosmos3_base_url = (
            os.getenv("COSMOS3_BASE_URL")
            or os.getenv("RAY_SERVE_BASE_URL")
            or "http://localhost:8000"
        )
        update_active_request(
            video,
            "model_wait",
            f"Cosmos3 diffusion: {video['name']}",
            model=model,
            backend=backend,
            prompt_source=prompt_source,
        )
        run_name = f"req-{int(time.time() * 1000)}-{str(video.get('name', 'x')).replace('/', '_')}"
        completion = post_cosmos3_generate(
            cosmos3_base_url,
            headers,
            vision_path=video["filepath"],
            prompt=effective_user_prompt,
            params=params,
            run_name=run_name,
        )
    else:
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
        if "batch_inference_correct" not in schema:
            dataset.add_sample_field("batch_inference_correct", fo.BooleanField)
        if "batch_inference_error" not in schema:
            dataset.add_sample_field("batch_inference_error", fo.StringField)
        sample = dataset[video["sample_id"]]
        sample["batch_inference_response"] = result.get("response") or ""
        sample["batch_inference_plan"] = result.get("plan") or {}
        sample["batch_inference_params"] = result.get("params") or {}
        sample["batch_inference_metrics"] = result.get("metrics") or {}
        sample["batch_inference_expected"] = result.get("expected") or {}
        sample["batch_inference_evaluation"] = result.get("evaluation") or {}
        if (result.get("evaluation") or {}).get("has_expected"):
            sample["batch_inference_correct"] = bool((result.get("evaluation") or {}).get("is_correct"))
        if result.get("json") is not None:
            sample["batch_inference_json"] = result["json"]
            label = result["json"].get("prediction_label")
            if label:
                sample["batch_inference_prediction"] = fo.Classification(label=str(label))
        if result.get("error"):
            sample["batch_inference_error"] = str(result["error"])
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


# ============================================================================
# Benchmark View — LingoQA evaluation engine
# ============================================================================
#
# Design contract (NIM Message-Shape Standing Order, NO client-side caps):
# - Outbound payload mirrors build.nvidia.com cosmos-reason2-8b verbatim:
#   one user message, content = [image_url, image_url, ..., text]; each
#   image_url.url is a base64 data: URL ("data:image/jpeg;base64,<b64>").
# - Model id is auto-detected from /v1/models on each run start; never
#   hardcoded.
# - NO max_tokens / max_completion_tokens / temperature / top_p sent unless
#   the user explicitly overrides. Server's max_model_len governs.
# - Lingo-Judge runs in-process: AutoModelForSequenceClassification on the
#   wayveai/Lingo-Judge checkpoint (DeBERTa-v3-base, 184M params). Two
#   reference answers per question; final score is max of the two sigmoids.
# - Streaming endpoint emits a 1-byte SSE comment every ~15s to keep long
#   fetches alive (multi-hour 1000-row eval). Partial results are flushed
#   to disk every 10 rows so a crash mid-run is recoverable.

LINGOQA_DATA_ROOTS = [
    Path(p) for p in (
        os.getenv("LINGOQA_DATA_ROOT") or "",
        "/home/horde/lingoqa-data",
        "/tmp/lingoqa-data",
    ) if p
]
BENCHMARK_RUN_ROOT = Path(os.getenv("BENCHMARK_RUN_ROOT", "/tmp/benchmark-runs"))
BENCHMARK_RUN_ROOT.mkdir(parents=True, exist_ok=True)

LINGOQA_CATEGORY_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("counting", ["how many", " count", "number of"]),
    ("action", ["what is the vehicle doing", "what action", "what is the ego", "what is the car doing", "what are you doing"]),
    ("justification", ["why ", "what is the reason", "justify"]),
    ("attention", ["pay attention", "what should you pay", "focus on", "watch out", "be aware"]),
    ("anticipation", ["about to", "going to happen", "next", "anticipate", "predict"]),
    ("reasoning_counterfactuals", ["if ", "would you", "what if", "instead", "counterfactual", "suppose"]),
    ("localisation", ["where ", "which lane", "left", "right", "position of", "located"]),
    ("identification", ["which ", "identify", "what kind", "what type", "what color", "what colour"]),
    ("description", ["describe", "what can you see", "what do you see", "what is visible"]),
]
LINGOQA_CATEGORY_ORDER = [
    "action",
    "justification",
    "attention",
    "identification",
    "localisation",
    "description",
    "counting",
    "anticipation",
    "reasoning_counterfactuals",
]

BENCHMARK_STATE: Dict[str, Any] = {
    "runs": {},  # run_id -> snapshot dict
    "judge_loaded": False,
    "judge_error": None,
    "lingoqa_loaded": False,
    "lingoqa_count": 0,
    # Datasets the user has resolved via /benchmark/resolve_url this session.
    # Keyed by dataset_config['id']. Populated by _resolve_url_to_dataset and
    # surfaced by /benchmark/datasets so the dropdown progressively grows.
    "resolved_datasets": {},
}
BENCHMARK_LOCK = threading.Lock()

# Module-level handles for the lazily-loaded judge.
_JUDGE_TOKENIZER = None
_JUDGE_MODEL = None
_JUDGE_DEVICE = None


def _lingoqa_dataset_dir() -> Optional[Path]:
    for root in LINGOQA_DATA_ROOTS:
        if not root.exists():
            continue
        # Two on-disk layouts are tolerated:
        # 1. <root>/evaluation/val.parquet + <root>/evaluation/images/images/val/...
        # 2. <root>/val.parquet + <root>/images/val/...
        for candidate in (root / "evaluation", root):
            parquet = candidate / "val.parquet"
            if parquet.exists():
                return candidate
    return None


def _lingoqa_image_root(base: Path) -> Path:
    # MANIFEST: parquet paths are relative to "<base>/images/" (note doubled
    # images/ wrapper after unzip).
    doubled = base / "images"
    if doubled.exists():
        return doubled
    return base


def categorize_lingoqa_question(question: str) -> str:
    q = question.lower()
    for cat, needles in LINGOQA_CATEGORY_KEYWORDS:
        for needle in needles:
            if needle in q:
                return cat
    return "uncategorized"


def load_lingoqa(parquet_path: Optional[Path] = None, image_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read LingoQA val.parquet and yield one record per row.

    Returns list of dicts: {question_id, segment_id, images: [absolute paths],
    question, answer, category}. Images-per-row is 5 (uniform); two reference
    answers exist per unique question_id and both rows are returned.
    """
    base = parquet_path.parent if parquet_path else _lingoqa_dataset_dir()
    if base is None:
        raise FileNotFoundError(
            "LingoQA val.parquet not found. Stage data at /home/horde/lingoqa-data "
            "or /tmp/lingoqa-data with the manifest layout."
        )
    parquet = parquet_path or (base / "val.parquet")
    img_root = image_root or _lingoqa_image_root(base)

    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(f"pyarrow required to read LingoQA parquet: {exc}") from exc

    table = pq.read_table(parquet)
    rows = table.to_pylist()
    out: List[Dict[str, Any]] = []
    for row in rows:
        raw_images = row.get("images")
        if hasattr(raw_images, "tolist"):
            raw_images = raw_images.tolist()
        if raw_images is None:
            raw_images = []
        images: List[str] = []
        for rel in raw_images:
            rel_str = str(rel)
            candidate = img_root / rel_str
            if not candidate.exists():
                # Try without leading "images/" (the parquet sometimes prefixes it).
                stripped = rel_str.split("/", 1)[1] if "/" in rel_str else rel_str
                alt = img_root / stripped
                candidate = alt if alt.exists() else candidate
            images.append(str(candidate))
        question = str(row.get("question") or "")
        out.append({
            "question_id": str(row.get("question_id") or ""),
            "segment_id": str(row.get("segment_id") or ""),
            "images": images,
            "question": question,
            "answer": str(row.get("answer") or ""),
            "category": categorize_lingoqa_question(question),
        })
    with BENCHMARK_LOCK:
        BENCHMARK_STATE["lingoqa_loaded"] = True
        BENCHMARK_STATE["lingoqa_count"] = len(out)
    return out


def group_lingoqa_by_question(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse two-reference rows into one record per question_id.

    Returns dicts: {question_id, segment_id, images, question, references: [a,b], category}.
    """
    bucket: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for s in samples:
        qid = s["question_id"]
        if qid not in bucket:
            bucket[qid] = {
                "question_id": qid,
                "segment_id": s["segment_id"],
                "images": s["images"],
                "question": s["question"],
                "references": [],
                "category": s["category"],
            }
            order.append(qid)
        bucket[qid]["references"].append(s["answer"])
    return [bucket[qid] for qid in order]


# =============================================================================
# Dataset dispatcher — any benchmark URL → canonical row shape.
#
# Canonical row (matches group_lingoqa_by_question output):
#   {question_id, segment_id, question, references: [str, ...],
#    images: [absolute_path, ...], category}
#
# Dispatch by dataset_id prefix:
#   "lingoqa-official"   → existing on-disk loader (KEPT WORKING)
#   "hf:owner/name"      → HuggingFace `datasets` library
#   "hf:owner/name@split"→ same, with split override
#   "arxiv:NNNN.NNNNN"   → discover paper's HF dataset, then dispatch hf:
#   "github:owner/repo"  → walk repo for parquet/json data + image dir
#   "gdrive:FOLDER_ID"   → gdown the folder, then look for parquet + images
#   "local:/abs/path"    → load a local parquet (advanced)
# =============================================================================

_HF_DATASETS_CACHE_ROOT = Path("/tmp/hf-datasets")


def _safe_dataset_dir_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s or "ds"))


def _normalize_hf_row(
    row: Dict[str, Any],
    row_idx: int,
    dataset_safe: str,
    image_cache_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Best-effort normalization of an HF dataset row to the canonical shape.

    Returns None if the row lacks a question OR references — we can't judge
    without ground truth. Image-less rows are allowed (some QA datasets are
    text-only); run_one_image_qa will simply send a text-only payload.
    """
    q = (
        row.get("question")
        or row.get("prompt")
        or row.get("query")
        or row.get("instruction")
        or ""
    )
    if isinstance(q, (list, tuple)) and q:
        q = q[0]
    q = str(q).strip()
    if not q:
        return None

    refs_raw = (
        row.get("answer")
        or row.get("answers")
        or row.get("references")
        or row.get("response")
        or row.get("responses")
        or row.get("output")
    )
    if isinstance(refs_raw, str):
        refs = [refs_raw]
    elif isinstance(refs_raw, (list, tuple)):
        refs = [str(r) for r in refs_raw if r is not None and str(r).strip()]
    elif refs_raw is None:
        refs = []
    else:
        refs = [str(refs_raw)]
    refs = [r for r in refs if r.strip()]
    if not refs:
        return None

    images_raw = (
        row.get("images")
        or row.get("image")
        or row.get("frames")
        or row.get("image_paths")
        or []
    )
    if not isinstance(images_raw, (list, tuple)):
        images_raw = [images_raw]
    image_paths: List[str] = []
    for i, img in enumerate(images_raw):
        if img is None:
            continue
        # PIL.Image-like (has .save and .convert)
        if hasattr(img, "save") and hasattr(img, "convert"):
            out = image_cache_dir / dataset_safe / f"{row_idx}" / f"{i}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                try:
                    img.convert("RGB").save(out, "JPEG", quality=88)
                except Exception:
                    continue
            image_paths.append(str(out))
        elif isinstance(img, str):
            if img.strip():
                image_paths.append(img)
        elif isinstance(img, dict):
            p = img.get("path") or img.get("bytes_path") or img.get("filename")
            if p:
                image_paths.append(str(p))

    qid = (
        row.get("question_id")
        or row.get("id")
        or row.get("uid")
        or f"{dataset_safe}-{row_idx}"
    )
    segment_id = (
        row.get("segment_id")
        or row.get("video_id")
        or row.get("clip_id")
        or ""
    )
    category = (
        row.get("category")
        or row.get("competence")
        or row.get("tag")
        or row.get("type")
        or "uncategorized"
    )
    return {
        "question_id": str(qid),
        "segment_id": str(segment_id),
        "question": q,
        "references": refs,
        "images": image_paths,
        "category": str(category),
    }


def _load_hf_dataset(repo: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load a HuggingFace dataset and return canonical rows.

    Imports `datasets` lazily so the rest of the file (LingoQA path, NIM
    serving, FiftyOne) stays importable on hosts without it.
    """
    # Special-case: if the user pasted a URL or repo id that points at the
    # LingoQA dataset and we have the local GDrive cache available, use it.
    # This keeps the smoke test fast and avoids hitting HF for a dataset we
    # already have on disk.
    if _LINGOQA_HF_RE.search(repo) and _lingoqa_dataset_dir() is not None:
        log(f"[dataset hf:{repo}] short-circuit → on-disk LingoQA cache")
        return group_lingoqa_by_question(load_lingoqa())

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"`datasets` library not installed. pip install datasets (got: {exc})"
        ) from exc

    use_split = split or "test"
    log(f"[dataset hf:{repo}] load_dataset(split={use_split!r})")
    try:
        ds = load_dataset(repo, split=use_split, trust_remote_code=False)
    except Exception as exc:
        # Fallback splits — many datasets only have 'train' or 'validation'.
        last_exc = exc
        for fallback in ("validation", "val", "train"):
            if fallback == use_split:
                continue
            try:
                log(f"[dataset hf:{repo}] split={use_split!r} failed, trying {fallback!r}")
                ds = load_dataset(repo, split=fallback, trust_remote_code=False)
                use_split = fallback
                last_exc = None
                break
            except Exception as exc2:
                last_exc = exc2
                continue
        if last_exc is not None:
            raise RuntimeError(f"HF load_dataset failed for {repo}: {last_exc}") from last_exc

    safe = _safe_dataset_dir_name(repo)
    image_cache_dir = _HF_DATASETS_CACHE_ROOT
    image_cache_dir.mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        try:
            normalized = _normalize_hf_row(dict(row), idx, safe, image_cache_dir)
        except Exception as exc:
            log(f"[dataset hf:{repo}] row {idx} normalize failed: {exc}")
            continue
        if normalized is not None:
            out.append(normalized)
    log(f"[dataset hf:{repo}] normalized {len(out)} rows from split={use_split!r}")
    return out


def _load_local_parquet(path: str) -> List[Dict[str, Any]]:
    """Load an arbitrary local parquet and normalize each row."""
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(f"pyarrow required to read local parquet: {exc}") from exc
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"local parquet not found: {p}")
    table = pq.read_table(p)
    rows = table.to_pylist()
    safe = _safe_dataset_dir_name(p.stem)
    image_cache_dir = _HF_DATASETS_CACHE_ROOT
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        try:
            normalized = _normalize_hf_row(row, idx, safe, image_cache_dir)
        except Exception:
            continue
        if normalized is not None:
            out.append(normalized)
    return out


def _load_github_dataset(repo: str) -> List[Dict[str, Any]]:
    """Walk a github repo's data/ or eval/ subdir for parquet files.

    Best-effort: clones via `git clone --depth 1`. First parquet whose rows
    normalize cleanly wins.
    """
    branch = None
    if "@" in repo:
        repo, branch = repo.split("@", 1)
    safe = _safe_dataset_dir_name(repo.replace("/", "__"))
    cache = Path("/tmp/github-datasets") / safe
    cache.mkdir(parents=True, exist_ok=True)
    repo_dir = cache / "repo"
    if not repo_dir.exists():
        url = f"https://github.com/{repo}.git"
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["-b", branch]
        cmd += [url, str(repo_dir)]
        log(f"[dataset github:{repo}] clone: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr[:400]}")
    # Find a parquet candidate.
    candidates: List[Path] = []
    for sub in ("data", "eval", "evaluation", "datasets"):
        d = repo_dir / sub
        if d.exists():
            candidates.extend(sorted(d.rglob("*.parquet")))
    if not candidates:
        candidates = sorted(repo_dir.rglob("*.parquet"))
    for parquet in candidates:
        try:
            rows = _load_local_parquet(str(parquet))
            if rows:
                log(f"[dataset github:{repo}] using {parquet} ({len(rows)} rows)")
                return rows
        except Exception as exc:
            log(f"[dataset github:{repo}] {parquet} unusable: {exc}")
            continue
    return []


def _load_gdrive_dataset(folder_id: str) -> List[Dict[str, Any]]:
    """Pull a GDrive folder with gdown and walk for parquet files."""
    safe = _safe_dataset_dir_name(folder_id)
    cache = Path("/tmp/gdrive-cache") / safe
    cache.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["gdown", "--folder", folder_id, "-O", str(cache)],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"gdown not installed: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gdown failed: {exc.stderr[:400]}") from exc
    for parquet in sorted(cache.rglob("*.parquet")):
        try:
            rows = _load_local_parquet(str(parquet))
            if rows:
                return rows
        except Exception:
            continue
    return []


def load_dataset_by_id(
    dataset_id: str,
    source_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Dispatch any benchmark dataset_id to its loader.

    Returns rows in the same shape as group_lingoqa_by_question(load_lingoqa()).
    Raises RuntimeError on unknown prefixes (preserves the existing error
    surface for true unknowns).
    """
    ds = str(dataset_id or "").strip()
    if not ds:
        raise RuntimeError("empty dataset_id")

    if ds == "lingoqa-official":
        return group_lingoqa_by_question(load_lingoqa())

    if ds.startswith("hf:"):
        spec = ds[3:]
        split: Optional[str] = None
        if "@" in spec:
            spec, split = spec.split("@", 1)
        return _load_hf_dataset(spec, split=split)

    if ds.startswith("arxiv:"):
        arxiv_id = ds[len("arxiv:"):]
        url = f"https://arxiv.org/abs/{arxiv_id}"
        if "discover_paper_source" not in globals():
            raise RuntimeError("discover_paper_source helper not available")
        try:
            discovery = globals()["discover_paper_source"](url, require_dataset=True) or {}
        except Exception as exc:
            raise RuntimeError(f"arxiv discovery failed for {arxiv_id}: {exc}") from exc
        hf_repo = discovery.get("selected_dataset")
        if not hf_repo:
            raise RuntimeError(
                f"arxiv:{arxiv_id} resolved with no HF dataset link; paste a "
                f"specific HF dataset URL instead."
            )
        log(f"[dataset arxiv:{arxiv_id}] dispatching to hf:{hf_repo}")
        return _load_hf_dataset(hf_repo)

    if ds.startswith("github:"):
        spec = ds[len("github:"):]
        return _load_github_dataset(spec)

    if ds.startswith("gdrive:"):
        return _load_gdrive_dataset(ds[len("gdrive:"):])

    if ds.startswith("local:"):
        return _load_local_parquet(ds[len("local:"):])

    raise RuntimeError(f"Unsupported dataset id: {dataset_id}")


def _ensure_judge() -> None:
    """Lazy-load wayveai/Lingo-Judge. Holds in memory after first call."""
    global _JUDGE_TOKENIZER, _JUDGE_MODEL, _JUDGE_DEVICE
    if _JUDGE_MODEL is not None:
        return
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception as exc:
        with BENCHMARK_LOCK:
            BENCHMARK_STATE["judge_error"] = f"transformers import failed: {exc}"
        raise
    log("Loading Lingo-Judge (wayveai/Lingo-Judge, base=microsoft/deberta-v3-base)")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained("wayveai/Lingo-Judge")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    _JUDGE_TOKENIZER = tokenizer
    _JUDGE_MODEL = model
    _JUDGE_DEVICE = device
    with BENCHMARK_LOCK:
        BENCHMARK_STATE["judge_loaded"] = True
        BENCHMARK_STATE["judge_error"] = None
    log(f"Lingo-Judge ready on {device}")


def lingo_judge_score(question: str, reference: str, prediction: str) -> float:
    """Run a single (q, ref, pred) triple through Lingo-Judge.

    Returns sigmoid probability in [0, 1]; > 0.5 is judged correct.
    """
    _ensure_judge()
    import torch

    text = f"[CLS]\nQuestion: {question}\nAnswer: {reference}\nStudent: {prediction}"
    encoded = _JUDGE_TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    ).to(_JUDGE_DEVICE)
    with torch.no_grad():
        out = _JUDGE_MODEL(**encoded)
    logit = out.logits.squeeze().detach().cpu().float().item()
    return float(1.0 / (1.0 + pow(2.718281828, -logit)))


def lingo_judge_score_max(question: str, references: List[str], prediction: str) -> Tuple[float, int]:
    """Return (max_score, index_of_winning_reference)."""
    scores = [lingo_judge_score(question, ref, prediction) for ref in references]
    if not scores:
        return 0.0, -1
    best = max(range(len(scores)), key=scores.__getitem__)
    return scores[best], best


def _strip_think_block(text: str) -> str:
    """Strip <think>...</think> and any <answer>...</answer> wrapper for the
    final-answer surfaced to Lingo-Judge.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", cleaned, flags=re.IGNORECASE)
    if m:
        cleaned = m.group(1)
    return cleaned.strip()


def _extract_think_block(text: str) -> str:
    m = re.search(r"<think>([\s\S]*?)</think>", text or "", flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_final_answer(prediction: str) -> str:
    """Strip every `<think>...</think>` block (incl. tags) from a prediction.

    Edge cases handled per Phase J amendment:
      - No `<think>` tag → return prediction.strip()
      - Unclosed `<think>` (no closing tag) → return everything AFTER the
        opening `<think>` tag (or empty string if nothing follows). This
        preserves the model's *attempt* at an answer instead of dropping it.
      - Multiple `<think>` blocks → strip all of them.
      - Leading/trailing whitespace stripped.

    The returned string is the "answer-only" payload sent to Lingo-Judge in
    `judge_mode="answer-only"` / `judge_mode="both"`. Goal: brief, gt-A/gt-B-
    style text so the judge's 512-token window holds the entire final answer.
    """
    if not prediction:
        return ""
    text = prediction
    has_open = re.search(r"<think>", text, flags=re.IGNORECASE)
    has_close = re.search(r"</think>", text, flags=re.IGNORECASE)
    if has_open and not has_close:
        # Unclosed: keep everything after the FIRST <think> opener.
        m = re.search(r"<think>([\s\S]*)", text, flags=re.IGNORECASE)
        text = m.group(1) if m else ""
    else:
        # Strip every closed block (greedy across all of them).
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # If the model also wrapped its answer in <answer>...</answer>, unwrap it.
    m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text, flags=re.IGNORECASE)
    if m:
        text = m.group(1)
    return text.strip()


# DeBERTa-v3-base (the Lingo-Judge backbone) hard-truncates inputs at 512
# tokens — see lingo_judge_score(). Reasoning-style models that emit long
# <think> chains can blow past this; we surface an approximate token count
# per row so the UI can call out "judge couldn't see the final answer."
#
# Heuristic: ~4 chars per token for English. Exact tokenizer counts would
# be nicer but adding tiktoken as a dep is out of scope — the goal here is
# directional, not precise.
JUDGE_MAX_TOKENS = 512


def _judge_text_tokens(question: str, reference: str, prediction: str) -> int:
    """Estimate the token count of the exact string the judge tokenizes:
    `[CLS]\\nQuestion: <q>\\nAnswer: <ref>\\nStudent: <pred>`.
    Uses a 4-char-per-token heuristic — directional, not exact.
    """
    full = f"[CLS]\nQuestion: {question or ''}\nAnswer: {reference or ''}\nStudent: {prediction or ''}"
    return max(1, len(full) // 4)


def _judge_text_for_row(question: str, reference: str, prediction: str) -> str:
    """The literal string fed to Lingo-Judge for one (q, ref, pred) triple.

    Surfaced in the detail panel so a reader can see WHAT the judge actually
    saw — including how a 1024-token reasoning trace gets clipped to 512.
    """
    return f"[CLS]\nQuestion: {question or ''}\nAnswer: {reference or ''}\nStudent: {prediction or ''}"


def benchmark_nim_base_url() -> str:
    return (
        os.getenv("BENCHMARK_NIM_BASE_URL")
        or os.getenv("VLLM_BASE_URL")
        or "http://localhost:8000/v1"
    )


def benchmark_nim_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = os.environ.get("NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY") or os.environ.get("VLLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def benchmark_detect_model() -> str:
    """Probe /v1/models on the live NIM and return the served model id."""
    base = benchmark_nim_base_url().rstrip("/")
    url = base + "/models"
    resp = requests.get(url, headers=benchmark_nim_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError(f"NIM /v1/models returned no entries: {url}")
    return data[0].get("id") or data[0].get("root") or ""


def run_one_image_qa(
    sample: Dict[str, Any],
    model: str,
    base_url: str,
    headers: Dict[str, str],
    *,
    system_prompt: Optional[str] = None,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    """LingoQA image-Q/A inference primitive.

    Mirrors run_one for video; uses the build.nvidia.com canonical shape:
    one user message, content = [image_url ... image_url, text]. Sends
    NO max_tokens, NO temperature, NO top_p — Alex standing order.
    Captures latency, model id, raw response, and parses <think>/<answer>.
    """
    if requests is None:
        raise RuntimeError(f"requests import failed: {REQUESTS_IMPORT_ERROR}")

    images = sample.get("images") or []
    content: List[Dict[str, Any]] = []
    for path in images:
        try:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
        except FileNotFoundError:
            raise FileNotFoundError(f"LingoQA image missing on disk: {path}")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": sample["question"]})

    system_text = system_prompt or (
        "You are a helpful assistant analyzing driving-scene images. "
        "Answer the question in 1-2 sentences. Think step-by-step inside "
        "<think>...</think> tags first if reasoning helps; place the final "
        "answer after the closing </think> tag (or inside <answer>...</answer>)."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ],
        # NO max_tokens / temperature / top_p — standing order.
    }
    url = base_url.rstrip("/") + "/chat/completions"
    started = time.monotonic()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    elapsed = time.monotonic() - started
    if resp.status_code >= 400:
        raise RuntimeError(f"NIM {resp.status_code}: {resp.text[:600]}")
    body = resp.json()
    raw_text = ""
    try:
        raw_text = body["choices"][0]["message"]["content"] or ""
    except Exception:
        raw_text = json.dumps(body)[:1000]
    return {
        "question_id": sample["question_id"],
        "segment_id": sample["segment_id"],
        "question": sample["question"],
        "category": sample.get("category") or "uncategorized",
        "model": model,
        "raw_response": raw_text,
        "reasoning_trace": _extract_think_block(raw_text),
        "prediction": _strip_think_block(raw_text) or raw_text.strip(),
        "latency_seconds": elapsed,
        "usage": body.get("usage") or {},
        "images": images,
    }


def _benchmark_run_dir(run_id: str) -> Path:
    path = BENCHMARK_RUN_ROOT / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _benchmark_snapshot_save(run_id: str, snap: Dict[str, Any]) -> None:
    path = _benchmark_run_dir(run_id) / "results.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snap, default=str, indent=2), encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------
# Drop-zone Benchmark View additions (sprint: "paste a URL → pick a model → hit Run")
# ----------------------------------------------------------------------------

BENCHMARK_HISTORY_PATH = BENCHMARK_RUN_ROOT / "history.json"
BENCHMARK_HISTORY_MAX = 20
BENCHMARK_MULTI_NIM_GLOB = BENCHMARK_RUN_ROOT / "multi-nim-20260517"


def _benchmark_history_load() -> List[Dict[str, Any]]:
    try:
        if not BENCHMARK_HISTORY_PATH.exists():
            return []
        data = json.loads(BENCHMARK_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data[-BENCHMARK_HISTORY_MAX:]
    except Exception as exc:
        log(f"[benchmark history] load failed: {exc}")
    return []


def _benchmark_history_append(entry: Dict[str, Any]) -> None:
    try:
        history = _benchmark_history_load()
        history.append(entry)
        history = history[-BENCHMARK_HISTORY_MAX:]
        BENCHMARK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = BENCHMARK_HISTORY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(history, default=str, indent=2), encoding="utf-8")
        tmp.replace(BENCHMARK_HISTORY_PATH)
    except Exception as exc:
        log(f"[benchmark history] append failed: {exc}")


def _benchmark_history_find(run_id: str) -> Optional[Dict[str, Any]]:
    for entry in _benchmark_history_load():
        if entry.get("run_id") == run_id:
            return entry
    return None


# URL kind detection: HuggingFace, Arxiv, GitHub, GDrive.
_RE_HF = re.compile(r"^https?://(?:www\.)?huggingface\.co/datasets/([^/?#\s]+/[^/?#\s]+)", re.I)
_RE_ARXIV = re.compile(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf|html)/([\w.\-/]+)", re.I)
_RE_GITHUB = re.compile(r"^https?://(?:www\.)?github\.com/([^/\s]+/[^/\s]+)", re.I)
_RE_GDRIVE = re.compile(r"^https?://(?:drive|docs)\.google\.com/(?:drive/folders/|file/d/|open\?id=)([\w-]+)", re.I)
_LINGOQA_HF_RE = re.compile(r"(runoob1|wayveai|wayve-ai)/lingoqa", re.I)


def _resolve_url_kind(url: str) -> Tuple[str, Optional[str]]:
    """Return (kind, identifier) for the given URL/string.

    kind ∈ {"hf_dataset", "arxiv", "github", "gdrive", "lingoqa", "unknown"}

    Ordering: real URL regexes (HF/Arxiv/GitHub/GDrive) WIN over the bare
    "lingoqa" shortcut. This means a paste of
    https://huggingface.co/datasets/runoob1/lingoqa resolves to
    ("hf_dataset", "runoob1/lingoqa") and exercises the dispatcher's hf:
    path. The bare lingoqa shortcut is reserved for the literal string
    "lingoqa" (zero-keystroke baseline).
    """
    s = (url or "").strip()
    if not s:
        return ("unknown", None)
    low = s.lower()
    # Bare keyword shortcut only — typed text, not a URL.
    if low == "lingoqa":
        return ("lingoqa", "lingoqa-official")
    m = _RE_HF.match(s)
    if m:
        return ("hf_dataset", m.group(1))
    m = _RE_ARXIV.match(s)
    if m:
        return ("arxiv", m.group(1))
    m = _RE_GITHUB.match(s)
    if m:
        return ("github", m.group(1))
    m = _RE_GDRIVE.match(s)
    if m:
        return ("gdrive", m.group(1))
    # Backstop: only treat as lingoqa if it's a wayveai/lingoqa-style
    # reference that did NOT match a URL pattern above (e.g. a bare
    # "wayveai/lingoqa" repo id).
    if _LINGOQA_HF_RE.search(s):
        return ("lingoqa", "lingoqa-official")
    return ("unknown", None)


def _resolve_url_to_dataset(url: str) -> Dict[str, Any]:
    """Parse url → dataset config candidate.

    For LingoQA: always resolves to the cached on-disk path.
    For HF/Arxiv/GitHub/GDrive: returns a best-effort discovery payload using
    existing helpers when available. Network calls are deferred to the run step.
    """
    kind, ident = _resolve_url_kind(url)
    result: Dict[str, Any] = {"kind": kind, "url": url, "identifier": ident}
    if kind == "lingoqa":
        base = _lingoqa_dataset_dir()
        result["dataset_config"] = {
            "id": "lingoqa-official",
            "name": "LingoQA (1000 rows, GDrive local cache)",
            "rows": 1000,
            "ready": bool(base and (base / "val.parquet").exists()),
            "path": str(base) if base else None,
            "judge": "lingo-judge",
        }
        result["display"] = "LingoQA detected (local cache)"
        return result
    if kind == "hf_dataset":
        result["dataset_config"] = {
            "id": f"hf:{ident}",
            "name": f"HuggingFace dataset: {ident}",
            "repo_id": ident,
            "ready": False,
            "judge": "lingo-judge",
        }
        result["display"] = f"HuggingFace dataset detected: {ident}"
        return result
    if kind == "arxiv":
        discovery: Dict[str, Any] = {}
        try:
            if "discover_paper_source" in globals():
                discovery = globals()["discover_paper_source"](url) or {}
        except Exception as exc:
            discovery = {"error": str(exc)}
        result["dataset_config"] = {
            "id": f"arxiv:{ident}",
            "name": f"Arxiv paper: {ident}",
            "arxiv_id": ident,
            "discovery": discovery,
            "ready": False,
            "judge": "lingo-judge",
        }
        result["display"] = f"Arxiv paper detected: {ident}"
        return result
    if kind == "github":
        result["dataset_config"] = {
            "id": f"github:{ident}",
            "name": f"GitHub repo: {ident}",
            "repo": ident,
            "ready": False,
            "judge": "lingo-judge",
        }
        result["display"] = f"GitHub repo detected: {ident}"
        return result
    if kind == "gdrive":
        result["dataset_config"] = {
            "id": f"gdrive:{ident}",
            "name": f"GDrive folder: {ident}",
            "gdrive_id": ident,
            "ready": False,
            "judge": "lingo-judge",
        }
        result["display"] = f"GDrive folder detected"
        return result
    result["dataset_config"] = None
    result["display"] = "Unknown URL kind — paste a HuggingFace dataset, Arxiv paper, GitHub repo, GDrive folder, or type 'lingoqa'."
    return result


def _resolve_url_to_dataset_cached(url: str) -> Dict[str, Any]:
    """Wrap _resolve_url_to_dataset and cache the dataset_config in
    BENCHMARK_STATE['resolved_datasets'] so /benchmark/datasets can expose
    the session's accumulated pool to the frontend dropdown."""
    payload = _resolve_url_to_dataset(url)
    cfg = (payload or {}).get("dataset_config") or {}
    ds_id = cfg.get("id")
    if ds_id:
        with BENCHMARK_LOCK:
            BENCHMARK_STATE.setdefault("resolved_datasets", {})[ds_id] = {
                **cfg,
                "source_url": url,
                "resolved_epoch": time.time(),
            }
    return payload


def _benchmark_available_models() -> List[Dict[str, Any]]:
    """Return [{id, name, endpoint, status, host}, ...].

    Reads from /tmp/benchmark-runs/multi-nim-20260517/*/identity.json if present;
    falls back to the local /v1/models on the active NIM base url.
    """
    out: List[Dict[str, Any]] = []
    multi_root = BENCHMARK_MULTI_NIM_GLOB
    if multi_root.exists() and multi_root.is_dir():
        for child in sorted(multi_root.iterdir()):
            identity = child / "identity.json"
            if not identity.exists():
                continue
            try:
                ident = json.loads(identity.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "id": ident.get("served_model_id") or ident.get("model") or child.name,
                "name": ident.get("name") or ident.get("served_model_id") or child.name,
                "endpoint": ident.get("endpoint") or ident.get("base_url"),
                "status": ident.get("status") or "ready",
                "host": ident.get("host") or ident.get("gpu") or "",
            })
    # Always include the local detection as the default option.
    try:
        local_model = benchmark_detect_model()
        local_base = benchmark_nim_base_url()
        local_entry = {
            "id": local_model,
            "name": local_model,
            "endpoint": local_base,
            "status": "ready",
            "host": "horde (local)",
        }
        # Dedupe by id — prefer the local entry if multi-nim list didn't include it.
        if not any(m.get("id") == local_entry["id"] for m in out):
            out.insert(0, local_entry)
    except Exception as exc:
        out.insert(0, {
            "id": "nvidia/Cosmos3-Super-Reasoner",
            "name": "nvidia/Cosmos3-Super-Reasoner",
            "endpoint": benchmark_nim_base_url() if "benchmark_nim_base_url" in globals() else "",
            "status": "error",
            "host": "horde (local)",
            "error": str(exc),
        })
    return out


def benchmark_get(run_id: str) -> Optional[Dict[str, Any]]:
    with BENCHMARK_LOCK:
        snap = BENCHMARK_STATE["runs"].get(run_id)
        return json.loads(json.dumps(snap, default=str)) if snap else None


def _benchmark_update(rid: str, **fields: Any) -> Dict[str, Any]:
    with BENCHMARK_LOCK:
        snap = BENCHMARK_STATE["runs"].setdefault(rid, {})
        snap.update(fields)
        snap["updated_epoch"] = time.time()
        snap.setdefault("run_id", rid)
        return json.loads(json.dumps(snap, default=str))


def _benchmark_append_result(rid: str, record: Dict[str, Any]) -> Dict[str, Any]:
    with BENCHMARK_LOCK:
        snap = BENCHMARK_STATE["runs"].setdefault(rid, {})
        results = snap.setdefault("results", [])
        results.append(record)
        snap["progress"] = {
            "done": len(results),
            "total": snap.get("total") or 0,
            "errors": sum(1 for r in results if r.get("error")),
        }
        snap["updated_epoch"] = time.time()
        return json.loads(json.dumps(snap, default=str))


def _benchmark_summarize(snap: Dict[str, Any]) -> Dict[str, Any]:
    results = [r for r in (snap.get("results") or []) if not r.get("error")]
    total = len(results)
    correct = sum(1 for r in results if r.get("judge_correct"))
    by_cat: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r.get("category") or "uncategorized"
        bucket = by_cat.setdefault(cat, {"total": 0, "correct": 0, "scores": []})
        bucket["total"] += 1
        if r.get("judge_correct"):
            bucket["correct"] += 1
        if r.get("judge_score") is not None:
            bucket["scores"].append(r["judge_score"])
    per_category = []
    for cat in LINGOQA_CATEGORY_ORDER + [c for c in by_cat if c not in LINGOQA_CATEGORY_ORDER]:
        if cat not in by_cat:
            continue
        b = by_cat[cat]
        acc = (b["correct"] / b["total"]) if b["total"] else 0.0
        avg = (sum(b["scores"]) / len(b["scores"])) if b["scores"] else None
        per_category.append({
            "category": cat,
            "total": b["total"],
            "correct": b["correct"],
            "accuracy": acc,
            "average_score": avg,
        })
    latencies = [r.get("latency_seconds") for r in results if r.get("latency_seconds") is not None]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    answer_lens = [len((r.get("prediction") or "").split()) for r in results]
    avg_len = (sum(answer_lens) / len(answer_lens)) if answer_lens else None
    reasoning_pct = (sum(1 for r in results if r.get("reasoning_trace")) / total) if total else 0.0
    # Judge-input truncation rate. A row is "truncated" if EITHER reference's
    # [CLS]\nQuestion:\nAnswer:\nStudent: string exceeded the judge's 512-token
    # window. Surfaced in the Recent Samples header so we can spot reasoning
    # models being silently under-scored at the population level.
    truncated_rows = sum(
        1 for r in results
        if r.get("judge_truncated_a") or r.get("judge_truncated_b")
    )
    truncated_rows_answer_only = sum(
        1 for r in results
        if r.get("judge_truncated_a_answer_only") or r.get("judge_truncated_b_answer_only")
    )
    # ---- Dual judge-mode A/B summary (Phase J amendment) ----
    # When the run is in mode "both" each row carries both judge_correct_*
    # fields. Surface side-by-side accuracy + delta so the UI tile can show
    # the headline "is the long-think prediction costing us X.Xpp?" finding.
    std_rows = [r for r in results if r.get("judge_correct_standard") is not None]
    ao_rows = [r for r in results if r.get("judge_correct_answer_only") is not None]
    acc_std = (sum(1 for r in std_rows if r.get("judge_correct_standard")) / len(std_rows)) if std_rows else None
    acc_ao = (sum(1 for r in ao_rows if r.get("judge_correct_answer_only")) / len(ao_rows)) if ao_rows else None
    delta_pp = ((acc_ao - acc_std) * 100.0) if (acc_std is not None and acc_ao is not None) else None
    # Count divergent rows (verdicts disagree between the two modes).
    divergent_rows = sum(
        1 for r in results
        if r.get("judge_correct_standard") is not None
        and r.get("judge_correct_answer_only") is not None
        and bool(r.get("judge_correct_standard")) != bool(r.get("judge_correct_answer_only"))
    )
    return {
        "total_predictions": total,
        "correct": correct,
        "overall_accuracy": (correct / total) if total else 0.0,
        "per_category": per_category,
        "average_latency_seconds": avg_latency,
        "average_answer_words": avg_len,
        "reasoning_trace_pct": reasoning_pct,
        "judge_truncated_rows": truncated_rows,
        "judge_truncated_pct": (truncated_rows / total) if total else 0.0,
        "judge_truncated_rows_answer_only": truncated_rows_answer_only,
        "judge_truncated_pct_answer_only": (truncated_rows_answer_only / total) if total else 0.0,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "accuracy_standard": acc_std,
        "accuracy_answer_only": acc_ao,
        "accuracy_delta_pp": delta_pp,
        "divergent_rows": divergent_rows,
    }


def run_lingoqa_benchmark(
    run_id: str,
    dataset_id: str,
    judge_id: str,
    sample_size: int,
    concurrency: int,
    seed: int,
    source_config: Optional[Dict[str, Any]] = None,
    judge_mode: str = "standard",
) -> None:
    """Benchmark worker thread. Mutates BENCHMARK_STATE['runs'][run_id].

    Despite the historical name, this worker now drives any dataset
    dispatched by load_dataset_by_id — not just LingoQA. The function name
    is kept for backwards compatibility with the API handler call site.

    judge_mode (Phase J amendment — dual judge-mode A/B):
      - "standard"    : judge sees result["prediction"] (default; back-compat).
      - "answer-only" : judge sees _extract_final_answer(prediction). The
                        <think>...</think> chain is removed before scoring so
                        the brief final answer fits in DeBERTa's 512-token
                        window — mirrors the brief gt-A/gt-B reference style.
      - "both"        : runs both scorers per row. Each result captures
                        judge_score_{standard,answer_only} and
                        judge_correct_{standard,answer_only}. The verdict
                        used for headline accuracy aliases to "standard" so
                        existing dashboards keep working.
    """
    judge_mode = judge_mode if judge_mode in ("standard", "answer-only", "both") else "standard"
    started = time.time()
    try:
        _benchmark_update(
            run_id,
            dataset=dataset_id,
            judge=judge_id,
            judge_mode=judge_mode,
            sample_size=sample_size,
            concurrency=concurrency,
            seed=seed,
            started_epoch=started,
            status="loading_dataset",
        )

        questions = load_dataset_by_id(dataset_id, source_config)
        if not questions:
            raise RuntimeError(
                f"Dataset {dataset_id} loaded but returned 0 rows "
                "(no questions with question + reference answer)"
            )

        if sample_size and sample_size < len(questions):
            import random as _random
            rng = _random.Random(seed)
            questions = rng.sample(questions, sample_size)

        _benchmark_update(
            run_id,
            total=len(questions),
            status="probing_model",
            results=[],
        )

        model = benchmark_detect_model()
        base = benchmark_nim_base_url()
        headers = benchmark_nim_headers()
        _benchmark_update(run_id, model=model, base_url=base, status="loading_judge")

        if judge_id == "lingo-judge":
            _ensure_judge()
        else:
            raise RuntimeError(f"Unsupported judge id: {judge_id}")

        _benchmark_update(run_id, status="running")
        log(f"[benchmark {run_id}] model={model} questions={len(questions)} concurrency={concurrency}")

        flush_every = 10
        results_collected = 0

        # ThreadPoolExecutor for model inference; judge runs serially after
        # each inference to avoid GPU contention.
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(run_one_image_qa, q, model, base, headers): q
                for q in questions
            }
            for fut in concurrent.futures.as_completed(futures):
                q = futures[fut]
                try:
                    result = fut.result()
                    refs = q.get("references") or []
                    pred_standard = result.get("prediction") or ""
                    pred_answer_only = _extract_final_answer(pred_standard)
                    # Capture the answer-only string for UI transparency even
                    # in standard mode — surfaces "what would have been sent"
                    # in the detail panel without re-running the judge.
                    result["prediction_answer_only"] = pred_answer_only
                    # ---- Run judge in requested mode(s) ----
                    score_standard = score_answer_only = None
                    ref_idx_standard = ref_idx_answer_only = -1
                    if judge_mode in ("standard", "both"):
                        score_standard, ref_idx_standard = lingo_judge_score_max(
                            q["question"], q["references"], pred_standard
                        )
                    if judge_mode in ("answer-only", "both"):
                        score_answer_only, ref_idx_answer_only = lingo_judge_score_max(
                            q["question"], q["references"], pred_answer_only
                        )
                    # ---- Headline (back-compat) fields: alias to the active
                    # mode for "standard" / "answer-only", or to standard when
                    # mode is "both" (Phase J amendment: A/B summary surfaces
                    # delta but existing dashboards keep reading judge_score).
                    if judge_mode == "answer-only":
                        primary_score, primary_idx = score_answer_only, ref_idx_answer_only
                    else:
                        primary_score, primary_idx = score_standard, ref_idx_standard
                    result["judge_mode"] = judge_mode
                    result["judge_score"] = primary_score
                    result["judge_correct"] = bool((primary_score or 0) > 0.5)
                    result["judge_winning_reference_index"] = primary_idx
                    # Dual fields (populated when computed; None otherwise).
                    result["judge_score_standard"] = score_standard
                    result["judge_correct_standard"] = bool((score_standard or 0) > 0.5) if score_standard is not None else None
                    result["judge_winning_reference_index_standard"] = ref_idx_standard if score_standard is not None else None
                    result["judge_score_answer_only"] = score_answer_only
                    result["judge_correct_answer_only"] = bool((score_answer_only or 0) > 0.5) if score_answer_only is not None else None
                    result["judge_winning_reference_index_answer_only"] = ref_idx_answer_only if score_answer_only is not None else None
                    result["references"] = q["references"]
                    # ---- Token-count transparency for BOTH judge-input strings
                    # so the UI can show whether the answer-only mode actually
                    # fit under the 512-token window (the whole point of the
                    # A/B). Per-row fields:
                    #   judge_input_tokens_a, _b            -> standard mode
                    #   judge_input_tokens_a_answer_only, _b_answer_only
                    #   judge_truncated_a, _b                -> standard mode
                    #   judge_truncated_a_answer_only, _b_answer_only
                    n_a_std = _judge_text_tokens(q["question"], refs[0] if len(refs) > 0 else "", pred_standard)
                    n_b_std = _judge_text_tokens(q["question"], refs[1], pred_standard) if len(refs) > 1 else None
                    n_a_ao = _judge_text_tokens(q["question"], refs[0] if len(refs) > 0 else "", pred_answer_only)
                    n_b_ao = _judge_text_tokens(q["question"], refs[1], pred_answer_only) if len(refs) > 1 else None
                    result["judge_input_tokens_a"] = n_a_std
                    result["judge_input_tokens_b"] = n_b_std
                    result["judge_truncated_a"] = n_a_std > JUDGE_MAX_TOKENS
                    result["judge_truncated_b"] = bool(n_b_std and n_b_std > JUDGE_MAX_TOKENS)
                    result["judge_input_tokens_a_answer_only"] = n_a_ao
                    result["judge_input_tokens_b_answer_only"] = n_b_ao
                    result["judge_truncated_a_answer_only"] = n_a_ao > JUDGE_MAX_TOKENS
                    result["judge_truncated_b_answer_only"] = bool(n_b_ao and n_b_ao > JUDGE_MAX_TOKENS)
                    result["judge_max_tokens"] = JUDGE_MAX_TOKENS
                    # Verbatim judge-input strings (both modes) so the detail
                    # panel can show A/B exactly. NOTE: not truncated for
                    # display by the producer — UI clips for rendering only.
                    result["judge_input_text_a"] = _judge_text_for_row(
                        q["question"], refs[0] if len(refs) > 0 else "", pred_standard
                    )
                    result["judge_input_text_b"] = (
                        _judge_text_for_row(q["question"], refs[1], pred_standard)
                        if len(refs) > 1 else None
                    )
                    result["judge_input_text_a_answer_only"] = _judge_text_for_row(
                        q["question"], refs[0] if len(refs) > 0 else "", pred_answer_only
                    )
                    result["judge_input_text_b_answer_only"] = (
                        _judge_text_for_row(q["question"], refs[1], pred_answer_only)
                        if len(refs) > 1 else None
                    )
                    result["error"] = None
                except Exception as exc:
                    log(f"[benchmark {run_id}] error on q={q.get('question_id')}: {exc}")
                    result = {
                        "question_id": q["question_id"],
                        "segment_id": q["segment_id"],
                        "question": q["question"],
                        "category": q.get("category"),
                        "references": q["references"],
                        "error": str(exc),
                        "judge_mode": judge_mode,
                        "judge_score": None,
                        "judge_correct": False,
                        "judge_score_standard": None,
                        "judge_correct_standard": None,
                        "judge_score_answer_only": None,
                        "judge_correct_answer_only": None,
                        "prediction": "",
                        "prediction_answer_only": "",
                        "raw_response": "",
                        "reasoning_trace": "",
                        "latency_seconds": None,
                        "images": q.get("images") or [],
                        "model": model,
                    }
                snap = _benchmark_append_result(run_id, result)
                results_collected += 1
                if results_collected % flush_every == 0 or results_collected == len(questions):
                    summary = _benchmark_summarize(snap)
                    snap = _benchmark_update(run_id, summary=summary)
                    _benchmark_snapshot_save(run_id, snap)

        final = _benchmark_update(
            run_id,
            status="complete",
            finished_epoch=time.time(),
            wall_seconds=time.time() - started,
            summary=_benchmark_summarize(benchmark_get(run_id) or {}),
        )
        _benchmark_snapshot_save(run_id, final)
        # Persist a one-line summary to /tmp/benchmark-runs/history.json so the
        # Benchmark View "Last run" tile + run-history modal survive restarts.
        try:
            summary_obj = final.get("summary") or {}
            _benchmark_history_append({
                "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "epoch": time.time(),
                "model": final.get("model") or "",
                "dataset": final.get("dataset") or "",
                "judge": final.get("judge") or "",
                "sample_size": final.get("sample_size"),
                "concurrency": final.get("concurrency"),
                "seed": final.get("seed"),
                "accuracy": summary_obj.get("overall_accuracy"),
                "total_predictions": summary_obj.get("total_predictions"),
                "wall_seconds": final.get("wall_seconds"),
                "report_url": f"/benchmark/report/{run_id}",
            })
        except Exception as exc:
            log(f"[benchmark {run_id}] history append failed: {exc}")
        log(f"[benchmark {run_id}] complete in {final['wall_seconds']:.1f}s")
    except Exception as exc:
        log(f"[benchmark {run_id}] FAILED: {exc}")
        final = _benchmark_update(
            run_id,
            status="error",
            error=str(exc),
            finished_epoch=time.time(),
        )
        try:
            _benchmark_snapshot_save(run_id, final)
        except Exception:
            pass


# =============================================================================
# Traceability bundle renderer (sprint goal: prove what each prediction saw).
#
# Each "worked example" in the benchmark report renders 4 artifacts so a reader
# can trace prediction -> input -> reasoning end-to-end:
#   1. frame_strip.png  : 5 input frames concatenated horizontally
#   2. frame_gif.gif    : same 5 frames at 1 fps, looped (video proxy)
#   3. ui_card.png      : PIL-rendered result card mirroring the Benchmark View
#   4. reasoning.png    : <think>...</think> chain as a styled card image
#
# Rendered on-demand at /benchmark/report build time (NOT at eval time) and
# cached under TRACEABILITY_ROOT/<run_id>/<model_safe>/<question_id>/. Frames
# are served raw via /lingoqa-frames/<segment_id>/<idx>.jpg for click-through.
# =============================================================================

TRACEABILITY_ROOT = Path(os.getenv("TRACEABILITY_ROOT", "/tmp/benchmark-artifacts"))
TRACEABILITY_ROOT.mkdir(parents=True, exist_ok=True)
NV_GREEN = (118, 185, 0)
NV_DARK = (26, 26, 26)
NV_SOFT = (240, 245, 232)
NV_WHITE = (255, 255, 255)
NV_LINE = (203, 213, 225)

try:
    from PIL import Image as _PIL_Image, ImageDraw as _PIL_ImageDraw, ImageFont as _PIL_ImageFont  # type: ignore
    PIL_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _pil_exc:  # pragma: no cover - surfaced at render time
    _PIL_Image = None  # type: ignore
    _PIL_ImageDraw = None  # type: ignore
    _PIL_ImageFont = None  # type: ignore
    PIL_IMPORT_ERROR = _pil_exc


def _safe_model_dir(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id or "unknown")[:64] or "unknown"


def _traceability_dir(run_id: str, model_id: str, question_id: str) -> Path:
    out = TRACEABILITY_ROOT / run_id / _safe_model_dir(model_id) / (question_id or "unknown")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_font(size: int, mono: bool = False) -> Any:
    """Best-effort font loader. Falls back to PIL default if no TTF found."""
    if _PIL_ImageFont is None:
        return None
    candidates_mono = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Courier.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ]
    candidates_sans = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in (candidates_mono if mono else candidates_sans):
        if os.path.exists(p):
            try:
                return _PIL_ImageFont.truetype(p, size=size)
            except Exception:
                continue
    try:
        return _PIL_ImageFont.load_default()
    except Exception:
        return None


def _resolve_segment_image(image_root: Optional[Path], segment_id: str, idx: int) -> Optional[Path]:
    """Find <root>/val/<seg>/<idx>.jpg across known layouts."""
    if not segment_id:
        return None
    roots: List[Path] = []
    if image_root:
        roots.append(Path(image_root))
    base = _lingoqa_dataset_dir()
    if base:
        roots.append(_lingoqa_image_root(base))
    # Walk known disk shapes (val/<seg>/<idx>.jpg or images/val/<seg>/<idx>.jpg).
    for root in roots:
        for tail in ("val", "images/val"):
            p = root / tail / segment_id / f"{idx}.jpg"
            if p.exists():
                return p
    # Last-chance fallback: scan the well-known absolute paths.
    for prefix in ("/home/horde/lingoqa-data/evaluation/images/val",
                   "/home/horde/lingoqa-data/evaluation/images/images/val",
                   "/tmp/lingoqa-data/evaluation/images/val",
                   "/tmp/lingoqa-data/evaluation/images/images/val"):
        p = Path(prefix) / segment_id / f"{idx}.jpg"
        if p.exists():
            return p
    return None


def render_frame_strip(segment_id: str, image_root: Optional[Path], output_path: Path,
                       cell: Tuple[int, int] = (320, 180), gutter: int = 4) -> Path:
    """5-frame horizontal strip (PNG). 320x180 cells, 4px white gutter, ~1600x180."""
    if _PIL_Image is None:
        raise RuntimeError(f"PIL unavailable: {PIL_IMPORT_ERROR}")
    cw, ch = cell
    strip_w = cw * 5 + gutter * 4
    canvas = _PIL_Image.new("RGB", (strip_w, ch), NV_WHITE)
    for idx in range(5):
        src = _resolve_segment_image(image_root, segment_id, idx)
        if src is None:
            # Missing frame placeholder.
            cell_img = _PIL_Image.new("RGB", (cw, ch), (235, 238, 245))
            draw = _PIL_ImageDraw.Draw(cell_img)
            font = _load_font(14)
            draw.text((10, ch // 2 - 8), f"frame {idx} missing", fill=NV_DARK, font=font)
        else:
            try:
                img = _PIL_Image.open(src).convert("RGB")
            except Exception as exc:
                cell_img = _PIL_Image.new("RGB", (cw, ch), (235, 238, 245))
                draw = _PIL_ImageDraw.Draw(cell_img)
                font = _load_font(12)
                draw.text((6, ch // 2 - 8), f"err: {exc}"[:60], fill=NV_DARK, font=font)
            else:
                # Letterbox-fit into cell.
                img.thumbnail((cw, ch), _PIL_Image.LANCZOS)
                cell_img = _PIL_Image.new("RGB", (cw, ch), NV_DARK)
                ox = (cw - img.width) // 2
                oy = (ch - img.height) // 2
                cell_img.paste(img, (ox, oy))
        canvas.paste(cell_img, (idx * (cw + gutter), 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def render_frame_gif(segment_id: str, image_root: Optional[Path], output_path: Path,
                     fps: float = 1.0, size: Tuple[int, int] = (640, 360)) -> Path:
    """Animated GIF over the 5 frames, infinite loop. ~5s at 1 fps."""
    if _PIL_Image is None:
        raise RuntimeError(f"PIL unavailable: {PIL_IMPORT_ERROR}")
    w, h = size
    duration_ms = int(round(1000.0 / max(fps, 0.1)))
    frames: List[Any] = []
    for idx in range(5):
        src = _resolve_segment_image(image_root, segment_id, idx)
        frame = _PIL_Image.new("RGB", (w, h), NV_DARK)
        if src is not None:
            try:
                img = _PIL_Image.open(src).convert("RGB")
                img.thumbnail((w, h), _PIL_Image.LANCZOS)
                ox = (w - img.width) // 2
                oy = (h - img.height) // 2
                frame.paste(img, (ox, oy))
            except Exception:
                pass
        else:
            draw = _PIL_ImageDraw.Draw(frame)
            font = _load_font(20)
            draw.text((20, h // 2), f"frame {idx} missing", fill=NV_WHITE, font=font)
        frames.append(frame.convert("P", palette=_PIL_Image.ADAPTIVE, colors=128))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError("no frames assembled for GIF")
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return output_path


def _wrap_text(text: str, font: Any, max_width: int, draw: Any) -> List[str]:
    """Word-wrap text to fit max_width pixels using the given font."""
    if not text:
        return [""]
    lines: List[str] = []
    for raw_line in str(text).splitlines() or [""]:
        words = raw_line.split(" ")
        cur = ""
        for word in words:
            trial = (cur + " " + word).strip() if cur else word
            try:
                bbox = draw.textbbox((0, 0), trial, font=font)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = len(trial) * 7
            if tw <= max_width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines or [""]


def render_result_card(example: Dict[str, Any], output_path: Path,
                       image_root: Optional[Path] = None,
                       size: Tuple[int, int] = (1200, 640)) -> Path:
    """Programmatic UI result-card screenshot (PIL fallback for headless browser).

    Mirrors the Benchmark View card: NV-dark header bar with model + verdict
    badge, frame strip thumbnail on the left, question/GT/prediction columns
    on the right, "Expand reasoning" affordance at the bottom.
    """
    if _PIL_Image is None:
        raise RuntimeError(f"PIL unavailable: {PIL_IMPORT_ERROR}")
    W, H = size
    canvas = _PIL_Image.new("RGB", (W, H), NV_WHITE)
    draw = _PIL_ImageDraw.Draw(canvas)

    # Header bar.
    header_h = 64
    draw.rectangle([0, 0, W, header_h], fill=NV_DARK)
    draw.rectangle([0, 0, 8, header_h], fill=NV_GREEN)
    title_font = _load_font(22)
    sub_font = _load_font(14)
    model_text = str(example.get("model") or "model")
    draw.text((24, 14), model_text[:64], fill=NV_WHITE, font=title_font)
    draw.text((24, 42), "Benchmark View — Result Card", fill=(118, 185, 0), font=sub_font)

    # Verdict badge (top right).
    correct = bool(example.get("judge_correct"))
    verdict_text = "PASS" if correct else "FAIL"
    badge_color = NV_GREEN if correct else (185, 28, 28)
    badge_w, badge_h = 110, 36
    bx0, by0 = W - badge_w - 16, 14
    draw.rectangle([bx0, by0, bx0 + badge_w, by0 + badge_h], fill=badge_color)
    badge_font = _load_font(20)
    try:
        bb = draw.textbbox((0, 0), verdict_text, font=badge_font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th = 60, 20
    draw.text((bx0 + (badge_w - tw) // 2, by0 + (badge_h - th) // 2 - 2), verdict_text,
              fill=NV_WHITE, font=badge_font)

    # Frame strip thumbnail on the left.
    strip_x, strip_y = 24, header_h + 20
    strip_w, strip_h = 520, 96
    segment_id = str(example.get("segment_id") or "")
    cell_w = (strip_w - 4 * 4) // 5
    for idx in range(5):
        src = _resolve_segment_image(image_root, segment_id, idx)
        cx = strip_x + idx * (cell_w + 4)
        if src and src.exists():
            try:
                img = _PIL_Image.open(src).convert("RGB")
                img.thumbnail((cell_w, strip_h), _PIL_Image.LANCZOS)
                tile = _PIL_Image.new("RGB", (cell_w, strip_h), NV_DARK)
                tile.paste(img, ((cell_w - img.width) // 2, (strip_h - img.height) // 2))
                canvas.paste(tile, (cx, strip_y))
            except Exception:
                draw.rectangle([cx, strip_y, cx + cell_w, strip_y + strip_h], fill=NV_LINE)
        else:
            draw.rectangle([cx, strip_y, cx + cell_w, strip_y + strip_h], fill=NV_LINE)
    cap_font = _load_font(12)
    draw.text((strip_x, strip_y + strip_h + 4),
              f"Segment {segment_id[:24]} | 5 frames", fill=(100, 116, 139), font=cap_font)

    # Right column: question + GT + prediction.
    col_x = strip_x + strip_w + 28
    col_w = W - col_x - 24
    body_font = _load_font(17)
    label_font = _load_font(13)
    mono_font = _load_font(15, mono=True)
    y = header_h + 20
    refs = example.get("references") or []

    def _block(label: str, value: str, font: Any, max_lines: int = 4) -> int:
        nonlocal y
        draw.text((col_x, y), label, fill=(100, 116, 139), font=label_font)
        y += 18
        lines = _wrap_text(value or "", font, col_w, draw)
        for line in lines[:max_lines]:
            draw.text((col_x, y), line, fill=NV_DARK, font=font)
            y += 22
        if len(lines) > max_lines:
            draw.text((col_x, y), "…", fill=NV_DARK, font=font)
            y += 22
        y += 8
        return y

    _block("QUESTION", str(example.get("question") or ""), body_font, max_lines=3)
    _block("GROUND TRUTH (A / B)",
           " | ".join([(refs[0] if refs else ""), (refs[1] if len(refs) > 1 else "")]),
           body_font, max_lines=3)
    _block("PREDICTION", str(example.get("prediction") or ""), body_font, max_lines=4)

    # Footer: latency + reasoning affordance.
    footer_y = H - 64
    draw.line([(24, footer_y), (W - 24, footer_y)], fill=NV_LINE, width=1)
    lat = example.get("latency_seconds")
    jscore = example.get("judge_score")
    foot_font = _load_font(14)
    foot_left = f"Latency {lat:.2f}s" if isinstance(lat, (int, float)) else "Latency —"
    if isinstance(jscore, (int, float)):
        foot_left += f"  ·  Judge prob {jscore:.3f}"
    draw.text((24, footer_y + 14), foot_left, fill=(71, 85, 105), font=foot_font)
    expander = "▸  Expand reasoning trace"
    try:
        bb = draw.textbbox((0, 0), expander, font=foot_font)
        ew = bb[2] - bb[0]
    except Exception:
        ew = 180
    draw.text((W - ew - 24, footer_y + 14), expander, fill=NV_GREEN, font=foot_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def render_reasoning_image(reasoning_text: str, output_path: Path,
                           width: int = 900, max_height: int = 1200,
                           max_chars: int = 1500) -> Path:
    """Render the <think> trace as a styled PNG card (NV-green header + mono body)."""
    if _PIL_Image is None:
        raise RuntimeError(f"PIL unavailable: {PIL_IMPORT_ERROR}")
    text = (reasoning_text or "").strip() or "(no reasoning trace emitted)"
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
        truncated = True
    body_font = _load_font(16, mono=True)
    header_font = _load_font(18)
    pad = 24
    header_h = 52
    body_max_w = width - 2 * pad
    # Probe wrapping on a throwaway canvas to compute height.
    probe = _PIL_Image.new("RGB", (10, 10), NV_WHITE)
    draw = _PIL_ImageDraw.Draw(probe)
    lines = _wrap_text(text, body_font, body_max_w, draw)
    try:
        bb = draw.textbbox((0, 0), "Hg", font=body_font)
        line_h = (bb[3] - bb[1]) + 6
    except Exception:
        line_h = 22
    body_h = max(40, len(lines) * line_h + pad * 2)
    total_h = min(max_height, header_h + body_h + 12)
    canvas = _PIL_Image.new("RGB", (width, total_h), NV_WHITE)
    draw = _PIL_ImageDraw.Draw(canvas)
    # Outer border.
    draw.rectangle([0, 0, width - 1, total_h - 1], outline=NV_LINE, width=1)
    # Header.
    draw.rectangle([0, 0, width, header_h], fill=NV_GREEN)
    draw.text((pad, 14), "REASONING TRACE", fill=NV_WHITE, font=header_font)
    if truncated:
        try:
            bb = draw.textbbox((0, 0), "(truncated)", font=header_font)
            tw = bb[2] - bb[0]
        except Exception:
            tw = 110
        draw.text((width - tw - pad, 18), "(truncated)", fill=NV_WHITE, font=_load_font(13))
    # Body bg.
    draw.rectangle([0, header_h, width, total_h], fill=NV_SOFT)
    y = header_h + pad
    max_y = total_h - pad
    for line in lines:
        if y + line_h > max_y:
            draw.text((pad, y), "…", fill=NV_DARK, font=body_font)
            break
        draw.text((pad, y), line, fill=NV_DARK, font=body_font)
        y += line_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def render_traceability_bundle(example: Dict[str, Any], image_root: Optional[Path],
                               output_dir: Path) -> Dict[str, Path]:
    """Master orchestrator: build all four artifacts for one example.

    Returns dict with keys frame_strip, frame_gif, ui_card, reasoning -> Path.
    Cached: if an artifact already exists, we re-use it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_id = str(example.get("segment_id") or "")
    paths = {
        "frame_strip": output_dir / "frame_strip.png",
        "frame_gif": output_dir / "frame_gif.gif",
        "ui_card": output_dir / "ui_card.png",
        "reasoning": output_dir / "reasoning.png",
    }
    if not paths["frame_strip"].exists():
        render_frame_strip(segment_id, image_root, paths["frame_strip"])
    if not paths["frame_gif"].exists():
        render_frame_gif(segment_id, image_root, paths["frame_gif"])
    if not paths["ui_card"].exists():
        render_result_card(example, paths["ui_card"], image_root=image_root)
    if not paths["reasoning"].exists():
        render_reasoning_image(str(example.get("reasoning_trace") or ""), paths["reasoning"])
    return paths


def _traceability_examples(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pick the worked examples for the report: 2 correct, 2 wrong, 2 rich-reasoning."""
    results = [r for r in (snap.get("results") or []) if not r.get("error")]
    correct = [r for r in results if r.get("judge_correct")][:2]
    wrong = [r for r in results if not r.get("judge_correct")][:2]
    rich = sorted(
        [r for r in results if r.get("reasoning_trace")],
        key=lambda r: len(r.get("reasoning_trace") or ""),
        reverse=True,
    )[:2]
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for bucket in (correct, wrong, rich):
        for r in bucket:
            key = (r.get("question_id"), r.get("model"))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out


def benchmark_render_report(run_id: str) -> str:
    """Render the in-app HTML report (arxiv-2312.14115 12-section layout)."""
    snap = benchmark_get(run_id) or {}
    summary = snap.get("summary") or _benchmark_summarize(snap) or {}
    results = [r for r in (snap.get("results") or []) if not r.get("error")]
    model = snap.get("model") or ""
    judge = snap.get("judge") or ""
    overall = summary.get("overall_accuracy") or 0.0
    correct = summary.get("correct") or 0
    total = summary.get("total_predictions") or 0

    # Pick qualitative examples.
    by_correct = [r for r in results if r.get("judge_correct")]
    by_wrong = [r for r in results if not r.get("judge_correct")]
    by_reasoning = sorted(
        [r for r in results if r.get("reasoning_trace")],
        key=lambda r: len(r.get("reasoning_trace") or ""),
        reverse=True,
    )
    qual_correct = by_correct[:2]
    qual_wrong = by_wrong[:2]
    qual_reasoning = by_reasoning[:2]

    per_cat = summary.get("per_category") or []
    max_bar_acc = max([c["accuracy"] for c in per_cat] + [0.01])

    def _esc(s: Any) -> str:
        return html.escape(str(s or ""))

    def _row(question: str, label: str, prediction: str, score: float, correct: bool) -> str:
        klass = "ok" if correct else "bad"
        score_pct = f"{score * 100:.1f}%" if isinstance(score, (int, float)) else ""
        return (
            "<tr>"
            f"<td>{_esc(question)}</td>"
            f"<td>{_esc(label)}</td>"
            f"<td>{_esc(prediction)}</td>"
            f"<td class='{klass}'>{score_pct}</td>"
            f"<td class='{klass}'>{'correct' if correct else 'miss'}</td>"
            "</tr>"
        )

    bars_html = "".join(
        f"<div class='cat-row'><div class='cat-name'>{_esc(c['category'])} (n={c['total']})</div>"
        f"<div class='cat-bar'><div class='cat-fill' style='width:{c['accuracy'] * 100:.1f}%'></div></div>"
        f"<div class='cat-pct'>{c['accuracy'] * 100:.1f}%</div></div>"
        for c in per_cat
    )

    qualitative_rows = "".join(
        _row(r["question"], (r.get("references") or [""])[0], r["prediction"], r.get("judge_score") or 0.0, bool(r.get("judge_correct")))
        for r in (qual_correct + qual_wrong)
    )

    reasoning_blocks = "".join(
        f"<div class='trace'>"
        f"<div class='trace-q'><strong>Question:</strong> {_esc(r['question'])}</div>"
        f"<div class='trace-gt'><strong>GT-A:</strong> {_esc((r.get('references') or [''])[0])}</div>"
        f"<div class='trace-gt'><strong>GT-B:</strong> {_esc((r.get('references') or ['', ''])[1] if len(r.get('references') or []) > 1 else '')}</div>"
        f"<div class='trace-cot'><strong>Reasoning trace:</strong><pre>{_esc(r['reasoning_trace'])}</pre></div>"
        f"<div class='trace-ans'><strong>Final answer:</strong> {_esc(r['prediction'])}</div>"
        f"<div class='trace-verdict'>Lingo-Judge: <strong>{'True' if r.get('judge_correct') else 'False'}</strong> (prob {(r.get('judge_score') or 0.0):.3f})</div>"
        f"</div>"
        for r in qual_reasoning
    )

    # --- Traceability bundles -------------------------------------------------
    # Build (or re-use cached) 4-artifact bundles for ~6 worked examples and
    # render them inline. Falls back to a friendly note if PIL is unavailable
    # or rendering fails on a per-example basis.
    base = _lingoqa_dataset_dir()
    image_root = _lingoqa_image_root(base) if base else None
    trace_blocks: List[str] = []
    if PIL_IMPORT_ERROR is not None:
        traceability_html = (
            f"<p class='trace'>Traceability bundles disabled: PIL import failed "
            f"({_esc(PIL_IMPORT_ERROR)}). Install pillow to enable.</p>"
        )
    else:
        examples_for_trace = _traceability_examples(snap)
        for idx, r in enumerate(examples_for_trace, start=1):
            qid = str(r.get("question_id") or f"q{idx}")
            seg = str(r.get("segment_id") or "")
            out_dir = _traceability_dir(run_id, str(r.get("model") or model), qid)
            try:
                paths = render_traceability_bundle(r, image_root, out_dir)
            except Exception as exc:
                trace_blocks.append(
                    f"<section class='trace-bundle'><h3>Example {idx} — {_esc(qid)}</h3>"
                    f"<p class='miss'>Bundle render failed: {_esc(exc)}</p></section>"
                )
                continue
            # Serve the bundle assets through /benchmark/artifact/<run_id>/<model_safe>/<qid>/<name>.
            # Always use the sanitized (slash-free) model directory name so the URL
            # has a stable 4-segment shape regardless of which model id is used.
            model_seg = _safe_model_dir(str(r.get("model") or model))
            base_url = (
                f"/benchmark/artifact/"
                f"{urllib.parse.quote(run_id, safe='')}/"
                f"{urllib.parse.quote(model_seg, safe='')}/"
                f"{urllib.parse.quote(qid, safe='')}"
            )
            verdict = "PASS" if r.get("judge_correct") else "FAIL"
            verdict_klass = "ok" if r.get("judge_correct") else "bad"
            click_links = "  ".join(
                f"<a class='frame-link' target='_blank' href='/lingoqa-frames/{urllib.parse.quote(seg, safe='')}/{i}.jpg'>frame {i}</a>"
                for i in range(5)
            )
            trace_blocks.append(
                "<section class='trace-bundle'>"
                f"<h3>Example {idx} — <span class='qid'>{_esc(qid)}</span> "
                f"<span class='verdict {verdict_klass}'>{verdict}</span></h3>"
                f"<div class='trace-q'><strong>Q:</strong> {_esc(r.get('question'))}</div>"
                f"<div class='trace-art'><div class='art-label'>Input frames (5, click to enlarge)</div>"
                f"<a target='_blank' href='/lingoqa-frames/{urllib.parse.quote(seg, safe='')}/0.jpg'>"
                f"<img class='strip' src='{base_url}/frame_strip.png' alt='Input frame strip'/></a>"
                f"<div class='frame-links'>{click_links}</div></div>"
                f"<div class='trace-art'><div class='art-label'>Animated GIF (1 fps, video proxy)</div>"
                f"<img class='gif' src='{base_url}/frame_gif.gif' alt='Animated GIF over 5 frames'/></div>"
                f"<div class='trace-art'><div class='art-label'>UI result card</div>"
                f"<img class='ui' src='{base_url}/ui_card.png' alt='Benchmark View UI card'/></div>"
                f"<div class='trace-art'><div class='art-label'>Reasoning trace</div>"
                f"<img class='reasoning' src='{base_url}/reasoning.png' alt='Reasoning trace'/></div>"
                f"<div class='trace-cap'>question_id: {_esc(qid)} · segment_id: {_esc(seg)} · "
                f"Lingo-Judge: <strong>{verdict}</strong> (prob {(r.get('judge_score') or 0.0):.3f})</div>"
                "</section>"
            )
        traceability_html = "".join(trace_blocks) or "<p>No worked examples available for traceability.</p>"

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>LingoQA Benchmark Report — {_esc(run_id)}</title>
<style>
:root {{ --green:#76B900; --dark:#1A1A1A; --bg:#fff; --soft:#F0F5E8; }}
body {{ margin:0; font-family:Inter,system-ui,sans-serif; color:var(--dark); background:var(--bg); font-size:18px; line-height:1.5; }}
.slide {{ max-width:1100px; margin:0 auto; padding:48px 32px; border-bottom:1px solid #e5e7eb; }}
.slide-title {{ font-size:32px; font-weight:700; margin:0 0 16px; color:var(--dark); }}
.slide-sub {{ font-size:20px; color:#475569; margin-bottom:24px; }}
.hero {{ font-size:120px; font-weight:800; color:var(--green); line-height:1; }}
.hero-sub {{ font-size:32px; margin:8px 0 16px; }}
.definition {{ background:var(--soft); border-left:6px solid var(--green); padding:16px 20px; font-size:18px; }}
.tile-grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; }}
.tile {{ border:1px solid #cbd5e1; border-left:6px solid var(--green); border-radius:8px; padding:18px; background:#fff; }}
.tile h4 {{ margin:0 0 8px; font-size:16px; color:#475569; text-transform:uppercase; letter-spacing:0.04em; }}
.tile .val {{ font-size:28px; font-weight:700; }}
.cat-row {{ display:grid; grid-template-columns:240px 1fr 80px; gap:12px; align-items:center; margin:8px 0; }}
.cat-name {{ font-size:18px; }}
.cat-bar {{ background:#e5e7eb; height:18px; border-radius:9px; overflow:hidden; border:1px solid #cbd5e1; }}
.cat-fill {{ background:var(--green); height:100%; }}
.cat-pct {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }}
table.qual {{ width:100%; border-collapse:collapse; margin-top:16px; }}
table.qual th, table.qual td {{ padding:12px; border-bottom:1px solid #e5e7eb; vertical-align:top; text-align:left; font-size:17px; }}
table.qual th {{ background:var(--dark); color:#fff; font-weight:600; }}
table.qual td.ok {{ background:#ECFDF5; }}
table.qual td.bad {{ background:#FEF2F2; }}
.trace {{ background:#f8fafc; border:1px solid #cbd5e1; border-left:6px solid var(--green); padding:18px; margin:12px 0; border-radius:8px; }}
.trace pre {{ background:var(--soft); padding:12px; border-radius:6px; white-space:pre-wrap; font-size:16px; }}
.ref-bar {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:16px; font-size:18px; }}
.ref-bar .pill {{ background:#fff; border:1px solid #cbd5e1; border-radius:999px; padding:6px 14px; }}
.ref-bar .pill.us {{ background:var(--green); color:#fff; border-color:var(--green); }}
.kv {{ display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; }}
.kv div:nth-child(odd) {{ color:#64748b; }}
.trace-bundle {{ background:#fff; border:1px solid #cbd5e1; border-left:6px solid var(--green); padding:18px 20px; margin:18px 0; border-radius:8px; }}
.trace-bundle h3 {{ margin:0 0 8px; font-size:20px; }}
.trace-bundle .qid {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:14px; color:#64748b; font-weight:normal; }}
.trace-bundle .verdict {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:13px; font-weight:700; margin-left:8px; vertical-align:middle; }}
.trace-bundle .verdict.ok {{ background:var(--green); color:#fff; }}
.trace-bundle .verdict.bad {{ background:#b91c1c; color:#fff; }}
.trace-bundle .trace-q {{ font-size:17px; color:var(--dark); margin:6px 0 14px; }}
.trace-bundle .trace-art {{ margin:12px 0; }}
.trace-bundle .art-label {{ font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px; }}
.trace-bundle img.strip {{ width:100%; max-width:1100px; height:auto; border:1px solid #cbd5e1; border-radius:6px; cursor:zoom-in; }}
.trace-bundle img.gif {{ max-width:640px; height:auto; border:1px solid #cbd5e1; border-radius:6px; }}
.trace-bundle img.ui {{ max-width:1100px; width:100%; height:auto; border:1px solid #cbd5e1; border-radius:6px; }}
.trace-bundle img.reasoning {{ max-width:900px; width:100%; height:auto; border:1px solid #cbd5e1; border-radius:6px; }}
.trace-bundle .frame-links {{ font-size:12px; color:#64748b; margin-top:4px; }}
.trace-bundle .frame-links a {{ color:var(--green); margin-right:8px; text-decoration:none; }}
.trace-bundle .frame-links a:hover {{ text-decoration:underline; }}
.trace-bundle .trace-cap {{ font-size:13px; color:#64748b; margin-top:10px; padding-top:10px; border-top:1px solid #e5e7eb; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.trace-bundle .miss {{ color:#b91c1c; font-size:14px; }}
</style></head><body>
<section class='slide' style='background:var(--dark); color:#fff;'>
  <div style='border-left:6px solid var(--green); padding-left:20px;'>
    <h1 style='font-size:44px; margin:0 0 12px; color:#fff;'>Cosmos3-Super-Reasoner on LingoQA</h1>
    <div style='font-size:24px; color:var(--green);'>Visual Question Answering for Autonomous Driving — {total}-row benchmark</div>
    <div style='font-size:16px; opacity:0.7; margin-top:24px;'>Run: {_esc(run_id)} · {_esc(time.strftime('%Y-%m-%d', time.localtime(snap.get('started_epoch') or time.time())))}</div>
    <div style='font-size:14px; opacity:0.5; margin-top:8px;'>Benchmark: Marcu et al., arXiv:2312.14115 (Wayve, 2024)</div>
  </div>
</section>

<section class='slide'>
  <h2 class='slide-title'>TL;DR</h2>
  <div class='tile-grid'>
    <div class='tile'><h4>Headline</h4><div class='val'>{overall * 100:.1f}%</div><div>Lingo-Judge accuracy on {total} predictions. Human ceiling 96.6%, LingoQA Baseline 60.8%, GPT-4V 59.6%.</div></div>
    <div class='tile'><h4>Reasoning behavior</h4><div class='val'>{(summary.get('average_answer_words') or 0):.1f} words</div><div>Avg answer length. Reasoning traces on {(summary.get('reasoning_trace_pct') or 0) * 100:.1f}% of questions.</div></div>
    <div class='tile'><h4>Per-category spread</h4><div class='val'>{(max([c['accuracy'] for c in per_cat] + [0]) - min([c['accuracy'] for c in per_cat] + [1])) * 100:.1f} pts</div><div>Gap between best and worst competency.</div></div>
  </div>
</section>

<section class='slide'>
  <h2 class='slide-title'>Headline Metric — Lingo-Judge</h2>
  <div class='hero'>{overall * 100:.1f}%</div>
  <div class='hero-sub'>{correct}/{total} predictions classified correct</div>
  <div class='definition'>Lingo-Judge is a DeBERTa-v3 classifier fine-tuned with LoRA. Score = max F_Judge(prediction, GT_j) over j in {{0,1}}. Judge does not see the images — pure text classification against human-written labels.</div>
  <div class='ref-bar'>
    <div class='pill'>Human (multi-frame) 96.6</div>
    <div class='pill'>Human (single-frame) 81.8</div>
    <div class='pill us'>This run {overall * 100:.1f}</div>
    <div class='pill'>LingoQA Baseline 60.8</div>
    <div class='pill'>GPT-4V 59.6</div>
    <div class='pill'>LLaVA-FT 59.0</div>
    <div class='pill'>BLIP-2-FT 52.2</div>
  </div>
</section>

<section class='slide'>
  <h2 class='slide-title'>Per-Category Breakdown</h2>
  <div>{bars_html}</div>
</section>

<section class='slide'>
  <h2 class='slide-title'>Qualitative Examples</h2>
  <table class='qual'><thead><tr><th>Question</th><th>Label</th><th>Prediction</th><th>L-J Prob.</th><th>L-J Class.</th></tr></thead>
  <tbody>{qualitative_rows}</tbody></table>
</section>

<section class='slide'>
  <h2 class='slide-title'>Reasoning Trace Examples</h2>
  {reasoning_blocks or '<p>No reasoning traces emitted in this run.</p>'}
</section>

<section class='slide'>
  <h2 class='slide-title'>Traceability Bundles — input frames, GIF, UI card, reasoning trace</h2>
  <p class='slide-sub'>Each worked example shows the model's actual input (5 frames + animated proxy), the live Benchmark View card, and the reasoning trace as embeddable artifacts. Click any frame strip to open the source JPGs at full resolution.</p>
  {traceability_html}
</section>

<section class='slide'>
  <h2 class='slide-title'>Methodology</h2>
  <div class='kv'>
    <div>Model</div><div>{_esc(model)}</div>
    <div>Served via</div><div>NIM OpenAI-compat /v1/chat/completions ({_esc(snap.get('base_url') or '')})</div>
    <div>Message shape</div><div>build.nvidia.com canonical: one user message, content = image_url[] (base64 data: URL) + text. No max_tokens.</div>
    <div>Frame count</div><div>5 frames per question (LingoQA uniform)</div>
    <div>Eval set</div><div>LingoQA val.parquet, 500 unique questions × 2 references = 1000 rows</div>
    <div>Metric</div><div>{_esc(judge)} (DeBERTa-v3-base, sigmoid &gt; 0.5)</div>
    <div>Total wall</div><div>{(snap.get('wall_seconds') or 0):.1f}s ({total} predictions, avg {(summary.get('average_latency_seconds') or 0):.2f}s/pred)</div>
  </div>
</section>

<section class='slide'>
  <h2 class='slide-title'>Judge Calibration</h2>
  <p>Lingo-Judge (paper Table 1): 95.0% validation accuracy, 0.950 Spearman, 0.993 Pearson — measured against human evaluators across 17 candidate models.</p>
  <p>Known failure mode: judge over-rates long-form incorrect answers (FUYU 45.4% judge vs 17.69% human). This run averages {(summary.get('average_answer_words') or 0):.1f} words/answer — flag if &gt; 30.</p>
</section>

<section class='slide'>
  <h2 class='slide-title'>Limitations</h2>
  <ul>
    <li>Zero-shot: model was not fine-tuned on LingoQA training data — peer is the paper's zero-shot row, not fine-tuned baseline.</li>
    <li>1,000 rows total (500 questions × 2 refs); ±3% per-category swings are noise.</li>
    <li>English-only, UK driving scenes, front-camera-only.</li>
    <li>Judge is out-of-distribution for reasoning-model answer styles; results should be spot-audited for verbose over-rating.</li>
  </ul>
</section>

<section class='slide'>
  <h2 class='slide-title'>Appendix — Reproducibility</h2>
  <div class='kv'>
    <div>Run id</div><div>{_esc(run_id)}</div>
    <div>NIM endpoint</div><div>{_esc(snap.get('base_url') or '')}</div>
    <div>Model id</div><div>{_esc(model)}</div>
    <div>Sample size</div><div>{_esc(snap.get('sample_size'))}</div>
    <div>Seed</div><div>{_esc(snap.get('seed'))}</div>
    <div>Started</div><div>{_esc(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snap.get('started_epoch') or time.time())))}</div>
    <div>Results JSON</div><div>/benchmark/results/{_esc(run_id)}.json</div>
  </div>
</section>
</body></html>"""


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cosmos BYO Video Batch Inference</title>
<meta name="description" content="Cosmos BYO Video Batch Inference with Basic View and LingoQA Benchmark View" />
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
/* Tab switcher (Basic / Benchmark) */
.tabs { display:flex; gap:0; padding:0 28px; border-bottom:1px solid var(--line); background:#fff; }
.tab-btn { background:transparent; color:#475569; border:0; border-bottom:3px solid transparent; padding:14px 22px; font-weight:650; cursor:pointer; font-size:14px; border-radius:0; }
.tab-btn.active { color:#76B900; border-bottom-color:#76B900; }
.tab-pane { display:none; }
.tab-pane.active { display:block; }
/* Benchmark View */
.bench-wrap { width:min(100% - 32px, 1680px); margin:0 auto; padding:24px 0; }
.bench-grid { display:grid; grid-template-columns:repeat(2, minmax(320px, 1fr)); gap:18px; }
.bench-card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px; }
.bench-card h3 { margin:0 0 12px; font-size:14px; color:#475569; text-transform:uppercase; letter-spacing:0.04em; }
.bench-card label { font-size:12px; color:#64748b; }
.bench-card select, .bench-card input { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:6px; font:inherit; }
.bench-section { margin-top:18px; background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px; }
.bench-section h3 { margin:0 0 14px; font-size:14px; color:#475569; text-transform:uppercase; letter-spacing:0.04em; }
.bench-hero { display:flex; align-items:center; gap:24px; flex-wrap:wrap; }
.bench-hero .num { font-size:64px; font-weight:800; color:#76B900; line-height:1; }
.bench-hero .ref-row { display:flex; flex-wrap:wrap; gap:8px; }
.bench-hero .ref-row span { background:#f8fafc; border:1px solid var(--line); border-radius:999px; padding:4px 12px; font-size:12px; color:#475569; }
.cat-chips { display:flex; flex-wrap:wrap; gap:8px; }
.cat-chip { background:#fff; border:1px solid var(--line); border-radius:999px; padding:6px 12px; font-size:12px; }
.cat-chip strong { color:#1A1A1A; }
.cat-chip.high { border-color:#76B900; background:#F0F5E8; }
.cat-chip.low { border-color:#FECACA; background:#FEF2F2; }
.bench-progress { height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; margin:8px 0; }
.bench-progress div { height:100%; background:#76B900; width:0; transition:width .2s ease; }
.sample-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:12px; }
.sample-card { border:1px solid var(--line); border-radius:8px; padding:10px; background:#f8fafc; cursor:pointer; transition:border-color .15s; }
.sample-card:hover { border-color:#76B900; }
.sample-card .verdict { font-weight:700; }
.sample-card.ok { border-left:4px solid #76B900; }
.sample-card.bad { border-left:4px solid #DC2626; }
.sample-card .q { font-size:13px; margin:6px 0; color:#1A1A1A; }
.sample-card .pred { font-size:12px; color:#475569; }
.sample-card img { width:100%; height:120px; object-fit:cover; border-radius:4px; background:#e5e7eb; }
/* Inline frame thumbnails on Recent Samples cards (Phase D — Lingo-Judge traceability) */
.sample-card .frame-thumbnails { display:flex; gap:3px; margin:6px 0 8px; overflow:hidden; }
.sample-card .frame-thumbnails .frame-thumbnail { width:80px; height:45px; object-fit:cover; border-radius:3px; background:#e5e7eb; border:1px solid #cbd5e1; display:block; flex:0 0 auto; }
.sample-card .frame-thumbnails .frame-thumbnail:hover { border-color:#76B900; }
.sample-card .frame-thumbnails .image-tag { font-size:11px; color:#475569; padding:2px 6px; background:#e5e7eb; border-radius:3px; }
.sample-card .details-toggle { font-size:11px; color:#475569; margin-top:6px; cursor:pointer; user-select:none; }
.sample-card .details-toggle:hover { color:#76B900; }
.sample-card .trunc-flag { display:inline-block; font-size:10px; padding:1px 5px; border-radius:3px; background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; margin-left:4px; vertical-align:middle; }
/* Per-row detail panel inside the modal */
.detail-frame-strip { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0; }
.detail-frame-strip figure { margin:0; }
.detail-frame-strip img { width:160px; height:90px; object-fit:cover; border-radius:4px; border:1px solid #cbd5e1; display:block; }
.detail-frame-strip figcaption { font-size:11px; color:#475569; text-align:center; margin-top:2px; }
.judge-breakdown { background:#F0F5E8; border:1px solid #c5e0a6; border-radius:6px; padding:12px; margin-top:12px; }
.judge-breakdown h4 { margin:0 0 8px; font-size:13px; color:#1A1A1A; text-transform:uppercase; letter-spacing:0.04em; }
.judge-breakdown .ref-block { background:#fff; border:1px solid var(--line); border-radius:4px; padding:8px 10px; margin-bottom:8px; }
.judge-breakdown .ref-block.winning { border-left:4px solid #76B900; }
.judge-breakdown .ref-block .ref-meta { font-size:12px; color:#475569; margin-bottom:4px; }
.judge-breakdown pre { background:#0f172a; color:#d1fae5; padding:8px; border-radius:4px; font-size:11px; max-height:160px; overflow:auto; white-space:pre-wrap; margin:4px 0; font-family:ui-monospace,Menlo,monospace; }
.judge-breakdown .trunc-marker { color:#fbbf24; font-weight:700; }
.trunc-callout { background:#FEF3C7; border:1px solid #FDE68A; border-left:4px solid #F59E0B; border-radius:4px; padding:10px 12px; margin:10px 0; font-size:13px; color:#78350F; }
.trunc-callout strong { color:#92400E; }
.trunc-summary-callout { background:#FEF3C7; border:1px solid #FDE68A; border-left:4px solid #F59E0B; border-radius:6px; padding:10px 14px; margin:10px 0; font-size:13px; color:#78350F; }
.provenance-footer { font-size:11px; color:#94a3b8; margin-top:12px; padding-top:8px; border-top:1px solid var(--line); font-family:ui-monospace,Menlo,monospace; }
.provenance-footer a { color:#76B900; text-decoration:none; }
.provenance-footer a:hover { text-decoration:underline; }
.bench-shape { background:#0f172a; color:#d1fae5; padding:12px; border-radius:6px; font-family:ui-monospace,Menlo,monospace; font-size:11px; white-space:pre-wrap; max-height:220px; overflow:auto; margin-top:8px; }
.bench-modal { display:none; position:fixed; inset:0; background:rgba(15,23,42,0.6); z-index:1000; align-items:center; justify-content:center; padding:24px; }
.bench-modal.open { display:flex; }
.bench-modal-inner { background:#fff; max-width:900px; max-height:90vh; overflow:auto; border-radius:12px; padding:24px; }
.bench-modal pre { background:#F0F5E8; padding:12px; border-radius:6px; white-space:pre-wrap; font-size:13px; max-height:240px; overflow:auto; }
/* Drop-zone layout */
.bench-lastrun { display:flex; justify-content:space-between; align-items:center; gap:14px; background:#F0F5E8; border:1px solid #c5e0a6; border-left:4px solid #76B900; border-radius:8px; padding:12px 16px; margin-bottom:14px; font-size:14px; color:#1A1A1A; }
.bench-lastrun-actions { display:flex; gap:8px; align-items:center; }
.bench-dropzone { background:#fff; border:1px solid var(--line); border-radius:12px; padding:24px; margin-bottom:18px; }
.bench-dropzone-row { display:flex; align-items:center; gap:12px; margin-bottom:12px; flex-wrap:wrap; }
.bench-dropzone-row:last-of-type { margin-bottom:0; }
.bench-url-input { flex:1 1 auto; min-width:260px; padding:14px 16px; font-size:15px; border:2px solid var(--line); border-radius:8px; font:inherit; transition:border-color .15s; }
.bench-url-input:focus { outline:none; border-color:#76B900; }
.bench-chip { background:#F0F5E8; border:1px solid #76B900; color:#1A1A1A; border-radius:999px; padding:6px 12px; font-size:12px; font-weight:600; white-space:nowrap; }
.bench-chip.unknown { background:#FEF2F2; border-color:#FECACA; color:#7f1d1d; }
.bench-model-select { flex:1 1 auto; min-width:260px; padding:12px 14px; font-size:14px; border:1px solid var(--line); border-radius:8px; font:inherit; background:#fff; }
.bench-primary-btn { flex:1 1 auto; background:#76B900; color:#fff; border:none; border-radius:8px; padding:16px 24px; font-size:16px; font-weight:700; cursor:pointer; transition:background .15s; }
.bench-primary-btn:hover { background:#5d9300; }
.bench-primary-btn:disabled { background:#94a3b8; cursor:not-allowed; }
.bench-ghost-row { gap:8px; }
.bench-ghost-btn { background:#fff; color:#475569; border:1px solid var(--line); border-radius:6px; padding:8px 14px; font-size:13px; cursor:pointer; transition:all .15s; }
.bench-ghost-btn:hover:not(:disabled) { border-color:#76B900; color:#1A1A1A; }
.bench-ghost-btn:disabled { opacity:0.5; cursor:not-allowed; }
.bench-ghost-link { color:#76B900; text-decoration:none; font-size:13px; font-weight:600; padding:6px 10px; border-radius:6px; border:1px solid transparent; }
.bench-ghost-link:hover { border-color:#76B900; }
.bench-dropzone-status { margin:8px 0 0; min-height:18px; }
.bench-advanced { background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:18px; margin-bottom:18px; }
.bench-history { background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px; margin-bottom:18px; }
.bench-history-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px; }
.bench-history-head h3 { margin:0; font-size:14px; color:#475569; text-transform:uppercase; letter-spacing:0.04em; }
.bench-history-rows { display:flex; flex-direction:column; gap:6px; }
.bench-history-row { display:grid; grid-template-columns:1.2fr 1fr 1.2fr 0.6fr auto; gap:12px; align-items:center; padding:10px 12px; border:1px solid var(--line); border-radius:6px; background:#f8fafc; font-size:13px; cursor:pointer; transition:border-color .15s; }
.bench-history-row:hover { border-color:#76B900; background:#F0F5E8; }
.bench-history-row .acc { font-weight:700; color:#76B900; }
.bench-history-row .actions { display:flex; gap:6px; }
.bench-history-empty { color:#94a3b8; font-size:13px; padding:12px; text-align:center; }
@media (min-width: 1440px) { main { width:calc(100% - 40px); } .bench-wrap { width:calc(100% - 40px); } }
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
<div class="tabs" role="tablist">
  <button class="tab-btn active" data-tab="basic" role="tab">Basic View</button>
  <button class="tab-btn" data-tab="benchmark" role="tab">Benchmark View</button>
</div>
<div id="basicPane" class="tab-pane active">
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
</div><!-- end basicPane -->
<div id="benchmarkPane" class="tab-pane">
  <div class="bench-wrap">
    <!-- Drop-zone layout: 2 inputs + 1 primary button. Power-user knobs moved
         into the collapsible Advanced panel below; everything still works. -->
    <div id="benchLastRunTile" class="bench-lastrun" style="display:none;">
      <span id="benchLastRunText">Last run: --</span>
      <span class="bench-lastrun-actions">
        <a id="benchLastRunReport" class="bench-ghost-link" href="#" target="_blank">Open Report</a>
        <button id="benchLastRunRerun" class="bench-ghost-btn">Re-run</button>
      </span>
    </div>

    <div class="bench-dropzone">
      <div class="bench-dropzone-row">
        <input id="benchUrlInput" class="bench-url-input"
               placeholder="Paste a benchmark URL — HuggingFace dataset, Arxiv paper, GitHub repo, or GDrive folder (or just type &quot;lingoqa&quot;)"
               autocomplete="off" spellcheck="false" />
        <span id="benchUrlChip" class="bench-chip" style="display:none;"></span>
      </div>
      <div class="bench-dropzone-row">
        <select id="benchModelSelect" class="bench-model-select">
          <option value="">Loading models...</option>
        </select>
      </div>
      <div class="bench-dropzone-row">
        <button id="benchRunBtn" class="bench-primary-btn">▶ Run Benchmark</button>
      </div>
      <div class="bench-dropzone-row bench-ghost-row">
        <button id="benchRerunLastBtn" class="bench-ghost-btn" disabled>↻ Re-run last</button>
        <button id="benchToggleAdvBtn" class="bench-ghost-btn" aria-expanded="false">⚙ Advanced</button>
        <button id="benchToggleHistBtn" class="bench-ghost-btn" aria-expanded="false">≡ Run history</button>
      </div>
      <p class="bench-dropzone-status hint" id="benchDropzoneStatus"></p>
    </div>

    <!-- Advanced panel: the prior 5-knob form. Collapsed by default. -->
    <div id="benchAdvanced" class="bench-advanced" style="display:none;">
      <div class="bench-grid">
        <div class="bench-card">
          <h3>Dataset</h3>
          <label>Source</label>
          <select id="benchDataset"><option value="lingoqa-official">LingoQA (official, GDrive local cache)</option></select>
          <label style="margin-top:10px;">Resolved from URL</label>
          <input id="benchResolvedSource" readonly value="" placeholder="(set automatically when you paste a URL above)" />
          <p class="hint" id="benchDatasetStatus">Probing dataset...</p>
        </div>
        <div class="bench-card">
          <h3>Eval Method</h3>
          <label>Judge</label>
          <select id="benchJudge"><option value="lingo-judge">Lingo-Judge (DeBERTa-v3-base, Wayve)</option></select>
          <p class="hint" id="benchJudgeStatus">Probing judge...</p>
          <label style="margin-top:10px;">Judge mode</label>
          <div id="benchJudgeMode" style="display:flex; flex-direction:column; gap:4px; font-size:13px;">
            <label><input type="radio" name="benchJudgeMode" value="standard" checked /> Standard <span style="color:#94a3b8;">— judge sees full prediction (incl. &lt;think&gt;)</span></label>
            <label><input type="radio" name="benchJudgeMode" value="answer-only" /> Answer-only <span style="color:#94a3b8;">— strip &lt;think&gt; before judging</span></label>
            <label><input type="radio" name="benchJudgeMode" value="both" /> Both (A/B) <span style="color:#94a3b8;">— score twice; surface delta</span></label>
          </div>
          <p class="hint">Brief gt-A/gt-B references favor short answers. Long &lt;think&gt; chains can push the judge past its 512-token window. Use "Both" to A/B the bias.</p>
        </div>
        <div class="bench-card">
          <h3>Model (auto-detected)</h3>
          <label>Served model id</label>
          <input id="benchModel" readonly value="probing /v1/models..." />
          <p class="hint">Auto-detected from the live NIM /v1/models endpoint. Never hardcoded.</p>
          <details><summary>Run Anywhere shape (build.nvidia.com canonical)</summary>
          <pre class="bench-shape">POST /v1/chat/completions
{
  "model": "&lt;served_id&gt;",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,&lt;b64_frame_1&gt;"}},
      ...
      {"type": "text", "text": "&lt;LingoQA question&gt;"}
    ]
  }]
  /* NO max_tokens, NO temperature, NO top_p — Alex standing order */
}</pre></details>
        </div>
        <div class="bench-card">
          <h3>Run Configuration</h3>
          <label>Sample size: <span id="benchSampleSizeVal">1000</span></label>
          <input id="benchSampleSize" type="range" min="10" max="1000" step="10" value="1000" />
          <label style="margin-top:10px;">Concurrency: <span id="benchConcurrencyVal">8</span></label>
          <input id="benchConcurrency" type="range" min="1" max="16" step="1" value="8" />
          <label style="margin-top:10px;">Seed: <span id="benchSeedVal">42</span></label>
          <input id="benchSeed" type="number" value="42" style="font:inherit;" />
          <div class="actions" style="margin-top:14px;">
            <button class="secondary" id="benchSmokeBtn">Smoke (5 rows)</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Run history modal: last 20 runs, persistent on disk. -->
    <div id="benchHistoryPanel" class="bench-history" style="display:none;">
      <div class="bench-history-head">
        <h3>Run history</h3>
        <span class="hint">Last 20 runs · click any row to re-run with the same params</span>
      </div>
      <div id="benchHistoryRows" class="bench-history-rows"></div>
    </div>

    <div class="bench-section">
      <h3>Live results</h3>
      <div class="bench-progress"><div id="benchBar"></div></div>
      <div class="kv" id="benchProgressKv" style="margin-bottom:12px;"></div>
      <div class="bench-hero">
        <div>
          <div class="num" id="benchAccuracy">--</div>
          <div style="font-size:14px; color:#475569;">Lingo-Judge Accuracy</div>
        </div>
        <div class="ref-row">
          <span>Human (multi-frame) 96.6</span>
          <span>Human (single-frame) 81.8</span>
          <span>LingoQA Baseline 60.8</span>
          <span>GPT-4V 59.6</span>
          <span>LLaVA-FT 59.0</span>
        </div>
      </div>
      <div style="margin-top:18px;">
        <h4 style="margin:0 0 8px; font-size:13px; color:#475569;">Per-category accuracy</h4>
        <div class="cat-chips" id="benchCats"></div>
      </div>
    </div>

    <div class="bench-section">
      <h3>Recent samples</h3>
      <p class="hint">Click a sample to expand the reasoning trace, frame strip, and judge breakdown. Frame thumbnails are clickable for full-res.</p>
      <div id="benchTruncStats"></div>
      <div class="sample-grid" id="benchSamples"></div>
    </div>

    <div class="bench-section">
      <h3>Report</h3>
      <p class="hint">Live arxiv-2312.14115-style report. Opens once a run completes (or partially mid-run).</p>
      <div class="actions">
        <a id="benchReportLink" class="export-link" href="#" target="_blank">Open report</a>
        <a id="benchJsonLink" class="export-link" href="#" target="_blank">Download results.json</a>
      </div>
    </div>
  </div>
  <div id="benchModal" class="bench-modal" role="dialog">
    <div class="bench-modal-inner">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <h3 id="benchModalTitle" style="margin:0;">Sample detail</h3>
        <button class="secondary" id="benchModalClose">Close</button>
      </div>
      <div id="benchModalBody"></div>
    </div>
  </div>
</div><!-- end benchmarkPane -->
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

// ====================================================================
// Benchmark View
// ====================================================================
let benchRunId = null;
let benchPollTimer = null;
let benchLastResults = [];

function benchSwitchTab(name){
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active', b.dataset.tab===name));
  document.getElementById('basicPane').classList.toggle('active', name==='basic');
  document.getElementById('benchmarkPane').classList.toggle('active', name==='benchmark');
  if(name==='benchmark'){
    benchProbeDataset(); benchProbeJudge(); benchProbeModel();
    benchLoadModels(); benchLoadHistory();
  }
}
document.querySelectorAll('.tab-btn').forEach(b=>{ b.onclick = ()=>benchSwitchTab(b.dataset.tab); });

async function benchProbeDataset(){
  try{
    const r = await fetch('/benchmark/datasets'); const j = await r.json();
    const lq = (j||[])[0] || {};
    el('benchDatasetStatus').textContent = lq.ready ? `LingoQA ready at ${lq.path} (1000 rows)` : 'LingoQA data not staged on this host. scp the dataset to /home/horde/lingoqa-data/.';
  }catch(e){ el('benchDatasetStatus').textContent = 'Probe failed: '+e.message; }
}
async function benchProbeJudge(){
  try{
    const r = await fetch('/benchmark/judges'); const j = await r.json();
    const lj = (j||[])[0] || {};
    el('benchJudgeStatus').textContent = lj.loaded ? 'Lingo-Judge loaded (DeBERTa-v3-base)' : 'Lingo-Judge will lazy-load on first run (~600 MB download, ~30s warm-up).';
  }catch(e){ el('benchJudgeStatus').textContent = 'Probe failed: '+e.message; }
}
async function benchProbeModel(){
  try{
    const r = await fetch('/benchmark/model'); const j = await r.json();
    el('benchModel').value = j.model || ('error: '+(j.error||'unknown'));
  }catch(e){ el('benchModel').value = 'probe failed: '+e.message; }
}

function benchClipText(s, max=180){ s=String(s||''); return s.length>max ? s.slice(0,max-1)+'...' : s; }
function benchPct(v){ return (Number(v||0)*100).toFixed(1)+'%'; }

function benchRenderSnap(snap){
  const summary = snap.summary || {};
  const progress = snap.progress || {done:0,total:0,errors:0};
  const total = progress.total || snap.total || 0;
  const done = progress.done || 0;
  const pct = total ? Math.round(100*done/total) : 0;
  el('benchBar').style.width = pct+'%';
  const elapsed = snap.started_epoch ? (((snap.finished_epoch||Date.now()/1000) - snap.started_epoch)) : 0;
  const eta = (done>0 && total>done) ? ((total-done) * (elapsed/done)) : null;
  const etaStr = eta ? (eta>60 ? `${Math.floor(eta/60)}m ${Math.round(eta%60)}s` : Math.round(eta)+'s') : '';
  const rows = [
    ['run id', snap.run_id||''],
    ['status', snap.status||''],
    ['model', snap.model||''],
    ['progress', `${done}/${total} (${pct}%)`],
    ['errors', String(progress.errors||0)],
    ['elapsed', Math.round(elapsed)+'s'],
    ['ETA', etaStr],
    ['avg latency', summary.average_latency_seconds ? summary.average_latency_seconds.toFixed(2)+'s' : ''],
    ['reasoning trace', summary.reasoning_trace_pct!==undefined ? benchPct(summary.reasoning_trace_pct) : ''],
  ];
  el('benchProgressKv').innerHTML = rows.filter(([_,v])=>v).map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v)}</div>`).join('');
  const acc = summary.overall_accuracy;
  el('benchAccuracy').textContent = acc !== undefined && acc !== null ? (acc*100).toFixed(1)+'%' : '--';
  // Per-category chips
  const cats = summary.per_category || [];
  const chipsHtml = cats.map(c=>{
    const a = c.accuracy||0;
    const cls = a>=0.7 ? 'high' : (a<=0.3 ? 'low' : '');
    return `<div class="cat-chip ${cls}"><strong>${esc(c.category)}</strong> ${(a*100).toFixed(1)}% (n=${c.total})</div>`;
  }).join('');
  el('benchCats').innerHTML = chipsHtml || '<div class="hint" style="color:#94a3b8;">no completed predictions yet</div>';
  // Aggregate truncation callout — surface when a meaningful fraction of rows
  // would have been clipped by the judge's 512-token window. >5% suggests a
  // re-judge sprint with shortened prompts is warranted.
  const truncRows = summary.judge_truncated_rows || 0;
  const truncPct = summary.judge_truncated_pct || 0;
  const totalRows = summary.total_predictions || 0;
  const truncEl = el('benchTruncStats');
  if(truncEl){
    let html = '';
    if(totalRows > 0 && truncRows > 0){
      const pctStr = (truncPct*100).toFixed(1);
      const warn = truncPct > 0.05;
      html += warn
        ? `<div class="trunc-summary-callout"><strong>⚠️ ${truncRows} / ${totalRows} (${pctStr}%) judge inputs truncated</strong> at ${summary.judge_max_tokens||512} tokens. Reasoning-style models with long &lt;think&gt; chains may be under-rated — consider a shortened-prompt re-judge sprint as follow-up.</div>`
        : `<div class="hint" style="color:#94a3b8; font-size:12px;">Judge truncation: ${truncRows} / ${totalRows} rows (${pctStr}%) exceeded ${summary.judge_max_tokens||512} tokens.</div>`;
    }
    // A/B summary tile — only when in "both" mode (both accuracies computed).
    if(summary.accuracy_standard !== null && summary.accuracy_standard !== undefined
       && summary.accuracy_answer_only !== null && summary.accuracy_answer_only !== undefined){
      const aStd = (summary.accuracy_standard*100).toFixed(1);
      const aAo  = (summary.accuracy_answer_only*100).toFixed(1);
      const delta = summary.accuracy_delta_pp;
      const deltaStr = (delta>=0?'+':'') + delta.toFixed(1) + 'pp';
      const aoTruncPct = (summary.judge_truncated_pct_answer_only||0)*100;
      const aoTruncRows = summary.judge_truncated_rows_answer_only || 0;
      const divergent = summary.divergent_rows || 0;
      const tileBg = Math.abs(delta) > 5 ? '#FEF3C7' : '#F0F5E8';
      const tileBorder = Math.abs(delta) > 5 ? '#F59E0B' : '#76B900';
      html += `<div style="background:${tileBg};border:1px solid ${tileBorder};border-left:4px solid ${tileBorder};border-radius:6px;padding:10px 14px;margin:8px 0;font-size:13px;">
        <strong>Judge mode A/B:</strong>
        Standard accuracy: <strong>${aStd}%</strong> ·
        Answer-only accuracy: <strong>${aAo}%</strong> ·
        Δ: <strong>${deltaStr}</strong> ·
        Divergent rows: ${divergent} / ${totalRows} ·
        Answer-only truncation: ${aoTruncRows} (${aoTruncPct.toFixed(1)}%)
      </div>`;
      if(Math.abs(delta) > 5){
        html += `<div class="trunc-summary-callout"><strong>Reasoning model: judge-input format may be biasing the score by ${deltaStr}.</strong> The answer-only mode strips &lt;think&gt; before scoring; a Δ &gt; 5pp suggests the long reasoning chain is materially affecting judge verdicts (positive Δ = answer-only scored higher = the reasoning chain was hurting the verdict).</div>`;
      }
    }
    truncEl.innerHTML = html;
  }
  // Sample grid — Phase D: inline frame thumbnails on every row so the user
  // can SEE what the model + judge worked from. LingoQA rows have segment_id
  // and use /lingoqa-frames/<seg>/<idx>.jpg; non-LingoQA rows fall back to a
  // tiny "📷 N images" tag (or first 1-3 inline images if accessible).
  const recent = snap.recent_results || [];
  benchLastResults = recent.slice();
  el('benchSamples').innerHTML = recent.map((r, idx)=>{
    // Color the card border based on STD verdict (current behavior). In
    // "both" mode the ans-only verdict is shown as a small extra dot.
    const stdCorrect = (r.judge_correct_standard !== null && r.judge_correct_standard !== undefined)
      ? r.judge_correct_standard : r.judge_correct;
    const cls = stdCorrect ? 'ok' : 'bad';
    const hasBoth = (r.judge_score_standard !== null && r.judge_score_standard !== undefined)
                 && (r.judge_score_answer_only !== null && r.judge_score_answer_only !== undefined);
    let verdict, score;
    if(r.error){ verdict = 'ERROR'; score = ''; }
    else if(hasBoth){
      const stdPct = (r.judge_score_standard*100).toFixed(0)+'%';
      const aoPct  = (r.judge_score_answer_only*100).toFixed(0)+'%';
      const stdMark = r.judge_correct_standard ? '✅' : '❌';
      const aoMark  = r.judge_correct_answer_only ? '✅' : '❌';
      const aoDotColor = r.judge_correct_answer_only ? '#76B900' : '#DC2626';
      const aoDot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${aoDotColor};margin:0 4px;" title="answer-only verdict"></span>`;
      verdict = `${stdMark} ${stdPct} std ${aoDot}/ ${aoMark} ${aoPct} ans-only`;
      score = '';
    } else {
      verdict = r.judge_correct ? '✅ correct' : '❌ miss';
      score = r.judge_score!==null && r.judge_score!==undefined ? (r.judge_score*100).toFixed(0)+'%' : '';
    }
    const truncFlag = (r.judge_truncated_a || r.judge_truncated_b)
      ? '<span class="trunc-flag" title="Judge input (standard mode) was truncated to 512 tokens — final answer may not have reached the judge.">⚠️ trunc</span>'
      : '';
    let thumbs = '';
    if(r.segment_id){
      const seg = encodeURIComponent(r.segment_id);
      const t = [];
      for(let i=0;i<5;i++){
        t.push(`<a href="/lingoqa-frames/${seg}/${i}.jpg" target="_blank" onclick="event.stopPropagation()"><img loading="lazy" class="frame-thumbnail" src="/lingoqa-frames/${seg}/${i}.jpg" alt="frame ${i}"/></a>`);
      }
      thumbs = `<div class="frame-thumbnails">${t.join('')}</div>`;
    } else if(Array.isArray(r.images) && r.images.length){
      // Non-LingoQA datasets: just show an image count tag (paths are local
      // disk and not currently served — future hook for /benchmark/artifact).
      thumbs = `<div class="frame-thumbnails"><span class="image-tag">📷 ${r.images.length} image${r.images.length>1?'s':''}</span></div>`;
    }
    return `<div class="sample-card ${cls}" data-idx="${idx}">
      <div class="verdict">${verdict} ${truncFlag} <span style="float:right; color:#475569;">${score}</span></div>
      <div class="q">${esc(benchClipText(r.question,90))}</div>
      ${thumbs}
      <div class="pred"><strong>pred:</strong> ${esc(benchClipText(r.prediction||r.error||'',120))}</div>
      <div class="pred"><strong>gt:</strong> ${esc(benchClipText((r.references||[''])[0],120))}</div>
      <div class="details-toggle">▾ Details</div>
    </div>`;
  }).join('') || '<div class="hint" style="color:#94a3b8;">no samples yet</div>';
  // Wire detail modal clicks. Anchors inside thumbnails stop propagation so a
  // click on a frame opens the full-res jpg without also opening the modal.
  document.querySelectorAll('.sample-card').forEach(c=>{ c.onclick = ()=>benchOpenSample(Number(c.dataset.idx)); });
  // Report + download links
  if(snap.run_id){
    el('benchReportLink').href = '/benchmark/report/'+snap.run_id;
    el('benchJsonLink').href = '/benchmark/results/'+snap.run_id+'.json';
  }
}

function benchOpenSample(idx){
  const r = benchLastResults[idx]; if(!r) return;
  el('benchModalTitle').textContent = r.judge_correct ? 'Correct: '+(r.category||'') : 'Miss: '+(r.category||'');
  // ---- Full 5-frame strip (LingoQA rows; lazy + clickable to full-res) ----
  let frameStripHtml = '';
  if(r.segment_id){
    const seg = encodeURIComponent(r.segment_id);
    const figs = [];
    for(let i=0;i<5;i++){
      figs.push(`<figure><a href="/lingoqa-frames/${seg}/${i}.jpg" target="_blank"><img loading="lazy" src="/lingoqa-frames/${seg}/${i}.jpg" alt="Frame ${i}"/></a><figcaption>Frame ${i}</figcaption></figure>`);
    }
    frameStripHtml = `<h4 style="margin-top:0;">Input frames</h4><div class="detail-frame-strip">${figs.join('')}</div>`;
  } else if(Array.isArray(r.images) && r.images.length){
    frameStripHtml = `<h4 style="margin-top:0;">Input</h4><div class="hint" style="font-size:12px;color:#94a3b8;">${r.images.length} image${r.images.length>1?'s':''} (paths only, see provenance footer).</div>`;
  }
  // ---- Token-count + truncation surfacing ----
  const reasoningText = r.reasoning_trace || '';
  const finalAnswer = r.prediction || '';
  // Char-based estimate of (think + answer) tokens — directional only.
  const combinedChars = reasoningText.length + finalAnswer.length;
  const approxTokens = Math.max(1, Math.floor(combinedChars / 4));
  const maxTok = r.judge_max_tokens || 512;
  const overMax = approxTokens > maxTok;
  const tokFlag = overMax ? ' <span class="trunc-flag">⚠️ TRUNCATED BY JUDGE</span>' : '';
  // ---- Judge breakdown per reference ----
  const refs = r.references || [];
  const winIdx = (r.judge_winning_reference_index !== undefined && r.judge_winning_reference_index !== null) ? r.judge_winning_reference_index : -1;
  function refBlock(label, refText, refIdx, judgeTextRaw, nTokens, truncated){
    if(!refText && refText !== '') return '';
    const isWin = refIdx === winIdx;
    const truncMarker = truncated ? '<span class="trunc-marker">↓ truncated here at '+maxTok+' tokens ↓</span>\\n' : '';
    // Clip the literal judge-input string for display at roughly 4*maxTok chars
    // so we render about what the tokenizer actually saw.
    const safeJudge = String(judgeTextRaw || '');
    const clip = truncated ? safeJudge.slice(0, maxTok*4) + '\\n[...truncated for display]' : safeJudge;
    const meta = `${label}${isWin?' <strong style="color:#76B900;">← winning ref</strong>':''} · ~${nTokens||0} tokens${truncated?' · TRUNCATED':''}`;
    return `<div class="ref-block ${isWin?'winning':''}">
      <div class="ref-meta">${meta}</div>
      <div style="font-size:12px; margin-bottom:4px;"><strong>Ref:</strong> ${esc(refText)}</div>
      <pre>${truncMarker}${esc(clip)}</pre>
    </div>`;
  }
  const score = (r.judge_score!==null && r.judge_score!==undefined) ? r.judge_score.toFixed(3) : 'n/a';
  const scoreLine = `judge score: <strong>${score}</strong> → ${r.judge_correct?'<span style="color:#76B900;">correct (&gt;0.5)</span>':'<span style="color:#DC2626;">miss (&le;0.5)</span>'}`;
  // ---- Build judge breakdown: standard mode (always shown), then answer-only
  // mode if it was scored (judge_mode "answer-only" or "both"). When both
  // are present the two panels render side-by-side so the A/B is visible.
  const stdBlock = `
    <div class="judge-breakdown">
      <h4>Judge breakdown — standard${r.judge_score_standard !== null && r.judge_score_standard !== undefined ? ` (score ${r.judge_score_standard.toFixed(3)} → ${r.judge_correct_standard?'<span style=\"color:#76B900;\">correct</span>':'<span style=\"color:#DC2626;\">miss</span>'})` : ''}</h4>
      <div style="font-size:12px; margin-bottom:8px; color:#475569;">judge sees full prediction (incl. &lt;think&gt;)${winIdx>=0?` · winning ref index: <strong>${winIdx}</strong>`:''}</div>
      ${refBlock('GT-A', refs[0]||'', 0, r.judge_input_text_a, r.judge_input_tokens_a, r.judge_truncated_a)}
      ${refs.length>1 ? refBlock('GT-B', refs[1]||'', 1, r.judge_input_text_b, r.judge_input_tokens_b, r.judge_truncated_b) : ''}
    </div>`;
  let aoBlock = '';
  if(r.judge_score_answer_only !== null && r.judge_score_answer_only !== undefined){
    const aoWinIdx = (r.judge_winning_reference_index_answer_only !== undefined && r.judge_winning_reference_index_answer_only !== null) ? r.judge_winning_reference_index_answer_only : -1;
    function aoRefBlock(label, refText, refIdx, judgeTextRaw, nTokens, truncated){
      const isWin = refIdx === aoWinIdx;
      const truncMarker = truncated ? '<span class="trunc-marker">↓ truncated here at '+maxTok+' tokens ↓</span>\\n' : '';
      const safeJudge = String(judgeTextRaw || '');
      const clip = truncated ? safeJudge.slice(0, maxTok*4) + '\\n[...truncated for display]' : safeJudge;
      const meta = `${label}${isWin?' <strong style="color:#76B900;">← winning ref</strong>':''} · ~${nTokens||0} tokens${truncated?' · TRUNCATED':''}`;
      return `<div class="ref-block ${isWin?'winning':''}">
        <div class="ref-meta">${meta}</div>
        <div style="font-size:12px; margin-bottom:4px;"><strong>Ref:</strong> ${esc(refText)}</div>
        <pre>${truncMarker}${esc(clip)}</pre>
      </div>`;
    }
    aoBlock = `
    <div class="judge-breakdown">
      <h4>Judge breakdown — answer-only (score ${r.judge_score_answer_only.toFixed(3)} → ${r.judge_correct_answer_only?'<span style="color:#76B900;">correct</span>':'<span style="color:#DC2626;">miss</span>'})</h4>
      <div style="font-size:12px; margin-bottom:8px; color:#475569;">judge sees only the final answer (&lt;think&gt; stripped) — mirrors brief gt-A/gt-B style</div>
      ${aoRefBlock('GT-A', refs[0]||'', 0, r.judge_input_text_a_answer_only, r.judge_input_tokens_a_answer_only, r.judge_truncated_a_answer_only)}
      ${refs.length>1 ? aoRefBlock('GT-B', refs[1]||'', 1, r.judge_input_text_b_answer_only, r.judge_input_tokens_b_answer_only, r.judge_truncated_b_answer_only) : ''}
    </div>`;
  }
  // A/B divergence callout: when both modes scored and verdicts disagree.
  let abCallout = '';
  if(r.judge_correct_standard !== null && r.judge_correct_standard !== undefined
     && r.judge_correct_answer_only !== null && r.judge_correct_answer_only !== undefined
     && Boolean(r.judge_correct_standard) !== Boolean(r.judge_correct_answer_only)){
    abCallout = `<div class="trunc-callout"><strong>A/B disagreement on this row.</strong> Standard verdict: ${r.judge_correct_standard?'correct':'miss'} (${r.judge_score_standard.toFixed(3)}) · Answer-only verdict: ${r.judge_correct_answer_only?'correct':'miss'} (${r.judge_score_answer_only.toFixed(3)}). The judge's input format changed the answer — a strong signal that long &lt;think&gt; chains are biasing the score.</div>`;
  }
  const judgeHtml = (aoBlock
    ? `<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">${stdBlock}${aoBlock}</div>${abCallout}`
    : stdBlock + abCallout);
  // ---- Truncation callout (yellow) on a miss when truncation happened ----
  const truncCallout = ((r.judge_truncated_a || r.judge_truncated_b) && !r.judge_correct)
    ? `<div class="trunc-callout"><strong>⚠️ Judge input was truncated to ${maxTok} tokens.</strong> The model's final answer may not have been visible to the judge — a known failure mode for reasoning-style models with long &lt;think&gt; chains. Consider a shortened-prompt re-judge for this row.</div>`
    : '';
  // ---- Reasoning + final answer block ----
  const reasoningBlock = reasoningText
    ? `<h4 style="margin-top:14px;">Reasoning trace + answer <span style="font-weight:400; color:#475569;">(~${approxTokens} tokens${tokFlag})</span></h4>
       <pre style="max-height:280px;">${esc('<think>\\n'+reasoningText+'\\n</think>\\n\\n'+finalAnswer)}</pre>`
    : `<h4 style="margin-top:14px;">Final answer <span style="font-weight:400; color:#475569;">(~${approxTokens} tokens${tokFlag})</span></h4>
       <pre>${esc(finalAnswer)}</pre>`;
  // ---- Provenance footer ----
  const provenance = `<div class="provenance-footer">
    question_id: ${esc(r.question_id||'')} · segment_id: ${esc(r.segment_id||'(none)')} · model: ${esc(r.model||'')} · latency: ${r.latency_seconds?r.latency_seconds.toFixed(2)+'s':'n/a'}
    ${r.segment_id?` · <a href="/lingoqa-frames/${encodeURIComponent(r.segment_id)}/0.jpg" target="_blank">open frame 0 →</a>`:''}
  </div>`;
  el('benchModalBody').innerHTML = `
    ${frameStripHtml}
    <div class="kv" style="margin-top:8px;">
      <div>question</div><div>${esc(r.question||'')}</div>
      <div>GT-A</div><div>${esc((r.references||[''])[0])}</div>
      <div>GT-B</div><div>${esc((r.references||['',''])[1]||'')}</div>
      <div>verdict</div><div>${r.judge_correct ? 'correct (sigmoid &gt; 0.5)' : 'miss (sigmoid &le; 0.5)'}</div>
      <div>latency</div><div>${r.latency_seconds ? r.latency_seconds.toFixed(2)+'s' : ''}</div>
    </div>
    ${truncCallout}
    ${reasoningBlock}
    ${judgeHtml}
    <h4 style="margin-top:14px;">Raw response</h4>
    <pre>${esc(r.raw_response||'')}</pre>
    ${provenance}
  `;
  el('benchModal').classList.add('open');
}
document.getElementById('benchModalClose').onclick = ()=>el('benchModal').classList.remove('open');
document.getElementById('benchModal').addEventListener('click', (e)=>{ if(e.target.id==='benchModal') el('benchModal').classList.remove('open'); });

async function benchPoll(){
  if(!benchRunId) return;
  try{
    const r = await fetch('/benchmark/status/'+benchRunId); const snap = await r.json();
    benchRenderSnap(snap);
    if(snap.status === 'complete' || snap.status === 'error'){
      clearInterval(benchPollTimer); benchPollTimer = null;
    }
  }catch(e){ console.error('benchPoll', e); }
}

async function benchStart(sample_size){
  try{
    // Prefer the most recently resolved dataset if its id matches the dropdown
    // value (i.e. the user pasted a URL and the resolver filled in the
    // dropdown). source_config is the dataset_config blob the backend handed
    // us — we forward it so the dispatcher can use repo_id/path hints.
    const dsValue = el('benchDataset').value;
    let source_config = null;
    if(benchUrlResolution && benchUrlResolution.dataset_config && benchUrlResolution.dataset_config.id === dsValue){
      source_config = benchUrlResolution.dataset_config;
    }
    // Phase J amendment: dual judge-mode A/B.
    const jmRadio = document.querySelector('input[name="benchJudgeMode"]:checked');
    const judge_mode = jmRadio ? jmRadio.value : 'standard';
    const body = {
      dataset: dsValue,
      judge: el('benchJudge').value,
      judge_mode: judge_mode,
      sample_size: sample_size!==undefined ? sample_size : Number(el('benchSampleSize').value),
      concurrency: Number(el('benchConcurrency').value),
      seed: Number(el('benchSeed').value),
      source_config: source_config,
    };
    const j = await api('/benchmark/run', body);
    benchRunId = j.run_id;
    el('benchAccuracy').textContent = '--';
    el('benchBar').style.width = '0%';
    if(benchPollTimer) clearInterval(benchPollTimer);
    benchPollTimer = setInterval(benchPoll, 1500);
    benchPoll();
  }catch(e){ alert(e.message); }
}
document.getElementById('benchSmokeBtn').onclick = ()=>benchStart(5);
['benchSampleSize','benchConcurrency','benchSeed'].forEach(id=>{
  const s = document.getElementById(id); if(!s) return;
  const vid = id+'Val'; const v = document.getElementById(vid);
  if(v){ s.oninput = ()=>{ v.textContent = s.value; }; }
});

// ====================================================================
// Drop-zone Benchmark View — URL resolve, models, history, last-run tile.
// One large URL input + one model dropdown + one big Run button.
// Power-user knobs live behind "⚙ Advanced" (unchanged 5-knob form).
// ====================================================================
let benchUrlResolution = null;       // {kind, dataset_config, display, ...}
let benchHistoryCache = [];          // raw array from /benchmark/history
let benchUrlDebounce = null;

function benchSetChip(text, kind){
  const chip = el('benchUrlChip');
  if(!text){ chip.style.display='none'; chip.textContent=''; return; }
  chip.style.display = 'inline-block';
  chip.textContent = text;
  chip.classList.toggle('unknown', kind === 'unknown');
}

async function benchResolveUrl(raw){
  const url = (raw||'').trim();
  if(!url){
    benchUrlResolution = null;
    benchSetChip('', null);
    el('benchDropzoneStatus').textContent = '';
    const r = el('benchResolvedSource'); if(r) r.value = '';
    return;
  }
  try{
    const resp = await fetch('/benchmark/resolve_url?url='+encodeURIComponent(url));
    const j = await resp.json();
    benchUrlResolution = j;
    const kind = j.kind || 'unknown';
    if(kind === 'lingoqa'){
      benchSetChip('✓ LingoQA detected (local cache)', kind);
    } else if(kind === 'hf_dataset'){
      benchSetChip('✓ HuggingFace dataset detected', kind);
    } else if(kind === 'arxiv'){
      benchSetChip('✓ Arxiv paper detected', kind);
    } else if(kind === 'github'){
      benchSetChip('✓ GitHub repo detected', kind);
    } else if(kind === 'gdrive'){
      benchSetChip('✓ GDrive folder detected', kind);
    } else {
      benchSetChip('? Unknown URL kind', 'unknown');
    }
    el('benchDropzoneStatus').textContent = j.display || '';
    const r = el('benchResolvedSource');
    const cfg = j.dataset_config || {};
    if(r){
      r.value = cfg.id ? (cfg.id + (cfg.ready ? ' [ready]' : (cfg.id !== 'lingoqa-official' ? ' [will load on Run]' : ''))) : '';
    }
    // Mirror the resolver's dataset_config.id into the Dataset <select>.
    // If the option doesn't exist yet (new HF/arxiv/github/gdrive URL),
    // append it before selecting. This replaces the old hard-coded reset
    // to 'lingoqa-official', which silently discarded the resolver's work.
    if(cfg.id){
      const ds = el('benchDataset');
      if(ds){
        let opt = Array.from(ds.options).find(o => o.value === cfg.id);
        if(!opt){
          opt = document.createElement('option');
          opt.value = cfg.id;
          opt.textContent = (cfg.name || cfg.id) + (cfg.ready ? '' : ' (will download on Run)');
          ds.appendChild(opt);
        } else {
          opt.textContent = (cfg.name || cfg.id) + (cfg.ready ? '' : ' (will download on Run)');
        }
        ds.value = cfg.id;
      }
    }
  }catch(e){
    benchSetChip('? resolve failed', 'unknown');
    el('benchDropzoneStatus').textContent = 'Resolve failed: '+e.message;
  }
}

async function benchLoadModels(){
  const sel = el('benchModelSelect');
  try{
    const r = await fetch('/benchmark/available_models');
    const list = await r.json();
    if(!Array.isArray(list) || list.length===0){
      sel.innerHTML = '<option value="">(no models available)</option>';
      return;
    }
    // Default to nvidia/Cosmos3-Super-Reasoner when present, else first entry.
    const def = list.find(m => (m.id||'').includes('Cosmos3-Super-Reasoner')) || list[0];
    sel.innerHTML = list.map(m=>{
      const id = m.id||'';
      const host = m.host||'';
      const status = m.status||'ready';
      const label = id + (host ? ' — '+host : '') + ' — '+status;
      const selected = (m === def) ? ' selected' : '';
      return `<option value="${esc(id)}"${selected}>${esc(label)}</option>`;
    }).join('');
  }catch(e){
    sel.innerHTML = `<option value="">(error: ${esc(e.message)})</option>`;
  }
}

function benchFmtAgo(epoch){
  if(!epoch) return '';
  const ageSec = Math.max(0, (Date.now()/1000) - Number(epoch));
  if(ageSec < 60) return Math.round(ageSec)+'s ago';
  if(ageSec < 3600) return Math.round(ageSec/60)+'m ago';
  if(ageSec < 86400) return (ageSec/3600).toFixed(1)+'h ago';
  return Math.round(ageSec/86400)+'d ago';
}

function benchFmtDataset(d){
  d = String(d||'');
  if(d === 'lingoqa-official') return 'LingoQA';
  if(d.startsWith('hf:'))     return 'HF: ' + d.slice(3);
  if(d.startsWith('arxiv:'))  return 'arXiv ' + d.slice(6);
  if(d.startsWith('github:')) return 'GitHub: ' + d.slice(7);
  if(d.startsWith('gdrive:')) return 'GDrive ' + d.slice(7, 15) + '…';
  if(d.startsWith('local:'))  return 'Local: ' + d.slice(6).split('/').pop();
  return d;
}

async function benchLoadHistory(){
  try{
    const r = await fetch('/benchmark/history');
    const list = await r.json();
    benchHistoryCache = Array.isArray(list) ? list : [];
    benchRenderHistory();
    benchRenderLastRunTile();
  }catch(e){
    benchHistoryCache = [];
    benchRenderHistory();
    benchRenderLastRunTile();
  }
}

function benchRenderLastRunTile(){
  const tile = el('benchLastRunTile');
  const rerunLast = el('benchRerunLastBtn');
  if(!benchHistoryCache.length){
    tile.style.display = 'none';
    if(rerunLast) rerunLast.disabled = true;
    return;
  }
  const last = benchHistoryCache[benchHistoryCache.length - 1];
  const acc = (last.accuracy!=null) ? (Number(last.accuracy)*100).toFixed(1)+'%' : '--';
  const ds = benchFmtDataset(last.dataset);
  const ago = benchFmtAgo(last.epoch);
  el('benchLastRunText').textContent = `Last run: ${acc} on ${ds} (${ago})`;
  const link = el('benchLastRunReport');
  link.href = last.report_url || ('/benchmark/report/'+last.run_id);
  link.textContent = 'Open Report';
  tile.style.display = 'flex';
  if(rerunLast) rerunLast.disabled = false;
}

function benchRenderHistory(){
  const rows = el('benchHistoryRows');
  if(!benchHistoryCache.length){
    rows.innerHTML = '<div class="bench-history-empty">No runs yet. Hit Run Benchmark above to start your first.</div>';
    return;
  }
  const ordered = benchHistoryCache.slice().reverse();
  rows.innerHTML = ordered.map(h=>{
    const acc = (h.accuracy!=null) ? (Number(h.accuracy)*100).toFixed(1)+'%' : '--';
    const ds = benchFmtDataset(h.dataset);
    const when = h.timestamp || benchFmtAgo(h.epoch);
    const model = h.model || '';
    return `<div class="bench-history-row" data-rid="${esc(h.run_id)}">
      <div title="${esc(h.run_id)}">${esc(when)}</div>
      <div title="${esc(model)}">${esc(model)}</div>
      <div>${esc(ds)}</div>
      <div class="acc">${acc}</div>
      <div class="actions">
        <a class="bench-ghost-link" href="${esc(h.report_url||('/benchmark/report/'+h.run_id))}" target="_blank" onclick="event.stopPropagation();">Open Report</a>
        <button class="bench-ghost-btn" data-rerun="${esc(h.run_id)}" onclick="event.stopPropagation(); benchRerunById('${esc(h.run_id)}');">Re-run</button>
      </div>
    </div>`;
  }).join('');
  rows.querySelectorAll('.bench-history-row').forEach(row=>{
    row.addEventListener('click', ()=>benchRerunById(row.dataset.rid));
  });
}

async function benchRerunById(runId){
  if(!runId) return;
  try{
    const resp = await fetch('/benchmark/rerun/'+encodeURIComponent(runId), {method:'POST'});
    const j = await resp.json();
    if(j.error){ alert(j.error); return; }
    benchRunId = j.run_id;
    el('benchAccuracy').textContent = '--';
    el('benchBar').style.width = '0%';
    if(benchPollTimer) clearInterval(benchPollTimer);
    benchPollTimer = setInterval(benchPoll, 1500);
    benchPoll();
  }catch(e){ alert(e.message); }
}

// Wire the drop-zone primary button + URL detection + ghost buttons.
const benchUrlInputEl = document.getElementById('benchUrlInput');
if(benchUrlInputEl){
  benchUrlInputEl.addEventListener('input', (e)=>{
    if(benchUrlDebounce) clearTimeout(benchUrlDebounce);
    benchUrlDebounce = setTimeout(()=>benchResolveUrl(e.target.value), 200);
  });
  benchUrlInputEl.addEventListener('paste', (e)=>{
    setTimeout(()=>benchResolveUrl(benchUrlInputEl.value), 50);
  });
}
document.getElementById('benchRunBtn').onclick = async function(){
  // If the user typed a URL, ensure we've resolved it; if unknown, warn.
  const urlVal = (el('benchUrlInput').value||'').trim();
  if(urlVal && (!benchUrlResolution || benchUrlResolution.kind === 'unknown')){
    await benchResolveUrl(urlVal);
    if(benchUrlResolution && benchUrlResolution.kind === 'unknown'){
      if(!confirm('URL kind unknown — run with the cached LingoQA dataset anyway?')) return;
    }
  }
  // Sync the selected model into the hidden auto-detected field so existing
  // /benchmark/run flow keeps using auto-detection on the local NIM. Multi-NIM
  // dispatch is a future hop; for now we surface the choice in the UI.
  benchStart();
};
const benchRerunLastBtn = document.getElementById('benchRerunLastBtn');
if(benchRerunLastBtn){
  benchRerunLastBtn.onclick = ()=>{
    if(!benchHistoryCache.length){ alert('No prior runs to re-run.'); return; }
    const last = benchHistoryCache[benchHistoryCache.length - 1];
    benchRerunById(last.run_id);
  };
}
const benchLastRunRerunBtn = document.getElementById('benchLastRunRerun');
if(benchLastRunRerunBtn){
  benchLastRunRerunBtn.onclick = ()=>{
    if(!benchHistoryCache.length) return;
    const last = benchHistoryCache[benchHistoryCache.length - 1];
    benchRerunById(last.run_id);
  };
}
const benchToggleAdvBtn = document.getElementById('benchToggleAdvBtn');
if(benchToggleAdvBtn){
  benchToggleAdvBtn.onclick = ()=>{
    const panel = el('benchAdvanced');
    const open = panel.style.display !== 'none';
    panel.style.display = open ? 'none' : 'block';
    benchToggleAdvBtn.setAttribute('aria-expanded', open ? 'false' : 'true');
  };
}
const benchToggleHistBtn = document.getElementById('benchToggleHistBtn');
if(benchToggleHistBtn){
  benchToggleHistBtn.onclick = ()=>{
    const panel = el('benchHistoryPanel');
    const open = panel.style.display !== 'none';
    if(!open) benchLoadHistory();
    panel.style.display = open ? 'none' : 'block';
    benchToggleHistBtn.setAttribute('aria-expanded', open ? 'false' : 'true');
  };
}
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
        elif self.path == "/benchmark/datasets":
            base = _lingoqa_dataset_dir()
            ready = base is not None and (base / "val.parquet").exists()
            out_datasets: List[Dict[str, Any]] = [{
                "id": "lingoqa-official",
                "name": "LingoQA (1000 rows)",
                "rows": 1000,
                "ready": bool(ready),
                "path": str(base) if base else None,
            }]
            # Session-resolved datasets — populated by /benchmark/resolve_url.
            with BENCHMARK_LOCK:
                resolved = dict(BENCHMARK_STATE.get("resolved_datasets") or {})
            for ds_id, cfg in resolved.items():
                if ds_id == "lingoqa-official":
                    continue
                out_datasets.append({
                    "id": ds_id,
                    "name": cfg.get("name") or ds_id,
                    "ready": bool(cfg.get("ready")),
                    "source_url": cfg.get("source_url"),
                    "resolved_epoch": cfg.get("resolved_epoch"),
                })
            self.send_json(out_datasets)
        elif self.path == "/benchmark/judges":
            self.send_json([{
                "id": "lingo-judge",
                "name": "Lingo-Judge (DeBERTa-v3-base, Wayve)",
                "loaded": bool(BENCHMARK_STATE.get("judge_loaded")),
                "error": BENCHMARK_STATE.get("judge_error"),
            }])
        elif self.path == "/benchmark/model":
            try:
                model = benchmark_detect_model()
                self.send_json({"model": model, "base_url": benchmark_nim_base_url()})
            except Exception as exc:
                self.send_json({"model": None, "error": str(exc), "base_url": benchmark_nim_base_url()})
        elif self.path.startswith("/benchmark/resolve_url"):
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            url_arg = (params.get("url") or [""])[0]
            try:
                self.send_json(_resolve_url_to_dataset_cached(url_arg))
            except Exception as exc:
                self.send_json({"kind": "unknown", "url": url_arg, "error": str(exc), "display": f"Resolve failed: {exc}"})
        elif self.path == "/benchmark/available_models":
            try:
                self.send_json(_benchmark_available_models())
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif self.path == "/benchmark/history":
            try:
                self.send_json(_benchmark_history_load())
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif self.path.startswith("/benchmark/status/"):
            run_id = self.path.rsplit("/", 1)[-1].split("?")[0]
            snap = benchmark_get(run_id)
            if not snap:
                self.send_error(404)
                return
            # Trim results in the status payload — full results available at
            # /benchmark/results/<run_id>.json. Keep the last 12 rows for the
            # live sample-grid view.
            light = dict(snap)
            results = light.get("results") or []
            light["recent_results"] = results[-12:]
            light["results"] = None
            light["result_count"] = len(results)
            self.send_json(light)
        elif self.path.startswith("/benchmark/results/") and self.path.endswith(".json"):
            run_id = self.path.rsplit("/", 1)[-1][:-5]
            snap = benchmark_get(run_id)
            if not snap:
                self.send_error(404)
                return
            self.send_json(snap)
        elif self.path.startswith("/benchmark/report/"):
            run_id = self.path.rsplit("/", 1)[-1].split("?")[0]
            if not benchmark_get(run_id):
                self.send_error(404)
                return
            self.send_text(benchmark_render_report(run_id), "text/html")
        elif self.path.startswith("/benchmark/artifact/"):
            # /benchmark/artifact/<run_id>/<model_safe>/<question_id>/<filename>
            tail = urllib.parse.urlparse(self.path).path[len("/benchmark/artifact/"):]
            parts = [urllib.parse.unquote(p) for p in tail.split("/") if p]
            if len(parts) != 4:
                self.send_error(404)
                return
            run_id, model_safe, qid, fname = parts
            # Defensive: filename whitelist only.
            if fname not in ("frame_strip.png", "frame_gif.gif", "ui_card.png", "reasoning.png"):
                self.send_error(404)
                return
            safe_model = _safe_model_dir(model_safe)
            artifact = TRACEABILITY_ROOT / run_id / safe_model / qid / fname
            # Resolve and sandbox under TRACEABILITY_ROOT to avoid path traversal.
            try:
                resolved = artifact.resolve()
                root_resolved = TRACEABILITY_ROOT.resolve()
                if root_resolved not in resolved.parents and resolved != root_resolved:
                    self.send_error(404)
                    return
            except Exception:
                self.send_error(404)
                return
            if not resolved.exists():
                self.send_error(404)
                return
            ctype = "image/gif" if fname.endswith(".gif") else "image/png"
            self.send_file(resolved, ctype)
        elif self.path.startswith("/lingoqa-frames/"):
            # /lingoqa-frames/<segment_id>/<idx>.jpg — click-to-enlarge support.
            tail = urllib.parse.urlparse(self.path).path[len("/lingoqa-frames/"):]
            parts = [urllib.parse.unquote(p) for p in tail.split("/") if p]
            if len(parts) != 2 or not parts[1].endswith(".jpg"):
                self.send_error(404)
                return
            segment_id = parts[0]
            try:
                idx = int(parts[1][:-4])
            except ValueError:
                self.send_error(404)
                return
            if not re.match(r"^[A-Za-z0-9_-]+$", segment_id) or idx < 0 or idx > 99:
                self.send_error(404)
                return
            src = _resolve_segment_image(None, segment_id, idx)
            if src is None or not src.exists():
                self.send_error(404)
                return
            self.send_file(src, "image/jpeg")
        elif self.path.startswith("/benchmark/stream/"):
            # Keepalive SSE stream for the live run panel. Emits a comment byte
            # every ~15s + a JSON status snapshot every poll.
            run_id = self.path.rsplit("/", 1)[-1].split("?")[0]
            if not benchmark_get(run_id):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                last_emit = 0.0
                while True:
                    snap = benchmark_get(run_id) or {}
                    status = snap.get("status") or "unknown"
                    payload = json.dumps({
                        "status": status,
                        "progress": snap.get("progress") or {},
                        "summary": snap.get("summary") or {},
                        "model": snap.get("model"),
                    })
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if status in ("complete", "error"):
                        break
                    # Keepalive comment every 15s to keep upstream proxies happy.
                    end = time.monotonic() + 5
                    while time.monotonic() < end:
                        time.sleep(1)
                        if time.monotonic() - last_emit > 15:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            last_emit = time.monotonic()
            except (BrokenPipeError, ConnectionResetError):
                pass
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
            elif self.path == "/benchmark/run":
                dataset_id = str(payload.get("dataset") or "lingoqa-official")
                judge_id = str(payload.get("judge") or "lingo-judge")
                sample_size = int(payload.get("sample_size") or 1000)
                concurrency = int(payload.get("concurrency") or 8)
                seed = int(payload.get("seed") or 42)
                # Phase J amendment: dual judge-mode A/B. Default "standard"
                # for backward compatibility.
                judge_mode = str(payload.get("judge_mode") or "standard")
                if judge_mode not in ("standard", "answer-only", "both"):
                    judge_mode = "standard"
                # source_config is the dataset_config dict the frontend got
                # back from /benchmark/resolve_url. Forwarded so the loader
                # can use repo_id, download paths, judge hints, etc.
                source_config = payload.get("source_config")
                if not isinstance(source_config, dict):
                    source_config = None
                run_id = f"bench-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
                _benchmark_update(
                    run_id,
                    status="queued",
                    dataset=dataset_id,
                    judge=judge_id,
                    judge_mode=judge_mode,
                    sample_size=sample_size,
                    concurrency=concurrency,
                    seed=seed,
                    source_config=source_config,
                    results=[],
                    progress={"done": 0, "total": 0, "errors": 0},
                )
                thread = threading.Thread(
                    target=run_lingoqa_benchmark,
                    args=(run_id, dataset_id, judge_id, sample_size, concurrency, seed, source_config),
                    kwargs={"judge_mode": judge_mode},
                    daemon=True,
                )
                thread.start()
                self.send_json({
                    "ok": True,
                    "run_id": run_id,
                    "dataset": dataset_id,
                    "judge": judge_id,
                    "judge_mode": judge_mode,
                    "sample_size": sample_size,
                    "concurrency": concurrency,
                    "seed": seed,
                })
            elif self.path.startswith("/benchmark/rerun/"):
                # POST /benchmark/rerun/<run_id> — clone params from history and fire a new run.
                old_run_id = self.path.rsplit("/", 1)[-1].split("?")[0]
                hist = _benchmark_history_find(old_run_id)
                if not hist:
                    self.send_json({"error": f"run_id not in history: {old_run_id}"}, status=404)
                    return
                dataset_id = str(hist.get("dataset") or "lingoqa-official")
                judge_id = str(hist.get("judge") or "lingo-judge")
                sample_size = int(hist.get("sample_size") or 1000)
                concurrency = int(hist.get("concurrency") or 8)
                seed = int(hist.get("seed") or 42)
                # Pull source_config from the prior run's live snapshot if
                # still in memory; falls back to None (loader will look it up
                # by dataset_id alone).
                prior_snap = benchmark_get(old_run_id) or {}
                source_config = prior_snap.get("source_config")
                if not isinstance(source_config, dict):
                    source_config = None
                # Carry the judge_mode forward from the prior run when
                # available; default to standard otherwise.
                judge_mode = str(prior_snap.get("judge_mode") or hist.get("judge_mode") or "standard")
                if judge_mode not in ("standard", "answer-only", "both"):
                    judge_mode = "standard"
                new_run_id = f"bench-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
                _benchmark_update(
                    new_run_id,
                    status="queued",
                    dataset=dataset_id,
                    judge=judge_id,
                    judge_mode=judge_mode,
                    sample_size=sample_size,
                    concurrency=concurrency,
                    seed=seed,
                    source_config=source_config,
                    results=[],
                    progress={"done": 0, "total": 0, "errors": 0},
                    rerun_of=old_run_id,
                )
                thread = threading.Thread(
                    target=run_lingoqa_benchmark,
                    args=(new_run_id, dataset_id, judge_id, sample_size, concurrency, seed, source_config),
                    kwargs={"judge_mode": judge_mode},
                    daemon=True,
                )
                thread.start()
                self.send_json({
                    "ok": True,
                    "run_id": new_run_id,
                    "rerun_of": old_run_id,
                    "dataset": dataset_id,
                    "judge": judge_id,
                    "judge_mode": judge_mode,
                    "sample_size": sample_size,
                    "concurrency": concurrency,
                    "seed": seed,
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
    Path(os.getenv("BATCH_INFERENCE_URL_FILE", "/tmp/byo_video_batch_inference_url.txt")).write_text(url, encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Cosmos BYO-video batch-inference frontend")
    sub = parser.add_subparsers(dest="cmd")
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--host", default=os.getenv("BATCH_INFERENCE_HOST", "0.0.0.0"))
    serve_p.add_argument("--port", type=int, default=int(os.getenv("BATCH_INFERENCE_PORT", "7861")))
    smoke_p = sub.add_parser("smoke")
    smoke_p.add_argument("--dataset", default=DEFAULT_DATASET)
    smoke_p.add_argument("--max-videos", type=int, default=int(os.getenv("BATCH_INFERENCE_SMOKE_VIDEOS", "2")))
    smoke_p.add_argument("--concurrency", type=int, default=int(os.getenv("BATCH_INFERENCE_CONCURRENCY", "2")))
    smoke_p.add_argument("--fps", type=float, default=float(os.getenv("BATCH_INFERENCE_FPS", "1")))
    smoke_p.add_argument("--max-tokens", type=int, default=int(os.getenv("BATCH_INFERENCE_MAX_TOKENS", "1024")))
    args = parser.parse_args()
    if args.cmd == "smoke":
        return smoke(args)
    serve(args.host if hasattr(args, "host") else "0.0.0.0", args.port if hasattr(args, "port") else 7861)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
