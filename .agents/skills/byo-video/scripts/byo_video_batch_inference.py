#!/usr/bin/env python3
"""Batch-inference frontend for Cosmos BYO-video inference.

This is intentionally self-contained: it serves a small browser UI, loads video
samples from a public Hugging Face dataset through FiftyOne when available, and
runs selected videos concurrently against the OpenAI-compatible vLLM/NIM server
that the BYO-video setup script starts.

Two views are exposed in the UI:
- Basic View: per-video inference against a Hugging Face dataset with prompt
  presets, parameter sliders, reference-video uploads, FiftyOne integration,
  and credential-safe spot/ablation comparison between the loaded endpoint and
  dynamically discovered hosted models.
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
import io
import json
import mimetypes
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, deque
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
REFERENCE_VIDEO_DIR = Path(os.getenv("BATCH_INFERENCE_REFERENCE_VIDEOS", "/tmp/byo_video_batch_inference_references"))
REFERENCE_VIDEO_MAX_BYTES = int(os.getenv("BATCH_INFERENCE_REFERENCE_VIDEO_MAX_BYTES", str(512 * 1024 * 1024)))
REFERENCE_VIDEO_TOTAL_MAX_BYTES = int(
    os.getenv("BATCH_INFERENCE_REFERENCE_VIDEO_TOTAL_MAX_BYTES", str(2 * 1024 * 1024 * 1024))
)
_HOSTED_DEFAULT_BASE_URL = os.getenv("NVIDIA_HOSTED_BASE_URL", "https://inference-api.nvidia.com/v1").rstrip("/")
NVIDIA_HOSTED_CATALOG_URL = os.getenv("NVIDIA_HOSTED_CATALOG_URL", _HOSTED_DEFAULT_BASE_URL + "/models").strip()
NVIDIA_HOSTED_INFERENCE_URL = os.getenv("NVIDIA_HOSTED_INFERENCE_URL", _HOSTED_DEFAULT_BASE_URL + "/chat/completions").strip()
NVIDIA_HOSTED_AUTH_HEADER = os.getenv("NVIDIA_HOSTED_AUTH_HEADER", "Authorization").strip()
NVIDIA_HOSTED_AUTH_SCHEME = os.getenv("NVIDIA_HOSTED_AUTH_SCHEME", "Bearer").strip()
COMPARISON_MAX_CASES = int(os.getenv("BATCH_INFERENCE_COMPARISON_MAX_CASES", "256"))
CURATED_HOSTED_MODELS = [
    {"id": "", "label": "Cosmos3 Super Reasoner — discover exact request id", "family": "Cosmos", "supports_video": True, "supports_image": True, "supports_text": True},
    {"id": "", "label": "Cosmos3 Nano Reasoner — discover exact request id", "family": "Cosmos", "supports_video": True, "supports_image": True, "supports_text": True},
    {"id": "", "label": "Gemma 4 — discover exact request id", "family": "Gemma", "supports_video": False, "supports_image": True, "supports_text": True},
    {"id": "", "label": "Nemotron 3 Nano Omni — discover exact request id", "family": "Nemotron", "supports_video": None, "supports_image": None, "supports_text": True},
    {"id": "", "label": "Qwen latest — discover exact request id", "family": "Qwen", "supports_video": None, "supports_image": None, "supports_text": True},
    {"id": "", "label": "Gemini 3.6 — discover exact request id", "family": "Gemini", "supports_video": None, "supports_image": None, "supports_text": True},
    {"id": "", "label": "Claude latest — discover exact request id", "family": "Claude", "supports_video": None, "supports_image": None, "supports_text": True},
    {"id": "", "label": "GPT 5.6 — discover exact request id", "family": "GPT", "supports_video": None, "supports_image": None, "supports_text": True},
    {"id": "", "label": "Kimi K2.6 — catalog must confirm video input", "family": "Kimi", "supports_video": None, "supports_image": None, "supports_text": True},
]


def write_private_json(path: Path, payload: Any) -> None:
    """Atomically persist JSON with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
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
REFERENCE_UPLOAD_LOCK = threading.Lock()
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
    "comparison": {
        "running": False,
        "catalog": [
            dict(
                item,
                source="curated",
                capability_source="curated",
                video_candidate=bool(item.get("supports_video")),
                comparison_media_strategy=(
                    "native_video" if item.get("supports_video") is True
                    else ("sampled_frames" if item.get("supports_image") is True else None)
                ),
            )
            for item in CURATED_HOSTED_MODELS
        ],
        "catalog_source": "curated",
        "catalog_warning": None,
        "run": None,
        "results": [],
        "history": [],
        "exports": [],
    },
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


SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(?:nvapi|sk|hf)[-_][A-Za-z0-9._-]{8,}\b"),
    re.compile(r"(?i)\bBearer\s+[^\s,;\"']+"),
    re.compile(r"(?i)(api[_-]?key|authorization|access[_-]?token|secret)(\s*[:=]\s*)[^\s,;]+"),
)
SECRET_KEY_NAMES = re.compile(r"(?i)(?:api[_-]?key|authorization|access[_-]?token|secret|credential)")


def redact_sensitive(value: Any, depth: int = 64) -> Any:
    """Return a JSON-safe credential-redacted copy of arbitrary runtime data."""
    if depth <= 0:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            out[key_text] = "[REDACTED]" if SECRET_KEY_NAMES.search(key_text) else redact_sensitive(item, depth - 1)
        return out
    if isinstance(value, (list, tuple, set)):
        return [redact_sensitive(item, depth - 1) for item in value]
    if isinstance(value, Path):
        return redact_sensitive(str(value), depth - 1)
    if isinstance(value, str):
        text = value
        for pattern in SECRET_VALUE_PATTERNS:
            if pattern.pattern.startswith("(?i)(api"):
                text = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text)
            else:
                text = pattern.sub("[REDACTED]", text)
        return text
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_sensitive(str(value), depth - 1)


def safe_error(exc: Any, limit: int = 1000) -> str:
    return str(redact_sensitive(str(exc)))[:limit]


def redact_runtime_secret(value: Any, secret: str, depth: int = 64) -> Any:
    """Redact an arbitrary per-request secret in addition to known token shapes."""
    if depth <= 0:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {str(key): redact_runtime_secret(item, secret, depth - 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [redact_runtime_secret(item, secret, depth - 1) for item in value]
    if isinstance(value, str):
        return redact_sensitive(value.replace(secret, "[REDACTED]") if secret else value)
    return redact_sensitive(value, depth)


def safe_runtime_error(exc: Any, secret: str, limit: int = 1000) -> str:
    return str(redact_runtime_secret(str(exc), secret))[:limit]


def redact_endpoint_references(value: Any, endpoints: Iterable[str], depth: int = 64) -> Any:
    replacements = set()
    for endpoint in endpoints:
        endpoint = str(endpoint or "").strip()
        if not endpoint:
            continue
        replacements.add(endpoint)
        try:
            parsed = urllib.parse.urlparse(endpoint)
            if parsed.netloc:
                replacements.add(parsed.netloc)
            if parsed.hostname:
                replacements.add(parsed.hostname)
        except Exception:
            pass
    if depth <= 0:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {str(key): redact_endpoint_references(item, replacements, depth - 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [redact_endpoint_references(item, replacements, depth - 1) for item in value]
    if isinstance(value, str):
        text = value
        for replacement in sorted(replacements, key=len, reverse=True):
            text = text.replace(replacement, "[ENDPOINT]")
        return text
    return value


def redact_comparison_payload(value: Any, secret: str, endpoint: str) -> Any:
    return redact_endpoint_references(redact_runtime_secret(value, secret), [endpoint])


def safe_comparison_error(exc: Any, secret: str, endpoint: str, limit: int = 1000) -> str:
    return str(redact_comparison_payload(str(exc), secret, endpoint))[:limit]


def safe_comparison_error_with_prompts(
    exc: Any,
    secret: str,
    endpoint: str,
    prompts: Iterable[Any],
    limit: int = 1000,
) -> str:
    value: Any = redact_comparison_payload(str(exc), secret, endpoint)
    for prompt in prompts:
        prompt_text = str(prompt or "")
        if prompt_text:
            value = redact_runtime_secret(value, prompt_text)
    return str(value)[:limit]


def log(message: str) -> None:
    message = str(redact_sensitive(str(message)))
    line = time.strftime("%H:%M:%S") + " " + message
    with STATE_LOCK:
        STATE["logs"].append(line)
        STATE["logs"] = STATE["logs"][-200:]
    print("[batch-inference]", message, flush=True)


def snapshot() -> Dict[str, Any]:
    with STATE_LOCK:
        data = json.loads(json.dumps(STATE, default=str))
    data = redact_sensitive(data)
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
            write_private_json(RESULTS_FILE, snapshot())
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
    reference_videos = [
        item for item in snapshot().get("videos", [])
        if item.get("source") == "reference_upload" and Path(str(item.get("filepath") or "")).is_file()
    ]
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
                "last_error": safe_error(exc),
            },
        )
        raise
    merged_videos = videos + [item for item in reference_videos if item.get("id") not in {video.get("id") for video in videos}]
    now = time.time()
    update_state(
        dataset_repo=repo_id,
        videos=merged_videos,
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
            "last_event": f"Loaded {len(videos)} dataset videos and retained {len(merged_videos) - len(videos)} reference uploads",
            "last_error": None,
        },
        batch_metrics=None,
    )
    return merged_videos


def load_dataset_worker(repo_id: str, max_videos: int) -> None:
    try:
        load_dataset(repo_id, max_videos)
    except Exception as exc:
        log(f"Dataset load failed: {exc}")
    finally:
        try:
            write_private_json(RESULTS_FILE, snapshot())
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
        default_base = (
            os.getenv("ALPAMAYO_BASE_URL", "http://localhost:8001/v1")
            if configured_backend == "alpamayo"
            else os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        )
        default_model = os.getenv("MODEL_NAME") or (
            os.getenv("ALPAMAYO_MODEL_ID", "nvidia/Alpamayo-1.5-10B")
            if configured_backend == "alpamayo"
            else ""
        )
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
    model_lower = (model or "").lower()
    return (
        "nim" in backend_lower
        or "alpamayo" in backend_lower
        or "alpamayo" in model_lower
        or model_prefers_file_url(model)
        or model_prefers_video_data(model, backend)
    )


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


def content_for_video(
    video_path: str,
    prompt: str,
    model: str,
    fps: float,
    max_pixels: int,
    max_frames: int,
    backend: str = "",
    force_frames: bool = False,
    force_native_video: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = get_video_meta(video_path)
    effective_max_frames = int(max_frames)
    if force_frames and "nim" in (backend or os.getenv("INFERENCE_BACKEND", "")).lower():
        fallback_limit = nim_frame_fallback_limit()
        if effective_max_frames <= 0 or effective_max_frames > fallback_limit:
            effective_max_frames = fallback_limit
    plan = estimate_plan(meta, fps, max_pixels, effective_max_frames, model if not force_frames else "", backend if not force_frames else "")
    if not force_frames and (force_native_video or model_uses_native_video(model, backend)):
        if plan.get("mode") != "native_video_url":
            plan.update({
                "mode": "native_video_url",
                "frames_passed": int(meta.get("total_frames") or 0),
                "visual_tokens_est": None,
                "note": "Video is passed as video_url; endpoint samples frames internally.",
            })
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


def validated_runtime_key(value: Any) -> str:
    key = str(value or "").strip()
    if not key:
        raise ClientInputError("A runtime API key is required for hosted model discovery and comparison")
    if len(key) < 8 or len(key) > 4096 or any(ch.isspace() for ch in key):
        raise ClientInputError("The runtime API key format is invalid")
    return key


def validated_hosted_endpoint(value: Any, label: str = "Hosted endpoint") -> str:
    """Validate a configured hosted URL before any credential-bearing request."""
    endpoint = str(value or "").strip()
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        parsed.port
    except Exception as exc:
        raise RuntimeError(f"{label} configuration is invalid") from exc
    allow_local_http = os.getenv("NVIDIA_HOSTED_ALLOW_HTTP", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    is_local_mock = (parsed.hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}
    valid_scheme = parsed.scheme == "https" or (
        parsed.scheme == "http" and allow_local_http and is_local_mock
    )
    if (
        not endpoint
        or any(character.isspace() or ord(character) < 0x20 for character in endpoint)
        or not valid_scheme
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            f"{label} must use HTTPS (or opt-in loopback HTTP for a local mock) without credentials, query, or fragment"
        )
    return endpoint.rstrip("/")


def hosted_model_family(model_id: str) -> str:
    lower = str(model_id or "").lower()
    for needle, family in (
        ("cosmos", "Cosmos"),
        ("nemotron", "Nemotron"),
        ("qwen", "Qwen"),
        ("gemma", "Gemma"),
        ("gemini", "Gemini"),
        ("kimi", "Kimi"),
        ("claude", "Claude"),
        ("gpt", "GPT"),
    ):
        if needle in lower:
            return family
    return "Other"


def _capability_text(raw: Dict[str, Any]) -> str:
    fields = [
        raw.get("capabilities"),
        raw.get("modalities"),
        raw.get("input_modalities"),
        raw.get("supported_modalities"),
        raw.get("features"),
        (
            {key: raw.get(key) for key in (
                "supports_video", "supports_image", "supports_text", "supports_vlm",
                "supports_vision", "supports_multimodal",
            ) if key in raw}
            if any(key in raw for key in (
                "supports_video", "supports_image", "supports_text", "supports_vlm",
                "supports_vision", "supports_multimodal",
            )) else None
        ),
    ]
    return " ".join(json.dumps(value, default=str) for value in fields if value is not None).lower()


def _top_level_capability_flags(raw: Dict[str, Any]) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    if "supports_video" in raw:
        flags["video"] = raw.get("supports_video")
    image_values = [
        raw.get(key) for key in (
            "supports_image", "supports_vlm", "supports_vision", "supports_multimodal",
        ) if key in raw
    ]
    if image_values:
        flags["image"] = True if True in image_values else (
            False if all(value is False for value in image_values) else image_values[0]
        )
    if "supports_text" in raw:
        flags["text"] = raw.get("supports_text")
    return flags


def _explicit_modality_state(value: Any, modality: str) -> Optional[bool]:
    modality = modality.lower()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if modality in normalized_key:
                if isinstance(item, bool):
                    return item
                if isinstance(item, (int, float)):
                    return bool(item)
                if isinstance(item, str) and item.strip().lower() in {"true", "yes", "supported", "enabled"}:
                    return True
                if isinstance(item, str) and item.strip().lower() in {"false", "no", "unsupported", "disabled"}:
                    return False
            nested = _explicit_modality_state(item, modality)
            if nested is not None:
                return nested
        return None
    if isinstance(value, (list, tuple, set)):
        states = [_explicit_modality_state(item, modality) for item in value]
        return True if True in states else (False if False in states else None)
    if isinstance(value, str):
        tokens = set(re.findall(r"[a-z0-9_+-]+", value.lower()))
        aliases = {
            "video": {"video", "video_url", "multiframe", "multi-frame"},
            "image": {"image", "image_url", "vision"},
            "text": {"text", "chat", "language"},
        }.get(modality, {modality})
        return True if tokens.intersection(aliases) else None
    return None


def hosted_model_capabilities(model_id: str, raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve explicit input modalities; unknown is never treated as video-capable."""
    row = raw or {}
    capability_text = _capability_text(row)
    explicit = bool(capability_text.strip())
    capability_values = [
        row.get("capabilities"), row.get("modalities"), row.get("input_modalities"),
        row.get("supported_modalities"), row.get("features"), _top_level_capability_flags(row),
    ]
    video_state = _explicit_modality_state(capability_values, "video")
    image_state = _explicit_modality_state(capability_values, "image")
    text_state = _explicit_modality_state(capability_values, "text")
    supports_video = video_state is True
    supports_image = image_state is True
    supports_text = text_state is not False
    if not explicit:
        curated = next((item for item in CURATED_HOSTED_MODELS if item["id"] == model_id), None)
        if curated:
            supports_video = bool(curated.get("supports_video"))
            supports_image = bool(curated.get("supports_image"))
            supports_text = bool(curated.get("supports_text", True))
            source = "curated"
        else:
            lower = model_id.lower()
            short_id = lower.rsplit("/", 1)[-1]
            known_native_video = short_id in {"cosmos3-nano-reasoner", "cosmos3-super-reasoner"} or bool(
                re.search(
                    r"(?:^|/)nemotron[-_.]?3[-_.]?nano[-_.]?omni[-_.]?30b[-_.]?a3b[-_.]?reasoning(?:$|[-_.:])",
                    lower,
                )
            )
            known_image_model = (
                known_native_video
                or bool(re.search(r"(?:^|/)qwen3(?:[._-]?5[-_.]?397b[-_.]?a17b|[._-]?6[-_.]?35b[-_.]?a3b)(?:$|[-_.:])", lower))
                or bool(re.search(r"(?:^|/)gemma[-_.]?4[-_.]?31b[-_.]?it(?:$|[-_.:])", lower))
                or bool(re.search(r"(?:^|/)gemini[-_.]?3[._-]?6[-_.]?flash(?:$|[-_.:])", lower))
                or bool(re.search(r"(?:^|/)claude[-_.]?opus[-_.]?(?:5|4[-_.]?8)(?:$|[-_.:])", lower))
                or bool(re.search(r"(?:^|/)gpt[-_.]?5[._-]?6[-_.]?sol(?:$|[-_.:])", lower))
                or bool(re.search(r"(?:^|/)kimi[-_.]?k2[._-]?6(?:$|[-_.:])", lower))
            )
            supports_video = known_native_video
            supports_image = known_image_model
            supports_text = True
            source = "confirmed_model_contract" if known_image_model else "unknown"
    else:
        source = "catalog"
    media_strategy = "native_video" if supports_video else ("sampled_frames" if supports_image else None)
    return {
        "supports_video": supports_video,
        "supports_image": supports_image,
        "supports_text": supports_text,
        "capability_source": source,
        "video_candidate": supports_video,
        "comparison_media_strategy": media_strategy,
    }


def curated_hosted_models() -> List[Dict[str, Any]]:
    return [
        dict(
            item,
            source="curated",
            capability_source="curated",
            video_candidate=bool(item.get("supports_video")),
            comparison_media_strategy=(
                "native_video" if item.get("supports_video") is True
                else ("sampled_frames" if item.get("supports_image") is True else None)
            ),
        )
        for item in CURATED_HOSTED_MODELS
    ]


def hosted_auth_headers(api_key: str) -> Dict[str, str]:
    header = NVIDIA_HOSTED_AUTH_HEADER
    if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", header):
        raise RuntimeError("Hosted auth header configuration is invalid")
    scheme = NVIDIA_HOSTED_AUTH_SCHEME
    if scheme and not re.fullmatch(r"[A-Za-z][A-Za-z0-9._~-]{0,31}", scheme):
        raise RuntimeError("Hosted auth scheme configuration is invalid")
    value = f"{scheme} {api_key}".strip() if scheme else api_key
    return {"Accept": "application/json", header: value}


def discover_hosted_models(api_key: Any) -> Dict[str, Any]:
    """Discover serverless models without retaining or returning the runtime key."""
    catalog_url = validated_hosted_endpoint(NVIDIA_HOSTED_CATALOG_URL, "Hosted catalog URL")
    key = validated_runtime_key(api_key)
    if requests is None:
        raise RuntimeError(f"requests import failed: {REQUESTS_IMPORT_ERROR}")
    headers = hosted_auth_headers(key)
    warning: Optional[str] = None
    source = "dynamic"
    try:
        response = requests.get(
            catalog_url,
            headers=headers,
            timeout=HTTP_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
        if response.status_code in (401, 403):
            raise ClientInputError("Hosted model discovery rejected the runtime key")
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Hosted model discovery returned HTTP {response.status_code}")
        body = response.json()
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("Hosted model discovery returned an unexpected response")
        models: List[Dict[str, Any]] = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or row.get("root") or row.get("name") or "").strip()
            if (
                not model_id
                or key in model_id
                or len(model_id) > 240
                or model_id in seen
                or "://" in model_id
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*", model_id)
            ):
                continue
            if redact_sensitive(model_id) != model_id:
                continue
            seen.add(model_id)
            capabilities = hosted_model_capabilities(model_id, row)
            models.append({
                "id": model_id,
                "label": str(redact_runtime_secret(str(row.get("name") or model_id)[:300], key)),
                "family": hosted_model_family(model_id),
                "source": "dynamic",
                **capabilities,
            })
        if not models:
            raise RuntimeError("Hosted model discovery returned no usable models")
        models.sort(key=lambda item: (not item["video_candidate"], item["family"] == "Other", item["family"], item["id"].lower()))
        models = redact_runtime_secret(models, key)
    except ClientInputError:
        raise
    except Exception as exc:
        models = curated_hosted_models()
        source = "curated"
        warning = "Live discovery was unavailable; showing curated defaults."
    finally:
        headers.clear()
        key = ""
    with STATE_LOCK:
        comparison = dict(STATE.get("comparison") or {})
        comparison.update({"catalog": models, "catalog_source": source, "catalog_warning": warning})
        STATE["comparison"] = comparison
    return redact_sensitive({"ok": True, "models": models, "source": source, "warning": warning})


def reference_video_storage_bytes() -> int:
    if not REFERENCE_VIDEO_DIR.exists():
        return 0
    total = 0
    for item in REFERENCE_VIDEO_DIR.iterdir():
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def store_reference_video(stream: Any, length: int, display_name: str) -> Path:
    """Write one bounded upload into a private, aggregate-quota-limited directory."""
    if length <= 0:
        raise ClientInputError("Reference upload is empty")
    if length > REFERENCE_VIDEO_MAX_BYTES:
        raise ClientInputError(
            f"Reference upload exceeds the {REFERENCE_VIDEO_MAX_BYTES // (1024 * 1024)} MiB limit"
        )
    safe_name = Path(display_name).name or "reference.mp4"
    suffix = Path(safe_name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ClientInputError("Reference upload must use a supported video extension")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(safe_name).stem).strip(".-")[:80] or "reference"
    with REFERENCE_UPLOAD_LOCK:
        REFERENCE_VIDEO_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        REFERENCE_VIDEO_DIR.chmod(0o700)
        if reference_video_storage_bytes() + length > REFERENCE_VIDEO_TOTAL_MAX_BYTES:
            raise ClientInputError("Reference upload storage quota is exhausted")
        destination = REFERENCE_VIDEO_DIR / f"{safe_stem}-{os.urandom(6).hex()}{suffix}"
        file_descriptor = os.open(
            str(destination),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        remaining = length
        try:
            os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "wb") as handle:
                file_descriptor = -1
                while remaining > 0:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ClientInputError("Reference upload ended before the declared content length")
                    handle.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            destination.unlink(missing_ok=True)
            raise
    return destination


def register_reference_video(path: Path, display_name: str) -> Dict[str, Any]:
    """Add an already-uploaded reference video without replacing dataset rows."""
    resolved = path.resolve()
    if not resolved.is_file() or resolved.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ClientInputError("Reference upload must be a supported video file")
    if resolved.stat().st_size <= 0:
        raise ClientInputError("Reference upload is empty")
    video = attach_meta({
        "id": video_id(str(resolved)),
        "name": Path(display_name).name or resolved.name,
        "filepath": str(resolved),
        "label": "Reference upload",
        "source": "reference_upload",
        "dataset_row": {"source": "reference_upload", "reference_video": True},
    })
    with STATE_LOCK:
        existing = list(STATE.get("videos") or [])
        existing = [item for item in existing if item.get("id") != video["id"]]
        existing.append(video)
        STATE["videos"] = existing
    try:
        write_private_json(RESULTS_FILE, snapshot())
    except Exception as exc:
        log(f"Could not save reference-video state: {safe_error(exc)}")
    return redact_sensitive(video)


def normalized_media_strategy(value: Any) -> Optional[str]:
    strategy = str(value or "").strip().lower()
    if not strategy or strategy == "auto":
        return None
    if strategy not in {"native_video", "sampled_frames"}:
        raise ClientInputError("Media strategy must be native_video or sampled_frames")
    return strategy


def comparison_catalog_rows() -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("id") or ""): item
        for item in ((snapshot().get("comparison") or {}).get("catalog") or [])
        if isinstance(item, dict) and item.get("id")
    }


def comparison_strategy_for_model(model_id: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[str]:
    override = normalized_media_strategy((overrides or {}).get(model_id))
    if override:
        return override
    row = comparison_catalog_rows().get(model_id) or {}
    strategy = normalized_media_strategy(row.get("comparison_media_strategy"))
    if strategy:
        return strategy
    if row.get("supports_video") is True:
        return "native_video"
    if row.get("supports_image") is True:
        return "sampled_frames"
    return None


def validate_comparison_payload_secrets(payload: Dict[str, Any], runtime_key: str) -> None:
    """Reject credentials pasted into model/workflow fields without echoing them."""
    relevant = {
        "models": payload.get("models"),
        "model_strategies": payload.get("model_strategies"),
        "variants": payload.get("variants"),
        "system_prompt": payload.get("system_prompt"),
        "user_prompt": payload.get("user_prompt"),
        "reasoning_model": payload.get("reasoning_model"),
    }
    serialized = json.dumps(relevant, default=str)
    if runtime_key and runtime_key in serialized:
        raise ClientInputError("Credentials must not be included in model or workflow fields")
    model_fields = json.dumps({
        "models": payload.get("models"),
        "model_strategies": payload.get("model_strategies"),
        "reasoning_model": payload.get("reasoning_model"),
    }, default=str)
    if redact_sensitive(model_fields) != model_fields:
        raise ClientInputError("Credentials must not be included in model fields")


def normalized_comparison_models(values: Iterable[Any], strategy_overrides: Optional[Dict[str, Any]] = None) -> List[str]:
    catalog = {
        str(item.get("id") or ""): item
        for item in ((snapshot().get("comparison") or {}).get("catalog") or [])
        if isinstance(item, dict) and item.get("id")
    }
    models: List[str] = []
    seen = set()
    blocked: List[str] = []
    for value in values or []:
        model_id = str(value or "").strip()
        if not model_id or model_id in seen:
            continue
        if (
            len(model_id) > 240
            or "://" in model_id
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*", model_id)
        ):
            raise ClientInputError("A selected hosted model id is invalid")
        if redact_sensitive(model_id) != model_id:
            raise ClientInputError("A selected hosted model id contains a credential-like value")
        catalog_row = catalog.get(model_id)
        strategy = comparison_strategy_for_model(model_id, strategy_overrides)
        if not strategy or (not catalog_row and not normalized_media_strategy((strategy_overrides or {}).get(model_id))):
            blocked.append(model_id)
            continue
        seen.add(model_id)
        models.append(model_id)
    if blocked:
        raise ClientInputError(
            "Reference-video comparison requires catalog-confirmed native video/image input or an explicit media strategy override; blocked: "
            + ", ".join(blocked[:8])
        )
    if not models:
        raise ClientInputError("Select at least one hosted model")
    return models[:24]


def normalized_comparison_variants(
    mode: str,
    raw_variants: Any,
    system_prompt: str,
    user_prompt: str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    mode = "ablation" if str(mode).lower() == "ablation" else "spot"
    if mode == "spot":
        return [{
            "id": "current-workflow",
            "label": "Current workflow",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "params": dict(params),
        }]
    variants: List[Dict[str, Any]] = []
    if isinstance(raw_variants, list):
        for index, raw in enumerate(raw_variants[:8]):
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label") or f"Variant {index + 1}").strip()[:100]
            variant_system = str(raw.get("system_prompt") if raw.get("system_prompt") is not None else system_prompt)
            variant_user = str(raw.get("user_prompt") if raw.get("user_prompt") is not None else user_prompt)
            variant_params = dict(params)
            overrides = raw.get("params")
            if isinstance(overrides, dict):
                for key in ("fps", "max_pixels", "max_tokens", "temperature", "top_p", "repetition_penalty", "max_frames", "reasoning_enabled", "reasoning_format"):
                    if key in overrides:
                        variant_params[key] = overrides[key]
            if "user_prompt" in raw:
                variant_params["prompt_mode"] = "runtime_form"
            variants.append({
                "id": f"variant-{index + 1}-{hashlib.sha1(label.encode('utf-8', 'ignore')).hexdigest()[:6]}",
                "label": label,
                "system_prompt": variant_system,
                "user_prompt": variant_user,
                "params": variant_params,
            })
    if not variants:
        plain_params = dict(params)
        plain_params.update({"prompt_mode": "runtime_form", "reasoning_enabled": False})
        variants = [
            {
                "id": "current-workflow",
                "label": "Current workflow",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "params": dict(params),
            },
            {
                "id": "plain-prompt",
                "label": "Reasoning scaffold removed",
                "system_prompt": system_prompt,
                "user_prompt": strip_reasoning_wrapper(user_prompt),
                "params": plain_params,
            },
        ]
    return variants


def loaded_comparison_target() -> Dict[str, Any]:
    server = detect_server()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    local_key = os.getenv("VLLM_API_KEY")
    if local_key:
        headers["Authorization"] = f"Bearer {local_key}"
    raw_model = str(server.get("model") or os.getenv("MODEL_NAME") or "loaded-model")
    model_path = raw_model.removeprefix("file://")
    safe_model = (
        Path(model_path).name or "loaded-model"
        if raw_model.startswith("file://") or Path(model_path).is_absolute()
        else raw_model
    )
    return {
        "source": "loaded",
        "label": "Loaded endpoint baseline",
        "model": safe_model,
        "backend": str(server.get("backend") or INFERENCE_BACKEND),
        "base_url": str(server.get("base_url") or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")),
        "headers": headers,
        "_runtime_secret": local_key or "",
    }


def _prompt_request_shape(text: Any) -> Dict[str, Any]:
    encoded = str(text or "").encode("utf-8", "ignore")
    return {
        "chars": len(str(text or "")),
        "sha256_16": hashlib.sha256(encoded).hexdigest()[:16],
    }


def comparison_request_shape(
    video: Dict[str, Any],
    target: Dict[str, Any],
    variant: Dict[str, Any],
    *,
    effective_system: Optional[str] = None,
    effective_user: Optional[str] = None,
    content: Optional[List[Dict[str, Any]]] = None,
    payload: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe a comparison request without prompts, auth, media bytes, or endpoint hosts."""
    backend = str(target.get("backend") or "")
    media_strategy = str(
        (plan or {}).get("media_strategy")
        or target.get("media_strategy")
        or "loaded_endpoint_default"
    )
    system_text = str(
        effective_system if effective_system is not None else variant.get("system_prompt") or ""
    )
    user_text = str(
        effective_user if effective_user is not None else variant.get("user_prompt") or ""
    )
    if content is None:
        inferred_type = "video_url" if media_strategy == "native_video" else "image_url"
        content_types = [inferred_type, "text"]
    else:
        content_types = list(dict.fromkeys(
            str(item.get("type") or "unknown") for item in content if isinstance(item, dict)
        ))
    request_parameters: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in ("temperature", "top_p", "max_tokens", "repetition_penalty"):
            if key in payload:
                request_parameters[key] = payload[key]
        if isinstance(payload.get("nvext"), dict):
            request_parameters["nvext"] = {
                key: value for key, value in payload["nvext"].items()
                if key in {"repetition_penalty"} and isinstance(value, (bool, int, float))
            }
    else:
        params = dict(variant.get("params") or {})
        for key in ("temperature", "top_p", "max_tokens", "repetition_penalty", "fps", "max_pixels", "max_frames"):
            value = params.get(key)
            if isinstance(value, (bool, int, float)):
                request_parameters[key] = value
    raw_path = urllib.parse.urlsplit(str(target.get("base_url") or "")).path.rstrip("/")
    if backend.lower().startswith("cosmos3") or backend.lower() == "cosmos3_native":
        transport = "cosmos_generate"
        request_path = raw_path or "/generate"
    else:
        transport = "openai_chat_completions"
        request_path = raw_path if raw_path.endswith("/chat/completions") else raw_path + "/chat/completions"
    media_plan = plan or {}
    return {
        "transport": transport,
        "path": request_path or "/chat/completions",
        "model": str(target.get("model") or ""),
        "messages": [
            {"role": "system", "content_types": ["text"], "text": _prompt_request_shape(system_text)},
            {"role": "user", "content_types": content_types, "text": _prompt_request_shape(user_text)},
        ],
        "parameters": request_parameters,
        "media": {
            "name": Path(str(video.get("name") or video.get("filepath") or "video")).name,
            "source": str(video.get("source") or "dataset"),
            "strategy": media_strategy,
            "frame_count": media_plan.get("frames_passed"),
            "payload": "omitted",
        },
    }


def run_comparison_case(
    video: Dict[str, Any],
    target: Dict[str, Any],
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one video/variant against one ephemeral target configuration."""
    started = time.monotonic()
    model = str(target.get("model") or "")
    backend = str(target.get("backend") or "")
    base_url = str(target.get("base_url") or "")
    if backend.lower() == "hosted_openai":
        base_url = validated_hosted_endpoint(base_url, "Hosted inference URL")
    media_strategy = str(target.get("media_strategy") or ("native_video" if model_uses_native_video(model, backend) else "sampled_frames"))
    headers = dict(target.get("headers") or {})
    params = dict(variant.get("params") or {})
    params.setdefault("reasoning_model", model)
    params.setdefault("reasoning_backend", backend)
    effective_system, effective_user, prompt_source = prompts_for_video(
        video,
        str(variant.get("system_prompt") or ""),
        str(variant.get("user_prompt") or ""),
        params,
    )
    fps = float(params.get("fps") or STATE["defaults"]["fps"])
    max_pixels = int(params.get("max_pixels") or STATE["defaults"]["max_pixels"])
    max_frames = int(params.get("max_frames") if params.get("max_frames") is not None else STATE["defaults"]["max_frames"])
    if media_strategy == "sampled_frames":
        content, plan = content_for_video(
            video["filepath"], effective_user, model, fps, max_pixels, max_frames, backend=backend, force_frames=True
        )
        # Hosted image-model examples put media first and the question last.
        content = [item for item in content if item.get("type") == "image_url"] + [
            item for item in content if item.get("type") == "text"
        ]
        plan["mode"] = "sampled_image_frames"
    elif media_strategy == "native_video":
        content, plan = content_for_video(
            video["filepath"], effective_user, model, fps, max_pixels, max_frames,
            backend=backend, force_native_video=True,
        )
    else:
        raise ClientInputError(f"Unsupported comparison media strategy: {media_strategy}")
    plan["media_strategy"] = media_strategy
    preprocessing_seconds = time.monotonic() - started
    if backend.lower().startswith("cosmos3") or backend.lower() == "cosmos3_native":
        completion = post_cosmos3_generate(
            base_url,
            headers,
            vision_path=video["filepath"],
            prompt=effective_user,
            params=params,
            run_name=f"compare-{int(time.time() * 1000)}-{video['id']}",
        )
    else:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": content},
            ],
            "temperature": float(params.get("temperature") if params.get("temperature") is not None else STATE["defaults"]["temperature"]),
            "top_p": float(params.get("top_p") if params.get("top_p") is not None else STATE["defaults"]["top_p"]),
        }
        repetition_penalty = float(params.get("repetition_penalty") if params.get("repetition_penalty") is not None else STATE["defaults"]["repetition_penalty"])
        if backend.lower() == "hosted_openai":
            payload["max_tokens"] = int(params.get("max_tokens") or STATE["defaults"]["max_tokens"])
        elif "nim" in backend.lower():
            payload["nvext"] = {"repetition_penalty": repetition_penalty}
        else:
            payload["max_tokens"] = int(params.get("max_tokens") or STATE["defaults"]["max_tokens"])
            payload["repetition_penalty"] = repetition_penalty
        try:
            completion = post_chat_completion(
                base_url,
                headers,
                payload,
                allow_redirects=backend.lower() != "hosted_openai",
            )
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if "nim" in backend.lower() and status_code in (400, 422) and plan.get("mode") == "native_video_url":
                fallback_content, fallback_plan = content_for_video(
                    video["filepath"], effective_user, model, fps, max_pixels, 0, backend=backend, force_frames=True
                )
                fallback_content = [item for item in fallback_content if item.get("type") == "image_url"] + [
                    item for item in fallback_content if item.get("type") == "text"
                ]
                fallback_plan["fallback_from_video_url"] = status_code
                payload["messages"][1]["content"] = fallback_content
                completion = post_chat_completion(
                    base_url,
                    headers,
                    payload,
                    allow_redirects=backend.lower() != "hosted_openai",
                )
                plan = fallback_plan
                media_strategy = "sampled_frames_fallback"
                plan["media_strategy"] = media_strategy
            else:
                raise
    response_text = str(completion.get("text") or "")
    metrics = dict(completion.get("metrics") or {})
    metrics["preprocessing_seconds"] = preprocessing_seconds
    metrics["e2e_seconds"] = time.monotonic() - started
    eval_input = {
        "response": response_text,
        "json": parse_json_from_text(response_text),
        "error": None,
    }
    return redact_comparison_payload({
        "video_id": video.get("id"),
        "video_name": video.get("name"),
        "video_source": video.get("source") or "dataset",
        "source": target.get("source"),
        "target": target.get("label"),
        "model": model,
        "media_strategy": media_strategy,
        "variant_id": variant.get("id"),
        "variant": variant.get("label"),
        "status": "ok",
        "latency_seconds": metrics.get("e2e_seconds"),
        "ttft_seconds": metrics.get("ttft_seconds"),
        "metrics": metrics,
        "plan": plan,
        "request_shape": comparison_request_shape(
            video,
            target,
            variant,
            effective_system=effective_system,
            effective_user=effective_user,
            content=content,
            payload=payload if not (backend.lower().startswith("cosmos3") or backend.lower() == "cosmos3_native") else None,
            plan=plan,
        ),
        "prompt_source": prompt_source,
        "response": response_text,
        "evaluation": evaluate_result(video, eval_input),
        "error": None,
    }, str(target.get("_runtime_secret") or ""), base_url)


def run_comparison_case_recorded(
    video: Dict[str, Any],
    target: Dict[str, Any],
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        return run_comparison_case(video, target, variant)
    except Exception as exc:
        secret = str(target.get("_runtime_secret") or "")
        return redact_comparison_payload({
            "video_id": video.get("id"),
            "video_name": video.get("name"),
            "video_source": video.get("source") or "dataset",
            "source": target.get("source"),
            "target": target.get("label"),
            "model": target.get("model"),
            "media_strategy": target.get("media_strategy") or "loaded_endpoint_default",
            "variant_id": variant.get("id"),
            "variant": variant.get("label"),
            "status": "error",
            "latency_seconds": time.monotonic() - started,
            "ttft_seconds": None,
            "metrics": {"e2e_seconds": time.monotonic() - started},
            "request_shape": comparison_request_shape(video, target, variant),
            "response": "",
            "evaluation": {},
            "error": safe_comparison_error_with_prompts(
                exc,
                secret,
                str(target.get("base_url") or ""),
                (variant.get("system_prompt"), variant.get("user_prompt")),
            ),
        }, secret, str(target.get("base_url") or ""))


def comparison_summary(results: List[Dict[str, Any]], wall_seconds: float) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in results:
        key = " | ".join((str(row.get("source") or ""), str(row.get("model") or ""), str(row.get("media_strategy") or ""), str(row.get("variant") or "")))
        item = grouped.setdefault(key, {
            "source": row.get("source"),
            "model": row.get("model"),
            "media_strategy": row.get("media_strategy"),
            "variant": row.get("variant"),
            "total": 0,
            "ok": 0,
            "errors": 0,
            "latencies": [],
            "error_messages": [],
        })
        item["total"] += 1
        if row.get("status") == "ok":
            item["ok"] += 1
        else:
            item["errors"] += 1
            if row.get("error"):
                item["error_messages"].append(safe_error(row["error"], 300))
        if row.get("latency_seconds") is not None:
            item["latencies"].append(row["latency_seconds"])
    rows: List[Dict[str, Any]] = []
    for item in grouped.values():
        latencies = item.pop("latencies")
        item["latency_seconds"] = stats_summary(latencies)
        item["error_messages"] = list(dict.fromkeys(item["error_messages"]))[:10]
        rows.append(item)
    rows.sort(key=lambda item: (item["source"] != "loaded", str(item["model"]), str(item["media_strategy"]), str(item["variant"])))
    errors = sum(1 for row in results if row.get("status") != "ok")
    return redact_sensitive({
        "total": len(results),
        "ok": len(results) - errors,
        "errors": errors,
        "wall_seconds": wall_seconds,
        "groups": rows,
    })


def run_hosted_comparison(
    ids: Iterable[str],
    hosted_models: List[str],
    media_strategies: Dict[str, str],
    api_key: str,
    mode: str,
    variants: List[Dict[str, Any]],
    concurrency: int,
    include_loaded: bool,
) -> None:
    started = time.monotonic()
    started_epoch = time.time()
    run_id = f"compare-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
    videos = selected_videos_for_ids(ids)
    targets: List[Dict[str, Any]] = []
    try:
        if include_loaded:
            targets.append(loaded_comparison_target())
        hosted_url = validated_hosted_endpoint(NVIDIA_HOSTED_INFERENCE_URL, "Hosted inference URL")
        hosted_headers = {"Content-Type": "application/json", **hosted_auth_headers(api_key)}
        targets.extend({
            "source": "hosted_endpoint",
            "label": "Hosted endpoint",
            "model": model,
            "backend": "hosted_openai",
            "media_strategy": media_strategies.get(model) or "native_video",
            "base_url": hosted_url,
            "headers": dict(hosted_headers),
            "_runtime_secret": api_key,
        } for model in hosted_models)
        hosted_headers.clear()
        total = len(videos) * len(targets) * len(variants)
        if not videos:
            raise ClientInputError("Load or upload at least one video before comparing endpoints")
        if not targets:
            raise ClientInputError("No comparison targets were selected")
        if total > COMPARISON_MAX_CASES:
            raise ClientInputError(
                f"Comparison expands to {total} cases; reduce videos, models, or variants to {COMPARISON_MAX_CASES} or fewer"
            )
        run_meta = {
            "run_id": run_id,
            "mode": "ablation" if mode == "ablation" else "spot",
            "status": "running",
            "started_epoch": started_epoch,
            "video_count": len(videos),
            "models": [{"source": target["source"], "model": target["model"], "media_strategy": target.get("media_strategy") or "loaded_endpoint_default"} for target in targets],
            "variants": [{"id": variant["id"], "label": variant["label"]} for variant in variants],
            "progress": {"done": 0, "total": total, "errors": 0},
        }
        with STATE_LOCK:
            comparison = dict(STATE.get("comparison") or {})
            comparison.update({"running": True, "run": run_meta, "results": []})
            STATE["comparison"] = comparison
        log(f"Comparison {run_id} started: {total} cases")
        results: List[Dict[str, Any]] = []
        errors = 0
        work = [(video, target, variant) for video in videos for target in targets for variant in variants]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(16, int(concurrency)))) as pool:
            future_map = {
                pool.submit(run_comparison_case_recorded, video, target, variant): (video, target, variant)
                for video, target, variant in work
            }
            for future in concurrent.futures.as_completed(future_map):
                video, target, variant = future_map[future]
                try:
                    result = future.result()
                    if result.get("status") != "ok":
                        errors += 1
                except Exception as exc:
                    errors += 1
                    result = redact_comparison_payload({
                        "video_id": video.get("id"),
                        "video_name": video.get("name"),
                        "video_source": video.get("source") or "dataset",
                        "source": target.get("source"),
                        "target": target.get("label"),
                        "model": target.get("model"),
                        "media_strategy": target.get("media_strategy") or "loaded_endpoint_default",
                        "variant_id": variant.get("id"),
                        "variant": variant.get("label"),
                        "status": "error",
                        "latency_seconds": None,
                        "ttft_seconds": None,
                        "metrics": {},
                        "request_shape": comparison_request_shape(video, target, variant),
                        "response": "",
                        "evaluation": {},
                        "error": safe_comparison_error_with_prompts(
                            exc,
                            str(target.get("_runtime_secret") or ""),
                            str(target.get("base_url") or ""),
                            (variant.get("system_prompt"), variant.get("user_prompt")),
                        ),
                    }, str(target.get("_runtime_secret") or ""), str(target.get("base_url") or ""))
                results.append(result)
                with STATE_LOCK:
                    comparison = dict(STATE.get("comparison") or {})
                    current_run = dict(comparison.get("run") or run_meta)
                    current_run["progress"] = {"done": len(results), "total": total, "errors": errors}
                    comparison.update({"run": current_run, "results": list(results)})
                    STATE["comparison"] = comparison
        wall_seconds = time.monotonic() - started
        summary = comparison_summary(results, wall_seconds)
        run_meta.update({
            "status": "complete",
            "finished_epoch": time.time(),
            "wall_seconds": wall_seconds,
            "progress": {"done": len(results), "total": total, "errors": errors},
            "summary": summary,
        })
        with STATE_LOCK:
            comparison = dict(STATE.get("comparison") or {})
            history = list(comparison.get("history") or [])[-19:]
            history.append({
                "run_id": run_id,
                "mode": run_meta["mode"],
                "status": "complete",
                "started_epoch": started_epoch,
                "summary": summary,
            })
            comparison.update({"running": False, "run": run_meta, "results": results, "history": history})
            STATE["comparison"] = comparison
        write_private_json(RESULTS_FILE, snapshot())
        log(f"Comparison {run_id} complete: {len(results) - errors} ok, {errors} errors")
    except Exception as exc:
        error = safe_comparison_error(exc, api_key, NVIDIA_HOSTED_INFERENCE_URL)
        with STATE_LOCK:
            comparison = dict(STATE.get("comparison") or {})
            current_run = dict(comparison.get("run") or {})
            current_run.update({"run_id": run_id, "status": "error", "error": error, "finished_epoch": time.time()})
            comparison.update({"running": False, "run": current_run})
            STATE["comparison"] = comparison
        log(f"Comparison {run_id} failed: {error}")
    finally:
        for target in targets:
            headers = target.get("headers")
            if isinstance(headers, dict):
                headers.clear()
            target["_runtime_secret"] = ""
        api_key = ""


def comparison_export_payload() -> Dict[str, Any]:
    comparison = snapshot().get("comparison") or {}
    run = comparison.get("run") or {}
    if not run or not comparison.get("results"):
        raise ClientInputError("Run a comparison before exporting")
    return redact_sensitive({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": run,
        "summary": run.get("summary") or comparison_summary(comparison.get("results") or [], float(run.get("wall_seconds") or 0)),
        "results": comparison.get("results") or [],
    })


def create_comparison_export(fmt: str) -> Path:
    fmt = str(fmt or "html").lower()
    if fmt not in {"html", "json", "csv"}:
        raise ClientInputError("Comparison export format must be html, json, or csv")
    payload = comparison_export_payload()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        EXPORT_DIR.chmod(0o700)
    except OSError:
        pass
    path = EXPORT_DIR / f"byo-video-comparison-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}.{fmt}"
    if fmt == "json":
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif fmt == "csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video", "video_source", "endpoint_source", "model", "media_strategy", "variant", "status", "latency_seconds", "ttft_seconds", "error", "response"])
            for row in payload["results"]:
                writer.writerow([
                    row.get("video_name"), row.get("video_source"), row.get("source"), row.get("model"), row.get("media_strategy"),
                    row.get("variant"), row.get("status"), row.get("latency_seconds"), row.get("ttft_seconds"),
                    row.get("error"), row.get("response"),
                ])
    else:
        summary_rows = []
        for item in (payload.get("summary") or {}).get("groups") or []:
            latency = item.get("latency_seconds") or {}
            summary_rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('source') or ''))}</td>"
                f"<td>{html.escape(str(item.get('model') or ''))}</td>"
                f"<td>{html.escape(str(item.get('media_strategy') or ''))}</td>"
                f"<td>{html.escape(str(item.get('variant') or ''))}</td>"
                f"<td>{html.escape(str(item.get('ok') or 0))}/{html.escape(str(item.get('total') or 0))}</td>"
                f"<td>{html.escape(str(round(float(latency.get('average') or 0), 3)))}</td>"
                f"<td>{html.escape(' | '.join(item.get('error_messages') or []))}</td>"
                "</tr>"
            )
        result_rows_html = []
        for row in payload["results"]:
            result_rows_html.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('video_name') or ''))}</td>"
                f"<td>{html.escape(str(row.get('source') or ''))}</td>"
                f"<td>{html.escape(str(row.get('model') or ''))}</td>"
                f"<td>{html.escape(str(row.get('media_strategy') or ''))}</td>"
                f"<td>{html.escape(str(row.get('variant') or ''))}</td>"
                f"<td>{html.escape(str(row.get('status') or ''))}</td>"
                f"<td>{html.escape(str(round(float(row.get('latency_seconds') or 0), 3)))}</td>"
                f"<td>{html.escape(str(row.get('error') or ''))}</td>"
                f"<td><pre>{html.escape(str(row.get('response') or ''))}</pre></td>"
                "</tr>"
            )
        run = payload.get("run") or {}
        path.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>BYO Video Endpoint Comparison</title>"
            "<style>body{font:14px system-ui;margin:32px;color:#1f2937}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #d8dee8;padding:8px;vertical-align:top;text-align:left}pre{white-space:pre-wrap;max-width:70ch;margin:0}</style>"
            "</head><body><h1>BYO Video Endpoint Comparison</h1>"
            f"<p>Run {html.escape(str(run.get('run_id') or ''))} · {html.escape(str(run.get('mode') or ''))} · status {html.escape(str(run.get('status') or ''))}</p>"
            "<h2>Summary</h2><table><thead><tr><th>Source</th><th>Model</th><th>Media strategy</th><th>Variant</th><th>OK</th><th>Avg latency (s)</th><th>Errors</th></tr></thead><tbody>"
            + "".join(summary_rows)
            + "</tbody></table><h2>Results</h2><table><thead><tr><th>Video</th><th>Source</th><th>Model</th><th>Media strategy</th><th>Variant</th><th>Status</th><th>Latency (s)</th><th>Error</th><th>Response</th></tr></thead><tbody>"
            + "".join(result_rows_html)
            + "</tbody></table></body></html>",
            encoding="utf-8",
        )
    # Defense in depth: verify serialized artifacts do not contain known secret shapes.
    if fmt in {"html", "json", "csv"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        redacted = str(redact_sensitive(raw))
        if redacted != raw:
            path.write_text(redacted, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


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
    if 200 <= resp.status_code < 300:
        return
    if 300 <= resp.status_code < 400:
        raise requests.HTTPError(
            f"{resp.status_code} redirect refused for configured endpoint",
            response=resp,
        )
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


def post_chat_completion(
    base_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    *,
    allow_redirects: bool = True,
) -> Dict[str, Any]:
    normalized_url = base_url.rstrip("/")
    url = normalized_url if normalized_url.endswith("/chat/completions") else normalized_url + "/chat/completions"
    request_started = time.monotonic()
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    stream_payload["stream_options"] = {"include_usage": True}
    resp = requests.post(
        url,
        headers=headers,
        json=stream_payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
        stream=True,
        allow_redirects=allow_redirects,
    )
    if resp.status_code >= 400 and "stream_options" in (resp.text or ""):
        stream_payload.pop("stream_options", None)
        resp = requests.post(
            url,
            headers=headers,
            json=stream_payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=allow_redirects,
        )
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
            write_private_json(RESULTS_FILE, snapshot())
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
    write_private_json(RESULTS_FILE, snapshot())
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
RF100_VL_DATA_ROOTS = [
    Path(p) for p in (
        os.getenv("RF100_VL_DATA_ROOT") or "",
        "/home/horde/rf100-vl",
        "/tmp/rf100-vl",
        str(BENCHMARK_RUN_ROOT / "rf100-vl"),
    ) if p
]
RF100_VL_EXPECTED_GROUPS = int(os.getenv("RF100_VL_EXPECTED_GROUPS", "100"))
RF100_VL_AUTO_DOWNLOAD = os.getenv("RF100_VL_AUTO_DOWNLOAD", "1").lower() not in ("0", "false", "no")
RF100_VL_DISCOVERY_CACHE = BENCHMARK_RUN_ROOT / "rf100-vl-profile.json"
RF100_VL_MAX_PIXELS = int(os.getenv("RF100_VL_MAX_PIXELS", str(128 * (32 ** 2))))
RF100_VL_MIN_PIXELS = int(os.getenv("RF100_VL_MIN_PIXELS", str(32 * (32 ** 2))))
RF100_VL_JPEG_QUALITY = int(os.getenv("RF100_VL_JPEG_QUALITY", "82"))
RF100_VL_PROMPT_MAX_CATEGORIES = int(os.getenv("RF100_VL_PROMPT_MAX_CATEGORIES", "160"))
RF100_VL_PROMPT_MAX_CHARS = int(os.getenv("RF100_VL_PROMPT_MAX_CHARS", "6000"))
RF100_VL_CONTEXT_RETRY_ATTEMPTS = int(os.getenv("RF100_VL_CONTEXT_RETRY_ATTEMPTS", "4"))
RF100_VL_TRANSIENT_RETRIES = int(os.getenv("RF100_VL_TRANSIENT_RETRIES", "1"))
RF100_VL_MAX_IN_FLIGHT_MULTIPLIER = int(os.getenv("RF100_VL_MAX_IN_FLIGHT_MULTIPLIER", "3"))
RF100_VL_REQUEST_TIMEOUT_SECONDS = float(os.getenv("RF100_VL_REQUEST_TIMEOUT_SECONDS", "240"))
RF100_VL_USE_PROJECT_CATEGORIES = os.getenv("RF100_VL_USE_PROJECT_CATEGORIES", "0").lower() in ("1", "true", "yes")
BENCHMARK_FLUSH_EVERY = int(os.getenv("BENCHMARK_FLUSH_EVERY", "100"))
BENCHMARK_FLUSH_SECONDS = float(os.getenv("BENCHMARK_FLUSH_SECONDS", "180"))
BENCHMARK_DURABLE_FLUSH_EVERY = int(os.getenv("BENCHMARK_DURABLE_FLUSH_EVERY", "500"))
BENCHMARK_DURABLE_FLUSH_SECONDS = float(os.getenv("BENCHMARK_DURABLE_FLUSH_SECONDS", "900"))
RF100_GATE_STATUS_PATH = Path(os.getenv("RF100_GATE_STATUS_PATH", "/tmp/rf100_last_heartbeat_status.json"))
AIR_SUPPORT_MIN_STABLE_SECONDS = int(os.getenv("AIR_SUPPORT_MIN_STABLE_SECONDS", str(30 * 60)))
AIR_SUPPORT_MIN_STABLE_IMAGES = int(os.getenv("AIR_SUPPORT_MIN_STABLE_IMAGES", "500"))
AIR_SUPPORT_MAX_ERROR_RATE = float(os.getenv("AIR_SUPPORT_MAX_ERROR_RATE", "0.01"))
CLAUDE_DOCS_PRESENTATIONS_DIR = Path(
    os.getenv(
        "CLAUDE_DOCS_PRESENTATIONS_DIR",
        "/Users/asotelo/Library/CloudStorage/GoogleDrive-asotelo@nvidia.com/My Drive/Claude Docs/presentations",
    )
)

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
    """Collapse LingoQA reference rows using the official evaluator key.

    The upstream evaluator groups references by
    (question_id, segment_id, question), not just question_id. Keeping the same
    key preserves the expected 500-row validation shape and avoids merging
    distinct clips that happen to reuse a question id.

    Returns dicts: {question_id, segment_id, images, question, references: [a,b], category}.
    """
    bucket: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str, str]] = []
    for s in samples:
        key = (str(s["question_id"]), str(s["segment_id"]), str(s["question"]))
        if key not in bucket:
            bucket[key] = {
                "question_id": key[0],
                "segment_id": s["segment_id"],
                "images": s["images"],
                "question": s["question"],
                "references": [],
                "category": s["category"],
            }
            order.append(key)
        answer = str(s.get("answer") or "")
        if answer and answer not in bucket[key]["references"]:
            bucket[key]["references"].append(answer)
    return [bucket[key] for key in order]


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


def _is_rf100_source_text(value: str) -> bool:
    low = (value or "").strip().lower()
    return (
        low in {"rf100-vl", "rf100vl", "roboflow100-vl", "roboflow100vl"}
        or "rf100-vl.org" in low
        or "roboflow/rf100-vl" in low
        or "2505.20612" in low
    )


def _is_lingoqa_source_text(value: str) -> bool:
    low = (value or "").strip().lower()
    return low == "lingoqa" or "wayveai/lingoqa" in low or "2312.14115" in low


def _rf100_dependency_status() -> Dict[str, Any]:
    out = {
        "package": "rf100vl",
        "package_ready": False,
        "roboflow_api_key_ready": bool(os.getenv("ROBOFLOW_API_KEY")),
        "auto_download": bool(RF100_VL_AUTO_DOWNLOAD),
        "blockers": [],
    }
    try:
        __import__("rf100vl")
        out["package_ready"] = True
    except Exception as exc:
        out["package_error"] = str(exc)
        out["blockers"].append("Install the rf100vl package on the benchmark host.")
    if not out["roboflow_api_key_ready"]:
        out["blockers"].append("Set ROBOFLOW_API_KEY on the benchmark host before downloading RF100-VL.")
    return out


def _rf100_candidate_annotation_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    patterns = [
        "**/_annotations.coco.json",
        "**/*_annotations*.json",
        "**/annotations/*.json",
        "**/instances*.json",
    ]
    seen: set = set()
    files: List[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            if path.name == "_annotations.coco.json":
                files.append(path)
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict) and data.get("images") and data.get("annotations") and data.get("categories"):
                files.append(path)
    return files


def _rf100_discovery_signature(root: Path, files: List[Path]) -> Dict[str, Any]:
    latest_mtime = 0.0
    for path in files:
        try:
            latest_mtime = max(latest_mtime, path.stat().st_mtime)
        except OSError:
            continue
    return {
        "root": str(root),
        "annotation_file_count": len(files),
        "latest_annotation_mtime": latest_mtime,
    }


def _rf100_find_root() -> Optional[Path]:
    for root in RF100_VL_DATA_ROOTS:
        if not root.exists():
            continue
        if _rf100_candidate_annotation_files(root):
            return root
        for child in sorted(root.iterdir()) if root.is_dir() else []:
            if child.is_dir() and _rf100_candidate_annotation_files(child):
                return child
    return None


def _rf100_download_if_possible(force_complete: bool = False) -> Path:
    existing = _rf100_find_root()
    if existing is not None and not force_complete:
        return existing
    if existing is not None and force_complete:
        profile = _rf100_discover(download=False)
        if int(profile.get("group_count") or 0) >= RF100_VL_EXPECTED_GROUPS:
            return existing
    dep = _rf100_dependency_status()
    if not RF100_VL_AUTO_DOWNLOAD:
        raise RuntimeError("RF100-VL data is not staged and RF100_VL_AUTO_DOWNLOAD=0.")
    if dep.get("blockers"):
        raise RuntimeError("RF100-VL cannot be downloaded yet: " + " ".join(dep["blockers"]))
    target = RF100_VL_DATA_ROOTS[0]
    target.mkdir(parents=True, exist_ok=True)
    try:
        from rf100vl import download_rf100vl  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"rf100vl import failed: {exc}") from exc
    log(f"[rf100-vl] downloading full dataset to {target}")
    download_rf100vl(path=str(target), api_key=os.getenv("ROBOFLOW_API_KEY"))
    found = _rf100_find_root()
    if found is None:
        raise RuntimeError(f"RF100-VL download finished but no COCO annotations were found under {target}")
    return found


def _rf100_group_name(root: Path, annotation_path: Path) -> str:
    parent = annotation_path.parent
    if parent.name.lower() in {"train", "valid", "val", "test"} and parent.parent != root:
        return parent.parent.name
    if parent.name.lower() == "annotations" and parent.parent != root:
        return parent.parent.name
    return parent.name if parent != root else annotation_path.stem


def _rf100_split_name(annotation_path: Path) -> str:
    parent = annotation_path.parent.name.lower()
    if parent in {"train", "valid", "val", "test"}:
        return "valid" if parent == "val" else parent
    return "unknown"


def _rf100_project_categories() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        from rf100vl import get_rf100vl_projects  # type: ignore
        projects = get_rf100vl_projects(api_key=os.getenv("ROBOFLOW_API_KEY"))
    except Exception:
        return mapping
    for item in projects or []:
        name = (
            getattr(item, "name", None)
            or getattr(item, "project_name", None)
            or getattr(item, "slug", None)
            or ""
        )
        category = (
            getattr(item, "category", None)
            or getattr(item, "domain", None)
            or getattr(item, "group", None)
            or ""
        )
        if name and category:
            mapping[str(name)] = str(category)
            mapping[_safe_dataset_dir_name(str(name)).lower()] = str(category)
    return mapping


def _rf100_image_path(annotation_path: Path, file_name: str) -> str:
    file_name = str(file_name or "")
    candidates = [
        annotation_path.parent / file_name,
        annotation_path.parent.parent / file_name,
        annotation_path.parent / Path(file_name).name,
        annotation_path.parent.parent / Path(file_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _rf100_prompt(
    categories: List[Dict[str, Any]],
    dataset_name: str,
    domain: str,
    *,
    max_categories: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    limit = max_categories if max_categories is not None else RF100_VL_PROMPT_MAX_CATEGORIES
    char_limit = max_chars if max_chars is not None else RF100_VL_PROMPT_MAX_CHARS
    category_entries = [
        f"{int(c.get('id'))}:{c.get('name')}"
        for c in categories[: max(0, limit)]
    ]
    category_text = "; ".join(category_entries)
    omitted = max(0, len(categories) - len(category_entries))
    if char_limit > 0 and len(category_text) > char_limit:
        clipped: List[str] = []
        used = 0
        for entry in category_entries:
            extra = len(entry) + (2 if clipped else 0)
            if used + extra > char_limit:
                omitted += 1
                continue
            clipped.append(entry)
            used += extra
        category_text = "; ".join(clipped)
    if omitted:
        category_text += f"; ... {omitted} categories omitted to stay within context"
    return (
        "You are evaluating RF100-VL object detection. Detect every visible instance "
        "of the requested categories in this image.\n"
        f"Dataset: {dataset_name}\n"
        f"Domain: {domain or 'unknown'}\n"
        "Return only valid JSON with this schema:\n"
        '{"detections":[{"category_id":integer,"category_name":"string","bbox":[x,y,width,height],"score":number}]}\n'
        "Use pixel coordinates relative to the image you see. Do not include prose outside the JSON.\n"
        "Categories:\n"
        f"{category_text}"
    )


def _rf100_discover(download: bool = False) -> Dict[str, Any]:
    root = _rf100_download_if_possible(force_complete=True) if download else _rf100_find_root()
    dep = _rf100_dependency_status()
    profile: Dict[str, Any] = {
        "adapter": "rf100-vl",
        "id": "rf100-vl",
        "name": "RF100-VL",
        "task": "object_detection",
        "metric": "COCO-style AP@[.50:.95]",
        "source_urls": [
            "https://rf100-vl.org/",
            "https://github.com/roboflow/rf100-vl",
            "https://arxiv.org/abs/2505.20612",
        ],
        "ready": False,
        "path": str(root) if root else None,
        "groups": [],
        "group_count": 0,
        "image_count": 0,
        "annotation_count": 0,
        "expected_groups": RF100_VL_EXPECTED_GROUPS,
        "full_gate_ready": False,
        "dependencies": dep,
        "blockers": list(dep.get("blockers") or []),
    }
    if root is None:
        profile["blockers"].insert(0, "RF100-VL COCO annotations were not found on this host.")
        return profile
    ann_files = _rf100_candidate_annotation_files(root)
    signature = _rf100_discovery_signature(root, ann_files)
    if not download and RF100_VL_DISCOVERY_CACHE.exists():
        try:
            cached = json.loads(RF100_VL_DISCOVERY_CACHE.read_text(encoding="utf-8"))
            if cached.get("_signature") == signature:
                cached["dependencies"] = dep
                cached["blockers"] = [
                    b for b in (cached.get("blockers") or [])
                    if "Install the rf100vl package" not in str(b)
                    and "ROBOFLOW_API_KEY" not in str(b)
                ]
                if dep.get("blockers") and not cached.get("ready"):
                    cached["blockers"].extend(dep.get("blockers") or [])
                return cached
        except Exception:
            pass
    category_map = _rf100_project_categories()
    groups: Dict[str, Dict[str, Any]] = {}
    image_count = 0
    annotation_count = 0
    for ann in ann_files:
        try:
            data = json.loads(ann.read_text(encoding="utf-8"))
        except Exception:
            continue
        group = _rf100_group_name(root, ann)
        split = _rf100_split_name(ann)
        key = group
        domain = category_map.get(group) or category_map.get(_safe_dataset_dir_name(group).lower()) or "uncategorized"
        bucket = groups.setdefault(key, {
            "id": key,
            "name": group,
            "domain": domain,
            "splits": [],
            "images": 0,
            "annotations": 0,
            "categories": len(data.get("categories") or []),
        })
        if split not in bucket["splits"]:
            bucket["splits"].append(split)
        bucket["images"] += len(data.get("images") or [])
        bucket["annotations"] += len(data.get("annotations") or [])
        image_count += len(data.get("images") or [])
        annotation_count += len(data.get("annotations") or [])
    group_list = sorted(groups.values(), key=lambda g: g["name"])
    profile.update({
        "ready": bool(group_list),
        "groups": group_list,
        "group_count": len(group_list),
        "image_count": image_count,
        "annotation_count": annotation_count,
        "full_gate_ready": len(group_list) >= RF100_VL_EXPECTED_GROUPS,
    })
    if len(group_list) < RF100_VL_EXPECTED_GROUPS:
        profile["blockers"].append(
            f"Full RF100-VL gate requires {RF100_VL_EXPECTED_GROUPS} groups; found {len(group_list)}."
        )
    profile["_signature"] = signature
    try:
        RF100_VL_DISCOVERY_CACHE.write_text(json.dumps(profile, default=str, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"[rf100-vl] discovery cache write failed: {exc}")
    return profile


def _load_rf100vl_dataset(
    source_config: Optional[Dict[str, Any]] = None,
    *,
    download: bool = False,
) -> List[Dict[str, Any]]:
    profile = _rf100_discover(download=download)
    if not profile.get("ready"):
        hf_repo = str((source_config or {}).get("hf_repo") or "probicheaux/rf100-vl")
        try:
            log(f"[rf100-vl] local COCO data unavailable; trying HF fallback {hf_repo}")
            return _load_rf100vl_hf_dataset(hf_repo, split=(source_config or {}).get("split"))
        except Exception as hf_exc:
            raise RuntimeError(
                "RF100-VL is not ready: "
                + " ".join(profile.get("blockers") or [])
                + f" HF fallback {hf_repo} also failed: {hf_exc}"
            ) from hf_exc
    root = Path(profile["path"])
    include_raw = (source_config or {}).get("include_groups") or []
    exclude_raw = (source_config or {}).get("exclude_groups") or []
    include = {str(x).strip() for x in include_raw if str(x).strip()}
    exclude = {str(x).strip() for x in exclude_raw if str(x).strip()}
    category_map = _rf100_project_categories()
    rows: List[Dict[str, Any]] = []
    for ann in _rf100_candidate_annotation_files(root):
        group = _rf100_group_name(root, ann)
        if include and group not in include:
            continue
        if group in exclude:
            continue
        try:
            data = json.loads(ann.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"RF100-VL COCO JSON unreadable at {ann}: {exc}") from exc
        images = {int(img.get("id")): img for img in (data.get("images") or []) if img.get("id") is not None}
        ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for a in data.get("annotations") or []:
            try:
                img_id = int(a.get("image_id"))
            except Exception:
                continue
            ann_by_image.setdefault(img_id, []).append(a)
        categories = [
            {"id": int(c.get("id")), "name": str(c.get("name") or c.get("id"))}
            for c in (data.get("categories") or [])
            if c.get("id") is not None
        ]
        cat_lookup = {int(c["id"]): c["name"] for c in categories}
        domain = category_map.get(group) or category_map.get(_safe_dataset_dir_name(group).lower()) or "uncategorized"
        split = _rf100_split_name(ann)
        prompt = _rf100_prompt(categories, group, domain)
        for img_id, img in images.items():
            gt = []
            for a in ann_by_image.get(img_id, []):
                if a.get("bbox") is None or a.get("category_id") is None:
                    continue
                try:
                    cat_id = int(a.get("category_id"))
                    bbox = [float(x) for x in a.get("bbox")]
                except Exception:
                    continue
                gt.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "category_name": cat_lookup.get(cat_id, str(cat_id)),
                    "bbox": bbox,
                    "area": a.get("area"),
                    "iscrowd": a.get("iscrowd", 0),
                })
            image_path = _rf100_image_path(ann, str(img.get("file_name") or ""))
            refs = [
                f"{len(gt)} COCO boxes across {len({g['category_id'] for g in gt})} categories",
                json.dumps(gt[:20], default=str),
            ]
            rows.append({
                "adapter": "rf100-vl",
                "task": "object_detection",
                "question_id": f"{group}:{img_id}",
                "segment_id": str(img_id),
                "question": prompt,
                "references": refs,
                "images": [image_path],
                "category": domain,
                "dataset_name": group,
                "split": split,
                "image_id": img_id,
                "image_width": img.get("width"),
                "image_height": img.get("height"),
                "ground_truth": gt,
                "categories": categories,
                "category_lookup": cat_lookup,
                "annotation_file": str(ann),
            })
    return rows


def _save_hf_image(img: Any, cache_dir: Path, row_idx: int) -> str:
    out = cache_dir / f"{row_idx}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(img, "save") and hasattr(img, "convert"):
        if not out.exists():
            img.convert("RGB").save(out, "JPEG", quality=88)
        return str(out)
    if isinstance(img, str):
        return img
    raise RuntimeError("HF RF100-VL row did not contain a supported image value")


def _load_rf100vl_hf_dataset(repo: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"`datasets` library not installed. pip install datasets (got: {exc})") from exc
    use_split = split or "test"
    try:
        ds = load_dataset(repo, split=use_split, trust_remote_code=False)
    except Exception as exc:
        last_exc = exc
        for fallback in ("validation", "valid", "train"):
            if fallback == use_split:
                continue
            try:
                ds = load_dataset(repo, split=fallback, trust_remote_code=False)
                use_split = fallback
                last_exc = None
                break
            except Exception as exc2:
                last_exc = exc2
        if last_exc is not None:
            raise RuntimeError(f"HF RF100-VL load_dataset failed for {repo}: {last_exc}") from last_exc
    safe = _safe_dataset_dir_name(repo.replace("/", "__"))
    cache_dir = _HF_DATASETS_CACHE_ROOT / safe / use_split
    rows: List[Dict[str, Any]] = []
    for idx, row_raw in enumerate(ds):
        row = dict(row_raw)
        annotations = row.get("annotations") or row.get("objects") or {}
        if not isinstance(annotations, dict):
            continue
        bboxes = annotations.get("bbox") or annotations.get("bboxes") or []
        cat_ids = annotations.get("category_id") or annotations.get("category_ids") or []
        cat_names = annotations.get("category_name") or annotations.get("category_names") or []
        image_id = row.get("image_id") or row.get("id") or idx
        dataset_name = str(row.get("dataset_name") or row.get("dataset_id") or "rf100-vl-hf")
        domain = str(row.get("domain") or row.get("category") or "uncategorized")
        image_path = _save_hf_image(row.get("image"), cache_dir, idx)
        categories_by_id: Dict[int, str] = {}
        gt = []
        for j, bbox in enumerate(bboxes):
            try:
                cat_id = int(cat_ids[j])
                bb = [float(x) for x in bbox]
            except Exception:
                continue
            name = str(cat_names[j]) if j < len(cat_names) and cat_names[j] is not None else str(cat_id)
            categories_by_id[cat_id] = name
            gt.append({
                "image_id": image_id,
                "category_id": cat_id,
                "category_name": name,
                "bbox": bb,
            })
        categories = [{"id": cid, "name": name} for cid, name in sorted(categories_by_id.items())]
        rows.append({
            "adapter": "rf100-vl",
            "task": "object_detection",
            "question_id": f"{dataset_name}:{image_id}",
            "segment_id": str(image_id),
            "question": _rf100_prompt(categories, dataset_name, domain),
            "references": [
                f"{len(gt)} COCO boxes across {len(categories)} categories",
                json.dumps(gt[:20], default=str),
            ],
            "images": [image_path],
            "category": domain,
            "dataset_name": dataset_name,
            "split": use_split,
            "image_id": image_id,
            "image_width": row.get("width"),
            "image_height": row.get("height"),
            "ground_truth": gt,
            "categories": categories,
            "category_lookup": categories_by_id,
            "annotation_file": f"hf:{repo}@{use_split}",
        })
    if not rows:
        raise RuntimeError(f"HF RF100-VL dataset {repo}@{use_split} produced no rows with annotations")
    return rows


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
    if refs_raw is None:
        indexed_refs = [
            row.get(k)
            for k in ("answer_1", "answer_2", "answer1", "answer2", "gt_a", "gt_b", "reference_1", "reference_2")
            if row.get(k) is not None
        ]
        if indexed_refs:
            refs_raw = indexed_refs
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
    if not images_raw:
        indexed_images = [
            row.get(f"image_{i}")
            for i in range(1, 6)
            if row.get(f"image_{i}") is not None
        ]
        if indexed_images:
            images_raw = indexed_images
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
    if "rf100-vl" in repo.lower():
        return _load_rf100vl_hf_dataset(repo, split=split)

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

    if ds in {"rf100-vl", "rf100vl", "roboflow100-vl"} or ds.startswith("rf100-vl:"):
        download = bool((source_config or {}).get("download", True))
        return _load_rf100vl_dataset(source_config, download=download)

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
    # Per official wayveai/LingoQA `benchmark/judge.py`, the protocol's
    # tokenizer call hardcodes `max_length=128` (NOT DeBERTa's 512 architectural
    # cap). We mirror that exactly so our scoring matches the upstream judge.
    encoded = _JUDGE_TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=LINGO_JUDGE_MAX_TOKENS,
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


# =============================================================================
# Fairness audit — 5-method judge pipeline (Phase K addition)
#
# Compares 5 judging methods side-by-side on the SAME predictions:
#   A. lingo_judge_max128            — OFFICIAL protocol (control). max_length=128.
#   B. lingo_judge_max512            — DeBERTa-v3-base's architectural ceiling.
#   C. lingo_judge_answer_only_max128— Protocol-correct: strip <think>, then max=128.
#   D. lingo_judge_head_tail_512     — Hack: first 256 tok + last 256 tok of full text.
#   E. llm_as_judge                  — Decoder-only LLM judge (configurable endpoint).
#
# All four DeBERTa variants share the SAME loaded model+tokenizer (see
# _ensure_judge) — only preprocessing differs. E hits a remote endpoint.
# =============================================================================

JUDGE_METHODS = (
    "lingo_judge_max128",
    "lingo_judge_max512",
    "lingo_judge_answer_only_max128",
    "lingo_judge_head_tail_512",
    "llm_as_judge",
)


def _lingo_judge_score_at(question: str, reference: str, prediction: str, max_length: int) -> float:
    """Lingo-Judge scorer with explicit max_length. Mirrors lingo_judge_score
    exactly but lets the caller pick the truncation cap.
    """
    _ensure_judge()
    import torch
    text = f"[CLS]\nQuestion: {question}\nAnswer: {reference}\nStudent: {prediction}"
    encoded = _JUDGE_TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(_JUDGE_DEVICE)
    with torch.no_grad():
        out = _JUDGE_MODEL(**encoded)
    logit = out.logits.squeeze().detach().cpu().float().item()
    return float(1.0 / (1.0 + pow(2.718281828, -logit)))


def _judge_head_tail(question: str, reference: str, prediction: str, max_total: int = 512) -> float:
    """Method D: head-tail token-level truncation.

    Tokenizes the FULL [CLS]/Question/Answer/Student string, takes the first
    half-max + last half-max token IDs (with a `[...]` separator decoded back
    into text), then re-tokenizes at max=max_total. Preserves both opening
    context (Q + ref) AND final answer (end of long <think> chains).
    """
    _ensure_judge()
    import torch
    text = f"[CLS]\nQuestion: {question}\nAnswer: {reference}\nStudent: {prediction}"
    # Tokenize WITHOUT truncation to get the full id list.
    full_ids = _JUDGE_TOKENIZER(text, return_tensors=None, truncation=False, padding=False)["input_ids"]
    # Separator token (decoded text representation, re-tokenized later).
    sep = " [...] "
    half = max_total // 2
    # If the full sequence is already ≤ max_total, just score it normally.
    if len(full_ids) <= max_total:
        encoded = _JUDGE_TOKENIZER(
            text, return_tensors="pt", truncation=True, max_length=max_total, padding=False
        ).to(_JUDGE_DEVICE)
    else:
        head_ids = full_ids[:half]
        tail_ids = full_ids[-half:]
        head_text = _JUDGE_TOKENIZER.decode(head_ids, skip_special_tokens=True)
        tail_text = _JUDGE_TOKENIZER.decode(tail_ids, skip_special_tokens=True)
        concat = head_text + sep + tail_text
        encoded = _JUDGE_TOKENIZER(
            concat, return_tensors="pt", truncation=True, max_length=max_total, padding=False
        ).to(_JUDGE_DEVICE)
    with torch.no_grad():
        out = _JUDGE_MODEL(**encoded)
    logit = out.logits.squeeze().detach().cpu().float().item()
    return float(1.0 / (1.0 + pow(2.718281828, -logit)))


def _judge_score_max_with(
    fn,
    question: str,
    references: List[str],
    prediction: str,
) -> Tuple[float, int]:
    """Generic max-over-references for ANY (q, ref, pred) -> float scorer."""
    scores = [fn(question, ref, prediction) for ref in references]
    if not scores:
        return 0.0, -1
    best = max(range(len(scores)), key=scores.__getitem__)
    return scores[best], best


# ---- LLM-as-judge (E) ------------------------------------------------------

BENCH_LLM_JUDGE_CONFIG_PATH = Path("/tmp/llm_judge_config.json")


def _llm_judge_config() -> Dict[str, Any]:
    """Resolve the LLM-judge endpoint config. Precedence:
      1. BENCH_LLM_JUDGE_URL / BENCH_LLM_JUDGE_MODEL env vars
      2. /tmp/llm_judge_config.json
      3. Sensible default placeholder (ready=false until Bronson provisions).
    """
    url = os.getenv("BENCH_LLM_JUDGE_URL")
    model = os.getenv("BENCH_LLM_JUDGE_MODEL")
    cfg: Dict[str, Any] = {}
    if BENCH_LLM_JUDGE_CONFIG_PATH.exists():
        try:
            cfg = json.loads(BENCH_LLM_JUDGE_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            log(f"[llm-judge] config parse failed: {exc}")
            cfg = {}
    url = url or cfg.get("url") or "http://0.0.0.0:8000/v1"
    model = model or cfg.get("model") or "meta-llama/Llama-3.1-8B-Instruct"
    return {"url": url, "model": model}


def _llm_judge_endpoint_ready(url: str, timeout: float = 3.0) -> bool:
    """Quick reachability probe — GET /v1/models on the endpoint."""
    try:
        probe = url.rstrip("/") + "/models"
        req = urllib.request.Request(probe, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


_LLM_VERDICT_RE = re.compile(
    r"VERDICT:\s*(YES|NO)\s*[\s,]*CONFIDENCE:\s*([0-9.]+)\s*[\s,]*REASON:\s*(.*)",
    re.IGNORECASE,
)


def _llm_as_judge(
    question: str,
    reference: str,
    prediction: str,
    endpoint_url: str,
    model_name: str,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Score one (q, ref, pred) triple via a decoder-only LLM judge.

    Returns {"verdict": "YES"|"NO"|None, "confidence": float|None,
             "reason": str, "score": 1.0|0.0|None, "error": str|None}.

    Uses the build.nvidia.com canonical OpenAI-compatible chat shape.
    NO max_tokens, NO temperature, NO top_p — Alex standing order.
    """
    prompt = (
        "You are evaluating whether a STUDENT's answer to a visual question is correct.\n\n"
        f"QUESTION: {question}\n"
        f"REFERENCE ANSWER (one of two acceptable references): {reference}\n"
        f"STUDENT'S ANSWER: {prediction}\n\n"
        "Is the student's answer semantically equivalent to the reference answer for the "
        "purposes of this question? The student may use different words or include reasoning, "
        "but the final factual claim must agree with the reference. Visual details that "
        "contradict the reference are wrong even if the rest of the answer is plausible.\n\n"
        "Reply with exactly one line in this format:\n"
        "VERDICT: YES|NO  CONFIDENCE: 0.XX  REASON: <one-sentence>"
    )
    body = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
    }
    url = endpoint_url.rstrip("/") + "/chat/completions"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        return {
            "verdict": None, "confidence": None, "reason": "",
            "score": None, "error": f"llm_judge_endpoint_unavailable: {exc}",
        }
    try:
        parsed = json.loads(raw)
        msg = (parsed.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):
            # Some servers return list-of-parts; collect text parts.
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    except Exception as exc:
        return {
            "verdict": None, "confidence": None, "reason": "",
            "score": None, "error": f"llm_judge_parse_failed: {exc}",
        }
    m = _LLM_VERDICT_RE.search(content)
    if not m:
        return {
            "verdict": None, "confidence": None, "reason": content.strip()[:200],
            "score": None,
            "error": f"llm_judge_no_verdict_line",
        }
    verdict = m.group(1).upper()
    try:
        confidence = float(m.group(2))
    except Exception:
        confidence = None
    reason = m.group(3).strip()
    score = 1.0 if verdict == "YES" else 0.0
    return {
        "verdict": verdict, "confidence": confidence, "reason": reason,
        "score": score, "error": None,
    }


def _llm_as_judge_max(
    question: str,
    references: List[str],
    prediction: str,
    endpoint_url: str,
    model_name: str,
) -> Tuple[Optional[float], int, Dict[str, Any]]:
    """Run LLM-as-judge against every reference; take the max score.

    Returns (max_score, winning_ref_idx, last_raw_record). max_score is None
    if every call errored (so callers can degrade gracefully).
    """
    raw_records: List[Dict[str, Any]] = []
    scores: List[Optional[float]] = []
    for ref in references:
        rec = _llm_as_judge(question, ref, prediction, endpoint_url, model_name)
        raw_records.append(rec)
        scores.append(rec.get("score"))
    valid = [(i, s) for i, s in enumerate(scores) if s is not None]
    if not valid:
        return None, -1, {"per_reference": raw_records}
    best_idx, best_score = max(valid, key=lambda kv: kv[1])
    return best_score, best_idx, {"per_reference": raw_records}


# ---- Rejudge dispatch ------------------------------------------------------

def _rejudge_one_row(
    row: Dict[str, Any],
    method: str,
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Re-score ONE existing prediction row through ONE judge method.

    Returns a dict {score, correct, winning_reference_index, error, method, ...}.
    Skips rows that have no prediction or are themselves errored — preserves
    the original row's error contract.
    """
    out: Dict[str, Any] = {
        "method": method,
        "question_id": row.get("question_id"),
        "score": None,
        "correct": None,
        "winning_reference_index": -1,
        "error": None,
    }
    if row.get("error"):
        out["error"] = f"upstream_error: {row.get('error')}"
        return out
    question = row.get("question") or ""
    references = row.get("references") or []
    prediction = row.get("prediction") or ""
    if not references:
        out["error"] = "no_references"
        return out
    try:
        if method == "lingo_judge_max128":
            score, idx = _judge_score_max_with(
                lambda q, r, p: _lingo_judge_score_at(q, r, p, 128),
                question, references, prediction,
            )
        elif method == "lingo_judge_max512":
            score, idx = _judge_score_max_with(
                lambda q, r, p: _lingo_judge_score_at(q, r, p, 512),
                question, references, prediction,
            )
        elif method == "lingo_judge_answer_only_max128":
            pred_ao = _extract_final_answer(prediction)
            score, idx = _judge_score_max_with(
                lambda q, r, p: _lingo_judge_score_at(q, r, p, 128),
                question, references, pred_ao,
            )
            out["prediction_used"] = pred_ao
        elif method == "lingo_judge_head_tail_512":
            score, idx = _judge_score_max_with(
                lambda q, r, p: _judge_head_tail(q, r, p, 512),
                question, references, prediction,
            )
        elif method == "llm_as_judge":
            cfg = llm_cfg or _llm_judge_config()
            score, idx, raw = _llm_as_judge_max(
                question, references, prediction, cfg["url"], cfg["model"],
            )
            out["llm_raw"] = raw
            if score is None:
                out["error"] = "llm_judge_endpoint_unavailable"
        else:
            out["error"] = f"unknown_method: {method}"
            return out
        out["score"] = score
        out["winning_reference_index"] = idx
        out["correct"] = bool((score or 0) > 0.5) if score is not None else None
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _benchmark_snapshot_load_from_disk(run_id: str) -> Optional[Dict[str, Any]]:
    """Load a previous run's results.json from disk. Used by /benchmark/rejudge
    + /benchmark/audit so historical runs (not in memory) can still be re-scored.
    Falls back to None if not found.
    """
    path = BENCHMARK_RUN_ROOT / run_id / "results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[rejudge] failed to load {path}: {exc}")
        return None


def _benchmark_load_any(run_id: str) -> Optional[Dict[str, Any]]:
    """In-memory snapshot first; fall back to disk results.json."""
    snap = benchmark_get(run_id)
    if snap and (snap.get("results") or snap.get("recent_results")):
        return snap
    return _benchmark_snapshot_load_from_disk(run_id)


def _rejudge_summarize(per_row: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize a list of per-row rejudge records into {accuracy, n_judged, n_errored}."""
    n_total = len(per_row)
    judged = [r for r in per_row if r.get("score") is not None]
    correct = sum(1 for r in judged if r.get("correct"))
    n_errored = sum(1 for r in per_row if r.get("error"))
    return {
        "accuracy": (correct / len(judged)) if judged else 0.0,
        "n_judged": len(judged),
        "n_errored": n_errored,
        "n_total": n_total,
    }


def run_rejudge(run_id: str, method: str) -> Dict[str, Any]:
    """Re-judge an EXISTING run's predictions through `method`.

    Writes /tmp/benchmark-runs/<run_id>/rejudge_<method>.json. Idempotent:
    if the file already exists, returns the cached content.
    """
    if method not in JUDGE_METHODS:
        raise RuntimeError(f"unknown judge_method: {method}")
    run_dir = BENCHMARK_RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"rejudge_{method}.json"
    if out_path.exists():
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # corrupt cache; recompute
    snap = _benchmark_load_any(run_id)
    if not snap:
        raise RuntimeError(f"run_id not found on disk or in memory: {run_id}")
    rows = snap.get("results") or []
    if not rows:
        raise RuntimeError(f"run_id {run_id} has no results to rejudge")
    # Resolve LLM config ONCE (avoid re-reading the file per row).
    llm_cfg = _llm_judge_config() if method == "llm_as_judge" else None
    log(f"[rejudge {run_id}] method={method} n_rows={len(rows)}")
    per_row: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        rec = _rejudge_one_row(row, method, llm_cfg=llm_cfg)
        per_row.append(rec)
        if (i + 1) % 50 == 0:
            log(f"[rejudge {run_id}] {method} {i+1}/{len(rows)}")
    summary = _rejudge_summarize(per_row)
    out = {
        "run_id": run_id,
        "method": method,
        "summary": summary,
        "per_row": per_row,
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, default=str, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    log(f"[rejudge {run_id}] {method} done. accuracy={summary['accuracy']:.3f}")
    return out


def run_audit(run_id: str) -> Dict[str, Any]:
    """Run ALL 5 judge methods on a run and write a single aggregated audit.json.

    Writes /tmp/benchmark-runs/<run_id>/audit.json. Per-method results are
    cached in rejudge_<method>.json (so a second audit call is fast).
    """
    snap = _benchmark_load_any(run_id)
    if not snap:
        raise RuntimeError(f"run_id not found on disk or in memory: {run_id}")
    rows = snap.get("results") or []
    n_predictions = len(rows)
    methods_data: Dict[str, Any] = {}
    per_method_rows: Dict[str, List[Dict[str, Any]]] = {}
    for method in JUDGE_METHODS:
        rj = run_rejudge(run_id, method)
        methods_data[method] = rj["summary"]
        per_method_rows[method] = rj["per_row"]
    # Build per-row matrix.
    qid_index: Dict[str, int] = {}
    for i, row in enumerate(rows):
        qid_index[str(row.get("question_id") or i)] = i
    per_row_out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        qid = str(row.get("question_id") or i)
        verdicts: Dict[str, Any] = {}
        scores: Dict[str, Any] = {}
        for method in JUDGE_METHODS:
            rec = per_method_rows[method][i] if i < len(per_method_rows[method]) else {}
            verdicts[method] = rec.get("correct")
            scores[method] = rec.get("score")
        # Divergent: verdicts != majority. Skip None verdicts in the vote.
        votes = [v for v in verdicts.values() if isinstance(v, bool)]
        if votes:
            yes_n = sum(1 for v in votes if v)
            no_n = len(votes) - yes_n
            majority = True if yes_n >= no_n else False
            divergent = [m for m, v in verdicts.items() if isinstance(v, bool) and v != majority]
        else:
            divergent = []
        per_row_out.append({
            "question_id": qid,
            "question": (row.get("question") or "")[:200],
            "verdicts": verdicts,
            "scores": scores,
            "divergent_methods": divergent,
        })
    # Divergence stats: rows with any disagreement, plus the method pair with
    # most disagreements across rows.
    rows_with_disagreement = sum(1 for r in per_row_out if r["divergent_methods"])
    pair_disagreements: Dict[str, int] = {}
    methods_list = list(JUDGE_METHODS)
    for r in per_row_out:
        for i, mi in enumerate(methods_list):
            vi = r["verdicts"].get(mi)
            if not isinstance(vi, bool):
                continue
            for mj in methods_list[i+1:]:
                vj = r["verdicts"].get(mj)
                if not isinstance(vj, bool):
                    continue
                if vi != vj:
                    key = f"{mi} vs {mj}"
                    pair_disagreements[key] = pair_disagreements.get(key, 0) + 1
    max_pair = ""
    max_pair_n = 0
    if pair_disagreements:
        max_pair, max_pair_n = max(pair_disagreements.items(), key=lambda kv: kv[1])
    audit = {
        "run_id": run_id,
        "n_predictions": n_predictions,
        "methods": methods_data,
        "per_row": per_row_out,
        "divergence_stats": {
            "rows_with_disagreement": rows_with_disagreement,
            "pair_disagreements": pair_disagreements,
            "max_pairwise_disagreement": max_pair,
            "max_pairwise_disagreement_count": max_pair_n,
        },
        "generated_epoch": time.time(),
    }
    out_path = BENCHMARK_RUN_ROOT / run_id / "audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(audit, default=str, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    log(f"[audit {run_id}] complete. {n_predictions} preds, {rows_with_disagreement} divergent rows")
    return audit


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
    style text so the judge's 128-token window (hardcoded in the official
    benchmark/judge.py) holds the entire final answer.
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


# Lingo-Judge truncation cap.
#
# IMPORTANT: The official wayveai/LingoQA `benchmark/judge.py`
# (https://github.com/wayveai/LingoQA/blob/main/benchmark/judge.py) hardcodes:
#     self.tokenizer(texts, return_tensors='pt', padding=True,
#                    truncation=True, max_length=128)
# i.e. the protocol clips judge inputs at 128 tokens — NOT DeBERTa-v3-base's
# 512 architectural cap. 128 is MORE aggressive than 512: a moderate reasoning
# trace (~90 tokens) is already over budget.
#
# Heuristic for the per-row truncation telemetry: ~4 chars per token for
# English. Exact tokenizer counts would be nicer but adding tiktoken as a dep
# is out of scope — the goal here is directional, not precise.
LINGO_JUDGE_MAX_TOKENS = 128  # Hardcoded in official wayveai/LingoQA benchmark/judge.py — DeBERTa's 512 architectural cap is NOT what the protocol uses
JUDGE_MAX_TOKENS = LINGO_JUDGE_MAX_TOKENS  # Back-compat alias for downstream readers


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
    saw — including how a 1024-token reasoning trace gets clipped to 128
    (the protocol's hardcoded ceiling in benchmark/judge.py).
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


def _extract_json_payload(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return {}
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start_candidates = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
    if not start_candidates:
        return {}
    start = min(start_candidates)
    for end_char in ("}", "]"):
        end = raw.rfind(end_char)
        if end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                continue
    return {}


def _coerce_bbox_xywh(value: Any, width: Optional[float] = None, height: Optional[float] = None) -> Optional[List[float]]:
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "width", "height")):
            try:
                return [float(value["x"]), float(value["y"]), max(0.0, float(value["width"])), max(0.0, float(value["height"]))]
            except Exception:
                return None
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            try:
                x1, y1, x2, y2 = float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"])
                return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            except Exception:
                return None
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0, y0, a, b = [float(value[i]) for i in range(4)]
    except Exception:
        return None
    # The RF100 prompt asks for COCO xywh. Lists are treated as xywh unless
    # image bounds make that impossible and x2/y2 interpretation fits.
    if width and height and a > x0 and b > y0:
        try:
            w_img = float(width)
            h_img = float(height)
            xywh_runs_out = (x0 + a > w_img * 1.05) or (y0 + b > h_img * 1.05)
            xyxy_fits = a <= w_img * 1.05 and b <= h_img * 1.05
            if xywh_runs_out and xyxy_fits:
                return [x0, y0, max(0.0, a - x0), max(0.0, b - y0)]
        except Exception:
            pass
    return [x0, y0, max(0.0, a), max(0.0, b)]


def _bbox_iou_xywh(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _rf100_parse_predictions(text: str, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = _extract_json_payload(text)
    if isinstance(payload, dict):
        raw_items = payload.get("detections") or payload.get("instances") or payload.get("predictions") or []
    elif isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict) and (
            payload[0].get("detections") or payload[0].get("instances") or payload[0].get("predictions")
        ):
            raw_items = payload[0].get("detections") or payload[0].get("instances") or payload[0].get("predictions") or []
        else:
            raw_items = payload
    else:
        raw_items = []
    cat_lookup = sample.get("category_lookup") or {}
    name_to_id = {str(v).lower(): int(k) for k, v in cat_lookup.items()}
    image_width = sample.get("image_width")
    image_height = sample.get("image_height")
    out: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        bbox = _coerce_bbox_xywh(
            item.get("bbox") or item.get("box") or item.get("bounding_box"),
            image_width,
            image_height,
        )
        if bbox is None:
            continue
        cat_id_raw = item.get("category_id")
        if cat_id_raw is None:
            cat_id_raw = item.get("class_id")
        if cat_id_raw is None:
            cat_id_raw = item.get("label_id")
        cat_name = item.get("category_name") or item.get("class_name") or item.get("label") or item.get("name")
        try:
            cat_id = int(cat_id_raw)
        except Exception:
            cat_id = name_to_id.get(str(cat_name or "").lower(), -1)
        if cat_id < 0:
            continue
        try:
            score = float(item.get("score") if item.get("score") is not None else item.get("confidence"))
        except Exception:
            score = 1.0
        out.append({
            "image_id": sample.get("image_id"),
            "category_id": cat_id,
            "category_name": str(cat_name or cat_lookup.get(cat_id) or cat_id),
            "bbox": bbox,
            "score": max(0.0, min(1.0, score)),
            "dataset_name": sample.get("dataset_name"),
        })
    return out


def _rf100_match_counts(gt: List[Dict[str, Any]], preds: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
    matched_gt: set = set()
    tp = 0
    fp = 0
    for pred in sorted(preds, key=lambda p: float(p.get("score") or 0), reverse=True):
        best_i = -1
        best_iou = 0.0
        for i, g in enumerate(gt):
            if i in matched_gt:
                continue
            if int(g.get("category_id")) != int(pred.get("category_id")):
                continue
            iou = _bbox_iou_xywh(g.get("bbox") or [0, 0, 0, 0], pred.get("bbox") or [0, 0, 0, 0])
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0 and best_iou >= threshold:
            matched_gt.add(best_i)
            tp += 1
        else:
            fp += 1
    fn = max(0, len(gt) - len(matched_gt))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _rf100_context_error(text: str) -> Optional[Dict[str, int]]:
    match = re.search(
        r"Input length\s*\(?(\d+)\)?\s*exceeds model'?s maximum context length\s*\(?(\d+)\)?",
        text or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {"input_length": int(match.group(1)), "max_context_length": int(match.group(2))}


def _rf100_pixel_attempts() -> List[int]:
    values = [
        RF100_VL_MAX_PIXELS,
        max(RF100_VL_MIN_PIXELS, RF100_VL_MAX_PIXELS // 2),
        max(RF100_VL_MIN_PIXELS, RF100_VL_MAX_PIXELS // 4),
        RF100_VL_MIN_PIXELS,
    ]
    out: List[int] = []
    for value in values[: max(1, RF100_VL_CONTEXT_RETRY_ATTEMPTS)]:
        if value > 0 and value not in out:
            out.append(value)
    return out or [RF100_VL_MAX_PIXELS]


def _rf100_encode_image(path: str, max_pixels: int, jpeg_quality: int) -> Tuple[str, Dict[str, Any]]:
    image_meta: Dict[str, Any] = {
        "source_path": path,
        "max_pixels": max_pixels,
        "jpeg_quality": jpeg_quality,
    }
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            rgb = img.convert("RGB")
            image_meta["original_size"] = [rgb.width, rgb.height]
            resized = resize_to_max_pixels(rgb, max_pixels)
            image_meta["sent_size"] = [resized.width, resized.height]
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            raw = buf.getvalue()
            image_meta["sent_bytes"] = len(raw)
            image_meta["scale_to_original"] = [
                (rgb.width / resized.width) if resized.width else 1.0,
                (rgb.height / resized.height) if resized.height else 1.0,
            ]
            return base64.b64encode(raw).decode("ascii"), image_meta
    except Exception as exc:
        image_meta["resize_error"] = str(exc)[:240]
        with open(path, "rb") as fh:
            raw = fh.read()
        image_meta["sent_bytes"] = len(raw)
        return base64.b64encode(raw).decode("ascii"), image_meta


def _rf100_scale_detections_to_original(
    detections: List[Dict[str, Any]],
    image_meta: Dict[str, Any],
    sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sent = image_meta.get("sent_size") or []
    original = image_meta.get("original_size") or []
    target_w = float(sample.get("image_width") or (original[0] if len(original) >= 1 else 0) or 0)
    target_h = float(sample.get("image_height") or (original[1] if len(original) >= 2 else 0) or 0)
    sent_w = float(sent[0]) if len(sent) >= 1 and sent[0] else target_w
    sent_h = float(sent[1]) if len(sent) >= 2 and sent[1] else target_h
    if not target_w or not target_h or not sent_w or not sent_h:
        return detections
    sx = target_w / sent_w
    sy = target_h / sent_h
    if abs(sx - 1.0) < 0.01 and abs(sy - 1.0) < 0.01:
        return detections
    scaled: List[Dict[str, Any]] = []
    for det in detections:
        item = dict(det)
        bbox = item.get("bbox") or []
        if isinstance(bbox, list) and len(bbox) >= 4:
            item["bbox_sent"] = list(bbox)
            x, y, w, h = [float(bbox[i]) for i in range(4)]
            item["bbox"] = [
                max(0.0, min(target_w, x * sx)),
                max(0.0, min(target_h, y * sy)),
                max(0.0, min(target_w, w * sx)),
                max(0.0, min(target_h, h * sy)),
            ]
            item["bbox_scaled_to_original"] = True
        scaled.append(item)
    return scaled


def _rf100_ap(gt: List[Dict[str, Any]], preds: List[Dict[str, Any]], threshold: float) -> float:
    cats = sorted({int(g.get("category_id")) for g in gt if g.get("category_id") is not None})
    if not cats:
        return 0.0
    ap_values: List[float] = []
    for cat in cats:
        gt_cat = [(idx, g) for idx, g in enumerate(gt) if int(g.get("category_id")) == cat]
        pred_cat = sorted(
            [p for p in preds if int(p.get("category_id", -1)) == cat],
            key=lambda p: float(p.get("score") or 0),
            reverse=True,
        )
        if not gt_cat:
            continue
        matched: set = set()
        tp_flags: List[int] = []
        fp_flags: List[int] = []
        for pred in pred_cat:
            best_idx = None
            best_iou = 0.0
            for gt_idx, g in gt_cat:
                if gt_idx in matched:
                    continue
                if str(g.get("image_key")) != str(pred.get("image_key")):
                    continue
                iou = _bbox_iou_xywh(g.get("bbox") or [0, 0, 0, 0], pred.get("bbox") or [0, 0, 0, 0])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx
            if best_idx is not None and best_iou >= threshold:
                matched.add(best_idx)
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
        if not pred_cat:
            ap_values.append(0.0)
            continue
        cum_tp = []
        cum_fp = []
        t = f = 0
        for tp_i, fp_i in zip(tp_flags, fp_flags):
            t += tp_i
            f += fp_i
            cum_tp.append(t)
            cum_fp.append(f)
        recalls = [v / len(gt_cat) for v in cum_tp]
        precisions = [cum_tp[i] / max(1, cum_tp[i] + cum_fp[i]) for i in range(len(cum_tp))]
        ap = 0.0
        for r in [i / 100 for i in range(101)]:
            p_at_r = max([p for p, rec in zip(precisions, recalls) if rec >= r] or [0.0])
            ap += p_at_r / 101.0
        ap_values.append(ap)
    value = sum(ap_values) / len(ap_values) if ap_values else 0.0
    return max(0.0, min(1.0, value))


def _rf100_summarize(snap: Dict[str, Any]) -> Dict[str, Any]:
    rows = [r for r in (snap.get("results") or []) if not r.get("error")]
    gt_all: List[Dict[str, Any]] = []
    pred_all: List[Dict[str, Any]] = []
    for r in rows:
        image_key = str(r.get("question_id") or r.get("image_id"))
        for g in r.get("ground_truth") or []:
            item = dict(g)
            item["image_key"] = image_key
            gt_all.append(item)
        for p in r.get("detections") or []:
            item = dict(p)
            item["image_key"] = image_key
            pred_all.append(item)
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    ap_by_threshold = {f"AP{int(t * 100)}": _rf100_ap(gt_all, pred_all, t) for t in thresholds}
    map_value = sum(ap_by_threshold.values()) / len(ap_by_threshold) if ap_by_threshold else 0.0
    by_group: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        group = r.get("dataset_name") or "unknown"
        b = by_group.setdefault(group, {"rows": [], "gt": [], "preds": []})
        b["rows"].append(r)
        image_key = str(r.get("question_id") or r.get("image_id"))
        for g in r.get("ground_truth") or []:
            item = dict(g)
            item["image_key"] = image_key
            b["gt"].append(item)
        for p in r.get("detections") or []:
            item = dict(p)
            item["image_key"] = image_key
            b["preds"].append(item)
    per_category = []
    for group, b in sorted(by_group.items()):
        ap50 = _rf100_ap(b["gt"], b["preds"], 0.5)
        ap_all = sum(_rf100_ap(b["gt"], b["preds"], t) for t in thresholds) / len(thresholds) if thresholds else 0.0
        per_category.append({
            "category": group,
            "total": len(b["rows"]),
            "correct": sum(1 for r in b["rows"] if r.get("judge_correct")),
            "accuracy": ap_all,
            "average_score": ap50,
            "ap": ap_all,
            "ap50": ap50,
        })
    latencies = [r.get("latency_seconds") for r in rows if r.get("latency_seconds") is not None]
    return {
        "task": "object_detection",
        "metric": "COCO-style AP@[.50:.95]",
        "total_predictions": len(rows),
        "total_ground_truth_boxes": len(gt_all),
        "total_predicted_boxes": len(pred_all),
        "map": map_value,
        "ap": map_value,
        "ap50": ap_by_threshold.get("AP50", 0.0),
        "ap75": ap_by_threshold.get("AP75", 0.0),
        "ap_by_threshold": ap_by_threshold,
        "overall_accuracy": map_value,
        "correct": sum(1 for r in rows if r.get("judge_correct")),
        "per_category": per_category,
        "average_latency_seconds": (sum(latencies) / len(latencies)) if latencies else None,
    }


def run_one_rf100_detection(
    sample: Dict[str, Any],
    model: str,
    base_url: str,
    headers: Dict[str, str],
    *,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError(f"requests import failed: {REQUESTS_IMPORT_ERROR}")
    images = sample.get("images") or []
    if not images:
        raise RuntimeError("RF100-VL sample has no image path")
    attempts: List[Dict[str, Any]] = []
    response_body: Optional[Dict[str, Any]] = None
    raw_text = ""
    elapsed = 0.0
    image_meta: Dict[str, Any] = {}
    prompt_text = str(sample["question"])
    prompt_variants = [prompt_text]
    categories = sample.get("categories") or []
    if categories:
        compact_prompt = _rf100_prompt(
            categories,
            str(sample.get("dataset_name") or ""),
            str(sample.get("category") or ""),
            max_categories=max(20, min(RF100_VL_PROMPT_MAX_CATEGORIES, len(categories))),
            max_chars=max(1200, RF100_VL_PROMPT_MAX_CHARS // 2),
        )
        if compact_prompt != prompt_text:
            prompt_variants.append(compact_prompt)
    last_exc: Optional[BaseException] = None
    for prompt_index, attempt_prompt in enumerate(prompt_variants):
        for max_pixels in _rf100_pixel_attempts():
            b64, image_meta = _rf100_encode_image(images[0], max_pixels, RF100_VL_JPEG_QUALITY)
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": attempt_prompt},
            ]
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return object detections as strict JSON only."},
                    {"role": "user", "content": content},
                ],
            }
            transient_tries = max(1, RF100_VL_TRANSIENT_RETRIES + 1)
            for transient_try in range(transient_tries):
                started = time.monotonic()
                try:
                    resp = requests.post(
                        base_url.rstrip("/") + "/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    )
                    elapsed = time.monotonic() - started
                except Exception as exc:
                    last_exc = exc
                    attempts.append({
                        **image_meta,
                        "prompt_index": prompt_index,
                        "transient_try": transient_try + 1,
                        "error": str(exc)[:600],
                    })
                    if transient_try + 1 < transient_tries:
                        time.sleep(min(8.0, 1.5 * (transient_try + 1)))
                        continue
                    break
                if resp.status_code < 400:
                    response_body = resp.json()
                    image_meta = {**image_meta, "prompt_index": prompt_index, "attempts": attempts + [{**image_meta, "status_code": resp.status_code}]}
                    break
                body_text = resp.text[:1200]
                context_error = _rf100_context_error(body_text)
                attempts.append({
                    **image_meta,
                    "prompt_index": prompt_index,
                    "status_code": resp.status_code,
                    "error": body_text[:600],
                    "context_error": context_error,
                })
                if context_error:
                    last_exc = RuntimeError(f"NIM {resp.status_code}: {body_text[:600]}")
                    break
                raise RuntimeError(f"NIM {resp.status_code}: {body_text[:600]}")
            if response_body is not None:
                break
        if response_body is not None:
            break
    if response_body is None:
        if last_exc:
            raise last_exc
        raise RuntimeError("RF100-VL request failed without a response")
    try:
        raw_text = response_body["choices"][0]["message"]["content"] or ""
    except Exception:
        raw_text = json.dumps(response_body)[:1000]
    detections = _rf100_parse_predictions(raw_text, sample)
    detections = _rf100_scale_detections_to_original(detections, image_meta, sample)
    gt = sample.get("ground_truth") or []
    image_metric = _rf100_match_counts(gt, detections, 0.5)
    pred_summary = f"{len(detections)} detections; F1@0.50={image_metric['f1']:.3f}"
    return {
        "adapter": "rf100-vl",
        "task": "object_detection",
        "question_id": sample["question_id"],
        "segment_id": sample["segment_id"],
        "image_id": sample.get("image_id"),
        "dataset_name": sample.get("dataset_name"),
        "split": sample.get("split"),
        "question": sample["question"],
        "category": sample.get("category") or "uncategorized",
        "model": model,
        "raw_response": raw_text,
        "reasoning_trace": _extract_think_block(raw_text),
        "prediction": pred_summary,
        "detections": detections,
        "ground_truth": gt,
        "references": sample.get("references") or [],
        "judge_score": image_metric["f1"],
        "judge_correct": image_metric["f1"] >= 0.5,
        "image_metrics": image_metric,
        "latency_seconds": elapsed,
        "usage": response_body.get("usage") or {},
        "images": images,
        "image_request": image_meta,
        "completed_epoch": time.time(),
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


def _benchmark_artifacts_dir(run_id: str) -> Path:
    path = _benchmark_run_dir(run_id) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _benchmark_leaderboard(snap: Dict[str, Any]) -> Dict[str, Any]:
    summary = snap.get("summary") or _benchmark_summary_for_snap(snap)
    adapter = snap.get("adapter") or _benchmark_adapter_for_dataset(snap.get("dataset") or "")
    metric_key = "map" if adapter == "rf100-vl" else "overall_accuracy"
    metric_label = "COCO-style mAP" if adapter == "rf100-vl" else "Lingo-Judge accuracy"
    return {
        "run_id": snap.get("run_id"),
        "adapter": adapter,
        "dataset": snap.get("dataset"),
        "metric": metric_label,
        "generated_epoch": time.time(),
        "models": [{
            "rank": 1,
            "model": snap.get("model") or "",
            "endpoint": snap.get("base_url") or "",
            "score": summary.get(metric_key) if summary.get(metric_key) is not None else summary.get("overall_accuracy"),
            "summary": summary,
        }],
    }


def _benchmark_write_trace_jsonl(run_id: str, snap: Dict[str, Any]) -> None:
    path = _benchmark_run_dir(run_id) / "trace.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in snap.get("results") or []:
            fh.write(json.dumps(row, default=str) + "\n")
    tmp.replace(path)


def _benchmark_write_lingoqa_predictions_csv(run_id: str, snap: Dict[str, Any]) -> Optional[Path]:
    adapter = snap.get("adapter") or _benchmark_adapter_for_dataset(snap.get("dataset") or "")
    if adapter != "lingoqa":
        return None
    path = _benchmark_run_dir(run_id) / "predictions.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["question_id", "segment_id", "answer"])
        writer.writeheader()
        for row in snap.get("results") or []:
            prediction = row.get("prediction_answer_only")
            if prediction is None:
                prediction = _extract_final_answer(row.get("prediction") or "")
            writer.writerow({
                "question_id": row.get("question_id") or "",
                "segment_id": row.get("segment_id") or "",
                "answer": prediction or "",
            })
    tmp.replace(path)
    return path


def _benchmark_write_executive_pptx(snap: Dict[str, Any], path: Path) -> None:
    summary = snap.get("summary") or _benchmark_summary_for_snap(snap)
    adapter = snap.get("adapter") or _benchmark_adapter_for_dataset(snap.get("dataset") or "")
    metric_name = "COCO-style mAP" if adapter == "rf100-vl" else "Lingo-Judge accuracy"
    score = summary.get("map") if adapter == "rf100-vl" else summary.get("overall_accuracy")
    score_txt = percent_value(score) if score is not None else "n/a"
    rows = [r for r in (snap.get("results") or []) if not r.get("error")]
    misses = [r for r in rows if not r.get("judge_correct")]
    wins = [r for r in rows if r.get("judge_correct")]
    per_cat = sorted(summary.get("per_category") or [], key=lambda c: c.get("accuracy") or 0, reverse=True)
    slides: List[List[str]] = []
    slides.append([
        ppt_text_shape(2, "Title", 0.65, 0.55, 11.8, 1.0, ["Benchmark Executive Summary"], 34, "76B900", True),
        ppt_text_shape(3, "Subtitle", 0.7, 1.45, 11.6, 0.7, [f"{snap.get('dataset')} | {snap.get('model')} | {time.strftime('%Y-%m-%d')}"], 16, "666666"),
        ppt_text_shape(4, "Metric", 0.7, 2.35, 5.6, 1.6, [score_txt, metric_name], 28, "FFFFFF", True, "1A1A1A"),
        ppt_text_shape(5, "Coverage", 6.6, 2.35, 5.6, 1.6, [str(summary.get("total_predictions") or len(rows)), "trace rows"], 28, "FFFFFF", True, "1A1A1A"),
        ppt_text_shape(6, "Claim", 0.7, 4.35, 11.5, 1.4, [
            "Use the leaderboard, per-group breakdown, and full trace repository to identify where Cosmos3 is strong and where alternate models should be tested next."
        ], 22, "1A1A1A", True, "F0F7E6"),
    ])
    slides.append([
        ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Leaderboard readout"], 30, "1A1A1A", True),
        ppt_text_shape(3, "Metrics", 0.8, 1.35, 11.7, 5.3, [
            f"Model: {snap.get('model') or 'unknown'}",
            f"Dataset: {snap.get('dataset') or 'unknown'}",
            f"Metric: {metric_name}",
            f"Score: {score_txt}",
            f"Errors: {(snap.get('progress') or {}).get('errors', 0)}",
            f"Run id: {snap.get('run_id')}",
        ], 22, "1A1A1A", False, "F0F7E6"),
    ])
    strong_lines = [
        f"{c.get('category')}: {percent_value(c.get('accuracy'))} on n={c.get('total')}"
        for c in per_cat[:8]
    ]
    weak_lines = [
        f"{c.get('category')}: {percent_value(c.get('accuracy'))} on n={c.get('total')}"
        for c in list(reversed(per_cat[-8:]))
    ]
    slides.append([
        ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Where it is strongest"], 30, "1A1A1A", True),
        ppt_text_shape(3, "Strong", 0.8, 1.35, 11.7, 5.3, strong_lines or ["No completed scored groups yet."], 20),
    ])
    slides.append([
        ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Where to improve or compare"], 30, "1A1A1A", True),
        ppt_text_shape(3, "Weak", 0.8, 1.35, 11.7, 5.3, weak_lines or ["No misses available yet."], 20, "1A1A1A", False, "FFFBED"),
    ])
    example_lines = []
    for row in (wins[:3] + misses[:5])[:8]:
        example_lines.append(
            f"{row.get('question_id')}: score {percent_value(row.get('judge_score')) or 'n/a'} - {bench_clip_for_ppt(row.get('prediction') or row.get('error') or '', 120)}"
        )
    slides.append([
        ppt_text_shape(2, "Title", 0.7, 0.5, 11.5, 0.7, ["Trace examples to pull into slides"], 30, "1A1A1A", True),
        ppt_text_shape(3, "Examples", 0.8, 1.35, 11.7, 5.3, example_lines or ["No examples yet."], 16),
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


def bench_clip_for_ppt(value: Any, max_len: int) -> str:
    text = str(value or "").replace("\n", " ")
    return text[: max_len - 3] + "..." if len(text) > max_len else text


def _benchmark_copy_pptx_to_claude_docs(run_id: str, pptx_path: Path) -> Optional[str]:
    if not pptx_path.exists() or not CLAUDE_DOCS_PRESENTATIONS_DIR.exists():
        return None
    dest = CLAUDE_DOCS_PRESENTATIONS_DIR / f"{run_id}-executive-summary.pptx"
    try:
        shutil.copy2(pptx_path, dest)
        return str(dest)
    except Exception as exc:
        log(f"[benchmark {run_id}] Claude Docs PPTX copy failed: {exc}")
        return None


def _benchmark_write_durable_artifacts(run_id: str, snap: Dict[str, Any], *, final: bool = False) -> Dict[str, Any]:
    run_dir = _benchmark_run_dir(run_id)
    artifacts_dir = _benchmark_artifacts_dir(run_id)
    leaderboard_path = run_dir / "leaderboard.json"
    leaderboard_path.write_text(json.dumps(_benchmark_leaderboard(snap), default=str, indent=2), encoding="utf-8")
    _benchmark_write_trace_jsonl(run_id, snap)
    metadata = {
        "run_id": run_id,
        "adapter": snap.get("adapter"),
        "dataset": snap.get("dataset"),
        "source_config": snap.get("source_config"),
        "model": snap.get("model"),
        "base_url": snap.get("base_url"),
        "judge": snap.get("judge"),
        "started_epoch": snap.get("started_epoch"),
        "finished_epoch": snap.get("finished_epoch"),
        "generated_epoch": time.time(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2), encoding="utf-8")
    artifact_urls = {
        "leaderboard": f"/benchmark/leaderboard/{run_id}",
        "trace": f"/benchmark/trace/{run_id}.jsonl",
        "metadata": f"/benchmark/artifact/{run_id}/metadata.json",
    }
    predictions_path = _benchmark_write_lingoqa_predictions_csv(run_id, snap)
    if predictions_path is not None:
        artifact_urls["predictions_csv"] = f"/benchmark/artifact/{run_id}/predictions.csv"
    if final:
        report_path = artifacts_dir / "report.html"
        pptx_path = artifacts_dir / "executive-summary.pptx"
        try:
            report_path.write_text(benchmark_render_report(run_id), encoding="utf-8")
            artifact_urls["report_html"] = f"/benchmark/artifact/{run_id}/artifacts/report.html"
        except Exception as exc:
            log(f"[benchmark {run_id}] report artifact failed: {exc}")
        try:
            _benchmark_write_executive_pptx(snap, pptx_path)
            artifact_urls["executive_pptx"] = f"/benchmark/artifact/{run_id}/artifacts/executive-summary.pptx"
            copied = _benchmark_copy_pptx_to_claude_docs(run_id, pptx_path)
            if copied:
                artifact_urls["claude_docs_pptx"] = copied
        except Exception as exc:
            log(f"[benchmark {run_id}] PPTX artifact failed: {exc}")
    return artifact_urls


# ----------------------------------------------------------------------------
# Drop-zone Benchmark View additions (sprint: "paste a URL → pick a model → hit Run")
# ----------------------------------------------------------------------------

BENCHMARK_HISTORY_PATH = BENCHMARK_RUN_ROOT / "history.json"
BENCHMARK_HISTORY_MAX = 20
BENCHMARK_MULTI_NIM_GLOB = BENCHMARK_RUN_ROOT / "multi-nim-20260517"


def _benchmark_history_load() -> List[Dict[str, Any]]:
    def from_run_dirs() -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for results_path in sorted(BENCHMARK_RUN_ROOT.glob("bench-*/results.json")):
            try:
                snap = json.loads(results_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary_obj = snap.get("summary") or _benchmark_summary_for_snap(snap)
            entries.append({
                "run_id": snap.get("run_id") or results_path.parent.name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(snap.get("started_epoch") or results_path.stat().st_mtime)),
                "epoch": snap.get("started_epoch") or results_path.stat().st_mtime,
                "model": snap.get("model") or "",
                "dataset": snap.get("dataset") or "",
                "adapter": snap.get("adapter") or _benchmark_adapter_for_dataset(snap.get("dataset") or ""),
                "judge": snap.get("judge") or "",
                "sample_size": snap.get("sample_size"),
                "concurrency": snap.get("concurrency"),
                "seed": snap.get("seed"),
                "accuracy": summary_obj.get("overall_accuracy"),
                "map": summary_obj.get("map"),
                "total_predictions": summary_obj.get("total_predictions"),
                "wall_seconds": snap.get("wall_seconds"),
                "report_url": f"/benchmark/report/{snap.get('run_id') or results_path.parent.name}",
            })
        return entries
    try:
        disk_entries = from_run_dirs()
        if not BENCHMARK_HISTORY_PATH.exists():
            return disk_entries[-BENCHMARK_HISTORY_MAX:]
        data = json.loads(BENCHMARK_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            by_id: Dict[str, Dict[str, Any]] = {}
            for entry in disk_entries + data:
                rid = entry.get("run_id")
                if rid:
                    by_id[str(rid)] = entry
            merged = sorted(by_id.values(), key=lambda e: float(e.get("epoch") or 0))
            return merged[-BENCHMARK_HISTORY_MAX:]
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


def _gate_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _gate_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _gate_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%" if float(value) <= 1 else f"{float(value):.1f}%"
    except Exception:
        return "n/a"


def _rf100_gate_status_load(path: Optional[Path] = None) -> Dict[str, Any]:
    status_path = path or RF100_GATE_STATUS_PATH
    if not status_path.exists():
        return {
            "ok": False,
            "error": f"Gate status snapshot not found: {status_path}",
            "path": str(status_path),
            "results": [],
            "statuses": {},
        }
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Gate status snapshot unreadable: {exc}",
            "path": str(status_path),
            "results": [],
            "statuses": {},
        }
    payload.setdefault("ok", True)
    payload.setdefault("path", str(status_path))
    payload.setdefault("results", [])
    payload.setdefault("statuses", {})
    return payload


def _rf100_gate_lane_rows(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    lanes: List[Dict[str, Any]] = []
    for row in status.get("results") or []:
        if not isinstance(row, dict):
            continue
        data = row.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        name = str(row.get("name") or "")
        gpu_text = str(data.get("gpu") or "")
        gpu_utils = []
        for line in gpu_text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpu_utils.append(_gate_int(parts[1]))
        lanes.append({
            "name": name,
            "brev_status": row.get("brev_status") or (status.get("statuses") or {}).get(name),
            "model": data.get("model") or "",
            "run_id": data.get("run_id") or "",
            "shard": f"{data.get('shard_index')}/{data.get('shard_count')}",
            "concurrency": _gate_int(data.get("concurrency")),
            "done": _gate_int(data.get("done")),
            "total": _gate_int(data.get("total")),
            "remaining": _gate_int(data.get("remaining")),
            "errors": _gate_int(data.get("errors")),
            "delta_done": _gate_int(data.get("delta_done")),
            "delta_errors": _gate_int(data.get("delta_errors")),
            "elapsed_h": _gate_number(data.get("elapsed_h")),
            "rate_h": _gate_number(data.get("rate_h")),
            "eta_h": _gate_number(data.get("eta_h"), default=-1.0),
            "gpu_util_max": max(gpu_utils) if gpu_utils else None,
            "gpu": gpu_text,
            "df": data.get("df") or [],
            "models": data.get("models") or {},
        })
    return lanes


def _rf100_gate_aggregate(lanes: List[Dict[str, Any]]) -> Dict[str, Any]:
    done = sum(_gate_int(l.get("done")) for l in lanes)
    total = sum(_gate_int(l.get("total")) for l in lanes)
    errors = sum(_gate_int(l.get("errors")) for l in lanes)
    delta_done = sum(_gate_int(l.get("delta_done")) for l in lanes)
    delta_errors = sum(_gate_int(l.get("delta_errors")) for l in lanes)
    rate_h = sum(_gate_number(l.get("rate_h")) for l in lanes)
    remaining = max(0, total - done) if total else None
    eta_h = (remaining / rate_h) if remaining is not None and rate_h > 0 else None
    return {
        "lanes": len(lanes),
        "done": done,
        "total": total,
        "remaining": remaining,
        "pct": (done / total) if total else None,
        "errors": errors,
        "delta_done": delta_done,
        "delta_errors": delta_errors,
        "rate_h": rate_h,
        "eta_h": eta_h,
    }


def _rf100_gate_air_support_guidance(c3_lanes: List[Dict[str, Any]], cr2_lanes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Operational guidance learned from Gate 1.

    The important product behavior: large RF100-VL runs should not start by
    fanning out five expensive Brevs. The UI should ask for more capacity only
    after a single lane has proved that endpoint, dataset staging, artifact
    flush, and scoring are stable enough to make duplication worth the cost.
    """
    all_lanes = c3_lanes + cr2_lanes
    stable_candidates = []
    for lane in all_lanes:
        done = _gate_int(lane.get("done"))
        errors = _gate_int(lane.get("errors"))
        elapsed_s = _gate_number(lane.get("elapsed_h")) * 3600
        error_rate = (errors / done) if done else 0.0
        gpu_ok = lane.get("gpu_util_max") is None or _gate_int(lane.get("gpu_util_max")) >= 50
        stable = (
            done >= AIR_SUPPORT_MIN_STABLE_IMAGES
            and elapsed_s >= AIR_SUPPORT_MIN_STABLE_SECONDS
            and error_rate <= AIR_SUPPORT_MAX_ERROR_RATE
            and _gate_number(lane.get("rate_h")) > 0
            and gpu_ok
        )
        stable_candidates.append({
            "name": lane.get("name"),
            "model": lane.get("model"),
            "done": done,
            "elapsed_minutes": round(elapsed_s / 60, 1) if elapsed_s else 0,
            "error_rate": error_rate,
            "rate_h": lane.get("rate_h"),
            "gpu_util_max": lane.get("gpu_util_max"),
            "stable": stable,
        })
    healthy_c3 = [l for l in c3_lanes if _gate_int(l.get("errors")) == 0 and _gate_number(l.get("rate_h")) > 0]
    healthy_cr2 = [
        l for l in cr2_lanes
        if _gate_int(l.get("errors")) == 0 and _gate_number(l.get("rate_h")) > 0 and _gate_int(l.get("delta_done")) > 0
    ]
    blocked = [
        l for l in cr2_lanes
        if _gate_int(l.get("delta_done")) == 0 or _gate_int(l.get("delta_errors")) > 0 or _gate_int(l.get("gpu_util_max") or 0) == 0
    ]
    decision = "hold"
    if len(healthy_c3) >= 1 and not blocked:
        decision = "scale_to_two"
    if len(healthy_c3) >= 2 and len(healthy_cr2) >= 1 and not blocked:
        decision = "scale_to_five"
    return {
        "decision": decision,
        "stable_lane_thresholds": {
            "min_images": AIR_SUPPORT_MIN_STABLE_IMAGES,
            "min_minutes": round(AIR_SUPPORT_MIN_STABLE_SECONDS / 60, 1),
            "max_error_rate": AIR_SUPPORT_MAX_ERROR_RATE,
            "required_signals": [
                "model id confirmed through /v1/models",
                "dataset rows are uniquely sharded with a manifest",
                "trace.jsonl and results.json flush every checkpoint",
                "throughput and error rate are stable over two windows",
                "no operator SSH intervention required during the window",
            ],
        },
        "stable_candidates": stable_candidates,
        "recommendations": [
            "Run one Brev lane for 30-45 minutes or at least 500 images before asking for air support.",
            "Scale from one to two lanes first; only scale to five after two lanes prove unique sharding and comparable throughput.",
            "Use a central manifest/lease file so new lanes claim non-overlapping image ids instead of duplicating work.",
            "Auto-pause scale-out when a lane has zero delta, rising error rows, unreachable SSH, dead proxy ports, or idle loaded GPUs.",
            "Surface cost burn, failed setup time, and duplicate inference risk in the UI before provisioning more GPUs.",
        ],
        "blocked_lanes": blocked,
    }


def _rf100_gate_learnings() -> List[Dict[str, str]]:
    return [
        {
            "title": "Air support is a product workflow, not a shell trick",
            "body": "Gate 1 needed constant human routing across Brev, SSH, vLLM, NIM, storage, endpoint identity, sharding, and cost. The UI should own that state explicitly.",
        },
        {
            "title": "Scale only after a stable canary",
            "body": "A single lane should prove model identity, request shape, dataset staging, artifact flushing, and low error rate for at least 30-45 minutes or 500 images before the UI asks for more Brevs.",
        },
        {
            "title": "Sharding must be leased, not remembered",
            "body": "Five machines should claim work from a durable manifest. Static modulo sharding is easy to duplicate or strand when a lane is restarted, replaced, or silently fails.",
        },
        {
            "title": "Cost controls need first-class states",
            "body": "Loaded-but-idle, dead proxy, stale progress, and SSH-unreachable are budget states. The UI should show them as billable risk with suggested actions rather than burying them in logs.",
        },
        {
            "title": "Partial reports must say what they can prove",
            "body": "A partial Gate 1 report can compare coverage, throughput, errors, and matched trace rows. It should not imply model quality parity unless the same image ids were scored by both models.",
        },
    ]


def _rf100_gate_interim_payload(status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    status = status or _rf100_gate_status_load()
    lanes = _rf100_gate_lane_rows(status)
    c3_lanes = [l for l in lanes if str(l.get("model")) == "nvidia/Cosmos3-Super-Reasoner-May17"]
    cr2_lanes = [l for l in lanes if str(l.get("model")) == "nvidia/cosmos-reason2-8b"]
    cr2_productive = [
        l for l in cr2_lanes
        if _gate_int(l.get("errors")) == 0 and _gate_int(l.get("done")) > 0 and l.get("name") not in {"rf100-cr2-dual-1", "rf100-cr2-dual-3", "nim-mb-cr2-8b-rf100"}
    ]
    cr2_waste = [
        l for l in cr2_lanes
        if l.get("name") in {"rf100-cr2-dual-1", "rf100-cr2-dual-3", "nim-mb-cr2-8b-rf100"}
        or _gate_int(l.get("delta_errors")) > 0
        or (_gate_int(l.get("delta_done")) == 0 and _gate_int(l.get("done")) > 0)
    ]
    comparison_statuses = status.get("comparison_statuses") or {
        key: (status.get("statuses") or {}).get(key)
        for key in ("qw3-8b-05110800", "nim-mb-omni", "nim-mb-gemma", "nim-mb-nano12b", "nim-mb-reason1")
    }
    payload = {
        "ok": bool(status.get("ok", True)),
        "source_status_path": status.get("path") or str(RF100_GATE_STATUS_PATH),
        "generated_epoch": time.time(),
        "dataset": "RF100-VL",
        "method": {
            "source": "RF100-VL official task framing: detection prompts over Roboflow 100 Vision Language image groups.",
            "metric": "COCO-style AP@[.50:.95], AP50, AP75, and per-group AP when matched trace rows are available.",
            "interim_caveat": "This Gate 1 interim report is based on partial live progress. Apples-to-apples quality comparison requires matched question_id/image_id trace rows for both models.",
        },
        "aggregates": {
            "c3": _rf100_gate_aggregate(c3_lanes),
            "cr2_all_attempted": _rf100_gate_aggregate(cr2_lanes),
            "cr2_productive": _rf100_gate_aggregate(cr2_productive),
            "cr2_waste": _rf100_gate_aggregate(cr2_waste),
        },
        "lanes": {
            "c3": c3_lanes,
            "cr2": cr2_lanes,
            "cr2_productive": cr2_productive,
            "cr2_waste": cr2_waste,
        },
        "comparison_statuses": comparison_statuses,
        "air_support": _rf100_gate_air_support_guidance(c3_lanes, cr2_lanes),
        "learnings": _rf100_gate_learnings(),
        "next_gate": [
            "Generate an interim report now, clearly marked partial.",
            "Merge raw trace rows by question_id before claiming apples-to-apples model quality.",
            "Run Cosmos3 on a matched CR2 subset or score only the intersection of completed rows.",
            "Add manifest-based sharding and scale-readiness prompts before any future RF100 full gate.",
            "Keep cost/risk visible: idle loaded GPU, proxy failure, SSH unreachable, stale delta, duplicate trace rows.",
        ],
    }
    return payload


def benchmark_render_rf100_gate_interim(payload: Optional[Dict[str, Any]] = None) -> str:
    payload = payload or _rf100_gate_interim_payload()
    generated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(payload.get("generated_epoch") or time.time()))
    ag = payload.get("aggregates") or {}
    c3 = ag.get("c3") or {}
    cr2_prod = ag.get("cr2_productive") or {}
    cr2_waste = ag.get("cr2_waste") or {}
    air = payload.get("air_support") or {}

    def esc(v: Any) -> str:
        return html.escape(str(v if v is not None else ""))

    def n(v: Any) -> str:
        try:
            return f"{int(float(v)):,}"
        except Exception:
            return "0"

    def h(v: Any) -> str:
        try:
            if v is None or float(v) < 0:
                return "n/a"
            return f"{float(v):.1f}h"
        except Exception:
            return "n/a"

    def lane_rows(lanes: List[Dict[str, Any]]) -> str:
        if not lanes:
            return "<tr><td colspan='9'>No lanes.</td></tr>"
        rows = []
        for lane in lanes:
            rows.append(
                "<tr>"
                f"<td>{esc(lane.get('name'))}</td>"
                f"<td>{esc(lane.get('model'))}</td>"
                f"<td>{esc(lane.get('run_id'))}</td>"
                f"<td>{esc(lane.get('shard'))}</td>"
                f"<td>{n(lane.get('done'))}/{n(lane.get('total'))}</td>"
                f"<td>{n(lane.get('delta_done'))}</td>"
                f"<td>{n(lane.get('errors'))}</td>"
                f"<td>{esc(lane.get('gpu_util_max') if lane.get('gpu_util_max') is not None else 'n/a')}%</td>"
                f"<td>{h(lane.get('eta_h'))}</td>"
                "</tr>"
            )
        return "".join(rows)

    learnings = "".join(
        f"<li><strong>{esc(item.get('title'))}</strong><br>{esc(item.get('body'))}</li>"
        for item in payload.get("learnings") or []
    )
    recs = "".join(f"<li>{esc(item)}</li>" for item in (air.get("recommendations") or []))
    next_gate = "".join(f"<li>{esc(item)}</li>" for item in payload.get("next_gate") or [])
    comparisons = payload.get("comparison_statuses") or {}
    comparison_rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in sorted(comparisons.items()))
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>RF100-VL Gate 1 Interim Report</title>
<style>
:root{{--green:#76B900;--dark:#1A1A1A;--line:#d8dee8;--soft:#F0F5E8;--warn:#92400e;--bad:#b91c1c;}}
body{{margin:0;font-family:Inter,system-ui,sans-serif;color:var(--dark);line-height:1.5;background:#fff;}}
header{{background:var(--dark);color:#fff;padding:34px 42px;border-left:8px solid var(--green);}}
h1{{margin:0 0 8px;font-size:36px;}} main{{max-width:1220px;margin:0 auto;padding:32px;}}
section{{margin:0 0 34px;}} h2{{font-size:18px;text-transform:uppercase;letter-spacing:.04em;border-bottom:2px solid var(--green);padding-bottom:6px;}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}} .tile{{border:1px solid var(--line);border-left:5px solid var(--green);border-radius:8px;padding:16px;background:#fff;}}
.tile .v{{font-size:28px;font-weight:800;color:var(--green);}} .tile .l{{font-size:12px;color:#64748b;text-transform:uppercase;}}
.note{{background:var(--soft);border-left:5px solid var(--green);padding:14px 18px;border-radius:0 8px 8px 0;}}
.warn{{background:#fff7ed;border-left-color:var(--warn);}} .bad{{background:#fef2f2;border-left-color:var(--bad);}}
table{{width:100%;border-collapse:collapse;font-size:13px;}} th{{background:var(--dark);color:#fff;text-align:left;padding:9px;}} td{{border-bottom:1px solid var(--line);padding:8px;vertical-align:top;}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:18px;}} code{{font-family:ui-monospace,Menlo,monospace;}}
@media(max-width:850px){{.tiles,.cols{{grid-template-columns:1fr;}}}}
</style></head><body>
<header><h1>RF100-VL Gate 1 Interim Report</h1>
<div>Generated {esc(generated)} | Source: <code>{esc(payload.get('source_status_path'))}</code></div></header>
<main>
<section><h2>Executive Readout</h2><div class='tiles'>
<div class='tile'><div class='v'>{_gate_percent(c3.get('pct'))}</div><div class='l'>Cosmos3 coverage</div></div>
<div class='tile'><div class='v'>{n(c3.get('done'))}</div><div class='l'>Cosmos3 images</div></div>
<div class='tile'><div class='v'>{n(cr2_prod.get('delta_done'))}</div><div class='l'>CR2 useful delta</div></div>
<div class='tile'><div class='v'>{n(cr2_waste.get('lanes'))}</div><div class='l'>CR2 waste lanes</div></div>
</div></section>
<section><h2>Method Guardrail</h2><div class='note warn'>
RF100-VL quality must be scored with COCO-style AP on matched image ids. This interim report can compare coverage, throughput, operational stability, and partial trace availability. It should not claim a model-quality winner until C3 and CR2 rows are joined by <code>question_id</code>/<code>image_id</code>.
</div></section>
<section><h2>Current Progress</h2>
<table><thead><tr><th>Track</th><th>Lanes</th><th>Done / Total</th><th>Delta</th><th>Errors</th><th>Rate</th><th>ETA</th></tr></thead><tbody>
<tr><td>Cosmos3 Super Reasoner May17</td><td>{n(c3.get('lanes'))}</td><td>{n(c3.get('done'))} / {n(c3.get('total'))}</td><td>{n(c3.get('delta_done'))}</td><td>{n(c3.get('errors'))}</td><td>{n(c3.get('rate_h'))}/h</td><td>{h(c3.get('eta_h'))}</td></tr>
<tr><td>Cosmos Reason 2 productive lanes</td><td>{n(cr2_prod.get('lanes'))}</td><td>{n(cr2_prod.get('done'))} / {n(cr2_prod.get('total'))}</td><td>{n(cr2_prod.get('delta_done'))}</td><td>{n(cr2_prod.get('errors'))}</td><td>{n(cr2_prod.get('rate_h'))}/h</td><td>{h(cr2_prod.get('eta_h'))}</td></tr>
<tr><td>Cosmos Reason 2 waste/stuck lanes</td><td>{n(cr2_waste.get('lanes'))}</td><td>{n(cr2_waste.get('done'))} / {n(cr2_waste.get('total'))}</td><td>{n(cr2_waste.get('delta_done'))}</td><td>{n(cr2_waste.get('errors'))}</td><td>{n(cr2_waste.get('rate_h'))}/h</td><td>{h(cr2_waste.get('eta_h'))}</td></tr>
</tbody></table></section>
<section><h2>Air Support Readiness</h2><div class='note bad'>
Decision: <strong>{esc(air.get('decision'))}</strong>. Gate 1 shows that scaling to five Brevs before a stable canary creates duplicate work, expensive idle states, and heavy SRE load.
</div><ul>{recs}</ul></section>
<section><h2>Lane Details</h2>
<h3>Cosmos3 Super Reasoner</h3><table><thead><tr><th>Brev</th><th>Model</th><th>Run id</th><th>Shard</th><th>Done</th><th>Delta</th><th>Errors</th><th>GPU</th><th>ETA</th></tr></thead><tbody>{lane_rows(payload.get('lanes', {}).get('c3') or [])}</tbody></table>
<h3>Cosmos Reason 2</h3><table><thead><tr><th>Brev</th><th>Model</th><th>Run id</th><th>Shard</th><th>Done</th><th>Delta</th><th>Errors</th><th>GPU</th><th>ETA</th></tr></thead><tbody>{lane_rows(payload.get('lanes', {}).get('cr2') or [])}</tbody></table></section>
<section><h2>Gate 1 Learnings</h2><ul>{learnings}</ul></section>
<section><h2>Next Gate Product Changes</h2><ul>{next_gate}</ul></section>
<section><h2>Stopped Comparisons</h2><table><thead><tr><th>Lane</th><th>Status</th></tr></thead><tbody>{comparison_rows}</tbody></table></section>
</main></body></html>"""


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
    if _is_rf100_source_text(s):
        return ("rf100_vl", "rf100-vl")
    if _is_lingoqa_source_text(s) and ("github.com" in low or "arxiv.org" in low):
        return ("lingoqa", "lingoqa-official")
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
    if kind == "rf100_vl":
        profile = _rf100_discover(download=False)
        result["dataset_config"] = {
            "id": "rf100-vl",
            "name": "RF100-VL (Roboflow 100 Vision Language)",
            "adapter": "rf100-vl",
            "task": "object_detection",
            "judge": "coco-ap",
            "metric": profile.get("metric"),
            "ready": bool(profile.get("ready")),
            "path": profile.get("path"),
            "rows": profile.get("image_count") or None,
            "groups": profile.get("group_count") or 0,
            "expected_groups": profile.get("expected_groups"),
            "full_gate_ready": bool(profile.get("full_gate_ready")),
            "blockers": profile.get("blockers") or [],
            "source_urls": profile.get("source_urls") or [],
        }
        result["display"] = (
            f"RF100-VL detected: {profile.get('group_count') or 0}/"
            f"{profile.get('expected_groups')} groups staged, "
            f"{profile.get('image_count') or 0} images."
        )
        result["profile"] = profile
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


def _benchmark_adapter_for_dataset(dataset_id: str) -> str:
    ds = str(dataset_id or "")
    if ds == "rf100-vl" or ds.startswith("rf100-vl:"):
        return "rf100-vl"
    return "lingoqa"


def _benchmark_analyze_source(url: str) -> Dict[str, Any]:
    resolved = _resolve_url_to_dataset_cached(url)
    cfg = resolved.get("dataset_config") or {}
    adapter = cfg.get("adapter") or _benchmark_adapter_for_dataset(cfg.get("id") or "")
    if adapter == "rf100-vl":
        profile = resolved.get("profile") or _rf100_discover(download=False)
        return {
            "ok": True,
            "adapter": "rf100-vl",
            "dataset_config": cfg,
            "resolved": resolved,
            "summary": {
                "title": "RF100-VL",
                "task": "Object detection over 100 Roboflow Universe datasets",
                "metric": "COCO-style AP@[.50:.95], AP50, AP75, per-group AP",
                "dataset": "164,149 images and 1,355,491 annotations across seven domains when fully staged",
                "flow": "Analyze source, stage/download COCO folders, scope groups, preview prompts, run model, score detections, export leaderboard and trace.",
            },
            "readiness": {
                "ready": bool(profile.get("ready")),
                "full_gate_ready": bool(profile.get("full_gate_ready")),
                "blockers": profile.get("blockers") or [],
                "path": profile.get("path"),
                "groups": profile.get("group_count") or 0,
                "expected_groups": profile.get("expected_groups"),
                "images": profile.get("image_count") or 0,
            },
            "stages": ["Source", "Understand", "Scope", "Preview", "Run", "Review", "Export"],
            "profile": profile,
        }
    base = _lingoqa_dataset_dir()
    return {
        "ok": True,
        "adapter": "lingoqa",
        "dataset_config": cfg or {
            "id": "lingoqa-official",
            "name": "LingoQA (1000 rows)",
            "adapter": "lingoqa",
            "judge": "lingo-judge",
        },
        "resolved": resolved,
        "summary": {
            "title": "LingoQA",
            "task": "Autonomous-driving visual question answering",
            "metric": "Lingo-Judge text classifier, max over two reference answers",
            "dataset": "Evaluation split has 100 scenarios and 1000 QA pairs.",
            "flow": "Analyze source, use local GDrive cache, scope rows, preview frames and questions, run VQA, score with Lingo-Judge, export trace.",
        },
        "readiness": {
            "ready": bool(base and (base / "val.parquet").exists()),
            "full_gate_ready": bool(base and (base / "val.parquet").exists()),
            "blockers": [] if base else ["Stage LingoQA evaluation data under /home/horde/lingoqa-data or /tmp/lingoqa-data."],
            "path": str(base) if base else None,
            "groups": 1,
            "expected_groups": 1,
            "images": 1000,
        },
        "stages": ["Source", "Understand", "Scope", "Preview", "Run", "Review", "Export"],
    }


def _benchmark_preview_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question_id": row.get("question_id"),
        "dataset_name": row.get("dataset_name"),
        "category": row.get("category"),
        "task": row.get("task") or "vqa",
        "question": row.get("question"),
        "references": row.get("references") or [],
        "images": row.get("images") or [],
        "ground_truth_count": len(row.get("ground_truth") or []),
        "categories": row.get("categories") or [],
    }


def _benchmark_load_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = str(payload.get("dataset") or payload.get("dataset_id") or "").strip()
    if not dataset_id and payload.get("url"):
        resolved = _resolve_url_to_dataset_cached(str(payload.get("url") or ""))
        dataset_id = ((resolved.get("dataset_config") or {}).get("id") or "")
    if not dataset_id:
        dataset_id = "lingoqa-official"
    source_config = payload.get("source_config")
    if not isinstance(source_config, dict):
        source_config = None
    adapter = _benchmark_adapter_for_dataset(dataset_id)
    if adapter == "rf100-vl":
        profile = _rf100_discover(download=False)
        if not profile.get("ready"):
            return {
                "ok": False,
                "adapter": adapter,
                "dataset": dataset_id,
                "profile": profile,
                "blockers": profile.get("blockers") or [],
                "preview": [],
            }
        rows = _load_rf100vl_dataset({**(source_config or {}), "download": False}, download=False)
    else:
        rows = load_dataset_by_id(dataset_id, source_config)
    limit = max(1, min(20, int(payload.get("preview_size") or 5)))
    groups = sorted({str(r.get("dataset_name") or r.get("category") or "default") for r in rows})
    return {
        "ok": True,
        "adapter": adapter,
        "dataset": dataset_id,
        "total": len(rows),
        "group_count": len(groups),
        "groups": groups[:200],
        "preview": [_benchmark_preview_row(r) for r in rows[:limit]],
    }


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
    # Operator-supplied external endpoints. This lets the same workflow compare
    # local and sibling Horde/NIM instances without hardcoding one-off IPs.
    # Preferred JSON:
    #   BENCHMARK_EXTRA_MODELS_JSON='[{"id":"model","endpoint":"http://host:8000/v1","host":"horde"}]'
    # Compact fallback:
    #   BENCHMARK_EXTRA_MODELS='model=http://host:8000/v1,other=http://host2:8000/v1'
    extra_json = os.getenv("BENCHMARK_EXTRA_MODELS_JSON", "").strip()
    if extra_json:
        try:
            extra = json.loads(extra_json)
            if isinstance(extra, dict):
                extra = [extra]
            for item in extra or []:
                if not isinstance(item, dict):
                    continue
                if not item.get("id") or not item.get("endpoint"):
                    continue
                out.append({
                    "id": str(item.get("id")),
                    "name": str(item.get("name") or item.get("id")),
                    "endpoint": str(item.get("endpoint")),
                    "status": str(item.get("status") or "ready"),
                    "host": str(item.get("host") or "external"),
                })
        except Exception as exc:
            out.append({
                "id": "extra-model-config-error",
                "name": "Extra model config error",
                "endpoint": "",
                "status": "error",
                "host": "env",
                "error": str(exc),
            })
    extra_csv = os.getenv("BENCHMARK_EXTRA_MODELS", "").strip()
    if extra_csv:
        for chunk in extra_csv.split(","):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                continue
            model_id, endpoint = chunk.split("=", 1)
            model_id = model_id.strip()
            endpoint = endpoint.strip()
            if model_id and endpoint:
                out.append({
                    "id": model_id,
                    "name": model_id,
                    "endpoint": endpoint,
                    "status": "ready",
                    "host": urllib.parse.urlparse(endpoint).hostname or "external",
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
    deduped: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for item in out:
        key = str(item.get("id") or "")
        if key and key in seen_ids:
            continue
        if key:
            seen_ids.add(key)
        deduped.append(item)
    return deduped


def _benchmark_model_config(selected_model: Optional[str] = None) -> Dict[str, Any]:
    selected = str(selected_model or "").strip()
    models = _benchmark_available_models()
    if selected:
        for item in models:
            if item.get("id") == selected:
                endpoint = item.get("endpoint") or benchmark_nim_base_url()
                return {
                    "id": item.get("id") or selected,
                    "endpoint": endpoint,
                    "headers": benchmark_nim_headers(),
                    "host": item.get("host") or "",
                    "status": item.get("status") or "ready",
                }
    model = benchmark_detect_model()
    return {
        "id": model,
        "endpoint": benchmark_nim_base_url(),
        "headers": benchmark_nim_headers(),
        "host": "horde (local)",
        "status": "ready",
    }


def benchmark_get(run_id: str) -> Optional[Dict[str, Any]]:
    with BENCHMARK_LOCK:
        snap = BENCHMARK_STATE["runs"].get(run_id)
        if snap:
            return json.loads(json.dumps(snap, default=str))
    disk = _benchmark_snapshot_load_from_disk(run_id)
    if disk:
        with BENCHMARK_LOCK:
            BENCHMARK_STATE["runs"][run_id] = disk
        return json.loads(json.dumps(disk, default=str))
    return None


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
        record.setdefault("completed_epoch", time.time())
        results.append(record)
        now = time.time()
        recent = [
            float(r.get("completed_epoch"))
            for r in results
            if r.get("completed_epoch") is not None
            and now - float(r.get("completed_epoch")) <= 15 * 60
        ]
        snap["progress"] = {
            "done": len(results),
            "total": snap.get("total") or 0,
            "errors": sum(1 for r in results if r.get("error")),
            "last_result_epoch": record.get("completed_epoch"),
            "recent_15m_done": len(recent),
            "recent_15m_images_per_hour": round(len(recent) * 4.0, 1),
        }
        snap["updated_epoch"] = now
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
    # [CLS]\nQuestion:\nAnswer:\nStudent: string exceeded the judge's
    # 128-token window (hardcoded `max_length=128` in the official wayveai/LingoQA
    # benchmark/judge.py — NOT DeBERTa's 512 architectural cap). Surfaced in
    # the Recent Samples header so we can spot reasoning models being
    # silently under-scored at the population level.
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


def _benchmark_summary_for_snap(snap: Dict[str, Any]) -> Dict[str, Any]:
    adapter = snap.get("adapter") or _benchmark_adapter_for_dataset(snap.get("dataset") or "")
    if adapter == "rf100-vl":
        return _rf100_summarize(snap)
    return _benchmark_summarize(snap)


def run_lingoqa_benchmark(
    run_id: str,
    dataset_id: str,
    judge_id: str,
    sample_size: int,
    concurrency: int,
    seed: int,
    source_config: Optional[Dict[str, Any]] = None,
    judge_mode: str = "standard",
    model_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Benchmark worker thread. Mutates BENCHMARK_STATE['runs'][run_id].

    Despite the historical name, this worker now drives any dataset
    dispatched by load_dataset_by_id — not just LingoQA. The function name
    is kept for backwards compatibility with the API handler call site.

    judge_mode (Phase J amendment — dual judge-mode A/B):
      - "standard"    : judge sees result["prediction"] (default; back-compat).
      - "answer-only" : judge sees _extract_final_answer(prediction). The
                        <think>...</think> chain is removed before scoring so
                        the brief final answer fits in the protocol's
                        128-token window (hardcoded `max_length=128` in the
                        official wayveai/LingoQA benchmark/judge.py — more
                        aggressive than DeBERTa's 512 architectural cap).
                        Mirrors the brief gt-A/gt-B reference style.
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
            adapter="lingoqa",
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

        model_info = model_config or _benchmark_model_config(None)
        model = model_info["id"]
        base = model_info["endpoint"]
        headers = model_info.get("headers") or benchmark_nim_headers()
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
                    # fit under the 128-token window (the protocol's hardcoded
                    # max_length=128 — see LINGO_JUDGE_MAX_TOKENS). The whole
                    # point of the A/B. Per-row fields:
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
                    summary = _benchmark_summary_for_snap(snap)
                    snap = _benchmark_update(run_id, summary=summary)
                    _benchmark_snapshot_save(run_id, snap)
                    _benchmark_write_durable_artifacts(run_id, snap, final=False)

        final = _benchmark_update(
            run_id,
            status="complete",
            finished_epoch=time.time(),
            wall_seconds=time.time() - started,
            summary=_benchmark_summary_for_snap(benchmark_get(run_id) or {}),
        )
        _benchmark_snapshot_save(run_id, final)
        artifacts = _benchmark_write_durable_artifacts(run_id, final, final=True)
        final = _benchmark_update(run_id, artifacts=artifacts)
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
                "map": summary_obj.get("map"),
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


def run_rf100vl_benchmark(
    run_id: str,
    dataset_id: str,
    sample_size: int,
    concurrency: int,
    seed: int,
    source_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> None:
    started = time.time()
    source_config = source_config or {}
    try:
        _benchmark_update(
            run_id,
            adapter="rf100-vl",
            dataset=dataset_id,
            judge="coco-ap",
            sample_size=sample_size,
            concurrency=concurrency,
            seed=seed,
            started_epoch=started,
            status="loading_dataset",
        )
        questions = load_dataset_by_id(dataset_id, source_config)
        if not questions:
            raise RuntimeError("RF100-VL loader returned 0 images")
        group_count = len({q.get("dataset_name") for q in questions})
        if source_config.get("gate_run") and group_count < RF100_VL_EXPECTED_GROUPS:
            raise RuntimeError(
                f"RF100-VL full gate requires {RF100_VL_EXPECTED_GROUPS} groups; "
                f"loaded {group_count}. Stage the official RF100-VL COCO download or provide missing groups."
            )
        shard_count = int(source_config.get("shard_count") or 1)
        shard_index = int(source_config.get("shard_index") or 0)
        if shard_count > 1:
            if shard_index < 0 or shard_index >= shard_count:
                raise RuntimeError(f"RF100-VL shard_index must be in [0, {shard_count - 1}], got {shard_index}")
            questions = [q for idx, q in enumerate(questions) if idx % shard_count == shard_index]
            if not questions:
                raise RuntimeError(f"RF100-VL shard {shard_index}/{shard_count} has 0 images")
        if sample_size and sample_size > 0 and sample_size < len(questions):
            import random as _random
            rng = _random.Random(seed)
            questions = rng.sample(questions, sample_size)
        model_info = model_config or _benchmark_model_config(None)
        model = model_info["id"]
        base = model_info["endpoint"]
        headers = model_info.get("headers") or benchmark_nim_headers()
        resume_existing = os.getenv("BENCHMARK_RESUME_EXISTING", "1").lower() not in ("0", "false", "no")
        retry_errors = os.getenv("BENCHMARK_RETRY_ERRORS", "1").lower() not in ("0", "false", "no")
        existing_results: List[Dict[str, Any]] = []
        if resume_existing:
            prior = _benchmark_snapshot_load_from_disk(run_id) or {}
            seen_question_ids: set = set()
            retry_count = 0
            for row in prior.get("results") or []:
                question_id = str(row.get("question_id") or "")
                if not question_id or question_id in seen_question_ids:
                    continue
                if row.get("error") and retry_errors:
                    retry_count += 1
                    continue
                existing_results.append(row)
                seen_question_ids.add(question_id)
            if seen_question_ids:
                original_count = len(questions)
                questions = [q for q in questions if str(q.get("question_id") or "") not in seen_question_ids]
                log(
                    f"[benchmark {run_id}] resume enabled: seeded {len(existing_results)} existing results; "
                    f"retrying {retry_count} errors; remaining {len(questions)}/{original_count}"
                )
        _benchmark_update(
            run_id,
            status="running",
            total=len(questions) + len(existing_results),
            model=model,
            base_url=base,
            results=existing_results,
            rf100_group_count=group_count,
            rf100_expected_groups=RF100_VL_EXPECTED_GROUPS,
            shard_count=shard_count,
            shard_index=shard_index,
            in_flight_limit=max(1, concurrency) * max(1, RF100_VL_MAX_IN_FLIGHT_MULTIPLIER),
            source_config=source_config,
        )
        flush_every = 25
        results_collected = 0
        pending = deque(questions)
        in_flight_limit = max(max(1, concurrency), max(1, concurrency) * max(1, RF100_VL_MAX_IN_FLIGHT_MULTIPLIER))
        last_flush = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures: Dict[Any, Dict[str, Any]] = {}

            def submit_more() -> None:
                while pending and len(futures) < in_flight_limit:
                    q_next = pending.popleft()
                    futures[ex.submit(run_one_rf100_detection, q_next, model, base, headers)] = q_next

            submit_more()
            while futures:
                done_futures, _ = concurrent.futures.wait(
                    futures,
                    timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done_futures:
                    snap_idle = benchmark_get(run_id) or {}
                    if snap_idle:
                        _benchmark_snapshot_save(run_id, snap_idle)
                    continue
                for fut in done_futures:
                    q = futures.pop(fut)
                    try:
                        result = fut.result()
                        result["error"] = None
                    except Exception as exc:
                        log(f"[benchmark {run_id}] RF100-VL error on image={q.get('question_id')}: {exc}")
                        result = {
                            "adapter": "rf100-vl",
                            "task": "object_detection",
                            "question_id": q.get("question_id"),
                            "segment_id": q.get("segment_id"),
                            "image_id": q.get("image_id"),
                            "dataset_name": q.get("dataset_name"),
                            "split": q.get("split"),
                            "question": q.get("question"),
                            "category": q.get("category"),
                            "references": q.get("references") or [],
                            "ground_truth": q.get("ground_truth") or [],
                            "detections": [],
                            "error": str(exc),
                            "judge_score": 0.0,
                            "judge_correct": False,
                            "prediction": "",
                            "raw_response": "",
                            "reasoning_trace": "",
                            "latency_seconds": None,
                            "images": q.get("images") or [],
                            "model": model,
                            "completed_epoch": time.time(),
                        }
                    snap = _benchmark_append_result(run_id, result)
                    results_collected += 1
                    now_flush = time.time()
                    if (
                        results_collected % flush_every == 0
                        or results_collected == len(questions)
                        or now_flush - last_flush >= 60
                    ):
                        summary = _benchmark_summary_for_snap(snap)
                        snap = _benchmark_update(run_id, summary=summary)
                        _benchmark_snapshot_save(run_id, snap)
                        _benchmark_write_durable_artifacts(run_id, snap, final=False)
                        last_flush = now_flush
                submit_more()
        final = _benchmark_update(
            run_id,
            status="complete",
            finished_epoch=time.time(),
            wall_seconds=time.time() - started,
            summary=_benchmark_summary_for_snap(benchmark_get(run_id) or {}),
        )
        _benchmark_snapshot_save(run_id, final)
        artifacts = _benchmark_write_durable_artifacts(run_id, final, final=True)
        final = _benchmark_update(run_id, artifacts=artifacts)
        _benchmark_snapshot_save(run_id, final)
        try:
            summary_obj = final.get("summary") or {}
            _benchmark_history_append({
                "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "epoch": time.time(),
                "adapter": "rf100-vl",
                "model": final.get("model") or "",
                "dataset": final.get("dataset") or "",
                "judge": "coco-ap",
                "sample_size": final.get("sample_size"),
                "concurrency": final.get("concurrency"),
                "seed": final.get("seed"),
                "accuracy": summary_obj.get("overall_accuracy"),
                "map": summary_obj.get("map"),
                "total_predictions": summary_obj.get("total_predictions"),
                "wall_seconds": final.get("wall_seconds"),
                "report_url": f"/benchmark/report/{run_id}",
            })
        except Exception as exc:
            log(f"[benchmark {run_id}] history append failed: {exc}")
        log(f"[benchmark {run_id}] RF100-VL complete in {final['wall_seconds']:.1f}s")
    except Exception as exc:
        log(f"[benchmark {run_id}] RF100-VL FAILED: {exc}")
        final = _benchmark_update(run_id, status="error", error=str(exc), finished_epoch=time.time())
        try:
            _benchmark_snapshot_save(run_id, final)
            artifacts = _benchmark_write_durable_artifacts(run_id, final, final=True)
            _benchmark_update(run_id, artifacts=artifacts)
        except Exception:
            pass


def run_benchmark_worker(
    run_id: str,
    dataset_id: str,
    judge_id: str,
    sample_size: int,
    concurrency: int,
    seed: int,
    source_config: Optional[Dict[str, Any]],
    judge_mode: str,
    selected_model: Optional[str],
) -> None:
    model_config = None
    try:
        model_config = _benchmark_model_config(selected_model)
    except Exception as exc:
        _benchmark_update(run_id, status="error", error=f"model endpoint not ready: {exc}", finished_epoch=time.time())
        return
    adapter = _benchmark_adapter_for_dataset(dataset_id)
    if adapter == "rf100-vl":
        run_rf100vl_benchmark(run_id, dataset_id, sample_size, concurrency, seed, source_config, model_config)
        return
    run_lingoqa_benchmark(
        run_id,
        dataset_id,
        judge_id,
        sample_size,
        concurrency,
        seed,
        source_config,
        judge_mode=judge_mode,
        model_config=model_config,
    )


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


def benchmark_render_rf100_report(run_id: str, snap: Dict[str, Any]) -> str:
    summary = snap.get("summary") or _rf100_summarize(snap)
    results = [r for r in (snap.get("results") or []) if not r.get("error")]
    errors = [r for r in (snap.get("results") or []) if r.get("error")]
    per_group = sorted(summary.get("per_category") or [], key=lambda c: c.get("accuracy") or 0, reverse=True)
    strong = per_group[:10]
    weak = list(reversed(per_group[-10:])) if per_group else []
    examples_good = [r for r in results if r.get("judge_correct")][:4]
    examples_miss = [r for r in results if not r.get("judge_correct")][:6]

    def _esc(s: Any) -> str:
        return html.escape(str(s or ""))

    def pct(v: Any) -> str:
        return percent_value(v) or "n/a"

    group_rows = "".join(
        f"<tr><td>{_esc(c.get('category'))}</td><td>{c.get('total')}</td>"
        f"<td>{pct(c.get('ap'))}</td><td>{pct(c.get('ap50'))}</td></tr>"
        for c in per_group[:100]
    )
    example_rows = "".join(
        f"<tr><td>{_esc(r.get('question_id'))}</td><td>{_esc(r.get('dataset_name'))}</td>"
        f"<td>{_esc(len(r.get('ground_truth') or []))}</td><td>{_esc(len(r.get('detections') or []))}</td>"
        f"<td>{pct(r.get('judge_score'))}</td><td>{_esc(r.get('prediction'))}</td></tr>"
        for r in (examples_good + examples_miss)
    )
    weak_rows = "".join(
        f"<li>{_esc(c.get('category'))}: AP {pct(c.get('ap'))}, AP50 {pct(c.get('ap50'))}, n={c.get('total')}</li>"
        for c in weak
    )
    strong_rows = "".join(
        f"<li>{_esc(c.get('category'))}: AP {pct(c.get('ap'))}, AP50 {pct(c.get('ap50'))}, n={c.get('total')}</li>"
        for c in strong
    )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>RF100-VL Benchmark Report - {_esc(run_id)}</title>
<style>
:root{{--green:#76B900;--dark:#1A1A1A;--line:#d8dee8;--soft:#F0F5E8;--bad:#b91c1c;}}
body{{margin:0;font-family:Inter,system-ui,sans-serif;color:var(--dark);line-height:1.5;background:#fff;}}
header{{background:var(--dark);color:#fff;padding:34px 42px;border-left:8px solid var(--green);}}
h1{{margin:0 0 8px;font-size:38px;}} main{{max-width:1180px;margin:0 auto;padding:32px;}}
section{{margin:0 0 34px;}} h2{{font-size:18px;text-transform:uppercase;letter-spacing:.04em;border-bottom:2px solid var(--green);padding-bottom:6px;}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}} .tile{{border:1px solid var(--line);border-left:5px solid var(--green);border-radius:8px;padding:16px;background:#fff;}}
.tile .v{{font-size:30px;font-weight:800;color:var(--green);}} .tile .l{{font-size:12px;color:#64748b;text-transform:uppercase;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}} th{{background:var(--dark);color:#fff;text-align:left;padding:9px;}} td{{border-bottom:1px solid var(--line);padding:8px;vertical-align:top;}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:18px;}} .note{{background:var(--soft);border-left:5px solid var(--green);padding:14px 18px;border-radius:0 8px 8px 0;}}
.bad{{color:var(--bad);font-weight:700;}} code{{font-family:ui-monospace,Menlo,monospace;}}
@media(max-width:800px){{.tiles,.cols{{grid-template-columns:1fr;}}}}
</style></head><body>
<header><h1>RF100-VL Benchmark Report</h1>
<div>Run {_esc(run_id)} | {_esc(snap.get('model'))} | {_esc(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snap.get('started_epoch') or time.time())))}</div></header>
<main>
<section><h2>Leaderboard</h2><div class='tiles'>
<div class='tile'><div class='v'>{pct(summary.get('map'))}</div><div class='l'>COCO-style mAP</div></div>
<div class='tile'><div class='v'>{pct(summary.get('ap50'))}</div><div class='l'>AP50</div></div>
<div class='tile'><div class='v'>{_esc(summary.get('total_predictions') or 0)}</div><div class='l'>images scored</div></div>
<div class='tile'><div class='v'>{_esc(summary.get('total_predicted_boxes') or 0)}</div><div class='l'>predicted boxes</div></div>
</div></section>
<section><h2>Gate Status</h2><div class='note'>
Dataset groups loaded: {_esc(snap.get('rf100_group_count') or 'unknown')} / {_esc(snap.get('rf100_expected_groups') or RF100_VL_EXPECTED_GROUPS)}.
Errors: {_esc(len(errors))}. Artifacts: <code>/tmp/benchmark-runs/{_esc(run_id)}</code>.
</div></section>
<section><h2>Where Cosmos3 Looks Strongest</h2><div class='cols'><div><h3>Strong groups</h3><ul>{strong_rows or '<li>No scored groups yet.</li>'}</ul></div><div><h3>Weak groups</h3><ul>{weak_rows or '<li>No misses yet.</li>'}</ul></div></div></section>
<section><h2>Per-Group Leaderboard</h2><table><thead><tr><th>Group</th><th>Images</th><th>AP</th><th>AP50</th></tr></thead><tbody>{group_rows}</tbody></table></section>
<section><h2>Trace Examples</h2><table><thead><tr><th>Image</th><th>Dataset</th><th>GT boxes</th><th>Pred boxes</th><th>F1@0.50</th><th>Prediction summary</th></tr></thead><tbody>{example_rows}</tbody></table></section>
<section><h2>Reproducibility</h2><table><tbody>
<tr><td>Dataset</td><td>{_esc(snap.get('dataset'))}</td></tr>
<tr><td>Model</td><td>{_esc(snap.get('model'))}</td></tr>
<tr><td>Endpoint</td><td>{_esc(snap.get('base_url'))}</td></tr>
<tr><td>Trace</td><td><a href='/benchmark/trace/{_esc(run_id)}.jsonl'>trace.jsonl</a></td></tr>
<tr><td>Leaderboard JSON</td><td><a href='/benchmark/leaderboard/{_esc(run_id)}'>leaderboard.json</a></td></tr>
</tbody></table></section>
</main></body></html>"""


def benchmark_render_report(run_id: str) -> str:
    """Render the in-app HTML report (arxiv-2312.14115 12-section layout)."""
    snap = _benchmark_load_any(run_id) or {}
    if (snap.get("adapter") or _benchmark_adapter_for_dataset(snap.get("dataset") or "")) == "rf100-vl":
        return benchmark_render_rf100_report(run_id, snap)
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
.compare-panel { margin-top:12px; border:1px solid var(--line); border-radius:8px; padding:14px; background:#f8fafc; }
.compare-grid { display:grid; grid-template-columns:repeat(2, minmax(220px, 1fr)); gap:10px 16px; }
.compare-grid .wide { grid-column:1 / -1; }
.compare-models { min-height:150px; }
.compare-summary { margin:12px 0; }
.compare-summary-row { display:grid; grid-template-columns:minmax(130px, 1.4fr) minmax(110px, 1fr) minmax(100px, 1fr) 72px 100px minmax(140px, 1.4fr); gap:8px; padding:7px 0; border-bottom:1px solid var(--line); font-size:12px; align-items:start; }
.compare-key-note { color:var(--muted); font-size:11px; margin:4px 0 0; }
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
.bench-flow { display:grid; grid-template-columns:repeat(7, minmax(96px, 1fr)); gap:8px; margin:12px 0 18px; }
.bench-flow-step { border:1px solid var(--line); border-radius:8px; padding:9px 10px; background:#fff; font-size:12px; color:#64748b; min-height:54px; }
.bench-flow-step.active { border-color:#76B900; background:#F0F5E8; color:#1A1A1A; }
.bench-flow-step.blocked { border-color:#F59E0B; background:#FEF3C7; color:#78350F; }
.bench-analysis { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px; margin-bottom:18px; }
.bench-analysis-grid { display:grid; grid-template-columns:minmax(240px, 0.9fr) minmax(0, 1.1fr); gap:14px; }
.bench-blockers { border-left:4px solid #F59E0B; background:#FEF3C7; padding:10px 12px; border-radius:0 6px 6px 0; color:#78350F; font-size:13px; }
.bench-gate-panel { background:#fff; border:1px solid var(--line); border-left:4px solid #76B900; border-radius:8px; padding:16px; margin:0 0 18px; }
.bench-gate-head { display:flex; justify-content:space-between; align-items:flex-start; gap:14px; margin-bottom:12px; }
.bench-gate-head h3 { margin:0 0 4px; font-size:14px; color:#475569; text-transform:uppercase; letter-spacing:0.04em; }
.bench-gate-grid { display:grid; grid-template-columns:repeat(4, minmax(120px, 1fr)); gap:10px; margin:12px 0; }
.bench-gate-stat { border:1px solid var(--line); border-radius:6px; padding:10px 12px; background:#f8fafc; min-width:0; }
.bench-gate-stat .value { font-size:20px; font-weight:800; color:#76B900; white-space:nowrap; }
.bench-gate-stat .label { font-size:11px; color:#64748b; text-transform:uppercase; }
.bench-gate-note { border-left:4px solid #76B900; background:#F0F5E8; padding:10px 12px; border-radius:0 6px 6px 0; font-size:13px; color:#1A1A1A; }
.bench-gate-note.hold { border-left-color:#F59E0B; background:#FEF3C7; color:#78350F; }
.bench-preview-list { display:grid; gap:8px; max-height:260px; overflow:auto; }
.bench-preview-row { border:1px solid var(--line); border-radius:6px; padding:8px; background:#f8fafc; font-size:12px; }
.bench-scope-controls { display:grid; gap:6px; font-size:13px; }
.bench-scope-controls label { display:flex; align-items:center; gap:6px; margin:2px 0; color:#1A1A1A; }
.bench-scope-controls input[type=radio] { width:auto; }
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
  .params, .export-sections, .compare-grid { grid-template-columns:1fr; }
  .compare-grid .wide { grid-column:auto; }
  .compare-summary-row { grid-template-columns:1fr; gap:2px; }
  .bench-flow, .bench-analysis-grid, .bench-gate-grid { grid-template-columns:1fr; }
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
  <label>Reference video uploads</label>
  <input id="referenceVideos" type="file" accept="video/*,.mp4,.mov,.m4v,.avi,.webm,.mkv" multiple />
  <p class="hint">Uploads are appended to the current dataset selection so the same references can be reused for batch runs and endpoint comparisons.</p>
  <div class="actions">
    <button class="secondary" id="referenceUploadBtn">Add reference videos</button>
  </div>
  <p class="hint" id="referenceUploadStatus"></p>
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
  <h3>Loaded vs hosted endpoint comparison</h3>
  <div class="compare-panel">
    <p class="hint">Run the currently selected dataset rows or uploaded references against the loaded endpoint baseline and one or more hosted models. Spot mode holds the workflow constant; ablation mode repeats cases across prompt variants.</p>
    <div class="compare-grid">
      <div>
        <label>Runtime API key</label>
        <input id="compareApiKey" type="password" value="" autocomplete="off" spellcheck="false" />
        <p class="compare-key-note">Use a fresh runtime key from <a href="https://inference.nvidia.com/key-management" target="_blank" rel="noreferrer">key management</a>. Enter it once per page session; this masked field keeps it until you click Clear key, refresh, or close the page. It is never included in server state, logs, results, or exports.</p>
      </div>
      <div>
        <label>Comparison mode</label>
        <select id="compareMode"><option value="spot">Spot comparison</option><option value="ablation">Workflow ablation</option></select>
        <label class="toggle-row"><input id="compareIncludeLoaded" type="checkbox" checked /><span><strong>Include loaded endpoint baseline</strong></span></label>
      </div>
      <div class="wide">
        <label>Hosted models (multi-select)</label>
        <select id="compareModels" class="compare-models" multiple size="8"></select>
        <label>Exact catalog model ID (optional)</label>
        <input id="compareCustomModel" value="" autocomplete="off" spellcheck="false" placeholder="Paste an exact ID returned by live discovery" />
        <label>Custom model media strategy</label>
        <select id="compareCustomStrategy"><option value="auto">Catalog capability (required)</option><option value="sampled_frames">Explicit deterministic sampled frames</option><option value="native_video">Explicit native video_url</option></select>
        <p class="hint" id="compareCatalogStatus">Curated defaults are shown until live discovery runs.</p>
      </div>
      <div class="wide">
        <label>Ablation variants (optional; one per line as label :: replacement user prompt)</label>
        <textarea id="compareVariants" placeholder="Concise answer :: Answer in one sentence.&#10;Reasoned answer :: Think step-by-step, then give a final answer."></textarea>
        <p class="hint">Blank ablation variants automatically compare the current workflow with its reasoning scaffold removed. Spot mode ignores this field.</p>
      </div>
    </div>
    <div class="actions">
      <button class="secondary" id="compareClearKeyBtn" type="button">Clear key</button>
      <button class="secondary" id="compareDiscoverBtn">Discover models</button>
      <button id="compareRunBtn">Compare selected videos</button>
      <button class="secondary compareExportBtn" data-format="html">Report</button>
      <button class="secondary compareExportBtn" data-format="json">JSON</button>
      <button class="secondary compareExportBtn" data-format="csv">CSV</button>
    </div>
    <p class="hint" id="compareStatus">No comparison has run yet.</p>
    <div class="progress"><div id="compareBar"></div></div>
    <div id="compareSummary" class="compare-summary"></div>
    <div class="results"><table><thead><tr><th>Video</th><th>Source</th><th>Model</th><th>Media</th><th>Variant</th><th>Status</th><th>Latency</th><th>TTFT</th><th>Error / response</th></tr></thead><tbody id="compareRows"></tbody></table></div>
    <div class="export-links" id="compareExportLinks"></div>
  </div>
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
      <div class="bench-dropzone-row" style="display:flex; gap:8px; align-items:center;">
        <button id="benchRunBtn" class="bench-primary-btn">▶ Run Benchmark</button>
        <button id="benchAnalyzeBtn" class="bench-ghost-btn">Analyze Source</button>
        <button id="benchLoadPreviewBtn" class="bench-ghost-btn">Load Preview</button>
        <button id="benchAuditBtn" class="bench-ghost-btn" disabled title="Run all 5 judge methods (lingo_judge x4 + llm_as_judge) on the last completed run">🔬 Fairness Audit</button>
      </div>
      <div class="bench-dropzone-row bench-ghost-row">
        <button id="benchRerunLastBtn" class="bench-ghost-btn" disabled>↻ Re-run last</button>
        <button id="benchToggleAdvBtn" class="bench-ghost-btn" aria-expanded="false">⚙ Advanced</button>
        <button id="benchToggleHistBtn" class="bench-ghost-btn" aria-expanded="false">≡ Run history</button>
        <button id="benchGateReportBtn" class="bench-ghost-btn">Gate 1 report</button>
      </div>
      <p class="bench-dropzone-status hint" id="benchDropzoneStatus"></p>
    </div>

    <div class="bench-flow" id="benchFlow">
      <div class="bench-flow-step active" data-stage="Source"><strong>Source</strong><br/>Paste site, repo, paper, or dataset.</div>
      <div class="bench-flow-step" data-stage="Understand"><strong>Understand</strong><br/>Task, metric, readiness.</div>
      <div class="bench-flow-step" data-stage="Scope"><strong>Scope</strong><br/>Full, smoke, or selected groups.</div>
      <div class="bench-flow-step" data-stage="Preview"><strong>Preview</strong><br/>Prompt, media, ground truth.</div>
      <div class="bench-flow-step" data-stage="Run"><strong>Run</strong><br/>Dispatch selected model.</div>
      <div class="bench-flow-step" data-stage="Review"><strong>Review</strong><br/>Leaderboard and traces.</div>
      <div class="bench-flow-step" data-stage="Export"><strong>Export</strong><br/>HTML, PPTX, trace repo.</div>
    </div>

    <div id="benchGatePanel" class="bench-gate-panel" style="display:none;">
      <div class="bench-gate-head">
        <div>
          <h3>RF100-VL Gate 1 interim</h3>
          <div id="benchGateSummary" class="hint">Loading partial benchmark state...</div>
        </div>
        <a id="benchGateReportLink" class="bench-ghost-link" href="/benchmark/gate_report/latest" target="_blank">Open report</a>
      </div>
      <div id="benchGateStats" class="bench-gate-grid"></div>
      <div id="benchGateAirSupport" class="bench-gate-note"></div>
    </div>

    <div id="benchAnalysisPanel" class="bench-analysis" style="display:none;">
      <div class="bench-analysis-grid">
        <div>
          <h3 style="margin:0 0 8px; font-size:14px; color:#475569;">Source analysis</h3>
          <div class="kv" id="benchAnalysisKv"></div>
          <div id="benchReadinessBlockers" style="margin-top:10px;"></div>
        </div>
        <div>
          <h3 style="margin:0 0 8px; font-size:14px; color:#475569;">Dataset preview</h3>
          <div id="benchPreviewSummary" class="hint">Load preview to inspect prompts and ground truth.</div>
          <div id="benchPreviewRows" class="bench-preview-list"></div>
        </div>
      </div>
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
          <p class="hint">Brief gt-A/gt-B references favor short answers. Long &lt;think&gt; chains can push the judge past its 128-token window (protocol hardcodes <code>max_length=128</code> in <code>benchmark/judge.py</code> — more aggressive than DeBERTa's 512 architectural cap). Use "Both" to A/B the bias.</p>
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
          <label>Scope</label>
          <div class="bench-scope-controls" id="benchScopeControls">
            <label><input type="radio" name="benchScopeMode" value="full" checked /> Full benchmark gate</label>
            <label><input type="radio" name="benchScopeMode" value="smoke" /> Smoke subset</label>
            <label><input type="radio" name="benchScopeMode" value="custom" /> Custom sample size</label>
          </div>
          <label>Sample size: <span id="benchSampleSizeVal">1000</span></label>
          <input id="benchSampleSize" type="range" min="10" max="1000" step="10" value="1000" />
          <label style="margin-top:10px;">Include groups (optional, comma-separated)</label>
          <input id="benchIncludeGroups" value="" placeholder="leave blank for all groups" />
          <label style="margin-top:10px;">Exclude groups (optional, comma-separated)</label>
          <input id="benchExcludeGroups" value="" placeholder="leave blank for none" />
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
          <div id="benchMetricLabel" style="font-size:14px; color:#475569;">Lingo-Judge Accuracy</div>
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
        <a id="benchLeaderboardLink" class="export-link" href="#" target="_blank">Leaderboard JSON</a>
        <a id="benchTraceLink" class="export-link" href="#" target="_blank">Trace JSONL</a>
        <a id="benchPptxLink" class="export-link" href="#" target="_blank">Executive PPTX</a>
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
let comparisonCatalogSig = '';
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
function setBusy(message){ document.getElementById('progressText').textContent = message; document.getElementById('bar').style.width = '12%'; if(el('sideBar')){ el('sideBar').style.width='12%'; el('statusTitle').textContent='Starting'; el('statusPct').textContent='12%'; el('runtimeNotice').className='status-note'; el('runtimeNotice').textContent=message; } ['loadBtn','runBtn','smokeBtn','foBtn','fitBudgetBtn','paperBtn','promptImportBtn','reasoningToggle','referenceUploadBtn','compareRunBtn','compareDiscoverBtn'].forEach(id=>{ if(document.getElementById(id)) document.getElementById(id).disabled=true; }); }
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
function selectedComparisonModels(){ return [...(el('compareModels')?.selectedOptions||[])].map(option=>option.value).filter(Boolean); }
function comparisonVariantsPayload(){ if(el('compareMode')?.value!=='ablation') return []; return String(el('compareVariants')?.value||'').split(/\\r?\\n/).map(line=>line.trim()).filter(Boolean).slice(0,8).map((line,index)=>{ const marker=line.indexOf('::'); if(marker<0) return {label:`Variant ${index+1}`,user_prompt:line}; return {label:line.slice(0,marker).trim()||`Variant ${index+1}`,user_prompt:line.slice(marker+2).trim()}; }); }
function syncComparisonCatalog(){
 const select=el('compareModels'); if(!select) return;
 const catalog=state?.comparison?.catalog||[]; const sig=JSON.stringify(catalog.map(item=>[item.id,item.label,item.comparison_media_strategy,item.capability_source]));
 if(sig===comparisonCatalogSig) return;
 const prior=new Set(selectedComparisonModels());
 select.innerHTML=catalog.map(item=>{ const strategy=item.comparison_media_strategy||''; const capable=!!strategy; const capability=strategy==='native_video'?'native video_url':(strategy==='sampled_frames'?'deterministic sampled image frames':'media capability unknown'); const label=item.id?(item.label&&item.label!==item.id?`${item.label} — ${item.id}`:item.id):(item.label||'Catalog hint'); return `<option value="${esc(item.id||'')}" ${capable&&item.id?'':'disabled'}>${esc(label)} [${esc(capability)}]</option>`; }).join('');
 let chosen=[...prior].filter(id=>catalog.some(item=>item.id===id&&item.comparison_media_strategy));
 if(!chosen.length) chosen=catalog.filter(item=>item.id&&item.comparison_media_strategy).slice(0,4).map(item=>item.id);
 [...select.options].forEach(option=>{ option.selected=chosen.includes(option.value); });
 comparisonCatalogSig=sig;
}
function renderComparison(){
 const comparison=state?.comparison||{}; syncComparisonCatalog();
 const run=comparison.run||{}; const progress=run.progress||{}; const total=Number(progress.total||0); const done=Number(progress.done||0); const pct=total?Math.round(100*done/total):(comparison.running?5:(run.status==='complete'?100:0));
 el('compareBar').style.width=pct+'%';
 const catalogText=comparison.catalog_source==='dynamic'?'Live catalog loaded':'Curated catalog defaults';
 el('compareCatalogStatus').textContent=comparison.catalog_warning?`${catalogText}. ${comparison.catalog_warning}`:`${catalogText}. Native-video models keep video_url input. Known multimodal/image models receive deterministic sampled image_url frames. Unknown entries stay blocked unless you explicitly choose a custom media strategy.`;
 el('compareStatus').textContent=comparison.running?`Running ${done}/${total} cases (${Number(progress.errors||0)} errors). The credential is omitted from run state and reports.`:run.status==='complete'?`Complete: ${done}/${total} cases, ${Number(progress.errors||0)} errors in ${sec(run.wall_seconds)}.`:run.status==='error'?`Comparison failed: ${clipText(run.error||'unknown error',260)}`:'No comparison has run yet.';
 const groups=(run.summary||{}).groups||[];
 el('compareSummary').innerHTML=groups.length?groups.map(item=>{ const latency=item.latency_seconds||{}; const errors=(item.error_messages||[]).join(' | '); return `<div class="compare-summary-row"><div><strong>${esc(item.model)}</strong><br>${esc(item.source||'')}</div><div>${esc(item.media_strategy||'')}</div><div>${esc(item.variant||'')}</div><div>${esc(item.ok||0)}/${esc(item.total||0)} ok</div><div>${sec(latency.average)} avg</div><div>${esc(errors)}</div></div>`; }).join(''):'<p class="hint">Summary will group model, media strategy, variant, status, latency, and errors.</p>';
 el('compareRows').innerHTML=(comparison.results||[]).map(row=>{ const detail=row.error?`<span style="color:var(--bad)">${esc(row.error)}</span>`:esc(clipText(row.response||'',320)); return `<tr><td>${esc(row.video_name||'')}</td><td>${esc(row.source||'')}</td><td>${esc(row.model||'')}</td><td>${esc(row.media_strategy||'')}</td><td>${esc(row.variant||'')}</td><td>${esc(row.status||'')}</td><td class="metric">${sec(row.latency_seconds)}</td><td class="metric">${sec(row.ttft_seconds)}</td><td>${detail}</td></tr>`; }).join('');
 el('compareExportLinks').innerHTML=(comparison.exports||[]).slice().reverse().map(item=>`<a class="export-link" href="${esc(item.url)}" target="_blank">${esc(String(item.format||'').toUpperCase())} | ${esc(item.filename||'')}</a>`).join('');
 const busy=!!(state.running||state.loading_dataset||comparison.running); el('compareRunBtn').disabled=busy; el('compareDiscoverBtn').disabled=busy; el('referenceUploadBtn').disabled=busy; [...document.querySelectorAll('.compareExportBtn')].forEach(button=>button.disabled=!!comparison.running||!(comparison.results||[]).length);
}
async function uploadReferenceVideos(){
 const files=[...(el('referenceVideos')?.files||[])]; if(!files.length){ alert('Choose one or more reference videos first.'); return; }
 el('referenceUploadStatus').textContent=`Uploading 0/${files.length}...`; el('referenceUploadBtn').disabled=true;
 try{ let done=0; for(const file of files){ const response=await fetch('/api/reference-video',{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream','X-File-Name':encodeURIComponent(file.name)},body:file}); const result=await response.json(); if(!response.ok) throw new Error(result.error||response.statusText); done++; el('referenceUploadStatus').textContent=`Uploaded ${done}/${files.length}: ${file.name}`; } el('referenceVideos').value=''; await poll(); }
 catch(error){ el('referenceUploadStatus').textContent=`Upload failed: ${error.message}`; alert(error.message); await poll(); }
 finally{ el('referenceUploadBtn').disabled=false; }
}
async function discoverComparisonModels(){
 let key=el('compareApiKey').value.trim(); if(!key){ alert('Enter a runtime API key for model discovery.'); return; }
 el('compareCatalogStatus').textContent='Discovering available models...';
 const pending=api('/api/compare/models',{api_key:key}); key='';
 try{ await pending; await poll(); }catch(error){ alert(error.message); await poll(); }
}
function clearComparisonKey(){ el('compareApiKey').value=''; }
async function runComparison(){
 let key=el('compareApiKey').value.trim(); if(!key){ alert('Enter a runtime API key for this comparison run.'); return; }
 const custom=String(el('compareCustomModel')?.value||'').trim(); const models=[...new Set([...selectedComparisonModels(),...(custom?[custom]:[])])]; if(!models.length){ alert('Select at least one hosted model with a catalog or explicit media strategy.'); return; }
 const customStrategy=String(el('compareCustomStrategy')?.value||'auto'); const model_strategies=custom&&customStrategy!=='auto'?{[custom]:customStrategy}:{};
 const body={api_key:key,ids:checkedIds(),models,model_strategies,mode:el('compareMode').value,variants:comparisonVariantsPayload(),include_loaded:el('compareIncludeLoaded').checked,concurrency:Number(el('concurrency').value),prompt_mode:promptMode(),system_prompt:el('systemPrompt').value,user_prompt:el('userPrompt').value,...params(),...reasoningPayload()};
 const pending=api('/api/compare/run',body); key=''; body.api_key='';
 try{ await pending; await poll(); }catch(error){ alert(error.message); await poll(); }
}
async function exportComparison(format){ try{ const item=await api('/api/compare/export',{format}); await poll(); window.open(item.url,'_blank'); }catch(error){ alert(error.message); await poll(); } }
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
 renderComparison(); document.getElementById('log').textContent = (state.logs||[]).join('\\n');
 const busy=!!(state.running||state.loading_dataset||(state.comparison||{}).running); document.getElementById('loadBtn').disabled = busy; document.getElementById('paperBtn').disabled = busy; document.getElementById('promptImportBtn').disabled = busy; document.getElementById('runBtn').disabled = busy || contextBlocked; document.getElementById('smokeBtn').disabled = busy || contextBlocked; document.getElementById('foBtn').disabled = busy; document.getElementById('reasoningToggle').disabled = busy; requestAnimationFrame(autosizePrompts); }
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
document.getElementById('referenceUploadBtn').onclick = uploadReferenceVideos;
document.getElementById('compareClearKeyBtn').onclick = clearComparisonKey;
document.getElementById('compareDiscoverBtn').onclick = discoverComparisonModels;
document.getElementById('compareRunBtn').onclick = runComparison;
document.querySelectorAll('.compareExportBtn').forEach(btn=>{ btn.onclick=()=>exportComparison(btn.dataset.format); });
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
  const isRf100 = (snap.adapter === 'rf100-vl') || String(snap.dataset||'').startsWith('rf100-vl');
  if(el('benchMetricLabel')) el('benchMetricLabel').textContent = isRf100 ? 'COCO-style mAP' : 'Lingo-Judge Accuracy';
  const acc = isRf100 ? (summary.map ?? summary.overall_accuracy) : summary.overall_accuracy;
  el('benchAccuracy').textContent = acc !== undefined && acc !== null ? (acc*100).toFixed(1)+'%' : '--';
  if(snap.status === 'complete'){ benchSetStage('Review'); }
  if(snap.status === 'error'){ benchSetStage('Run', true); }
  // Per-category chips
  const cats = summary.per_category || [];
  const chipsHtml = cats.map(c=>{
    const a = c.accuracy||0;
    const cls = a>=0.7 ? 'high' : (a<=0.3 ? 'low' : '');
    return `<div class="cat-chip ${cls}"><strong>${esc(c.category)}</strong> ${(a*100).toFixed(1)}% (n=${c.total})</div>`;
  }).join('');
  el('benchCats').innerHTML = chipsHtml || '<div class="hint" style="color:#94a3b8;">no completed predictions yet</div>';
  // Aggregate truncation callout — surface when a meaningful fraction of rows
  // would have been clipped by the judge's 128-token window (hardcoded
  // `max_length=128` in benchmark/judge.py; more aggressive than DeBERTa's
  // 512 architectural cap). >5% suggests a re-judge sprint with shortened
  // prompts is warranted.
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
        ? `<div class="trunc-summary-callout"><strong>⚠️ ${truncRows} / ${totalRows} (${pctStr}%) judge inputs truncated</strong> at ${summary.judge_max_tokens||128} tokens (protocol hardcodes <code>max_length=128</code> in <code>benchmark/judge.py</code>). Reasoning-style models with long &lt;think&gt; chains may be under-rated — consider a shortened-prompt re-judge sprint as follow-up.</div>`
        : `<div class="hint" style="color:#94a3b8; font-size:12px;">Judge truncation: ${truncRows} / ${totalRows} rows (${pctStr}%) exceeded ${summary.judge_max_tokens||128} tokens (protocol's hardcoded <code>max_length=128</code>).</div>`;
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
      ? `<span class="trunc-flag" title="Judge input (standard mode) was truncated at 128 tokens (the protocol's hardcoded max_length=128 in benchmark/judge.py — more aggressive than DeBERTa's 512 architectural cap). Final answer may not have reached the judge.">⚠️ trunc</span>`
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
    if(el('benchLeaderboardLink')) el('benchLeaderboardLink').href = '/benchmark/leaderboard/'+snap.run_id;
    if(el('benchTraceLink')) el('benchTraceLink').href = '/benchmark/trace/'+snap.run_id+'.jsonl';
    if(el('benchPptxLink')) el('benchPptxLink').href = '/benchmark/artifact/'+snap.run_id+'/artifacts/executive-summary.pptx';
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
  // Default 128: the official wayveai/LingoQA benchmark/judge.py hardcodes
  // `max_length=128` in its tokenizer call. NOT DeBERTa's 512 architectural cap.
  const maxTok = r.judge_max_tokens || 128;
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
  const protocolNote = `<div style="font-size:11px; color:#94a3b8; margin:4px 0 8px 0; font-style:italic;">Truncated at ${maxTok} tokens (the protocol's hardcoded <code>max_length=128</code> in <code>benchmark/judge.py</code>, more aggressive than DeBERTa's 512 architectural cap).</div>`;
  const stdBlock = `
    <div class="judge-breakdown">
      <h4>Judge breakdown — standard${r.judge_score_standard !== null && r.judge_score_standard !== undefined ? ` (score ${r.judge_score_standard.toFixed(3)} → ${r.judge_correct_standard?'<span style=\"color:#76B900;\">correct</span>':'<span style=\"color:#DC2626;\">miss</span>'})` : ''}</h4>
      <div style="font-size:12px; margin-bottom:8px; color:#475569;">judge sees full prediction (incl. &lt;think&gt;)${winIdx>=0?` · winning ref index: <strong>${winIdx}</strong>`:''}</div>
      ${protocolNote}
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
      ${protocolNote}
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
    const scope = benchScopePayload(sample_size);
    if(sample_size!==undefined){
      scope.mode = 'smoke';
      scope.sample_size = sample_size;
      scope.gate_run = false;
    }
    const body = {
      dataset: dsValue,
      judge: el('benchJudge').value,
      judge_mode: judge_mode,
      sample_size: scope.sample_size,
      concurrency: Number(el('benchConcurrency').value),
      seed: Number(el('benchSeed').value),
      source_config: source_config,
      scope: scope,
      selected_model: el('benchModelSelect') ? el('benchModelSelect').value : '',
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
let benchAnalysisCache = null;
let benchPreviewCache = null;

function benchSetStage(activeStage, blocked=false){
  document.querySelectorAll('.bench-flow-step').forEach(step=>{
    const isActive = step.dataset.stage === activeStage;
    step.classList.toggle('active', isActive && !blocked);
    step.classList.toggle('blocked', isActive && blocked);
  });
}

function benchSplitGroups(value){
  return String(value||'').split(',').map(s=>s.trim()).filter(Boolean);
}

function benchScopePayload(sampleOverride){
  const modeEl = document.querySelector('input[name="benchScopeMode"]:checked');
  const mode = modeEl ? modeEl.value : 'full';
  const sample = sampleOverride!==undefined ? sampleOverride : Number(el('benchSampleSize').value);
  return {
    mode: mode,
    sample_size: mode === 'full' ? 0 : (mode === 'smoke' ? 5 : sample),
    include_groups: benchSplitGroups(el('benchIncludeGroups')?.value),
    exclude_groups: benchSplitGroups(el('benchExcludeGroups')?.value),
    gate_run: mode === 'full',
    download: true,
    hf_repo: 'probicheaux/rf100-vl',
  };
}

function benchRenderAnalysis(analysis){
  benchAnalysisCache = analysis;
  const panel = el('benchAnalysisPanel');
  if(panel) panel.style.display = 'block';
  const summary = analysis.summary || {};
  const ready = analysis.readiness || {};
  const rows = [
    ['adapter', analysis.adapter || ''],
    ['task', summary.task || ''],
    ['metric', summary.metric || ''],
    ['dataset', summary.dataset || ''],
    ['groups', ready.expected_groups ? `${ready.groups||0}/${ready.expected_groups}` : String(ready.groups||'')],
    ['images', ready.images!=null ? String(ready.images) : ''],
    ['path', ready.path || ''],
  ];
  el('benchAnalysisKv').innerHTML = rows.filter(([_,v])=>v).map(([k,v])=>`<div>${esc(k)}</div><div>${esc(v)}</div>`).join('');
  const blockers = ready.blockers || [];
  if(blockers.length){
    el('benchReadinessBlockers').innerHTML = `<div class="bench-blockers"><strong>Readiness blockers</strong><ul>${blockers.map(b=>`<li>${esc(b)}</li>`).join('')}</ul></div>`;
    benchSetStage('Understand', true);
  } else {
    el('benchReadinessBlockers').innerHTML = '<div class="hint" style="color:#76B900;">Ready for preview and run.</div>';
    benchSetStage('Scope');
  }
}

function benchRenderPreview(data){
  benchPreviewCache = data;
  const total = data.total || 0;
  const groups = data.group_count || 0;
  const blockers = data.blockers || [];
  if(blockers.length){
    el('benchPreviewSummary').innerHTML = `<div class="bench-blockers"><strong>Preview blocked</strong><ul>${blockers.map(b=>`<li>${esc(b)}</li>`).join('')}</ul></div>`;
    el('benchPreviewRows').innerHTML = '';
    benchSetStage('Preview', true);
    return;
  }
  el('benchPreviewSummary').textContent = `${total} rows across ${groups} groups. Showing ${Math.min((data.preview||[]).length, 20)} preview rows.`;
  el('benchPreviewRows').innerHTML = (data.preview||[]).map(r=>{
    const gt = r.ground_truth_count ? `${r.ground_truth_count} GT boxes` : benchClipText((r.references||[''])[0]||'', 100);
    return `<div class="bench-preview-row"><strong>${esc(r.question_id||'')}</strong><br/>
      <span>${esc(r.dataset_name || r.category || '')}</span><br/>
      <span>${esc(benchClipText(r.question||'', 180))}</span><br/>
      <span style="color:#64748b;">${esc(gt)}</span></div>`;
  }).join('') || '<div class="hint">No preview rows available.</div>';
  benchSetStage('Preview');
}

function benchGateInt(v){
  const n = Number(v || 0);
  return Number.isFinite(n) ? Math.round(n).toLocaleString() : '0';
}

function benchGatePct(v){
  const n = Number(v);
  return Number.isFinite(n) ? (n * 100).toFixed(1) + '%' : '--';
}

function benchGateEta(v){
  const n = Number(v);
  if(!Number.isFinite(n) || n < 0) return 'n/a';
  if(n < 1) return Math.round(n * 60) + 'm';
  return n.toFixed(1) + 'h';
}

function benchRenderGateReport(payload){
  const panel = el('benchGatePanel');
  if(!panel) return;
  panel.style.display = 'block';
  const ag = payload.aggregates || {};
  const c3 = ag.c3 || {};
  const cr2 = ag.cr2_productive || {};
  const waste = ag.cr2_waste || {};
  const air = payload.air_support || {};
  const generated = payload.generated_epoch ? new Date(Number(payload.generated_epoch) * 1000).toLocaleString() : '';
  el('benchGateSummary').textContent = generated
    ? `Partial RF100-VL status generated ${generated}. Quality comparisons remain restricted to matched trace rows.`
    : 'Partial RF100-VL status. Quality comparisons remain restricted to matched trace rows.';
  el('benchGateStats').innerHTML = [
    ['C3 coverage', benchGatePct(c3.pct), `${benchGateInt(c3.done)} / ${benchGateInt(c3.total)} images`],
    ['C3 ETA', benchGateEta(c3.eta_h), `${benchGateInt(c3.rate_h)} images/hour`],
    ['CR2 useful coverage', benchGatePct(cr2.pct), `${benchGateInt(cr2.done)} / ${benchGateInt(cr2.total)} images`],
    ['CR2 stuck/waste lanes', benchGateInt(waste.lanes), `${benchGateInt(waste.errors)} errors observed`],
  ].map(([label,value,sub])=>`<div class="bench-gate-stat"><div class="value">${esc(value)}</div><div class="label">${esc(label)}</div><div class="hint">${esc(sub)}</div></div>`).join('');
  const decision = air.decision || 'hold';
  const recs = (air.recommendations || []).slice(0, 3).map(r=>`<li>${esc(r)}</li>`).join('');
  const note = el('benchGateAirSupport');
  note.classList.toggle('hold', decision === 'hold');
  note.innerHTML = `<strong>Air support decision: ${esc(decision)}</strong><ul>${recs}</ul>`;
  const link = el('benchGateReportLink');
  if(link) link.href = '/benchmark/gate_report/latest';
}

async function benchLoadGateReport(){
  try{
    const r = await fetch('/benchmark/gate_report/latest.json');
    const payload = await r.json();
    if(payload.error){ throw new Error(payload.error); }
    benchRenderGateReport(payload);
    benchSetStage('Review');
  }catch(e){
    const panel = el('benchGatePanel');
    if(panel) panel.style.display = 'block';
    el('benchGateSummary').textContent = 'Gate report unavailable: '+e.message;
    el('benchGateStats').innerHTML = '';
    const note = el('benchGateAirSupport');
    note.classList.add('hold');
    note.textContent = 'No Gate 1 status snapshot is available on this host.';
    benchSetStage('Review', true);
  }
}

async function benchAnalyzeSource(){
  const url = (el('benchUrlInput').value||'').trim() || 'lingoqa';
  try{
    const resp = await api('/benchmark/analyze_source', {url});
    benchRenderAnalysis(resp);
    if(resp.dataset_config && resp.dataset_config.id){
      const ds = el('benchDataset');
      let opt = Array.from(ds.options).find(o=>o.value===resp.dataset_config.id);
      if(!opt){
        opt = document.createElement('option');
        opt.value = resp.dataset_config.id;
        ds.appendChild(opt);
      }
      opt.textContent = resp.dataset_config.name || resp.dataset_config.id;
      ds.value = resp.dataset_config.id;
    }
  }catch(e){
    el('benchDropzoneStatus').textContent = 'Analyze failed: '+e.message;
    benchSetStage('Understand', true);
  }
}

async function benchLoadPreview(){
  try{
    const body = {
      dataset: el('benchDataset').value,
      source_config: benchUrlResolution && benchUrlResolution.dataset_config ? benchUrlResolution.dataset_config : null,
      scope: benchScopePayload(),
      preview_size: 5,
    };
    if(body.source_config){ body.source_config = {...body.source_config, ...body.scope}; }
    const data = await api('/benchmark/load', body);
    benchRenderPreview(data);
  }catch(e){
    el('benchPreviewSummary').textContent = 'Preview failed: '+e.message;
    benchSetStage('Preview', true);
  }
}

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
    await benchAnalyzeSource();
  }catch(e){
    benchSetChip('? resolve failed', 'unknown');
    el('benchDropzoneStatus').textContent = 'Resolve failed: '+e.message;
    benchSetStage('Source', true);
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
  const auditBtn = el('benchAuditBtn');
  if(!benchHistoryCache.length){
    tile.style.display = 'none';
    if(rerunLast) rerunLast.disabled = true;
    if(auditBtn) auditBtn.disabled = true;
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
  if(auditBtn){ auditBtn.disabled = false; auditBtn.dataset.runId = last.run_id; }
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
  if(!benchAnalysisCache){
    await benchAnalyzeSource();
  }
  // Sync the selected model into the hidden auto-detected field so existing
  // /benchmark/run flow can dispatch the selected local or discovered endpoint.
  benchSetStage('Run');
  benchStart();
};
const benchAnalyzeBtn = document.getElementById('benchAnalyzeBtn');
if(benchAnalyzeBtn){ benchAnalyzeBtn.onclick = benchAnalyzeSource; }
const benchLoadPreviewBtn = document.getElementById('benchLoadPreviewBtn');
if(benchLoadPreviewBtn){ benchLoadPreviewBtn.onclick = benchLoadPreview; }
const benchRerunLastBtn = document.getElementById('benchRerunLastBtn');
if(benchRerunLastBtn){
  benchRerunLastBtn.onclick = ()=>{
    if(!benchHistoryCache.length){ alert('No prior runs to re-run.'); return; }
    const last = benchHistoryCache[benchHistoryCache.length - 1];
    benchRerunById(last.run_id);
  };
}
const benchAuditBtn = document.getElementById('benchAuditBtn');
if(benchAuditBtn){
  benchAuditBtn.onclick = async ()=>{
    const runId = benchAuditBtn.dataset.runId
      || (benchHistoryCache.length ? benchHistoryCache[benchHistoryCache.length-1].run_id : '');
    if(!runId){ alert('No completed run to audit.'); return; }
    benchAuditBtn.disabled = true;
    const origText = benchAuditBtn.textContent;
    benchAuditBtn.textContent = '🔬 Auditing (5 methods)...';
    try{
      const resp = await fetch('/benchmark/audit', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({run_id: runId}),
      });
      const j = await resp.json();
      if(j.error){ alert('Audit failed: '+j.error); return; }
      // Minimal modal — open the audit JSON in a new tab so the reader can
      // inspect the full matrix. The 6x5 modal can be a later polish.
      const w = window.open('/benchmark/audit/'+encodeURIComponent(runId), '_blank');
      if(!w){ alert('Audit complete: '+JSON.stringify(j.methods, null, 2)); }
    }catch(e){ alert('Audit failed: '+e.message); }
    finally{
      benchAuditBtn.disabled = false;
      benchAuditBtn.textContent = origText;
    }
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
const benchGateReportBtn = document.getElementById('benchGateReportBtn');
if(benchGateReportBtn){ benchGateReportBtn.onclick = benchLoadGateReport; }
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
            rf_profile = _rf100_discover(download=False)
            out_datasets.append({
                "id": "rf100-vl",
                "name": "RF100-VL (Roboflow 100 Vision Language)",
                "rows": rf_profile.get("image_count") or None,
                "groups": rf_profile.get("group_count") or 0,
                "expected_groups": rf_profile.get("expected_groups"),
                "ready": bool(rf_profile.get("ready")),
                "full_gate_ready": bool(rf_profile.get("full_gate_ready")),
                "path": rf_profile.get("path"),
                "blockers": rf_profile.get("blockers") or [],
            })
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
            }, {
                "id": "coco-ap",
                "name": "COCO-style AP evaluator (RF100-VL)",
                "loaded": True,
                "error": None,
            }])
        elif self.path == "/benchmark/judge_protocol_note":
            # Lazy-fetch documentation: explains the 128 vs 512 distinction so
            # the UI can surface it without inlining the prose at page load.
            self.send_json({
                "max_length": LINGO_JUDGE_MAX_TOKENS,
                "deberta_architectural_cap": 512,
                "source_file": "https://github.com/wayveai/LingoQA/blob/main/benchmark/judge.py",
                "source_line": (
                    "self.tokenizer(texts, return_tensors='pt', padding=True, "
                    "truncation=True, max_length=128)"
                ),
                "summary": (
                    "The official wayveai/LingoQA `benchmark/judge.py` hardcodes "
                    "`max_length=128` in its tokenizer call. The protocol's "
                    "truncation ceiling is 128, NOT DeBERTa-v3-base's 512 "
                    "architectural cap. 128 is materially more aggressive: a "
                    "reasoning trace beyond ~80-90 tokens is silently clipped."
                ),
                "implication": (
                    "Reasoning-style predictions whose <think>...</think> chain "
                    "pushes total judge input past 128 tokens will be scored "
                    "against an answer-less prefix. Use judge_mode=answer-only "
                    "or judge_mode=both to strip <think> before scoring."
                ),
                "koi_finding": (
                    "Cosmos Benchmarks has no in-house LingoQA scorer; "
                    "vlmeval_lingoqa in vlmeval_mapping.json is a forward "
                    "declaration. VLMEvalKit has no lingoqa.py handler. Fix is "
                    "to strip <think> before passing prediction to the judge."
                ),
            })
        elif self.path == "/benchmark/model":
            try:
                model = benchmark_detect_model()
                self.send_json({"model": model, "base_url": benchmark_nim_base_url()})
            except Exception as exc:
                self.send_json({"model": None, "error": str(exc), "base_url": benchmark_nim_base_url()})
        elif self.path == "/benchmark/llm_judge_endpoint":
            try:
                cfg = _llm_judge_config()
                ready = _llm_judge_endpoint_ready(cfg["url"])
                self.send_json({"url": cfg["url"], "model": cfg["model"], "ready": ready})
            except Exception as exc:
                self.send_json({"url": None, "model": None, "ready": False, "error": str(exc)})
        elif self.path.startswith("/benchmark/audit/"):
            # GET /benchmark/audit/<run_id> — return cached audit.json if present.
            run_id = self.path.rsplit("/", 1)[-1].split("?")[0]
            audit_path = BENCHMARK_RUN_ROOT / run_id / "audit.json"
            if not audit_path.exists():
                self.send_error(404)
                return
            try:
                self.send_json(json.loads(audit_path.read_text(encoding="utf-8")))
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
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
        elif self.path == "/benchmark/gate_report/latest.json":
            try:
                self.send_json(_rf100_gate_interim_payload())
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif self.path == "/benchmark/gate_report/latest":
            try:
                self.send_text(benchmark_render_rf100_gate_interim(), "text/html")
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif self.path == "/benchmark/history":
            try:
                self.send_json(_benchmark_history_load())
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
        elif self.path.startswith("/benchmark/leaderboard/"):
            run_id = self.path.rsplit("/", 1)[-1].split("?")[0]
            snap = benchmark_get(run_id)
            if not snap:
                self.send_error(404)
                return
            self.send_json(_benchmark_leaderboard(snap))
        elif self.path.startswith("/benchmark/trace/") and self.path.endswith(".jsonl"):
            run_id = self.path.rsplit("/", 1)[-1][:-6]
            trace_path = BENCHMARK_RUN_ROOT / run_id / "trace.jsonl"
            if not trace_path.exists():
                snap = benchmark_get(run_id)
                if snap:
                    _benchmark_write_trace_jsonl(run_id, snap)
            if not trace_path.exists():
                self.send_error(404)
                return
            self.send_file(trace_path, "application/jsonl", download_name=trace_path.name)
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
            # Supports both durable run artifacts:
            #   /benchmark/artifact/<run_id>/artifacts/report.html
            #   /benchmark/artifact/<run_id>/artifacts/executive-summary.pptx
            # and legacy traceability images:
            #   /benchmark/artifact/<run_id>/<model_safe>/<question_id>/<filename>
            tail = urllib.parse.urlparse(self.path).path[len("/benchmark/artifact/"):]
            parts = [urllib.parse.unquote(p) for p in tail.split("/") if p]
            if len(parts) >= 2:
                run_id = parts[0]
                rel = Path(*parts[1:])
                artifact = BENCHMARK_RUN_ROOT / run_id / rel
                try:
                    resolved = artifact.resolve()
                    root_resolved = (BENCHMARK_RUN_ROOT / run_id).resolve()
                    if root_resolved not in resolved.parents and resolved != root_resolved:
                        self.send_error(404)
                        return
                except Exception:
                    self.send_error(404)
                    return
                if resolved.exists() and resolved.is_file():
                    ctype = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
                    self.send_file(resolved, ctype, download_name=resolved.name)
                    return
            if len(parts) != 4:
                self.send_error(404)
                return
            run_id, model_safe, qid, fname = parts
            if fname not in ("frame_strip.png", "frame_gif.gif", "ui_card.png", "reasoning.png"):
                self.send_error(404)
                return
            safe_model = _safe_model_dir(model_safe)
            artifact = TRACEABILITY_ROOT / run_id / safe_model / qid / fname
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
            if self.path == "/api/reference-video":
                self.handle_reference_upload()
                return
            payload = self.read_json()
            if self.path == "/api/compare/models":
                key = validated_runtime_key(payload.pop("api_key", None))
                try:
                    self.send_json(discover_hosted_models(key))
                finally:
                    key = ""
            elif self.path == "/api/compare/run":
                snap = snapshot()
                comparison = snap.get("comparison") or {}
                if snap.get("running") or snap.get("loading_dataset") or comparison.get("running"):
                    raise RuntimeError("A batch, dataset load, or comparison is already running")
                key = validated_runtime_key(payload.pop("api_key", None))
                try:
                    validate_comparison_payload_secrets(payload, key)
                except Exception:
                    key = ""
                    raise
                strategy_overrides = payload.get("model_strategies")
                if not isinstance(strategy_overrides, dict):
                    strategy_overrides = {}
                models = normalized_comparison_models(payload.get("models") or [], strategy_overrides)
                media_strategies = {
                    model: comparison_strategy_for_model(model, strategy_overrides) or ""
                    for model in models
                }
                ids = payload.get("ids") or []
                system_prompt = str(payload.get("system_prompt") or DEFAULT_SYSTEM)
                user_prompt = str(payload.get("user_prompt") or DEFAULT_PROMPT)
                params = params_from_payload(payload)
                mode = "ablation" if str(payload.get("mode") or "spot").lower() == "ablation" else "spot"
                variants = normalized_comparison_variants(
                    mode,
                    payload.get("variants"),
                    system_prompt,
                    user_prompt,
                    params,
                )
                if key in json.dumps({"models": models, "media_strategies": media_strategies, "variants": variants}, default=str):
                    key = ""
                    raise ClientInputError("The runtime API key must not be included in model ids or workflow prompts")
                include_loaded = bool_param(payload.get("include_loaded", True))
                videos = selected_videos_for_ids(ids)
                target_count = len(models) + (1 if include_loaded else 0)
                total_cases = len(videos) * target_count * len(variants)
                if not videos:
                    key = ""
                    raise ClientInputError("Load or upload at least one video before comparing endpoints")
                if total_cases > COMPARISON_MAX_CASES:
                    key = ""
                    raise ClientInputError(
                        f"Comparison expands to {total_cases} cases; reduce videos, models, or variants to {COMPARISON_MAX_CASES} or fewer"
                    )
                thread = threading.Thread(
                    target=run_hosted_comparison,
                    args=(
                        ids,
                        models,
                        media_strategies,
                        key,
                        mode,
                        variants,
                        max(1, min(16, int(payload.get("concurrency") or 4))),
                        include_loaded,
                    ),
                    daemon=True,
                )
                thread.start()
                key = ""
                self.send_json({
                    "ok": True,
                    "status": "running",
                    "mode": mode,
                    "video_count": len(videos),
                    "target_count": target_count,
                    "variant_count": len(variants),
                    "total_cases": total_cases,
                })
            elif self.path == "/api/compare/export":
                fmt = str(payload.get("format") or "html")
                path = create_comparison_export(fmt)
                item = {
                    "ok": True,
                    "format": path.suffix.lstrip(".").lower(),
                    "filename": path.name,
                    "url": "/api/export/" + urllib.parse.quote(path.name),
                    "bytes": path.stat().st_size,
                }
                with STATE_LOCK:
                    comparison = dict(STATE.get("comparison") or {})
                    comparison["exports"] = (comparison.get("exports") or [])[-9:] + [item]
                    STATE["comparison"] = comparison
                self.send_json(item)
            elif self.path == "/api/load":
                snap = snapshot()
                if snap.get("running") or snap.get("loading_dataset") or (snap.get("comparison") or {}).get("running"):
                    raise RuntimeError("A batch, dataset load, or comparison is already running")
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
                    if snap.get("running") or snap.get("loading_dataset") or (snap.get("comparison") or {}).get("running"):
                        raise RuntimeError("A batch, dataset load, or comparison is already running")
                    thread = threading.Thread(target=import_paper_worker, args=(source, max_videos, load_now), daemon=True)
                    thread.start()
                    self.send_json({"ok": True, "status": "importing", "source": source, "max_videos": max_videos})
                else:
                    discovery = import_paper_source(source, max_videos, load_now)
                    self.send_json(discovery)
            elif self.path == "/api/prompts/import":
                snap = snapshot()
                if snap.get("running") or snap.get("loading_dataset") or (snap.get("comparison") or {}).get("running"):
                    raise RuntimeError("A batch, dataset load, comparison, or prompt import is already running")
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
                if snap.get("running") or snap.get("loading_dataset") or (snap.get("comparison") or {}).get("running"):
                    raise RuntimeError("A batch, dataset load, or comparison is already running")
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
                if snap.get("running") or snap.get("loading_dataset") or (snap.get("comparison") or {}).get("running"):
                    raise RuntimeError("A batch, dataset load, or comparison is already running")
                max_videos = int_payload(payload, "max_videos", 2)
                concurrency = int_payload(payload, "concurrency", 2)
                load_dataset(DEFAULT_DATASET, max_videos)
                ids = [v["id"] for v in snapshot()["videos"] if v.get("source") != "reference_upload"]
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
            elif self.path == "/benchmark/analyze_source":
                url = str(payload.get("url") or payload.get("source") or "")
                if not url:
                    self.send_json({"error": "url required"}, status=400)
                    return
                self.send_json(_benchmark_analyze_source(url))
            elif self.path == "/benchmark/load":
                self.send_json(_benchmark_load_preview(payload))
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
                scope = payload.get("scope")
                if isinstance(scope, dict):
                    source_config = {**(source_config or {}), **scope}
                    if scope.get("mode") == "full":
                        sample_size = 0
                    elif scope.get("mode") == "smoke":
                        sample_size = int(scope.get("sample_size") or 5)
                    elif scope.get("sample_size") is not None:
                        sample_size = int(scope.get("sample_size") or sample_size)
                selected_model = str(payload.get("selected_model") or "").strip() or None
                run_id = f"bench-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
                adapter = _benchmark_adapter_for_dataset(dataset_id)
                _benchmark_update(
                    run_id,
                    status="queued",
                    adapter=adapter,
                    dataset=dataset_id,
                    judge=judge_id,
                    judge_mode=judge_mode,
                    sample_size=sample_size,
                    concurrency=concurrency,
                    seed=seed,
                    source_config=source_config,
                    selected_model=selected_model,
                    results=[],
                    progress={"done": 0, "total": 0, "errors": 0},
                )
                thread = threading.Thread(
                    target=run_benchmark_worker,
                    args=(run_id, dataset_id, judge_id, sample_size, concurrency, seed, source_config, judge_mode, selected_model),
                    daemon=True,
                )
                thread.start()
                self.send_json({
                    "ok": True,
                    "run_id": run_id,
                    "adapter": adapter,
                    "dataset": dataset_id,
                    "judge": judge_id,
                    "judge_mode": judge_mode,
                    "sample_size": sample_size,
                    "concurrency": concurrency,
                    "seed": seed,
                    "selected_model": selected_model,
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
                selected_model = str(prior_snap.get("selected_model") or hist.get("model") or "").strip() or None
                # Carry the judge_mode forward from the prior run when
                # available; default to standard otherwise.
                judge_mode = str(prior_snap.get("judge_mode") or hist.get("judge_mode") or "standard")
                if judge_mode not in ("standard", "answer-only", "both"):
                    judge_mode = "standard"
                adapter = _benchmark_adapter_for_dataset(dataset_id)
                new_run_id = f"bench-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
                _benchmark_update(
                    new_run_id,
                    status="queued",
                    adapter=adapter,
                    dataset=dataset_id,
                    judge=judge_id,
                    judge_mode=judge_mode,
                    sample_size=sample_size,
                    concurrency=concurrency,
                    seed=seed,
                    source_config=source_config,
                    selected_model=selected_model,
                    results=[],
                    progress={"done": 0, "total": 0, "errors": 0},
                    rerun_of=old_run_id,
                )
                thread = threading.Thread(
                    target=run_benchmark_worker,
                    args=(new_run_id, dataset_id, judge_id, sample_size, concurrency, seed, source_config, judge_mode, selected_model),
                    daemon=True,
                )
                thread.start()
                self.send_json({
                    "ok": True,
                    "run_id": new_run_id,
                    "rerun_of": old_run_id,
                    "adapter": adapter,
                    "dataset": dataset_id,
                    "judge": judge_id,
                    "judge_mode": judge_mode,
                    "sample_size": sample_size,
                    "concurrency": concurrency,
                    "seed": seed,
                    "selected_model": selected_model,
                })
            elif self.path == "/benchmark/rejudge":
                # Re-score an existing run through ONE alternate judge method.
                # Body: {"run_id": "...", "judge_method": "lingo_judge_max128|..."}.
                # Idempotent — caches output in rejudge_<method>.json on disk.
                target_run = str(payload.get("run_id") or "").strip()
                method = str(payload.get("judge_method") or "").strip()
                if not target_run:
                    self.send_json({"error": "run_id required"}, status=400)
                    return
                if method not in JUDGE_METHODS:
                    self.send_json({
                        "error": f"judge_method must be one of {list(JUDGE_METHODS)}",
                    }, status=400)
                    return
                try:
                    result = run_rejudge(target_run, method)
                    # Trim per_row in the wire response — full result is on disk.
                    light = dict(result)
                    full_rows = light.get("per_row") or []
                    light["per_row_count"] = len(full_rows)
                    light["per_row_sample"] = full_rows[:3]
                    light["per_row"] = None
                    light["artifact_path"] = str(
                        BENCHMARK_RUN_ROOT / target_run / f"rejudge_{method}.json"
                    )
                    self.send_json(light)
                except Exception as exc:
                    log(f"[rejudge] {target_run} {method} failed: {exc}")
                    self.send_json({"error": str(exc)}, status=400 if "not found" in str(exc) else 500)
            elif self.path == "/benchmark/audit":
                # Run ALL 5 judge methods on an existing run. Writes audit.json.
                # Body: {"run_id": "..."}.
                target_run = str(payload.get("run_id") or "").strip()
                if not target_run:
                    self.send_json({"error": "run_id required"}, status=400)
                    return
                try:
                    audit = run_audit(target_run)
                    # Trim per_row for the wire response — full audit on disk.
                    light = dict(audit)
                    full_rows = light.get("per_row") or []
                    light["per_row_count"] = len(full_rows)
                    light["per_row_sample"] = full_rows[:3]
                    light["per_row"] = None
                    light["artifact_path"] = str(BENCHMARK_RUN_ROOT / target_run / "audit.json")
                    self.send_json(light)
                except Exception as exc:
                    log(f"[audit] {target_run} failed: {exc}")
                    self.send_json({"error": str(exc)}, status=400 if "not found" in str(exc) else 500)
            else:
                self.send_error(404)
        except Exception as exc:
            error = safe_error(exc)
            log(f"API error: {error}")
            with STATE_LOCK:
                progress = dict(STATE.get("progress") or {})
                progress.update({
                    "updated_epoch": time.time(),
                    "last_event": "API error",
                    "last_error": error,
                })
                STATE["progress"] = progress
            self.send_json({"error": error}, status=400 if isinstance(exc, ClientInputError) else 500)

    def handle_reference_upload(self) -> None:
        snap = snapshot()
        if snap.get("running") or snap.get("loading_dataset") or (snap.get("comparison") or {}).get("running"):
            raise RuntimeError("Wait for the active batch or comparison before uploading a reference video")
        try:
            length = int(self.headers.get("content-length", "0") or "0")
        except ValueError as exc:
            raise ClientInputError("Reference upload has an invalid content length") from exc
        if length <= 0:
            raise ClientInputError("Reference upload is empty")
        if length > REFERENCE_VIDEO_MAX_BYTES:
            raise ClientInputError(
                f"Reference upload exceeds the {REFERENCE_VIDEO_MAX_BYTES // (1024 * 1024)} MiB limit"
            )
        raw_name = urllib.parse.unquote(self.headers.get("X-File-Name", "reference.mp4"))
        display_name = Path(raw_name).name or "reference.mp4"
        destination: Optional[Path] = None
        try:
            destination = store_reference_video(self.rfile, length, display_name)
            video = register_reference_video(destination, display_name)
        except Exception:
            try:
                if destination is not None:
                    destination.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        self.send_json({"ok": True, "video": video})

    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        if not length:
            return {}
        if length > 10 * 1024 * 1024:
            raise ClientInputError("JSON request body is too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(redact_sensitive(data), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str, download_name: Optional[str] = None) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
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
    ids = [v["id"] for v in snapshot()["videos"] if v.get("source") != "reference_upload"]
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
