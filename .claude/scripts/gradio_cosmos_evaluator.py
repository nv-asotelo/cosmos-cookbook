#!/usr/bin/env python3
"""Gradio frontend for the Cosmos Evaluator REST services."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - launch-time guard
    raise SystemExit("gradio is required. Install with: pip install gradio") from exc

try:
    import requests
except ImportError as exc:  # pragma: no cover - launch-time guard
    raise SystemExit("requests is required. Install with: pip install requests") from exc


PORT = int(os.environ.get("GRADIO_PORT", "7860"))
SHARE = os.environ.get("GRADIO_SHARE", "true").lower() not in {"0", "false", "no"}

NIM_URL = os.environ.get("COSMOS_EVALUATOR_NIM_URL", "http://localhost:8000").rstrip("/")
VLM_URL = os.environ.get("COSMOS_EVALUATOR_VLM_URL", "http://localhost:8083").rstrip("/")
CONTROL_URL = os.environ.get("COSMOS_EVALUATOR_CONTROL_URL", "http://localhost:8090").rstrip("/")
ATTRIBUTE_URL = os.environ.get("COSMOS_EVALUATOR_ATTRIBUTE_URL", "http://localhost:8086").rstrip("/")
HALLUCINATION_URL = os.environ.get("COSMOS_EVALUATOR_HALLUCINATION_URL", "http://localhost:8085").rstrip("/")
OBSTACLE_URL = os.environ.get("COSMOS_EVALUATOR_OBSTACLE_URL", "http://localhost:8082").rstrip("/")

DATA_DIR = Path(os.environ.get("COSMOS_EVALUATOR_DATA_DIR", "~/cosmos-evaluator/checks/sample_data/cosmos_public")).expanduser()
CONTAINER_DATA_PREFIX = os.environ.get("COSMOS_EVALUATOR_CONTAINER_DATA_PREFIX", "/data").rstrip("/")
REQUEST_TIMEOUT_S = float(os.environ.get("COSMOS_EVALUATOR_REQUEST_TIMEOUT_S", "7200"))
RUN_REQUEST_TIMEOUT_S = float(os.environ.get("COSMOS_EVALUATOR_RUN_TIMEOUT_S", "180"))
HTTP_TIMEOUT_S = float(os.environ.get("COSMOS_EVALUATOR_HTTP_TIMEOUT_S", "15"))

SAMPLE_BASENAME = os.environ.get(
    "COSMOS_EVALUATOR_SAMPLE_VIDEO",
    "01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000_0_Morning.30fps.mp4",
)
SAMPLE_API_PATH = f"{CONTAINER_DATA_PREFIX}/{SAMPLE_BASENAME}"

DEFAULT_PRESET = {
    "name": "environment",
    "weather": "Clear Sky",
    "time_of_day_illumination": "Morning",
    "region_geography": "Dense City Center",
    "road_surface_conditions": "Dry",
}

DEFAULT_ENDPOINT = "qwen3.5-397b-a17b"
DEFAULT_ENDPOINTS = [DEFAULT_ENDPOINT, "cosmos3-super-reasoner", "cosmos3-nano-reasoner"]
RUN_JOBS: Dict[str, Dict[str, Any]] = {}
RUN_JOBS_LOCK = threading.Lock()


CSS = """
:root {
  --nv-bg: #f5f7fb;
  --nv-panel: #ffffff;
  --nv-panel-2: #eef3f8;
  --nv-border: #d5dbe5;
  --nv-green: #76b900;
  --nv-text: #141820;
  --nv-muted: #596273;
  --nv-red: #b42318;
  --nv-yellow: #b54708;
}
body, .gradio-container {
  background: var(--nv-bg) !important;
  color: var(--nv-text) !important;
  color-scheme: light !important;
  font-family: Inter, Arial, sans-serif !important;
}
.gradio-container { max-width: 1500px !important; margin: 0 auto !important; }
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .tabitem,
.gradio-container .tabs,
.gradio-container .input-container,
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  background: #ffffff !important;
  color: var(--nv-text) !important;
  border-color: var(--nv-border) !important;
}
.gradio-container label,
.gradio-container p,
.gradio-container .prose,
.gradio-container .block-info,
.gradio-container .wrap {
  color: var(--nv-text) !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder { color: #7b8494 !important; }
.nv-appbar {
  height: 46px; border-bottom: 1px solid var(--nv-border); background: #ffffff;
  display: flex; align-items: center; gap: 24px; padding: 0 18px; color: #202938;
}
.nv-logo { font-weight: 800; color: #101828; }
.nv-logo span { color: var(--nv-green); }
.nv-nav { display: flex; gap: 18px; color: #667085; font-size: 13px; }
.nv-hero {
  border: 1px solid var(--nv-border); border-radius: 8px; padding: 20px 24px; margin-bottom: 14px;
  background: linear-gradient(90deg, #ffffff, #f6f9fc 74%, #edf7e7);
}
.nv-hero h1 { margin: 0; color: #101828; font-size: 30px; line-height: 1.15; letter-spacing: 0; }
.nv-hero p { color: #475467; max-width: 820px; margin: 8px 0 0; }
.nv-pill {
  display: inline-block; border: 1px solid var(--nv-border); border-radius: 999px; padding: 3px 8px;
  margin: 10px 6px 0 0; color: #344054; font-size: 12px; background: #ffffff;
}
.status-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 8px;
  margin: 0 0 10px;
}
.status-card { border: 1px solid var(--nv-border); border-radius: 8px; background: var(--nv-panel); padding: 10px; }
.status-title { color: #667085; font-size: 12px; margin-bottom: 4px; }
.status-ok { color: #2f6b00; font-weight: 700; }
.status-warn { color: #a15c07; font-weight: 700; }
.status-bad { color: #b42318; font-weight: 700; }
.api-note { color: #475467; font-size: 13px; line-height: 1.45; }
.warning-box {
  border: 1px solid #fedf89; background: #fffbeb; color: #7a2e0e; border-radius: 8px;
  padding: 10px 12px; margin: 6px 0;
}
.ok-box {
  border: 1px solid #b7e08b; background: #f3fbe9; color: #2f5f00; border-radius: 8px;
  padding: 10px 12px; margin: 6px 0;
}
.run-progress {
  border: 1px solid var(--nv-border); background: #ffffff; border-radius: 8px; padding: 10px 12px; margin: 8px 0;
}
.run-progress-track { height: 10px; background: #e7ebf0; border-radius: 999px; overflow: hidden; }
.run-progress-fill { height: 100%; background: linear-gradient(90deg, #76b900, #f97316); }
.run-progress-meta { display: grid; gap: 3px; margin-top: 8px; color: #344054; font-size: 13px; }
.run-progress-meta code { color: #7a2e0e; background: #fff7ed; }
.score-table-wrap {
  border: 1px solid var(--nv-border); border-radius: 8px; overflow: hidden; background: #ffffff; margin: 10px 0 14px;
}
.score-table-title { padding: 10px 12px; color: #344054; border-bottom: 1px solid var(--nv-border); font-weight: 700; }
.score-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 14px; line-height: 1.45; }
.score-table col.check { width: 20%; }
.score-table col.score { width: 8%; }
.score-table col.preset { width: 18%; }
.score-table col.explanation { width: 54%; }
.score-table th {
  text-align: left; background: #eef3f8; color: #1d2939; padding: 10px 12px; border-bottom: 1px solid var(--nv-border);
}
.score-table td {
  vertical-align: top; color: #1d2939; padding: 10px 12px; border-top: 1px solid #e4e7ec; overflow-wrap: anywhere;
  word-break: normal; white-space: normal;
}
.score-table td + td, .score-table th + th { border-left: 1px solid #e4e7ec; }
.score-table .score-cell { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; color: #2f6b00; }
.score-table .empty-cell { color: #667085; text-align: center; }
"""


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _set_job(job_id: str, **updates: Any) -> None:
    with RUN_JOBS_LOCK:
        job = RUN_JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = time.time()


def _snapshot_job(job_id: str) -> Dict[str, Any]:
    with RUN_JOBS_LOCK:
        return dict(RUN_JOBS.get(job_id, {}))


def _eta_label(job: Dict[str, Any]) -> str:
    if job.get("done") or int(job.get("progress") or 0) >= 100:
        return "complete"
    eta_seconds = job.get("eta_seconds")
    eta_started_at = job.get("eta_started_at")
    if not eta_seconds:
        return "estimating"
    elapsed = time.time() - float(eta_started_at or time.time())
    remaining = max(0, int(round(float(eta_seconds) - elapsed)))
    timeout = int(RUN_REQUEST_TIMEOUT_S)
    if job.get("api_call") == "POST /process/preset":
        return f"about {remaining}s, timeout {timeout}s"
    return f"about {remaining}s"


def _progress_html(job: Optional[Dict[str, Any]] = None) -> str:
    job = job or {}
    progress = max(0, min(100, int(job.get("progress") or 0)))
    status = html.escape(str(job.get("status") or "Ready."))
    api_call = html.escape(str(job.get("api_call") or "none"))
    eta = html.escape(_eta_label(job) if job else "not running")
    return (
        '<div class="run-progress">'
        '<div class="run-progress-track">'
        f'<div class="run-progress-fill" style="width: {progress}%"></div>'
        '</div>'
        '<div class="run-progress-meta">'
        f"<div><strong>{progress}%</strong> {status}</div>"
        f"<div>API call: <code>{api_call}</code></div>"
        f"<div>ETA: {eta}</div>"
        "</div></div>"
    )


def _endpoint_is_local_nim(endpoint: str, endpoints: List[Dict[str, Any]]) -> bool:
    meta = _endpoint_lookup(endpoints).get(endpoint, {})
    base_url = str(meta.get("base_url") or "")
    return bool(meta.get("nim_image")) or "cosmos3-nim" in base_url or "localhost:8000" in base_url


def _service_get(base_url: str, path: str, timeout: float = HTTP_TIMEOUT_S) -> Tuple[bool, Any]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        if resp.ok:
            return True, body
        return False, {"status_code": resp.status_code, "body": body}
    except Exception as exc:
        return False, {"error": str(exc)}


def _service_post(base_url: str, path: str, payload: Dict[str, Any], timeout: float = REQUEST_TIMEOUT_S) -> Tuple[bool, Any]:
    try:
        resp = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=timeout)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        if resp.ok:
            return True, body
        return False, {"status_code": resp.status_code, "body": body}
    except Exception as exc:
        return False, {"error": str(exc)}


def _unwrap_success(body: Any) -> Any:
    if isinstance(body, dict) and body.get("success") is True and "data" in body:
        return body["data"]
    return body


def _detect_nim_model(nim_url: str) -> Tuple[Optional[str], Any]:
    ok, body = _service_get(nim_url, "/v1/models")
    if not ok:
        return None, body
    try:
        model = body.get("data", [{}])[0].get("id")
    except Exception:
        model = None
    return model, body


def _runtime(control_url: str) -> Tuple[Dict[str, Any], Any]:
    ok, body = _service_get(control_url, "/runtime/vlm")
    if not ok:
        return {}, body
    data = _unwrap_success(body)
    return data if isinstance(data, dict) else {}, body


def _runtime_endpoints(control_url: str) -> Tuple[List[Dict[str, Any]], Any]:
    ok, body = _service_get(control_url, "/runtime/vlm/endpoints")
    if not ok:
        return [], body
    data = _unwrap_success(body)
    endpoints = data.get("available_endpoints", []) if isinstance(data, dict) else []
    return endpoints if isinstance(endpoints, list) else [], body


def _health_map(vlm_url: str, attribute_url: str, hallucination_url: str, obstacle_url: str) -> Dict[str, Any]:
    services = {
        "Evaluator service": (vlm_url, "/health"),
        "Attribute": (attribute_url, "/health"),
        "Hallucination": (hallucination_url, "/health"),
        "Obstacle": (obstacle_url, "/health"),
    }
    result: Dict[str, Any] = {}
    for name, (base, path) in services.items():
        ok, body = _service_get(base, path)
        data = _unwrap_success(body)
        status = data.get("status") if isinstance(data, dict) else None
        result[name] = {"ok": ok, "status": status or ("healthy" if ok else "error"), "body": body}
    return result


def _route_uses_local_nim(active_endpoint: Optional[str]) -> bool:
    return bool(active_endpoint and "cosmos3" in str(active_endpoint))


def _status_html(health: Dict[str, Any], runtime: Dict[str, Any], nim_model: Optional[str]) -> str:
    active_endpoint = str(runtime.get("active_endpoint") or "")
    active = active_endpoint or "unknown"
    model = str(nim_model or "unreachable")
    local_nim_title = "Local Cosmos3 NIM"
    local_nim_value = f"{model} (active route)" if _route_uses_local_nim(active_endpoint) else f"{model} (standby only)"
    cards = [
        ("Active evaluator target", active, bool(runtime)),
        (local_nim_title, local_nim_value, bool(nim_model)),
    ]
    for name, data in health.items():
        ok = bool(data.get("ok"))
        cards.append((name, str(data.get("status") or "unknown"), ok))
    rendered = []
    for title, value, ok in cards:
        cls = "status-ok" if ok else "status-bad"
        rendered.append(
            f'<div class="status-card"><div class="status-title">{html.escape(title)}</div>'
            f'<div class="{cls}">{html.escape(value)}</div></div>'
        )
    return '<div class="status-grid">' + "".join(rendered) + "</div>"


def _endpoint_lookup(endpoints: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("endpoint")): item for item in endpoints if item.get("endpoint")}


def _normalize_model(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _mismatch_message(endpoint: str, endpoints: List[Dict[str, Any]], nim_model: Optional[str]) -> str:
    meta = _endpoint_lookup(endpoints).get(endpoint, {})
    expected = str(meta.get("model") or "")
    base_url = str(meta.get("base_url") or "")
    is_local_nim = bool(meta.get("nim_image")) or "cosmos3-nim" in base_url or "localhost:8000" in base_url
    if not endpoint or not expected or not is_local_nim:
        return ""
    if not nim_model:
        return f"Selected endpoint {endpoint} expects {expected}, but the local NIM /v1/models endpoint is not reachable."
    if _normalize_model(expected) != _normalize_model(nim_model):
        return f"Selected endpoint {endpoint} expects {expected}, but the loaded NIM model is {nim_model}."
    return ""


def _safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")
    return stem or f"upload_{int(time.time())}.mp4"


def _uploaded_file_path(file_value: Any) -> Optional[str]:
    if file_value is None:
        return None
    if isinstance(file_value, dict):
        candidate = file_value.get("path") or file_value.get("name")
    else:
        candidate = getattr(file_value, "path", None) or getattr(file_value, "name", None) or str(file_value)
    if not candidate:
        return None
    path = Path(str(candidate))
    return str(path) if path.exists() else None


def _local_video_path_from_server_path(server_path: str) -> Optional[str]:
    path = (server_path or "").strip()
    if not path:
        return None
    direct = Path(path)
    if direct.exists():
        return str(direct)
    prefix = f"{CONTAINER_DATA_PREFIX}/"
    if path.startswith(prefix):
        candidate = DATA_DIR / path[len(prefix):]
        if candidate.exists():
            return str(candidate)
    return None


def _video_preview_value(source: str, upload: Any, server_path: str) -> Optional[str]:
    if source == "Uploaded video":
        return _uploaded_file_path(upload)
    if source == "Server path":
        return _local_video_path_from_server_path(server_path)
    return sample_video_value()


def _upload_video_change(
    upload: Any,
    server_path: str,
    weather: str,
    time_of_day: str,
    geography: str,
    road_surface: str,
) -> Tuple[Any, Optional[str], str]:
    source = "Uploaded video" if _uploaded_file_path(upload) else "Sample video"
    video_value = _video_preview_value(source, upload, server_path)
    payload_text = _preview_payload(source, upload, server_path, weather, time_of_day, geography, road_surface)
    return gr.update(value=source), video_value, payload_text


def _stage_upload(file_value: Any) -> str:
    if file_value is None:
        raise ValueError("Choose an uploaded video or switch the source to the sample/server path.")
    src = _uploaded_file_path(file_value)
    if not src:
        raise ValueError("Uploaded file is no longer available.")
    src_path = Path(src)
    if not src_path.exists():
        raise ValueError(f"Uploaded file is no longer available: {src}")
    if src_path.suffix.lower() != ".mp4":
        raise ValueError("Only MP4 uploads are supported for the evaluator preset API.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest_name = _safe_filename(src_path.name)
    dest = DATA_DIR / dest_name
    if dest.exists():
        dest = DATA_DIR / f"{src_path.stem}_{int(time.time())}{src_path.suffix}"
    shutil.copy2(src_path, dest)
    return f"{CONTAINER_DATA_PREFIX}/{dest.name}"


def _video_api_path(source: str, upload: Any, server_path: str) -> str:
    if source == "Uploaded video":
        return _stage_upload(upload)
    if source == "Server path":
        path = (server_path or "").strip()
        if not path:
            raise ValueError("Provide a server path such as /data/my_video.mp4.")
        return path
    return SAMPLE_API_PATH


def _build_payload(
    video_path: str,
    weather: str,
    time_of_day: str,
    geography: str,
    road_surface: str,
) -> Dict[str, Any]:
    return {
        "augmented_video_url": video_path,
        "preset_conditions": {
            "name": "environment",
            "weather": weather,
            "time_of_day_illumination": time_of_day,
            "region_geography": geography,
            "road_surface_conditions": road_surface,
        },
    }


def _preview_payload(
    source: str,
    upload: Any,
    server_path: str,
    weather: str,
    time_of_day: str,
    geography: str,
    road_surface: str,
) -> str:
    if source == "Uploaded video" and upload is not None:
        upload_path = _uploaded_file_path(upload)
        name = _safe_filename(Path(upload_path).name if upload_path else "upload.mp4")
        video_path = f"{CONTAINER_DATA_PREFIX}/{name}"
    elif source == "Server path":
        video_path = (server_path or "").strip() or f"{CONTAINER_DATA_PREFIX}/my_video.mp4"
    else:
        video_path = SAMPLE_API_PATH
    return _json(
        _build_payload(
            video_path,
            weather,
            time_of_day,
            geography,
            road_surface,
        )
    )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _curl_commands(
    nim_url: str,
    vlm_url: str,
    control_url: str,
    attribute_url: str,
    hallucination_url: str,
    obstacle_url: str,
    endpoint: str,
    payload_text: str,
) -> str:
    endpoint_payload = _shell_quote(json.dumps({"endpoint": endpoint}, indent=2))
    preset_payload = _shell_quote(payload_text)
    return "\n\n".join(
        [
            f"curl -fsS {nim_url.rstrip('/')}/v1/models",
            f"curl -fsS {vlm_url.rstrip('/')}/health",
            f"curl -fsS {control_url.rstrip('/')}/runtime/vlm",
            f"curl -fsS {control_url.rstrip('/')}/runtime/vlm/endpoints",
            "curl -fsS -X POST "
            f"{control_url.rstrip('/')}/runtime/vlm/switch "
            "-H 'Content-Type: application/json' "
            f"-d {endpoint_payload}",
            "curl -fsS -X POST "
            f"{vlm_url.rstrip('/')}/process/preset "
            "-H 'Content-Type: application/json' "
            f"-d {preset_payload}",
            f"curl -fsS {attribute_url.rstrip('/')}/health",
            f"curl -fsS {hallucination_url.rstrip('/')}/health",
            f"curl -fsS {obstacle_url.rstrip('/')}/health",
        ]
    )


def _response_summary(body: Any) -> Tuple[str, List[List[Any]]]:
    data = _unwrap_success(body)
    result = data.get("result", data) if isinstance(data, dict) else {}
    environment = result.get("environment", result) if isinstance(result, dict) else {}
    if not isinstance(environment, dict):
        return "No environment result was returned.", []
    overall = environment.get("overall_score")
    frames = environment.get("frames_used")
    seconds = environment.get("processing_time_s")
    model = environment.get("model")
    lines = []
    if overall is not None:
        lines.append(f"Overall score: **{overall}**")
    if model:
        lines.append(f"Model: `{model}`")
    if frames is not None:
        lines.append(f"Frames used: `{frames}`")
    if seconds is not None:
        lines.append(f"Processing time: `{seconds}s`")
    details = environment.get("scoring_details") or {}
    rows: List[List[Any]] = []
    if isinstance(details, dict):
        for key, item in details.items():
            if isinstance(item, dict):
                rows.append(
                    [
                        key,
                        item.get("score"),
                        item.get("preset"),
                        item.get("explanation"),
                    ]
                )
    return "\n\n".join(lines) if lines else "Evaluator returned a response.", rows


def _score_label(score: Any) -> str:
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        if float(score).is_integer():
            return str(int(score))
        return f"{float(score):.3g}"
    return str(score if score is not None else "")


def _details_html(rows: List[List[Any]]) -> str:
    body_rows = []
    for row in rows:
        check, score, preset, explanation = (list(row) + ["", "", "", ""])[:4]
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(check or ''))}</td>"
            f'<td class="score-cell">{html.escape(_score_label(score))}</td>'
            f"<td>{html.escape(str(preset or ''))}</td>"
            f"<td>{html.escape(str(explanation or ''))}</td>"
            "</tr>"
        )
    if not body_rows:
        body_rows.append('<tr><td class="empty-cell" colspan="4">Run the evaluator to see scoring details.</td></tr>')
    return (
        '<div class="score-table-wrap">'
        '<div class="score-table-title">Scoring details</div>'
        '<table class="score-table">'
        '<colgroup><col class="check"><col class="score"><col class="preset"><col class="explanation"></colgroup>'
        "<thead><tr><th>Check</th><th>Score</th><th>Preset</th><th>Explanation</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def refresh_state(
    nim_url: str,
    vlm_url: str,
    control_url: str,
    attribute_url: str,
    hallucination_url: str,
    obstacle_url: str,
) -> Tuple[str, Any, Any, Any, Any, str]:
    nim_model, nim_raw = _detect_nim_model(nim_url)
    runtime, runtime_raw = _runtime(control_url)
    endpoints, endpoints_raw = _runtime_endpoints(control_url)
    health = _health_map(vlm_url, attribute_url, hallucination_url, obstacle_url)
    choices = [item.get("endpoint") for item in endpoints if item.get("endpoint")] or DEFAULT_ENDPOINTS
    active = runtime.get("active_endpoint") if isinstance(runtime, dict) else None
    value = active if active in choices else choices[0]
    if active and not _route_uses_local_nim(str(active)):
        model_md = (
            f"Active evaluator target: `{active}` hosted endpoint. "
            f"The local Cosmos3 NIM is standby only: `{nim_model or 'unreachable'}`."
        )
    else:
        model_md = f"Active evaluator target uses local Cosmos3 NIM: `{nim_model or 'unreachable'}`."
    return (
        _status_html(health, runtime, nim_model),
        gr.update(choices=choices, value=value),
        runtime_raw,
        endpoints_raw,
        {"nim": nim_raw, "health": health},
        model_md,
    )


def switch_endpoint(control_url: str, endpoint: str) -> Tuple[str, Any]:
    ok, body = _service_post(control_url, "/runtime/vlm/switch", {"endpoint": endpoint}, timeout=HTTP_TIMEOUT_S)
    if ok:
        runtime, runtime_raw = _runtime(control_url)
        active = runtime.get("active_endpoint") if isinstance(runtime, dict) else None
        if active == endpoint:
            return f"Confirmed active endpoint: `{endpoint}`.", {"switch": body, "runtime": runtime_raw}
        return f"Switch requested for `{endpoint}`, but runtime reports `{active or 'unknown'}`.", {
            "switch": body,
            "runtime": runtime_raw,
        }
    return f"Endpoint switch failed for `{endpoint}`.", body


def run_evaluator(
    source: str,
    upload: Any,
    server_path: str,
    weather: str,
    time_of_day: str,
    geography: str,
    road_surface: str,
    endpoint: str,
    allow_mismatch: bool,
    use_edited_json: bool,
    payload_text: str,
    nim_url: str,
    vlm_url: str,
    control_url: str,
) -> Tuple[str, str, str, str, str]:
    try:
        started = time.time()
        endpoints, _ = _runtime_endpoints(control_url)
        nim_model, _ = _detect_nim_model(nim_url)
        mismatch = _mismatch_message(endpoint, endpoints, nim_model)
        if mismatch and not allow_mismatch:
            return (
                f'<div class="warning-box">{html.escape(mismatch)} Enable "Allow endpoint/model mismatch" in Advanced View to run anyway.</div>',
                _details_html([]),
                _json({"blocked": True, "reason": mismatch}),
                payload_text,
                "Blocked before API request.",
            )
            return

        if endpoint:
            switched, switch_body = _service_post(control_url, "/runtime/vlm/switch", {"endpoint": endpoint}, timeout=HTTP_TIMEOUT_S)
            if not switched:
                return (
                    '<div class="warning-box">Runtime switch failed.</div>',
                    _details_html([]),
                    _json(switch_body),
                    payload_text,
                    "Runtime switch failed before preset request.",
                )
                return

        if use_edited_json:
            payload = json.loads(payload_text)
        else:
            video_path = _video_api_path(source, upload, server_path)
            payload = _build_payload(
                video_path,
                weather,
                time_of_day,
                geography,
                road_surface,
            )
            payload_text = _json(payload)

        ok, body = _service_post(vlm_url, "/process/preset", payload, timeout=RUN_REQUEST_TIMEOUT_S)
        summary, rows = _response_summary(body)
        status = '<div class="ok-box">Preset evaluator completed.</div>' if ok else '<div class="warning-box">Preset evaluator returned an error.</div>'
        elapsed = round(time.time() - started, 2)
        return (
            status + "\n\n" + summary,
            _details_html(rows),
            _json(body),
            _json(payload),
            f"{'Success' if ok else 'Request failed'} in {elapsed}s.",
        )
    except Exception as exc:
        return (
            f'<div class="warning-box">{html.escape(str(exc))}</div>',
            _details_html([]),
            _json({"error": str(exc)}),
            payload_text,
            "Request failed before completion.",
        )


def _finish_job(
    job_id: str,
    result_md: str,
    details_html: str,
    raw_response: str,
    payload_text: str,
    run_status: str,
    ok: bool,
) -> None:
    _set_job(
        job_id,
        progress=100,
        status="Complete." if ok else "Finished with an error.",
        api_call="done",
        eta_seconds=0,
        done=True,
        delivered=False,
        result_md=result_md,
        details_html=details_html,
        raw_response=raw_response,
        payload_text=payload_text,
        run_status=run_status,
    )


def _track_job(job_id: str, progress: int, status: str, api_call: str, eta_seconds: float) -> None:
    _set_job(
        job_id,
        progress=progress,
        status=status,
        api_call=api_call,
        eta_seconds=eta_seconds,
        eta_started_at=time.time(),
        run_status=status,
    )


def _run_evaluator_job(
    job_id: str,
    source: str,
    upload: Any,
    server_path: str,
    weather: str,
    time_of_day: str,
    geography: str,
    road_surface: str,
    endpoint: str,
    allow_mismatch: bool,
    use_edited_json: bool,
    payload_text: str,
    nim_url: str,
    vlm_url: str,
    control_url: str,
) -> None:
    started = time.time()
    try:
        _track_job(job_id, 8, "Reading available evaluator targets.", "GET /runtime/vlm/endpoints", 5)
        endpoints, _ = _runtime_endpoints(control_url)
        nim_model = None
        if _endpoint_is_local_nim(endpoint, endpoints):
            _track_job(job_id, 18, "Checking local Cosmos3 NIM model.", "GET /v1/models", 8)
            nim_model, _ = _detect_nim_model(nim_url)

        mismatch = _mismatch_message(endpoint, endpoints, nim_model)
        if mismatch and not allow_mismatch:
            _finish_job(
                job_id,
                f'<div class="warning-box">{html.escape(mismatch)} Enable "Allow endpoint/model mismatch" in Advanced View to run anyway.</div>',
                _details_html([]),
                _json({"blocked": True, "reason": mismatch}),
                payload_text,
                "Blocked before API request.",
                False,
            )
            return

        if endpoint:
            _track_job(job_id, 28, f"Confirming active evaluator target {endpoint}.", "POST /runtime/vlm/switch", 6)
            switched, switch_body = _service_post(control_url, "/runtime/vlm/switch", {"endpoint": endpoint}, timeout=HTTP_TIMEOUT_S)
            if not switched:
                _finish_job(
                    job_id,
                    '<div class="warning-box">Runtime switch failed.</div>',
                    _details_html([]),
                    _json(switch_body),
                    payload_text,
                    "Runtime switch failed before preset request.",
                    False,
                )
                return

        _track_job(job_id, 42, "Preparing evaluator payload.", "build preset payload", 4)
        if use_edited_json:
            payload = json.loads(payload_text)
        else:
            video_path = _video_api_path(source, upload, server_path)
            payload = _build_payload(video_path, weather, time_of_day, geography, road_surface)
            payload_text = _json(payload)
            _set_job(job_id, payload_text=payload_text)

        preset_eta = 30 if source == "Sample video" else min(120, max(30, RUN_REQUEST_TIMEOUT_S / 2))
        _track_job(job_id, 60, "Posting preset request and waiting for evaluator response.", "POST /process/preset", preset_eta)
        ok, body = _service_post(vlm_url, "/process/preset", payload, timeout=RUN_REQUEST_TIMEOUT_S)

        _track_job(job_id, 92, "Rendering score summary and raw response.", "render response", 3)
        summary, rows = _response_summary(body)
        status = '<div class="ok-box">Preset evaluator completed.</div>' if ok else '<div class="warning-box">Preset evaluator returned an error.</div>'
        elapsed = round(time.time() - started, 2)
        _finish_job(
            job_id,
            status + "\n\n" + summary,
            _details_html(rows),
            _json(body),
            _json(payload),
            f"{'Success' if ok else 'Request failed'} in {elapsed}s.",
            ok,
        )
    except Exception as exc:
        _finish_job(
            job_id,
            f'<div class="warning-box">{html.escape(str(exc))}</div>',
            _details_html([]),
            _json({"error": str(exc)}),
            payload_text,
            "Request failed before completion.",
            False,
        )


def start_evaluator_job(
    source: str,
    upload: Any,
    server_path: str,
    weather: str,
    time_of_day: str,
    geography: str,
    road_surface: str,
    endpoint: str,
    allow_mismatch: bool,
    use_edited_json: bool,
    payload_text: str,
    nim_url: str,
    vlm_url: str,
    control_url: str,
) -> Tuple[str, str, str, str, str, str, str]:
    job_id = uuid.uuid4().hex
    if source == "Uploaded video" and not use_edited_json:
        staged_path = _stage_upload(upload)
        source = "Server path"
        upload = None
        server_path = staged_path
        payload_text = _preview_payload(source, upload, server_path, weather, time_of_day, geography, road_surface)
    _set_job(
        job_id,
        progress=3,
        status="Queued evaluator run.",
        api_call="start background job",
        eta_seconds=5,
        eta_started_at=time.time(),
        done=False,
        delivered=False,
        result_md='<div class="ok-box">Evaluator run started. Polling job status.</div>',
        details_html=_details_html([]),
        raw_response="{}",
        payload_text=payload_text,
        run_status="Starting evaluator job.",
    )
    thread = threading.Thread(
        target=_run_evaluator_job,
        args=(
            job_id,
            source,
            upload,
            server_path,
            weather,
            time_of_day,
            geography,
            road_surface,
            endpoint,
            allow_mismatch,
            use_edited_json,
            payload_text,
            nim_url,
            vlm_url,
            control_url,
        ),
        daemon=True,
    )
    thread.start()
    job = _snapshot_job(job_id)
    return (
        job_id,
        _progress_html(job),
        job["result_md"],
        job["details_html"],
        job["raw_response"],
        job["payload_text"],
        job["run_status"],
    )


def poll_evaluator_job(job_id: str) -> Tuple[Any, Any, Any, Any, Any, Any]:
    if not job_id:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    job = _snapshot_job(job_id)
    if not job:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    if job.get("done") and job.get("delivered"):
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    if job.get("done"):
        _set_job(job_id, delivered=True)
    return (
        _progress_html(job),
        job.get("result_md", gr.update()),
        job.get("details_html", _details_html([])),
        job.get("raw_response", "{}"),
        job.get("payload_text", gr.update()),
        job.get("run_status", gr.update()),
    )


def refresh_commands(
    nim_url: str,
    vlm_url: str,
    control_url: str,
    attribute_url: str,
    hallucination_url: str,
    obstacle_url: str,
    endpoint: str,
    payload_text: str,
) -> str:
    return _curl_commands(nim_url, vlm_url, control_url, attribute_url, hallucination_url, obstacle_url, endpoint, payload_text)


def fetch_config(vlm_url: str, control_url: str, attribute_url: str, hallucination_url: str, obstacle_url: str) -> Tuple[Any, Any, Any, Any, Any]:
    _, vlm_config = _service_get(vlm_url, "/config")
    _, runtime = _service_get(control_url, "/runtime/vlm")
    _, attr_config = _service_get(attribute_url, "/config")
    _, hall_config = _service_get(hallucination_url, "/config")
    _, obs_config = _service_get(obstacle_url, "/config")
    return vlm_config, runtime, attr_config, hall_config, obs_config


def sample_video_value() -> Optional[str]:
    local_sample = DATA_DIR / SAMPLE_BASENAME
    return str(local_sample) if local_sample.exists() else None


def build_app() -> gr.Blocks:
    initial_payload = _preview_payload(
        "Sample video",
        None,
        SAMPLE_API_PATH,
        DEFAULT_PRESET["weather"],
        DEFAULT_PRESET["time_of_day_illumination"],
        DEFAULT_PRESET["region_geography"],
        DEFAULT_PRESET["road_surface_conditions"],
    )
    initial_commands = _curl_commands(
        NIM_URL,
        VLM_URL,
        CONTROL_URL,
        ATTRIBUTE_URL,
        HALLUCINATION_URL,
        OBSTACLE_URL,
        DEFAULT_ENDPOINT,
        initial_payload,
    )

    with gr.Blocks(title="Cosmos Evaluator Workbench", css=CSS) as demo:
        gr.HTML(
            """
            <div class="nv-appbar">
              <div class="nv-logo"><span>NVIDIA</span></div>
              <div class="nv-nav"><span>Cosmos Evaluator</span><span>Runtime Switch</span><span>Preset API</span></div>
            </div>
            <div class="nv-hero">
              <h1>Cosmos Evaluator Workbench</h1>
              <p>Run the evaluator smoke example, stage your own MP4, choose the single active evaluator target, and inspect the REST commands that drive the service.</p>
              <span class="nv-pill">Basic evaluator</span><span class="nv-pill">Advanced API console</span><span class="nv-pill">Local NIM diagnostics</span>
            </div>
            """
        )

        status_html = gr.HTML(value="<div class=\"api-note\">Click Refresh status to query services.</div>")
        refresh_btn = gr.Button("Refresh status", variant="secondary")
        evaluator_job_id = gr.Textbox(label="Evaluator job id", value="", visible=False)
        evaluator_timer = gr.Timer(1.0, active=True)

        with gr.Tabs():
            with gr.TabItem("Basic View"):
                with gr.Row():
                    with gr.Column(scale=5):
                        video_preview = gr.Video(label="Video preview", value=sample_video_value(), height=260)
                        source = gr.Radio(
                            label="Video source",
                            choices=["Sample video", "Uploaded video", "Server path"],
                            value="Sample video",
                        )
                        upload = gr.File(label="Upload MP4", file_types=[".mp4"], file_count="single")
                        server_path = gr.Textbox(
                            label="Server path",
                            value=SAMPLE_API_PATH,
                            info="Path visible to evaluator containers, usually /data/<filename>.",
                        )
                    with gr.Column(scale=4):
                        endpoint = gr.Dropdown(label="Active evaluator target", choices=DEFAULT_ENDPOINTS, value=DEFAULT_ENDPOINT)
                        confirm_endpoint_btn = gr.Button("Confirm endpoint change", variant="secondary")
                        endpoint_confirm_status = gr.Markdown("Endpoint change has not been confirmed in this session.")
                        nim_model_md = gr.Markdown("Active route unknown. Local Cosmos3 NIM standby: `unknown`.")
                        gr.Markdown(
                            "Only the active evaluator target is used for a run. "
                            "The local Cosmos3 NIM is shown for diagnostics and standby switching."
                        )
                        weather = gr.Textbox(label="Weather", value=DEFAULT_PRESET["weather"])
                        time_of_day = gr.Textbox(
                            label="Time of day / illumination",
                            value=DEFAULT_PRESET["time_of_day_illumination"],
                        )
                        geography = gr.Textbox(label="Region / geography", value=DEFAULT_PRESET["region_geography"])
                        road_surface = gr.Textbox(label="Road surface", value=DEFAULT_PRESET["road_surface_conditions"])
                        run_btn = gr.Button("Run evaluator", variant="primary")
                        run_status = gr.Markdown("Ready.")
                        progress_status = gr.HTML(value=_progress_html())

                result_md = gr.Markdown("Run the evaluator to see the score summary.")
                details_table = gr.HTML(value=_details_html([]))
                raw_response = gr.Code(label="Raw evaluator response", language="json", lines=16)

            with gr.TabItem("Advanced View"):
                with gr.Row():
                    nim_url = gr.Textbox(label="Local Cosmos3 NIM URL", value=NIM_URL)
                    vlm_url = gr.Textbox(label="Evaluator service URL", value=VLM_URL)
                    control_url = gr.Textbox(label="Runtime/control URL", value=CONTROL_URL)
                with gr.Row():
                    attribute_url = gr.Textbox(label="Attribute URL", value=ATTRIBUTE_URL)
                    hallucination_url = gr.Textbox(label="Hallucination URL", value=HALLUCINATION_URL)
                    obstacle_url = gr.Textbox(label="Obstacle URL", value=OBSTACLE_URL)

                allow_mismatch = gr.Checkbox(
                    label="Allow endpoint/model mismatch",
                    value=False,
                    info="Only affects local Cosmos3 targets. Hosted Qwen does not use the local NIM model.",
                )
                use_edited_json = gr.Checkbox(
                    label="Use edited JSON payload for Basic View run",
                    value=False,
                    info="When disabled, Basic View fields generate the payload.",
                )
                payload_preview = gr.Code(label="Preset request body", value=initial_payload, language="json", lines=16)
                commands = gr.Code(label="Generated API commands", value=initial_commands, language="shell", lines=20)
                switch_btn = gr.Button("Switch to selected endpoint", variant="secondary")
                switch_status = gr.Markdown()
                switch_raw = gr.JSON(label="Runtime switch response")

                with gr.Row():
                    fetch_config_btn = gr.Button("Fetch config and runtime JSON")
                with gr.Row():
                    runtime_raw = gr.JSON(label="Runtime")
                    endpoints_raw = gr.JSON(label="Endpoints")
                with gr.Row():
                    health_raw = gr.JSON(label="NIM and health")
                with gr.Row():
                    vlm_config = gr.JSON(label="VLM config")
                    attr_config = gr.JSON(label="Attribute config")
                with gr.Row():
                    hall_config = gr.JSON(label="Hallucination config")
                    obs_config = gr.JSON(label="Obstacle config")

        refresh_inputs = [nim_url, vlm_url, control_url, attribute_url, hallucination_url, obstacle_url]
        refresh_outputs = [status_html, endpoint, runtime_raw, endpoints_raw, health_raw, nim_model_md]
        refresh_btn.click(refresh_state, inputs=refresh_inputs, outputs=refresh_outputs)
        demo.load(refresh_state, inputs=refresh_inputs, outputs=refresh_outputs)

        preview_inputs = [
            source,
            upload,
            server_path,
            weather,
            time_of_day,
            geography,
            road_surface,
        ]
        for component in [source, server_path, weather, time_of_day, geography, road_surface]:
            component.change(_preview_payload, inputs=preview_inputs, outputs=payload_preview)
        source.change(_video_preview_value, inputs=[source, upload, server_path], outputs=video_preview)
        server_path.change(_video_preview_value, inputs=[source, upload, server_path], outputs=video_preview)
        upload.change(
            _upload_video_change,
            inputs=[upload, server_path, weather, time_of_day, geography, road_surface],
            outputs=[source, video_preview, payload_preview],
        )

        command_inputs = [nim_url, vlm_url, control_url, attribute_url, hallucination_url, obstacle_url, endpoint, payload_preview]
        for component in command_inputs:
            component.change(refresh_commands, inputs=command_inputs, outputs=commands)

        run_outputs = [evaluator_job_id, progress_status, result_md, details_table, raw_response, payload_preview, run_status]
        poll_outputs = [progress_status, result_md, details_table, raw_response, payload_preview, run_status]
        run_btn.click(
            start_evaluator_job,
            inputs=[
                source,
                upload,
                server_path,
                weather,
                time_of_day,
                geography,
                road_surface,
                endpoint,
                allow_mismatch,
                use_edited_json,
                payload_preview,
                nim_url,
                vlm_url,
                control_url,
            ],
            outputs=run_outputs,
            queue=False,
        )
        evaluator_timer.tick(poll_evaluator_job, inputs=[evaluator_job_id], outputs=poll_outputs, queue=False)
        confirm_endpoint_btn.click(switch_endpoint, inputs=[control_url, endpoint], outputs=[endpoint_confirm_status, switch_raw])
        switch_btn.click(switch_endpoint, inputs=[control_url, endpoint], outputs=[switch_status, switch_raw])
        fetch_config_btn.click(
            fetch_config,
            inputs=[vlm_url, control_url, attribute_url, hallucination_url, obstacle_url],
            outputs=[vlm_config, runtime_raw, attr_config, hall_config, obs_config],
        )

    return demo


def _public_url(local_url: str, share_url: Optional[str]) -> str:
    explicit = os.environ.get("PUBLIC_GRADIO_URL") or os.environ.get("GRADIO_PUBLIC_URL")
    if explicit:
        return explicit
    if share_url:
        return str(share_url).rstrip("/")
    host = os.environ.get("BYO_VIDEO_LOCAL_HOST")
    if host:
        return f"http://{host}:{PORT}"
    return local_url.rstrip("/")


def _write_launch_markers(local_url: str, share_url: Optional[str] = None) -> str:
    url = _public_url(local_url, share_url)
    try:
        Path("/tmp/gradio_url.txt").write_text(url + "\n", encoding="utf-8")
        Path("/tmp/gradio_live.flag").write_text(url + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"[launch] Could not write Gradio status files: {exc}", flush=True)
    print(f"[launch] {url}", flush=True)
    return url


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    initial_local_url = f"http://127.0.0.1:{PORT}"
    demo = build_app()
    demo.queue(default_concurrency_limit=4)
    result = demo.launch(server_name="0.0.0.0", server_port=PORT, share=SHARE, prevent_thread_lock=True)
    local_url = initial_local_url
    share_url = None
    if isinstance(result, tuple):
        if len(result) > 1 and result[1]:
            local_url = str(result[1])
        if len(result) > 2 and result[2]:
            share_url = str(result[2])
    _write_launch_markers(local_url, share_url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        demo.close()


if __name__ == "__main__":
    main()
