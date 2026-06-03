#!/usr/bin/env python3
"""Gradio frontend for the Cosmos Evaluator REST services."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
import re
import shutil
import time
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
DEFAULT_KEYFRAME_INTERVAL_S = 2.0
DEFAULT_KEYFRAME_WIDTH = 640
DEFAULT_LONG_VIDEO_MAX_FRAMES = 5

DEFAULT_ENDPOINTS = ["cosmos3-super-reasoner", "cosmos3-nano-reasoner", "qwen3.5-397b-a17b"]


CSS = """
:root {
  --nv-bg: #050505;
  --nv-panel: #121212;
  --nv-panel-2: #1b1b1b;
  --nv-border: #343434;
  --nv-green: #76b900;
  --nv-text: #f3f3f3;
  --nv-muted: #b9b9b9;
  --nv-red: #ef4444;
  --nv-yellow: #f59e0b;
}
body, .gradio-container {
  background: var(--nv-bg) !important;
  color: var(--nv-text) !important;
  font-family: Inter, Arial, sans-serif !important;
}
.gradio-container { max-width: 1500px !important; margin: 0 auto !important; }
.nv-appbar {
  height: 46px; border-bottom: 1px solid var(--nv-border); background: #080808;
  display: flex; align-items: center; gap: 24px; padding: 0 18px; color: #ddd;
}
.nv-logo { font-weight: 800; color: white; }
.nv-logo span { color: var(--nv-green); }
.nv-nav { display: flex; gap: 18px; color: #aaa; font-size: 13px; }
.nv-hero {
  border: 1px solid var(--nv-border); border-radius: 8px; padding: 20px 24px; margin-bottom: 14px;
  background: linear-gradient(90deg, #0b0b0b, #111 74%, #151515);
}
.nv-hero h1 { margin: 0; color: white; font-size: 30px; line-height: 1.15; letter-spacing: 0; }
.nv-hero p { color: #d8d8d8; max-width: 820px; margin: 8px 0 0; }
.nv-pill {
  display: inline-block; border: 1px solid var(--nv-border); border-radius: 999px; padding: 3px 8px;
  margin: 10px 6px 0 0; color: #d6d6d6; font-size: 12px; background: #171717;
}
.status-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 8px;
  margin: 0 0 10px;
}
.status-card { border: 1px solid var(--nv-border); border-radius: 8px; background: var(--nv-panel); padding: 10px; }
.status-title { color: #aaa; font-size: 12px; margin-bottom: 4px; }
.status-ok { color: #86efac; font-weight: 700; }
.status-warn { color: #fde68a; font-weight: 700; }
.status-bad { color: #fca5a5; font-weight: 700; }
.api-note { color: #c7c7c7; font-size: 13px; line-height: 1.45; }
.warning-box {
  border: 1px solid #92400e; background: #251504; color: #fde68a; border-radius: 8px;
  padding: 10px 12px; margin: 6px 0;
}
.ok-box {
  border: 1px solid #365314; background: #111d07; color: #d9f99d; border-radius: 8px;
  padding: 10px 12px; margin: 6px 0;
}
"""


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


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
        "VLM": (vlm_url, "/health"),
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


def _status_html(health: Dict[str, Any], runtime: Dict[str, Any], nim_model: Optional[str]) -> str:
    active = html.escape(str(runtime.get("active_endpoint") or "unknown"))
    model = html.escape(str(nim_model or "unreachable"))
    cards = [
        ("NIM model", model, bool(nim_model)),
        ("Runtime", active, bool(runtime)),
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


def _stage_upload(file_value: Any) -> str:
    if file_value is None:
        raise ValueError("Choose an uploaded video or switch the source to the sample/server path.")
    src = getattr(file_value, "name", None) or str(file_value)
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
    analysis_mode: str,
    keyframe_interval_s: float,
    keyframe_width: int,
    max_frames: int,
) -> Dict[str, Any]:
    payload = {
        "augmented_video_url": video_path,
        "preset_conditions": {
            "name": "environment",
            "weather": weather,
            "time_of_day_illumination": time_of_day,
            "region_geography": geography,
            "road_surface_conditions": road_surface,
        },
    }
    if analysis_mode == "Long video analysis":
        payload["preset_check_config"] = {
            "keyframe_interval_s": float(keyframe_interval_s),
            "keyframe_width": int(keyframe_width),
            "max_frames": int(max_frames),
        }
    return payload


def _preview_payload(
    source: str,
    upload: Any,
    server_path: str,
    weather: str,
    time_of_day: str,
    geography: str,
    road_surface: str,
    analysis_mode: str,
    keyframe_interval_s: float,
    keyframe_width: int,
    max_frames: int,
) -> str:
    if source == "Uploaded video" and upload is not None:
        name = _safe_filename(getattr(upload, "name", None) or str(upload))
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
            analysis_mode,
            keyframe_interval_s,
            keyframe_width,
            max_frames,
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
    model_md = f"Loaded NIM model: `{nim_model or 'unreachable'}`"
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
        return f"Active endpoint switched to `{endpoint}`.", body
    return f"Endpoint switch failed for `{endpoint}`.", body


def run_evaluator(
    source: str,
    upload: Any,
    server_path: str,
    weather: str,
    time_of_day: str,
    geography: str,
    road_surface: str,
    analysis_mode: str,
    keyframe_interval_s: float,
    keyframe_width: int,
    max_frames: int,
    endpoint: str,
    allow_mismatch: bool,
    use_edited_json: bool,
    payload_text: str,
    nim_url: str,
    vlm_url: str,
    control_url: str,
) -> Tuple[str, List[List[Any]], Any, str, str]:
    try:
        endpoints, _ = _runtime_endpoints(control_url)
        nim_model, _ = _detect_nim_model(nim_url)
        mismatch = _mismatch_message(endpoint, endpoints, nim_model)
        if mismatch and not allow_mismatch:
            return (
                f'<div class="warning-box">{html.escape(mismatch)} Enable "Allow endpoint/model mismatch" in Advanced View to run anyway.</div>',
                [],
                {"blocked": True, "reason": mismatch},
                payload_text,
                "Blocked before API request.",
            )

        if endpoint:
            switched, switch_body = _service_post(control_url, "/runtime/vlm/switch", {"endpoint": endpoint}, timeout=HTTP_TIMEOUT_S)
            if not switched:
                return (
                    "Runtime switch failed.",
                    [],
                    switch_body,
                    payload_text,
                    "Runtime switch failed before preset request.",
                )

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
                analysis_mode,
                keyframe_interval_s,
                keyframe_width,
                max_frames,
            )
            payload_text = _json(payload)

        ok, body = _service_post(vlm_url, "/process/preset", payload)
        summary, rows = _response_summary(body)
        status = '<div class="ok-box">Preset evaluator completed.</div>' if ok else '<div class="warning-box">Preset evaluator returned an error.</div>'
        return status + "\n\n" + summary, rows, body, _json(payload), "Success." if ok else "Request failed."
    except Exception as exc:
        return (
            f'<div class="warning-box">{html.escape(str(exc))}</div>',
            [],
            {"error": str(exc)},
            payload_text,
            "Request failed before completion.",
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
        "Long video analysis",
        DEFAULT_KEYFRAME_INTERVAL_S,
        DEFAULT_KEYFRAME_WIDTH,
        DEFAULT_LONG_VIDEO_MAX_FRAMES,
    )
    initial_commands = _curl_commands(
        NIM_URL,
        VLM_URL,
        CONTROL_URL,
        ATTRIBUTE_URL,
        HALLUCINATION_URL,
        OBSTACLE_URL,
        "cosmos3-super-reasoner",
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
              <p>Run the evaluator smoke example, stage your own MP4, switch the active VLM endpoint, and inspect the REST commands that drive the service.</p>
              <span class="nv-pill">Basic evaluator</span><span class="nv-pill">Advanced API console</span><span class="nv-pill">Cosmos3 NIM runtime</span>
            </div>
            """
        )

        status_html = gr.HTML(value="<div class=\"api-note\">Click Refresh status to query services.</div>")
        refresh_btn = gr.Button("Refresh status", variant="secondary")

        with gr.Tabs():
            with gr.TabItem("Basic View"):
                with gr.Row():
                    with gr.Column(scale=5):
                        sample_video = gr.Video(label="Sample video", value=sample_video_value(), height=260)
                        source = gr.Radio(
                            label="Video source",
                            choices=["Sample video", "Uploaded video", "Server path"],
                            value="Sample video",
                        )
                        analysis_mode = gr.Radio(
                            label="Analysis mode",
                            choices=["Long video analysis", "Normal evaluator behavior"],
                            value="Long video analysis",
                            info=(
                                "Long mode sends at most 5 sampled frames across the clip for Cosmos3 NIM. "
                                "Normal mode leaves the evaluator's original frame sampling untouched."
                            ),
                        )
                        upload = gr.File(label="Upload MP4", file_types=[".mp4"], file_count="single")
                        server_path = gr.Textbox(
                            label="Server path",
                            value=SAMPLE_API_PATH,
                            info="Path visible to evaluator containers, usually /data/<filename>.",
                        )
                    with gr.Column(scale=4):
                        endpoint = gr.Dropdown(label="Evaluator VLM endpoint", choices=DEFAULT_ENDPOINTS, value="cosmos3-super-reasoner")
                        nim_model_md = gr.Markdown("Loaded NIM model: `unknown`")
                        weather = gr.Textbox(label="Weather", value=DEFAULT_PRESET["weather"])
                        time_of_day = gr.Textbox(
                            label="Time of day / illumination",
                            value=DEFAULT_PRESET["time_of_day_illumination"],
                        )
                        geography = gr.Textbox(label="Region / geography", value=DEFAULT_PRESET["region_geography"])
                        road_surface = gr.Textbox(label="Road surface", value=DEFAULT_PRESET["road_surface_conditions"])
                        run_btn = gr.Button("Run evaluator", variant="primary")

                result_md = gr.Markdown("Run the evaluator to see the score summary.")
                details_table = gr.Dataframe(
                    headers=["Check", "Score", "Preset", "Explanation"],
                    datatype=["str", "number", "str", "str"],
                    label="Scoring details",
                    interactive=False,
                )
                raw_response = gr.JSON(label="Raw evaluator response")

            with gr.TabItem("Advanced View"):
                with gr.Row():
                    nim_url = gr.Textbox(label="NIM URL", value=NIM_URL)
                    vlm_url = gr.Textbox(label="VLM URL", value=VLM_URL)
                    control_url = gr.Textbox(label="Runtime/control URL", value=CONTROL_URL)
                with gr.Row():
                    attribute_url = gr.Textbox(label="Attribute URL", value=ATTRIBUTE_URL)
                    hallucination_url = gr.Textbox(label="Hallucination URL", value=HALLUCINATION_URL)
                    obstacle_url = gr.Textbox(label="Obstacle URL", value=OBSTACLE_URL)

                allow_mismatch = gr.Checkbox(
                    label="Allow endpoint/model mismatch",
                    value=False,
                    info="Use only when the selected endpoint is hosted or you are intentionally testing an unloaded local NIM.",
                )
                use_edited_json = gr.Checkbox(
                    label="Use edited JSON payload for Basic View run",
                    value=False,
                    info="When disabled, Basic View fields generate the payload.",
                )
                with gr.Row():
                    keyframe_interval_s = gr.Slider(
                        minimum=0.5,
                        maximum=30.0,
                        value=DEFAULT_KEYFRAME_INTERVAL_S,
                        step=0.5,
                        label="Long mode sampling interval seconds",
                        info="Candidate frame spacing before selecting up to 5 frames across the clip.",
                    )
                    keyframe_width = gr.Number(
                        value=DEFAULT_KEYFRAME_WIDTH,
                        precision=0,
                        label="Long mode keyframe width",
                    )
                    max_frames = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=DEFAULT_LONG_VIDEO_MAX_FRAMES,
                        step=1,
                        label="Long mode max frames",
                        info="Cosmos3 NIM accepts at most 5 images in one prompt.",
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
                request_status = gr.Markdown()

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
            analysis_mode,
            keyframe_interval_s,
            keyframe_width,
            max_frames,
        ]
        for component in preview_inputs:
            component.change(_preview_payload, inputs=preview_inputs, outputs=payload_preview)

        command_inputs = [nim_url, vlm_url, control_url, attribute_url, hallucination_url, obstacle_url, endpoint, payload_preview]
        for component in command_inputs:
            component.change(refresh_commands, inputs=command_inputs, outputs=commands)

        run_btn.click(
            run_evaluator,
            inputs=[
                source,
                upload,
                server_path,
                weather,
                time_of_day,
                geography,
                road_surface,
                analysis_mode,
                keyframe_interval_s,
                keyframe_width,
                max_frames,
                endpoint,
                allow_mismatch,
                use_edited_json,
                payload_preview,
                nim_url,
                vlm_url,
                control_url,
            ],
            outputs=[result_md, details_table, raw_response, payload_preview, request_status],
        )
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
