#!/usr/bin/env python3
"""
Gradio frontend for Cosmos Predict world generation NIMs.

The UI mirrors the public NVIDIA Build page for
build.nvidia.com/nvidia/cosmos-predict1-5b as closely as practical in Gradio:
dark Build styling, model-card header, "Try It" request/response layout,
World Creation Mode, first-frame/first-9-frame conditioning, prompt, seed, and
the hosted OpenAPI request preview.

Self-hosted NIM compatibility is preserved through POST /v1/infer:
  payload: {"prompt", "video"|"image" (base64), "seed", "guidance_scale",
            "steps", "resolution", "num_output_frames", "fps"}
  response: {"b64_video": "<base64 mp4>", "seed": <int>}

Hosted Build OpenAPI schema observed 2026-05-12:
  request: {"prompt": string, "input_image_index"?: 0|1|null, "seed"?: int|null}
  response: {"asset_url": string}

Useful env:
  NIM_HOST / NIM_PORT             local NIM target, defaults localhost:8000
  GRADIO_PORT                     UI port, defaults 7860
  GRADIO_SHARE                    true/false, defaults true
  COSMOS_MODEL_COLLECTION         default HF collection family
  COSMOS_MODEL_ID                 default model id/ref shown in the selector
  COSMOS_PREDICT_BACKEND_SCHEMA   local_nim or build_openapi, defaults local_nim
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

import gradio as gr  # type: ignore

NIM_HOST = os.environ.get("NIM_HOST", "localhost")
NIM_PORT = int(os.environ.get("NIM_PORT", "8000"))
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "true").lower() in {"1", "true", "yes", "on"}
DEFAULT_MODEL_ID = (
    os.environ.get("COSMOS_MODEL_ID")
    or os.environ.get("NIM_SERVED_MODEL_NAME")
    or os.environ.get("MODEL_NAME")
    or os.environ.get("MODEL_ID")
    or "nvidia/cosmos-predict1-7b-video2world"
)
DEFAULT_SCHEMA = os.environ.get("COSMOS_PREDICT_BACKEND_SCHEMA", "local_nim")
NIM_IMAGE = os.environ.get("NIM_IMAGE") or os.environ.get("IMAGE") or ""
IS_COSMOS3_GENERATOR = "cosmos3" in f"{DEFAULT_MODEL_ID} {NIM_IMAGE}".lower() and "gen" in f"{DEFAULT_MODEL_ID} {NIM_IMAGE}".lower()
DEFAULT_COLLECTION = os.environ.get("COSMOS_MODEL_COLLECTION") or ("cosmos3" if IS_COSMOS3_GENERATOR else "cosmos-predict1")
DEFAULT_GUIDANCE = float(os.environ.get("COSMOS_GUIDANCE_SCALE", "6" if IS_COSMOS3_GENERATOR else "7"))
DEFAULT_STEPS = int(os.environ.get("COSMOS_VIDEO_STEPS", "35"))
STAGED_CHECKPOINT_FILE = Path(os.environ.get("PREDICT_STAGED_MODEL_FILE", "/tmp/nvidia_build_predict_staged_model.json"))
INFER_URL = f"http://{NIM_HOST}:{NIM_PORT}/v1/infer"
NIM_BASE_URL = f"http://{NIM_HOST}:{NIM_PORT}/v1"
BUILD_MODEL_URL = "https://build.nvidia.com/nvidia/cosmos-predict1-5b"

COLLECTION_CHOICES = [
    "cosmos3",
    "nvidia-cosmos-2",
    "cosmos-predict25",
    "cosmos-reason1",
    "cosmos-predict1",
    "cosmos",
]

MODEL_HINTS = {
    "cosmos3": DEFAULT_MODEL_ID if "cosmos3" in DEFAULT_MODEL_ID.lower() or "cosmos3" in NIM_IMAGE.lower() else "nvidia/cosmos-3",
    "nvidia-cosmos-2": "nvidia/cosmos-2",
    "cosmos-predict25": "nvidia/cosmos-predict2.5",
    "cosmos-reason1": "nvidia/cosmos-reason1",
    "cosmos-predict1": "nvidia/cosmos-predict1-7b-video2world",
    "cosmos": "nvidia/cosmos",
}

LOCAL_VIDEO_PARAMS = {
    "height": int(os.environ.get("COSMOS_VIDEO_HEIGHT", "704")),
    "width": int(os.environ.get("COSMOS_VIDEO_WIDTH", "1280")),
    "frames_count": int(os.environ.get("COSMOS_VIDEO_FRAMES", "121")),
    "frames_per_sec": int(os.environ.get("COSMOS_VIDEO_FPS", "24")),
}

BUILD_VIDEO_SPEC = {
    "height": 640,
    "width": 1024,
    "input_video_frames": 9,
    "input_image_index": 0,
}
PROGRESS_FRAME_COUNT = 6

CSS = """
:root {
  --nv-green: #76b900;
  --nv-bg: #0c0c0c;
  --nv-panel: #161616;
  --nv-border: #343434;
  --nv-text: #f5f5f5;
  --nv-muted: #b7b7b7;
}
.gradio-container {
  background: var(--nv-bg) !important;
  color: var(--nv-text) !important;
  font-family: "NVIDIA Sans", Inter, system-ui, sans-serif !important;
}
.nv-hero {
  min-height: 168px;
  border: 1px solid var(--nv-border);
  border-radius: 8px;
  padding: 24px;
  background:
    linear-gradient(90deg, rgba(12,12,12,0.98) 0%, rgba(12,12,12,0.84) 50%, rgba(12,12,12,0.3) 100%),
    url("https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-predict1-5b.jpg");
  background-position: center right;
  background-size: cover;
}
.nv-eyebrow {
  color: var(--nv-muted);
  font-size: 14px;
  margin: 0 0 2px;
}
.nv-title {
  font-size: 34px;
  line-height: 1.1;
  margin: 0;
  font-weight: 600;
  overflow-wrap: anywhere;
}
.nv-copy {
  max-width: 760px;
  color: #e5e5e5;
  margin-top: 8px;
}
.nv-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}
.nv-badge {
  border: 1px solid #4b4b4b;
  background: #202020;
  color: #eeeeee;
  border-radius: 16px;
  padding: 3px 10px;
  font-size: 12px;
}
.nv-badge-green {
  border-color: var(--nv-green);
  color: #d8ff9a;
}
.nv-panel {
  border: 1px solid var(--nv-border);
  border-radius: 8px;
  background: var(--nv-panel);
  padding: 14px 16px;
}
.nv-link a {
  color: var(--nv-green) !important;
}
button.primary {
  background: var(--nv-green) !important;
  color: #0c0c0c !important;
  border-color: var(--nv-green) !important;
}
.nv-progress {
  min-height: 360px;
  border-radius: 8px;
  background: #080808;
  padding: 26px 0 18px;
}
.nv-progress-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
  border-bottom: 1px solid #4a4a4a;
  margin-bottom: 24px;
  padding-bottom: 26px;
}
.nv-progress-head p {
  margin: 0;
  color: #d7d7d7;
  font-size: 18px;
  line-height: 28px;
}
.nv-progress-head strong {
  color: #f2f2f2;
}
.nv-progress-head span {
  flex: 0 0 auto;
  color: var(--nv-green);
  font-size: 13px;
  font-weight: 800;
}
.nv-progress-track {
  height: 6px;
  overflow: hidden;
  border-radius: 999px;
  background: #262626;
}
.nv-progress-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--nv-green), #b6ff45);
  box-shadow: 0 0 18px rgba(118, 185, 0, 0.32);
}
.nv-frame-strip {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 14px;
  margin-top: 28px;
}
.nv-frame {
  position: relative;
  height: 118px;
  overflow: hidden;
  border: 1px solid #202020;
  border-radius: 10px;
  background: #111;
}
.nv-frame img,
.nv-frame-placeholder {
  display: block;
  width: 100%;
  height: 100%;
  filter: blur(var(--frame-blur));
  opacity: var(--frame-opacity);
  object-fit: cover;
  transform: scale(1.04);
}
.nv-frame-placeholder {
  background:
    linear-gradient(90deg, rgba(118, 185, 0, 0.18), transparent),
    linear-gradient(135deg, #2a2a2a, #090909 60%, #1f1f1f);
}
.nv-frame span {
  position: absolute;
  right: 8px;
  bottom: 7px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.68);
  color: #eeeeee;
  font-size: 11px;
  font-weight: 800;
  line-height: 18px;
  padding: 0 7px;
}
.nv-progress-note {
  margin-top: 16px;
  color: var(--nv-muted);
  font-size: 13px;
  line-height: 20px;
}
.nv-progress-error .nv-progress-fill {
  background: #f07178;
}
"""


def _gpu_info() -> tuple[str, int]:
    """Return (gpu_name, free_mib). Uses nvidia-smi to avoid a torch dependency."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        line = (out.stdout or "").strip().splitlines()[0]
        name, free = [s.strip() for s in line.split(",")]
        return name, int(free)
    except Exception:
        return "CPU", 0


def _b64_file(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _input_path(file_obj) -> str:
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return str(file_obj.get("path") or file_obj.get("name") or "")
    return str(getattr(file_obj, "name", file_obj) or "")


def _image_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{_b64_file(path)}"


def _progress_frame_sources(world_mode: str, video_file, image_file) -> list[str]:
    if world_mode in {"Image-to-World", "Image-to-Video"}:
        image_path = _input_path(image_file)
        if not image_path or not Path(image_path).exists():
            return []
        return [_image_data_url(image_path)] * PROGRESS_FRAME_COUNT

    video_path = _input_path(video_file)
    if not video_path or not Path(video_path).exists() or not shutil.which("ffmpeg"):
        return []

    frame_dir = Path("/tmp/cosmos_predict_progress_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame in frame_dir.glob("frame_*.jpg"):
        frame.unlink(missing_ok=True)
    out_pattern = str(frame_dir / "frame_%02d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf",
        "fps=1,scale=360:202:force_original_aspect_ratio=increase,crop=360:202",
        "-frames:v", str(PROGRESS_FRAME_COUNT),
        out_pattern,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)
    except Exception:
        return []
    return [_image_data_url(str(path)) for path in sorted(frame_dir.glob("frame_*.jpg"))[:PROGRESS_FRAME_COUNT]]


def _redacted_payload(payload: dict) -> dict:
    result = dict(payload)
    for key in ("video", "image"):
        value = result.get(key)
        if isinstance(value, str):
            result[key] = f"<{len(value):,} base64 chars>"
    return result


def _diagnostic_markdown(
    source: str,
    error: str,
    issue: str,
    likely_cause: str,
    suggestions: list[str],
    endpoint: str | None = None,
    payload: dict | None = None,
) -> str:
    source_label = {
        "backend": "Backend",
        "frontend": "Frontend/API",
        "parameters": "Parameters",
    }.get(source, source.title())
    suggestion_lines = "\n".join(f"- {item}" for item in suggestions)
    details = [
        "### Request failed",
        f"**Failure source:** {source_label}",
        f"**Error:** {error}",
        f"**Why it failed:** {issue}",
        f"**Likely cause:** {likely_cause}",
    ]
    if endpoint:
        details.append(f"**Backend endpoint:** `{endpoint}`")
    if suggestion_lines:
        details.append(f"**What to check next:**\n{suggestion_lines}")
    if payload:
        details.append("**Redacted request payload:**")
        details.append(f"```json\n{json.dumps(_redacted_payload(payload), indent=2)}\n```")
    return "\n\n".join(details)


def _progress_html(percent: float, frame_sources: list[str], source_name: str = "", state: str = "working") -> str:
    percent = max(0, min(100, percent))
    total_frames = int(LOCAL_VIDEO_PARAMS["frames_count"])
    generated_frames = 0 if state == "idle" else max(1, min(total_frames, round((percent / 100) * total_frames)))
    state_class = " nv-progress-error" if state == "error" else ""
    if state == "idle":
        header = "<strong>Output:</strong> Generated frames will appear here during inference."
    elif state == "complete":
        header = "<strong>Generation complete:</strong> Preparing the final video preview."
    else:
        header = "<strong>The autoregressive model is working:</strong> Predicting future frames for you..."
    frames = []
    for idx in range(PROGRESS_FRAME_COUNT):
        threshold = (idx / PROGRESS_FRAME_COUNT) * 86
        readiness = max(0, min(1, (percent - threshold) / 34))
        blur = max(0, 10 - readiness * 10)
        opacity = 0.38 + readiness * 0.62
        frame_number = min(total_frames, max(1, round(((idx + 1) / PROGRESS_FRAME_COUNT) * total_frames)))
        src = frame_sources[idx % len(frame_sources)] if frame_sources else ""
        if src:
            content = f'<img src="{src}" alt="">'
        else:
            content = '<div class="nv-frame-placeholder"></div>'
        frames.append(
            f'<div class="nv-frame" style="--frame-blur:{blur:.2f}px;--frame-opacity:{opacity:.2f}">'
            f'{content}<span>Frame {frame_number}</span></div>'
        )
    source = html.escape(source_name or "conditioning media")
    return f"""
<div class="nv-progress{state_class}">
  <div class="nv-progress-head">
    <p>{header}</p>
    <span>{generated_frames} / {total_frames} frames</span>
  </div>
  <div class="nv-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{round(percent)}">
    <span class="nv-progress-fill" style="width: {max(4, percent):.1f}%"></span>
  </div>
  <div class="nv-frame-strip">{''.join(frames)}</div>
  <div class="nv-progress-note">Conditioning source: {source}. The final MP4 replaces this preview as soon as inference completes.</div>
</div>
"""


def _preprocess_video(src_path: str) -> tuple[str, str]:
    # Cosmos Predict's TorchScript autoencoder expects exactly 5s × 1280×704 × 24fps.
    # Mismatched input shape fails inside the encoder forward pass (HTTP 500
    # observed 2026-05-08 on a 0:45 1920×1080 clip). Normalize before send.
    if not shutil.which("ffmpeg"):
        return src_path, "WARNING: ffmpeg missing; sending video as-is (autoencoder may reject)"
    out_path = "/tmp/cosmos_predict_input.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-vf",
        "scale=1280:704:force_original_aspect_ratio=decrease,"
        "pad=1280:704:(ow-iw)/2:(oh-ih)/2:black",
        "-r", "24",
        "-frames:v", "121",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        out_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        return out_path, "Prepared 121 frames @ 24fps, 1280x704, h264 for local Cosmos Predict NIM"
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", errors="replace")[-200:]
        return src_path, f"WARNING: ffmpeg failed (code {e.returncode}); sending as-is. {tail}"
    except Exception as e:
        return src_path, f"WARNING: ffmpeg error: {e}; sending as-is"


def _seed_value(seed: int | float | None) -> int | None:
    if seed is None:
        return None
    seed_int = int(seed)
    if seed_int < 0:
        return None
    return seed_int


def build_openapi_payload(prompt: str, input_image_index: int, seed: int | float | None) -> dict:
    payload: dict = {
        "prompt": prompt or "",
        "input_image_index": int(input_image_index),
    }
    seed_int = _seed_value(seed)
    if seed_int is not None:
        payload["seed"] = seed_int
    return payload


def _is_cosmos3_generator() -> bool:
    return IS_COSMOS3_GENERATOR


def _nim_resolution_key() -> str:
    height = int(LOCAL_VIDEO_PARAMS["height"])
    if height <= 256:
        return "256"
    if height <= 480:
        return "480"
    return "720"


def _nim_frame_count(value: int | float) -> int:
    requested = max(25, round(float(value)))
    remainder = (requested - 1) % 4
    return requested if remainder == 0 else requested + (4 - remainder)


def build_local_payload(
    world_mode: str,
    media_b64: str,
    prompt: str,
    guidance_scale: float,
    steps: int,
    seed: int | float | None,
) -> dict:
    if _is_cosmos3_generator():
        payload = {
            "prompt": prompt or "",
            "guidance_scale": min(7.0, max(1.0, float(guidance_scale))),
            "steps": int(steps),
            "resolution": _nim_resolution_key(),
            "num_output_frames": _nim_frame_count(LOCAL_VIDEO_PARAMS["frames_count"]),
            "fps": float(LOCAL_VIDEO_PARAMS["frames_per_sec"]),
        }
        if media_b64:
            payload["image"] = media_b64
        seed_int = _seed_value(seed)
        if seed_int is not None:
            payload["seed"] = seed_int
        return payload

    payload = {
        "prompt": prompt or "",
        "guidance_scale": float(guidance_scale),
        "steps": int(steps),
        "video_params": LOCAL_VIDEO_PARAMS,
    }
    if media_b64:
        media_field = "video" if world_mode in {"Video-to-World", "Video-to-Video"} else "image"
        payload[media_field] = media_b64
    seed_int = _seed_value(seed)
    if seed_int is not None:
        payload["seed"] = seed_int
    return payload


def preview_request(
    collection: str,
    model_id: str,
    world_mode: str,
    prompt: str,
    input_image_index: int,
    guidance_scale: float,
    steps: int,
    seed: int,
    backend_schema: str,
):
    if backend_schema == "build_openapi":
        payload = build_openapi_payload(prompt, input_image_index, seed)
        endpoint = "POST https://ai.api.nvidia.com/v1/infer"
    else:
        payload = build_local_payload(
            world_mode,
            "<base64 input video/image omitted>",
            prompt,
            guidance_scale,
            steps,
            seed,
        )
        endpoint = f"POST {INFER_URL}"
    return json.dumps({
        "endpoint": endpoint,
        "collection": collection,
        "model": model_id,
        "payload": payload,
    }, indent=2)


def _post_infer(payload: dict) -> dict:
    req = urllib.request.Request(
        INFER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
        return json.loads(body)


def generate(
    collection: str,
    model_id: str,
    world_mode: str,
    video_file,
    image_file,
    prompt: str,
    input_image_index: int,
    guidance_scale: float,
    steps: int,
    seed: int,
    backend_schema: str,
):
    prep_info = ""
    if world_mode == "Text-to-Video":
        source_path = ""
        source_name = "prompt only"
        frame_sources = []
    else:
        source_path = _input_path(video_file if world_mode == "Video-to-World" else image_file)
        source_name = Path(source_path).name if source_path else "conditioning media"
        frame_sources = _progress_frame_sources(world_mode, video_file, image_file)

    if _is_cosmos3_generator() and world_mode == "Video-to-World":
        yield _progress_html(0, [], "unsupported mode", "error"), None, _diagnostic_markdown(
            "frontend",
            "The staged Cosmos3 generator NIM supports Text-to-Video and Image-to-Video.",
            "Video-to-World is not exposed by this /v1/infer contract.",
            "The selected backend is the staging cosmos3-gen NIM, whose OpenAPI schema accepts prompt plus optional image.",
            ["Switch to Text-to-Video or Image-to-Video."],
            INFER_URL,
        ), None
        return

    yield _progress_html(5, frame_sources, source_name), None, (
        "**The autoregressive model is working:** Preparing conditioning media."
    ), None

    if world_mode == "Text-to-Video":
        media_b64 = ""
        prep_info = "Prompt-only generation; no conditioning media included."
    elif world_mode == "Video-to-World":
        if not source_path:
            yield _progress_html(0, [], "missing video", "error"), None, _diagnostic_markdown(
                "frontend",
                "Upload a video for Video-to-World mode.",
                "No conditioning video was included in the request.",
                "Generate was pressed before an upload or example clip was loaded.",
                ["Upload an MP4 input or choose Image-to-World and upload an image."],
            ), None
            return
        prep_path, prep_info = _preprocess_video(source_path)
        media_b64 = _b64_file(prep_path)
    else:
        if not source_path:
            yield _progress_html(0, [], "missing image", "error"), None, _diagnostic_markdown(
                "frontend",
                f"Upload an image for {world_mode} mode.",
                "No conditioning image was included in the request.",
                "Generate was pressed before an image was loaded.",
                ["Upload a JPEG/PNG input or choose Text-to-Video."]
            ), None
            return
        media_b64 = _b64_file(source_path)

    yield _progress_html(24, frame_sources, source_name), None, (
        f"{prep_info or 'Conditioning media prepared.'}\n\nBuilding Cosmos Predict request."
    ), None

    if backend_schema == "build_openapi":
        payload = build_openapi_payload(prompt, input_image_index, seed)
        # Hosted Build returns an asset_url and accepts no media bytes in the
        # public OpenAPI schema. Route execution to the local NIM unless a future
        # setup layer supplies a hosted transport.
        preview = json.dumps({
            "collection": collection,
            "model": model_id,
            "hosted_schema_payload": payload,
            "note": "Execution still uses local NIM base64 /v1/infer in this BYO-video app.",
        }, indent=2)
    else:
        preview = None

    payload = build_local_payload(world_mode, media_b64, prompt, guidance_scale, steps, seed)
    request_preview = json.dumps(_redacted_payload(payload), indent=2)
    yield _progress_html(36, frame_sources, source_name), None, (
        "**The autoregressive model is working:** Predicting future frames for you..."
    ), request_preview

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_post_infer, payload)
            percent = 42
            while not future.done():
                yield _progress_html(percent, frame_sources, source_name), None, (
                    f"**The autoregressive model is working:** Predicting future frames for you... "
                    f"{round((percent / 100) * LOCAL_VIDEO_PARAMS['frames_count'])}/{LOCAL_VIDEO_PARAMS['frames_count']} frames"
                ), request_preview
                percent = min(96, percent + (5 if percent < 70 else 2))
                time.sleep(2)
            data = future.result()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500] if hasattr(e, "read") else str(e)
        yield _progress_html(100, frame_sources, source_name, "error"), None, _diagnostic_markdown(
            "backend",
            f"HTTP {e.code}: {err_body}",
            f"The backend rejected the request with HTTP {e.code}.",
            "The backend was reachable, but rejected the request. Check model availability, endpoint contract, input shape, and backend logs.",
            ["Inspect the backend log for the matching request.", "Confirm this backend supports POST /v1/infer and the selected model."],
            INFER_URL,
            payload,
        ), request_preview
        return
    except Exception as e:
        yield _progress_html(100, frame_sources, source_name, "error"), None, _diagnostic_markdown(
            "backend",
            f"Cannot reach the Predict backend at {INFER_URL}.",
            str(e)[:300],
            "No Predict-compatible service is listening at the configured backend URL from the Gradio process.",
            [
                "Start the Predict backend/NIM on port 8000 inside the Brev instance.",
                "Or restart this UI with NIM_HOST/NIM_PORT pointing to the active Predict endpoint.",
                "This is not caused by the frontend default generation parameters.",
            ],
            INFER_URL,
            payload,
        ), request_preview
        return

    b64_video = data.get("b64_video")
    if not b64_video:
        asset_url = data.get("asset_url")
        if asset_url:
            yield _progress_html(100, frame_sources, source_name, "complete"), None, (
                f"Hosted response asset_url: {asset_url}"
            ), json.dumps(data, indent=2)[:2000]
            return
        yield _progress_html(100, frame_sources, source_name, "error"), None, (
            _diagnostic_markdown(
                "backend",
                "Backend returned JSON, but no generated video or asset URL was present.",
                f"Response contract mismatch. Response keys: {list(data.keys())}",
                "The target service is not a Cosmos Predict generation endpoint or returned an unexpected schema.",
                ["Confirm the backend model and endpoint are Predict-compatible.", "If using hosted Build schema, use a transport that returns asset_url."],
                INFER_URL,
                payload,
            )
        ), json.dumps(data)[:1000]
        return

    out_path = "/tmp/cosmos_predict_output.mp4"
    Path(out_path).write_bytes(base64.b64decode(b64_video))
    info = (
        f"Generated {len(b64_video)//1000} KB mp4. "
        f"seed={data.get('seed', 'n/a')} | collection={collection} | model={model_id}"
    )
    if prep_info:
        info = f"{prep_info}\n\n{info}"
    response = {k: v for k, v in data.items() if k != "b64_video"}
    if preview:
        response["build_openapi_preview"] = json.loads(preview)
    yield _progress_html(100, frame_sources, source_name, "complete"), out_path, info, json.dumps(response, indent=2)


def update_inputs(mode):
    if mode == "Text-to-Video":
        return gr.update(visible=False), gr.update(visible=False), gr.update(value=0)
    if mode == "Video-to-World":
        return gr.update(visible=True), gr.update(visible=False), gr.update(value=0)
    return gr.update(visible=False), gr.update(visible=True), gr.update(value=0)


def update_model_hint(collection: str):
    return MODEL_HINTS.get(collection, DEFAULT_MODEL_ID)


def hero_markup(model_id: str) -> str:
    display_model = html.escape(model_id or DEFAULT_MODEL_ID)
    image_badge = f'<span class="nv-badge">Staged image: {html.escape(NIM_IMAGE.split("/")[-1])}</span>' if NIM_IMAGE else ""
    return f"""
<div class="nv-hero">
  <p class="nv-eyebrow">nvidia</p>
  <h1 class="nv-title">{display_model}</h1>
  <p class="nv-copy">Generates physics-aware video from a text prompt or conditioning image for physical AI development.</p>
  <div class="nv-badges">
    <span class="nv-badge nv-badge-green">Free Endpoint</span>
    <span class="nv-badge">Physical AI</span>
    <span class="nv-badge">robotics</span>
    <span class="nv-badge">text-to-video</span>
    <span class="nv-badge">image-to-video</span>
    <span class="nv-badge">NIM local: {NIM_BASE_URL}</span>
    {image_badge}
  </div>
</div>
"""


def update_model_selection(collection: str):
    model = update_model_hint(collection)
    return model, hero_markup(model)


def schema_note(backend_schema: str):
    if backend_schema == "build_openapi":
        return (
            "Build OpenAPI preview: prompt, input_image_index, seed; response asset_url. "
            "Generation still posts local base64 media to the self-hosted NIM."
        )
    if _is_cosmos3_generator():
        return "Local Cosmos3 Generator NIM: prompt plus optional base64 image, guidance, steps, seed, resolution, num_output_frames, and fps."
    return "Local NIM: prompt plus optional base64 image/video, guidance, steps, seed, and video_params."


def _write_launch_markers(launch_result) -> None:
    local_url = f"http://127.0.0.1:{GRADIO_PORT}"
    share_url = None
    if isinstance(launch_result, tuple):
        if len(launch_result) >= 2:
            local_url = str(launch_result[1] or local_url)
        if len(launch_result) >= 3:
            share_url = launch_result[2]
    public_url = str(share_url or local_url)
    if not share_url and ("127.0.0.1" in public_url or "localhost" in public_url or "0.0.0.0" in public_url):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2)
            sock.connect(("8.8.8.8", 80))
            host_ip = sock.getsockname()[0]
            sock.close()
            public_url = public_url.replace("127.0.0.1", host_ip).replace("localhost", host_ip).replace("0.0.0.0", host_ip)
        except Exception:
            pass
    Path("/tmp/gradio_url.txt").write_text(public_url + "\n", encoding="utf-8")
    Path("/tmp/gradio_live.flag").write_text(public_url + "\n", encoding="utf-8")


def _write_initial_launch_markers() -> None:
    local_url = f"http://127.0.0.1:{GRADIO_PORT}"
    Path("/tmp/gradio_url.txt").write_text(local_url + "\n", encoding="utf-8")
    Path("/tmp/gradio_live.flag").write_text(local_url + "\n", encoding="utf-8")


def _remember_staged_checkpoint() -> None:
    if not NIM_IMAGE:
        return
    payload = {
        "image": NIM_IMAGE,
        "served_model": DEFAULT_MODEL_ID,
        "backend": "nim_local",
        "base_url": NIM_BASE_URL,
        "infer_url": INFER_URL,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        STAGED_CHECKPOINT_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        STAGED_CHECKPOINT_FILE.chmod(0o600)
    except Exception:
        pass


_gpu_name, _free_mib = _gpu_info()
if DEFAULT_COLLECTION not in COLLECTION_CHOICES:
    COLLECTION_CHOICES.append(DEFAULT_COLLECTION)
_remember_staged_checkpoint()

with gr.Blocks(title=DEFAULT_MODEL_ID, css=CSS) as demo:
    hero = gr.Markdown(hero_markup(DEFAULT_MODEL_ID))
    gr.Markdown(
        f"""
<div class="nv-panel nv-link">
GPU: {_gpu_name} | VRAM free: {_free_mib:,} MiB | Build page:
<a href="{BUILD_MODEL_URL}" target="_blank">{BUILD_MODEL_URL}</a>
</div>
"""
    )
    with gr.Row():
        with gr.Column(scale=1):
            collection = gr.Dropdown(
                choices=COLLECTION_CHOICES,
                value=DEFAULT_COLLECTION,
                label="Hugging Face collection",
                info="Setup can route this frontend from Cosmos model family selections.",
            )
            model_id = gr.Textbox(
                label="Model",
                value=DEFAULT_MODEL_ID,
                info="Shown for routing/context; local NIM selection still happens in setup.",
            )
            world_mode = gr.Radio(
                choices=["Text-to-Video", "Image-to-Video", "Video-to-World"],
                value="Text-to-Video",
                label="World Creation Mode",
                info="Text-to-Video is prompt-only; Image-to-Video uses one conditioning image."
            )
            video_in = gr.Video(label="Input Video (mp4)", visible=False)
            image_in = gr.Image(label="Input Image", type="filepath", visible=False)
            prompt = gr.Textbox(
                label="Prompt",
                value="A smooth first-person robot manipulation video in a greenhouse. The robot arm reaches toward a ripe red apple, gently grasps it, twists, and places it into a harvest bin. Natural daylight, stable camera, realistic physics.",
                lines=3,
                max_lines=5,
            )
            with gr.Accordion("Request schema", open=True):
                backend_schema = gr.Radio(
                    choices=[
                        ("Local self-hosted NIM /v1/infer", "local_nim"),
                        ("NVIDIA Build OpenAPI preview", "build_openapi"),
                    ],
                    value=DEFAULT_SCHEMA if DEFAULT_SCHEMA in {"local_nim", "build_openapi"} else "local_nim",
                    label="API contract",
                )
                schema_status = gr.Markdown(schema_note(DEFAULT_SCHEMA))
                input_image_index = gr.Slider(
                    0,
                    1,
                    value=BUILD_VIDEO_SPEC["input_image_index"],
                    step=1,
                    label="input_image_index",
                    info="Hosted schema field. 0 is the first input image/frame; max is 1.",
                )
            with gr.Accordion("Generation parameters", open=True):
                guidance = gr.Slider(1.0, 10.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance scale (CFG)")
                steps = gr.Slider(1, 50, value=DEFAULT_STEPS, step=1, label="Steps")
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                gr.Markdown(
                    "Quick staging defaults are controlled by COSMOS_VIDEO_HEIGHT/WIDTH/FRAMES/FPS. "
                    "For the Cosmos3 generator NIM, start with 256p, 25 frames, and 35 steps for quality previews."
                )
            submit = gr.Button("Generate New World", variant="primary")
        with gr.Column(scale=1):
            progress_panel = gr.HTML(value=_progress_html(0, [], "waiting for input", "idle"))
            video_out = gr.Video(label="Generated Future World", interactive=False)
            status = gr.Markdown()
            request_preview = gr.Code(label="Request preview", language="json")
            response = gr.Code(label="Response (JSON, video stripped)", language="json")
    collection.change(update_model_selection, collection, [model_id, hero])
    model_id.change(hero_markup, model_id, hero)
    world_mode.change(update_inputs, world_mode, [video_in, image_in, input_image_index])
    backend_schema.change(schema_note, backend_schema, schema_status)
    preview_inputs = [
        collection,
        model_id,
        world_mode,
        prompt,
        input_image_index,
        guidance,
        steps,
        seed,
        backend_schema,
    ]
    for component in preview_inputs:
        component.change(preview_request, preview_inputs, request_preview)
    demo.load(preview_request, preview_inputs, request_preview)
    submit.click(generate,
                 [
                     collection,
                     model_id,
                     world_mode,
                     video_in,
                     image_in,
                     prompt,
                     input_image_index,
                     guidance,
                     steps,
                     seed,
                     backend_schema,
                 ],
                 [progress_panel, video_out, status, response])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    _write_initial_launch_markers()
    result = demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        prevent_thread_lock=True,
    )
    _write_launch_markers(result)
    if GRADIO_SHARE:
        for _ in range(30):
            share_url = getattr(demo, "share_url", None)
            if share_url:
                _write_launch_markers((None, getattr(demo, "local_url", None), share_url))
                break
            time.sleep(1)
    while True:
        time.sleep(3600)
