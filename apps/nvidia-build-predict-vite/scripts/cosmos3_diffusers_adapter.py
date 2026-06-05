#!/usr/bin/env python3
"""Small Ray-compatible HTTP adapter for Diffusers Cosmos3 generation.

The Vite Generator already speaks a simple `/info` + `/generate` contract for
Cosmos3 Ray Serve. This adapter keeps that surface while using Hugging Face
Diffusers' `Cosmos3OmniDiffusersPipeline` locally.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from diffusers.utils import load_image
from diffusers_cosmos3 import Cosmos3OmniDiffusersPipeline
import diffusers_cosmos3.pipeline as cosmos3_pipeline
from diffusers_cosmos3.pipeline import save_img_or_video

# The horde host currently exposes a cuDNN stack that fails on the VAE's 3D
# convolutions. Native CUDA kernels pass the same ops, so keep cuDNN off here.
torch.backends.cudnn.enabled = False


HOST = os.environ.get("DIFFUSERS_ADAPTER_HOST", "0.0.0.0")
PORT = int(os.environ.get("DIFFUSERS_ADAPTER_PORT", "8010"))
MODEL_ID = os.environ.get("DIFFUSERS_MODEL_ID", "nvidia/Cosmos3-Nano")
OUTPUT_DIR = Path(os.environ.get("DIFFUSERS_OUTPUT_DIR", "/tmp/cosmos3_diffusers_outputs"))
UPLOAD_DIR = Path(os.environ.get("DIFFUSERS_UPLOAD_DIR", "/tmp/cosmos3_diffusers_uploads"))
DTYPE = torch.bfloat16

_PIPELINE: Cosmos3OmniDiffusersPipeline | None = None
_PIPELINE_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
_LAST_ERROR: str | None = None


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _pipeline() -> Cosmos3OmniDiffusersPipeline:
    global _PIPELINE, _LAST_ERROR
    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        try:
            torch.set_float32_matmul_precision("high")
            pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                device_map="cuda",
            )
            _PIPELINE = pipe
            _LAST_ERROR = None
            return pipe
        except Exception as exc:  # pragma: no cover - operational diagnostic
            _LAST_ERROR = f"{type(exc).__name__}: {exc}"
            raise


def _parse_aspect_ratio(raw: Any) -> tuple[int, int]:
    value = str(raw or "16,9").replace(":", ",")
    try:
        left, right = [int(part.strip()) for part in value.split(",", 1)]
        if left > 0 and right > 0:
            return left, right
    except Exception:
        pass
    return 16, 9


def _dimensions(resolution: Any, aspect_ratio: Any) -> tuple[int, int]:
    height = int(float(resolution or 720))
    if height <= 0:
        height = 720
    width_ratio, height_ratio = _parse_aspect_ratio(aspect_ratio)
    width = round(height * width_ratio / height_ratio)
    width = max(16, int(round(width / 16)) * 16)
    height = max(16, int(round(height / 16)) * 16)
    return height, width


def _materialize_data_url(data_url: str | None) -> str | None:
    if not data_url:
        return None
    header, _, payload = data_url.partition(",")
    if not payload:
        return None
    mime = "image/png"
    if header.startswith("data:"):
        mime = header[5:].split(";", 1)[0] or mime
    ext = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }.get(mime, "png")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"upload-{time.time_ns()}.{ext}"
    path.write_bytes(base64.b64decode(payload))
    return str(path)


def _conditioning_image(payload: dict[str, Any]):
    vision_path = payload.get("vision_path")
    data_url_path = _materialize_data_url(payload.get("mediaDataUrl"))
    source = data_url_path or vision_path
    if not source:
        return None
    return load_image(source)


def _safe_int(value: Any, fallback: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        number = int(float(value))
    except Exception:
        number = fallback
    number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _safe_float(value: Any, fallback: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except Exception:
        number = fallback
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _sample_args_defaults(mode: str) -> dict[str, Any]:
    framework_root = Path(os.environ.get("COSMOS3_FRAMEWORK_ROOT", Path.cwd()))
    default_path = framework_root / "cosmos_framework" / "inference" / "defaults" / mode / "sample_args.json"
    if default_path.is_file():
        try:
            return json.loads(default_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "duration_template": "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS.",
        "resolution_template": "This video is of {height}x{width} resolution.",
        "negative_metadata_mode": "none",
        "inverse_duration_template": "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS.",
        "inverse_resolution_template": "This video is not of {height}x{width} resolution.",
        "negative_prompt_keep_metadata": True,
        "num_steps": 35,
        "guidance": 6.0,
        "guidance_interval": None,
        "normalize_cfg": False,
        "shift": 10.0,
        "sigma_max": 80.0,
    }


def _write_pipeline_sample_args(payload: dict[str, Any], *, has_image: bool) -> None:
    mode = "image2video" if has_image else "text2video"
    sample_args_dir = Path(cosmos3_pipeline.__file__).parent / "sample_args"
    sample_args_dir.mkdir(parents=True, exist_ok=True)

    defaults = _sample_args_defaults(mode)
    defaults["model_mode"] = mode
    defaults["negative_prompt"] = str(payload.get("negative_prompt") or defaults.get("negative_prompt") or "")
    defaults["num_steps"] = _safe_int(payload.get("num_steps"), int(defaults.get("num_steps") or 35), minimum=1, maximum=80)
    defaults["guidance"] = _safe_float(payload.get("guidance"), float(defaults.get("guidance") or 6.0), minimum=0.0, maximum=20.0)
    defaults["guidance_interval"] = payload.get("guidance_interval", defaults.get("guidance_interval"))
    if payload.get("normalize_cfg") is not None:
        defaults["normalize_cfg"] = bool(payload.get("normalize_cfg"))
    defaults["shift"] = _safe_float(payload.get("shift"), float(defaults.get("shift") or 10.0), minimum=0.0)
    defaults["sigma_max"] = _safe_float(payload.get("sigma_max"), float(defaults.get("sigma_max") or 80.0), minimum=0.0)

    for required_mode in ("image2video", "text2video"):
        output = sample_args_dir / f"{required_mode}.json"
        data = defaults if required_mode == mode else _sample_args_defaults(required_mode)
        data = dict(data)
        data["model_mode"] = required_mode
        data["negative_prompt"] = str(data.get("negative_prompt") or "")
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "cosmos3-diffusers-adapter/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback
        path = urlparse(self.path).path
        if path == "/health":
            _json_response(self, 200, {"status": "ok", "loaded": _PIPELINE is not None, "last_error": _LAST_ERROR})
            return
        if path == "/info":
            _json_response(
                self,
                200,
                {
                    "models": [MODEL_ID],
                    "backend": "diffusers",
                    "output_dir": str(OUTPUT_DIR),
                    "loaded": _PIPELINE is not None,
                    "last_error": _LAST_ERROR,
                    "capabilities": {"text_to_video": True, "image_to_video": True, "action_policy": False},
                    "environment": {
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "cudnn_enabled": torch.backends.cudnn.enabled,
                        "cosmos3_version": "diffusers",
                    },
                },
            )
            return
        _json_response(self, 404, {"status": "error", "message": f"Unknown path: {path}"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback
        path = urlparse(self.path).path
        if path != "/generate":
            _json_response(self, 404, {"status": "error", "message": f"Unknown path: {path}"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            _json_response(self, 400, {"status": "error", "message": f"Invalid JSON: {exc}"})
            return

        sample_name = str(payload.get("name") or f"diffusers-{int(time.time())}").replace("/", "-")
        sample_dir = OUTPUT_DIR / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample_args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        try:
            height, width = _dimensions(payload.get("resolution"), payload.get("aspect_ratio"))
            num_frames = _safe_int(payload.get("num_frames"), 121, minimum=1, maximum=189)
            fps = _safe_float(payload.get("fps"), 24.0, minimum=1.0, maximum=60.0)
            seed = payload.get("seed")
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(int(float(seed)))

            image = _conditioning_image(payload)
            pipe = _pipeline()
            with _GENERATION_LOCK:
                _write_pipeline_sample_args(payload, has_image=image is not None)
                result_frames = pipe(
                    prompt=str(payload.get("prompt") or ""),
                    negative_prompt=payload.get("negative_prompt") or None,
                    image=image,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    fps=fps,
                    condition_frame_indexes=payload.get("condition_frame_indexes_vision") or None,
                    generator=generator,
                    output_type="video",
                )

            video_path = sample_dir / "vision.mp4"
            if not result_frames:
                raise RuntimeError("Diffusers pipeline did not return frames")
            save_img_or_video(result_frames[0], str(video_path.with_suffix("")), fps=fps)
            (sample_dir / "sample_outputs.json").write_text(
                json.dumps({"status": "success", "files": [str(video_path)]}, indent=2),
                encoding="utf-8",
            )
            _json_response(
                self,
                200,
                {
                    "status": "success",
                    "message": "",
                    "outputs": [{"files": [str(video_path)], "content": None}],
                },
            )
        except Exception as exc:  # pragma: no cover - operational diagnostic
            traceback_text = traceback.format_exc()
            (sample_dir / "error.log").write_text(traceback_text, encoding="utf-8")
            _json_response(
                self,
                500,
                {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "stack_trace": traceback_text,
                    "outputs": [],
                },
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"cosmos3 diffusers adapter listening on {HOST}:{PORT} model={MODEL_ID}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
