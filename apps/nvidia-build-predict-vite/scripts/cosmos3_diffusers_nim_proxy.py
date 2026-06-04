#!/usr/bin/env python3
"""NIM-style `/v1/infer` compatibility proxy for Cosmos3 Diffusers.

The legacy Gradio Predict surface speaks the staging NIM payload:
`prompt`, optional base64 `image`, `steps`, `resolution`, `num_output_frames`,
`fps`, `guidance_scale`, and `seed`. This proxy translates that small contract
to the local Diffusers adapter's `/generate` endpoint and returns `b64_video`.
"""

from __future__ import annotations

import base64
import json
import os
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HOST = os.environ.get("NIM_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("NIM_PROXY_PORT", "8000"))
DIFFUSERS_BASE_URL = os.environ.get("DIFFUSERS_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
MODEL_ID = os.environ.get("MODEL_ID") or os.environ.get("COSMOS_MODEL_ID") or "nvidia/Cosmos3-Nano"


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


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


def _resolution(payload: dict[str, Any]) -> str:
    raw = payload.get("resolution")
    if raw:
        return str(raw)
    video_params = payload.get("video_params") if isinstance(payload.get("video_params"), dict) else {}
    return str(video_params.get("height") or 720)


def _frames(payload: dict[str, Any]) -> int:
    video_params = payload.get("video_params") if isinstance(payload.get("video_params"), dict) else {}
    return _safe_int(payload.get("num_output_frames") or video_params.get("frames_count"), 121, maximum=189)


def _fps(payload: dict[str, Any]) -> float:
    video_params = payload.get("video_params") if isinstance(payload.get("video_params"), dict) else {}
    return _safe_float(payload.get("fps") or video_params.get("frames_per_sec"), 24.0, minimum=1.0, maximum=60.0)


def _media_data_url(payload: dict[str, Any]) -> str | None:
    image_b64 = payload.get("image")
    if isinstance(image_b64, str) and image_b64.strip():
        if image_b64.startswith("data:"):
            return image_b64
        return f"data:image/png;base64,{image_b64}"
    return None


def _post_json(url: str, payload: dict[str, Any], *, timeout: int = 1800) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


class Handler(BaseHTTPRequestHandler):
    server_version = "cosmos3-diffusers-nim-proxy/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback
        path = self.path.split("?", 1)[0]
        if path == "/v1/models":
            _json_response(self, 200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
            return
        if path in {"/health", "/v1/health"}:
            _json_response(self, 200, {"status": "ok", "backend": "diffusers", "base_url": DIFFUSERS_BASE_URL})
            return
        _json_response(self, 404, {"error": f"Unknown path: {path}"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback
        path = self.path.split("?", 1)[0]
        if path != "/v1/infer":
            _json_response(self, 404, {"error": f"Unknown path: {path}"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            _json_response(self, 400, {"error": f"Invalid JSON: {exc}"})
            return

        media_data_url = _media_data_url(payload)
        diffusers_payload = {
            "name": f"gradio-{int(time.time())}",
            "model": MODEL_ID,
            "prompt": payload.get("prompt") or "",
            "negative_prompt": payload.get("negative_prompt") or "",
            "resolution": _resolution(payload),
            "aspect_ratio": "16,9",
            "num_frames": _frames(payload),
            "fps": _fps(payload),
            "num_steps": _safe_int(payload.get("steps"), 35, maximum=80),
            "guidance": _safe_float(payload.get("guidance_scale"), 6.0, minimum=0.0, maximum=20.0),
            "seed": payload.get("seed"),
            "model_mode": "image2video" if media_data_url else "text2video",
        }
        if media_data_url:
            diffusers_payload["mediaDataUrl"] = media_data_url

        try:
            data = _post_json(f"{DIFFUSERS_BASE_URL}/generate", diffusers_payload)
            outputs = data.get("outputs") if isinstance(data, dict) else None
            files = outputs[0].get("files") if outputs and isinstance(outputs[0], dict) else None
            video_path = Path(files[0]) if files else None
            if not video_path or not video_path.exists():
                raise RuntimeError(f"Diffusers response did not include a readable video path: {data!r}")
            _json_response(
                self,
                200,
                {
                    "b64_video": base64.b64encode(video_path.read_bytes()).decode("ascii"),
                    "seed": payload.get("seed"),
                    "backend": "diffusers",
                    "output_path": str(video_path),
                    "raw": data,
                },
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            _json_response(self, exc.code, {"error": body, "backend": "diffusers"})
        except Exception as exc:
            _json_response(
                self,
                500,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "backend": "diffusers",
                    "stack_trace": traceback.format_exc(),
                },
            )


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"cosmos3 Diffusers NIM proxy listening on {HOST}:{PORT} -> {DIFFUSERS_BASE_URL}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
