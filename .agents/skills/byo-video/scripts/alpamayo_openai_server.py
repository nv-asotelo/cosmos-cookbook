# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible BYO-video adapter for Alpamayo 1.5 VQA.

The BYO-video frontends already know how to talk to local vLLM/NIM servers via
``/v1/models`` and ``/v1/chat/completions``. Alpamayo runs in-process through
Transformers, so this small server gives it the same wire shape.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import re
import time
import urllib.request
from urllib.parse import unquote, urlparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from PIL import Image

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

LOGGER = logging.getLogger("alpamayo1_5.byo_openai_server")

DEFAULT_MODEL_ID = "nvidia/Alpamayo-1.5-10B"
DATA_URL_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)


@dataclass
class MediaItem:
    mime: str
    data: bytes
    kind: str


@dataclass
class CompletionResult:
    answer: str
    reasoning: str
    prompt_tokens: int
    completion_tokens: int


class AlpamayoEngine:
    """Lazy Alpamayo model/processor holder."""

    def __init__(
        self,
        model_id: str,
        model_path: str | None,
        device: str,
        dtype: torch.dtype,
        attn_implementation: str,
        frames_per_video: int,
        max_decoded_video_frames: int,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.frames_per_video = frames_per_video
        self.max_decoded_video_frames = max_decoded_video_frames
        self.model: Alpamayo1_5 | None = None
        self.processor: Any | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        load_source = self.model_path or self.model_id
        LOGGER.info("Loading %s from %s on %s with %s attention", self.model_id, load_source, self.device, self.attn_implementation)
        kwargs: dict[str, Any] = {"dtype": self.dtype}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        self.model = Alpamayo1_5.from_pretrained(load_source, **kwargs).to(self.device)
        self.model.eval()
        self.processor = helper.get_processor(self.model.tokenizer)

    def complete(self, request: dict[str, Any]) -> CompletionResult:
        self.load()
        assert self.model is not None
        assert self.processor is not None

        prompt, media_items = extract_prompt_and_media(request.get("messages", []))
        if not prompt:
            prompt = os.getenv("ALPAMAYO_DEFAULT_PROMPT", "Describe the scene.")
        if not media_items:
            raise ValueError("Alpamayo BYO requests must include at least one image_url or video_url data URL.")

        frames = self._frames_from_media(media_items)
        if not frames:
            raise ValueError("Could not decode any image frames from the request media.")

        frame_tensor = torch.stack(frames, dim=0)
        messages = helper.create_vqa_message(frame_tensor, question=prompt, camera_indices=None)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_tokens = int(inputs.input_ids.shape[-1])
        model_inputs = helper.to_device({"tokenized_data": inputs}, self.device)

        max_tokens = int(request.get("max_tokens") or request.get("max_completion_tokens") or 256)
        top_p = float(request.get("top_p") or 0.98)
        temperature = max(float(request.get("temperature", 0.6)), 1e-5)
        num_samples = max(1, int(request.get("n") or 1))
        top_k = request.get("top_k")
        top_k_int = int(top_k) if top_k is not None else None

        seed = request.get("seed")
        if seed is None and os.getenv("ALPAMAYO_SEED"):
            seed = int(os.environ["ALPAMAYO_SEED"])
        if seed is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        with torch.inference_mode():
            autocast_enabled = self.device.startswith("cuda")
            with torch.autocast("cuda", dtype=self.dtype, enabled=autocast_enabled):
                extra = self.model.generate_text(
                    data=model_inputs,
                    top_p=top_p,
                    top_k=top_k_int,
                    temperature=temperature,
                    num_samples=num_samples,
                    max_generation_length=max_tokens,
                )

        answer = array_text(extra.get("answer")) or array_text(next(iter(extra.values()), None))
        reasoning = array_text(extra.get("cot"))
        completion_tokens = estimate_tokens(answer) + estimate_tokens(reasoning)
        return CompletionResult(
            answer=answer,
            reasoning=reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _frames_from_media(self, media_items: list[MediaItem]) -> list[torch.Tensor]:
        frames: list[torch.Tensor] = []
        for item in media_items:
            if item.kind == "image" or item.mime.startswith("image/"):
                frames.append(image_bytes_to_tensor(item.data))
                continue
            frames.extend(sample_video_frames(item.data, self.frames_per_video, self.max_decoded_video_frames))
        return frames


def array_text(value: Any) -> str:
    if value is None:
        return ""
    arr = np.asarray(value, dtype=object).reshape(-1)
    return str(arr[0]) if arr.size else ""


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.split()) * 1.3)) if text else 0


def decode_data_url(url: str) -> tuple[str, bytes] | None:
    match = DATA_URL_RE.match(url or "")
    if not match:
        return None
    mime = match.group(1) or "application/octet-stream"
    payload = match.group(3) or ""
    if match.group(2):
        return mime, base64.b64decode(payload)
    from urllib.parse import unquote_to_bytes

    return mime, unquote_to_bytes(payload)


def fetch_url(url: str) -> tuple[str, bytes]:
    with urllib.request.urlopen(url, timeout=float(os.getenv("ALPAMAYO_MEDIA_TIMEOUT", "30"))) as response:
        mime = response.headers.get_content_type() or "application/octet-stream"
        return mime, response.read()


def _media_roots() -> list[str]:
    raw = os.getenv("ALPAMAYO_MEDIA_ROOTS") or os.getenv("BYO_VIDEO_SERVER_MEDIA_ROOTS") or ""
    roots = [part for part in raw.split(":") if part.strip()]
    roots.extend(["/tmp", os.path.expanduser("~"), "/home/horde", "/mnt", "/data"])
    resolved: list[str] = []
    for root in roots:
        path = os.path.abspath(os.path.expanduser(root))
        if os.path.isdir(path) and path not in resolved:
            resolved.append(path)
    return resolved


def fetch_file_url(url: str) -> tuple[str, bytes]:
    parsed = urlparse(url)
    path = unquote(parsed.path or url)
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise ValueError(f"Media file does not exist: {path}")
    if os.getenv("ALPAMAYO_ALLOW_ANY_FILE_URL", "").lower() not in {"1", "true", "yes", "on"}:
        roots = _media_roots()
        if not any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots):
            raise ValueError("Media file is outside ALPAMAYO_MEDIA_ROOTS/BYO_VIDEO_SERVER_MEDIA_ROOTS.")
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".mp4": "video/mp4",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    with open(path, "rb") as handle:
        return mime, handle.read()


def media_from_url(raw_url: str, kind: str) -> MediaItem:
    decoded = decode_data_url(raw_url)
    if decoded is not None:
        mime, data = decoded
        return MediaItem(mime=mime, data=data, kind=kind)
    if raw_url.startswith(("http://", "https://")):
        mime, data = fetch_url(raw_url)
        return MediaItem(mime=mime, data=data, kind=kind)
    if raw_url.startswith("file://") or raw_url.startswith("/"):
        mime, data = fetch_file_url(raw_url)
        return MediaItem(mime=mime, data=data, kind=kind)
    raise ValueError("Only data:, file:, http:, https:, and absolute local media URLs are supported.")


def value_url(value: Any, key: str) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("url") or value.get(key)
    return None


def extract_prompt_and_media(messages: list[Any]) -> tuple[str, list[MediaItem]]:
    text_parts: list[str] = []
    media_items: list[MediaItem] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
            elif part_type == "image_url":
                url = value_url(part.get("image_url"), "image_url")
                if url:
                    media_items.append(media_from_url(url, "image"))
            elif part_type == "video_url":
                url = value_url(part.get("video_url"), "video_url")
                if url:
                    media_items.append(media_from_url(url, "video"))
    return "\n".join(chunk.strip() for chunk in text_parts if chunk.strip()).strip(), media_items


def image_bytes_to_tensor(data: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.asarray(image)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def sample_indices(total: int, desired: int) -> set[int]:
    if total <= desired:
        return set(range(total))
    return {round(i * (total - 1) / (desired - 1)) for i in range(desired)}


def sample_video_frames(data: bytes, desired: int, max_decoded: int) -> list[torch.Tensor]:
    try:
        import av
    except ImportError as exc:
        raise ValueError("PyAV is required to decode video payloads.") from exc

    container = av.open(io.BytesIO(data))
    try:
        stream = next((candidate for candidate in container.streams if candidate.type == "video"), None)
        if stream is None:
            raise ValueError("Video payload did not contain a video stream.")
        decoded: list[Image.Image] = []
        for index, frame in enumerate(container.decode(stream)):
            if index >= max_decoded:
                break
            decoded.append(frame.to_image().convert("RGB"))
        if not decoded:
            return []
        keep = sample_indices(len(decoded), max(1, desired))
        return [
            torch.from_numpy(np.asarray(image)).permute(2, 0, 1).contiguous()
            for index, image in enumerate(decoded)
            if index in keep
        ]
    finally:
        container.close()


def chat_completion_response(
    request: dict[str, Any],
    result: CompletionResult,
    model_id: str,
) -> dict[str, Any]:
    created = int(time.time())
    message: dict[str, Any] = {
        "role": "assistant",
        "content": result.answer,
    }
    if result.reasoning:
        message["reasoning_content"] = result.reasoning
    return {
        "id": f"chatcmpl-alpamayo-{created}",
        "object": "chat.completion",
        "created": created,
        "model": request.get("model") or model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def stream_chat_completion(handler: BaseHTTPRequestHandler, response: dict[str, Any]) -> None:
    choice = response["choices"][0]
    message = choice["message"]
    created = response["created"]
    model = response["model"]
    chunk_base = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache, no-transform")
    handler.send_header("Connection", "close")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()

    if message.get("reasoning_content"):
        payload = {
            **chunk_base,
            "choices": [{"index": 0, "delta": {"reasoning_content": message["reasoning_content"]}}],
        }
        handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
    payload = {
        **chunk_base,
        "choices": [{"index": 0, "delta": {"content": message.get("content", "")}}],
    }
    handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
    done = {**chunk_base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    handler.wfile.write(f"data: {json.dumps(done)}\n\n".encode("utf-8"))
    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


class AlpamayoOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "AlpamayoOpenAI/0.1"
    engine: AlpamayoEngine

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self.send_json({"status": "ok", "model": self.engine.model_id})
            return
        if self.path.rstrip("/") == "/v1/models":
            self.send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.engine.model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "nvidia",
                            "max_model_len": int(os.getenv("ALPAMAYO_MAX_MODEL_LEN", "32768")),
                        }
                    ],
                }
            )
            return
        self.send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {self.path}")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {self.path}")
            return
        try:
            request = self.read_json()
            result = self.engine.complete(request)
            response = chat_completion_response(request, result, self.engine.model_id)
            if request.get("stream"):
                stream_chat_completion(self, response)
            else:
                self.send_json(response)
        except ValueError as exc:
            LOGGER.warning("Bad request: %s", exc)
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - surfaced over HTTP for remote ops
            LOGGER.exception("Completion failed")
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        if length <= 0:
            raise ValueError("Missing JSON request body.")
        return json.loads(self.rfile.read(length))

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error" if status.value < 500 else "server_error",
                }
            },
            status=status,
        )


def dtype_from_env(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def build_engine(args: argparse.Namespace) -> AlpamayoEngine:
    device = args.device or os.getenv("ALPAMAYO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    return AlpamayoEngine(
        model_id=args.model_id or os.getenv("MODEL_ID") or os.getenv("ALPAMAYO_MODEL_ID") or DEFAULT_MODEL_ID,
        model_path=args.model_path or os.getenv("ALPAMAYO_MODEL_PATH"),
        device=device,
        dtype=dtype_from_env(args.dtype or os.getenv("ALPAMAYO_DTYPE", "bfloat16")),
        attn_implementation=args.attn_implementation or os.getenv("ALPAMAYO_ATTENTION", "eager"),
        frames_per_video=int(args.frames or os.getenv("ALPAMAYO_FRAMES", "4")),
        max_decoded_video_frames=int(
            args.max_decoded_video_frames or os.getenv("ALPAMAYO_MAX_DECODED_VIDEO_FRAMES", "96")
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Alpamayo 1.5 through an OpenAI-compatible BYO-video API.")
    parser.add_argument("--host", default=os.getenv("ALPAMAYO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ALPAMAYO_PORT", "8001")))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--max-decoded-video-frames", type=int, default=None)
    parser.add_argument("--preload", action="store_true", help="Load the model before accepting requests.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=os.getenv("ALPAMAYO_LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    engine = build_engine(args)
    if args.preload:
        engine.load()
    handler_cls = type("BoundAlpamayoOpenAIHandler", (AlpamayoOpenAIHandler,), {"engine": engine})
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    LOGGER.info("Serving %s at http://%s:%s/v1", engine.model_id, args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
