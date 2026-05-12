#!/usr/bin/env python3
"""
Build.NVIDIA-style Cosmos Reason Gradio frontend for an OpenAI-compatible
vLLM/NIM server.

Target default for the Horde BYO-video flow:
  VLLM_BASE_URL=http://localhost:8000/v1
  model auto-detected from /models, e.g. nvidia/Cosmos3-Nano-Reasoner
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import pathlib
import socket
import subprocess
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - launch-time guard
    raise SystemExit("gradio is required. Install with: pip install gradio") from exc

try:
    import requests
except ImportError as exc:  # pragma: no cover - launch-time guard
    raise SystemExit("requests is required. Install with: pip install requests") from exc


BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = os.environ.get("VLLM_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))
MODEL_OVERRIDE = os.environ.get("MODEL_NAME") or os.environ.get("VLLM_MODEL")
PORT = int(os.environ.get("GRADIO_PORT", "7860"))
SHARE = os.environ.get("GRADIO_SHARE", "true").lower() not in {"0", "false", "no"}
_DEFAULT_SOURCE_DIR = pathlib.Path("/tmp/build-nvidia-com")
_LOCAL_SOURCE_DIR = pathlib.Path("/Users/asotelo/cosmos-cookbook-reports/build-nvidia-com")
LOCAL_SOURCE_DIR = pathlib.Path(os.environ.get("BUILD_NVIDIA_SOURCE_DIR", _DEFAULT_SOURCE_DIR))
if not LOCAL_SOURCE_DIR.exists() and _LOCAL_SOURCE_DIR.exists():
    LOCAL_SOURCE_DIR = _LOCAL_SOURCE_DIR

DEFAULT_SYSTEM = os.environ.get("DEFAULT_SYSTEM_PROMPT", "You are a helpful assistant.")
DEFAULT_USER = os.environ.get(
    "DEFAULT_USER_PROMPT",
    "What is happening in this video or image?",
)
REASONING_SUFFIX = (
    "Answer the question using the following format:\n\n"
    "<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag."
)
RACE_CAR_PROMPT = (
    "Describe the video. Add timestamps in mm:ss format.\n\n"
    "Answer the question using the following format:\n\n"
    "<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag and include the timestamps."
)

ALLOWED_EXTENSIONS = {".mp4", ".jpg", ".jpeg", ".png"}

BUILD_CSS = """
:root {
  --nv-bg: #050505;
  --nv-panel: #121212;
  --nv-panel-2: #1a1a1a;
  --nv-border: #3a3a3a;
  --nv-green: #76b900;
  --nv-text: #f4f4f4;
  --nv-muted: #b7b7b7;
}
body,
.gradio-container {
  background: var(--nv-bg) !important;
  color: var(--nv-text) !important;
  font-family: Inter, Arial, sans-serif !important;
}
.gradio-container {
  max-width: 1600px !important;
  margin: 0 auto !important;
}
.nv-appbar {
  height: 48px;
  border-bottom: 1px solid var(--nv-border);
  background: #080808;
  display: flex;
  align-items: center;
  gap: 28px;
  padding: 0 18px;
  color: #e8e8e8;
  font-size: 14px;
}
.nv-logo {
  font-weight: 800;
  letter-spacing: 0;
  color: white;
}
.nv-logo span {
  color: var(--nv-green);
}
.nv-nav {
  display: flex;
  gap: 22px;
  color: #b8b8b8;
}
.nv-hero {
  min-height: 154px;
  border: 1px solid var(--nv-border);
  border-radius: 8px;
  padding: 22px 28px;
  background:
    linear-gradient(90deg, rgba(10,10,10,0.98), rgba(10,10,10,0.82) 58%, rgba(10,10,10,0.12)),
    url("https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-reason2-8b.jpg");
  background-position: center right;
  background-size: cover;
}
.nv-publisher {
  color: #b5b5b5;
  font-size: 15px;
  margin: 0 0 2px;
  text-transform: lowercase;
}
.nv-title-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.nv-title-row h1 {
  color: white;
  font-size: 31px;
  line-height: 1.15;
  margin: 0;
  font-weight: 600;
  letter-spacing: 0;
}
.nv-pill-solid {
  display: inline-flex;
  align-items: center;
  height: 20px;
  border-radius: 10px;
  background: #2a2a2a;
  border: 1px solid #555;
  color: #f5f5f5;
  padding: 0 8px;
  font-size: 12px;
}
.nv-desc {
  max-width: 680px;
  color: #e7e7e7;
  margin: 8px 0 0;
  font-size: 15px;
}
.nv-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 12px;
}
.nv-tags span {
  border: 1px solid #4b6666;
  color: #bfefef;
  background: rgba(20, 50, 50, 0.35);
  border-radius: 12px;
  padding: 2px 8px;
  font-size: 12px;
  text-transform: lowercase;
}
.nv-tabs {
  display: none;
  gap: 22px;
  border-bottom: 1px solid var(--nv-border);
  margin-top: 18px;
  color: #cfcfcf;
}
.nv-tabs span {
  padding: 12px 0 10px;
}
.nv-tabs span:first-child {
  color: white;
  border-bottom: 2px solid var(--nv-green);
}
.section-heading {
  border-bottom: 1px solid var(--nv-border);
  color: white;
  font-size: 22px;
  line-height: 1.2;
  padding-bottom: 10px;
  margin-bottom: 12px;
  font-weight: 600;
}
.input-column,
.output-column {
  border-color: var(--nv-border) !important;
}
.api-panel {
  border: 1px solid var(--nv-border);
  border-radius: 4px;
  background: var(--nv-panel);
  padding: 12px;
  color: #e8e8e8;
  margin-bottom: 12px;
}
.api-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--nv-border);
  padding-bottom: 8px;
  margin-bottom: 8px;
}
.api-credit {
  border: 1px solid var(--nv-border);
  background: #0f0f0f;
  border-radius: 4px;
  padding: 6px 9px;
  color: #d7d7d7;
}
.api-actions {
  display: flex;
  gap: 6px;
}
.api-actions span {
  background: var(--nv-green);
  color: #071100;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 700;
  padding: 5px 10px;
}
.backend-note {
  color: var(--nv-muted);
  font-size: 12px;
  line-height: 1.45;
}
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .block-title {
  color: #eeeeee !important;
}
.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap,
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
  background: var(--nv-panel) !important;
  color: var(--nv-text) !important;
  border-color: var(--nv-border) !important;
}
.gradio-container .secondary-button {
  background: #1d1d1d !important;
  border-color: #555 !important;
  color: #f3f3f3 !important;
}
.gradio-container .primary {
  background: var(--nv-green) !important;
  border-color: var(--nv-green) !important;
  color: #071100 !important;
}
.gradio-container pre,
.gradio-container code {
  background: #080808 !important;
  color: #e9e9e9 !important;
}
"""


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def detect_model(timeout: float = 5.0) -> Tuple[str, str]:
    if MODEL_OVERRIDE:
        return MODEL_OVERRIDE, "env"

    try:
        resp = requests.get(f"{BASE_URL}/models", headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data") or []
        if models and models[0].get("id"):
            return str(models[0]["id"]), "server"
    except Exception as exc:
        print(f"[model] Could not detect model from {BASE_URL}/models: {exc}", flush=True)

    return "nvidia/Cosmos3-Nano-Reasoner", "fallback"


def _file_path(upload: Any) -> Optional[str]:
    if upload is None:
        return None
    if isinstance(upload, (str, os.PathLike)):
        return os.fspath(upload)
    for attr in ("path", "name"):
        value = getattr(upload, attr, None)
        if value:
            return os.fspath(value)
    if isinstance(upload, dict):
        for key in ("path", "name"):
            if upload.get(key):
                return os.fspath(upload[key])
    return None


def _media_part(path: str) -> Dict[str, Any]:
    ext = pathlib.Path(path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Supported uploads are .mp4, .jpg, .jpeg, and .png")

    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")

    if ext == ".mp4":
        return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}}

    mime = mimetypes.guess_type(path)[0] or ("image/png" if ext == ".png" else "image/jpeg")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _build_messages(upload: Any, user_prompt: str, system_prompt: str) -> List[Dict[str, Any]]:
    user_prompt = (user_prompt or "").strip()
    system_prompt = (system_prompt or "").strip()
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    path = _file_path(upload)
    if path:
        content: Any = [_media_part(path), {"type": "text", "text": user_prompt}]
    else:
        content = user_prompt

    messages.append({"role": "user", "content": content})
    return messages


def _request_preview(
    model: str,
    upload: Any,
    user_prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    top_k: int,
    seed: int,
    fps: float,
) -> str:
    path = _file_path(upload)
    if path:
        ext = pathlib.Path(path).suffix.lower()
        media_type = "video_url" if ext == ".mp4" else "image_url"
        media_obj = {media_type: {"url": f"data:{'video/mp4' if ext == '.mp4' else 'image/...'};base64,<uploaded-file>"}}
        user_content: Any = [{"type": media_type, **media_obj}, {"type": "text", "text": user_prompt}]
    else:
        user_content = user_prompt

    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
        "top_k": top_k,
        "stream": True,
    }
    if path and pathlib.Path(path).suffix.lower() == ".mp4":
        body["media_io_kwargs"] = {"video": {"fps": fps}}
    return json.dumps(body, indent=2)


def _stream_text(resp: requests.Response) -> Iterable[str]:
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if text is None:
            message = choice.get("message") or {}
            text = message.get("content")
        if text:
            yield text


def run_inference(
    upload: Any,
    user_prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    top_k: int,
    seed: int,
    fps: float,
    use_reasoning: bool,
) -> Generator[Tuple[str, str, str], None, None]:
    model, source = detect_model(timeout=3.0)
    prompt = (user_prompt or "").strip()
    if use_reasoning and REASONING_SUFFIX not in prompt:
        prompt = f"{prompt}\n\n{REASONING_SUFFIX}".strip()

    preview = _request_preview(
        model, upload, prompt, system_prompt, max_tokens, temperature, top_p,
        repetition_penalty, top_k, seed, fps
    )
    yield "", f"Detected model: `{model}` ({source}). Preparing request...", preview

    try:
        messages = _build_messages(upload, prompt, system_prompt)
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "seed": int(seed),
            "top_k": int(top_k),
            "stream": True,
        }
        path = _file_path(upload)
        if path and pathlib.Path(path).suffix.lower() == ".mp4":
            body["media_io_kwargs"] = {"video": {"fps": float(fps)}}

        started = time.time()
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=_headers(),
            json=body,
            stream=True,
            timeout=600,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")

        output = ""
        for chunk in _stream_text(resp):
            output += chunk
            yield output, f"Streaming from `{BASE_URL}`. Elapsed {time.time() - started:.1f}s.", preview
        if not output:
            try:
                data = resp.json()
                output = data["choices"][0]["message"]["content"]
            except Exception:
                output = "(No text returned.)"
        yield output, f"Complete. Elapsed {time.time() - started:.1f}s.", preview
    except Exception as exc:
        yield "", f"Error: {exc}", preview


def reset_ui() -> Tuple[None, str, str, str, str, str]:
    model, source = detect_model(timeout=1.5)
    preview = _request_preview(model, None, DEFAULT_USER, DEFAULT_SYSTEM, 4096, 0.3, 0.3, 1.2, 20, 42, 4.0)
    return None, DEFAULT_USER, DEFAULT_SYSTEM, "", f"Detected model: `{model}` ({source}).", preview


def build_app() -> gr.Blocks:
    model, source = detect_model(timeout=2.0)
    initial_preview = _request_preview(model, None, DEFAULT_USER, DEFAULT_SYSTEM, 4096, 0.3, 0.3, 1.2, 20, 42, 4.0)

    with gr.Blocks(title="cosmos-reason2-8b | NVIDIA NIM") as demo:
        gr.HTML(
            """
            <div class="nv-appbar">
              <div class="nv-logo"><span>NVIDIA</span></div>
              <div class="nv-nav"><span>Explore</span><span>Models</span><span>Blueprints</span><span>GPUs</span><span>Docs</span></div>
            </div>
            <div class="nv-hero">
              <p class="nv-publisher">nvidia</p>
              <div class="nv-title-row">
                <h1>cosmos-reason2-8b</h1>
                <span class="nv-pill-solid">Downloadable</span>
              </div>
              <p class="nv-desc">Vision language model that excels in understanding the physical world using structured reasoning on videos or images.</p>
              <div class="nv-tags">
                <span>Physical AI</span><span>autonomous vehicles</span><span>industrial</span>
                <span>reasoning</span><span>robotics</span><span>smart cities</span>
                <span>video understanding</span><span>vision language model</span>
              </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=6, elem_classes=["input-column"]):
                gr.HTML('<div class="section-heading">Input</div>')
                upload = gr.File(
                    label="Input",
                    file_types=[".mp4", ".jpg", ".jpeg", ".png"],
                    file_count="single",
                )
                user_prompt = gr.Textbox(
                    label="User Prompt",
                    value=DEFAULT_USER,
                    lines=7,
                    max_lines=12,
                    info=(
                        "Your question or task. Aim for up to 400 tokens; max 1000 tokens. "
                        "Enable reasoning with the option below."
                    ),
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=DEFAULT_SYSTEM,
                    lines=4,
                    max_lines=8,
                    info="Defines AI role/rules for session. Max 250 tokens.",
                )
                with gr.Row():
                    reset = gr.Button("Reset")
                    run = gr.Button("Run", variant="primary")

            with gr.Column(scale=5, elem_classes=["output-column"]):
                gr.HTML('<div class="section-heading">Output</div>')
                gr.HTML(
                    f"""
                    <div class="api-panel">
                      <div class="api-top">
                        <div class="api-credit">Using free API <span style="color:#8b8b8b">for development</span></div>
                      </div>
                      <div class="backend-note">
                        Local backend: <code>{BASE_URL}</code><br>
                        Model: <code>{model}</code> ({source})
                      </div>
                    </div>
                    """
                )
                with gr.Accordion("API / code", open=True):
                    code = gr.Code(label="Request body", value=initial_preview, language="json", lines=16)
                with gr.Accordion("Generation settings", open=False):
                    reasoning = gr.Checkbox(label="Append reasoning format instruction", value=False)
                    max_tokens = gr.Slider(1, 4096, value=4096, step=1, label="Max tokens")
                    temperature = gr.Slider(0, 2, value=0.3, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.01, 1, value=0.3, step=0.01, label="Top P")
                    repetition_penalty = gr.Slider(0.1, 2, value=1.2, step=0.05, label="Repetition penalty")
                    top_k = gr.Slider(1, 100, value=20, step=1, label="Top K")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    fps = gr.Slider(0.25, 8, value=4.0, step=0.25, label="Video FPS")

        status = gr.Markdown(f"Detected model: `{model}` ({source}).")
        output = gr.Textbox(label="Output", lines=16)

        race_car = LOCAL_SOURCE_DIR / "Race Car.mp4"
        if race_car.exists():
            gr.Examples(
                examples=[[str(race_car), RACE_CAR_PROMPT, DEFAULT_SYSTEM, 4096, 0.6, 0.3, 1.2, 20, 42, 6.0, False]],
                inputs=[
                    upload, user_prompt, system_prompt, max_tokens, temperature, top_p,
                    repetition_penalty, top_k, seed, fps, reasoning
                ],
                label="View Examples",
            )

        run.click(
            fn=run_inference,
            inputs=[
                upload, user_prompt, system_prompt, max_tokens, temperature, top_p,
                repetition_penalty, top_k, seed, fps, reasoning
            ],
            outputs=[output, status, code],
        )
        reset.click(
            fn=reset_ui,
            inputs=[],
            outputs=[upload, user_prompt, system_prompt, output, status, code],
        )

    return demo


def _detect_host_ip() -> str | None:
    explicit = os.environ.get("PUBLIC_GRADIO_HOST") or os.environ.get("BYO_VIDEO_LOCAL_HOST")
    if explicit:
        return explicit.strip()
    try:
        output = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2
        ).stdout
        for token in output.split():
            if token.count(".") == 3 and not token.startswith("127."):
                return token
    except Exception:
        pass
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        sock.connect(("8.8.8.8", 80))
        host = sock.getsockname()[0]
        sock.close()
        if host and not host.startswith("127."):
            return host
    except Exception:
        pass
    return None


def _public_url(local_url: str | None, share_url: str | None) -> str:
    if share_url:
        return share_url.rstrip("/")
    explicit = os.environ.get("PUBLIC_GRADIO_URL")
    if explicit:
        return explicit.rstrip("/")
    url = local_url or f"http://0.0.0.0:{PORT}"
    host_ip = _detect_host_ip()
    if host_ip and ("localhost" in url or "127.0.0.1" in url or "0.0.0.0" in url):
        url = url.replace("localhost", host_ip).replace("127.0.0.1", host_ip).replace("0.0.0.0", host_ip)
    return url.rstrip("/")


if __name__ == "__main__":
    demo = build_app()
    app, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=SHARE,
        prevent_thread_lock=True,
        theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
        css=BUILD_CSS,
    )
    public_url = _public_url(local_url, share_url)
    print(f"[launch] {public_url}", flush=True)
    try:
        with open("/tmp/gradio_url.txt", "w", encoding="utf-8") as fh:
            fh.write(public_url)
        with open("/tmp/gradio_live.flag", "w", encoding="utf-8") as fh:
            fh.write(public_url)
    except Exception as exc:
        print(f"[launch] Could not write Gradio status files: {exc}", flush=True)
    demo.block_thread()
