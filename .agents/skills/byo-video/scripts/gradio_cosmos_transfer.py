#!/usr/bin/env python3
"""
Gradio frontend for Cosmos Transfer 1 (7B) and Transfer 2.5 (2B).

UX schema replicated from build.nvidia.com playground for both models
(per /tmp/cosmos_transfer_ux.json from 2026-05-08 sprint Phase A research).

This is a SKELETON — Cosmos Transfer NIM images are NGC-DENIED on the
current API key (verified 2026-05-08 against every tag/naming variant).
Same entitlement-gap class as cosmos-reason2-32b. The Gradio is ready to
deploy the moment the NIM team grants allowlist access.

UX shape (faithful replica):
  - RGB video upload (source)
  - Optional control-modality video upload + modality selector
  - Prompt textbox (free-form, ≤ 300 words per Model Card)
  - Sliders: guidance, per-modality control_weight (edge/seg/depth/vis),
    seed (int, -1 = random)
  - Output: generated mp4 video + improved-prompt readback (NIM auto-rewrite)

NIM API (inferred from cosmos-transfer1 cookbook recipe configs — confirm
against live NIM when entitlement clears):
  POST http://localhost:8000/v1/infer
  payload: {prompt, input_video_path|input_video, guidance,
            edge:{control_weight,control_path}, seg:{control_weight,control_path,mask_path},
            depth:{control_weight,control_path}, vis:{control_weight}}
  response: {b64_video|video, improved_prompt}

Required env:
  NIM_HOST           default localhost
  NIM_PORT           default 8000
  TRANSFER_VARIANT   "1-7b" (default) or "2-5-2b" — switches the served
                     model id (nvidia/cosmos-transfer1-7b vs
                     nvidia/cosmos-transfer2-5-2b) and tags the title
"""

import base64
import json
import os
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

import gradio as gr  # type: ignore

NIM_HOST = os.environ.get("NIM_HOST", "localhost")
NIM_PORT = int(os.environ.get("NIM_PORT", "8000"))
INFER_URL = f"http://{NIM_HOST}:{NIM_PORT}/v1/infer"
NIM_BASE_URL = f"http://{NIM_HOST}:{NIM_PORT}/v1"

VARIANT = os.environ.get("TRANSFER_VARIANT", "1-7b")
_VARIANT_LABELS = {
    "1-7b":   ("Cosmos Transfer 1 — 7B",   "nvidia/cosmos-transfer1-7b"),
    "2-5-2b": ("Cosmos Transfer 2.5 — 2B", "nvidia/cosmos-transfer2-5-2b"),
}
_TITLE, _MODEL_ID = _VARIANT_LABELS.get(VARIANT, _VARIANT_LABELS["1-7b"])


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


SERVER_MEDIA_EXTENSIONS = {".mp4", ".mov", ".jpg", ".jpeg", ".png", ".webp"}


def _is_url_source(value: str) -> bool:
    return str(value or "").strip().lower().startswith(("http://", "https://"))


def _is_file_url_source(value: str) -> bool:
    return str(value or "").strip().lower().startswith("file://")


def _path_from_file_url(value: str) -> str:
    import urllib.parse

    parsed = urllib.parse.urlparse(str(value or ""))
    return urllib.parse.unquote(parsed.path or "")


def _source_suffix(value: str) -> str:
    text = str(value or "").strip()
    if _is_url_source(text) or _is_file_url_source(text):
        import urllib.parse

        text = urllib.parse.urlparse(text).path
    return Path(text).suffix.lower()


def _resolve_server_source(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    suffix = _source_suffix(text)
    if suffix not in SERVER_MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported server media extension: {suffix or '<none>'}")
    if _is_url_source(text):
        return text
    if _is_file_url_source(text):
        text = _path_from_file_url(text)
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Server media path does not exist: {path}")
    return str(path)


def _b64_source(source: str) -> str:
    if _is_url_source(source):
        with urllib.request.urlopen(source, timeout=120) as response:
            return base64.b64encode(response.read()).decode("ascii")
    return _b64_file(source)


def _server_example_choices() -> list[tuple[str, str]]:
    roots = os.environ.get("BYO_VIDEO_SERVER_EXAMPLE_DIRS", "/tmp/nvidia-build-reason-vite/public/examples:/tmp/examples")
    choices: list[tuple[str, str]] = [("None", "")]
    for raw_root in [part for part in roots.split(":") if part.strip()]:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in SERVER_MEDIA_EXTENSIONS:
                choices.append((f"{root.name}/{path.name}", str(path)))
    return choices[:40]


def generate(rgb_video, control_video, rgb_server_source, control_server_source, control_modality, prompt,
             guidance, edge_w, seg_w, depth_w, vis_w, seed):
    try:
        rgb_source = _resolve_server_source(rgb_server_source) or rgb_video
        ctrl_source = _resolve_server_source(control_server_source) or control_video
    except Exception as exc:
        return None, f"❌ Server media source error: {exc}", None, None
    if not rgb_source:
        return None, "❌ Upload an RGB video.", None, None
    payload = {
        "prompt": prompt or "",
        "input_video": _b64_source(rgb_source),
        "guidance": float(guidance),
    }
    if ctrl_source and control_modality and control_modality != "none":
        ctrl_b64 = _b64_source(ctrl_source)
        # Per docs, each modality is its own block with control_weight + control_path.
        # control_path here is the b64 payload (NIM may accept either path or b64).
        payload[control_modality] = {
            "control_weight": float({"edge": edge_w, "seg": seg_w,
                                      "depth": depth_w, "vis": vis_w}.get(control_modality, 1.0)),
            "control_path": ctrl_b64,
        }
    if seed is not None and seed >= 0:
        payload["seed"] = int(seed)

    req = urllib.request.Request(
        INFER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500] if hasattr(e, "read") else str(e)
        return None, f"❌ HTTP {e.code}: {body}", json.dumps(payload, indent=2)[:1500], None
    except Exception as e:
        return None, f"❌ Request failed: {str(e)[:300]}", json.dumps(payload, indent=2)[:1500], None

    b64v = data.get("b64_video") or data.get("video")
    if not b64v:
        return None, f"❌ No video in response. Keys: {list(data.keys())}", json.dumps(data)[:1000], None

    out_path = "/tmp/cosmos_transfer_output.mp4"
    Path(out_path).write_bytes(base64.b64decode(b64v))
    improved = data.get("improved_prompt", "")
    info = f"✅ Generated mp4 · variant={VARIANT}"
    return out_path, info, json.dumps({k: v for k, v in data.items()
                                       if k not in ("b64_video", "video")}, indent=2), improved


_gpu_name, _free_mib = _gpu_info()

with gr.Blocks(title=_TITLE) as demo:
    gr.Markdown(
        f"# {_TITLE}\n"
        f"**Load:** on demand &nbsp;·&nbsp; **GPU:** {_gpu_name} &nbsp;·&nbsp; "
        f"**VRAM free:** {_free_mib:,} MiB &nbsp;·&nbsp; "
        f"**Backend:** NIM local Docker (`{NIM_BASE_URL}`)\n\n"
        "Generate a physics-aware video world state from an RGB source video, "
        "optionally guided by a spatial control modality (edge/seg/depth/blur). "
        "Replicates the build.nvidia.com playground UX for the self-hosted NIM."
    )
    with gr.Row():
        with gr.Column(scale=1):
            rgb_in = gr.Video(label="RGB source video (mp4)")
            ctrl_in = gr.Video(label="Control video (optional)")
            with gr.Accordion("Server media source", open=False):
                rgb_example = gr.Dropdown(
                    label="RGB server example",
                    choices=_server_example_choices(),
                    value="",
                    info="Choose media already staged on this machine.",
                )
                rgb_server = gr.Textbox(
                    label="RGB URL or local path",
                    value="",
                    placeholder="/tmp/source.mp4",
                    info="Server source wins over uploaded media and avoids browser upload.",
                )
                ctrl_example = gr.Dropdown(
                    label="Control server example",
                    choices=_server_example_choices(),
                    value="",
                )
                ctrl_server = gr.Textbox(
                    label="Control URL or local path",
                    value="",
                    placeholder="/tmp/control.mp4",
                )
            ctrl_mod = gr.Radio(
                choices=["none", "edge", "seg", "depth", "vis"],
                value="none",
                label="Control modality",
                info="Edge (Canny) · Seg (mask) · Depth · Vis (blurred RGB). "
                     "All control inputs must share spatio-temporal dims with the RGB source.",
            )
            prompt = gr.Textbox(
                label="Prompt", lines=3, max_lines=6,
                value="The camera follows several vehicles driving through a quiet "
                      "suburban neighborhood on a rainy day...",
            )
            with gr.Accordion("Generation parameters", open=False):
                guidance = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Guidance scale")
                edge_w = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="edge.control_weight")
                seg_w  = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="seg.control_weight")
                depth_w= gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="depth.control_weight")
                vis_w  = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="vis.control_weight")
                seed   = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
            submit = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            video_out = gr.Video(label="Generated video", interactive=False)
            status = gr.Markdown()
            improved = gr.Textbox(label="AI-improved prompt", lines=3, interactive=False)
            response = gr.Code(label="Response (JSON, video stripped)", language="json")
    rgb_example.change(lambda value: value or "", rgb_example, rgb_server)
    ctrl_example.change(lambda value: value or "", ctrl_example, ctrl_server)
    submit.click(
        generate,
        [rgb_in, ctrl_in, rgb_server, ctrl_server, ctrl_mod, prompt, guidance, edge_w, seg_w, depth_w, vis_w, seed],
        [video_out, status, response, improved],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
