#!/usr/bin/env python3
"""
Gradio frontend for Cosmos Predict 1 (cosmos-predict1-7b-video2world).

UX schema replicated from build.nvidia.com/nvidia/cosmos-predict1-5b
(per /tmp/cosmos_predict_ux.json from 2026-05-08 sprint Phase A).

The build.nvidia.com playground exposes a "World Creation Mode" radio toggle
between Video-to-World and Image-to-World. We replicate that as the primary
control. Inputs adapt: Video-to-World takes an mp4; Image-to-World takes a
single still. Both produce a generated mp4 video.

NIM API: POST http://localhost:8000/v1/infer
  payload: {"prompt", "video"|"image" (base64), "seed", "guidance_scale",
            "steps", "prompt_upsampling", "video_params": {...}}
  response: {"b64_video": "<base64 mp4>", "seed": <int>}

Required env:
  NIM_HOST           default localhost
  NIM_PORT           default 8000
  NIM_ALLOW_URL_INPUT  on the NIM container — enables URL inputs (we use base64)

Runtime: serves on 0.0.0.0:7860 (Gradio default). frpc tunnel exposes a public URL.
"""

import base64
import json
import os
import shutil
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

import gradio as gr  # type: ignore

NIM_HOST = os.environ.get("NIM_HOST", "localhost")
NIM_PORT = int(os.environ.get("NIM_PORT", "8000"))
INFER_URL = f"http://{NIM_HOST}:{NIM_PORT}/v1/infer"
NIM_BASE_URL = f"http://{NIM_HOST}:{NIM_PORT}/v1"


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


def _preprocess_video(src_path: str) -> tuple[str, str]:
    # Cosmos Predict's TorchScript autoencoder expects exactly 5s × 1280×704 × 24fps.
    # Mismatched input shape fails inside the encoder forward pass (HTTP 500
    # observed 2026-05-08 on a 0:45 1920×1080 clip). Normalize before send.
    if not shutil.which("ffmpeg"):
        return src_path, "⚠ ffmpeg missing — sending video as-is (autoencoder may reject)"
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
        return out_path, "✂ 121 frames @ 24fps · 1280×704 · h264 (Cosmos Predict spec)"
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", errors="replace")[-200:]
        return src_path, f"⚠ ffmpeg failed (code {e.returncode}); sending as-is. {tail}"
    except Exception as e:
        return src_path, f"⚠ ffmpeg error: {e}; sending as-is"


def generate(world_mode: str, video_file, image_file, prompt: str,
             guidance_scale: float, steps: int, seed: int,
             prompt_upsampling: bool):
    prep_info = ""
    if world_mode == "Video-to-World":
        if not video_file:
            return None, "❌ Upload a video for Video-to-World mode.", None
        media_field = "video"
        prep_path, prep_info = _preprocess_video(video_file)
        media_b64 = _b64_file(prep_path)
    else:
        if not image_file:
            return None, "❌ Upload an image for Image-to-World mode.", None
        media_field = "image"
        media_b64 = _b64_file(image_file)

    payload = {
        "prompt": prompt or "",
        media_field: media_b64,
        "guidance_scale": float(guidance_scale),
        "steps": int(steps),
        "video_params": {
            "height": 704,
            "width": 1280,
            "frames_count": 121,
            "frames_per_sec": 24,
        },
    }
    if seed is not None and seed >= 0:
        payload["seed"] = int(seed)
    # NOTE: build.nvidia.com playground exposes a "prompt_upsampling" toggle but
    # the self-hosted cosmos-predict1-7b-video2world NIM rejects this field
    # ("extra_forbidden" 422 — observed 2026-05-08). Schema for the hosted API
    # differs from the self-host. Field omitted; the NIM uses its built-in
    # default (true). The UI checkbox is preserved for future-compat with NIMs
    # that DO accept it; today its value is informational only.
    _ = prompt_upsampling  # intentionally unused — see note above

    req = urllib.request.Request(
        INFER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
            data = json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500] if hasattr(e, "read") else str(e)
        return None, f"❌ HTTP {e.code}: {err_body}", json.dumps(payload, indent=2)[:2000]
    except Exception as e:
        return None, f"❌ Request failed: {str(e)[:300]}", json.dumps(payload, indent=2)[:2000]

    b64_video = data.get("b64_video")
    if not b64_video:
        return None, f"❌ No b64_video in response. Response keys: {list(data.keys())}", json.dumps(data)[:1000]

    out_path = "/tmp/cosmos_predict_output.mp4"
    Path(out_path).write_bytes(base64.b64decode(b64_video))
    info = f"✅ Generated {len(b64_video)//1000} KB mp4 · seed={data.get('seed', 'n/a')}"
    if prep_info:
        info = f"{prep_info}\n\n{info}"
    return out_path, info, json.dumps({k: v for k, v in data.items() if k != "b64_video"}, indent=2)


def update_inputs(mode):
    if mode == "Video-to-World":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


_gpu_name, _free_mib = _gpu_info()

with gr.Blocks(title="Cosmos Predict 1 (Video2World)") as demo:
    gr.Markdown(
        "# Cosmos Predict 1 — Video2World\n"
        f"**Load:** on demand &nbsp;·&nbsp; **GPU:** {_gpu_name} &nbsp;·&nbsp; "
        f"**VRAM free:** {_free_mib:,} MiB &nbsp;·&nbsp; "
        f"**Backend:** NIM local Docker (`{NIM_BASE_URL}`)\n\n"
        "Generate future frames of a physics-aware world state from a video or image input. "
        "Replicates the build.nvidia.com playground UX for the self-hosted NIM."
    )
    with gr.Row():
        with gr.Column(scale=1):
            world_mode = gr.Radio(
                choices=["Video-to-World", "Image-to-World"],
                value="Video-to-World",
                label="World Creation Mode",
                info="Drive future-frame generation from a video (continuation) or image (first-frame)."
            )
            video_in = gr.Video(label="Input Video (mp4)", visible=True)
            image_in = gr.Image(label="Input Image", type="filepath", visible=False)
            prompt = gr.Textbox(
                label="Prompt",
                value="A first person view from a robot working in a chemical plant.",
                lines=3,
                max_lines=5,
            )
            with gr.Accordion("Generation parameters", open=False):
                guidance = gr.Slider(1.0, 10.0, value=7.0, step=0.5, label="Guidance scale (CFG)")
                steps = gr.Slider(1, 50, value=35, step=1, label="Steps")
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                upsample = gr.Checkbox(value=True, label="Prompt upsampling (auto-rewrite for quality)")
            submit = gr.Button("Generate New World", variant="primary")
        with gr.Column(scale=1):
            video_out = gr.Video(label="Generated Future World", interactive=False)
            status = gr.Markdown()
            response = gr.Code(label="Response (JSON, video stripped)", language="json")
    world_mode.change(update_inputs, world_mode, [video_in, image_in])
    submit.click(generate,
                 [world_mode, video_in, image_in, prompt, guidance, steps, seed, upsample],
                 [video_out, status, response])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
