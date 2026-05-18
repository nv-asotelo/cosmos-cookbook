"""Cosmos3 Omni Generator (with upload + telemetry).

A wrapper for `cosmos3.ray.gradio` that:
  * adds a real image upload widget (drag-and-drop) for i2v / i2i,
  * adds a real video upload widget for v2v / robotics-policy modes,
  * surfaces a top status bar with VRAM free / SSD free / backend name,
  * keeps everything else (generate(), components, examples) reused unchanged
    from the upstream cosmos3 Gradio.

Upload routing into `vision_path`:
  - If a video is uploaded, its path goes into `vision_path` (takes precedence).
  - Otherwise, if an image is uploaded, its path goes into `vision_path`.
  - If both are cleared, `vision_path` is removed and the example's default
    is restored on the next preset reload.

Telemetry source-of-truth:
  - GPU/VRAM: `nvidia-smi --query-gpu` shelled out locally (this script runs
    on the same host as Ray Serve, so local probes describe the inference box).
  - SSD: `shutil.disk_usage("/")`.
  - Backend: detected from listening ports + process command lines.
    Priority: cosmos3_native (Ray Serve on :8000) > vLLM > NIM > Gradio-only.

Telemetry refresh: 5s tick via gr.Timer.

Run on the same host as Ray Serve, with the cosmos3 venv:
    cd ~/cosmos3
    uv run --no-sync python ~/cosmos3_upload_gradio.py \
        --host 0.0.0.0 --port 8080 \
        --server-host localhost --server-port 8000 \
        --server-output-dir outputs/ray_serve
"""

import json
import os
import re
import shutil
import subprocess
from functools import partial
from pathlib import Path

import gradio as gr

from cosmos3.args import OmniSampleOverrides
from cosmos3.common.args import tyro_cli
from cosmos3.ray.gradio import (
    Args,
    COMPONENTS,
    EXCLUDE_FIELDS,
    INPUTS_DIR,
    build_components,
    generate,
    get_info,
    load_input,
)

NVIDIA_GREEN = "#76B900"
NVIDIA_DARK = "#1A1A1A"
SERVER_MEDIA_EXTENSIONS = {".mp4", ".mov", ".jpg", ".jpeg", ".png", ".webp"}


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


def _gpu_probe() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().splitlines()
        if not out:
            return {}
        # first GPU only — the cosmos3 deployment is single-GPU
        parts = [x.strip() for x in out[0].split(",")]
        name, used, free, total, util, temp, power = parts[:7]
        return {
            "name": name,
            "mem_used_gib": round(int(used) / 1024, 1),
            "mem_free_gib": round(int(free) / 1024, 1),
            "mem_total_gib": round(int(total) / 1024, 1),
            "util_pct": int(util),
            "temp_c": int(temp),
            "power_w": float(power),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _disk_probe(path: str = "/") -> dict:
    try:
        u = shutil.disk_usage(path)
        return {
            "path": path,
            "total_gib": round(u.total / 2**30, 1),
            "used_gib": round(u.used / 2**30, 1),
            "free_gib": round(u.free / 2**30, 1),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


_CHECKPOINT_RE = re.compile(r"--checkpoint-path[=\s]+([^\s]+)")


def _backend_probe() -> dict:
    """Detect which model-serving backend is bound on this host plus the
    actually loaded checkpoint name (standing order: never hardcode model id).

    Order: Ray Serve (cosmos3_native) > vLLM > NIM (Triton) > unknown.
    """
    try:
        ps = subprocess.check_output(
            ["ps", "-Ao", "args"], text=True, timeout=4, errors="replace"
        )
    except Exception as exc:
        return {"name": "unknown", "checkpoint": None, "error": f"{type(exc).__name__}: {exc}"}
    ps_l = ps.lower()
    checkpoint = None
    for line in ps.splitlines():
        if "cosmos3.ray.serve" in line:
            m = _CHECKPOINT_RE.search(line)
            if m:
                checkpoint = m.group(1)
                break
    if "cosmos3.ray.serve" in ps_l or "ray::proxyactor" in ps_l:
        return {
            "name": "Ray Serve (cosmos3_native)",
            "port": 8000,
            "framework": "cosmos3",
            "checkpoint": checkpoint,
        }
    if "vllm" in ps_l:
        return {"name": "vLLM", "port": 8000, "framework": "vllm", "checkpoint": checkpoint}
    if "triton" in ps_l or "nim_llm" in ps_l or "/opt/nim" in ps_l:
        return {"name": "NIM (Triton)", "port": 8000, "framework": "nim", "checkpoint": checkpoint}
    return {"name": "unknown / not detected", "port": None, "framework": "unknown", "checkpoint": checkpoint}


def _telemetry_markdown() -> str:
    gpu = _gpu_probe()
    disk = _disk_probe("/")
    backend = _backend_probe()
    gpu_str = (
        f"**{gpu.get('name','?')}** · VRAM **{gpu.get('mem_free_gib','?')} GB free** / "
        f"{gpu.get('mem_total_gib','?')} GB · util {gpu.get('util_pct','?')}% · "
        f"{gpu.get('temp_c','?')}°C · {gpu.get('power_w','?')} W"
        if "error" not in gpu else f"GPU probe error: {gpu['error']}"
    )
    disk_str = (
        f"SSD **{disk.get('free_gib','?')} GB free** / {disk.get('total_gib','?')} GB on `{disk.get('path','/')}`"
        if "error" not in disk else f"Disk probe error: {disk['error']}"
    )
    backend_str = f"Backend: **{backend['name']}**"
    checkpoint_str = (
        f"Model: <b style='color:{NVIDIA_GREEN}'>{backend.get('checkpoint')}</b>"
        if backend.get("checkpoint")
        else "Model: <i style='color:#aaa'>detecting…</i>"
    )
    return (
        f"<div style='padding:8px 12px;background:{NVIDIA_DARK};color:#fff;"
        f"border-left:4px solid {NVIDIA_GREEN};font-size:13px;line-height:1.6'>"
        f"<b style='color:{NVIDIA_GREEN}'>HOST TELEMETRY</b> &nbsp;&nbsp; "
        f"{checkpoint_str} &nbsp;·&nbsp; {backend_str} &nbsp;·&nbsp; {gpu_str} &nbsp;·&nbsp; {disk_str}"
        f"</div>"
    )


def update_extra_with_vision(image_path, video_path, server_source, current_json):
    """Inject the uploaded media path into extra_input.vision_path.

    Server source takes precedence over uploads, then video, then image.
    Clearing all removes vision_path.
    """
    try:
        data = json.loads(current_json) if current_json else {}
    except json.JSONDecodeError:
        data = {}
    media_path = (server_source or "").strip() or video_path or image_path
    if media_path:
        data["vision_path"] = str(media_path)
    else:
        data.pop("vision_path", None)
    return json.dumps(data, indent=2)


def ui_builder(args: Args) -> gr.Blocks:
    info = get_info(args)
    available_models = info["models"]
    if len(available_models) == 0:
        raise ValueError("No models available")

    default_example = "t2i"
    examples: dict[str, Path] = {}
    for p in INPUTS_DIR.rglob("*.json"):
        if "internal" in p.parts:
            continue
        if p.stem in examples:
            raise ValueError(f"Duplicate example file: {p}")
        examples[p.stem] = p
    if default_example not in examples:
        default_example = next(iter(sorted(examples.keys())), "")

    with gr.Blocks(
        title="Cosmos3 Omni Generator (with upload + telemetry)",
        css=f"""
        .telemetry-strip {{ margin-bottom: 8px; }}
        .gradio-container .gr-button.primary {{ background: {NVIDIA_GREEN} !important; }}
        """,
    ) as ui:
        telemetry_md = gr.Markdown(_telemetry_markdown(), elem_classes=["telemetry-strip"])
        gr.Markdown("# Cosmos3 Omni Generator")
        gr.Markdown(
            "Upload an **image** (i2v / i2i) or a **video** (v2v / robotics policy) "
            "to drive vision-conditioned generation. The uploaded file path is "
            "injected into **Extra Arguments → `vision_path`** automatically. "
            "Video uploads take precedence; clear both to revert to the example default."
        )
        with gr.Accordion("Environment", open=False):
            gr.JSON(value=info["environment"])

        example_dropdown = gr.Dropdown(
            choices=["", *sorted(examples.keys())],
            value=default_example,
            label="Input preset",
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_input = gr.Dropdown(
                    value=available_models[0],
                    choices=available_models,
                    label="Model",
                )
                generate_btn = gr.Button("Generate", variant="primary")

                image_upload = gr.Image(
                    type="filepath",
                    label="Conditioning image (drag-and-drop or click to upload)",
                    sources=["upload", "clipboard"],
                    height=200,
                )
                video_upload = gr.Video(
                    label="Conditioning video (drag-and-drop or click to upload, for v2v / robotics-policy)",
                    sources=["upload"],
                    height=200,
                )
                with gr.Accordion("Server media source", open=False):
                    server_example = gr.Dropdown(
                        label="Server example",
                        choices=_server_example_choices(),
                        value="",
                        info="Choose media already staged on this machine.",
                    )
                    server_source = gr.Textbox(
                        label="Server media URL or local path",
                        value="",
                        placeholder="/tmp/source.mp4",
                        info="Takes precedence over uploads and avoids browser upload.",
                    )

                components = build_components(OmniSampleOverrides, COMPONENTS)

                with gr.Accordion("Extra Arguments", open=False):
                    extra_json = OmniSampleOverrides(name="").model_dump_json(
                        indent=2,
                        exclude={*COMPONENTS, *EXCLUDE_FIELDS},
                    )
                    extra_input = gr.Code(
                        extra_json,
                        language="json",
                        lines=10,
                    )

            with gr.Column(scale=1):
                media_output = gr.Gallery(label="Media", allow_preview=True)

                with gr.Accordion("Request", open=False):
                    request_output = gr.JSON()

                with gr.Accordion("Response", open=False):
                    response_output = gr.JSON()

        load_input_kwargs = dict(
            fn=partial(load_input, examples=examples),
            inputs=[example_dropdown],
            outputs=[model_input, *components.values(), extra_input],
        )
        example_dropdown.change(**load_input_kwargs)
        ui.load(**load_input_kwargs)

        image_upload.change(
            fn=update_extra_with_vision,
            inputs=[image_upload, video_upload, server_source, extra_input],
            outputs=[extra_input],
        )
        video_upload.change(
            fn=update_extra_with_vision,
            inputs=[image_upload, video_upload, server_source, extra_input],
            outputs=[extra_input],
        )
        server_source.change(
            fn=update_extra_with_vision,
            inputs=[image_upload, video_upload, server_source, extra_input],
            outputs=[extra_input],
        )
        server_example.change(
            fn=lambda value, image_path, video_path, current_json: (
                value or "",
                update_extra_with_vision(image_path, video_path, value or "", current_json),
            ),
            inputs=[server_example, image_upload, video_upload, extra_input],
            outputs=[server_source, extra_input],
        )

        generate_btn.click(
            fn=partial(generate, args=args),
            inputs=[model_input, *components.values(), extra_input],
            outputs=[media_output, request_output, response_output],
        )

        telemetry_tick = gr.Timer(5.0)
        telemetry_tick.tick(fn=_telemetry_markdown, outputs=[telemetry_md])

    return ui


def main():
    args = tyro_cli(Args, description=__doc__)
    ui = ui_builder(args)
    ui.queue(default_concurrency_limit=1, max_size=8)
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=True,
        allowed_paths=[str(args.server_output_dir), "/tmp"],
    )


if __name__ == "__main__":
    main()
