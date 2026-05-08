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


def generate(rgb_video, control_video, control_modality, prompt,
             guidance, edge_w, seg_w, depth_w, vis_w, seed):
    if not rgb_video:
        return None, "❌ Upload an RGB video.", None, None
    payload = {
        "prompt": prompt or "",
        "input_video": _b64_file(rgb_video),
        "guidance": float(guidance),
    }
    if control_video and control_modality and control_modality != "none":
        ctrl_b64 = _b64_file(control_video)
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
    submit.click(
        generate,
        [rgb_in, ctrl_in, ctrl_mod, prompt, guidance, edge_w, seg_w, depth_w, vis_w, seed],
        [video_out, status, response, improved],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
