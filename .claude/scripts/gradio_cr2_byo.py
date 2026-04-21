#!/usr/bin/env python3
"""
Cosmos Reason2 BYO-Video Gradio Demo
Version: 2026-04-20
Canonical source: ~/.claude/scripts/gradio_cr2_byo.py
Deploy to instances via byo_video_setup.py (reads this file).

UI best practices from vlm-race sprint:
  - Video clip characterization (resolution, FPS, duration on upload)
  - Demo prompt dropdown with presets
  - Advanced settings accordion (fps, max_pixels, max_tokens, system prompt)
  - Streaming output for live token display
  - BF16 dtype (was FP16)
  - qwen_vl_utils.process_vision_info for proper max_pixels resize (torchvision fallback)
"""
import os, sys, json, time, threading, warnings
warnings.filterwarnings("ignore")

try:
    import torch
except ImportError:
    print("[ERROR] torch not installed."); sys.exit(1)

try:
    import gradio as gr
except ImportError:
    print("[ERROR] gradio not installed. Run: pip install gradio"); sys.exit(1)

try:
    import av as _av_module
    _AV_OK = True
except ImportError:
    _av_module = None
    _AV_OK = False

try:
    import qwen_vl_utils
except ImportError:
    print("[ERROR] qwen_vl_utils not installed (should be in cosmos-reason2 venv)."); sys.exit(1)

try:
    import transformers
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        TextIteratorStreamer,
    )
except ImportError as e:
    print(f"[ERROR] transformers not installed: {e}"); sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
HOME       = os.path.expanduser("~")
MODEL_DIR  = os.environ.get("MODEL_DIR",  f"{HOME}/cosmos-reason2/models/Cosmos-Reason2-2B")
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/Cosmos-Reason2-2B")
OUT_FILE   = os.environ.get("OUT_FILE",   "/tmp/byo_video_reason2_results.json")
PORT       = int(os.environ.get("GRADIO_PORT", "7860"))
SHARE      = os.environ.get("GRADIO_SHARE", "true").lower() != "false"
LOW_VRAM   = os.environ.get("LOW_VRAM", "false").lower() == "true"

# VRAM tier from setup script (env vars override low-vram binary check)
DEFAULT_FPS        = int(os.environ.get("GRADIO_FPS", "4" if LOW_VRAM else "8"))
DEFAULT_MAX_PIXELS = int(os.environ.get("GRADIO_MAX_PIXELS", str(128*(32**2) if LOW_VRAM else 4096*(32**2))))
DEFAULT_MAX_TOKENS = 512

# Auto-cap constants — keep inference under TARGET_INFER_S
# PREFILL_TPS: empirical prefill throughput in tokens/s for this GPU tier.
# Passed by setup script via GRADIO_PREFILL_TPS. Conservative default = 23 (RTX PRO 6000).
PREFILL_TPS      = float(os.environ.get("GRADIO_PREFILL_TPS", "23"))
TARGET_INFER_S   = 55.0    # target max inference seconds (budget under 60s)
TEXT_TOKENS      = 50      # approximate text tokens (empirical with qwen_vl_utils path)
_EMPIRICAL_VPF   = 128     # tokens per frame at _BASELINE_PX (calibrated: RTX PRO 6000 Blackwell)
_BASELINE_PX     = 524288  # reference max_pixels for _EMPIRICAL_VPF calibration
_PIXEL_TIERS     = [131072, 262144, 524288, 1048576, 2097152]


def _est_tokens(n_frames: int, max_pixels: int) -> int:
    # Empirical: 8 frames × 524288 px → 1055 tokens; linear in both dimensions.
    return int(n_frames * max_pixels / _BASELINE_PX * _EMPIRICAL_VPF) + TEXT_TOKENS


def _auto_cap(n_frames: int, current_max_pixels: int):
    """Return (capped_pixels, est_s). Steps down pixel tiers until est < TARGET_INFER_S."""
    for px in reversed(_PIXEL_TIERS):
        if px > current_max_pixels:
            continue
        est_s = _est_tokens(n_frames, px) / PREFILL_TPS
        if est_s <= TARGET_INFER_S:
            return px, est_s
    # Below all tiers — use minimum, report expected time
    px = _PIXEL_TIERS[0]
    return px, _est_tokens(n_frames, px) / PREFILL_TPS

DEFAULT_SYSTEM = "You are a helpful assistant that analyzes videos."
DEFAULT_PROMPT = "Describe what is happening in this video. What are the key actions, objects, and events?"

DEMO_PROMPTS = [
    ("General description",    "Describe what is happening in this video. What are the key actions, objects, and events?"),
    ("Safety analysis",        "Identify any safety hazards, risks, or unsafe behaviors visible in this video. Be specific."),
    ("Non-expert summary",     "Summarize this video in plain language for someone with no domain expertise. Focus on what is happening and why it matters."),
    ("Action recognition",     "List every distinct action or motion performed in this video, in the order they occur."),
    ("Environment & setting",  "Describe the environment: location, lighting, weather, surface type, and any background context."),
    ("Object inventory",       "List all objects, equipment, and people visible. Note their state (moving/stationary, interacting/isolated)."),
    ("Anomaly detection",      "Identify anything unusual, unexpected, or out of place in this video compared to normal operation."),
    ("Temporal summary",       "Break this video into time segments and describe what changes in each segment."),
]

# ── Video metadata ─────────────────────────────────────────────────────────────
def get_video_meta(path: str):
    """Return (width, height, fps, duration_s) using PyAV if available."""
    class Meta:
        width = height = fps = duration_s = 0
    m = Meta()
    if not _AV_OK or not path:
        return m
    try:
        container = _av_module.open(path)
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream:
            m.width  = stream.width
            m.height = stream.height
            m.fps    = float(stream.average_rate) if stream.average_rate else 0.0
            m.duration_s = float(container.duration) / 1_000_000 if container.duration else 0.0
        container.close()
    except Exception:
        pass
    return m


def get_free_vram_mib() -> int:
    try:
        return torch.cuda.mem_get_info()[0] // (1024 * 1024)
    except Exception:
        return 0


# ── Model load ────────────────────────────────────────────────────────────────
print(f"[demo] Loading model from {MODEL_DIR} ...", flush=True)
t0 = time.time()
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
load_time = time.time() - t0

gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
free_vram = get_free_vram_mib()
print(f"[demo] Model ready in {load_time:.1f}s  |  GPU: {gpu_name}  |  VRAM free: {free_vram:,} MiB", flush=True)
if os.environ.get("GRADIO_FPS") is not None:
    tier = os.environ.get("GRADIO_FPS", "?")  # tier label not passed separately; use fps as proxy
    print(f"[demo] VRAM tier: {tier}fps | fps={DEFAULT_FPS}, max_pixels={DEFAULT_MAX_PIXELS:,}", flush=True)
elif LOW_VRAM:
    print(f"[demo] LOW_VRAM mode: fps={DEFAULT_FPS}, max_pixels={DEFAULT_MAX_PIXELS}", flush=True)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(video_path, user_prompt, system_prompt, fps, max_pixels, max_new_tokens):
    if video_path is None:
        yield "Upload a video first.", "*—*"
        return
    if not user_prompt.strip():
        user_prompt = DEFAULT_PROMPT

    fps            = int(fps)
    max_pixels     = int(max_pixels)
    max_new_tokens = int(max_new_tokens)

    # Auto-cap max_pixels at inference time (race-condition-safe — UI slider update may lag)
    m = get_video_meta(video_path)
    if m.duration_s > 0:
        n_frames = max(1, int(m.duration_s * fps))
        capped_px, est_s = _auto_cap(n_frames, max_pixels)
        if capped_px < max_pixels:
            print(f"[auto-cap] {max_pixels:,} → {capped_px:,} px ({n_frames} frames, ~{est_s:.0f}s est)", flush=True)
            max_pixels = capped_px
            yield f"⏳ Auto-cap: resolution reduced to {capped_px:,} px/frame (~{est_s:.0f}s est)…", "*—*"
        else:
            yield f"⏳ Preprocessing {n_frames} frames at {max_pixels:,} px/frame…", "*—*"
    else:
        yield "⏳ Preprocessing video frames…", "*—*"

    # fps and max_pixels embedded in the video dict so qwen_vl_utils.fetch_video
    # applies smart_resize before tokenization (bypassing the transformers pipeline
    # which ignores max_pixels). torchvision fallback handles missing FFmpeg.
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [
            {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
            {"type": "text",  "text": user_prompt},
        ]},
    ]

    t1 = time.time()
    # Step 1: text prompt (no tokenization — vision handled separately)
    text_prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Step 2: load and resize video frames via qwen_vl_utils (respects max_pixels).
    # fps=[sample_fps] from video_kwargs doubles token count (confirmed empirically) — omit it.
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)

    # Step 3: tokenize text + visual together
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    tokens_in = inputs["input_ids"].shape[1]

    yield f"⚡ Prefilling {tokens_in:,} tokens…", "*—*"

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=180.0,  # 8B models need 60-120s prefill before first token
    )
    gen_thread = threading.Thread(
        target=model.generate,
        kwargs={
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.05,
            "streamer": streamer,
        },
        daemon=True,
    )
    gen_thread.start()

    parts    = []
    ttft_s   = None
    t_gen    = time.time()

    for chunk in streamer:
        if chunk:
            if ttft_s is None:
                ttft_s = time.time() - t_gen
            parts.append(chunk)
            yield "".join(parts), "*generating…*"

    gen_thread.join(timeout=10)
    infer_time = time.time() - t1
    response   = "".join(parts)
    tokens_out = len(processor.tokenizer.encode(response, add_special_tokens=False))

    result = {
        "model":        MODEL_NAME,
        "prompt":       user_prompt,
        "response":     response,
        "load_time_s":  round(load_time, 1),
        "infer_time_s": round(infer_time, 1),
        "ttft_s":       round(ttft_s, 2) if ttft_s else None,
        "tokens_in":    tokens_in,
        "tokens_out":   tokens_out,
        "fps":          fps,
        "status":       "success",
    }
    try:
        with open(OUT_FILE, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    metrics = "\n\n".join([
        f"**Inference:** {infer_time:.1f}s",
        f"**TTFT:** {ttft_s:.2f}s" if ttft_s else "",
        f"**Tokens in:** {tokens_in:,}  |  **Tokens out:** {tokens_out:,}",
        f"**Model load:** {load_time:.1f}s",
        f"*Results saved → {OUT_FILE}*",
    ]).strip()

    print(f"[done] {infer_time:.1f}s · {tokens_out} tok out · ttft={ttft_s:.2f}s", flush=True)
    yield response, metrics


# ── Gradio UI ─────────────────────────────────────────────────────────────────
LOW_VRAM_NOTICE = "\n> ⚠ **Low-VRAM mode** — fps=1, reduced resolution. Use clips under 60s." if LOW_VRAM else ""

with gr.Blocks(
    title="Cosmos Reason2 — BYO Video Demo",
    theme=gr.themes.Base(primary_hue="green", font=gr.themes.GoogleFont("Inter")),
) as demo:

    gr.Markdown(f"""# 🌌 Cosmos Reason2 — BYO Video Demo
**Model:** `{MODEL_NAME}` &nbsp;·&nbsp; **Load:** {load_time:.1f}s &nbsp;·&nbsp; **GPU:** {gpu_name} &nbsp;·&nbsp; **VRAM free:** {free_vram:,} MiB
{LOW_VRAM_NOTICE}
Upload any MP4 and ask the model a question about it.
""")

    # ── Input row ─────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="Upload Video (MP4)", sources=["upload"], height=280)
            clip_info   = gr.Markdown("*Upload a video to see clip info*")

        with gr.Column(scale=1):
            demo_picker = gr.Dropdown(
                label="Demo Prompt",
                choices=[p[0] for p in DEMO_PROMPTS],
                value=DEMO_PROMPTS[0][0],
                info="Presets for quick testing",
            )
            run_btn = gr.Button("▶  Run Inference", variant="primary", size="lg")

    # ── Advanced settings ─────────────────────────────────────────────────────
    with gr.Accordion("⚙️  Advanced Settings", open=False):
        with gr.Row():
            system_box = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM, lines=2)
            user_box   = gr.Textbox(label="User Prompt", value=DEFAULT_PROMPT, lines=2)

        with gr.Row():
            fps_slider = gr.Slider(
                minimum=1, maximum=8, step=1, value=DEFAULT_FPS,
                label="Video sampling rate (fps)",
                info="Higher = more frames = more tokens = slower inference",
            )
            maxpx_slider = gr.Slider(
                minimum=128*(32**2), maximum=4096*(32**2), step=128*(32**2),
                value=DEFAULT_MAX_PIXELS,
                label="Max pixels per frame",
                info="Reduce on low-VRAM systems",
            )
            maxtok_slider = gr.Slider(
                minimum=64, maximum=2048, step=64, value=DEFAULT_MAX_TOKENS,
                label="Max output tokens",
            )

    # ── Output ────────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            response_out = gr.Textbox(label="Model Response", lines=16, interactive=False)
        with gr.Column(scale=1):
            status_out  = gr.Markdown("*—*", label="Status")
            metrics_out = gr.Markdown("*—*", label="Metrics")

    # ── Events ────────────────────────────────────────────────────────────────
    def on_upload(path, fps_val):
        if not path:
            return "*Upload a video to see clip info*", gr.update()
        m = get_video_meta(path)
        if not m.width:
            return "*Clip info unavailable (PyAV not installed)*", gr.update()

        fps_val   = max(1, int(fps_val))
        n_frames  = max(1, int(m.duration_s * fps_val))
        base_info = (f"**Resolution:** {m.width}×{m.height}"
                     f"  |  **FPS:** {m.fps:.1f}"
                     f"  |  **Duration:** {m.duration_s:.1f}s"
                     f"  |  **Frames sampled:** {n_frames}")

        est_s_full = _est_tokens(n_frames, DEFAULT_MAX_PIXELS) / PREFILL_TPS
        capped_px, est_s = _auto_cap(n_frames, DEFAULT_MAX_PIXELS)

        if capped_px < DEFAULT_MAX_PIXELS:
            note = (f"\n> ⚠ **Auto-cap:** resolution reduced to **{capped_px:,} px/frame**"
                    f" → estimated **~{est_s:.0f}s** inference"
                    f" (was ~{est_s_full:.0f}s at full resolution)")
            return base_info + note, gr.update(value=capped_px)
        else:
            return base_info + f"  |  **Est. inference:** ~{est_s:.0f}s", gr.update()

    video_input.change(on_upload, inputs=[video_input, fps_slider], outputs=[clip_info, maxpx_slider])

    def on_demo(name):
        for n, p in DEMO_PROMPTS:
            if n == name:
                return p
        return DEFAULT_PROMPT

    demo_picker.change(on_demo, inputs=[demo_picker], outputs=[user_box])

    def _run(video_path, user_prompt, system_prompt, fps, max_pixels, max_new_tokens):
        for text, metrics in run_inference(video_path, user_prompt, system_prompt, fps, max_pixels, max_new_tokens):
            yield text, metrics

    run_btn.click(
        fn=_run,
        inputs=[video_input, user_box, system_box, fps_slider, maxpx_slider, maxtok_slider],
        outputs=[response_out, metrics_out],
    )

    SAMPLE_VIDEO = f"{HOME}/cosmos-reason2/assets/sample.mp4"
    if os.path.exists(SAMPLE_VIDEO):
        gr.Examples(
            examples=[[SAMPLE_VIDEO, DEFAULT_PROMPT, DEFAULT_SYSTEM, DEFAULT_FPS, DEFAULT_MAX_PIXELS, DEFAULT_MAX_TOKENS]],
            inputs=[video_input, user_box, system_box, fps_slider, maxpx_slider, maxtok_slider],
            label="Sample video",
        )

demo.launch(server_name="0.0.0.0", server_port=PORT, share=SHARE)
