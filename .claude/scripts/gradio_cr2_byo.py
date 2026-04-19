#!/usr/bin/env python3
"""
Cosmos Reason2 BYO-Video Gradio Demo
Upload any MP4 and get a model response.
"""
import os, json, time, tempfile, warnings
warnings.filterwarnings("ignore")

import torch
import transformers
from transformers import video_processing_utils
from transformers.video_utils import load_video as _load_video
import gradio as gr

# PyAV backend patch (required on Hyperstack / Horde — FFmpeg not in system PATH)
def _patched_fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
    if isinstance(video_url_or_urls, list):
        return list(zip(*[
            _patched_fetch_videos(self, x, sample_indices_fn=sample_indices_fn)
            for x in video_url_or_urls
        ]))
    return _load_video(video_url_or_urls, backend="pyav", sample_indices_fn=sample_indices_fn)

video_processing_utils.BaseVideoProcessor.fetch_videos = _patched_fetch_videos

MODEL_DIR  = os.environ.get("MODEL_DIR",  "/home/shadeform/cosmos-reason2/models/Cosmos-Reason2-2B")
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/Cosmos-Reason2-2B")
OUT_FILE   = os.environ.get("OUT_FILE",   "/tmp/byo_video_reason2_results.json")
PORT       = int(os.environ.get("GRADIO_PORT", "7860"))
LOW_VRAM   = os.environ.get("LOW_VRAM", "false").lower() == "true"
FPS        = 1 if LOW_VRAM else 4
MAX_PIXELS = 128 * (32**2) if LOW_VRAM else 4096 * (32**2)

print(f"[demo] Loading model from {MODEL_DIR} ...", flush=True)
t0 = time.time()
model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.float16, device_map="auto", attn_implementation="sdpa",
)
processor = transformers.Qwen3VLProcessor.from_pretrained(MODEL_DIR)
PIXELS_PER_TOKEN = 32 ** 2
processor.image_processor.size = {"shortest_edge": 256 * PIXELS_PER_TOKEN, "longest_edge": MAX_PIXELS}
processor.video_processor.size = {"shortest_edge": 256 * PIXELS_PER_TOKEN, "longest_edge": MAX_PIXELS}
load_time = time.time() - t0
print(f"[demo] Model ready in {load_time:.1f}s {'(LOW_VRAM mode: fps=1)' if LOW_VRAM else ''}", flush=True)


def run_inference(video_path, prompt, system_prompt):
    if video_path is None:
        return "Upload a video first.", "{}"
    if not prompt.strip():
        prompt = "Describe this video in detail."

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [
            {"type": "video", "video": video_path},
            {"type": "text",  "text": prompt},
        ]},
    ]

    t1 = time.time()
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", fps=FPS,
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    n = inputs["input_ids"].shape[1]
    response = processor.decode(out_ids[0][n:], skip_special_tokens=True)
    infer_time = time.time() - t1

    result = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "response": response,
        "load_time_s": round(load_time, 1),
        "infer_time_s": round(infer_time, 1),
        "status": "success",
    }
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    return response, f"Inference: {infer_time:.1f}s  |  Model load: {load_time:.1f}s  |  Results saved to {OUT_FILE}"


DEFAULT_SYSTEM = "You are a helpful assistant that analyzes videos."
DEFAULT_PROMPT = "Describe this video in detail."

LOW_VRAM_NOTICE = "\n> ⚠ **Low-VRAM mode** — fps=1, reduced resolution. Use clips under 60s for best results." if LOW_VRAM else ""

with gr.Blocks(title="Cosmos Reason2 — BYO Video Demo", theme=gr.themes.Default()) as demo:
    gr.Markdown(
        f"""
# Cosmos Reason2 — BYO Video Demo
**Model:** `{MODEL_NAME}` &nbsp;|&nbsp; **Load:** {load_time:.1f}s &nbsp;|&nbsp; **GPU:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"} &nbsp;|&nbsp; **fps:** {FPS}
{LOW_VRAM_NOTICE}
Upload any MP4 and ask the model a question about it.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload your video (MP4)")
            prompt_input = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=3,
            )
            system_input = gr.Textbox(
                label="System prompt (optional)",
                value=DEFAULT_SYSTEM,
                lines=2,
            )
            run_btn = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=1):
            response_output = gr.Textbox(label="Model response", lines=18, interactive=False)
            timing_output = gr.Textbox(label="Timing", lines=1, interactive=False)

    run_btn.click(
        fn=run_inference,
        inputs=[video_input, prompt_input, system_input],
        outputs=[response_output, timing_output],
    )

    gr.Examples(
        examples=[["/home/shadeform/cosmos-reason2/assets/sample.mp4", DEFAULT_PROMPT, DEFAULT_SYSTEM]],
        inputs=[video_input, prompt_input, system_input],
        label="Sample video",
    )

SHARE = os.environ.get("GRADIO_SHARE", "true").lower() != "false"
demo.launch(server_name="0.0.0.0", server_port=PORT, share=SHARE)
