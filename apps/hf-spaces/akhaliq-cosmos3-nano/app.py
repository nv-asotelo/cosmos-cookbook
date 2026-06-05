import spaces
import os
import torch
from PIL import Image
from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video, encode_video, load_image
import tempfile
import uuid

# ─── Model Loading ───────────────────────────────────────────────────────────
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=False,
)

app = Server(title="Cosmos3 Nano — World Foundation Model")

# ─── Negative prompts ────────────────────────────────────────────────────────
NEGATIVE_PROMPT_VIDEO = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)

NEGATIVE_PROMPT_I2V = (
    "The video captures a series of frames showing macroblocking artifacts, chromatic aberration, "
    "high-frequency noise, and rolling shutter distortion. It includes static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, bit-depth compression artifacts, color banding, unnatural transitions, "
    "outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual "
    "noise, and flickering. Avoid moiré patterns, edge halos, and temporal aliasing. Furthermore, the content "
    "defies common sense, generating illogical scenarios, nonsensical entities, absurd character behaviors, "
    "and conceptual paradoxes that violate basic human reasoning and everyday reality. The video looks like a "
    "surreal or glitchy hallucination. Overall, the video is of poor quality."
)


# ─── Dynamic GPU duration estimator for ZeroGPU ─────────────────────────────
# Rough per-step time scales with resolution × num_frames.
# Benchmarked: ~11.4s/step at 1280×720×189 on RTX Pro 6000 Blackwell.
# Reference compute: 1280 * 720 * 189 = 174,182,400
_REF_COMPUTE = 1280 * 720 * 189
_REF_STEP_TIME = 11.4  # seconds per step at reference compute


def _estimate_duration(num_frames, height, width, num_inference_steps, **kwargs):
    """Estimate GPU seconds needed, with overhead buffer."""
    compute = width * height * num_frames
    step_time = _REF_STEP_TIME * (compute / _REF_COMPUTE)
    estimated = step_time * num_inference_steps
    # Add 30s buffer for VAE decode, model init, etc.
    return min(int(estimated + 30), 300)


def _estimate_duration_t2i(prompt, height, width, num_inference_steps):
    return _estimate_duration(num_frames=1, height=height, width=width,
                              num_inference_steps=num_inference_steps)


def _estimate_duration_t2v(prompt, negative_prompt, num_frames, height, width,
                           fps, num_inference_steps):
    return _estimate_duration(num_frames=num_frames, height=height, width=width,
                              num_inference_steps=num_inference_steps)


def _estimate_duration_i2v(image_path, prompt, negative_prompt, num_frames,
                           height, width, fps, num_inference_steps):
    return _estimate_duration(num_frames=num_frames, height=height, width=width,
                              num_inference_steps=num_inference_steps)


def _estimate_duration_sound(prompt, negative_prompt, num_frames, height, width,
                             fps, num_inference_steps):
    return _estimate_duration(num_frames=num_frames, height=height, width=width,
                              num_inference_steps=num_inference_steps)


# ─── API endpoints ───────────────────────────────────────────────────────────


@app.api()
@spaces.GPU(duration=_estimate_duration_t2i, size="xlarge")
def text_to_image(
    prompt: str,
    height: int = 720,
    width: int = 1280,
    num_inference_steps: int = 25,
) -> FileData:
    """Generate a single image from a text prompt using Cosmos3-Nano."""
    result = pipe(
        prompt=prompt,
        num_frames=1,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
    )
    out_path = os.path.join(tempfile.gettempdir(), f"cosmos3_t2i_{uuid.uuid4().hex[:8]}.jpg")
    result.video[0].save(out_path, format="JPEG", quality=90)
    return FileData(path=out_path)


@app.api()
@spaces.GPU(duration=_estimate_duration_t2v, size="xlarge")
def text_to_video(
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 121,
    height: int = 544,
    width: int = 960,
    fps: float = 24.0,
    num_inference_steps: int = 25,
) -> FileData:
    """Generate a video from a text prompt using Cosmos3-Nano."""
    neg = negative_prompt if negative_prompt else NEGATIVE_PROMPT_VIDEO
    result = pipe(
        prompt=prompt,
        negative_prompt=neg,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
        num_inference_steps=num_inference_steps,
    )
    out_path = os.path.join(tempfile.gettempdir(), f"cosmos3_t2v_{uuid.uuid4().hex[:8]}.mp4")
    export_to_video(result.video, out_path, fps=int(fps), macro_block_size=1)
    return FileData(path=out_path)


@app.api()
@spaces.GPU(duration=_estimate_duration_i2v, size="xlarge")
def image_to_video(
    image_path: FileData,
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 121,
    height: int = 544,
    width: int = 960,
    fps: float = 24.0,
    num_inference_steps: int = 25,
) -> FileData:
    """Generate a video conditioned on an input image and text prompt."""
    image = Image.open(image_path["path"]).convert("RGB")
    neg = negative_prompt if negative_prompt else NEGATIVE_PROMPT_I2V
    result = pipe(
        prompt=prompt,
        negative_prompt=neg,
        image=image,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
        num_inference_steps=num_inference_steps,
    )
    out_path = os.path.join(tempfile.gettempdir(), f"cosmos3_i2v_{uuid.uuid4().hex[:8]}.mp4")
    export_to_video(result.video, out_path, fps=int(fps), macro_block_size=1)
    return FileData(path=out_path)


@app.api()
@spaces.GPU(duration=_estimate_duration_sound, size="xlarge")
def text_to_video_with_sound(
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 121,
    height: int = 544,
    width: int = 960,
    fps: float = 24.0,
    num_inference_steps: int = 25,
) -> FileData:
    """Generate a video with synchronized audio from a text prompt."""
    neg = negative_prompt if negative_prompt else NEGATIVE_PROMPT_VIDEO
    result = pipe(
        prompt=prompt,
        negative_prompt=neg,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
        enable_sound=True,
        num_inference_steps=num_inference_steps,
    )
    out_path = os.path.join(tempfile.gettempdir(), f"cosmos3_sound_{uuid.uuid4().hex[:8]}.mp4")
    encode_video(
        result.video,
        fps=int(fps),
        audio=result.sound,
        audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
        output_path=out_path,
    )
    return FileData(path=out_path)


# ─── Serve frontend ──────────────────────────────────────────────────────────


@app.get("/")
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


app.launch(show_error=True)
