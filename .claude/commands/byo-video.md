# /byo-video — Cosmos BYO-Video Demo

Launch an interactive Gradio web demo for Cosmos Reason2: upload any MP4 in a browser, get model output. Handles environment detection, dependency bootstrap, model download, and web UI launch.

Default output: **Gradio web UI at a `gradio.live` public URL** — user uploads video in browser.
Headless output (programmatic): JSON at `/tmp/byo_video_reason2_results.json`.

**Canonical scripts:**
- `gradio_cr2_byo.py` — Gradio app (deploy to instance before running)
- `byo_video_setup.py` — Bootstrap + launch (installs deps, downloads model, starts Gradio)

**Primary launch command (all environments):**
```bash
python3 /tmp/byo_video_setup.py
```
Streams live step-by-step progress with ETAs. Prints a clickable `gradio.live` URL at the end. URL also written to `/tmp/gradio_url.txt`.

---

## Supported models

| Model | Size | Min VRAM | Use case |
|---|---|---|---|
| Cosmos Reason2 (CR2-2B) | 2B VLM | 40 GB | Video understanding: robotics, AV, Metropolis |
| Cosmos Reason2 (CR2-8B) | 8B VLM | 80 GB | Same, higher quality |

---

## VRAM auto-selection

Agent runs this before any setup:
```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
```

- ≥ 80,000 MiB → `nvidia/Cosmos-Reason2-8B`
- ≥ 40,000 MiB → `nvidia/Cosmos-Reason2-2B`
- < 40,000 MiB → CR2-2B with LOW_VRAM mode (fps=1, reduced resolution)

---

## AGENT PROTOCOL

**USER MUST BRING:**

| What | How |
|---|---|
| Environment | "local", "brev", or SSH host |
| HF_TOKEN | Required — gated model access |
| BYO video | Path on instance, or upload via UI once live |

**AGENT RUNS AUTONOMOUSLY:**
- VRAM check and model size selection
- All deploy steps (base64 encode + remote write)
- Bootstrap via `byo_video_setup.py` (streams progress)
- URL capture from `/tmp/gradio_url.txt`

---

## Deploy — Brev

```bash
# Create instance
brev create <name> --gpu-name H100 --type hyperstack_H100

# Wait until SHELL = READY
brev ls

# Deploy scripts
for script in byo_video_setup gradio_cr2_byo; do
  B64=$(base64 -i path/to/${script}.py | tr -d '\n')
  brev exec <name> "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done

# Launch (streams live progress)
brev exec <name> "export HF_TOKEN=hf_... && export PATH=~/.local/bin:~/.cargo/bin:\$PATH && python3 /tmp/byo_video_setup.py"
```

---

## Deploy — SSH instance

```bash
# Deploy scripts
for script in byo_video_setup gradio_cr2_byo; do
  B64=$(base64 -i path/to/${script}.py | tr -d '\n')
  ssh user@<host> "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done

# Launch detached (survives SSH timeout during long model loads)
ssh user@<host> "nohup env HF_TOKEN=hf_... PATH=~/.local/bin:~/.cargo/bin:\$PATH python3 /tmp/byo_video_setup.py > /tmp/byo_video.log 2>&1 &"

# Poll for URL
ssh user@<host> "grep -E 'gradio.live|Running on' /tmp/byo_video.log | tail -3"
```

---

## Local (no cloud)

```bash
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git ~/cosmos-reason2
cd ~/cosmos-reason2
uv sync --extra cu128
uv pip install "av==16.1.0" gradio

export HF_TOKEN=hf_...
uv run huggingface-cli download nvidia/Cosmos-Reason2-2B --local-dir ~/cosmos-reason2/models/Cosmos-Reason2-2B

export MODEL_DIR=~/cosmos-reason2/models/Cosmos-Reason2-2B
export MODEL_NAME=nvidia/Cosmos-Reason2-2B
uv run python /tmp/gradio_cr2_byo.py
```

Open `http://localhost:7860`.

---

## Gradio UI features

- **Video upload:** drag-and-drop MP4; clip info (resolution · FPS · duration) shown on upload
- **Demo prompts:** dropdown with presets for quick testing (non-expert summary, safety check, action description, etc.)
- **Advanced settings** (collapsible):
  - System prompt and user prompt (editable)
  - Video sampling rate (fps) — affects frame count and token count
  - Max pixels per frame — reduce on low-VRAM systems
  - Max output tokens
- **Streaming output:** tokens appear live as the model generates
- **Timing:** inference time and model load time displayed after completion
- **Results JSON:** saved to `/tmp/byo_video_reason2_results.json` after each run

---

## Env vars

| Var | Default | Notes |
|---|---|---|
| `MODEL_DIR` | `~/cosmos-reason2/models/Cosmos-Reason2-2B` | Local weights path |
| `MODEL_NAME` | `nvidia/Cosmos-Reason2-2B` | HF model ID (display only) |
| `HF_TOKEN` | — | Required for gated model download |
| `LOW_VRAM` | `false` | Set `true` for <40GB GPUs (fps=1, low resolution) |
| `GRADIO_PORT` | `7860` | Port |
| `GRADIO_SHARE` | `true` | Set `false` to disable public tunnel |
| `OUT_FILE` | `/tmp/byo_video_reason2_results.json` | Results output path |

---

## PyAV backend patch (always required)

FFmpeg is not in PATH on most cloud GPU images (Hyperstack, Lambda, Horde). The patch is embedded in `gradio_cr2_byo.py`. If writing a custom script, prepend:

```python
from transformers import video_processing_utils
from transformers.video_utils import load_video as _load_video

def _patched_fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
    if isinstance(video_url_or_urls, list):
        return list(zip(*[
            _patched_fetch_videos(self, x, sample_indices_fn=sample_indices_fn)
            for x in video_url_or_urls
        ]))
    return _load_video(video_url_or_urls, backend="pyav", sample_indices_fn=sample_indices_fn)

video_processing_utils.BaseVideoProcessor.fetch_videos = _patched_fetch_videos
```

---

## Timing benchmarks

| Environment | GPU | Model | Load | Inference |
|---|---|---|---|---|
| Brev Hyperstack | H100 PCIe | CR2-2B | 1.7s | 44.1s |
| Brev Hyperstack | H100 PCIe | CR2-2B | 10.6s | 42.9s |
| Horde / A40 | A40 | CR2-2B | 2.9s | 76.9s |

---

## Common failure modes

| Symptom | Fix |
|---|---|
| Blank Gradio page | Tunnel hiccup — restart the process, new URL is generated |
| Black frames / video error | PyAV not installed or patch not applied. `uv pip install "av==16.1.0"` |
| `torch.cuda.is_available()` → False | Wrong CUDA/torch build. Run `uv sync --extra cu128` in `cosmos-reason2/` dir |
| `uv sync --extra cu128` fails | Wrong CUDA driver. Check `nvidia-smi` shows CUDA 12.x |
| OOM during inference | Reduce fps or max_pixels in Advanced Settings, or use CR2-2B instead of 8B |
| Inference times out | Model too large for GPU, or video too long. Enable LOW_VRAM mode or shorten clip |
| `gradio.live` tunnel fails | Set `GRADIO_SHARE=false` and use SSH port-forward: `ssh -L 7860:localhost:7860 user@host` |
