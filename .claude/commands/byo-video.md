# /byo-video — Cosmos BYO-Video Demo

Upload a video, pick a prompt, get structured analysis from Cosmos Reason2.

**Canonical scripts (stable, versioned — do not read from /tmp/):**
- `~/.claude/scripts/gradio_cr2_byo.py`  — Gradio demo app (vLLM + HF + NIM backends)
- `~/.claude/scripts/byo_video_setup.py` — Bootstrap + launch (HF Transformers mode)

---

## Supported models

| Model | Size | Min VRAM | Preferred backend |
|---|---|---|---|
| Cosmos Reason2 (CR2-2B) | 2B VLM | 40 GB | HF Transformers |
| Cosmos Reason2 (CR2-8B) | 8B VLM | 80 GB | vLLM (hot-swap via dropdown) |
| Cosmos Reason2 (CR2-8B-NVFP4) | 8B VLM | 80 GB | vLLM |
| Cosmos Reason2 (CR2-8B-FP8) | 8B VLM | 80 GB | vLLM |

CR2-2B runs on workstation hardware (≥40GB). CR2-8B requires an H100 or A100 80GB; use vLLM backend for sub-second TTFT (HF gives ~44s on H100 vs ~174ms on vLLM).

---

## Disk requirements

**3× 8B variants on-disk needs ≥200GB free.**

| Checkpoint | Disk |
|---|---|
| Cosmos-Reason2-8B (BF16) | ~16 GB |
| Cosmos-Reason2-8B-FP8 | ~10 GB |
| Cosmos-Reason2-8B-NVFP4 | ~7 GB |
| Cosmos-Reason2-2B-FP8 | ~3 GB |
| OS + Python env (cosmos-reason2) | ~35 GB |
| HF download cache (temp) | = model size while downloading |
| Video uploads | up to 1 GB each |
| **Total recommended** | **≥200 GB** |

**Important:** `snapshot_download` creates hardlinks between the HF cache and the local model dir. `du -sh` on each directory double-counts the blocks. The HF cache (`~/.cache/huggingface/hub/`) can be deleted safely once a model is installed — the files in `cosmos-reason2/models/` are the real copies. Use the Storage panel in the Gradio UI to monitor and clean.

---

## VRAM auto-selection

The setup script (`byo_video_setup.py`) handles tier selection automatically:

**Model selection** — CR2-2B for the live demo (8B fails the <60s inference target on non-H100 GPUs). Force 8B via `MODEL_NAME=nvidia/Cosmos-Reason2-8B` env var.
- ≥ 40000 MiB free → `nvidia/Cosmos-Reason2-2B`
- < 40000 MiB → CR2-2B with LOW_VRAM mode (fps=1, reduced resolution)

**FPS/pixel tier** — detected by GPU name (not free VRAM), because workstation GPUs have large VRAM but slower prefill compute:

| Tier | GPU names | fps | max_pixels | Expected TTFT (HF) |
|---|---|---|---|---|
| H100/A100 | H100, A100, H200, GB200 | 8 | 1,048,576 | ~18s TTFT (vLLM: ~174ms) |
| High-VRAM (non-H100) | RTX PRO, A40, A30, etc. | 8 | 524,288 | ~18s TTFT |
| Low-VRAM | < 40GB free | 4 | 131,072 | ~80–120s |

---

## Backend comparison: HF Transformers vs vLLM

| Metric | HF Transformers | vLLM |
|---|---|---|
| TTFT (H100, CR2-8B, 2K tokens) | ~18–44s | ~174ms |
| TTFT (H100, CR2-8B, 4K tokens) | — | ~341ms |
| TTFT (H100, CR2-8B, 8K tokens) | — | ~724ms |
| Prefill throughput | ~23 tok/s (RTX PRO 6000) | ~12,000 tok/s (H100) |
| Model load time | 10–20s (cached) | 60–90s (server restart) |
| Hot-swap to new checkpoint | ~30s (Python re-load) | ~60–90s (vLLM restart) |
| Concurrent requests | blocked per request | full batching |
| Quantization support | FP8 via HF auto | FP8, NVFP4 native |

**When to use HF mode:** Demos where startup time matters more than TTFT. CR2-2B on workstations. Environments where vLLM is not installed.

**When to use vLLM mode:** Benchmarking, customer demos requiring sub-second TTFT, any 8B variant. The Gradio dropdown triggers live server hot-swap between NVFP4/FP8/BF16 without restarting the UI.

---

## Environment detection (LOCAL FIRST)

Check in this order:

1. **Local GPU**: `nvidia-smi` succeeds → local path, no cloud needed
2. **Brev**: `brev ls` succeeds → Brev path
3. **Horde**: `HORDE_API_KEY` is set → Horde path
4. **Nebius**: `NEBIUS_ENDPOINT` is set → Nebius path
5. **None**: ask "Which environment?"

---

## Primary flow — Brev web demo (most common)

### Step 1 — Create or reuse instance

```bash
# massedcompute_H100 gives 1TB disk (required for multi-model) at $3.58/hr
brev create byo-video-vllm --type massedcompute_H100

# Or list existing
brev ls
```

**Why massedcompute_H100?** 1TB disk. The hyperstack_H100 ($2.28/hr) provisions ~97GB OS disk — too small to hold 3× 8B variants + download cache + video uploads. massedcompute gives sufficient headroom.

Wait for STATUS: RUNNING. Then confirm the instance is reachable:
```bash
brev exec byo-video-vllm "nvidia-smi"
```

### Step 2 — Bootstrap

Run once per fresh instance.

```bash
# Install uv
brev exec byo-video-vllm "curl -LsSf https://astral.sh/uv/install.sh | sh"

# Clone cosmos-reason2
brev exec byo-video-vllm "git clone https://github.com/nvidia-cosmos/cosmos-reason2.git ~/cosmos-reason2"

# Install dependencies
brev exec byo-video-vllm "cd ~/cosmos-reason2 && export PATH=~/.local/bin:~/.cargo/bin:\$PATH && uv sync --extra cu128"
# Note: byo_video_setup.py auto-detects CUDA 12.8 (driver <575.x) and downgrades
# vLLM to 0.11.0 automatically. On CUDA 12.9+ instances this step is not needed.

# Install PyAV
brev exec byo-video-vllm "cd ~/cosmos-reason2 && export PATH=~/.local/bin:~/.cargo/bin:\$PATH && uv pip install av==16.1.0"

# Install vLLM (for vLLM backend)
brev exec byo-video-vllm "cd ~/cosmos-reason2 && export PATH=~/.local/bin:~/.cargo/bin:\$PATH && uv pip install vllm"

# Download NVFP4 (first model, smallest, ~7GB)
brev exec byo-video-vllm "cd ~/cosmos-reason2 && export PATH=~/.local/bin:~/.cargo/bin:\$PATH && export HF_TOKEN=hf_... && uv run hf download nvidia/Cosmos-Reason2-8B-NVFP4 --local-dir models/Cosmos-Reason2-8B-NVFP4"
```

### Step 3 — Start vLLM server

```bash
brev exec byo-video-vllm "nohup ~/cosmos-reason2/.venv/bin/vllm serve ~/cosmos-reason2/models/Cosmos-Reason2-8B-NVFP4 --served-model-name nvidia/Cosmos-Reason2-8B-NVFP4 --port 8000 --dtype auto --trust-remote-code --max-model-len 32768 --gpu-memory-utilization 0.85 > /tmp/vllm.log 2>&1 &"
```

Wait ~60–90s for vLLM to load, then confirm:
```bash
brev exec byo-video-vllm "curl -s http://localhost:8000/v1/models"
```

### Step 4 — Deploy scripts and launch Gradio

```bash
# Deploy scripts
brev copy ~/.claude/scripts/gradio_cr2_byo.py byo-video-vllm:/tmp/gradio_cr2_byo.py
brev copy ~/.claude/scripts/byo_video_setup.py byo-video-vllm:/tmp/byo_video_setup.py

# Launch Gradio in vLLM mode
brev exec byo-video-vllm "nohup bash -c 'cd ~/cosmos-reason2 && export PATH=~/.local/bin:~/.cargo/bin:\$PATH && INFERENCE_BACKEND=vllm MODEL_SIZE=8B .venv/bin/python /tmp/gradio_cr2_byo.py' > /tmp/gradio.log 2>&1 &"

# Wait ~15s for Gradio to start, then get URL
brev exec byo-video-vllm "cat /tmp/gradio_url.txt"
```

### Step 5 — Kill alert (required)

When Alex is done:
```bash
/Users/asotelo/.nvcowork/bin/pa --agent tpm --print \
  "Send a Microsoft Teams message to chat ID 48:notes with this text: \
  'byo-video demo done on brev/byo-video-vllm (H100). Instance can be terminated.'" \
  --max-duration 90
```

Do NOT auto-terminate. Notify Alex and wait for explicit kill confirmation.

---

## vLLM checkpoint hot-swap

The Gradio dropdown triggers live vLLM server restart when you select a different checkpoint:

1. Select checkpoint → orange banner: "⚠ Restarting vLLM for {model}…"
2. Polls `/v1/models` every 5s (~60–90s total)
3. Banner turns green: "✅ vLLM now serving {model} ({Ns})"
4. If model is not on disk → grey banner + **"Download & Load"** button
5. "Download & Load" → HF Hub download → auto-swap vLLM

Dropdown includes all 2B, 8B, and 32B variants. Models not on disk show "not on disk" banner until downloaded.

---

## Local web demo (no cloud billing)

Same Gradio app, runs on your own GPU machine.

```bash
# Bootstrap (once)
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git ~/cosmos-reason2
cd ~/cosmos-reason2
uv sync --extra cu128
uv pip install "av==16.1.0" gradio
export HF_TOKEN=hf_...
uv run hf download nvidia/Cosmos-Reason2-2B \
  --local-dir ~/cosmos-reason2/models/Cosmos-Reason2-2B

# Launch (HF mode, CR2-2B)
export MODEL_DIR=~/cosmos-reason2/models/Cosmos-Reason2-2B
export MODEL_NAME=nvidia/Cosmos-Reason2-2B
cd ~/cosmos-reason2
uv run python /tmp/gradio_cr2_byo.py
```

---

## Headless inference (programmatic / no browser)

```bash
export HF_TOKEN=hf_...
export BYO_VIDEO=/path/to/video.mp4
export MODEL_DIR=~/cosmos-reason2/models/Cosmos-Reason2-2B
export MODEL_NAME=nvidia/Cosmos-Reason2-2B
export OUT_FILE=/tmp/byo_video_reason2_results.json
cd ~/cosmos-reason2
uv run python /tmp/smoke_cr2_byo.py
```

---

## Horde (SSH-based — asotelo org)

**SSH username is `horde`** — confirmed 2026-04-17. Not `ubuntu`, `nvidia`, or `root`.

Deploy scripts:
```bash
for script in byo_video_setup gradio_cr2_byo; do
  B64=$(base64 -i ~/.claude/scripts/${script}.py | tr -d '\n')
  ssh -i ~/.ssh/id_ed25519 horde@<ip> \
    "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done
```

Run setup:
```bash
ssh -i ~/.ssh/id_ed25519 horde@<ip> \
  "export HF_TOKEN=hf_... && python3 /tmp/byo_video_setup.py"
```

**Horde capacity API is stale** — reports availability that doesn't match actual pool. Confirmed 2026-04-17. Retry or use Brev. Existing instance `asotelo-uzof99` (A40) is the reliable fallback.

---

## Gradio app: backend modes

| Mode | Env var | Notes |
|---|---|---|
| HF Transformers | `INFERENCE_BACKEND=hf` (default) | Loads model into Python; ~10-20s load, ~18-44s TTFT |
| vLLM | `INFERENCE_BACKEND=vllm` | Requires vLLM server on port 8000; sub-second TTFT |
| NIM (NVCF) | `INFERENCE_BACKEND=nim` | NGC API key required; no local GPU |
| NIM local | `INFERENCE_BACKEND=nim_local` | Container on local GPU; 5-image limit |

**Canonical source:** `~/.claude/scripts/gradio_cr2_byo.py` — deploy via `brev copy` or base64.

---

## AGENT PROTOCOL

**USER MUST BRING:**

| What | How to provide |
|---|---|
| Environment | "local", "brev", "horde", or "nebius" |
| Instance name / IP | `brev ls` to confirm, or paste the IP |
| HF_TOKEN | `export HF_TOKEN=hf_...` in session |
| BYO video file | Path to MP4 on instance, or upload via Gradio UI |

**AGENT RUNS AUTONOMOUSLY:** environment detection, deploy steps, bootstrap, URL capture, kill alert.

---

## PyAV backend patch (always apply)

Required on Hyperstack and Horde — FFmpeg not in system PATH. Already embedded in `gradio_cr2_byo.py` and `smoke_cr2_byo.py`.

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

| Environment | GPU | Model | Backend | TTFT | Pass <60s? |
|---|---|---|---|---|---|
| Brev Hyperstack | H100 PCIe | CR2-2B BF16 | HF | ~44s | ✅ |
| Brev Hyperstack | H100 PCIe | CR2-2B BF16 | HF | ~43s (2nd run) | ✅ |
| Horde | A40 | CR2-2B BF16 | HF | ~77s | ❌ |
| Horde | RTX PRO 6000 Blackwell 97GB | CR2-2B BF16 | HF | ~55s | ✅ |
| Horde | RTX PRO 6000 Blackwell 97GB | CR2-8B BF16 | HF | >62s | ❌ |
| Brev H100 | H100 | CR2-8B NVFP4 | vLLM | ~174ms (2K tok) | ✅ |
| Brev H100 | H100 | CR2-8B NVFP4 | vLLM | ~341ms (4K tok) | ✅ |
| Brev H100 | H100 | CR2-8B NVFP4 | vLLM | ~724ms (8K tok) | ✅ |

---

## Common failure modes

| Symptom | Fix |
|---|---|
| Port 7860 unreachable | Check `ss -tlnp` on instance. Gradio failed — check `/tmp/gradio.log`. |
| Horde: no public share URL | NVIDIA network blocks outgoing to gradio.live. Normal — access via SSH tunnel: `ssh -L 7860:localhost:7860 horde@<ip>` then open localhost:7860. |
| Horde: `ffprobe` not found | Horde's snap ffmpeg doesn't include ffprobe in PATH. Fix: `wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O /tmp/ff.tar.xz && cd /tmp && tar -xf ff.tar.xz && sudo cp /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe`. |
| Black frames / torchcodec error | PyAV patch not applied. Confirm `gradio_cr2_byo.py` was deployed. |
| OOM during inference | Kill other GPU processes. CR2-8B needs ~80GB VRAM. |
| vLLM 0.12.0 fails to start | CUDA 12.8 (driver 570.x) is incompatible. `byo_video_setup.py` auto-detects and pins vLLM to 0.11.0. Manual fix: `cd ~/cosmos-reason2 && uv pip install vllm==0.11.0`. |
| Horde SSH rejected | Username must be `horde`. Key: `~/.ssh/id_ed25519`. |
| vLLM "Server not running" | vLLM process died. Restart with vLLM serve command from Step 3. |
| Download fails ENOSPC | Disk full. Open Storage panel in Gradio UI. Clean HF cache or delete partial downloads. 32B models need ≥64GB free — use massedcompute_H100. |
| Dropdown swap stalls >150s | vLLM failed to start. SSH to instance, check `cat /tmp/vllm.log`. |
| BF16 8B not on disk | Click "Download & Load" from dropdown banner. Needs ~16GB free + HF token. |
| FP8 download fails with 401 | CR2-8B-FP8 is a gated HuggingFace model — requires HF_TOKEN with accepted license. Use "Apply Token" in Gradio Advanced Settings → HuggingFace Auth. |
| HF 429 rate limit on download | Script retries with sleep. Usually succeeds by attempt 3–4. |
| hyperstack_H100 disk full at 97GB | Wrong instance type. Use massedcompute_H100 (1TB). hyperstack OS disk is too small for multi-model. |
| Gradio theme error on startup | Gradio 6.0 moved `theme=` from `gr.Blocks()` to `demo.launch()`. Deploy the latest canonical `gradio_cr2_byo.py` from `~/.claude/scripts/`. |
| `flashinfer-cubin` version mismatch after vLLM downgrade | `RuntimeError: flashinfer-cubin version (0.6.6) does not match flashinfer version (0.5.3)`. Set `FLASHINFER_DISABLE_VERSION_CHECK=1` in env. `byo_video_setup.py` sets this automatically in `launch_env`. |
| vLLM startup fails: `ninja` not found | `FileNotFoundError: ninja` from flashinfer JIT. Run `sudo apt-get install -y ninja-build`. `byo_video_setup.py` Step 6c does this automatically. |
