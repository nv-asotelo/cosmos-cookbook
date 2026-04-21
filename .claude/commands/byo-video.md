# /byo-video — Cosmos BYO-Video Demo

Two modes:

**VLM Race (multi-model comparison) — use `/vlm-race` skill instead:**
Side-by-side comparison of Cosmos Reason 2, Nemotron-Nano-12B-v2, and Qwen3-VL.
The VLM Race is now its own skill with full documentation. See `/vlm-race`.
The section below is kept for reference only.

**Single-model (CR2 only) — original mode:**
Launch Cosmos Reason2 only. Faster startup. See `## Primary flow — Brev web demo` below.

Default output (both modes): **Gradio web UI at a `gradio.live` public URL** — user uploads video in browser.

**Canonical scripts (stable, versioned — do not read from /tmp/):**
- `~/.claude/scripts/gradio_compare_vlm.py` — VLM Race 3-model comparison app (new)
- `~/.claude/scripts/compare_vlm_setup.py`  — VLM Race bootstrap + launch (new)
- `~/.claude/scripts/gradio_cr2_byo.py`      — Single-model CR2 app (original)
- `~/.claude/scripts/byo_video_setup.py`     — Single-model bootstrap + launch (original)

---

## VLM Race — Multi-Model Comparison

### Models
| Column | Model | HF ID | Notes |
|---|---|---|---|
| 🌌 | Cosmos Reason 2 | `nvidia/Cosmos-Reason2-2B` or `8B` | Auto-sized by VRAM; checkpoint selectable in Advanced |
| 🤖 | Nemotron-Nano-12B-v2 | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | Fixed; gated — requires HF_TOKEN |
| 🔷 | Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct` or `8B` | Paired with CR2 size |

All models use BF16 (torch.bfloat16) by default.

### VRAM requirements
| Config | VRAM needed | Fits on |
|---|---|---|
| 2B variants (CR2-2B + Nem-12B + Qwen-2B) | ~32 GB | A40 (48GB), A100 (40/80GB), H100 (80GB) |
| 8B variants (CR2-8B + Nem-12B + Qwen-8B) | ~56 GB | A100 80GB, H100 80GB |
| RTX5070 (12 GB) | CR2-2B only; Nem-12B OOM | Graceful OOM error shown in UI |

### Deploy — Brev (agent provisions instance)

```bash
# Step 1 — Create instance (agent runs this)
brev create vlm-race --gpu-name H100 --type hyperstack_H100

# Step 2 — Wait for SHELL READY
brev ls   # repeat until SHELL column shows READY

# Step 3 — Deploy both scripts
for script in gradio_compare_vlm compare_vlm_setup; do
  B64=$(base64 -i ~/.claude/scripts/${script}.py | tr -d '\n')
  brev exec vlm-race "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done

# Step 4 — Launch (agent runs; streams to terminal)
brev exec vlm-race "export HF_TOKEN=hf_... && python3 /tmp/compare_vlm_setup.py"

# Step 5 — Capture URL (if needed separately)
brev exec vlm-race "cat /tmp/gradio_url.txt"
```

### Deploy — Horde (10.57.234.230, SSH key default)

```bash
# Deploy scripts
for script in gradio_compare_vlm compare_vlm_setup; do
  B64=$(base64 -i ~/.claude/scripts/${script}.py | tr -d '\n')
  ssh horde@10.57.234.230 "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done

# Launch
ssh horde@10.57.234.230 "export HF_TOKEN=hf_... && python3 /tmp/compare_vlm_setup.py"

# Capture URL
ssh horde@10.57.234.230 "cat /tmp/gradio_url.txt"
```

### Deploy — asotelo-dt (local workstation, RTX5070)

Same SSH pattern as Horde. Note: Nemotron-12B will OOM on RTX5070 — app handles gracefully.

```bash
ssh asotelo@asotelo-dt "python3 /tmp/compare_vlm_setup.py"
```

### Race UI
- **Basic mode:** upload video, pick demo prompt, click Start Comparison
- **Advanced mode:** edit system/user prompts, toggle thinking/reasoning, tune fps/resolution/max_tokens, change CR2 checkpoint
- **Metrics per column:** TTFT · Inference time · E2E time · Tokens in/out
- **Clip info:** resolution · fps · duration shown on upload
- **Winner badge:** 🏆 on whichever model has lowest inference_s among successful runs
- **OOM handling:** column shows error + "try quantized variant" message; other columns continue
- **Timeout:** per-model, default 240s; configurable in Advanced

### Env vars (VLM Race)
| Var | Default | Notes |
|---|---|---|
| `CR2_CHECKPOINT` | auto | Override CR2 model (any HF ID or local path) |
| `RACE_TIMEOUT_S` | 300 | Per-model timeout in seconds |
| `NEMOTRON_MODEL` | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | Override Nemotron model |
| `QWEN_MODEL_2B` / `QWEN_MODEL_8B` | Qwen3-VL defaults | Override Qwen3-VL model |
| `GRADIO_PORT` | 7860 | Port |
| `GRADIO_SHARE` | true | Set to false to disable public link |

### Results
JSON saved to `/tmp/vlm_race_results.json` after each run. Contains per-model: status, text, TTFT, inference_s, e2e_s, tokens_in, tokens_out, winner flag.

---

**Primary launch command (all environments):**
```bash
python3 /tmp/byo_video_setup.py
```
This script shows live step-by-step progress with ETAs for every install stage, then prints a clickable hyperlink to the Gradio UI. The URL is also written to `/tmp/gradio_url.txt` for agent capture. Deploy it to the instance before running (see deploy section below).

---

## AGENT PROTOCOL — what the agent runs vs what the user must bring

**USER MUST BRING (ask if not already provided):**

| What | How to provide |
|---|---|
| Environment | Say which: "local", "brev", "horde", or "nebius" |
| Instance name / IP | `brev ls` to confirm, or paste the IP |
| HF_TOKEN | `export HF_TOKEN=hf_...` in this session |
| BYO video file | Path to an MP4 on the instance, or upload via the Gradio UI once it's live |

**AGENT RUNS AUTONOMOUSLY (do not ask the user to run these):**
- All environment detection commands (`nvidia-smi`, `brev ls`, etc.)
- All deploy steps (base64 encode, `brev exec`, SSH copy)
- Bootstrap and launch via `byo_video_setup.py`
- URL capture from `/tmp/gradio_url.txt`
- Kill alert via pa-cli when Alex says done

**Crisp interaction target:** Alex provides the 4 items above, then the agent runs everything else without prompting for shell commands.

---

---

## Supported models

| Model | Size | Min VRAM | Use case |
|---|---|---|---|
| Cosmos Reason2 (CR2-2B) | 2B VLM | 40 GB | Video understanding: robotics, AV, Metropolis |
| Cosmos Reason2 (CR2-8B) | 8B VLM | 80 GB | Same, higher quality |
| Cosmos Transfer2.5 | Gen | 80 GB+ | Video-to-video generation, sim2real |
| Cosmos Predict2 | Gen | 80 GB+ | World model generation |

Transfer2.5 and Predict2 are datacenter-only (H100/A100 80GB+). Reason2 at 2B runs on workstation hardware (≥40GB).

---

## VRAM auto-selection

The setup script (`byo_video_setup.py`) handles all of this automatically. Rules as of 2026-04-21:

**Model selection** — always CR2-2B for the live demo (8B fails the <60s inference target on non-H100 GPUs). Force 8B via `MODEL_NAME=nvidia/Cosmos-Reason2-8B` env var if quality > speed.
- ≥ 40000 MiB free → `nvidia/Cosmos-Reason2-2B`
- < 40000 MiB → CR2-2B with LOW_VRAM mode (fps=1, reduced resolution)

**FPS/pixel tier** — detected by GPU name (not free VRAM), because workstation GPUs like RTX PRO 6000 have large VRAM but slower prefill compute:
| Tier | GPU names | fps | max_pixels | Expected inference |
|---|---|---|---|---|
| H100/A100 | H100, A100, H200, GB200 | 2 | 1,048,576 | ~44s (validated H100) |
| High-VRAM (non-H100) | RTX PRO, A40, A30, etc. | 1 | 524,288 | ~54s (validated RTX PRO 6000 97GB) |
| Low-VRAM | < 40GB free | 1 | 131,072 | ~80-120s |

**Pre-kill**: setup script kills any process on port 7860 BEFORE measuring VRAM, so stale processes don't skew the tier selection.

Transfer2.5 / Predict2: abort if < 80000. Do not proceed.

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
# Create new instance (name it something meaningful)
brev create <name> --gpu-name H100 --type hyperstack_H100

# Or list existing
brev ls
```

Wait for STATUS: RUNNING. Then get the public IP:
```bash
brev exec <name> "curl -s ifconfig.me"
```

### Step 2 — Bootstrap

Run once per fresh instance. All commands via `brev exec <name> "<cmd>"`.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone cosmos-reason2 (provides model code + sample video)
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git ~/cosmos-reason2

# Install dependencies
cd ~/cosmos-reason2 && export PATH=~/.local/bin:~/.cargo/bin:$PATH && uv sync --extra cu128

# Install PyAV (required on Hyperstack — FFmpeg not in PATH)
uv pip install "av==16.1.0"

# Download model weights (CR2-2B ~8GB, first-run only, ~5-10 min on cloud bandwidth)
export HF_TOKEN=hf_...
uv run huggingface-cli download nvidia/Cosmos-Reason2-2B \
  --local-dir ~/cosmos-reason2/models/Cosmos-Reason2-2B
```

### Step 3 — Deploy scripts and launch

Two scripts must be present on the instance:
- `/tmp/byo_video_setup.py` — shows live progress + ETAs, launches Gradio, prints clickable URL
- `/tmp/gradio_cr2_byo.py` — the Gradio app itself (called by setup script)

Both live at `~/.claude/scripts/` on Alex's Mac (canonical, versioned). Deploy via base64:

```bash
# Deploy both scripts to the instance
for script in byo_video_setup gradio_cr2_byo; do
  B64=$(base64 -i ~/.claude/scripts/${script}.py | tr -d '\n')
  brev exec <name> "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done
```

Then run setup (streams live progress to this terminal):
```bash
brev exec <name> "export HF_TOKEN=hf_... && \
  export PATH=~/.local/bin:~/.cargo/bin:\$PATH && \
  python3 /tmp/byo_video_setup.py"
```

The script will:
1. Detect GPU and auto-select model (CR2-2B ≥40GB, CR2-8B ≥80GB)
2. Check each prerequisite with ✓ / ⟳ status and ETA
3. Install anything missing (uv, repos, deps, model weights)
4. Launch Gradio with live model-load progress
5. Print a **clickable hyperlink** to the public `gradio.live` URL

### Step 4 — The URL appears at the end

Example output:
```
──────────────────────────────────────────────────────────────
  Cosmos Reason2 Demo — Ready
──────────────────────────────────────────────────────────────
  URL:  https://xxxxxxxxxxxx.gradio.live
  Upload any MP4 → type a prompt → click Run Inference
  Link valid for 72h. Kill instance when done.
──────────────────────────────────────────────────────────────
```

The URL is an OSC 8 hyperlink — click it directly in iTerm2 or Terminal.app (macOS). Valid for 72 hours.

### Step 5 — Kill alert (required)

When Alex is done:
```bash
/Users/asotelo/.nvcowork/bin/pa --agent tpm --print \
  "Send a Microsoft Teams message to chat ID 48:notes with this text: \
  'Cosmos Reason2 web demo done on brev/<name> (H100 PCIe). \
  Instance can be terminated now.'" \
  --max-duration 90
```

Do NOT auto-terminate. Notify Alex and wait for explicit kill confirmation.

---

## Local web demo (no cloud billing)

Same Gradio app, runs on the user's own GPU machine.

Prerequisites:
```bash
export HF_TOKEN=hf_...
# Check VRAM — must be ≥40GB for CR2-2B
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
```

Bootstrap (once):
```bash
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git ~/cosmos-reason2
cd ~/cosmos-reason2
uv sync --extra cu128
uv pip install "av==16.1.0" gradio
uv run huggingface-cli download nvidia/Cosmos-Reason2-2B \
  --local-dir ~/cosmos-reason2/models/Cosmos-Reason2-2B
```

Launch:
```bash
export MODEL_DIR=~/cosmos-reason2/models/Cosmos-Reason2-2B
export MODEL_NAME=nvidia/Cosmos-Reason2-2B
cd ~/cosmos-reason2
uv run python /tmp/gradio_cr2_byo.py
```

Open `http://localhost:7860` in any browser. No kill alert needed — local machine.

**Claude auth on local:** If running via Claude Code on a headless SSH machine, check auth first:
```bash
claude auth status
```
If `loggedIn: false`: `claude auth login --console` (API/console.anthropic.com users) or `claude auth login` (Claude.ai). On SSH — a URL prints; open it in your local browser.

---

## Headless inference (programmatic / no browser)

For CI or batch use where no browser is needed. Results go to JSON only.

```bash
export HF_TOKEN=hf_...
export BYO_VIDEO=/path/to/video.mp4
export MODEL_DIR=~/cosmos-reason2/models/Cosmos-Reason2-2B
export MODEL_NAME=nvidia/Cosmos-Reason2-2B
export OUT_FILE=/tmp/byo_video_reason2_results.json
export PROVIDER=brev  # or horde, local
cd ~/cosmos-reason2
uv run python /tmp/smoke_cr2_byo.py
```

Results at `/tmp/byo_video_reason2_results.json`.

---

## Horde (SSH-based — asotelo org)

Horde instances are created via REST API, accessed via SSH.

**Critical:** SSH username is `horde` — confirmed empirically 2026-04-17. Not `ubuntu`, `nvidia`, `root`, or `asotelo`.

Agent steps:
1. Create instance via Horde API v4 (`POST /api/v4/instances`) — or use existing `asotelo-uzof99`
2. Poll `GET /api/v4/instances/<id>` until `status: running`
3. Deploy scripts to instance (canonical source is `~/.claude/scripts/`):
   ```bash
   for script in byo_video_setup gradio_cr2_byo; do
     B64=$(base64 -i ~/.claude/scripts/${script}.py | tr -d '\n')
     ssh -i ~/.ssh/id_ed25519 horde@<ip> \
       "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
   done
   ```
4. Run setup (streams live output here):
   ```bash
   ssh -i ~/.ssh/id_ed25519 horde@<ip> \
     "export HF_TOKEN=hf_... && python3 /tmp/byo_video_setup.py"
   ```
5. URL prints at the end AND is written to `/tmp/gradio_url.txt` — read it back with `cat /tmp/gradio_url.txt` via ssh

**HOME on Horde is `/home/horde/`** — setup script uses `os.path.expanduser("~")` so it adapts automatically.

**Horde capacity API is stale.** Reports availability that doesn't match actual pool — confirmed across all SKUs as of 2026-04-17. Expect provisioning failures even when API shows GPUs available. Retry or use Brev. Existing instance `asotelo-uzof99` (A40) is the reliable fallback.

---

## Nebius (OpenAI-compatible API endpoint)

Nebius runs Reason2 as a vLLM serving endpoint — no SSH, no Gradio needed. Useful for API integration testing, not for interactive browser demos.

```python
from openai import OpenAI
client = OpenAI(base_url="https://<instance>.nebius.ai/v1", api_key="<nebius-key>")
response = client.chat.completions.create(
    model="nvidia/Cosmos-Reason2-2B",
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,<b64>"}},
        {"type": "text", "text": "Describe what is happening in this video."}
    ]}]
)
```

Write result to `/tmp/byo_video_reason2_results.json`. Send Teams kill alert when done.

---

## Gradio app script

Canonical source: **`~/.claude/scripts/gradio_cr2_byo.py`** (versioned 2026-04-18).

Features: PyAV backend patch, LOW_VRAM mode (fps=1, reduced resolution for <40GB VRAM), model header display, results saved to OUT_FILE.

Deploy to instances via base64 as shown in the deploy section. The setup script reads this file and deploys it to `/tmp/gradio_cr2_byo.py` on the remote instance.

**Do not embed the source here.** The canonical file is the source of truth.

---

## Cosmos cookbook structure (as of 2026-04-17)

The cookbook repo restructured. **`deploy/` directory no longer exists.** Old shell script paths are invalid.

| What | New location |
|---|---|
| Brev Reason2 setup script | `docs/getting_started/brev/reason2/setup_script.sh` |
| Worker safety recipe (Python) | `docs/recipes/inference/reason2/worker_safety/worker_safety.py` |
| Transfer2.5 real augmentation | `docs/recipes/inference/transfer2_5/inference-real-augmentation/inference.md` |
| Predict2 ITS | `docs/recipes/inference/predict2/inference-its/inference.md` |

For BYO-video inference, **do not use cookbook scripts** — use `gradio_cr2_byo.py` (web demo) or `smoke_cr2_byo.py` (headless) directly against the `cosmos-reason2` repo environment.

---

## PyAV backend patch (always apply)

Required on Hyperstack and most Horde images — FFmpeg is not in system PATH so torchcodec fails. Already embedded in both `gradio_cr2_byo.py` and `smoke_cr2_byo.py`.

If writing a new inference script, prepend this before loading the processor:

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

| Environment | GPU | Model | Tier | Load | Inference | Pass <60s? |
|---|---|---|---|---|---|---|
| Brev Hyperstack | H100 PCIe | CR2-2B | fps=2, 1M px | 1.7s | 44.1s | ✅ |
| Brev Hyperstack | H100 PCIe | CR2-2B | fps=2, 1M px | 10.6s | 42.9s | ✅ |
| Horde | A40 | CR2-2B | fps=1, 512K px | 2.9s | 76.9s | ❌ (old — no tier) |
| Horde | RTX PRO 6000 Blackwell (97GB) | CR2-2B | fps=1, 512K px | 3.4s | 54.8s | ✅ |
| Horde | RTX PRO 6000 Blackwell (97GB) | CR2-8B | fps=1, 1M px | 10.4s | >62s TTFT | ❌ |

Notes:
- "Inference" = preprocess + prefill + decode (TTFT-dominated for video inference)
- Old A40 result was without VRAM tier tuning; with fps=1 it would likely be ~55-65s
- CR2-8B fails the <60s target on all non-H100 GPUs tested; use CR2-2B for demos

---

## Common failure modes

| Symptom | Fix |
|---|---|
| Port 7860 unreachable | Check `ss -tlnp | grep 7860` on instance. If not listening, Gradio failed to start — check `/tmp/gradio_demo.log`. |
| Video upload fails in browser | Gradio temp dir issue. Try with sample.mp4 via Examples button first. |
| Black frames / torchcodec error | PyAV patch not applied. Confirm `gradio_cr2_byo.py` was deployed (not a custom script). |
| OOM during inference | VRAM too low. Kill other GPU processes first. CR2-2B needs ~10GB GPU RAM peak. |
| `uv sync --extra cu128` fails | Wrong CUDA driver. Check `nvidia-smi` shows CUDA 12.x driver. |
| Horde SSH rejected | Username must be `horde`. Not `ubuntu`, `nvidia`, `root`, or `asotelo`. SSH key: `~/.ssh/id_ed25519` (not a .pem file). |
| Horde "No GPUs available" | Capacity API is stale. Try different time window or use Brev instead. |
| `claude auth status` → loggedIn: false | `claude auth login --console` (API users). On SSH: copy URL, open in local browser. |
| Wrong VRAM tier selected (old Gradio using memory) | Setup script now kills port 7860 BEFORE measuring VRAM. Re-run setup to get clean tier. |
| Inference >60s on high-VRAM workstation GPU | GPU name doesn't match H100/A100/H200/GB200 → fps=1 tier applies. If inference still slow, check that `GRADIO_FPS=1` is in the Gradio process env. |
| HF 429 rate limit on download | Script retries 5× with 30s sleep. Common on shared Horde IP. Usually succeeds by attempt 3-4. |
