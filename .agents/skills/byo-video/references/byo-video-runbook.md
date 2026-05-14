# /byo-video — Cosmos BYO-Video Demo

> Compatibility note: this runbook contains legacy Claude Code tool names and
> `~/.claude/scripts` examples. Before executing, apply the adapter in
> `../SKILL.md`; for shared use, prefer the bundled `../scripts/` directory.

**Single-model deployment skill.** Deploys any supported model (Cosmos Reason2, Nemotron-Nano-12B-v2-VL, Qwen3-VL, etc.) to one of four frontend modes. When presenting a frontend picker, use these descriptive labels rather than raw implementation names, then map the selected option to `BYO_VIDEO_FRONTEND`:

- **Guided dataset batch UI** -> `BYO_VIDEO_FRONTEND=batch_inference` — guided browser UI for HF public dataset selection, concurrent video processing, worker-safety smoke testing, result export, and FiftyOne result writeback when available. Gradio still launches as a live sidecar.
- **Single-video upload UI** -> `BYO_VIDEO_FRONTEND=gradio` — default NVIDIA Build-style Gradio skin for upload-one-video/image workflows, prompt presets, reasoning on/off indicators, backend controls, and parameter tuning.
- **Dataset browser + result viewer** -> `BYO_VIDEO_FRONTEND=fiftyone` — guided batch UI with FiftyOne installed and available for dataset browsing, sample inspection, and result review. Gradio still launches as a live sidecar.
- **NVIDIA Build-style model playground (Default)** -> `BYO_VIDEO_FRONTEND=nvidia_build` — model-specific Gradio surface that follows the corresponding build.nvidia.com playground for Cosmos / Cosmos3 / Cosmos Predict / Cosmos Reason selections.

Default to `BYO_VIDEO_FRONTEND=nvidia_build` for shareable model-page demos or
unspecified frontend requests. Select `batch_inference` only when the user asks for
guided dataset loading, concurrent batch inference, worker-safety smoke testing,
or result writeback.

Every selection serves **Gradio web UI** on `GRADIO_PORT` (default `7860`) and writes `/tmp/gradio_url.txt` plus `/tmp/gradio_live.flag`. Dataset/batch selections additionally serve the **Batch Inference web UI** on `BATCH_INFERENCE_PORT` (default `7861`) and write `/tmp/byo_video_batch_inference_url.txt`. (Internal identifiers `batch_inference` / `BATCH_INFERENCE_PORT` are retained; user-visible prose says "batch inference".)

**Canonical scripts (stable, versioned — do not read from /tmp/):**
- `~/.claude/scripts/gradio_cr2_byo.py`  — default Build-style Gradio app (all supported models)
- `~/.claude/scripts/byo_video_setup.py` — Bootstrap + launch script
- `~/.claude/scripts/byo_video_batch_inference.py` — Batch Inference frontend for HF dataset batch inference and FiftyOne support (filename retains `batch_inference` for backward compat)
- `~/.claude/scripts/byo_video_runtime_guide.py` — friendly CLI guide for Claude Code-assisted dataset loads, guarded runs, prompt shaping, and exports

---

## NIM Message-Shape — Standing Order

**Every NIM payload must match the build.nvidia.com snippet for that model.** No `file://` URLs. No HTML-tag-in-text shapes. No client-side frame extraction when the model accepts `video_url` natively.

**Canonical shape (use this for every model the runbook supports unless build.nvidia.com publishes a different one):**

```python
{"role": "user", "content": [
  {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
  {"type": "text",      "text": prompt}
]}
```

This works on hosted (`integrate.api.nvidia.com`), local NIM (`nvcr.io/nim/...` on Brev port 8000), and vLLM-served checkpoints — without any server-side flag. The 2026-05-08 nem-12b incident showed `file://` returns HTTP 400 (`Cannot load local files without --allowed-local-media-path`) on default vLLM containers; build.nvidia.com hosted endpoints reject `file://` outright. Base64 `data:` URLs are the only shape that works everywhere.

**When adding a new model to `KNOWN_VLM_NIMS`:** fetch its build.nvidia.com snippet first (or `python3 ~/.claude/scripts/nim_catalog.py upstream`), copy the `messages` payload shape verbatim, then wire it in. Do not extrapolate from another model.

---

**Primary launch command (all environments):**
```bash
BYO_VIDEO_FRONTEND=batch_inference \
BATCH_INFERENCE_DATASET=pjramg/Safe_Unsafe_Test \
python3 /tmp/byo_video_setup.py
```
This script shows live step-by-step progress with ETAs for every install stage, then prints clickable hyperlinks for Gradio and any selected companion frontend plus the CLI guide command when deployed. The Gradio URL is always written to `/tmp/gradio_url.txt`; the batch-inference URL is written to `/tmp/byo_video_batch_inference_url.txt` when using the guided dataset batch UI or FiftyOne flow. Deploy it to the instance before running (see deploy section below).

### Batch Inference HF Dataset Smoke

Use this mode whenever the user asks for a guided alternative to Gradio, HF public dataset input, concurrent video processing, or FiftyOne support.

Required smoke dataset: `pjramg/Safe_Unsafe_Test`.

The Batch Inference frontend preloads the worker-safety system/user prompt from the recipe, lets the user change the HF dataset id and max videos, select any number of loaded videos, choose concurrency, and run all selected videos concurrently against the local vLLM/NIM OpenAI-compatible endpoint. When FiftyOne loaded the dataset, each result is written back to the sample as:

- `runtime_agent_response`
- `runtime_agent_json`
- `runtime_agent_prediction`
- `runtime_agent_error` when applicable

Headless smoke command after setup and vLLM/NIM are live:

```bash
cd ~/cosmos-reason2
VLLM_BASE_URL=http://localhost:8000/v1 \
uv run python /tmp/byo_video_batch_inference.py smoke \
  --dataset pjramg/Safe_Unsafe_Test \
  --max-videos 2 \
  --concurrency 2
```

Browser smoke path:

1. Open the Batch Inference URL.
2. Confirm `pjramg/Safe_Unsafe_Test` is in the dataset field.
3. Click **Run smoke** or load the dataset, select videos, then click **Run selected videos**.
4. Use **Open FiftyOne** after the dataset loads to inspect samples and batch-inference prediction fields (FiftyOne sample fields retain the `runtime_agent_*` field names for backward compat with existing writers).

---

## Cosmos3 OSS Backend Routing

Cosmos3 ships **two distinct architectures** under one family, served by **different stacks**:

| Cosmos3 model | Class | HF library | Backend | Picker `MODEL_SIZE` |
|---|---|---|---|---|
| `nvidia/Cosmos3-Nano-Reasoner` | Reasoner (chat VLM) | transformers, `qwen3_vl` | vLLM (existing path) | `C3-8B` |
| `nvidia/Cosmos3-Super-Reasoner` | Reasoner (chat VLM) | transformers, `qwen3_vl` | vLLM (existing path) | `C3-super` |
| `nvidia/Cosmos3-Nano` | Generator (diffusion video) | diffusers, `Cosmos3OmniDiffusersPipeline` | cosmos3 upstream package | `C3-NANO-GEN` |
| `nvidia/Cosmos3-Super` | Generator (diffusion video) | diffusers, `Cosmos3OmniDiffusersPipeline` | cosmos3 upstream package | `C3-SUPER-GEN` |

**Reasoners** load identically to other VLMs — `INFERENCE_BACKEND=vllm`, served on `localhost:8000`. Current HF API checks return auth-required for `nvidia/Cosmos3-Nano-Reasoner` and `nvidia/Cosmos3-Super-Reasoner`, so set `HF_TOKEN` with the required access before launch. For `MODEL_SIZE=C3-8B` and `BYO_VIDEO_FRONTEND=nvidia_build`, the primary surface is the Vite Build skin in `apps/nvidia-build-reason-vite` on port `5173`; Gradio still launches as the required fallback sidecar and writes `/tmp/gradio_url.txt` plus `/tmp/gradio_live.flag`.

**Predict / generator playgrounds** use the matching Vite Build skin in `apps/nvidia-build-predict-vite` on port `5174` when `BYO_VIDEO_FRONTEND=nvidia_build` and the selected model is a Cosmos Predict / Video2World / generator class. Set `BYO_VIDEO_LAUNCH_BATCH_INFERENCE=1` to keep the guided Batch Inference UI live alongside either Vite skin.

**Next.js source-capture frontends** live in `apps/nvidia-build-reason-next` and `apps/nvidia-build-predict-next`. They share the same `/api/models`, `/api/active-model`, `/api/reason`, and `/api/predict` contract as the Vite apps and are useful as editable reference captures; launch them manually on ports `3000` and `3001` when comparing against the Vite primaries.

**Generators** require the upstream `nvidia-cosmos/cosmos3` package and a different runtime:

```
git clone https://github.com/nvidia-cosmos/cosmos3.git ~/cosmos3
cd ~/cosmos3 && uv sync --all-extras --group=cu130-train
```

Then launch via the bundled helper `scripts/cosmos3_native_launch.sh` (deployed to `/tmp/cosmos3_native_launch.sh` on the target):

```bash
COSMOS3_CHECKPOINT=Cosmos3-Nano  bash /tmp/cosmos3_native_launch.sh   # or Cosmos3-Super
```

The helper starts `python -m cosmos3.ray.serve` on `:8000` and `python -m cosmos3.ray.gradio --host 0.0.0.0 --port 8080`, writes `/tmp/gradio_url.txt` (`http://<host>:8080`) and `/tmp/gradio_live.flag`, and leaves PIDs in `/tmp/cosmos3_serve.pid` and `/tmp/cosmos3_gradio.pid` for clean teardown.

**Smoke trace (horde@10.57.233.111, RTX PRO 6000 Blackwell, driver 575, 2026-05-12):**
- `uv sync --all-extras --group=cu130-train` completed (~5 min, +11 GB venv).
- `cosmos3.scripts.inference --help` and `cosmos3.ray.gradio --help` return clean.
- `import cosmos3` resolves to `/home/horde/cosmos3/cosmos3/__init__.py`.
- Full weight-download + Ray Serve boot deferred to follow-up commit (Cosmos3-Nano ~30 GB; Cosmos3-Super ~60 GB).

**Open integration items** (follow-up):

- `byo_video_setup.py` Step 9 / Step 10 currently branch only on `vllm`, `hf`, and `nim_local`. The `cosmos3_native` path is wired into `_MODEL_CONFIGS` and the runbook here, but Step 10 still calls into the legacy Gradio app. Until the setup script branches on `INFERENCE_BACKEND=cosmos3_native`, invoke the helper manually on the target after the install lands.
- `cosmos3.ray.gradio` exposes no `--share` flag, but `scripts/cosmos3_upload_gradio.py` (our wrapper) does call `ui.queue()` + `ui.launch(share=True)`. The gradio.live tunnel can fail to register on networks that block outbound frpc; if it does, the LAN URL still works (`http://<host>:8080`) and SSH port-forward (`ssh -L 8080:localhost:8080 <user@host>`) is the most VPN-tolerant fallback.

---

## CUDA wheel gotcha (cu130 vs cu128 — 2026-05-12 learning)

`uv sync --all-extras --group=cu130-train` is the README-recommended install. On a host whose NVIDIA driver reports CUDA runtime ≤ 12.9 (`nvidia-smi` shows `CUDA Version: 12.9` even on driver 575.x), the Ray Serve workers will crash on first inference with:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12090).
```

Switching the install group to `--group=cu128-train` is necessary but **not sufficient** — uv's group system swaps CUDA sidecar packages (torchvision, torchao, torchcodec) but does not re-pin PyTorch itself. After the group swap you get a torch+cu130 / torchvision+cu128 mismatch, producing:

```
RuntimeError: Detected that PyTorch and torchvision were compiled with different CUDA major versions.
PyTorch has CUDA Version=13.0 and torchvision has CUDA Version=12.8.
```

The fix is documented in `docs/setup.md` under "Advanced: custom torch/cuda versions" but easy to miss. Force-install both with an explicit backend pin:

```bash
cd ~/cosmos3
uv pip install 'torch==2.10.0' 'torchvision==0.25.0' --reinstall --torch-backend=cu128
```

Then verify it stuck — `torch.version.cuda` lies (it returns the package-metadata version, which is set in Python), so call the native check directly:

```bash
uv run --no-sync python -c \
  'import torch, torchvision; torchvision.extension._check_cuda_version(); print("OK")'
```

Finally, **the launcher must use `uv run --no-sync python -m ...`** when invoking `cosmos3.ray.serve` / `cosmos3.ray.gradio`. Without `--no-sync`, `uv run` auto-resolves dependencies on every invocation and re-installs the cu130 wheel from its cache, silently undoing the manual pin. The bundled `scripts/cosmos3_native_launch.sh` already passes `--no-sync`.

---

## i2v defaults + ETA (single-GPU horde @ RTX PRO 6000 Blackwell)

Source: internal benchmark `Cosmos 3 vs Cosmos Predict 2.5.xlsx` (pivot table) — wall-clock seconds to generate one video clip, CUDA Graph disabled, single-GPU configurations. Use these as your operator defaults when standing up a basic UI on horde.

### Recommended i2v defaults

| Field | Value | Why |
|---|---|---|
| Input preset | `i2v` | Loads prompt + a known-good `vision_path` from `inputs/omni/i2v.json` |
| Model | `Cosmos3-Nano` (only served checkpoint) | The Generator EA1 checkpoint |
| Resolution | `480p` | The PDF benchmark anchor for this GPU |
| Aspect ratio | `16:9` (default) | Form default; covered in the chart |
| FPS | `24` | Cinematic, inside the 10–30 supported range; doesn't move ETA much |
| Num frames | Form default (model default 189) | Lower this to halve diffusion time at the cost of clip duration |
| Seed | `0` | Pin for reproducibility; leave blank for random |
| Sampler | `unipc` | Default in `OmniSetupArgs` |
| Num steps | 35 (default) | What the live `cosmos3.ray.serve` uses |
| Vision path | URL or `/path/on/server` | The upload wrapper sets this automatically when you drop an image |

### ETA table (1× GPU, 480p i2v, steady-state)

From the internal Diffusion Speed leaderboard:

| GPU | C3-Nano (s) | Predict 2.5-2B (s) | Speedup |
|---|---|---|---|
| B300 | 12.84 | 83.59 | 6.51× |
| B200 | 13.69 | 25.60 | 1.87× |
| H200 141GB HBM3 | 25.41 | 43.89 | 1.73× |
| H100 80GB HBM3 | 25.51 | 44.81 | 1.76× |
| H200 NVL | 28.20 | 46.54 | 1.65× |
| H100 NVL | 35.15 | 58.92 | 1.68× |
| **RTX PRO 6000 Blackwell** | **68.80** | **74.30** | **1.08×** |
| H20 | 112.17 | 167.11 | 1.49× |

### Wall-clock expectations on RTX PRO 6000 Blackwell

| Scenario | Wall-clock | Notes |
|---|---|---|
| **First call** after server boot (cold) | 3 – 4 min | `torch.compile=True` JIT warmup. Cosmos3-Nano live run on 2026-05-12 took 3 min 51 s end-to-end. |
| **Steady-state** (compile cached) | ~70 s | The 68.8 s benchmark + ~2 s decode/I/O. |
| 720p instead of 480p (extrapolated) | ~140–180 s | Not in the chart for this GPU; ~2–2.5× the 480p time on other Blackwell rows. |
| `num_frames` halved (189 → 96) | ~½ of above | Diffusion is per-frame; cuts time roughly linearly. |

### Operator checklist for a basic UI demo

1. Confirm Ray Serve is warm before the demo — issue one disposable generation ≥ 5 min beforehand so `torch.compile` lands and the next click runs at steady-state.
2. Pin Resolution=480p, FPS=24, Seed=0 in the form. Reproducible + ~70 s per clip.
3. Pre-upload the conditioning image (the upload widget writes to `/tmp` and updates `vision_path` automatically; first-fetch from a remote URL adds 1–3 s).
4. If demo audience is on VPN: use the gradio.live URL when available (`ui.launch(share=True)` in `cosmos3_upload_gradio.py`) — long-poll reconnect tolerates short network drops mid-inference. If frpc is blocked, fall back to SSH port-forward.
5. The mp4 lands in `~/cosmos3/outputs/ray_serve/generate_<timestamp>_<short>/vision.mp4` on the server; the Gradio gallery streams it from `allowed_paths`.

---

## SKILL PROTOCOL — Hybrid: main session + observer

The main session owns **PHASE 0–1** (pre-checks + picker). These phases are interactive;
`AskUserQuestion` resolves all user decisions before anything runs on a GPU.

Immediately after the picker answers are resolved, the main session spawns a **background
observer** for **PHASE 2–6** (provision → shell ready → deploy scripts → setup → frontend live).
The observer handles all long-running work autonomously, keeping the main session free for
conversation. On completion the observer writes `/tmp/byo_video_observer_result.json`; the
main session reads it via the task completion notification and either displays the final URL
panel or fires `AskUserQuestion` for recovery.

**All user decisions happen in PHASE 1 before the observer spawns.** This includes the
"restart stopped instance vs provision fresh" choice — PHASE 0 data is used to build the Q3
options dynamically (see PICKER section).

**Never use Teams, Slack, or any external notification in this skill.** All user communication
happens in the Claude Code UI — panels, `AskUserQuestion`, and completion text.

### PHASE 0 — Pre-checks

0. **Clear stale session artifacts first** (one Bash call before anything else):
   ```bash
   rm -f /tmp/byo_video_observer_result.json /tmp/byo_video_progress.json
   ```
   These files persist across Claude Code sessions. If not cleared, the progress cron will
   read a prior session's result and report a false success or failure. This step is mandatory.

That is the entire PHASE 0. **There is no `brev ls` probe** — the picker presents Brev as one of three target options statically, and the observer handles `brev start`/`brev create` only after the user explicitly picks Brev. Skipping the probe (and the dynamic restart slot it enabled) trades one nice-to-have for: no Brev auth dependency at skill open, no false-start on logged-out brev CLI, and a clean code path for SSH-only / local-only users.

Display the initial panel without an instance list (the picker has every option anyway):

```
╔══════════════════════════════════════════════════════════════╗
║  Cosmos BYO-Video  ·  Pre-checks                             ║
╠══════════════════════════════════════════════════════════════╣
║  Stale-artifact cleanup: ✓                                   ║
╚══════════════════════════════════════════════════════════════╝
```

### PHASE 1 — Picker

Load `AskUserQuestion` via ToolSearch: `query: "select:AskUserQuestion"`, then fire all 3
questions in a single call (see PICKER section below for the full call).

**Q3 offers three options, always the same set (no dynamic slot):**

- `{ label: "New Brev instance", description: "Agent provisions appropriate GPU tier for your model — requires brev CLI login" }` → `DEPLOY_TARGET=brev:new`.
- `{ label: "SSH target", description: "Provide user@host or IP — any GPU machine you can SSH into" }` → follow-up AskUserQuestion for host.
- `{ label: "Local machine", description: "Run on this Mac — agent checks nvidia-smi locally first" }` → `DEPLOY_TARGET=local`.

If the user wants to restart a specific stopped Brev instance, they pick **New Brev instance** and the observer's brev path probes `brev ls` itself at PHASE 2 — moving the brev dependency out of pre-checks and into the path that actually uses it.

Once all answers are received, resolve `MODEL_ID`, `MODEL_SIZE`, `INFERENCE_BACKEND`,
`DEPLOY_TARGET` per the answer→env var mapping table.

If Q3 answer is "Existing Brev" or "SSH target", fire one follow-up `AskUserQuestion` to collect
the instance name or host.

Record `PROVISION_START_TS` immediately after all answers are in. Then spawn the observer
(see HANDOFF section) and stay available for conversation.

---

## PICKER — Main session PHASE 1

Run after PHASE 0. Load `AskUserQuestion` via ToolSearch first: `query: "select:AskUserQuestion"`

Then fire with all 3 questions in a single call:

```
AskUserQuestion({
  questions: [
    {
      question: "Backend?",
      header: "Backend",
      multiSelect: false,
      options: [
        { label: "vLLM (Recommended)", description: "~15–20 min setup, quantization support, fast inference" },
        { label: "HF Transformers", description: "~8–12 min setup, no quantization, simpler" },
        { label: "NIM (local Docker)", description: "Pull nvcr.io NIM container, serve OpenAI-compatible API on port 8000 — needs NGC_API_KEY + Docker on the target" }
      ]
    },
    {
      question: "Model?",
      header: "Model",
      multiSelect: false,
      options: [
        { label: "Cosmos Reason2 2B", description: "nvidia/Cosmos-Reason2-2B (public, ≥40GB VRAM)" },
        { label: "Cosmos Reason2 8B", description: "nvidia/Cosmos-Reason2-8B (public, ≥80GB VRAM)" },
        { label: "Cosmos Reason2 32B", description: "nvidia/Cosmos-Reason2-32B (public, ≥141GB VRAM — H200 required)" },
        { label: "Something else", description: "Cosmos 3 (private), Cosmos Transfer, Nemotron, non-NVIDIA models" }
      ]
    },
    {
      question: "Environment?",
      header: "Environment",
      multiSelect: false,
      options: [
        // Static 3-slot set — no Phase 0 brev probe.
        { label: "New Brev instance", description: "Agent provisions appropriate GPU tier — requires brev CLI login" },
        { label: "SSH target", description: "Provide user@host or IP — any GPU machine you can SSH into" },
        { label: "Local machine", description: "Run on this Mac — agent checks nvidia-smi locally" }
      ]
    }
  ]
})
```

**Answer → env var mapping:**

| Question | Answer | Sets |
|---|---|---|
| Q1 Backend | vLLM | `INFERENCE_BACKEND=vllm` |
| Q1 Backend | HF Transformers | `INFERENCE_BACKEND=hf` |
| Q1 Backend | NIM (local Docker) | `INFERENCE_BACKEND=nim_local` · `NIM_IMAGE=nvcr.io/nim/nvidia/<model-short-id>:latest` (resolved from MODEL_ID) · requires `NGC_API_KEY` |
| Q1 Backend | Cosmos3 native (auto) | `INFERENCE_BACKEND=cosmos3_native` — auto-selected when `MODEL_SIZE ∈ {C3-NANO-GEN, C3-SUPER-GEN}`; wraps the upstream `nvidia-cosmos/cosmos3` Ray Serve + Gradio stack. See "Cosmos3 OSS Backend Routing" below. |
| Q2 Model | Cosmos Reason2 2B | `MODEL_ID=nvidia/Cosmos-Reason2-2B` · `MODEL_SIZE=2B` |
| Q2 Model | Cosmos Reason2 8B | `MODEL_ID=nvidia/Cosmos-Reason2-8B` · `MODEL_SIZE=8B` |
| Q2 Model | Cosmos Reason2 32B | `MODEL_ID=nvidia/Cosmos-Reason2-32B` · `MODEL_SIZE=32B` |
| Q2 Model | Something else | Fire SOMETHING ELSE sub-picker (see below) |
| Q3 Env | New Brev instance | `DEPLOY_TARGET=brev:new` |
| Q3 Env | Existing Brev | Follow-up AskUserQuestion: "Instance name?" → `DEPLOY_TARGET=brev:<name>` |
| Q3 Env | SSH target | Follow-up AskUserQuestion: "user@host or IP?" → `DEPLOY_TARGET=ssh:<user@host>` |
| Q3 Env | Local machine | `DEPLOY_TARGET=local` |

**SOMETHING ELSE sub-picker** — fire immediately when user selects "Something else" for Q2. Because `AskUserQuestion` caps at 4 options per question, fan out by family:

```
AskUserQuestion({
  questions: [
    {
      question: "Which family?",
      header: "Family",
      multiSelect: false,
      options: [
        { label: "Cosmos 3 (OSS)", description: "Public Cosmos3 — Reasoners (chat VLM) or Generators (diffusion video)" },
        { label: "Cosmos Transfer / Nemotron", description: "Generation + multimodal — Cosmos Transfer 2.5, Nemotron-Nano-12B-v2-VL" },
        { label: "Non-NVIDIA models", description: "Best-effort only, not officially supported" }
      ]
    }
  ]
})
```

**If "Cosmos 3 (OSS)" → fire the Cosmos3 family picker:**

```
AskUserQuestion({
  questions: [
    {
      question: "Which Cosmos3 OSS model?",
      header: "Cosmos3",
      multiSelect: false,
      options: [
        { label: "Cosmos3-Nano-Reasoner",  description: "nvidia/Cosmos3-Nano-Reasoner — chat VLM, ~16 GB BF16, vLLM-served, HF_TOKEN may be required" },
        { label: "Cosmos3-Super-Reasoner", description: "nvidia/Cosmos3-Super-Reasoner — chat VLM, ~60 GB BF16, vLLM-served, HF_TOKEN may be required" },
        { label: "Cosmos3-Nano (Generator)",  description: "nvidia/Cosmos3-Nano — diffusion video gen (t2i/t2v/i2v), ~30 GB; needs cosmos3 upstream package" },
        { label: "Cosmos3-Super (Generator)", description: "nvidia/Cosmos3-Super — diffusion video gen (t2i/t2v/i2v), ~60 GB; needs cosmos3 upstream package" }
      ]
    }
  ]
})
```

| Selection | Sets |
|---|---|
| Cosmos3-Nano-Reasoner | `HF_TOKEN=<token>` · `MODEL_ID=nvidia/Cosmos3-Nano-Reasoner` · `MODEL_SIZE=C3-8B` · `INFERENCE_BACKEND=vllm` |
| Cosmos3-Super-Reasoner | `HF_TOKEN=<token>` · `MODEL_ID=nvidia/Cosmos3-Super-Reasoner` · `MODEL_SIZE=C3-super` · `INFERENCE_BACKEND=vllm` |
| Cosmos3-Nano (Generator) | `MODEL_ID=nvidia/Cosmos3-Nano` · `MODEL_SIZE=C3-NANO-GEN` · `INFERENCE_BACKEND=cosmos3_native` |
| Cosmos3-Super (Generator) | `MODEL_ID=nvidia/Cosmos3-Super` · `MODEL_SIZE=C3-SUPER-GEN` · `INFERENCE_BACKEND=cosmos3_native` |

**If "Cosmos Transfer / Nemotron" → fire:**

```
AskUserQuestion({
  questions: [
    {
      question: "Which model?",
      header: "Model",
      multiSelect: false,
      options: [
        { label: "Cosmos Transfer 2.5", description: "nvidia/Cosmos-Transfer2.5 (generation model, ≥80GB VRAM)" },
        { label: "Nemotron-Nano-12B-v2-VL", description: "nvidia/Nemotron-Nano-12B-v2-VL-BF16 (12B, gated, vLLM only, ≥40GB VRAM)" }
      ]
    }
  ]
})
```

| Selection | Sets |
|---|---|
| Cosmos Transfer 2.5 | `MODEL_ID=nvidia/Cosmos-Transfer2.5` · `MODEL_SIZE=32B` |
| Nemotron-Nano-12B-v2-VL | `MODEL_ID=nvidia/Nemotron-Nano-12B-v2-VL-BF16` · `MODEL_SIZE=NEM-12B` |

**If "Non-NVIDIA models":** show disclaimer inline, then fire Qwen sub-picker.

**Non-NVIDIA disclaimer (display inline before sub-picker):**
> ⚠️ Non-NVIDIA models: NVIDIA does not officially support or guarantee setup for third-party models. This is best-effort only.

**Qwen sub-picker:**

```
AskUserQuestion({
  questions: [
    {
      question: "Which non-NVIDIA model?",
      header: "Model",
      multiSelect: false,
      options: [
        { label: "Qwen3-VL-2B-Instruct", description: "Qwen/Qwen3-VL-2B-Instruct (public, ~8GB VRAM)" },
        { label: "Qwen3-VL-8B-Instruct", description: "Qwen/Qwen3-VL-8B-Instruct (public, ~20GB VRAM)" },
        { label: "Qwen3-VL-32B-Instruct", description: "Qwen/Qwen3-VL-32B-Instruct (public, ~64GB VRAM)" }
      ]
    }
  ]
})
```

| Selection | Sets |
|---|---|
| Qwen3-VL-2B-Instruct | `MODEL_ID=Qwen/Qwen3-VL-2B-Instruct` · `MODEL_SIZE=QW3-2B` |
| Qwen3-VL-8B-Instruct | `MODEL_ID=Qwen/Qwen3-VL-8B-Instruct` · `MODEL_SIZE=QW3-8B` |
| Qwen3-VL-32B-Instruct | `MODEL_ID=Qwen/Qwen3-VL-32B-Instruct` · `MODEL_SIZE=QW3-32B` |

Once `MODEL_ID`, `MODEL_SIZE`, `INFERENCE_BACKEND`, `DEPLOY_TARGET` are all resolved: record
`PROVISION_START_TS` and spawn the observer (see HANDOFF section below).

---

## EXECUTION PROTOCOL — Phase 1 (main session) + Phases 2–6 (observer)

The main session executes only PHASE 0–1. After the picker resolves all answers it spawns the
observer and stays free for conversation.

**PROVISION_START_TS** is recorded in PHASE 1 immediately after all picker answers are in and
passed to the observer as part of the spawn prompt.

The observer handles PHASE 2–6. It outputs clean checklist panels — commands run silently
underneath. See **OBSERVER PROTOCOL** section for the observer's checklist format and
per-phase instructions.

---

### HANDOFF — Spawn observer after PHASE 1

After all picker answers are resolved, display a brief handoff message and spawn the observer:

```
Setting up your demo — observer is running in the background (~15–20 min).
I'll show you the frontend URL when it's live. Feel free to keep chatting.
```

Spawn:

```
Agent({
  subagent_type: "general-purpose",
  run_in_background: True,
  name: "byo-video-observer",
  prompt: """
OBSERVER TASK — Cosmos BYO-Video PHASE 2–6

You are a background observer. Read the full OBSERVER PROTOCOL section of the /byo-video
skill. Execute PHASE 2 through PHASE 6 exactly as described there.

Inherited state (fill in actual values):
  DEPLOY_TARGET:       <brev:new | brev:<name> | ssh:<user@host> | local>
  MODEL_ID:            <model_id>
  MODEL_SIZE:          <model_size>
  INFERENCE_BACKEND:   <backend>
  RATE:                <rate_per_hour>
  PROVISION_START_TS:  <epoch_seconds>

On success write to /tmp/byo_video_observer_result.json on the LOCAL machine:
  {"status":"live","url":"<frontend_url>","elapsed_s":<N>,"cost":<N>,"instance":"<name>","rate":<rate>,"gpu":"<gpu_label>","model_id":"<model_id>","model_size":"<model_size>","backend":"<backend>","frontend":"<batch_inference|gradio|fiftyone>"}

On unrecoverable failure write:
  {"status":"failed","phase":<N>,"error":"<one-line error>","instance":"<name>"}

Then exit.
"""
})
```

**After spawning the observer, immediately start the progress loop:**

Load `CronCreate` via ToolSearch: `query: "select:CronCreate"`, then create:

```
CronCreate({
  cron: "*/1 * * * *",
  prompt: "Check /byo-video observer progress: read /tmp/byo_video_progress.json and /tmp/byo_video_observer_result.json on the local machine; display a one-line status update with the current checklist phase and elapsed time. If result JSON exists with status='live', display the LIVE final panel and cancel this cron job (load CronDelete via ToolSearch first). If result JSON exists with status='failed', fire AskUserQuestion with recovery options (see HALT-AND-ASK) and cancel this cron job.",
  recurring: true
})
```

Store the returned job ID as `PROGRESS_LOOP_ID`. Cancel it with `CronDelete(PROGRESS_LOOP_ID)` when the observer completes (success or failure).

After spawning: stay available for conversation.
When the task completion notification fires, OR when the progress loop detects a result JSON, read `/tmp/byo_video_observer_result.json`:
- `status == "live"` → display the LIVE final panel (see PHASE 6 completion template). Cancel the progress loop.
- `status == "failed"` → fire `AskUserQuestion` with recovery options (see HALT-AND-ASK). Cancel the progress loop.

---

### OBSERVER PROTOCOL — Standing Rule

> **OBSERVER RULE: No silent exits.**
> On ANY unrecoverable error — SSH failure, brev create failure, vLLM error, setup script error, all providers exhausted — the observer MUST:
> 1. Write `/tmp/byo_video_observer_result.json` on the local machine with `{"status":"failed","phase":<N>,"error":"<one-line>","instance":"<name>"}`.
> 2. Then `exit`.
>
> Never exit without writing this file. The main session's progress loop reads it every minute — an absent file means "still running," so a silent exit leaves the user waiting forever.

---

### OBSERVER PROTOCOL — PHASE 2: PROVISION

**This phase and all subsequent phases run inside the observer. Do not prompt the user.**

**HF_TOKEN:** Auto-read by the setup script from `~/.cache/huggingface/token` on the remote
instance. Do NOT ask. Do NOT pass as a CLI arg.

**For `DEPLOY_TARGET=brev:new`:**
1. Look up MODEL_SIZE in MODEL_GPU_REQUIREMENTS to select provider type.
2. Run `brev create <name> --type <provider_type>` — one Bash call.
   Name convention: `cr2-<modelsize>-<timestamp-short>` (e.g., `cr2-2b-0505`)
3. `brev create` blocks until shell ready. Proceed to PHASE 4.

**CRITICAL — `cloudCredId` error halt:** If `brev create` output contains `cloudCredId or workspaceGroupId must be specified on request`, do NOT rotate to a fallback provider — this error is an org-level credential gap that applies to all provider types. Write failure JSON immediately and exit:
`{"status":"failed","phase":2,"error":"Brev org missing cloud credential — brev create blocked for all providers. Fix in Brev dashboard org settings.","instance":"<name>"}`

**For `DEPLOY_TARGET=brev:<name>` (restart stopped instance):**
1. Run `brev start <name>` — one Bash call.
2. Poll `brev ls` every 30s until STATUS = RUNNING and SHELL = READY. Then proceed to PHASE 4.

**For `DEPLOY_TARGET=ssh:<user@host>`:**
1. Verify: `ssh -i ~/.ssh/id_ed25519 <user@host> "echo ok"`
2. Proceed directly to PHASE 4 on success.

**Observer checklist** — emit at the start of PHASE 2 and reprint (with updates) on each
phase transition and every 30s poll:

```
╔══════════════════════════════════════════════════════════════╗
║  Cosmos BYO-Video · setting up...  (elapsed: Xm Ys)          ║
║  <GPU label> · $<rate>/hr · <MODEL_ID> (<MODEL_SIZE>, <be>)  ║
╠══════════════════════════════════════════════════════════════╣
║  [→] Provisioning instance                                   ║
║  [ ] Shell ready                                             ║
║  [ ] Scripts deployed                                        ║
║  [ ] Installing dependencies          ETA ~8 min             ║
║  [ ] Downloading model weights        ETA ~5 min             ║
║  [ ] Starting Gradio                  ETA ~1 min             ║
╠══════════════════════════════════════════════════════════════╣
║  Cost so far: $0.00  |  Est. total setup: ~$<low>–$<high>    ║
║  Brev dashboard: https://brev.dev                            ║
╚══════════════════════════════════════════════════════════════╝
```

Mark each row `[✓]` when complete, `[→]` when active. Update elapsed and cost each reprint.
Raw bash output is never surfaced in the checklist.

Write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 2, "status": "provisioning", "elapsed_s": 0, "instance": "<name>", "checklist": {"provision": "active", "shell": "pending", "scripts": "pending", "deps": "pending", "weights": "pending", "gradio": "pending"}}
```

Cost estimates by GPU tier:
- H100 SXM ($3.54/hr): 10 min setup ~$0.59 · 20 min ~$1.18
- H200 SXM ($4.20/hr): 20 min setup ~$1.40 · 30 min ~$2.10

---

### OBSERVER PROTOCOL — PHASE 3: WAIT FOR SHELL READY

Applies when `brev create` did not already block until shell ready (e.g., restart path).
Poll `brev ls` every 30s. Update the checklist `[→] Shell ready` row each poll.

**On UNHEALTHY:**
1. Update checklist: `[!] UNHEALTHY — recovering (3 min window)`.
2. Poll every 30s for 3 minutes.
3. If recovered: run `brev exec <name> "nvidia-smi"` — if GPU responds, continue.
4. If still UNHEALTHY after 3 minutes:
   - Try `brev reset <name>`. If succeeds → re-enter poll loop.
   - If reset fails: `brev delete <name>`, rotate to next provider (PROVIDER FALLBACK table).
     Emit one line: `✗ <type> UNHEALTHY — deleted, trying <next-type>`. Re-enter PHASE 2.
   - If all providers exhausted: write failure result and exit.
     `{"status":"failed","phase":3,"error":"All providers exhausted — UNHEALTHY","instance":"<last-name>"}`

On each poll, write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 3, "status": "waiting_shell", "elapsed_s": <N>, "instance": "<name>", "checklist": {"provision": "done", "shell": "active", "scripts": "pending", "deps": "pending", "weights": "pending", "gradio": "pending"}}
```

On SHELL READY: update checklist `[✓] Shell ready`, proceed to PHASE 4.

---

### OBSERVER PROTOCOL — PHASE 4: DEPLOY SCRIPTS

Deploy the BYO-video scripts from the active skill-local scripts directory. Prefer
`.agents/skills/byo-video/scripts/` in this repository; fall back to legacy
`~/.claude/scripts/` only when running outside the repo.

**Step 0 — Auto-deploy HF token (before scripts, always):**
Check if a cached HF token exists locally. If so, deploy it to the instance so the setup script can authenticate without user interaction.
```bash
# Check local token
cat ~/.cache/huggingface/token 2>/dev/null || echo "NO_TOKEN"
```
If a token is found (starts with `hf_`):
```bash
brev exec <name> "mkdir -p ~/.cache/huggingface"
brev exec <name> "echo <token> > ~/.cache/huggingface/token"
```
This prevents the Step 2 HF auth failure that requires a full setup restart.

```bash
SCRIPT_DIR="${COSMOS_AGENT_SCRIPTS_DIR:-$PWD/.agents/skills/byo-video/scripts}"
```

**Step 1** — Deploy required Python scripts:

```bash
for script in byo_video_setup gradio_cr2_byo gradio_cosmos_predict gradio_cosmos_reason_build byo_video_batch_inference byo_video_runtime_guide; do
  B64=$(base64 -i "$SCRIPT_DIR/${script}.py" | tr -d '\n')
  brev exec <name> "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done
```

**Step 2** — Deploy optional NIM helpers when `INFERENCE_BACKEND=nim_local`:

```bash
B64=$(base64 -i "$SCRIPT_DIR/nim_catalog.py" | tr -d '\n')
brev exec <name> "python3 -c \"import base64; open('/tmp/nim_catalog.py','wb').write(base64.b64decode('${B64}'))\""
B64=$(base64 -i "$SCRIPT_DIR/nim_launch.sh" | tr -d '\n')
brev exec <name> "python3 -c \"import base64; open('/tmp/nim_launch.sh','wb').write(base64.b64decode('${B64}'))\" && chmod +x /tmp/nim_launch.sh"
```

If a script exceeds the shell argument buffer, use `brev copy "$SCRIPT_DIR/<script>" <name>:/tmp/<script>` instead of base64.

Update checklist: `[✓] Scripts deployed`

Write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 4, "status": "scripts_deployed", "elapsed_s": <N>, "instance": "<name>", "checklist": {"provision": "done", "shell": "done", "scripts": "done", "deps": "pending", "weights": "pending", "gradio": "pending"}}
```

---

### OBSERVER PROTOCOL — PHASE 4-NIM: NIM-LOCAL ADDENDUM

**Only when `INFERENCE_BACKEND=nim_local`. Otherwise skip to Phase 5.**

The NIM (local Docker) backend pulls a NIM container from `nvcr.io/nim/nvidia/<model-short>:latest` and runs it on the target's port 8000. The Gradio app talks to the container via the standard OpenAI-compatible client (same code path as vLLM, just a different `VLLM_BASE_URL`). Works on any reachable target — Brev, SSH host, or local Docker — provided NGC_API_KEY and Docker are present. The user supplies the target via the Phase 1 picker; the skill never assumes a default host.

**Source of truth for available VLM NIMs — fetch this URL every time the user selects NIM:**

> [https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html](https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html)

The `~/.claude/scripts/nim_catalog.py` helper does this automatically (and is also deployed to `/tmp/nim_catalog.py` on the target). The agent must:

1. **In the Phase 1 picker (main session)** — when the user selects "NIM (local Docker)" as the backend, run `python3 ~/.claude/scripts/nim_catalog.py upstream` to refresh the canonical model list from the URL above. If a model name appears upstream that is NOT in `KNOWN_VLM_NIMS` (the slug map inside `nim_catalog.py`), surface a one-line note to the user (e.g., *"Heads-up: docs added `Foo VLM`; the byo-video skill doesn't know its nvcr.io short-id yet — file a one-line PR adding it to KNOWN_VLM_NIMS, or use Custom Checkpoint ID."*).
2. **In observer Phase 4 (script deploy)** — deploy `nim_catalog.py` alongside the other scripts (see deploy block below).
3. **In runtime debug (the morning batch-inference loop)** — re-run `nim_catalog.py upstream` each session start to detect upstream catalog changes (deprecations, new models, version bumps).

**Runtime NIM swap — out-of-band (not via a Gradio button):**

A previous version (commit `4ed5951`) wired a "Switch to selected NIM" button inside Gradio that streamed `nim_launch.sh` output to the browser. It was removed because a 5-15 min docker pull + 1-3 min vLLM warmup blows past Gradio's streaming heartbeat, so the UI froze even when the backend was making progress. The dropdown still lists every VLM NIM in the upstream catalog (so the user knows their options), but the swap itself is performed out-of-band by one of:

1. **Runtime agent path (preferred)** — when the user says *"switch the NIM to X"*, the agent runs the SSH commands below, watches `docker logs -f cosmos-nim`, confirms `/v1/models` is back up, and tells the user to refresh the Gradio page. Gradio re-queries `/v1/models` on every page load and picks up the new served model id automatically.

2. **Manual SSH path (no Claude needed)** — the Gradio info panel in `nim_local` mode displays this snippet directly:
   ```bash
   ssh <user@host>
   docker rm -f cosmos-nim
   MODEL=<short-id> CONTAINER_NAME=cosmos-nim PORT=8000 \
     NGC_API_KEY="$NGC_API_KEY" bash /tmp/nim_launch.sh
   # Then reload the Gradio page.
   ```
   Valid short-ids come from `python3 /tmp/nim_catalog.py list --no-probe` (or the static panel inside Gradio).

**Agent runbook for "switch the NIM" requests:**
1. Run `python3 ~/.claude/scripts/nim_catalog.py upstream` to refresh the catalog from `docs.nvidia.com/nim/vision-language-models/latest/introduction.html`. Surface any upstream model name not in `KNOWN_VLM_NIMS` as a one-line note.
2. Resolve the user's request (e.g. *"Cosmos Reason2 2B"*) to a short-id (e.g. `cosmos-reason2-2b`) via the slug map in `nim_catalog.py`.
3. SSH to the target. Run `docker rm -f cosmos-nim` then `MODEL=<short-id> NGC_API_KEY="$NGC_API_KEY" bash /tmp/nim_launch.sh`. Stream the log so the user can see progress. Do not put credentials in argv.
4. Verify with `curl -sf http://localhost:8000/v1/models | jq '.data[0].id'` — confirm the served model id matches.
5. Tell the user to refresh the Gradio page.

**Required env on the target:**
- `NGC_API_KEY` (must start with `nvapi-`) — used by both `docker login nvcr.io` and the running container
- `HF_TOKEN` — **not needed**; NIM ships the model

**Deploy `nim_launch.sh` AND `nim_catalog.py` alongside the other scripts (Phase 4 step 1):**
```bash
# nim_launch.sh
brev exec <name> "python3 -c \"import base64; open('/tmp/nim_launch.sh','wb').write(base64.b64decode('<B64_NIM_SH>'))\""
brev exec <name> "chmod +x /tmp/nim_launch.sh"
# nim_catalog.py — used by Gradio at startup AND by the batch inference
brev exec <name> "python3 -c \"import base64; open('/tmp/nim_catalog.py','wb').write(base64.b64decode('<B64_NIM_CATALOG>'))\""
```
(For SSH targets: replace `brev exec <name>` with `ssh -i ~/.ssh/id_ed25519 <user@host>`.)

**`byo_video_setup.py` handles the NIM launch automatically** when `INFERENCE_BACKEND=nim_local`:
- Step 2/2b — HF auth is **skipped**
- Step 3 — NGC_API_KEY is **mandatory** (script exits if missing)
- Step 9 (HF weights download) — **skipped**
- Step 9-NIM (new) — runs `bash /tmp/nim_launch.sh` which:
  1. Reuses an existing healthy container with the same name (idempotent)
  2. Otherwise: `docker login nvcr.io`, `docker pull` the image, `docker run -d` per official build.nvidia.com style (`--gpus all --ipc host --shm-size=32GB --ulimit memlock=-1 --ulimit stack=67108864 -e NGC_API_KEY -p 8000:8000`). It uses container-internal `/opt/nim/.cache` by default; set `NIM_CACHE_MODE=host` only when you intentionally want to bind-mount `$LOCAL_NIM_CACHE`.
  3. Forwards known per-NIM env overrides such as `NIM_MAX_MODEL_LEN`, `NIM_MODEL_PROFILE`, `NIM_MEDIA_IO_KWARGS`, and comma-separated `NIM_EXTRA_ENV=KEY=VALUE,...`.
  4. Waits up to 1800s for `GET /v1/models` by default; Omni and Gemma use 2400s because first boot can run 20-30 min.
- Step 10 — Gradio always launches with `VLLM_BASE_URL=http://localhost:8000/v1`; Batch Inference / FiftyOne selections launch their companion UI after Gradio is live. The Gradio and Batch Inference frontends auto-detect the served model name via `/v1/models` (so `_SERVER_MODEL_ID` or the Batch Inference `server.model` field matches the NIM-served id, e.g. `nvidia/cosmos-reason2-8b`).

**NIM image short-id resolution (in `byo_video_setup.py`):**
```
MODEL_ID=nvidia/Cosmos-Reason2-8B   →  short=cosmos-reason2-8b   →  nvcr.io/nim/nvidia/cosmos-reason2-8b:latest
MODEL_ID=nvidia/Cosmos-Reason2-2B   →  short=cosmos-reason2-2b   →  nvcr.io/nim/nvidia/cosmos-reason2-2b:latest
MODEL_ID=nvidia/Cosmos-Reason2-32B  →  short=cosmos-reason2-32b  →  nvcr.io/nim/nvidia/cosmos-reason2-32b:latest
MODEL_ID=google/gemma-4-31b-it      →  short=gemma-4-31b-it      →  nvcr.io/nim/google/gemma-4-31b-it:latest
```
`byo_video_setup.py` resolves vendor/image/env through `nim_catalog.py`, so non-NVIDIA vendor images such as Gemma use the correct `nvcr.io/nim/google/...` path. Override at any time with `NIM_MODEL_SHORT` or full `NIM_IMAGE` env var.

**NIM payload rules learned from the smoke sprints:**
- Send base64 `data:` `video_url` first for every video-capable NIM. No `file://`.
- Do not send `max_tokens` to NIM `/v1/chat/completions`; server `max_model_len` governs.
- Do not enforce client-side caps on frames, tokens, fps, pixels, or resolution. If a NIM cannot handle the request, let it return the service-owned 4xx.
- Cosmos Reason1 7B is the known exception: it rejects native `video_url`, so Gradio / Batch Inference retry with image-frame fallback after a 400/422.
- Nemotron Nano frame mode has a 5-image prompt limit, which is why the default path must be native `video_url`.

**NIM-8B-FP8-THINK-EOS bug (greedy decode):** the FP8-quantized cosmos-reason2-8b NIM emits a bare `<think>` opener then an EOS-like token at `temperature=0`, finishing in 2-3 tokens with no reasoning trace and no final answer. Visible symptom in Gradio: response shows only `<think>` (or appears empty) and the run completes in <1s with `tok=2` or `tok=3` in `gradio_demo.log`. Workaround: keep temperature ≥ 0.3. The Gradio app defaults the slider to 0.6 in `nim_local` mode and clamps server-side calls to ≥0.3 as a safety net. Runtime monitor rule `nim_local_think_eos_truncation` flags any `[vllm done] X.Xs · 1|2|3 tok` line.

**Failure modes the runtime monitor catches** (`byo_video_runtime_monitor.py` rule names):
- `nim_local_container_down` — Gradio reports `[NIM] Container not responding at` (port 8000 unreachable)
- `nim_local_image_unauthorized` — NGC denied the pull (key invalid or model not allowlisted)
- `vllm_disconnect` — generic port-8000 failure (covers both vLLM and NIM-local container)

---

### OBSERVER PROTOCOL — PHASE 5: SETUP LAUNCH + LOG TAIL

**This phase runs inside the observer subagent, not the main session.**

Record `SETUP_DISPATCHED_AT` = now.

**Pre-launch VRAM check (mandatory — runs before setup):**
```bash
brev exec <name> "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1"
```
(For SSH: replace `brev exec <name>` with `ssh -i ~/.ssh/id_ed25519 <user@host>`.)

Look up `MODEL_GPU_REQUIREMENTS[MODEL_SIZE].min_vram` in MB (multiply GB values by 1000):
`2B`→40000 · `8B`→80000 · `32B`→141000 · `C3-8B`→40000 · `C3-32B`→141000 · `NEM-12B`→40000 · `QW3-2B`→8000 · `QW3-8B`→20000 · `QW3-32B`→64000 · `C3-super`→141000

If `free_vram_mb < min_vram_mb`:
  Write to `/tmp/byo_video_observer_result.json` on the LOCAL machine and exit immediately:
  ```json
  {"status":"failed","phase":5,"error":"insufficient_vram","vram_free_mb":<free>,"vram_needed_mb":<needed>,"model_size":"<MODEL_SIZE>","instance":"<name>"}
  ```
  Do NOT launch setup. The main session handles model switching via AskUserQuestion (see HALT-AND-ASK).

If `free_vram_mb >= min_vram_mb`: proceed with the user's requested model — do not substitute.

Launch setup (one Bash call — nohup so brev exec returns immediately). Pass `MODEL_NAME` and `MODEL_SIZE` explicitly so the setup script does not auto-select a different model:
```bash
brev exec <name> "nohup bash -c 'export INFERENCE_BACKEND=<backend> MODEL_ID=<model_id> MODEL_NAME=<model_id> MODEL_SIZE=<model_size> BYO_VIDEO_FRONTEND=<nvidia_build|batch_inference|gradio|fiftyone> BATCH_INFERENCE_DATASET=pjramg/Safe_Unsafe_Test BREV_RATE_PER_HOUR=<rate> PATH=~/.local/bin:~/.cargo/bin:$PATH && python3 /tmp/byo_video_setup.py > /tmp/byo_video_setup.log 2>&1' &"
```

**Log tail loop (every 30s):** Tail the log, print a compact status line (not a full panel —
the observer is headless). Look for step markers and errors. On each poll, write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 5, "status": "<active_step>", "elapsed_s": <N>, "instance": "<name>", "checklist": {"provision": "done", "shell": "done", "scripts": "done", "deps": "active|done", "weights": "active|done|pending", "gradio": "pending"}}
```
(Update `deps` to `"done"` after Step 6 completes, `weights` to `"done"` after Step 9 completes.)

```bash
brev exec <name> "tail -60 /tmp/byo_video_setup.log 2>/dev/null || echo '(log not yet written)'"
```

Step markers to detect (scan log for these strings):

```
Step 1: GPU detect       Step 2: HF auth         Step 3: NGC API key
Step 4: uv               Step 5: cosmos-reason2  Step 6: uv sync
Step 7: PyAV/frontend    Step 9: weights          Step 10: frontend launch
```

**Error detection:** Scan for `Traceback`, `Error:`, `exit status 1`, `FAILED`.

On error:
1. Retry once: re-run the setup launch command, re-enter log tail loop.
2. If error recurs: write failure result and exit:
   ```json
   {"status":"failed","phase":5,"error":"<one-line error>","instance":"<name>"}
   ```
   Write to `/tmp/byo_video_observer_result.json` on the local machine, then exit.
   The main session handles user recovery via `AskUserQuestion`.

---

### OBSERVER PROTOCOL — PHASE 6: FRONTEND LIVE + URL CAPTURE

Update checklist: `[→] Starting frontend`. Poll every 30s for the live flag. Every mode writes `/tmp/gradio_live.flag`; Batch Inference and FiftyOne modes also write `/tmp/byo_video_batch_inference_live.flag` (filename retains the `batch_inference` token for backward compat). On each poll, write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 6, "status": "waiting_frontend", "elapsed_s": <N>, "instance": "<name>", "checklist": {"provision": "done", "shell": "done", "scripts": "done", "deps": "done", "weights": "done", "frontend": "active"}}
```

```bash
brev exec <name> "cat /tmp/gradio_live.flag 2>/dev/null; cat /tmp/byo_video_batch_inference_live.flag 2>/dev/null"
```

When the Gradio flag is present, read the URLs:
```bash
brev exec <name> "printf 'gradio='; cat /tmp/gradio_url.txt 2>/dev/null; printf 'batch_inference='; cat /tmp/byo_video_batch_inference_url.txt 2>/dev/null || true"
```

**Compute total cost:** `(time.time() - PROVISION_START_TS) / 3600 * rate_per_hour`

Write success result to `/tmp/byo_video_observer_result.json` on the **local machine**:
```json
{"status":"live","url":"<primary_frontend_url>","gradio_url":"<gradio_url>","runtime_agent_url":"<runtime_agent_url_or_null>","elapsed_s":<N>,"cost":<computed>,"instance":"<name>","rate":<rate>,"gpu":"<gpu_label>","model_id":"<model_id>","model_size":"<model_size>","backend":"<backend>","frontend":"<batch_inference|gradio|fiftyone>"}
```
Then exit. The main session reads this file on task completion and displays the final panel.

**LIVE final panel** (displayed by main session on success):

```
╔══════════════════════════════════════════════════════════════╗
║  Cosmos BYO-Video  ·  LIVE  ✓                                ║
║  <GPU label> · $<rate>/hr · elapsed: <N>m <S>s               ║
╠══════════════════════════════════════════════════════════════╣
║  [✓] Pre-checks                                              ║
║  [✓] Model & environment selected                            ║
║  [✓] Instance: <name> (<GPU type>)                           ║
║  [✓] SHELL READY                                             ║
║  [✓] Scripts deployed                                        ║
║  [✓] Setup complete                                          ║
║  [✓] Frontend live                                           ║
╠══════════════════════════════════════════════════════════════╣
║  Gradio URL: <gradio_url>                                    ║
║  Runtime URL: <runtime_agent_url_or_blank>                   ║
║  Total setup cost: ~$<computed>  |  Link valid 72h           ║
║  Kill the Brev instance when you're done to stop billing.    ║
╚══════════════════════════════════════════════════════════════╝
```

Do NOT auto-terminate the instance.

---

### POST-LIVE — Guided Companion (recommended for casual users)

After the LIVE panel, offer a friendlier post-deployment helper before handing
the user the expert browser UI. This helper is for users who want Claude Code to
talk them through the same capabilities as the Batch Inference HTML screen:
dataset load, paper import, prompt selection, context guard, guarded batch run,
progress, result review, and export.

Fire `AskUserQuestion`:

```
AskUserQuestion: "The BYO-video frontend is live. How much guidance do you want for the first run?"
Options:
  "Guided companion (Recommended)" — "Claude asks one question at a time and can run the CLI guide alongside the browser UI"
  "Monitor only"                   — "Keep the expert UI, but watch for runtime errors and stalled runs"
  "Skip"                           — "Use the URL directly; you can start the guide later from the terminal"
```

If the user chooses the guided companion, spawn a post-deployment subagent:

```
Agent(
  name: "byo-video-guide",
  prompt: """
You are the BYO-video post-deployment guide for a casual Claude Code user.
Use AskUserQuestion for one decision at a time. Never ask open-ended questions
when a small menu will do.

Start by asking which surface they want to use. Use descriptive labels rather
than raw implementation names:
1. Guided dataset batch UI — load HF datasets, select videos, run concurrent inference, and export/write back results
2. CLI companion — walk through dataset loading, guarded runs, prompt shaping, and exports from the terminal
3. Single-video upload UI — upload one MP4/image, tune prompt settings, and inspect the response
4. Dataset browser + result viewer — inspect loaded samples and written predictions in FiftyOne

Then ask what they want to accomplish:
1. Worker-safety smoke test
2. Load a Hugging Face dataset
3. Import prompts/datasets from a paper or Hugging Face paper page
4. Shape structured JSON output for an inference run
5. Export a report, raw file, spreadsheet, or PowerPoint

For Batch Inference work, prefer:
  python3 /tmp/byo_video_runtime_guide.py --url <frontend_url> wizard
or specific commands:
  python3 /tmp/byo_video_runtime_guide.py --url <frontend_url> status
  python3 /tmp/byo_video_runtime_guide.py --url <frontend_url> load pjramg/Safe_Unsafe_Test --max-videos 2
  python3 /tmp/byo_video_runtime_guide.py --url <frontend_url> paper https://huggingface.co/papers/2603.29281 --max-videos 2
  python3 /tmp/byo_video_runtime_guide.py --url <frontend_url> run --first 2 --concurrency 1
  python3 /tmp/byo_video_runtime_guide.py --url <frontend_url> export --format pptx

Keep all safety guards on. Do not set allow_over_context. If the context guard
blocks a run, explain the budget issue and offer: Fit to context in the HTML UI,
lower max frames/FPS/max pixels, run fewer videos, or explicitly stress test
only after the user confirms they accept possible 400 errors.

For Gradio users, guide browser steps instead of inventing APIs: upload one MP4,
choose the prompt preset, use build.nvidia.com defaults for NIM-local when
appropriate, then run. If Gradio shows a generic error, inspect
/tmp/gradio_demo.log and optionally start the runtime monitor.

For structured output, help the user choose a schema first (classification,
safety inspection, dataset evaluation, or custom JSON fields). When the dataset
has expected answers, encourage an evaluation run and show match/miss summaries.

During a run, surface progress, elapsed time, ETA, error count, and the most
recent runtime error. End by helping the user export the artifact they need.
"""
)
```

The companion CLI is also safe to run manually from the instance:

```bash
python3 /tmp/byo_video_runtime_guide.py --url "$(cat /tmp/byo_video_batch_inference_url.txt 2>/dev/null || cat /tmp/gradio_url.txt)" wizard
python3 /tmp/byo_video_runtime_guide.py --url <frontend_url> status
python3 /tmp/byo_video_runtime_guide.py --url <frontend_url> export --format html --sections overview,run_metrics,evaluation,recommendations
```

---

### POST-LIVE — Runtime Monitor (optional, opt-in)

Even with the guided companion, keep the monitor available because Gradio and some
backend paths can swallow server-side errors as a generic browser toast. Errors
at preprocess, prefill, or generate stages are fully visible in `/tmp/gradio_demo.log`
or `/tmp/byo_video_batch_inference.log` on the remote, but may be invisible in the browser.

The **runtime monitor** is a lightweight polling daemon (`~/.claude/scripts/byo_video_runtime_monitor.py`)
that watches the remote Gradio process, GPU stats, and log tail; pattern-matches errors
against a rule catalog (cuda_oom, pyav_decode, preprocess_empty, shape_mismatch,
vllm_disconnect, process_killed, broken_pipe, torch_compile_fail, generic_traceback,
process_disappeared); and saves per-inference metrics for later review.

If the user chooses "Monitor only" above, or asks for extra watching while the
guided companion runs, fire `AskUserQuestion` to offer the monitor:

```
AskUserQuestion: "Spawn the runtime monitor to watch for errors and capture session metrics?"
Options:
  "Yes (Recommended)"  — "Polls every 30s; surfaces errors with suggested fixes; saves all inference metrics for later questions"
  "Skip"               — "No background watcher. Spawn later with: python3 ~/.claude/scripts/byo_video_runtime_monitor.py poll --remote <host>"
```

If yes, launch the daemon:

```bash
nohup python3 ~/.claude/scripts/byo_video_runtime_monitor.py poll \
  --remote <user@host> --rate <rate_per_hour> \
  --session-id <session_name> --model-id <model_id> \
  > /tmp/byo_video_runtime_monitor.out 2>&1 &
```

Then schedule a 1-min cron progress loop that reads the alert stream:

```
CronCreate({
  cron: "*/1 * * * *",
  prompt: "Read /tmp/byo_video_runtime_state.json and /tmp/byo_video_runtime_alerts.jsonl. Compare alert line count to /tmp/byo_video_runtime_alerts_seen.txt (default 0). For each new alert: print one-line summary 'severity rule: summary — fix'. Update the seen counter. Then print one-line state: 'gradio=alive|dead | GPU util=N% mem=Xused/Yfree | alerts=N metrics=N'. If gradio_alive=false, fire AskUserQuestion (load via ToolSearch) with options 'Restart Gradio' / 'Investigate log' / 'Abort'. Cancel this cron when user dismisses the monitor.",
  recurring: true
})
```

**Output files (all local /tmp):**
- `byo_video_runtime_state.json` — current snapshot (overwritten each poll)
- `byo_video_runtime_alerts.jsonl` — append-only alert stream
- `byo_video_session_metrics.jsonl` — append-only inference history (prompt, response, ttft_s, gen_s, tokens, model_id)

**Querying session data later:**
```bash
python3 ~/.claude/scripts/byo_video_runtime_monitor.py status
python3 ~/.claude/scripts/byo_video_runtime_monitor.py alerts -v
python3 ~/.claude/scripts/byo_video_runtime_monitor.py metrics
```

**Stopping the monitor:**
```bash
python3 ~/.claude/scripts/byo_video_runtime_monitor.py stop
```
Also cancel the cron loop (`CronDelete <id>`) when stopping.

The monitor is read-only on the remote. Polling cost: one SSH bundle every 30s, ~1KB.

---

### SSH DEPLOYMENTS (non-Brev)

PHASE 0–4 run in the main session with SSH commands replacing `brev exec`. After scripts are
deployed, the same HANDOFF applies — spawn the observer with `DEPLOY_TARGET=ssh:<user@host>`.
The observer runs PHASE 5–6 via SSH instead of `brev exec`.

Deploy scripts:
```bash
SCRIPT_DIR="${COSMOS_AGENT_SCRIPTS_DIR:-$PWD/.agents/skills/byo-video/scripts}"
for script in byo_video_setup gradio_cr2_byo gradio_cosmos_predict gradio_cosmos_reason_build byo_video_batch_inference byo_video_runtime_guide; do
  B64=$(base64 -i "$SCRIPT_DIR/${script}.py" | tr -d '\n')
  ssh -i ~/.ssh/id_ed25519 <user@host> \
    "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done

tar -C apps -czf - _shared nvidia-build-reason-vite nvidia-build-predict-vite nvidia-build-reason-next nvidia-build-predict-next | \
  ssh -i ~/.ssh/id_ed25519 <user@host> \
    "rm -rf /tmp/_shared /tmp/nvidia-build-reason-vite /tmp/nvidia-build-predict-vite /tmp/nvidia-build-reason-next /tmp/nvidia-build-predict-next && tar -C /tmp -xzf -"
```

Launch setup:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "nohup bash -c 'export INFERENCE_BACKEND=<backend> MODEL_ID=<model_id> MODEL_NAME=<model_id> MODEL_SIZE=<model_size> BYO_VIDEO_FRONTEND=<nvidia_build|batch_inference|gradio|fiftyone> BYO_VIDEO_LAUNCH_BATCH_INFERENCE=1 BATCH_INFERENCE_DATASET=pjramg/Safe_Unsafe_Test && python3 /tmp/byo_video_setup.py > /tmp/byo_video_setup.log 2>&1' &"
```

Cosmos3 Nano Reasoner Vite Build skin on the verified Horde host:

```bash
ssh -i ~/.ssh/id_ed25519 horde@10.57.232.110 \
  "nohup bash -c 'export HF_TOKEN=<hf_token_with_access> INFERENCE_BACKEND=vllm MODEL_ID=nvidia/Cosmos3-Nano-Reasoner MODEL_NAME=nvidia/Cosmos3-Nano-Reasoner MODEL_SIZE=C3-8B BYO_VIDEO_FRONTEND=nvidia_build REASON_VITE_PORT=5173 BATCH_INFERENCE_DATASET=pjramg/Safe_Unsafe_Test && python3 /tmp/byo_video_setup.py > /tmp/byo_video_setup.log 2>&1' &"
```

Tail logs:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "tail -60 /tmp/byo_video_setup.log 2>/dev/null"
```

URL capture:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "cat /tmp/gradio_live.flag 2>/dev/null; cat /tmp/byo_video_batch_inference_live.flag 2>/dev/null"
ssh -i ~/.ssh/id_ed25519 <user@host> "printf 'reason_vite='; cat /tmp/nvidia_build_reason_vite_url.txt 2>/dev/null; printf 'predict_vite='; cat /tmp/nvidia_build_predict_vite_url.txt 2>/dev/null; printf 'reason_next='; cat /tmp/nvidia_build_reason_next_url.txt 2>/dev/null; printf 'predict_next='; cat /tmp/nvidia_build_predict_next_url.txt 2>/dev/null; printf 'gradio='; cat /tmp/gradio_url.txt 2>/dev/null; printf 'batch_inference='; cat /tmp/byo_video_batch_inference_url.txt 2>/dev/null || true"
```

---

---

## HALT-AND-ASK PROTOCOL

During PHASE 0–1 (main session): all user decisions are captured in the picker. If an edge case
requires a follow-up question, fire `AskUserQuestion` before spawning the observer.

During PHASE 2–6 (observer): the observer never calls `AskUserQuestion`. On unrecoverable
failure it writes `/tmp/byo_video_observer_result.json` and exits. The main session reads the
result on task completion notification and fires `AskUserQuestion` for recovery.

**Load AskUserQuestion** before firing: `ToolSearch({ query: "select:AskUserQuestion" })`.

### Exception triggers and question templates

**Observer failure (any phase) — triggered by main session progress loop detecting failure JSON:**

Read `/tmp/byo_video_progress.json` for last-known state before firing AskUserQuestion.

```
AskUserQuestion: "Observer failed at Phase <N>: <error>.
Last checklist state: [✓] provision [✓] shell [→] scripts [ ] deps [ ] weights [ ] gradio
What next?"
Options:
  "Retry — spawn a new observer on the same instance"
  "Provision a new instance (current will be deleted)"
  "Abort — delete instance and exit"
```

(Replace the checklist state line with actual values read from `/tmp/byo_video_progress.json` — use `[✓]` for `"done"`, `[→]` for `"active"`, `[ ]` for `"pending"`.)

On "Retry": spawn a new observer with the same state, `DEPLOY_TARGET=brev:<existing-name>`.
On "New instance": delete the failed instance, spawn a new observer with `DEPLOY_TARGET=brev:new`.
On "Abort": `brev delete <name>`, exit skill.

**Insufficient VRAM — triggered by main session detecting `"insufficient_vram"` in failure JSON:**

Read `vram_free_mb` and `vram_needed_mb` from the failure JSON. Build options from MODEL_GPU_REQUIREMENTS — include only models whose `min_vram_mb` ≤ `vram_free_mb`.

```
AskUserQuestion: "The requested model (<MODEL_SIZE>, needs ~<vram_needed_mb/1000>GB VRAM) won't fit on <instance>
(<vram_free_mb/1000>GB free). Switch to a model that fits, or abort?"
Options (show only what fits — examples for 40GB free):
  "Cosmos Reason2 2B (needs ~40GB)"    → MODEL_ID=nvidia/Cosmos-Reason2-2B, MODEL_SIZE=2B
  "Cosmos3-Nano-Reasoner (needs ~40GB)" → MODEL_ID=nvidia/Cosmos3-Nano-Reasoner, MODEL_SIZE=C3-8B
  "Abort — exit without deleting instance"
```

On model switch: update MODEL_ID, MODEL_SIZE, MODEL_NAME to the new selection, respawn observer.
On Abort: exit skill. Do not delete the instance (user may want it for other work).

**Rule:** Never silently substitute a different model. If the requested model does not fit and there
is no confirmed user-approved alternative, always ask. A silent downgrade is a broken demo.

**Rule:** For any exception not listed here, apply the closest matching template. The goal is
one clear question with 2–3 concrete options. Never ask open-ended questions mid-deployment.

---

### MODEL_GPU_REQUIREMENTS — Read before provisioning any instance

| MODEL_SIZE | Min VRAM | Brev type | Rate | Avoid |
|---|---|---|---|---|
| `2B` | 40 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `8B` | 80 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | A40 (48GB too tight) |
| `32B` | 141 GB | `gpu-h200-sxm.1gpu-16vcpu-200gb` | $4.20/hr | H100 single-GPU |
| `C3-8B` | 40 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `C3-32B` | 141 GB | `gpu-h200-sxm.1gpu-16vcpu-200gb` | $4.20/hr | H100 single-GPU |
| `NEM-12B` | 40 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `QW3-2B` | 8 GB | Any GPU ≥8GB | varies | — |
| `QW3-8B` | 20 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `QW3-32B` | 64 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `C3-super` | TBD (32B) | `gpu-h200-sxm.1gpu-16vcpu-200gb` | $4.20/hr | — |

> **C3-super** is a native MODEL_SIZE key in both `byo_video_setup.py` and `gradio_cr2_byo.py`. Requires H200 SXM (141GB VRAM). VLLM_TIMEOUT is 420s for this size (32B torch.compile takes ~5 min).

**32B and C3-32B — H200 only.** H100 80GB is insufficient. If the user has an existing H100 and selects 32B: warn them and offer to provision an H200.

**H200 fallback order for 32B/C3-32B:**

| Priority | Type | Rate |
|---|---|---|
| 0 | Existing stopped H200 (from brev ls) | existing rate |
| 1 | `gpu-h200-sxm.1gpu-16vcpu-200gb` | $4.20/hr |
| 2 | `digitalocean_H200_sxm5` | $4.13/hr |

### PROVIDER FALLBACK — Auto-rotation on provisioning failure

Rotate silently (no AskUserQuestion) on UNHEALTHY after 3 min, `brev create` error, or reset failure. Emit one line per rotation: `✗ <type> failed — trying <next-type>`.

**Standard (all models except 32B/C3-32B):**

| Priority | Type | GPU | Rate |
|---|---|---|---|
| 1 | `gpu-h100-sxm.1gpu-16vcpu-200gb` | H100 SXM 80GB | $3.54/hr |
| 2 | `hyperstack_A100_80G` | A100 80GB | $1.62/hr |
| 3 | `hyperstack_H100` | H100 80GB | $2.28/hr |
| 4 | `scaleway_A40` | A40 48GB | $1.10/hr (2B LOW_VRAM only) |

If rotating to a tier >$1.50/hr more expensive than the current one: display a one-line cost warning, wait 30s (allows interruption), then proceed.

Only AskUserQuestion when all providers exhausted — present the failure summary and options.

---

---

## Supported models

| Model | Size | Min VRAM | MODEL_SIZE | Use case |
|---|---|---|---|---|
| Cosmos Reason2 BF16 | 2B VLM | 40 GB | `2B` | Video understanding: robotics, AV, Metropolis |
| Cosmos Reason2 FP8 | 2B VLM | 24 GB | `2B` | Same, quantized |
| Cosmos Reason2 BF16 | 8B VLM | 80 GB | `8B` | Higher quality video understanding |
| Cosmos Reason2 BF16 | 32B VLM | 141 GB | `32B` | Public; H200 SXM minimum (H100 80GB insufficient) |
| Cosmos3-Nano-Reasoner | 8B VLM | 40 GB | `C3-8B` | HF_TOKEN required when HF API returns 401; verify access before launch |
| Cosmos3-Reasoner 2B/32B | 2B/32B | 40/80+ GB | `C3-2B`, `C3-32B` | Gated HF_TOKEN; nvidia org required |
| Nemotron-Nano-12B-v2-VL BF16 | 12B VLM | 40 GB | `NEM-12B` | vLLM-only; gated; opencv backend |
| Nemotron-Nano-12B-v2-VL FP8 | 12B VLM | 24 GB | `NEM-12B` | FP8 quantized |
| Qwen3-VL-2B Instruct/FP8/Thinking | 2B VLM | 8 GB | `QW3-2B` | Public (no HF_TOKEN); vLLM-only |
| Qwen3-VL-8B Instruct/FP8/Thinking | 8B VLM | 20 GB | `QW3-8B` | Public; higher quality |
| Qwen3-VL-32B Instruct/FP8/Thinking | 32B VLM | 64+ GB | `QW3-32B` | Public; best quality |
| Cosmos Transfer2.5 | Gen | 80 GB+ | — | Video-to-video generation, sim2real |
| Cosmos Predict2 | Gen | 80 GB+ | — | World model generation |

Transfer2.5 and Predict2 are datacenter-only (H100/A100 80GB+).

### Nemotron-Nano-12B-v2-VL notes

- **Gated model** — HF_TOKEN required with nvidia org access
- **vLLM 0.14.0+** — vLLM 0.11.0 is ABI-incompatible with torch 2.9.0+cu128. `byo_video_setup.py` auto-pins to 0.14.0 on CUDA 12.8 (driver < 575). Do not pin lower.
- **opencv backend** — `VLLM_VIDEO_LOADER_BACKEND=opencv` is injected automatically. PyAV is not supported.
- **Base64 video protocol** — frontends send `data:video/...;base64,...` `video_url` content. Do not use `file://`; it fails on hosted NIMs and on default vLLM/NIM containers without local-media flags.
- **FLASHINFER bypass** — `FLASHINFER_DISABLE_VERSION_CHECK=1` is injected automatically (flashinfer package/cubin version mismatch in cosmos-reason2 venv).
- **NVFP4-QAD variant** — requires a special vLLM build; not available in standard setup. Use BF16 or FP8.
- **Launch**: `MODEL_SIZE=NEM-12B INFERENCE_BACKEND=vllm python3 /tmp/byo_video_setup.py`

### Qwen3-VL notes

- **Public model** — no HF_TOKEN required
- **vLLM only** — uses the same base64 `video_url` protocol as Nemotron; HF inference backend not supported
- **No `file://` dependency** — the base64 payload path works without `--allowed-local-media-path /tmp`
- **Variant picker** — Gradio checkpoint dropdown shows Instruct, FP8, and Thinking for each size
- **Thinking variant** — extended reasoning mode; max_tokens should be ≥2048 for best results
- **VRAM**: 2B~8GB, 8B~20GB, 32B~64GB (FP8 halves these)
- **Launch**: `MODEL_SIZE=QW3-8B INFERENCE_BACKEND=vllm python3 /tmp/byo_video_setup.py`

---

## VLM NIM Catalog (for `INFERENCE_BACKEND=nim_local`)

Canonical catalog: `~/.claude/scripts/nim_catalog.py` → `KNOWN_VLM_NIMS`. The agent walks BOTH `https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html` AND every versioned release-notes page in `KNOWN_RELEASE_VERSIONS` to keep older containers (e.g. Cosmos Reason1 7B) discoverable when a newer release drops them from the introduction table. When a name appears upstream that is NOT in `KNOWN_VLM_NIMS`, surface it to the user as a one-line PR-this hint.

Two catalogs to keep separate:

| Catalog | URL | What it lists | Auth |
|---|---|---|---|
| **Self-host Docker images** (used by `nim_local`) | `nvcr.io/nim/<vendor>/<short-id>:latest` (entitlements at `docs.nvidia.com/nim/...`) | Containers you can `docker pull` and run on your own GPU | NGC_API_KEY (`nvapi-…`) |
| **Hosted serverless API** (separate path) | `integrate.api.nvidia.com/v1/chat/completions` (catalog at `build.nvidia.com`) | Subset of NIMs NVIDIA serves for you | NGC_API_KEY |

CR2-2B is in the Docker catalog only. CR2-8B is in both. The "[NIM] Skipped — not in public NVCF catalog" message in Gradio refers to the hosted-API catalog and does NOT mean the local container is unavailable.

### Video-capable NIMs (relevant to `/byo-video`)

| Family | Short-id | Min VRAM | Notes |
|---|---|---|---|
| Cosmos Reason2 | `cosmos-reason2-2b` | 20 GB | FP8; reasoning; temp ≥ 0.3 to avoid `<think>+EOS` bug at greedy decode. Container only — not on hosted API. |
| Cosmos Reason2 | `cosmos-reason2-8b` | 40 GB | FP8; reasoning; Efficient Video Sampling (EVS); same temp constraint as 2B. Container + hosted API. |
| Cosmos Reason2 | `cosmos-reason2-32b` | 80 GB | Preview/private: default NGC key saw `DENIED` on 2026-05-07; needs allowlist before self-serve smoke. |
| Cosmos Reason1 | `cosmos-reason1-7b` | 24 GB | Older release; preserved via release-notes walk. Known native-`video_url` exception; frontends retry with frames. |
| Nemotron | `nemotron-3-nano-omni-30b-a3b-reasoning` | 80 GB | H200 recommended; first boot can take ~25 min, so use 2400s wait. |
| Nemotron | `nemotron-nano-12b-v2-vl` | 40 GB | Prefer native `video_url`; image-frame fallback hits 5-image prompt limit. Use container-internal cache. |
| Mistral | `ministral-14b-instruct-2512` | 40 GB | Tool calling; 100k context on L40S. |
| Qwen | `qwen3.5-35b-a3b` | 40 GB | MoE; high-concurrency video constraints. |
| Qwen | `qwen3.5-122b-a10b` | 140 GB | MoE; KV cache saturation risk on long video. Multi-GPU. |
| Qwen | `qwen3.5-397b-a17b` | 400 GB | Video not enabled by default — config flag required. Multi-node. |
| Gemma | `gemma-4-31b-it` | 80 GB | Requires `NIM_MAX_MODEL_LEN=131072`; default 262144 over-allocates KV cache and exits. H200 recommended. |

### Image-only NIMs (filtered OUT of `/byo-video` by `list_video_nims()`)

`mistral-medium-3.5-128b` · `mistral-small-4-119b-2603` · `mistral-small-3.2-24b-instruct-2506` · `mistral-large-3-675b-instruct-2512` · `kimi-k2.5` · `kimi-k2.6` · `qwen3.6-27b` · `qwen3.6-35b-a3b` · `llama-3.1-nemotron-nano-vl-8b-v1` · `llama-3.2-11b-vision-instruct` · `llama-3.2-90b-vision-instruct` · `llama-4-maverick-17b-128e-instruct` · `llama-4-scout-17b-16e-instruct` · `nemotron-parse-v1.2` · `nemotron-3-content-safety`

Adding a new NIM:

1. Find its row in either the latest [introduction](https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html) or a [release notes](https://docs.nvidia.com/nim/vision-language-models/1.7.0/release-notes.html) page.
2. Read the model card linked from that row — note image short-id, served model ID, VRAM minimum, and whether the card mentions multi-frame video input.
3. Append a new `NimImage(...)` line to `KNOWN_VLM_NIMS` in `~/.claude/scripts/nim_catalog.py` with `supports_video=` set correctly. Append the new release version to `KNOWN_RELEASE_VERSIONS` if it's newer than the latest entry.
4. Mirror the change to `<repo>/.claude/scripts/nim_catalog.py` and commit.

Querying from the agent or runbook:

```bash
python3 ~/.claude/scripts/nim_catalog.py upstream      # raw upstream model-name list
python3 ~/.claude/scripts/nim_catalog.py list --no-probe  # full known catalog as JSON
python3 -c "from nim_catalog import list_video_nims; \
            [print(n.short_id, n.min_vram_mb) for n in list_video_nims()]"
```

---

## VRAM auto-selection

The setup script (`byo_video_setup.py`) handles all of this automatically. Rules as of 2026-04-21:

**Model selection** — always CR2-2B for the live demo (8B fails the <60s inference target on non-H100 GPUs). Force 8B via `MODEL_NAME=nvidia/Cosmos-Reason2-8B` env var if quality > speed.
- ≥ 40000 MiB free → `nvidia/Cosmos-Reason2-2B`
- < 40000 MiB → CR2-2B with LOW_VRAM mode (fps=1, reduced resolution)

**FPS/pixel tier** — detected by GPU name (not free VRAM), because workstation GPUs like RTX PRO 6000 have large VRAM but slower prefill compute:
| Tier | Condition | fps | max_pixels | Expected inference |
|---|---|---|---|---|
| H100/A100 | GPU name contains H100, A100, H200, GB200 | 2 | 1,048,576 | ~44s (validated H100) |
| High-VRAM (non-H100) | ≥40GB free, non-H100 (RTX PRO, A40, A30, etc.) | 1 | 524,288 | ~54s (validated RTX PRO 6000 97GB) |
| Low-VRAM | <40GB free | 1 | 131,072 | ~80-120s |
| Ultra-Low-VRAM | <8GB free (RTX 5070, GTX 3090, etc.) | 1 | 65,536 | ~120-180s; may OOM on videos >5s at 720p — pre-resize to 360p |

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
# Create new instance — use model-aware GPU type (see MODEL_GPU_REQUIREMENTS)
# For 32B: brev create <name> --gpu-name H200 --type gpu-h200-sxm.1gpu-16vcpu-200gb
# For ≤8B: brev create <name> --gpu-name H100 --type gpu-h100-sxm.1gpu-16vcpu-200gb

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

Then run setup (streams live progress to this terminal).
Note: pass env vars as separate exports in the remote quoted command — HF_TOKEN is auto-read by the script from `~/.cache/huggingface/token` on the instance (do not pass it manually):
```bash
brev exec <name> "export INFERENCE_BACKEND=vllm MODEL_ID=nvidia/Cosmos-Reason2-2B BREV_RATE_PER_HOUR=3.70 PATH=~/.local/bin:~/.cargo/bin:$PATH && python3 /tmp/byo_video_setup.py"
```

The script will:
1. Print a 9-step setup dashboard and begin executing
2. Detect GPU + VRAM tier (H100 / High-VRAM / Low-VRAM / Ultra-Low-VRAM)
3. Validate HF token (whoami check — fails fast if expired)
4. Install any missing deps (uv, repos, PyAV, Gradio)
5. Download model weights with retry logic
6. Launch Gradio and print a **clickable hyperlink** to the public `gradio.live` URL
7. Report elapsed time and credits spent throughout

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

### Step 5 — Kill instance when done

Delete the instance from the Brev dashboard or run `brev delete <name>`. Do NOT auto-terminate.

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

For public HF dataset batch smoke, use the Batch Inference UI instead:

```bash
cd ~/cosmos-reason2
VLLM_BASE_URL=http://localhost:8000/v1 \
uv run python /tmp/byo_video_batch_inference.py smoke \
  --dataset pjramg/Safe_Unsafe_Test \
  --max-videos 2 \
  --concurrency 2
```

Results at `/tmp/byo_video_batch_inference_results.json`.

---

## Horde (SSH-based — asotelo org)

Horde instances are created via REST API, accessed via SSH.

**Critical:** SSH username is `horde` — confirmed empirically 2026-04-17. Not `ubuntu`, `nvidia`, `root`, or `asotelo`.

Agent steps:
1. Create instance via Horde API v4 (`POST /api/v4/instances`) or use the explicit user-provided Horde host. The setup script is scratch-safe: it installs `uv`, clones `~/cosmos-reason2`, creates the venv, downloads weights, starts vLLM/NIM when requested, always launches Gradio, and launches the selected companion frontend when applicable.
2. Poll `GET /api/v4/instances/<id>` until `status: running`
3. Deploy scripts to instance (canonical source is the BYO-video skill scripts directory):
   ```bash
   SCRIPT_DIR="${COSMOS_AGENT_SCRIPTS_DIR:-$PWD/.agents/skills/byo-video/scripts}"
   for script in byo_video_setup gradio_cr2_byo gradio_cosmos_predict gradio_cosmos_reason_build byo_video_batch_inference byo_video_runtime_guide; do
     B64=$(base64 -i "$SCRIPT_DIR/${script}.py" | tr -d '\n')
     ssh -i ~/.ssh/id_ed25519 horde@<ip> \
       "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
   done

   tar -C apps -czf - _shared nvidia-build-reason-vite nvidia-build-predict-vite nvidia-build-reason-next nvidia-build-predict-next | \
     ssh -i ~/.ssh/id_ed25519 horde@<ip> \
       "rm -rf /tmp/_shared /tmp/nvidia-build-reason-vite /tmp/nvidia-build-predict-vite /tmp/nvidia-build-reason-next /tmp/nvidia-build-predict-next && tar -C /tmp -xzf -"
   ```
4. Run setup (streams live output here):
   ```bash
   ssh -i ~/.ssh/id_ed25519 horde@<ip> \
     "export HF_TOKEN=<hf_token_with_access> INFERENCE_BACKEND=vllm MODEL_ID=nvidia/Cosmos3-Nano-Reasoner MODEL_NAME=nvidia/Cosmos3-Nano-Reasoner MODEL_SIZE=C3-8B BYO_VIDEO_FRONTEND=nvidia_build BYO_VIDEO_LAUNCH_BATCH_INFERENCE=1 REASON_VITE_PORT=5173 BATCH_INFERENCE_DATASET=pjramg/Safe_Unsafe_Test && python3 /tmp/byo_video_setup.py"
   ```
5. URLs print at the end. Reason Vite writes `/tmp/nvidia_build_reason_vite_url.txt`; Predict Vite writes `/tmp/nvidia_build_predict_vite_url.txt`; Gradio is always written to `/tmp/gradio_url.txt`; Batch Inference / FiftyOne companion UI is written to `/tmp/byo_video_batch_inference_url.txt` when selected or when `BYO_VIDEO_LAUNCH_BATCH_INFERENCE=1` is set. Optional Next.js captures can write `/tmp/nvidia_build_reason_next_url.txt` and `/tmp/nvidia_build_predict_next_url.txt` when started manually. Read all back with `cat /tmp/nvidia_build_reason_vite_url.txt 2>/dev/null; cat /tmp/nvidia_build_predict_vite_url.txt 2>/dev/null; cat /tmp/nvidia_build_reason_next_url.txt 2>/dev/null; cat /tmp/nvidia_build_predict_next_url.txt 2>/dev/null; cat /tmp/gradio_url.txt; cat /tmp/byo_video_batch_inference_url.txt 2>/dev/null` via ssh.
6. Smoke test the worker-safety dataset after the frontend is live:
   ```bash
   ssh -i ~/.ssh/id_ed25519 horde@<ip> \
     "cd ~/cosmos-reason2 && VLLM_BASE_URL=http://localhost:8000/v1 uv run python /tmp/byo_video_batch_inference.py smoke --dataset pjramg/Safe_Unsafe_Test --max-videos 2 --concurrency 2"
   ```

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

Write result to `/tmp/byo_video_reason2_results.json`. Delete the endpoint when done.

---

## Gradio app script

Canonical source: **`.agents/skills/byo-video/scripts/gradio_cr2_byo.py`**; mirror tracked legacy changes into `.claude/scripts/gradio_cr2_byo.py`.

Features (as of 2026-04-21):
- **Checkpoint selector** — Advanced Settings accordion has a preset dropdown for Cosmos, Cosmos3, Nemotron, Qwen, and VLM NIMs plus a custom model ID field. Any HF model ID or local path is accepted.
- **On-demand load/unload** — switching checkpoints does `del model → gc.collect() → cuda.empty_cache()` before loading the next variant. VRAM is confirmed free before loading.
- **Run All Variants button** — sequential benchmark: FP8 → NVFP4 → NIM, each unloaded before the next. Results saved to `/tmp/byo_video_benchmark.json`.
- **Right-side status panel** — replaces the grey loading box. Shows Step N/5 WIP tracker (resolve / load / preprocess / prefill / generate) with ✅/⟳/— per step, plus live token metrics: prefill count, generated count (running), TTFT, inference time.
- **NIM mode** — local Docker NIMs use the OpenAI-compatible `/v1/chat/completions` path. Hosted NVCF is still supported for catalog entries. Both send base64 `video_url` first, omit `max_tokens`, and only retry image frames after a 400/422.
- LOW_VRAM mode, PyAV backend, qwen_vl_utils pipeline, auto-cap, results JSON — all retained from 2026-04-20.

Deploy to instances via base64 as shown in the deploy section.

**Do not embed the source here.** The canonical file is the source of truth.

### Checkpoint selector usage

| Method | How |
|---|---|
| Preset (base, FP8, NVFP4, NIM) | Advanced Settings → Checkpoint dropdown |
| Custom HF ID | Advanced Settings → Custom Checkpoint ID field (overrides dropdown) |
| Custom local path | Same custom field — accepts `/path/to/model` |
| Env var (headless) | `export CR2_CHECKPOINT=nvidia/Cosmos-Reason2-2B-FP8` before setup |

For multi-variant benchmarking without the UI, click **Run All Variants** — runs FP8 → NVFP4 → NIM sequentially.

### NIM mode requirements

| Item | Value |
|---|---|
| NGC_API_KEY | `nvapi-...` prefix (set in env before setup, passed to Gradio) |
| Model identifier | `nvidia/cosmos-reason2-2b` (NVCF catalog) |
| Video input | Base64 `data:` `video_url`; frame fallback only after service 400/422 |
| Auth header | `Authorization: Bearer $NGC_API_KEY` |
| Output | Streamed via SSE, same JSON results format as HF path |

Set `NGC_API_KEY` before running `byo_video_setup.py` — it passes it through to the Gradio process env.

### Setup script — multi-variant mode

Set `MULTI_VARIANT=true` to also pre-download FP8 and NVFP4 model weights during setup:

```bash
export HF_TOKEN=hf_...
export NGC_API_KEY=nvapi-...
export MULTI_VARIANT=true
python3 /tmp/byo_video_setup.py
```

Without `MULTI_VARIANT=true`, only the base model downloads. FP8/NVFP4 will download from HF on first use in Gradio (with HF_TOKEN).

### Env vars (single-model mode)

| Var | Default | Notes |
|---|---|---|
| `MODEL_NAME` | `nvidia/Cosmos-Reason2-2B` | HF model ID to load at startup |
| `MODEL_DIR` | `~/cosmos-reason2/models/Cosmos-Reason2-2B` | Local path |
| `HF_TOKEN` | — | Required for gated model download |
| `NGC_API_KEY` | — | Required for NIM mode (`nvapi-` prefix) |
| `MULTI_VARIANT` | false | `true` = also download FP8 + NVFP4 during setup |
| `GRADIO_PORT` | 7860 | Gradio server port |
| `GRADIO_SHARE` | true | Set `false` to disable public link |
| `GRADIO_FPS` | tier-based | Passed by setup script |
| `GRADIO_MAX_PIXELS` | tier-based | Passed by setup script |
| `GRADIO_PREFILL_TPS` | tier-based | Passed by setup script |
| `LOW_VRAM` | auto | Set `true` to force low-VRAM mode |
| `OUT_FILE` | `/tmp/byo_video_reason2_results.json` | Per-run results JSON path |

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
| Brev H100 | H100 | CR2-2B-FP8 | fps=8, 1M px | TBD | TBD | ⏳ pending smoke gate |
| Brev H100 | H100 | CR2-8B-NVFP4 | fps=8, 1M px | TBD | TBD | ⏳ pending smoke gate |
| Brev H100 | H100 | NIM 2B (NVCF) | 8 frames | N/A | TBD | ⏳ pending smoke gate |

Notes:
- "Inference" = preprocess + prefill + decode (TTFT-dominated for video inference)
- Old A40 result was without VRAM tier tuning; with fps=1 it would likely be ~55-65s
- CR2-8B fails the <60s target on all non-H100 GPUs tested; use CR2-2B for demos
- FP8/NVFP4/NIM timing benchmarks TBD — update after Bronson smoke gate (byo-video Checkpoint & NIM Sprint, 2026-04-21)

---

## Dynamic Reconfiguration — Switching Model Class at Runtime

**When invoked with `<instance> <model-class>` arguments**, the skill changes the active model without touching the static launch scripts.

Usage:
```
/byo-video byo-video-vllm 8B
/byo-video byo-video-vllm 2B
/byo-video 10.0.1.103 8B
```

### Known instances (Hyperstack, private subnet 10.0.1.0/24)

| Brev name | IP | Backend | Default size | NGC env |
|---|---|---|---|---|
| `byo-video-vllm` | 10.0.1.103 | vLLM 0.11.0 | 8B NVFP4 | `/home/shadeform/.ngc_env` |
| `byo-video-nim` | 10.0.1.248 | HF Transformers | 2B BF16 | — |

### Models on disk (byo-video-vllm)

| Size | Local path | HF ID |
|---|---|---|
| 2B FP8 | `~/cosmos-reason2/models/Cosmos-Reason2-2B-FP8` | `nvidia/Cosmos-Reason2-2B-FP8` |
| 2B BF16 | `~/cosmos-reason2/models/Cosmos-Reason2-2B` | `nvidia/Cosmos-Reason2-2B` |
| 8B NVFP4 | `~/cosmos-reason2/models/Cosmos-Reason2-8B-NVFP4` | `nvidia/Cosmos-Reason2-8B-NVFP4` |
| 8B FP8 | `~/cosmos-reason2/models/Cosmos-Reason2-8B-FP8` | `nvidia/Cosmos-Reason2-8B-FP8` |
| 8B BF16 | `~/cosmos-reason2/models/Cosmos-Reason2-8B` | `nvidia/Cosmos-Reason2-8B` |
| 32B BF16 | public (May 2026) — no HF_TOKEN, ~66 GB | `nvidia/Cosmos-Reason2-32B` |
| 32B AV | public (May 2026) — no HF_TOKEN, ~66 GB | `nvidia/Cosmos-Reason2-32B-AV` |

### Reconfiguration procedure (agent runs all steps)

**Step 1 — Kill Gradio:**
```bash
brev exec <instance> "pkill -f gradio_cr2_byo"
```
Then wait 3s, verify port is free:
```bash
brev exec <instance> "ss -tlnp | grep 7860"
```

**Step 2 — Kill vLLM server:**
```bash
brev exec <instance> "pkill -f 'vllm serve'"
```
Wait 5s for GPU memory to release:
```bash
brev exec <instance> "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
```

**Step 3 — Deploy updated gradio script (if changed):**
```bash
B64=$(base64 -i ~/.claude/scripts/gradio_cr2_byo.py | tr -d '\n')
brev exec <instance> "python3 -c \"import base64; open('/tmp/gradio_cr2_byo.py','wb').write(base64.b64decode('${B64}'))\""
```

**Step 4 — Start vLLM with target model:**

For 2B FP8 (vLLM, real FP8 kernels):
```bash
brev exec <instance> "nohup bash -c 'source /home/shadeform/.ngc_env && cd /home/shadeform/cosmos-reason2 && .venv/bin/vllm serve models/Cosmos-Reason2-2B-FP8 --served-model-name nvidia/Cosmos-Reason2-2B-FP8 --port 8000 --dtype auto --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.90 > /tmp/vllm_server.log 2>&1 &'"
```

For 8B NVFP4 (current default):
```bash
brev exec <instance> "nohup bash -c 'source /home/shadeform/.ngc_env && cd /home/shadeform/cosmos-reason2 && .venv/bin/vllm serve models/Cosmos-Reason2-8B-NVFP4 --served-model-name nvidia/Cosmos-Reason2-8B-NVFP4 --port 8000 --dtype auto --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.85 > /tmp/vllm_server.log 2>&1 &'"
```

For 8B FP8:
```bash
brev exec <instance> "nohup bash -c 'source /home/shadeform/.ngc_env && cd /home/shadeform/cosmos-reason2 && .venv/bin/vllm serve models/Cosmos-Reason2-8B-FP8 --served-model-name nvidia/Cosmos-Reason2-8B-FP8 --port 8000 --dtype auto --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.85 > /tmp/vllm_server.log 2>&1 &'"
```

For 32B BF16 (needs smaller max-model-len to fit 80GB — download first):
```bash
brev exec <instance> "nohup bash -c 'source /home/shadeform/.ngc_env && cd /home/shadeform/cosmos-reason2 && .venv/bin/vllm serve models/Cosmos-Reason2-32B --served-model-name nvidia/Cosmos-Reason2-32B --port 8000 --dtype bfloat16 --trust-remote-code --max-model-len 4096 --gpu-memory-utilization 0.95 > /tmp/vllm_server.log 2>&1 &'"
```

**Step 5 — Wait for vLLM ready (poll /v1/models, up to 120s):**
```bash
brev exec <instance> "for i in \$(seq 1 24); do curl -sf http://localhost:8000/v1/models && break || sleep 5; done"
```

**Step 6 — Start Gradio with correct MODEL_SIZE:**

Do NOT edit `launch_gradio_vllm.sh`. Run inline:
```bash
brev exec <instance> "nohup bash -c 'source /home/shadeform/.ngc_env && cd /home/shadeform/cosmos-reason2 && INFERENCE_BACKEND=vllm VLLM_BASE_URL=http://localhost:8000/v1 MODEL_SIZE=<SIZE> .venv/bin/python /tmp/gradio_cr2_byo.py > /tmp/gradio_demo.log 2>&1 &'"
```

Replace `<SIZE>` with `2B`, `8B`, or `32B`.

**Step 7 — Get URL:**
```bash
brev exec <instance> "sleep 20 && cat /tmp/gradio_url.txt"
```

### Downloading missing models

If the target model isn't on disk yet, download before starting vLLM:
```bash
brev exec <instance> "cd /home/shadeform/cosmos-reason2 && .venv/bin/huggingface-cli download nvidia/Cosmos-Reason2-8B-FP8 --local-dir models/Cosmos-Reason2-8B-FP8"
```
HF_TOKEN required for gated models. `Cosmos-Reason2-8B-FP8` is public.

### 32B feasibility notes

- **For /byo-video: H200 is the minimum viable GPU.** H100 single-GPU (80GB) is insufficient for comfortable inference under the skill's time constraints. Use `gpu-h200-sxm.1gpu-16vcpu-200gb` (141GB VRAM) — empirically confirmed by Alex with c3r-32b.
- 32B BF16 weights are ~66GB, but vLLM requires additional VRAM for KV cache, activations, and overhead. On H100 80GB you can technically load with `--max-model-len 4096 --gpu-memory-utilization 0.95`, but this leaves almost no room for KV cache and will OOM on longer video sequences.
- CR2-32B is **public** as of May 2026 — no HF_TOKEN required. Download: `huggingface-cli download nvidia/Cosmos-Reason2-32B --local-dir models/Cosmos-Reason2-32B`.
- In vLLM mode with 8B loaded, `run_all_variants` will send 32B requests to the 8B server — the table `Notes` column will show `vLLM serves Cosmos-Reason2-8B-NVFP4` to flag the mismatch.
- For true 32B benchmarking: restart vLLM with 32B model using this skill, then run `Run All Variants` with `MODEL_SIZE=32B`.

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
| `brev login` fails with EOF | `brev login` requires a browser handoff — it cannot run via `! brev login` in Claude Code (non-TTY). Open a separate terminal tab, run `brev login` there, complete the browser prompt, then return. |
| vLLM Connection refused on first inference | `byo_video_setup.py` now auto-starts vLLM before Gradio (Step 9b). If running the Gradio script manually, start vLLM first: `nohup .venv/bin/vllm serve <model_dir> --port 8000 ... &` then poll `curl localhost:8000/v1/models`. |
| Nemotron: `no module named 'mamba_ssm'` or `selective_scan_cuda` | vLLM PyPI build doesn't include mamba-ssm. Use vLLM nightly Docker: `vllm/vllm-openai:nightly-8bff831f0aa239006f34b721e63e1340e3472067` or `nvcr.io/nvidia/vllm:25.12.post1-py3`. |
| Nemotron: `video_url not supported` or `unsupported content type` | vLLM version doesn't support `video_url` message type. Requires vLLM nightly; PyPI ≤0.11.0 unsupported. |
| Nemotron/NIM: 400 error on inference | Confirm the deployed frontend is current and sends base64 `video_url`. If the service still rejects it, inspect the response body; Batch Inference / Gradio retry frame fallback only for 400/422. |
