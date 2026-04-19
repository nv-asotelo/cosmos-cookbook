# /vlm-race — VLM Race: Multi-Model Video Comparison

> **Status: Scaffolded — sequential load/unload architecture in progress.**
> Current implementation loads all models simultaneously (shared VRAM bandwidth). True apples-to-apples comparison requires sequential load/unload (see Architecture section below).

Side-by-side inference comparison: upload one video, get outputs from multiple VLMs with TTFT, inference time, token counts, and a winner badge.

**Canonical scripts:**
- `gradio_compare_vlm.py` — Gradio app (3-column race UI)
- `compare_vlm_setup.py` — Bootstrap + launch

**Results:** JSON at `/tmp/vlm_race_results.json` after each run.

---

## Models

| Column | Model | HF ID | VRAM (BF16) |
|---|---|---|---|
| 🌌 Cosmos Reason 2 | CR2-2B or CR2-8B | `nvidia/Cosmos-Reason2-{2,8}B` | ~5 GB / ~17 GB |
| 🤖 Nemotron-Nano-12B | Fixed | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | ~26 GB |
| 🔷 Qwen3-VL | Paired with CR2 size | `Qwen/Qwen3-VL-{2,8}B-Instruct` | ~5 GB / ~17 GB |

All models BF16. Sequential inference on one GPU.

**Nemotron dependency:** Requires `mamba-ssm` CUDA extension. Pre-built wheels only available for torch ≤ 2.6. Instances running torch 2.9+cu128 get a graceful error card. Full 3/3 requires CUDA toolkit + torch 2.6 (e.g. `nvidia/cuda:12.4-devel` Docker image).

---

## Architecture — Current vs Target

### Current (shared bandwidth)
All models loaded into VRAM at startup. Each model runs inference in turn, sharing memory bandwidth with idle models. Reported times are not comparable to single-model baselines.

```
[load CR2] [load Nemotron] [load Qwen] → [infer CR2 | all 3 in VRAM] → [infer Qwen | all 3 in VRAM]
```

### Target (sequential load/unload — next sprint)
Each model gets the full GPU for its run. Results held in CPU memory between runs.

```
[load CR2] → [infer CR2] → [unload CR2] → [load Qwen] → [infer Qwen] → [unload Qwen] → compare
```

Implementation: `RACE_MODE=sequential_loadunload` env var. `init_slots()` loads no models at startup. Each model loads immediately before inference and is unloaded (`del model; torch.cuda.empty_cache()`) immediately after.

---

## VRAM auto-selection

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
```

| Free VRAM | CR2 size | Qwen3-VL size | Nemotron |
|---|---|---|---|
| ≥ 80 GB | 8B | 8B | 12B (if mamba-ssm compatible) |
| ≥ 40 GB | 2B | 2B | 12B (VRAM pre-check; OOM card if insufficient) |
| < 40 GB | 2B | 2B | — (OOM card) |

---

## Deploy — Brev

```bash
# Create instance (H100 80GB — enables 8B models)
brev create vlm-race --gpu-name H100 --type hyperstack_H100

# Wait for SHELL = READY
brev ls

# Deploy scripts
for script in gradio_compare_vlm compare_vlm_setup; do
  B64=$(base64 -i path/to/${script}.py | tr -d '\n')
  brev exec vlm-race "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done

# Launch (streams live progress; URL at end + written to /tmp/gradio_url.txt)
brev exec vlm-race "export HF_TOKEN=hf_... && export PATH=~/.local/bin:~/.cargo/bin:\$PATH && python3 /tmp/compare_vlm_setup.py"
```

---

## Deploy — SSH instance

```bash
# Deploy scripts
for script in gradio_compare_vlm compare_vlm_setup; do
  B64=$(base64 -i path/to/${script}.py | tr -d '\n')
  ssh user@<host> "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done

# Launch detached
ssh user@<host> "nohup env HF_TOKEN=hf_... PYTHONUNBUFFERED=1 PATH=~/.local/bin:~/.cargo/bin:\$PATH python3 /tmp/gradio_compare_vlm.py > /tmp/vlm_race.log 2>&1 &"

# Poll for URL
ssh user@<host> "grep -E 'ready|ERR|gradio.live' /tmp/vlm_race.log | tail -10"
```

---

## Gradio UI features

- **Model status table:** live load status (ready / error / OOM) per model at launch
- **Video upload:** clip info (resolution · FPS · duration) shown on upload
- **Demo prompts:** dropdown presets for quick testing
- **Advanced settings** (collapsible): system/user prompts, fps, max_pixels, max_tokens, CR2 checkpoint override, per-model timeout
- **Live inference status:** per-second stopwatch, phase label (Preprocessing / Prefilling / Generating), unicode progress bar
- **Winner badge:** 🏆 on lowest `inference_s` among successful runs
- **Results JSON:** saved to `/tmp/vlm_race_results.json` after each run

---

## Env vars

| Var | Default | Notes |
|---|---|---|
| `HF_TOKEN` | — | Required for Nemotron (gated model) |
| `RACE_TIMEOUT_S` | `300` | Per-model inference timeout in seconds |
| `GRADIO_SHARE` | `true` | Set `false` for local machines (avoids tunnel hang) |
| `GRADIO_PORT` | `7860` | Port |
| `CR2_DIR_2B` / `CR2_DIR_8B` | `~/models/Cosmos-Reason2-{2,8}B` | Local weight path |
| `NEMOTRON_DIR` | `~/models/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | Local weight path |
| `QWEN_DIR_2B` / `QWEN_DIR_8B` | `~/models/Qwen3-VL-{2,8}B-Instruct` | Local weight path |
| `CR2_CHECKPOINT` | auto | Override CR2 model (any HF ID or local path) |

---

## Timing benchmarks

| GPU | Models loaded | CR2 load | Qwen3-VL load | CR2 inference |
|---|---|---|---|---|
| H100 PCIe (Brev) | CR2-8B + Qwen3-VL-8B | 3–4s | 3–4s | ~130–180s (8B, shared VRAM) |
| A40 | CR2-2B + Qwen3-VL-2B | 2–3s | 2–3s | ~77s (from byo-video baseline) |

Note: times above reflect shared-VRAM mode. Sequential load/unload will match single-model byo-video baselines.

---

## Roadmap — Configurable comparison modes

| Mode | Left | Right | Purpose |
|---|---|---|---|
| CR2 vs Nemotron | Cosmos Reason 2 | Nemotron-Nano-12B | Architectural comparison |
| CR2 vs Qwen3-VL | Cosmos Reason 2 | Qwen3-VL | Open-source vs NVIDIA |
| CR2 checkpoint race | CR2-2B | CR2-8B | Size vs quality tradeoff |

Implementation: mode selector in UI (`RACE_MODE` env var). Each pairwise mode loads only 2 models, reducing VRAM pressure and enabling CR2-8B + Nemotron-12B on 40GB GPUs.

---

## Common failure modes

| Symptom | Fix |
|---|---|
| Nemotron: `selective_scan_cuda undefined symbol` | mamba-ssm built for torch 2.6, running torch 2.9. Error card shown. Need CUDA toolkit instance to rebuild. |
| Nemotron: `mamba-ssm cannot be imported` | Not installed. `cd ~/cosmos-reason2 && uv pip install 'mamba-ssm @ https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'` |
| All models fail to load | `torch.cuda.is_available()` → False. Wrong CUDA/torch build. `uv sync --extra cu128` in cosmos-reason2 dir. |
| Gradio tunnel hangs | Set `GRADIO_SHARE=false`, access via `http://localhost:7860` or SSH port-forward. |
| Inference appears stuck | First-token prefill on 8B models takes 60–120s for video input. Normal. Stopwatch shows elapsed time. |
| Blank Gradio page | Tunnel hiccup on launch — restart process, new URL is generated. |
