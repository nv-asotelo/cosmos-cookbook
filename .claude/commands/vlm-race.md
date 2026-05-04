# /vlm-race — VLM Race: Multi-Model Video Comparison

Side-by-side inference comparison across Cosmos Reason 2, Nemotron-Nano-12B-v2, and Qwen3-VL.
Upload one video, get all model outputs with TTFT, inference time, tokens in/out, and a winner badge.

**Canonical scripts (stable, versioned — do not read from /tmp/):**
- `~/.claude/scripts/gradio_compare_vlm.py` — Gradio app (3-model race UI)
- `~/.claude/scripts/compare_vlm_setup.py`  — Bootstrap + launch (deps, model download, Gradio start)

**Results:** JSON at `/tmp/vlm_race_results.json` after each run (per-model status, text, TTFT, inference_s, e2e_s, tokens_in, tokens_out, winner flag).

---

## Models

| Column | Model | HF ID | VRAM (BF16) |
|---|---|---|---|
| 🌌 Cosmos Reason 2 | CR2-2B or CR2-8B | `nvidia/Cosmos-Reason2-{2,8}B` | ~5 GB / ~17 GB |
| 🤖 Nemotron-Nano-12B | Fixed | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | ~26 GB |
| 🔷 Qwen3-VL | Paired with CR2 size | `Qwen/Qwen3-VL-{2,8}B-Instruct` | ~5 GB / ~17 GB |

All models BF16. Sequential inference on one GPU — parallel racing requires multi-GPU (not implemented).

**Nemotron dependency:** requires `mamba-ssm` CUDA extension. Pre-built wheels only available for
torch ≤ 2.6. Instances running torch 2.9+cu128 (Brev Hyperstack, Horde A40) get a graceful error
card — other two models continue. Full 3/3 requires an instance with CUDA toolkit + torch 2.6
(e.g. Docker image `nvidia/cuda:12.4-devel`).

---

## VRAM auto-selection

Agent runs this before any setup:
```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
```

| Free VRAM | CR2 size | Qwen3-VL size | Nemotron | Expected |
|---|---|---|---|---|
| ≥ 80 GB | 8B | 8B | 12B | 3/3 (if mamba-ssm compatible) |
| ≥ 40 GB | 2B | 2B | 12B | 2/3 — Nemotron OOM card |
| < 40 GB | 2B | 2B | — | 2/3 — Nemotron OOM card |

---

## AGENT PROTOCOL

**USER MUST BRING:**

| What | How |
|---|---|
| Environment | "local", "brev", or "horde" |
| Instance name / IP | `brev ls` to confirm, or paste IP |
| HF_TOKEN | Required — Nemotron is gated |
| BYO video | Path on instance, or upload via UI once live |

**AGENT RUNS AUTONOMOUSLY (do not ask the user to run these):**
- VRAM check and model size selection
- All deploy steps (base64 encode, brev exec / ssh copy)
- Bootstrap via `compare_vlm_setup.py` (streams progress to terminal)
- URL capture from `/tmp/gradio_url.txt`
- Kill alert via pa-cli when Alex says done

**Crisp target:** Alex provides the 4 items above, agent runs everything else to URL.

---

## Deploy — Brev (recommended)

```bash
# Create instance (H100 80GB — enables all 3 models at 8B)
brev create vlm-race --gpu-name H100 --type hyperstack_H100

# Wait for SHELL READY
brev ls   # repeat until SHELL = READY

# Deploy both scripts
for script in gradio_compare_vlm compare_vlm_setup; do
  B64=$(base64 -i ~/.claude/scripts/${script}.py | tr -d '\n')
  brev exec vlm-race "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done

# Launch (streams live progress; URL printed at end + written to /tmp/gradio_url.txt)
brev exec vlm-race "export HF_TOKEN=hf_... && export PATH=~/.local/bin:~/.cargo/bin:\$PATH && python3 /tmp/compare_vlm_setup.py"

# Or: launch detached (survives SSH timeout; poll log instead)
brev exec vlm-race "nohup env HF_TOKEN=hf_... PYTHONUNBUFFERED=1 ... /home/shadeform/cosmos-reason2/.venv/bin/python3 /tmp/gradio_compare_vlm.py > /tmp/vlm_race_demo.log 2>&1 &"
brev exec vlm-race "grep -E 'ready|ERR|Running on' /tmp/vlm_race_demo.log"
```

**Brev org:** always use `asotelo-test-org` — NCA orgs are not Alex's budget.

---

## Deploy — Horde (asotelo org, A40 48 GB)

**SSH username is `horde`** — confirmed empirically. Not `ubuntu`, `nvidia`, or `asotelo`.

```bash
# Deploy scripts
for script in gradio_compare_vlm compare_vlm_setup; do
  B64=$(base64 -i ~/.claude/scripts/${script}.py | tr -d '\n')
  ssh horde@10.57.234.230 "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done

# Launch detached (model loads take ~3 min; nohup prevents SSH timeout kill)
ssh horde@10.57.234.230 "nohup env HF_TOKEN=hf_... PYTHONUNBUFFERED=1 CR2_DIR_2B=~/models/Cosmos-Reason2-2B NEMOTRON_DIR=~/models/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 QWEN_DIR_2B=~/models/Qwen3-VL-2B-Instruct GRADIO_SHARE=true ~/cosmos-reason2/.venv/bin/python3 /tmp/gradio_compare_vlm.py > /tmp/vlm_race.log 2>&1 &"

# Poll for URL (wait ~90s for models to load)
ssh horde@10.57.234.230 "grep -E 'ready|ERR|Running on|gradio.live' /tmp/vlm_race.log"
```

**Horde capacity API is stale** — reports availability that doesn't match actual pool. Existing instance
at 10.57.234.230 (A40) is the reliable fallback. Gradio share tunnel may fail on Horde — use SSH
port-forward (`ssh -L 7860:localhost:7860 horde@10.57.234.230`) as fallback.

---

## Deploy — asotelo-dt (local workstation, RTX5070 12 GB)

Nemotron OOM on RTX5070 — expected, handled gracefully with error card. CR2-2B + Qwen3-VL-2B run fine.

```bash
# Deploy (use deploy_dt.py script)
python3 /tmp/deploy_dt.py

# Launch (GRADIO_SHARE=false — avoids tunnel hang on local machine)
ssh asotelo@asotelo-dt.local "nohup env GRADIO_SHARE=false PYTHONUNBUFFERED=1 CR2_DIR_2B=~/models/Cosmos-Reason2-2B NEMOTRON_DIR=~/models/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 QWEN_DIR_2B=~/models/Qwen3-VL-2B-Instruct ~/models/venv/bin/python3 /tmp/gradio_compare_vlm.py > /tmp/vlm_race.log 2>&1 &"

# Access at http://localhost:7860 (no tunnel needed — local machine)
```

---

## Race UI

- **Basic:** upload MP4, pick demo prompt, click Start Comparison
- **Advanced:** edit system/user prompts, toggle thinking/reasoning, tune fps / resolution / max_tokens, change CR2 checkpoint
- **Metrics per column:** TTFT · Inference time · E2E time · Tokens in/out
- **Winner badge:** 🏆 on lowest `inference_s` among successful runs
- **OOM / error cards:** column shows reason + suggestion; other columns continue
- **Timeout:** per-model, default 300s; configurable in Advanced (`RACE_TIMEOUT_S` env var)
- **Clip info:** resolution · fps · duration shown on upload

---

## Env vars

| Var | Default | Notes |
|---|---|---|
| `HF_TOKEN` | — | Required for Nemotron (gated model) |
| `RACE_TIMEOUT_S` | 300 | Per-model inference timeout in seconds |
| `GRADIO_SHARE` | `true` | Set to `false` to disable public tunnel (use on local machines) |
| `GRADIO_PORT` | 7860 | Port |
| `CR2_DIR_2B` / `CR2_DIR_8B` | `~/models/Cosmos-Reason2-{2,8}B` | Local weight path override |
| `NEMOTRON_DIR` | `~/models/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | Local weight path override |
| `QWEN_DIR_2B` / `QWEN_DIR_8B` | `~/models/Qwen3-VL-{2,8}B-Instruct` | Local weight path override |
| `CR2_CHECKPOINT` | auto | Override CR2 model (any HF ID or local path) |
| `NEMOTRON_MODEL` | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | Override Nemotron HF ID |

---

## Timing benchmarks

| Environment | GPU | Models loaded | Load | Inference (CR2) |
|---|---|---|---|---|
| Brev Hyperstack | H100 PCIe | CR2-8B + Qwen3-VL-8B | 3–10s each | ~130–180s expected |
| Horde | A40 | CR2-2B + Qwen3-VL-2B | 2–4s each | ~77s (from byo-video A40 baseline) |
| asotelo-dt | RTX5070 | CR2-2B + Qwen3-VL-2B | 2–3s each | ~44s (extrapolated from H100 byo-video) |

Nemotron inference not yet benchmarked — mamba-ssm dependency unresolved on current instances.

---

## Common failure modes

| Symptom | Fix |
|---|---|
| Nemotron: `selective_scan_cuda undefined symbol` | mamba-ssm built for torch 2.6, running torch 2.9. Error card shown. Need CUDA toolkit instance to rebuild. |
| Nemotron: `mamba-ssm cannot be imported` | mamba-ssm not installed. `cd ~/cosmos-reason2 && uv pip install 'mamba-ssm @ https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'` |
| All models fail to load | Check `torch.cuda.is_available()` — if False, wrong CUDA/torch version. `uv sync --extra cu128` in cosmos-reason2 dir. |
| Gradio tunnel hangs (no URL) | Set `GRADIO_SHARE=false`, access via `http://localhost:7860` or SSH port-forward. |
| Inference hangs past timeout | Timeout fires at `RACE_TIMEOUT_S`; check `[TIMEOUT]` line in log. If no timeout log line, process was SIGKILL'd by OS OOM — check VRAM. |
| Port 7860 unreachable | `ss -tlnp | grep 7860` on instance. If not listening, Gradio crashed — check log. |
| Black frames / video error | PyAV patch missing. Confirm `gradio_compare_vlm.py` was deployed (not a custom script). |
| Horde SSH rejected | Username must be `horde`. Not `ubuntu`, `nvidia`, `root`, or `asotelo`. |
| Brev instance not in asotelo-test-org | Wrong org active. Switch org before `brev create`. |
| `brev login` fails with EOF | `brev login` requires a browser handoff — it cannot run via `! brev login` in Claude Code (non-TTY). Open a separate terminal tab, run `brev login` there, complete the browser prompt, then return to Claude Code. |

---

## Kill alert (required when done)

```bash
python3 ~/.claude/scripts/send_teams.py "VLM Race done on brev/vlm-race (H100 PCIe). Instance can be terminated now."
```

Do NOT auto-terminate. Notify Alex and wait for explicit kill confirmation.

---

## Roadmap — Next sprint scope

Current app: fixed 3-model race (all models run on every inference).

Alex's target: **configurable pairwise comparison modes**:

| Mode | Left | Right | Purpose |
|---|---|---|---|
| CR2 vs Nemotron | Cosmos Reason 2 | Nemotron-Nano-12B | Architectural comparison, same video |
| CR2 vs Qwen3-VL | Cosmos Reason 2 | Qwen3-VL | Open-source vs NVIDIA, same video |
| CR2 checkpoint race | CR2-2B | CR2-8B | Size vs quality tradeoff, same video |

Implementation: mode selector in Gradio UI (radio/dropdown). Each mode loads only 2 models,
freeing VRAM — enables CR2-8B + Nemotron-12B on 40GB GPUs (combined ~43GB vs 48GB A40).
Scripts: extend `gradio_compare_vlm.py` with `--mode` flag / env var `RACE_MODE`.
