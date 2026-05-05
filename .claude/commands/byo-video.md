# /byo-video — Cosmos BYO-Video Demo

**Single-model deployment skill.** Deploys any supported model (Cosmos Reason2, Nemotron-Nano-12B-v2-VL, Qwen3-VL, etc.) to a Gradio web UI at a `gradio.live` public URL — user uploads video in browser.

For multi-model side-by-side comparison, use `/vlm-race` (separate skill, separate instance required).

Default output: **Gradio web UI at a `gradio.live` public URL** — user uploads video in browser.

**Canonical scripts (stable, versioned — do not read from /tmp/):**
- `~/.claude/scripts/gradio_cr2_byo.py`       — Gradio app (all supported models)
- `~/.claude/scripts/byo_video_setup.py`      — Bootstrap + launch script
- `~/.claude/scripts/cosmos_deploy_monitor.py` — Local deployment monitor (state machine, checklist, provider rotation)

---

**Primary launch command (all environments):**
```bash
python3 /tmp/byo_video_setup.py
```
This script shows live step-by-step progress with ETAs for every install stage, then prints a clickable hyperlink to the Gradio UI. The URL is also written to `/tmp/gradio_url.txt` for agent capture. Deploy it to the instance before running (see deploy section below).

---

## AGENT PROTOCOL — Main session handles interactive phases

**Phase split.** `AskUserQuestion` is only available in the main session context — subagent
environments do not have it as a deferred tool. Therefore:

- **Main session** runs PHASE 0 (pre-checks) and PHASE 1 (picker) using `AskUserQuestion`.
- **Observer agent** is spawned after all parameters are resolved and handles PHASE 2–6
  (provisioning → deployment → URL capture) with the resolved values baked into its prompt.

### Main session — PHASE 0: Pre-checks

1. Run `brev ls` via Bash. Capture the full output.
2. Note stopped H100/H200 instances (priority-0 candidates for Q3).
3. Note any RUNNING instances (may be reusable).

Display initial panel:

```
╔══════════════════════════════════════════════════════════════╗
║  Cosmos BYO-Video  ·  Pre-checks                             ║
╠══════════════════════════════════════════════════════════════╣
║  Existing instances:                                         ║
║    [list each: name | status | GPU | type]                   ║
║    (or "None found")                                         ║
╚══════════════════════════════════════════════════════════════╝
```

### Main session — PHASE 1: Picker

Load `AskUserQuestion` via ToolSearch: `query: "select:AskUserQuestion"`, then fire all 3
questions in a single call (see PICKER section below for the full call).

Once all answers are received, resolve `MODEL_ID`, `MODEL_SIZE`, `INFERENCE_BACKEND`,
`DEPLOY_TARGET` per the answer→env var mapping table.

If Q3 answer is "Existing Brev" or "SSH target", fire one follow-up `AskUserQuestion` to collect
the instance name or host before spawning the observer.

### Main session — Spawn observer (after PHASE 1 complete)

With all parameters resolved, spawn the observer:

```python
Agent({
  description: "byo-video observer",
  prompt: f"""
You are the byo-video observer. You own PHASE 2 through PHASE 6 of the Cosmos BYO-Video
deployment (provisioning → scripts → setup → URL capture). PHASE 0 and PHASE 1 are already
complete — do NOT re-run them.

Resolved parameters:
  MODEL_ID           = <MODEL_ID>
  MODEL_SIZE         = <MODEL_SIZE>
  INFERENCE_BACKEND  = <INFERENCE_BACKEND>
  DEPLOY_TARGET      = <DEPLOY_TARGET>

Brev pre-check output (from PHASE 0):
<brev_ls_output>

Your rules:
- Never run anything silently. Every action produces visible output.
- Never use Bash(run_in_background=true).
- Display the LIVE STATUS PANEL after every phase change.
- If blocked: show the panel with the block reason, then call AskUserQuestion (load via
  ToolSearch first: query "select:AskUserQuestion") with recovery options.
- Never ask the user to run commands themselves.

Begin with PHASE 2 immediately using the resolved parameters above.
"""
})
```

The observer executes PHASE 2–6 as defined in the OBSERVER PROTOCOL below and returns when
a Gradio URL is live (or an unrecoverable error occurs).

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
        { label: "HF Transformers", description: "~8–12 min setup, no quantization, simpler" }
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
        { label: "New Brev instance", description: "Agent provisions appropriate GPU tier for your model" },
        { label: "Existing Brev", description: "Provide instance name when prompted" },
        { label: "SSH target", description: "Provide user@host or IP when prompted" },
        { label: "Local machine", description: "Agent runs nvidia-smi locally" }
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
| Q2 Model | Cosmos Reason2 2B | `MODEL_ID=nvidia/Cosmos-Reason2-2B` · `MODEL_SIZE=2B` |
| Q2 Model | Cosmos Reason2 8B | `MODEL_ID=nvidia/Cosmos-Reason2-8B` · `MODEL_SIZE=8B` |
| Q2 Model | Cosmos Reason2 32B | `MODEL_ID=nvidia/Cosmos-Reason2-32B` · `MODEL_SIZE=32B` |
| Q2 Model | Something else | Fire SOMETHING ELSE sub-picker (see below) |
| Q3 Env | New Brev instance | `DEPLOY_TARGET=brev:new` |
| Q3 Env | Existing Brev | Follow-up AskUserQuestion: "Instance name?" → `DEPLOY_TARGET=brev:<name>` |
| Q3 Env | SSH target | Follow-up AskUserQuestion: "user@host or IP?" → `DEPLOY_TARGET=ssh:<user@host>` |
| Q3 Env | Local machine | `DEPLOY_TARGET=local` |

**SOMETHING ELSE sub-picker** — fire immediately when user selects "Something else" for Q2:

```
AskUserQuestion({
  questions: [
    {
      question: "Which model?",
      header: "Model",
      multiSelect: false,
      options: [
        { label: "Cosmos3-Nano-Reasoner", description: "nvidia/Cosmos3-Nano-Reasoner (8B, private, ≥40GB VRAM, HF_TOKEN required)" },
        { label: "Cosmos3-Reasoner-32B", description: "nvidia/Cosmos3-Reasoner-32B (32B, gated, nvidia org + HF_TOKEN required)" },
        { label: "Cosmos Transfer 2.5", description: "nvidia/Cosmos-Transfer2.5 (generation model, ≥80GB VRAM)" },
        { label: "Nemotron-Nano-12B-v2-VL", description: "nvidia/Nemotron-Nano-12B-v2-VL-BF16 (12B, gated, vLLM only, ≥40GB VRAM)" },
        { label: "Non-NVIDIA models", description: "Best-effort only, not officially supported" }
      ]
    }
  ]
})
```

| Selection | Sets |
|---|---|
| Cosmos3-Nano-Reasoner | `MODEL_ID=nvidia/Cosmos3-Nano-Reasoner` · `MODEL_SIZE=C3-8B` |
| Cosmos3-Reasoner-32B | `MODEL_ID=nvidia/Cosmos3-Reasoner-32B` · `MODEL_SIZE=C3-32B` |
| Cosmos Transfer 2.5 | `MODEL_ID=nvidia/Cosmos-Transfer2.5` · `MODEL_SIZE=32B` |
| Nemotron-Nano-12B-v2-VL | `MODEL_ID=nvidia/Nemotron-Nano-12B-v2-VL-BF16` · `MODEL_SIZE=NEM-12B` |
| Non-NVIDIA models | Show disclaimer inline, then fire Qwen sub-picker |

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

Once `MODEL_ID`, `MODEL_SIZE`, `INFERENCE_BACKEND`, `DEPLOY_TARGET` are all resolved: spawn the
OBSERVER AGENT as described in the AGENT PROTOCOL section above. No further questions from the
main session.

---

## OBSERVER PROTOCOL

The observer receives resolved parameters in its prompt and executes PHASE 2–6. Each phase ends
with a panel update. The panel is the user's single source of truth — it shows instance, GPU,
rate, elapsed time, ETA, cost, and checklist state. Never replace panel updates with prose
summaries.

**Elapsed timer rule:** `PROVISION_START_TS` is recorded once at the top of PHASE 2. On every
30s poll cycle (PHASE 3, PHASE 5, PHASE 6), compute `elapsed = int(time.time() - PROVISION_START_TS)`
and embed it in the panel header line as `elapsed: Xm Ys`. Reprint the full panel on every poll
cycle — not only on phase transitions. This keeps elapsed time accurate to within 30s without
flooding the session with rapid refreshes.

---

### PHASE 2 — PROVISION

Record `PROVISION_START_TS` = now (Python: `import time; PROVISION_START_TS = time.time()`).

**HF_TOKEN:** Auto-read by the setup script from `~/.cache/huggingface/token` on the remote instance.
Do NOT ask the user for it. Do NOT pass it as a CLI arg. The script handles it.

**Look up MODEL_SIZE in MODEL_GPU_REQUIREMENTS** (see reference section below) to select GPU type.

**For `DEPLOY_TARGET=brev:new`:**

1. Check PHASE 0 pre-check results for stopped instances with a compatible GPU:
   - If found: fire `AskUserQuestion`: "Found stopped <name> (<GPU type>) — restart it or provision fresh?"
     Options: "Restart <name> (existing rate)" · "Provision new <type> (~$<rate>/hr)"
   - If not found: proceed directly to `brev create`.

2. Run `brev create`:
   - `brev create <name> --type <provider_type>` — one Bash call
   - Name convention: `cr2-<modelsize>-<timestamp-short>` (e.g., `cr2-2b-0505`)
   - Provider type from MODEL_GPU_REQUIREMENTS table

3. Display cost context immediately:

```
╔══════════════════════════════════════════════════════════════╗
║  Cosmos BYO-Video · <instance>  (elapsed: Ns)                ║
║  <GPU label> · $<rate>/hr · <MODEL_ID> (<MODEL_SIZE>, <be>)  ║
╠══════════════════════════════════════════════════════════════╣
║  [✓] Pre-checks                                              ║
║  [✓] Model & environment selected                            ║
║  [→] Provisioning: <instance> (<GPU type>)                   ║
║  [ ] SHELL READY                                             ║
║  [ ] Scripts deployed                                        ║
║  [ ] Setup running                                           ║
║  [ ] Gradio live                                             ║
╠══════════════════════════════════════════════════════════════╣
║  Cost so far: $0.00  |  Est. setup cost: ~$<low>–$<high>     ║
║  Brev dashboard: https://brev.dev                            ║
╚══════════════════════════════════════════════════════════════╝
```

Cost estimates by GPU tier:
- H100 SXM ($3.54/hr): 10 min setup ~$0.59 · 20 min ~$1.18
- H200 SXM ($4.20/hr): 20 min setup ~$1.40 · 30 min ~$2.10

**IDLE BILLING HANDLER** — triggers if instance reaches SHELL READY but setup hasn't launched for >5 min:

```
⚠️  <name> is idle — <elapsed> at $<rate>/hr. $<burned> spent so far.
    Setup not yet started.
    Options: [s] Start setup now  [d] Delete instance  [w] Wait 5 more min
```

Do NOT auto-terminate. If [w]: reset timer, repeat at +5 min.

---

### PHASE 3 — WAIT FOR SHELL READY

Poll `brev ls` every 30s via Bash (one call per poll). Reprint the full panel each poll with
updated elapsed time computed from `PROVISION_START_TS`:

```
║  Cosmos BYO-Video · <instance>  (elapsed: Xm Ys)                ║
...
║  [→] Waiting for SHELL READY (polling every 30s)                 ║
```

**On UNHEALTHY:**
1. Warn immediately — display panel with `[!] UNHEALTHY — recovery window: 3m 00s`.
2. Poll every 30s for 3 minutes. Update countdown each poll.
3. If recovered: run `brev exec <name> "nvidia-smi"` — if GPU responds, continue. If not, treat as BLOCKED.
4. If still UNHEALTHY after 3 minutes: auto-recover silently.
   - Try `brev reset <name>`. If succeeds → re-enter poll loop.
   - If reset fails: `brev delete <name>`, rotate to next provider in PROVIDER FALLBACK table. Emit one line: `✗ <type> UNHEALTHY — deleted, trying <next-type>`.
   - Keep rotating until SHELL READY or all providers exhausted.
   - Only AskUserQuestion when all providers have been tried and all failed.

On SHELL READY: update panel `[✓] SHELL READY`, proceed to PHASE 4.

---

### PHASE 4 — DEPLOY SCRIPTS

Read and base64-encode each script. One Bash call per file. One Bash call per deploy.

```bash
# Step 1: read + encode byo_video_setup.py (Bash call 1)
python3 -c "import base64, sys; sys.stdout.write(base64.b64encode(open('/Users/asotelo/.claude/scripts/byo_video_setup.py','rb').read()).decode())"
# → capture B64_SETUP

# Step 2: deploy byo_video_setup.py to instance (Bash call 2)
brev exec <name> "python3 -c \"import base64; open('/tmp/byo_video_setup.py','wb').write(base64.b64decode('<B64_SETUP>'))\""

# Step 3: read + encode gradio_cr2_byo.py (Bash call 3)
python3 -c "import base64, sys; sys.stdout.write(base64.b64encode(open('/Users/asotelo/.claude/scripts/gradio_cr2_byo.py','rb').read()).decode())"
# → capture B64_GRADIO

# Step 4: deploy gradio_cr2_byo.py to instance (Bash call 4)
brev exec <name> "python3 -c \"import base64; open('/tmp/gradio_cr2_byo.py','wb').write(base64.b64decode('<B64_GRADIO>'))\""
```

If `gradio_cr2_byo.py` is >98KB, use `brev copy` instead of base64 (avoids brev exec argument buffer overflow):
```bash
brev copy <name> ~/.claude/scripts/gradio_cr2_byo.py /tmp/gradio_cr2_byo.py
```

Update panel: `[✓] Scripts deployed`

---

### PHASE 5 — SETUP LAUNCH + LOG TAIL

Record `SETUP_DISPATCHED_AT` = now.

Launch setup (one Bash call — nohup so brev exec returns immediately):
```bash
brev exec <name> "nohup bash -c 'export INFERENCE_BACKEND=<backend> MODEL_ID=<model_id> BREV_RATE_PER_HOUR=<rate> PATH=~/.local/bin:~/.cargo/bin:$PATH && python3 /tmp/byo_video_setup.py > /tmp/byo_video_setup.log 2>&1' &"
```

Update panel: `[→] Setup running`

**Log tail loop (every 30s):** Tail the log, then reprint the full panel with updated elapsed
from `PROVISION_START_TS`. Do not emit intermediate prose between panels.

```bash
brev exec <name> "tail -60 /tmp/byo_video_setup.log 2>/dev/null || echo '(log not yet written)'"
```

Parse log output for step completion markers. Update the step-level checklist in the panel,
including updated `elapsed: Xm Ys` in the header:

```
║  [→] Setup running                                           ║
║       [✓] Step 1: GPU detect + VRAM tier                    ║
║       [✓] Step 2: HF auth + token validate                  ║
║       [✓] Step 3: NGC API key                               ║
║       [✓] Step 4: uv package manager                        ║
║       [✓] Step 5: cosmos-reason2 repo                       ║
║       [→] Step 6: uv sync + CUDA libs  (running ~8 min)     ║
║       [ ] Step 7: PyAV + Gradio + requests                  ║
║       [ ] Step 9: Model weights download                    ║
║       [ ] Step 10: Gradio launch                            ║
```

Detect errors in log: scan for `Traceback`, `Error:`, `exit status 1`, `FAILED`. If found:
→ Update panel with `[✗] Setup failed` + error snippet.
→ AskUserQuestion: "Setup failed at Step N: <error>. What next?" with options:
  - "Retry setup on this instance"
  - "Provision a new instance"
  - "Abort and delete instance"
→ Do not attempt recovery without user direction.

---

### PHASE 6 — GRADIO LIVE + URL CAPTURE

Poll every 30s for the live flag:
```bash
brev exec <name> "cat /tmp/gradio_live.flag 2>/dev/null"
```

When flag is present, read the URL:
```bash
brev exec <name> "cat /tmp/gradio_url.txt 2>/dev/null"
```

**Compute total cost:** `(time.time() - PROVISION_START_TS) / 3600 * rate_per_hour`

Display final panel:

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
║  [✓] Gradio live                                             ║
╠══════════════════════════════════════════════════════════════╣
║  URL:  <gradio_url>                                          ║
║  Total setup cost: ~$<computed>  |  Link valid 72h           ║
║  Brev dashboard: https://brev.dev                            ║
╚══════════════════════════════════════════════════════════════╝
```

Send kill alert:
```bash
~/.claude/scripts/teams-notify.sh "Cosmos demo live at <url>. Kill when ready."
```

Do NOT auto-terminate the instance. Wait for explicit kill instruction from the user.

---

### SSH DEPLOYMENTS (non-Brev)

The observer handles SSH targets with the same panel, minus the `brev ls` polling.

Deploy scripts:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "python3 -c \"import base64; open('/tmp/byo_video_setup.py','wb').write(base64.b64decode('<B64>'))\""
ssh -i ~/.ssh/id_ed25519 <user@host> "python3 -c \"import base64; open('/tmp/gradio_cr2_byo.py','wb').write(base64.b64decode('<B64>'))\""
```

Launch setup:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "nohup bash -c 'export INFERENCE_BACKEND=<backend> MODEL_ID=<model_id> && python3 /tmp/byo_video_setup.py > /tmp/byo_video_setup.log 2>&1' &"
```

Tail logs:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "tail -60 /tmp/byo_video_setup.log 2>/dev/null"
```

URL capture:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "cat /tmp/gradio_live.flag 2>/dev/null"
ssh -i ~/.ssh/id_ed25519 <user@host> "cat /tmp/gradio_url.txt 2>/dev/null"
```

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
| Cosmos3-Nano-Reasoner | 8B VLM | 40 GB | `C3-8B` | Public; was Cosmos3-Reasoner-8B-Private |
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
- **file:// video protocol** — Gradio copies video to `/tmp/gradio_upload.mp4` and sends `file:///tmp/gradio_upload.mp4` to vLLM. `--allowed-local-media-path /tmp` is required (set automatically by setup script NEM-12B config).
- **FLASHINFER bypass** — `FLASHINFER_DISABLE_VERSION_CHECK=1` is injected automatically (flashinfer package/cubin version mismatch in cosmos-reason2 venv).
- **NVFP4-QAD variant** — requires a special vLLM build; not available in standard setup. Use BF16 or FP8.
- **Launch**: `MODEL_SIZE=NEM-12B INFERENCE_BACKEND=vllm python3 /tmp/byo_video_setup.py`

### Qwen3-VL notes

- **Public model** — no HF_TOKEN required
- **vLLM only** — uses same `file://` video URL protocol as Nemotron; HF inference backend not supported
- **`--allowed-local-media-path /tmp`** required — set automatically in QW3-* configs
- **Variant picker** — Gradio checkpoint dropdown shows Instruct, FP8, and Thinking for each size
- **Thinking variant** — extended reasoning mode; max_tokens should be ≥2048 for best results
- **VRAM**: 2B~8GB, 8B~20GB, 32B~64GB (FP8 halves these)
- **Launch**: `MODEL_SIZE=QW3-8B INFERENCE_BACKEND=vllm python3 /tmp/byo_video_setup.py`

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

### Step 5 — Kill alert (required)

When Alex is done:
```bash
~/.claude/scripts/teams-notify.sh "Cosmos demo done on brev/<name>. Kill when ready."
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

Canonical source: **`~/.claude/scripts/gradio_cr2_byo.py`** (versioned 2026-04-21).

Features (as of 2026-04-21):
- **Checkpoint selector** — Advanced Settings accordion has a preset dropdown (CR2-2B base, CR2-2B FP8, CR2-8B NVFP4, NIM 2B) plus a custom model ID field. Any HF model ID or local path is accepted.
- **On-demand load/unload** — switching checkpoints does `del model → gc.collect() → cuda.empty_cache()` before loading the next variant. VRAM is confirmed free before loading.
- **Run All Variants button** — sequential benchmark: FP8 → NVFP4 → NIM, each unloaded before the next. Results saved to `/tmp/byo_video_benchmark.json`.
- **Right-side status panel** — replaces the grey loading box. Shows Step N/5 WIP tracker (resolve / load / preprocess / prefill / generate) with ✅/⟳/— per step, plus live token metrics: prefill count, generated count (running), TTFT, inference time.
- **NIM mode** — calls NVCF API (`https://integrate.api.nvidia.com/v1/chat/completions`) via NGC_API_KEY (nvapi- prefix). Extracts up to 8 JPEG frames from the video and sends them as image content. No local model load needed.
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
| Video input | Up to 8 JPEG frames extracted by PyAV at selected fps |
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
| Nemotron: 400 error from vLLM on inference | Check that `--allowed-local-media-path /tmp` is in the vLLM serve command (set automatically by `byo_video_setup.py`). |
