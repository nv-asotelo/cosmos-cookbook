# CLAUDE.md — Worker Safety · Cosmos Reason 2 Inference

## Step 0 — Install and authenticate Brev CLI

Before any `brev` command can run, the Brev CLI must be installed and the user must be logged in.

### Check if brev is already installed

```bash
which brev
brev --version
```

If `brev` is found and the version is ≥ 0.6.322, skip to "Authenticate" below.

### Install brev-cli (if not found)

**macOS:**
```bash
brew install brevdev/homebrew-brev/brev
```

**Linux / WSL (curl installer):**
```bash
# Download first, inspect, then execute — do NOT pipe directly to bash
curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh -o /tmp/install-brev.sh
# Review the script before running: cat /tmp/install-brev.sh
sudo bash /tmp/install-brev.sh
```

> **Security notice:** The Linux installer runs with `sudo` and modifies system paths.
> The script is open-source at https://github.com/brevdev/brev-cli — review it before execution.
> Never pipe unknown scripts directly to `sudo bash` without inspection.

Verify installation:
```bash
which brev
brev --version
```

### Authenticate (REQUIRED — cannot be automated)

```
⚠️  HUMAN ACTION REQUIRED — Claude cannot complete this step.
```

`brev login` opens a browser for OAuth. This must be done by the user, once per machine, before any `brev create` or `brev exec` call will work.

**Tell the user:**
> To use this recipe, you need to authenticate with Brev once. Please run the following in your terminal:
>
> ```bash
> brev login
> ```
>
> A browser window will open. Log in with your NVIDIA SSO or GitHub account. When the terminal shows "Logged in!", you're done. Claude will continue from there.

After the user confirms login is complete, proceed to the Quick Start below.

---

## Quick Start for Agents

```bash
# ── PRE-FLIGHT CHECKLIST ────────────────────────────────────────────────────
# [✓] brev installed (Step 0 above)
# [✓] brev login completed by user (browser OAuth — one-time per machine)
# [✓] User has accepted NVIDIA Open Model License at:
#     https://huggingface.co/nvidia/Cosmos-Reason2-2B
# [✓] HF_TOKEN set: export HF_TOKEN=hf_...

# ── STEP 1: Select org ──────────────────────────────────────────────────────
brev org set asotelo-test-org              # ALWAYS set org first — wrong org = wrong bill

# ── STEP 2: Select best available instance (run before brev create) ─────────
# Priority order: availability > inference speed > cost.
# Query each tier and use the first one with results. Do NOT skip this step.
#
# Tier 1 — B200 (native fp4, ~5× faster than A100, ~$6.76/hr)
brev search --gpu-name B200 --sort price
#
# Tier 2 — H200 (3.5× faster than A100, ~$4.20/hr, effectively same session cost as A100)
brev search --gpu-name H200 --sort price
#
# Tier 3 — H100 (2.5× faster than A100, 34% cheaper total cost for multi-run sessions)
# H100 is 34% cheaper than A100 for multi-run sessions (2.5x faster — sessions finish sooner)
brev search --gpu-name H100 --sort price
#
# Tier 4 — A100 80G fallback (verified working, massedcompute preferred)
brev search --gpu-name A100 --min-vram 80 --sort price

# ── STEP 3: Create instance using chosen type ───────────────────────────────
# Replace <CHOSEN_TYPE> with the first available result from Step 2 above.
# Default if all tiers available: use H200 (best availability:speed:cost ratio).
brev create worker-safety \
  --type <CHOSEN_TYPE> \
  --startup-script @launch.sh             # launch.sh must be in the same directory

# Monitor (poll until complete):
brev exec worker-safety "tail -20 ~/inference.log"

# Port-forward FiftyOne to localhost:
brev port-forward worker-safety -p 5151:5151
# Then open: http://localhost:5151
```

**Minimum inputs from user:** `HF_TOKEN` (once per session). `brev login` must have been run at least once on this machine. `brev` must be installed (see Step 0).

---

## Model

`nvidia/Cosmos-Reason2-2B`

- **Access:** Gated — accept NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-2B
- **Size:** ~4.6 GB
- **License:** NVIDIA Open Model License
- **Auth:** `export HF_TOKEN=hf_...` — do NOT use `huggingface-cli login` (interactive)

## Compute Requirements

1× NVIDIA GPU with 40 GB+ VRAM. Select the best available instance using the priority ladder below.

**Priority order: availability → inference speed → cost**

| Priority | GPU | Example instance | $/hr | vs A100 speed | Session cost (3 runs) | Notes |
|----------|-----|-----------------|------|---------------|-----------------------|-------|
| 1 | B200 | `verda_B200` | ~$6.76 | ~5× | ~$8.65 | Native fp4 kernel; Blackwell arch; single provider |
| 2 | **H200** ★ | `gpu-h200-sxm.1gpu-16vcpu-200gb` | ~$4.20 | ~3.5× | ~$6.72 | **Recommended default.** Widely available; same total cost as A100; 3.5× faster. |
| 3 | H100 | `hyperstack_H100` | ~$2.28 | ~2.5× | ~$4.52 | **34% cheaper than A100 for multi-run sessions (2.5× faster).** Native fp8 (Hopper). Preferred over A100 for any session with 2+ runs. |
| 4 | A100 80G | `massedcompute_A100_sxm4_80G` | ~$1.49 | 1× | ~$6.85 | Fallback only. GCP A100 types fail with capacity errors — use massedcompute. |

> **Why H200 over A100?** The A100's $1.49/hr looks cheaper, but a 3-run experiment session takes 4.6 hours ($6.85 total). The same session on an H200 at $4.20/hr completes in ~1.6 hours ($6.72) — effectively the same total cost at 3.5× the speed. For anyone onboarding to Cosmos Reason, A100 is the wrong default.

> **Why not always B200?** B200 is currently single-provider (`verda`) — if that provider is at capacity, there is no fallback within the same tier. H200 is available across multiple providers. Always query availability before deciding.

**Instance user:** depends on provider. massedcompute/shadeform instances use `shadeform`; nebius uses `user`. Use `$HOME` throughout — do not hardcode `/home/ubuntu` or any other username.

---

## Step-by-Step Execution

### Step 1 — Select org

```bash
brev org set asotelo-test-org
```

`asotelo-test-org` is the correct personal org. **Never skip this step.** Brev CLI org context resets between calls — set it before every `brev create`.

### Step 2 — Select best available instance

Query each tier in order. Use the first tier that returns results. The agent must run these queries at runtime — do not assume any instance type is available without checking.

```bash
brev search --gpu-name B200 --sort price
brev search --gpu-name H200 --sort price
brev search --gpu-name H100 --sort price
brev search --gpu-name A100 --min-vram 80 --sort price
```

**Decision logic:**
- If B200 results returned → use first result (native fp4, highest throughput)
- If H200 results returned → use first result (default recommendation)
- If H100 results returned → use first result (cheapest total for 3+ run sessions)
- Fallback → `massedcompute_A100_sxm4_80G` (verified working; GCP A100 types fail)

### Step 3 — System dependencies

```bash
sudo apt-get update -y
sudo apt-get install -y curl ffmpeg git git-lfs
git lfs install
```

### Step 4 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Step 5 — HuggingFace authentication

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

Do NOT run `huggingface-cli login` — it is interactive and will hang. The `HF_TOKEN` environment variable is sufficient for model downloads.

### Step 6 — Clone cosmos-reason2

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-reason2.git $HOME/cosmos-reason2
cd $HOME/cosmos-reason2
```

`GIT_LFS_SKIP_SMUDGE=1` is required — the public LFS quota on cosmos-reason2 is frequently exhausted. Skip LFS on clone; model weights are downloaded separately via HuggingFace in Step 8.

### Step 7 — Clone cosmos-cookbook

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git $HOME/cosmos-cookbook
```

### Step 8 — Create Python environment

```bash
cd $HOME/cosmos-reason2
uv sync --extra cu128
```

Do NOT run `source .venv/bin/activate`. The activated venv does NOT persist across `brev exec` subshell calls. Always invoke Python as:

```bash
$HOME/cosmos-reason2/.venv/bin/python3
```

Or use `uv run` from within the cosmos-reason2 directory.

### Step 9 — Install recipe dependencies

```bash
cd $HOME/cosmos-reason2
uv pip install fiftyone
```

Do NOT use `pip install` — uv venvs have no pip binary. Always use `uv pip install`.

### Step 10 — Download model weights

```bash
HF_TOKEN=$HF_TOKEN $HOME/cosmos-reason2/.venv/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/Cosmos-Reason2-2B',
    local_dir='$HOME/cosmos-reason2/models/Cosmos-Reason2-2B',
    token='$HF_TOKEN'
)
print('Model download complete')
"
```

### Step 11 — Copy recipe files

```bash
cp $HOME/cosmos-cookbook/docs/recipes/inference/reason2/worker_safety/worker_safety.py \
   $HOME/cosmos-reason2/worker_safety.py
cp -r $HOME/cosmos-cookbook/docs/recipes/inference/reason2/worker_safety/assets \
   $HOME/cosmos-reason2/assets
cp $HOME/cosmos-cookbook/docs/recipes/inference/reason2/worker_safety/run_headless.py \
   $HOME/cosmos-reason2/run_headless.py
```

### Step 12 — Run inference headlessly

```bash
cd $HOME/cosmos-reason2
nohup $HOME/cosmos-reason2/.venv/bin/python3 run_headless.py \
  --results ~/inference_results.json > ~/inference.log 2>&1 &
echo "Inference PID: $!"
```

Use `run_headless.py` instead of executing `worker_safety.py` directly. The recipe's `session.wait()` call blocks forever on a headless machine. `run_headless.py` patches `fo.launch_app` to a no-op before running the recipe.

Monitor progress:
```bash
tail -f ~/inference.log
```

Serve FiftyOne when done:
```bash
$HOME/cosmos-reason2/.venv/bin/python3 run_headless.py --serve --port 5151
```

### Step 13 — Verify results

```bash
$HOME/cosmos-reason2/.venv/bin/python3 -c "
import json
data = json.load(open('$HOME/inference_results.json'))
labeled = [r for r in data if r.get('safety_label')]
errors  = [r for r in data if r.get('cosmos_error')]
print(f'Total: {len(data)}  Classified: {len(labeled)}  Errors: {len(errors)}')
"
```

---

## Known Issues

| # | Symptom | Fix |
|---|---------|-----|
| 1 | `brev create` fails with capacity error | Run the instance selection ladder in Step 2 — query B200 → H200 → H100 → A100 in order and use the first available type |
| 2 | GCP A100 types fail (`a2-ultragpu`, `a2-highgpu`) | Use `massedcompute_A100_sxm4_80G` for A100 — GCP A100 quota is frequently exhausted |
| 3 | `pip: command not found` | Use `uv pip install` instead of `pip install` |
| 4 | `source .venv/bin/activate` has no effect in `brev exec` | Use full path: `$HOME/cosmos-reason2/.venv/bin/python3` |
| 5 | `GIT_LFS_SKIP_SMUDGE` not set → clone fails/hangs | Set `GIT_LFS_SKIP_SMUDGE=1` before every git clone |
| 6 | `huggingface-cli login` hangs | Use `export HF_TOKEN=hf_...` — no interactive login needed |
| 7 | `DatasetNotFoundError: safe-unsafe-worker-behavior` | Use `fo.list_datasets()[0]` — dataset name is the HF slug |
| 8 | `session.wait()` blocks headless execution | Use `run_headless.py` — patches `fo.launch_app` to no-op |
| 9 | FiftyOne port unreachable | Bind with `address="0.0.0.0"`, then `brev port-forward -p 5151:5151` |
| 10 | Wrong org billed | Always run `brev org set asotelo-test-org` before `brev create` |
| 11 | Instance user is not `ubuntu` | Use `$HOME` everywhere; massedcompute/shadeform uses `shadeform`, nebius uses `user` |
| 12 | `torch._inductor` stalls nvfp4/fp8 inference for 4+ hours | Set `os.environ["TORCHDYNAMO_DISABLE"] = "1"` as the **first line** before any torch import |

---

## Performance Optimization

Reference timings on A100 SXM4 80G. H200/H100 will be faster — scale by multiplier in the instance table above.

| Configuration | max_new_tokens | fps | Est. time (40 videos) on A100 | Est. time on H200 |
|---------------|---------------|-----|-------------------------------|-------------------|
| Baseline float16 | 1024 | 4 | ~83 min | ~24 min |
| Optimized float16 | 512 | 2 | ~50–60 min (est.) | ~15–17 min |
| nvfp4 quantized | 512 | 2 | ~25–35 min (est.) | ~7–10 min |
| fp8 quantized | 512 | 2 | ~25–35 min (est.) | ~7–10 min |

**TORCHDYNAMO_DISABLE=1 is required for nvfp4 and fp8 quantized checkpoints.** Without it, `torch._inductor` triggers JIT kernel compilation on first forward pass — 16 compile_worker processes, GPU at <40% utilization, 0 videos processed after 4+ hours. Add this before any torch import:

```python
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # must be before any torch import
import torch
import transformers
```

**To quantize before inference:**
```bash
cd $HOME/cosmos-reason2
$HOME/cosmos-reason2/.venv/bin/python3 scripts/quantize.py \
  --model $HOME/cosmos-reason2/models/Cosmos-Reason2-2B \
  -o $HOME/cosmos-reason2/models/quantized
```
Quantization takes ~25 min one-time. Use the quantized model for all subsequent runs.

---

## Success Criteria

| Check | Command | Expected |
|-------|---------|----------|
| GPU visible | `nvidia-smi` | H200/H100/A100 80 GB shown (whichever was provisioned) |
| CUDA available | `python3 -c "import torch; print(torch.cuda.is_available())"` | `True` |
| Inference complete | `cat ~/inference_results.json \| python3 -c "import json,sys; d=json.load(sys.stdin); print(len([r for r in d if r['safety_label']]), 'classified')"` | `40 classified` (or N/40) |
| No errors | Same script, check `cosmos_error` | `0 errors` |
| FiftyOne loads | `python3 -c "import fiftyone as fo; print(fo.list_datasets())"` | Dataset name printed |

---

## Using Your Own Videos

The default setup runs inference on `pjramg/Safe_Unsafe_Test` (40 HuggingFace videos). To classify your own warehouse footage instead, use `run_headless.py --video-dir`. No changes to `worker_safety.py` are needed.

### Step 1 — Upload your videos to the instance

```bash
# From your LOCAL machine
brev copy ./my_videos/ worker-safety:/home/shadeform/my_videos/
```

Supported video formats: `.mp4` `.avi` `.mov` `.mkv` `.webm` `.m4v`

### Step 2 — Run inference on your videos

```bash
# On the instance (via brev exec or brev shell)
$HOME/cosmos-reason2/.venv/bin/python3 run_headless.py \
  --video-dir ~/my_videos/ \
  --results ~/my_results.json
```

### Step 3 — Browse results in FiftyOne

```bash
$HOME/cosmos-reason2/.venv/bin/python3 run_headless.py \
  --video-dir ~/my_videos/ \
  --results ~/my_results.json \
  --serve --port 5151
```

Then on your local machine: `brev port-forward worker-safety -p 5151:5151` → open `http://localhost:5151`

### Using a different HuggingFace dataset

```bash
$HOME/cosmos-reason2/.venv/bin/python3 run_headless.py \
  --hf-dataset your-org/your-dataset \
  --results ~/my_results.json
```

The `--hf-dataset` flag is ignored when `--video-dir` is also set.

### How it works

`run_headless.py` intercepts `fo.load_dataset()` and `fouh.load_from_hub()` before executing `worker_safety.py`. Your local video directory is pre-loaded into a FiftyOne dataset named `custom_inference`. The recipe sees the same FiftyOne API regardless of data source. Output format is identical (`inference_results.json`).

---

## Companion Files

| File | Purpose |
|------|---------|
| `run_headless.py` | Headless inference runner — replaces interactive REPL |
| `launch.sh` | Brev startup script — full env + inference, one command |
| `restore_labels.py` | Migrate labels to new instance without re-running inference |

---

## Cosmos Metadata

| Field | Value |
|-------|-------|
| Workload | inference |
| Domain | domain:industrial |
| Technique | technique:reasoning |
| Tags | inference, reason-2, safety |
| Summary | Zero-shot warehouse safety inspection using Cosmos Reason 2 to classify worker behaviors from video without custom model training. |
