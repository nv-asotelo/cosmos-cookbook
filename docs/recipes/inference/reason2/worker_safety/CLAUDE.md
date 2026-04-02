# CLAUDE.md — Cosmos Reason 2 · Brev Inference Guide

This file provides reusable Brev CLI patterns for deploying any Cosmos Reason 2 inference recipe on remote GPU instances. Recipe-specific setup steps (environment, dependencies, scripts) are defined in the recipe's own documentation; this file covers infrastructure selection, org management, and known Brev/GPU issues.

---

## Step 0 — Install and authenticate Brev CLI

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
# [✓] User has accepted NVIDIA Open Model License on HuggingFace
# [✓] HF_TOKEN set: export HF_TOKEN=hf_...
# [✓] NGC_API_KEY set if recipe uses NIM endpoints: export NGC_API_KEY=nvapi-...

# ── STEP 1: Select org ──────────────────────────────────────────────────────
brev org set asotelo-test-org              # ALWAYS set org first — wrong org = wrong bill

# ── STEP 2: Select best available instance ──────────────────────────────────
# Priority order: stoppable > non-stoppable, then availability > speed > cost.
# Stoppable instances (FEATURES column shows "S") can be paused without deletion
# — preserving disk state and avoiding re-setup costs on resume.
# Query each tier and use the first stoppable result before falling back to non-stoppable.
#
# Tier 1 — B200 stoppable (native fp4, ~5× faster than A100)
brev search --gpu-name B200 --sort price --stoppable
#
# Tier 2 — H200 stoppable (3.5× faster, recommended default)
brev search --gpu-name H200 --sort price --stoppable
#
# Tier 3 — H100 stoppable (2.5× faster, cheapest total for multi-run sessions)
brev search --gpu-name H100 --sort price --stoppable
#
# Tier 4 — A100 80G stoppable fallback
brev search --gpu-name A100 --min-vram 80 --sort price --stoppable
#
# ── If no stoppable instances found, fall back to non-stoppable ─────────────
brev search --gpu-name B200 --sort price
brev search --gpu-name H200 --sort price
brev search --gpu-name H100 --sort price
brev search --gpu-name A100 --min-vram 80 --sort price

# ── STEP 3: Create instance using chosen type ───────────────────────────────
# Replace <INSTANCE_NAME> with a descriptive name for this recipe deployment.
# Replace <CHOSEN_TYPE> with the first available result from Step 2.
# Add --startup-script @launch.sh if a launch.sh exists for the recipe.
brev create <INSTANCE_NAME> --type <CHOSEN_TYPE>

# Monitor:
brev exec <INSTANCE_NAME> "tail -20 ~/setup.log"

# Port-forward (if recipe serves a UI or API):
brev port-forward <INSTANCE_NAME> -p <REMOTE_PORT>:<LOCAL_PORT>
```

**Minimum inputs from user:** `HF_TOKEN` (once per session). `brev login` must have been run at least once on this machine. `brev` must be installed (see Step 0).

---

## Compute Requirements

1× NVIDIA GPU with 40 GB+ VRAM. Select the best available instance using the priority ladder below.

**Priority order: stoppable first, then availability → inference speed → cost**

Stoppable instances preserve disk state when paused — no re-download of model weights, no re-setup of the environment. For recipes with large model downloads (4–70+ GB), this can save 30–60+ minutes per resumed session.

| Priority | GPU | Example instance | $/hr | vs A100 speed | Stoppable | Notes |
|----------|-----|-----------------|------|---------------|-----------|-------|
| 1 | B200 stoppable | *(query at runtime)* | ~$6.76 | ~5× | Yes | Native fp4; Blackwell arch; check availability |
| 2 | **H200 stoppable** ★ | `gpu-h200-sxm.1gpu-16vcpu-200gb` (nebius) | ~$4.20 | ~3.5× | Yes | **Recommended default.** Widely available; preserves state on stop. |
| 3 | H100 stoppable | *(query at runtime)* | ~$2.28 | ~2.5× | Yes | 34% cheaper than A100 for multi-run sessions. Native fp8 (Hopper). |
| 4 | A100 80G stoppable | *(query at runtime)* | ~$1.49 | 1× | Yes | Stoppable A100 if available. |
| 5 | H200 non-stoppable | `digitalocean_H200_sxm5` | ~$4.13 | ~3.5× | No | Use only if no stoppable H200 found. |
| 6 | H100 non-stoppable | `hyperstack_H100` | ~$2.28 | ~2.5× | No | Use only if no stoppable H100 found. |
| 7 | A100 80G non-stoppable | `massedcompute_A100_sxm4_80G` | ~$1.49 | 1× | No | Last resort. GCP A100 types fail with capacity errors — use massedcompute. |

> **Why stoppable first?** Non-stoppable instances must be deleted to stop billing. All environment setup, model downloads, and installs are lost. For iterative recipe development or multi-session work, a stoppable instance at a slightly higher rate is almost always cheaper in total.

> **Why H200 over A100?** The A100's $1.49/hr looks cheaper, but a 3-run experiment session takes ~4.6 hours ($6.85 total). The same session on H200 at $4.20/hr completes in ~1.6 hours ($6.72) — same total cost at 3.5× the speed.

**Instance user:** depends on provider. massedcompute/shadeform instances use `shadeform`; nebius uses `user`. Use `$HOME` throughout — do not hardcode `/home/ubuntu` or any other username.

---

## General Setup Pattern (any recipe)

These steps apply to all Cosmos Reason 2 inference recipes. Recipe-specific steps (model name, script paths, dependencies) come from the recipe's own documentation.

### System dependencies

```bash
sudo apt-get update -y
sudo apt-get install -y curl ffmpeg git git-lfs
git lfs install
```

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### HuggingFace authentication

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

Do NOT run `huggingface-cli login` — it is interactive and will hang. The `HF_TOKEN` environment variable is sufficient for all model downloads.

### Clone cosmos-reason2

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-reason2.git $HOME/cosmos-reason2
```

`GIT_LFS_SKIP_SMUDGE=1` is required — the public LFS quota on cosmos-reason2 is frequently exhausted. Model weights are downloaded separately via HuggingFace.

### Clone cosmos-cookbook

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git $HOME/cosmos-cookbook
```

### Create Python environment

```bash
cd $HOME/cosmos-reason2
uv sync --extra cu128
```

Do NOT run `source .venv/bin/activate`. The activated venv does NOT persist across `brev exec` subshell calls. Always invoke Python as:

```bash
$HOME/cosmos-reason2/.venv/bin/python3
```

Or use `uv run` from within the cosmos-reason2 directory.

### Install additional dependencies

```bash
cd $HOME/cosmos-reason2
uv pip install <recipe-specific-packages>
```

Do NOT use `pip install` — uv venvs have no pip binary. Always use `uv pip install`.

### Download model weights

```bash
HF_TOKEN=$HF_TOKEN $HOME/cosmos-reason2/.venv/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/<MODEL_NAME>',
    local_dir='$HOME/cosmos-reason2/models/<MODEL_NAME>',
    token='$HF_TOKEN'
)
print('Model download complete')
"
```

---

## Performance Optimization

**TORCHDYNAMO_DISABLE=1 is required for nvfp4 and fp8 quantized checkpoints.** Without it, `torch._inductor` triggers JIT kernel compilation on first forward pass — 16 compile_worker processes, GPU at <40% utilization, stalled for 4+ hours. Add this before any torch import:

```python
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # must be before any torch import
import torch
import transformers
```

Reference timings (A100 SXM4 80G baseline; H200/H100 scale by multiplier in instance table):

| Configuration | max_new_tokens | Est. time (40 videos) on A100 | Est. time on H200 |
|---------------|---------------|-------------------------------|-------------------|
| Baseline float16 | 1024 | ~83 min | ~24 min |
| Optimized float16 | 512 | ~50–60 min | ~15–17 min |
| nvfp4 quantized | 512 | ~25–35 min | ~7–10 min |
| fp8 quantized | 512 | ~25–35 min | ~7–10 min |

**To quantize before inference:**
```bash
cd $HOME/cosmos-reason2
$HOME/cosmos-reason2/.venv/bin/python3 scripts/quantize.py \
  --model $HOME/cosmos-reason2/models/<MODEL_NAME> \
  -o $HOME/cosmos-reason2/models/quantized
```

Quantization takes ~25 min one-time. Use the quantized model for all subsequent runs.

---

## Known Issues

| # | Symptom | Fix |
|---|---------|-----|
| 1 | `brev create` fails with capacity error | Run the instance selection ladder in Step 2 — query stoppable tiers first, then fall back to non-stoppable |
| 2 | GCP A100 types fail (`a2-ultragpu`, `a2-highgpu`) | Use `massedcompute_A100_sxm4_80G` for A100 — GCP A100 quota is frequently exhausted |
| 3 | `pip: command not found` | Use `uv pip install` instead of `pip install` |
| 4 | `source .venv/bin/activate` has no effect in `brev exec` | Use full path: `$HOME/cosmos-reason2/.venv/bin/python3` |
| 5 | `GIT_LFS_SKIP_SMUDGE` not set → clone fails/hangs | Set `GIT_LFS_SKIP_SMUDGE=1` before every git clone |
| 6 | `huggingface-cli login` hangs | Use `export HF_TOKEN=hf_...` — no interactive login needed |
| 7 | `session.wait()` blocks headless execution | Use a headless runner that patches `fo.launch_app` to a no-op |
| 8 | FiftyOne port unreachable | Bind with `address="0.0.0.0"`, then `brev port-forward -p 5151:5151` |
| 9 | Wrong org billed | Always run `brev org set asotelo-test-org` before `brev create` |
| 10 | Instance user is not `ubuntu` | Use `$HOME` everywhere; massedcompute/shadeform uses `shadeform`, nebius uses `user` |
| 11 | `torch._inductor` stalls nvfp4/fp8 inference for 4+ hours | Set `os.environ["TORCHDYNAMO_DISABLE"] = "1"` as the **first line** before any torch import |
| 12 | Non-stoppable instance deleted, setup lost | Prefer stoppable instances (FEATURES "S") — see GPU selection ladder |

---

## Success Criteria (general)

| Check | Command | Expected |
|-------|---------|----------|
| GPU visible | `nvidia-smi` | GPU shown with expected VRAM |
| CUDA available | `$HOME/cosmos-reason2/.venv/bin/python3 -c "import torch; print(torch.cuda.is_available())"` | `True` |
| Model loaded | `ls $HOME/cosmos-reason2/models/` | Model directory present |

Recipe-specific success checks are defined in each recipe's documentation.

---

## Cosmos Metadata

| Field | Value |
|-------|-------|
| Workload | inference |
| Domain | general |
| Technique | technique:reasoning |
| Tags | inference, reason-2, brev |
| Summary | Reusable Brev CLI infrastructure guide for deploying Cosmos Reason 2 inference recipes on remote GPU instances. |
