# CLAUDE.md — Worker Safety · Cosmos Reason 2 Inference

## Quick Start for Agents

```bash
# ── PRE-FLIGHT (human must do these once) ──────────────────────────────────
# 1. brev login                            ← browser OAuth, one-time per machine
# 2. Accept NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-2B
# 3. Set HF_TOKEN in your shell: export HF_TOKEN=hf_...

# ── AGENT EXECUTION ────────────────────────────────────────────────────────
brev org set asotelo-test-org              # ALWAYS set org first — wrong org = wrong bill

brev create worker-safety \
  --type massedcompute_A100_sxm4_80G_DGX \
  --startup-script @launch.sh             # launch.sh must be in the same directory

# Monitor (poll until complete):
brev exec worker-safety "tail -20 ~/inference.log"

# Port-forward FiftyOne to localhost:
brev port-forward worker-safety -p 5151:5151
# Then open: http://localhost:5151
```

**Minimum inputs from user:** `HF_TOKEN` (once per session). `brev login` must have been run at least once on this machine.

---

## Model

`nvidia/Cosmos-Reason2-2B`

- **Access:** Gated — accept NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-2B
- **Size:** ~4.6 GB
- **License:** NVIDIA Open Model License
- **Auth:** `export HF_TOKEN=hf_...` — do NOT use `huggingface-cli login` (interactive)

## Compute Requirements

1× NVIDIA GPU with 40 GB+ VRAM. Tested: `massedcompute_A100_sxm4_80G_DGX` (A100 80 GB SXM4).

**Verified working instance type:** `massedcompute_A100_sxm4_80G_DGX`
GCP A100 types (e.g. `g2-standard-96`) fail with capacity errors. Use massedcompute.

**Instance user:** `shadeform` (not `ubuntu`). Use `$HOME` throughout — do not hardcode `/home/ubuntu`.

---

## Step-by-Step Execution

### Step 1 — Select org

```bash
brev org set asotelo-test-org
```

`asotelo-test-org` is the correct personal org. **Never skip this step.** Brev CLI org context resets between calls — set it before every `brev create`.

### Step 2 — System dependencies

```bash
sudo apt-get update -y
sudo apt-get install -y curl ffmpeg git git-lfs
git lfs install
```

### Step 3 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Step 4 — HuggingFace authentication

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

Do NOT run `huggingface-cli login` — it is interactive and will hang. The `HF_TOKEN` environment variable is sufficient for model downloads.

### Step 5 — Clone cosmos-reason2

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-reason2.git $HOME/cosmos-reason2
cd $HOME/cosmos-reason2
```

`GIT_LFS_SKIP_SMUDGE=1` is required — the public LFS quota on cosmos-reason2 is frequently exhausted. Skip LFS on clone; model weights are downloaded separately via HuggingFace in Step 7.

### Step 6 — Clone cosmos-cookbook

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git $HOME/cosmos-cookbook
```

### Step 7 — Create Python environment

```bash
cd $HOME/cosmos-reason2
uv sync --extra cu128
```

Do NOT run `source .venv/bin/activate`. The activated venv does NOT persist across `brev exec` subshell calls. Always invoke Python as:

```bash
$HOME/cosmos-reason2/.venv/bin/python3
```

Or use `uv run` from within the cosmos-reason2 directory.

### Step 8 — Install recipe dependencies

```bash
cd $HOME/cosmos-reason2
uv pip install fiftyone
```

Do NOT use `pip install` — uv venvs have no pip binary. Always use `uv pip install`.

### Step 9 — Download model weights

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

### Step 10 — Copy recipe files

```bash
cp $HOME/cosmos-cookbook/docs/recipes/inference/reason2/worker_safety/worker_safety.py \
   $HOME/cosmos-reason2/worker_safety.py
cp -r $HOME/cosmos-cookbook/docs/recipes/inference/reason2/worker_safety/assets \
   $HOME/cosmos-reason2/assets
cp $HOME/cosmos-cookbook/docs/recipes/inference/reason2/worker_safety/run_headless.py \
   $HOME/cosmos-reason2/run_headless.py
```

### Step 11 — Run inference headlessly

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

### Step 12 — Verify results

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
| 1 | `brev create` fails with capacity error | Use `massedcompute_A100_sxm4_80G_DGX` — GCP A100 types fail |
| 2 | `pip: command not found` | Use `uv pip install` instead of `pip install` |
| 3 | `source .venv/bin/activate` has no effect in `brev exec` | Use full path: `$HOME/cosmos-reason2/.venv/bin/python3` |
| 4 | `GIT_LFS_SKIP_SMUDGE` not set → clone fails/hangs | Set `GIT_LFS_SKIP_SMUDGE=1` before every git clone |
| 5 | `huggingface-cli login` hangs | Use `export HF_TOKEN=hf_...` — no interactive login needed |
| 6 | `DatasetNotFoundError: safe-unsafe-worker-behavior` | Use `fo.list_datasets()[0]` — dataset name is the HF slug |
| 7 | `session.wait()` blocks headless execution | Use `run_headless.py` — patches `fo.launch_app` to no-op |
| 8 | FiftyOne port unreachable | Bind with `address="0.0.0.0"`, then `brev port-forward -p 5151:5151` |
| 9 | Wrong org billed | Always run `brev org set asotelo-test-org` before `brev create` |
| 10 | Instance user is `shadeform` not `ubuntu` | Use `$HOME` everywhere; never hardcode `/home/ubuntu` |
| 11 | Recipe ends in Python REPL with `session.wait()` | Use `run_headless.py` — avoids interactive shell requirement |

---

## Performance Optimization

| Configuration | max_new_tokens | fps | Est. inference time (40 videos) |
|---------------|---------------|-----|----------------------------------|
| Baseline float16 | 1024 | 4 | ~83 min |
| Optimized float16 | 512 | 2 | ~25–35 min (est.) |
| nvfp4 quantized | 512 | 2 | ~15–25 min (est.) |
| fp8 quantized | 512 | 2 | ~15–25 min (est.) |

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
| GPU visible | `nvidia-smi` | A100 80 GB shown |
| CUDA available | `python3 -c "import torch; print(torch.cuda.is_available())"` | `True` |
| Inference complete | `cat ~/inference_results.json \| python3 -c "import json,sys; d=json.load(sys.stdin); print(len([r for r in d if r['safety_label']]), 'classified')"` | `40 classified` (or N/40) |
| No errors | Same script, check `cosmos_error` | `0 errors` |
| FiftyOne loads | `python3 -c "import fiftyone as fo; print(fo.list_datasets())"` | Dataset name printed |

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
