Deploy a Cosmos Cookbook recipe to a Brev GPU instance using the 3-file framework.

## Usage
/cosmos-brev-deploy <recipe-path> [provider]

Examples:
  /cosmos-brev-deploy reason2/worker_safety
  /cosmos-brev-deploy transfer1/inference-its-weather-augmentation nebius
  /cosmos-brev-deploy post_training/reason1/spatial-ai-warehouse hyperstack

## What This Does
Guides through launching any Cosmos recipe on a Brev-provisioned GPU. Handles all 6
runtime bugs documented in the April 2026 Brev GPU Sprint (5 H100s, 0 completions —
all environment setup failures, not model failures).

## Steps

### 0. PRE-PROCESS — gather before first Brev command

Ask these in a single message block. Do not start Brev work until all are answered.

```
1. Recipe: cosmos3-reasoner-2b | cosmos3-reasoner-8b | <other recipe path>
2. Model size: 2B [default] | 8B [needs ~80 GB H100]
3. Backend: hf-transformers [default] | vllm [H100 only, sub-second TTFT]
4. Provider: massedcompute_H100 [default] | hyperstack_H100 | nebius
5. Org: [confirm — default asotelo-test-org]
```

**HF_TOKEN:** Use `brev env set HF_TOKEN=hf_...` — stored as a Brev secret, not in command history. Never pass as command-line arg. For Cosmos3-Reasoner: requires nvidia org membership (confirm `hf auth whoami` shows `orgs: nvidia`). NGC not required.

### 1. Pre-flight Gate (Non-negotiable)

Run these checks before touching Brev:

- Is `HF_TOKEN` set via brev env or `echo $HF_TOKEN`? Required for all gated nvidia/ models.
- Is `COOKBOOK` set? (`echo $COOKBOOK` — must point to cosmos-cookbook repo root)
- Does `deploy/<category>/<recipe>/demo.sh` exist? Check with `ls deploy/`.
- Does `deploy/<category>/<recipe>/brev.yaml` exist? Read it for hardware spec.
- Has a smoke test been run on this recipe at least once on real hardware?

If any check fails, explain what's missing before proceeding.

### 2. Read the Recipe's brev.yaml

Read `deploy/<category>/<recipe>/brev.yaml` and report:
- GPU type and count
- Disk requirement
- Estimated cost (if listed)
- Region recommendation

### 3. Check Org Context

**CRITICAL:** Always confirm the user has set their Brev org before creating instances.

```bash
# Show active org and list existing instances — confirm you're in the right org
brev ls

# Switch org if needed (e.g., to the safe test org)
brev set-org asotelo-test-org
```

Default safe org for testing: `asotelo-test-org`

Do NOT create instances in a production org without the user explicitly naming the org.
If the user hasn't specified which org, ask before creating.

### 4. Create the Brev Instance

```bash
brev create <instance-name> --file deploy/<category>/<recipe>/brev.yaml
```

Wait for the instance to show as Running:
```bash
brev ls
```

### 5. Provider-Specific Notes

**Nebius H100 SXM:**
- User: `ubuntu`, HOME: `/home/ubuntu`
- `brev stop` works correctly — pauses billing
- git-lfs NOT pre-installed (brev-env.sh handles this)
- pip blocked by PEP 668 (brev-env.sh handles this)

**Hyperstack H100/A100:**
- User: `ubuntu`, HOME: `/home/shadeform`
- **`brev stop` is a NO-OP — VM keeps running and billing**
- Use `brev delete` to actually terminate and stop charges
- git-lfs NOT pre-installed (brev-env.sh handles this)
- OS disk is ~97GB — too small for multi-model byo-video use (FP8+BF16+NVFP4). Use MassedCompute instead.

**MassedCompute H100 (byo-video only):**
- Use `--type massedcompute_H100` when provisioning for byo-video with multiple model variants
- Provides ~1TB disk (vs Hyperstack ~97GB) — required for FP8 + NVFP4 + BF16 side-by-side
- SSH username: confirm with `brev ls` (typically `ubuntu`)
- `brev stop` behavior: test before relying on it — MassedCompute policies differ from Hyperstack

### 6. Deploy the Recipe

```bash
brev exec <instance-name> -- bash -c "
  export HF_TOKEN='$HF_TOKEN'
  export COOKBOOK=/workspace/cosmos-cookbook
  git clone https://github.com/nvidia-cosmos/cosmos-cookbook \$COOKBOOK
  bash \$COOKBOOK/deploy/<category>/<recipe>/demo.sh
"
```

The demo.sh sources `deploy/shared/brev-env.sh` which handles all environment setup.

### 7. The 6 Brev Bugs — Already Fixed by brev-env.sh

Every `demo.sh` sources `deploy/shared/brev-env.sh` which patches these in order:

| # | Bug | Symptom | Root Cause |
|---|-----|---------|------------|
| 1 | Wrong HOME | `mkdir: cannot create '/root': Permission denied` | `brev exec` passes LOCAL machine's HOME |
| 2 | Empty /workspace/ | `cd /workspace/cosmos-reason2 — not found` | `brev.yaml setup:` block does NOT run on bare creates |
| 3 | Missing git-lfs | `git: 'lfs' is not a git command` | Bare Nebius/Hyperstack Ubuntu images lack git-lfs |
| 4 | HF auth before venv | `huggingface-cli: command not found` | CLI is inside the uv venv, not system-wide |
| 5 | pip blocked PEP 668 | `error: externally-managed-environment` | Ubuntu 24.04 blocks bare `pip install` in uv venv |
| 6 | uv missing after HOME fix | `uv: command not found` | uv installed to wrong HOME path |

**You do NOT need to fix these manually.** Sourcing `brev-env.sh` resolves all 6.

### 8. Monitor Progress

**Option A — Tail the output:**
```bash
brev exec <instance-name> -- tail -f /tmp/<recipe>_output.json
```

**Option B — Watch GPU:**
```bash
brev exec <instance-name> -- watch -n 5 nvidia-smi
```

**For overnight runs, run the crash monitor:**
```bash
bash deploy/shared/brev-monitor.sh <instance-name> [--provider hyperstack]
```

### 9. Retrieve Results

```bash
# Pull output files
brev scp <instance-name>:/tmp/<recipe>_output.json ./results/

# For post-training — pull checkpoints
brev scp <instance-name>:/workspace/cosmos-reason2/checkpoints/ ./checkpoints/ -r
```

### 10. Teardown

```bash
# Nebius — stop to pause billing
brev stop <instance-name>

# Hyperstack — MUST use delete (stop is a no-op on Hyperstack)
brev delete <instance-name>
```

## The 3-File Framework

Every recipe deployment has exactly 3 files:

```
deploy/
  shared/
    brev-env.sh          # Universal bootstrap — fixes all 6 bugs. ONE file for all recipes.
  <category>/<recipe>/
    demo.sh              # Recipe logic only — no environment code
    brev.yaml            # Hardware spec only — no logic
```

**To add Brev support to a new recipe:**
1. `demo.sh`: Set 4-5 env vars, source `brev-env.sh`, add recipe-specific commands
2. `brev.yaml`: Specify GPU type, count, disk size
3. Never put environment setup in demo.sh — all of that lives in brev-env.sh

## Troubleshooting

**Instance stuck at "Creating":**
- Check Brev dashboard for quota/capacity issues on selected GPU type
- Try a different region: Nebius has better H100 availability than Hyperstack usually

**`command not found` after brev-env.sh:**
- The venv activated but a command is missing — check COSMOS_EXTRA_DEPS in demo.sh
- Re-source: `source $BREV_COSMOS_DIR/.venv/bin/activate`

**Model download fails:**
- Confirm `HF_TOKEN` is set AND the model license was accepted on huggingface.co/nvidia
- For Cosmos3-Reasoner private models: confirm `hf auth whoami` shows `orgs: nvidia` — model is gated to nvidia org members only. NGC is NOT required.
- NGC auth is only required for NIM endpoint mode (not standard inference).

## Cosmos3-Reasoner on Brev (quick reference)

```bash
# 1. Set HF token as Brev secret (not in command history)
brev env set HF_TOKEN=hf_...

# 2. Create massedcompute_H100 instance
#    2B: ~60GB disk. 8B: ~100GB with cache. 32B: 1TB minimum (--disk 1024 or brev.yaml disk: 1024)
brev create cosmos3-reasoner-demo --type massedcompute_H100

# 3. Bootstrap and launch (choose MODEL_SIZE)
brev exec cosmos3-reasoner-demo "MODEL_SIZE=C3-2B python3 /tmp/byo_video_setup.py"
brev exec cosmos3-reasoner-demo "MODEL_SIZE=C3-8B python3 /tmp/byo_video_setup.py"
brev exec cosmos3-reasoner-demo "MODEL_SIZE=C3-32B python3 /tmp/byo_video_setup.py"
```

**C3-32B vLLM flags (single H100 80GB):**
```bash
VLLM_VIDEO_LOADER_BACKEND=opencv \
SKIP_HF_PRELOAD=1 \
nohup ~/cosmos-reason2/.venv/bin/vllm serve ~/cosmos-reason2/models/Cosmos3-Reasoner-32B \
  --served-model-name nvidia/Cosmos3-Reasoner-32B-Private \
  --port 8000 \
  --dtype auto \
  --trust-remote-code \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.93 \
  > /tmp/vllm.log 2>&1 &
```
Weight size: ~60-70GB BF16 (25 safetensor files). If OOM occurs, reduce `--max-model-len` or use multi-GPU.

**vLLM flags required for video (apply to any Cosmos/Nemotron VLM on Brev):**
```bash
VLLM_VIDEO_LOADER_BACKEND=opencv \
SKIP_HF_PRELOAD=1 \
vllm serve <model-path> \
  --max-model-len 32768 \    # MAXLEN-001: 8192 default breaks video queries
  --trust-remote-code \
  --gpu-memory-utilization 0.85
```

**Out of disk during model download:**
- Most inference recipes: need 20–80GB; post-training: 100–600GB
- Check remaining: `df -h /workspace`
- Use a larger disk in brev.yaml: `disk: 500`
