---
name: cosmos-setup
description: Validate a Cosmos Cookbook environment before running GPU recipes. Checks GPU, Python, Docker, HuggingFace auth, optional NGC, disk, uv, just, git-lfs, ffmpeg, and Brev-specific prerequisites.
compatibility: Codex, Claude Code, and Kimi Code CLI. Translate AskUserQuestion to the host agent's native question mechanism when needed.
---

Validate the environment for running Cosmos Cookbook recipes and set up any missing dependencies.

Agent compatibility: when this skill says `AskUserQuestion`, use the native
interactive question tool for the current agent. If no such tool is available,
ask a concise direct question and wait for the user before continuing.

Steps:
1. Check NVIDIA GPU: run `nvidia-smi` and report GPU model, VRAM, driver version, and CUDA version. If no GPU is found, warn the user that GPU recipes cannot run locally.
   - Also report CPU architecture: `uname -m`. If aarch64 (ARM64): warn that standard x86_64 PyPI wheels will not install. Verify `python3 -c "import torch; print(torch.cuda.is_available())"` before proceeding with any recipe.
2. Check Python version: run `python3 --version` and verify it is 3.10 or higher.
3. Check Docker: run `docker --version` to confirm Docker is available (required for several post-training recipes).
4. Check HuggingFace token: run `hf auth whoami` first.
   - If logged in: confirm `orgs: nvidia` is present for Cosmos3-Reasoner private models. If the nvidia org is missing, warn — model downloads will fail with 403.
   - If not logged in: use the `AskUserQuestion` tool with this prompt:
     > "HuggingFace authentication required.
     > Run in your terminal: **`hf auth login`**
     > Paste your HF token when prompted (no browser). Token is stored in `~/.cache/huggingface/token`.
     > Type **done** when complete."
     After the user replies, re-run `hf auth whoami`. If still not logged in, use `AskUserQuestion` once more. After three failures, halt.
   - Do NOT ask for the token as a CLI argument — it ends up in shell history.
   - For Cosmos3-Reasoner models: verify `hf auth whoami` shows `orgs: nvidia` before any model download.
5. Check NGC API key: NGC is only required for NIM endpoint mode. For all other Cosmos recipes (including Cosmos3-Reasoner), NGC is optional. If `echo $NGC_API_KEY` is empty, note it's not needed for standard inference. Only warn if the user explicitly wants NIM mode.
6. Check disk space: run `df -h /` and warn if less than 100GB free (post-training recipes need 100–600GB).
7. Check uv: run `uv --version`. If missing, run: `curl -LsSf https://astral.sh/uv/install.sh | sh` then reload PATH with `export PATH="$HOME/.local/bin:$PATH"`. Note: `source $HOME/.local/bin/env` only works in interactive shells — use the export form in non-interactive/SSH sessions.
8. Check just: run `just --version`. If missing, run: `uv tool install -U rust-just`
9. Check git-lfs: run `git lfs version`. If missing, run: `sudo apt-get install -y git-lfs && git lfs install`
10. Check ffmpeg: run `ffmpeg -version 2>/dev/null | head -1`. If missing, run: `sudo apt-get install -y ffmpeg`. If apt-get fails or ffmpeg is still not in PATH (common on Hyperstack/snap-only environments), try: `sudo snap install ffmpeg && sudo ln -s /snap/bin/ffmpeg /usr/local/bin/ffmpeg`.
    - On Horde: `sudo cp /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe` (if amd64 static binary pre-downloaded).
11. Report a summary table: GPU, Arch, Python, Docker, HuggingFace (with nvidia org), NGC (optional), Disk, uv, just, git-lfs, ffmpeg — each ✓ or ✗ with version.
12. If everything checks out, print: "Environment ready. Use /cosmos-run-recipe <recipe-name> to execute a recipe."

## Model Family Selection

| Family | HF repo prefix | Access | Setup path |
|---|---|---|---|
| Cosmos3-Reasoner | nvidia/Cosmos3-Reasoner-*-Private | 🔒 nvidia org (private) | cosmos-reason2 repo (Qwen3-VL compatible) |
| Cosmos Reason2 | nvidia/Cosmos-Reason2-* | Public | cosmos-reason2 repo |
| Cosmos Transfer | nvidia/Cosmos-Transfer* | Public | cosmos-transfer repo |

Cosmos3-Reasoner uses `MODEL_SIZE=C3-2B`, `MODEL_SIZE=C3-8B`, or `MODEL_SIZE=C3-32B` with `byo_video_setup.py`. The model uses the same HF Transformers API as CR2 (Qwen3-VL family). C3-32B requires 1TB+ disk and a single H100 80GB with `--tensor-parallel-size 1 --gpu-memory-utilization 0.93`.

## Brev Cloud Deployment Check

If the user is on a Brev-provisioned instance (check: `test -d /workspace`), run these additional checks:

**Brev Pre-flight Gate** (must pass before starting any recipe):
1. Confirm `COOKBOOK` is set: `echo $COOKBOOK` (must be the cosmos-cookbook repo root)
2. Confirm `HF_TOKEN` is set: `echo $HF_TOKEN` (empty = model downloads will fail)
3. Check that `deploy/shared/brev-env.sh` exists in `$COOKBOOK`
4. Check Brev org: run `brev ls` — confirm the active org is the intended test org, NOT a production org
5. Check provider: Nebius (`HOME=/home/ubuntu`) vs Hyperstack (`HOME=/home/shadeform`)
   - **Hyperstack critical:** `brev stop` is a NO-OP — always use `brev delete` to avoid runaway billing

**Known Brev Bugs — all fixed by `deploy/shared/brev-env.sh`:**
These bugs caused 5 H100s to run overnight with 0 completions in April 2026. Every demo.sh
sources brev-env.sh which patches all of them automatically — but warn the user if they're
writing a custom script that bypasses brev-env.sh:

| Bug | Symptom | Fix |
|-----|---------|-----|
| Wrong HOME | `mkdir: cannot create '/root': Permission denied` | Auto-detect HOME via `getent passwd $(id -un)` |
| Empty /workspace/ | `cd /workspace/cosmos-X — not found` | Clone in brev-env.sh, not brev.yaml setup: block |
| Missing git-lfs | `git: 'lfs' is not a git command` | `apt-get install git-lfs` before any clone |
| HF auth before venv | `hf: command not found` or `huggingface-cli: command not found` | Run HF login AFTER `source .venv/bin/activate` (or use `uv run hf whoami`) |
| pip blocked PEP 668 | `error: externally-managed-environment` | Always use `uv pip install`, never bare `pip` |
| uv wrong path | `uv: command not found` after install | Fix HOME first, then install uv |

Use `/cosmos-brev-deploy <recipe>` to deploy any recipe with all bugs handled automatically.
