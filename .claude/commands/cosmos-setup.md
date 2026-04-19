Validate the environment for running Cosmos Cookbook recipes and set up any missing dependencies.

Steps:
1. Check NVIDIA GPU: run `nvidia-smi` and report GPU model, VRAM, driver version, and CUDA version. If no GPU is found, warn the user that GPU recipes cannot run locally.
2. Check Python version: run `python3 --version` and verify it is 3.10 or higher.
3. Check Docker: run `docker --version` to confirm Docker is available (required for several post-training recipes).
4. Check HuggingFace login: run `huggingface-cli whoami`. If not logged in, instruct the user: "Run `huggingface-cli login` and paste your HF_TOKEN. You also need to accept the NVIDIA Open Model License at https://huggingface.co/nvidia for gated models."
5. Check NGC API key: run `echo $NGC_API_KEY`. If empty, tell the user: "Set your NGC API key: export NGC_API_KEY=<your-key>. Get one at https://org.ngc.nvidia.com/setup/api-keys"
6. Check disk space: run `df -h /` and warn if less than 100GB free (post-training recipes need 100–600GB).
7. Check uv: run `uv --version`. If missing, run: `curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env`
8. Check just: run `just --version`. If missing, run: `uv tool install -U rust-just`
9. Check git-lfs: run `git lfs version`. If missing, run: `sudo apt-get install -y git-lfs && git lfs install`
10. Check ffmpeg: run `ffmpeg -version 2>/dev/null | head -1`. If missing, run: `sudo apt-get install -y ffmpeg`
11. Report a summary table: GPU, Python, Docker, HuggingFace, NGC, Disk, uv, just, git-lfs, ffmpeg — each ✓ or ✗ with version.
12. If everything checks out, print: "Environment ready. Use /cosmos-run-recipe <recipe-name> to execute a recipe."

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
| HF auth before venv | `huggingface-cli: command not found` | Run HF login AFTER `source .venv/bin/activate` |
| pip blocked PEP 668 | `error: externally-managed-environment` | Always use `uv pip install`, never bare `pip` |
| uv wrong path | `uv: command not found` after install | Fix HOME first, then install uv |

Use `/cosmos-brev-deploy <recipe>` to deploy any recipe with all bugs handled automatically.
