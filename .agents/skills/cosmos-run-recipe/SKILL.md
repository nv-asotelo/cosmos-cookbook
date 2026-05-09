---
name: cosmos-run-recipe
description: Run a Cosmos Cookbook recipe end-to-end with agent assistance. Fuzzy-match recipe names, read AGENTS.md or CLAUDE.md guidance, check compute and environment variables, execute entry points, and recover from documented gotchas.
compatibility: Codex, Claude Code, and Kimi Code CLI.
---

Run a Cosmos Cookbook recipe end-to-end with full agent assistance.

$ARGUMENTS is the recipe name or path (e.g., "worker-safety", "predict2/cosmos_policy"). Leave blank to choose interactively.

Steps:
1. If no recipe name is given, read the Recipe Index in AGENTS.md or CLAUDE.md if present, otherwise read docs/recipes/all_recipes.md, and display all available recipes grouped by category. Ask the user to choose one.
2. Map $ARGUMENTS to a recipe directory by searching under docs/recipes/ for a matching directory name. Use fuzzy matching (e.g., "worker-safety" matches "worker_safety", "carla" matches "inference-carla-sdg-augmentation").
3. Look for AGENTS.md, then CLAUDE.md, in the matched recipe directory. If found, use the first one as the primary guide. If neither exists, fall back to reading inference.md, post_training.md, or README.md in that directory and proceed from there — do NOT hard-fail or block on the absence of agent guidance.
4. Check compute requirements from the recipe's agent guide: run `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` and compare available VRAM to the requirement. If insufficient, say: "This recipe requires [X]GB VRAM but you have [Y]GB. You can provision a matching instance on NVIDIA Brev at https://brev.nvidia.com. Do you have a Brev API token?"
5. Verify all required environment variables listed in the agent guide are set. For any missing variable, show the user exactly what to export and wait for confirmation before continuing.
6. Run the Setup Prerequisites checklist from the agent guide. For each unchecked item, ask the user to confirm it is done or offer to run the setup step automatically.
7. Execute the Entry Points commands from the agent guide one at a time, showing each command before running it and its output afterward.
8. For POST-TRAINING recipes: after launching the training job, print the Monitoring command from the agent guide and return control to the user. Do NOT wait for training to complete.
9. For INFERENCE recipes: run to completion and show the Expected Output described in the agent guide.
10. If any command fails: read the error, check the Gotchas section of the agent guide if present, attempt a fix, and re-run. If the fix attempt fails, explain the error and ask the user for help.

## Environment Notes

**CUDA extras:** Always use `uv sync --extra cu128`. This installs vLLM 0.12.0 + torch 2.9.0. The `byo_video_setup.py` script auto-detects the CUDA driver and downgrades to vLLM 0.11.0 (torch 2.8.0) when the driver is < 575.x (CUDA 12.8, e.g. Hyperstack H100 with driver 570.x). Do not use `cu128_torch28` — that extra is defined in a workspace member only, not at the top-level repo. Use `cu130` only on instances confirmed to have CUDA 13.0+.

**Video backend:** If `torchcodec` fails with "ffmpeg not found" or "codec error", the PyAV backend is the correct fallback. The `gradio_cr2_byo.py` and `smoke_cr2_byo.py` scripts ship with a PyAV monkey-patch that applies automatically — do not remove it.

**SSH tunnel:** When the recipe output includes a localhost URL (e.g., `http://localhost:7860`), the SSH tunnel command (`ssh -L 7860:localhost:7860 ...`) runs on the **user's laptop**, not on the GPU instance. Instruct the user to run it in a local terminal tab.

**FiftyOne recipes:** If the recipe guide or docs mention FiftyOne, verify `python -c "import fiftyone as fo; print(fo.__version__)"` from the same Python environment that will run the recipe. If missing, install inside the active venv with `uv pip install -U fiftyone` for uv-managed environments, or `python -m pip install -U fiftyone` when pip is available. For headless GPU instances, prefer explicit app settings such as `FIFTYONE_ADDRESS=0.0.0.0`, `FIFTYONE_PORT=5151`, and recipe-specific `*_FIFTYONE_WAIT=0` flags so smoke tests complete instead of blocking on `session.wait()`. If the user wants to view the app, give them a local SSH tunnel command for the chosen port.

**Cosmos3-Reasoner recipes:** Use `MODEL_SIZE=C3-2B` (default) or `MODEL_SIZE=C3-8B`. Both use the `cosmos-reason2` working directory (Qwen3-VL compatible). HF_TOKEN with nvidia org access required — check `hf auth whoami` confirms `orgs: nvidia`. NGC not required.

**aarch64 instances (Horde 2026-04-27 pool):** Verify `python3 -c "import torch; print(torch.cuda.is_available())"` before running any recipe. If False, the CUDA Python environment needs to be set up for ARM64+CUDA first — do not proceed until this passes.
