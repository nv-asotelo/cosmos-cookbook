Run a Cosmos Cookbook recipe end-to-end with full agent assistance.

$ARGUMENTS is the recipe name or path (e.g., "worker-safety", "predict2/cosmos_policy"). Leave blank to choose interactively.

Steps:
1. If no recipe name is given, read the Recipe Index in CLAUDE.md (repo root) and display all available recipes grouped by category. Ask the user to choose one.
2. Map $ARGUMENTS to a recipe directory by searching under docs/recipes/ for a matching directory name. Use fuzzy matching (e.g., "worker-safety" matches "worker_safety", "carla" matches "inference-carla-sdg-augmentation").
3. Look for CLAUDE.md in the matched recipe directory. If found, use it as the primary guide. If no CLAUDE.md exists, fall back to reading inference.md, post_training.md, or README.md in that directory and proceed from there — do NOT hard-fail or block on the absence of CLAUDE.md.
4. Check compute requirements from the recipe's CLAUDE.md: run `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` and compare available VRAM to the requirement. If insufficient, say: "This recipe requires [X]GB VRAM but you have [Y]GB. You can provision a matching instance on NVIDIA Brev at https://brev.nvidia.com. Do you have a Brev API token?"
5. Verify all required environment variables listed in the CLAUDE.md are set. For any missing variable, show the user exactly what to export and wait for confirmation before continuing.
6. Run Setup Prerequisites checklist from CLAUDE.md. For each unchecked item, ask the user to confirm it is done or offer to run the setup step automatically.
7. Execute the Entry Points commands from CLAUDE.md one at a time, showing each command before running it and its output afterward.
8. For POST-TRAINING recipes: after launching the training job, print the Monitoring command from CLAUDE.md and return control to the user. Do NOT wait for training to complete.
9. For INFERENCE recipes: run to completion and show the Expected Output described in CLAUDE.md.
10. If any command fails: read the error, check the Gotchas section of CLAUDE.md (if present), attempt a fix, and re-run. If the fix attempt fails, explain the error and ask the user for help.

## Environment Notes

**CUDA extras:** Always use `uv sync --extra cu128`. This installs vLLM 0.12.0 + torch 2.9.0. The `byo_video_setup.py` script auto-detects the CUDA driver and downgrades to vLLM 0.11.0 (torch 2.8.0) when the driver is < 575.x (CUDA 12.8, e.g. Hyperstack H100 with driver 570.x). Do not use `cu128_torch28` — that extra is defined in a workspace member only, not at the top-level repo. Use `cu130` only on instances confirmed to have CUDA 13.0+.

**Video backend:** If `torchcodec` fails with "ffmpeg not found" or "codec error", the PyAV backend is the correct fallback. The `gradio_cr2_byo.py` and `smoke_cr2_byo.py` scripts ship with a PyAV monkey-patch that applies automatically — do not remove it.

**SSH tunnel:** When the recipe output includes a localhost URL (e.g., `http://localhost:7860`), the SSH tunnel command (`ssh -L 7860:localhost:7860 ...`) runs on the **user's laptop**, not on the GPU instance. Instruct the user to run it in a local terminal tab.
