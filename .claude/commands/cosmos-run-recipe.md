Run a Cosmos Cookbook recipe end-to-end with full agent assistance.

$ARGUMENTS is the recipe name or path (e.g., "worker-safety", "predict2/cosmos_policy"). Leave blank to choose interactively.

Steps:
1. If no recipe name is given, read the Recipe Index in CLAUDE.md (repo root) and display all available recipes grouped by category. Ask the user to choose one.
2. Map $ARGUMENTS to a recipe directory by searching under docs/recipes/ for a matching directory name. Use fuzzy matching (e.g., "worker-safety" matches "worker_safety", "carla" matches "inference-carla-sdg-augmentation").
3. Read the CLAUDE.md in the matched recipe directory. If no CLAUDE.md exists, say so and suggest running /cosmos-setup first.
4. Check compute requirements from the recipe's CLAUDE.md: run `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` and compare available VRAM to the requirement. If insufficient, say: "This recipe requires [X]GB VRAM but you have [Y]GB. You can provision a matching instance on NVIDIA Brev at https://brev.nvidia.com. Do you have a Brev API token?"
5. Verify all required environment variables listed in the CLAUDE.md are set. For any missing variable, show the user exactly what to export and wait for confirmation before continuing.
6. Run Setup Prerequisites checklist from CLAUDE.md. For each unchecked item, ask the user to confirm it is done or offer to run the setup step automatically.
7. Execute the Entry Points commands from CLAUDE.md one at a time, showing each command before running it and its output afterward.
8. For POST-TRAINING recipes: after launching the training job, print the Monitoring command from CLAUDE.md and return control to the user. Do NOT wait for training to complete.
9. For INFERENCE recipes: run to completion and show the Expected Output described in CLAUDE.md.
10. If any command fails: read the error, check the Gotchas section of CLAUDE.md, attempt a fix, and re-run. If the fix attempt fails, explain the error and ask the user for help.
