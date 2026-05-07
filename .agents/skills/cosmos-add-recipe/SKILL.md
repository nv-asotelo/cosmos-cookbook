---
name: cosmos-add-recipe
description: Scaffold a new Cosmos Cookbook recipe from templates, including docs, SUMMARY.md, assets, and matching AGENTS.md plus CLAUDE.md agent guidance files with validation.
compatibility: Codex, Claude Code, and Kimi Code CLI.
---

Scaffold a new Cosmos Cookbook recipe from the official templates.

$ARGUMENTS format: "<recipe-name> <type>" where type is one of: inference, post-training, data-curation, end2end

Steps:
1. Parse $ARGUMENTS. If not provided or incomplete, ask the user for:
   a. Recipe slug (e.g., "my-robot-safety-detector") — used as directory name
   b. Type: inference / post-training / data-curation / end2end
   c. Model(s) used: e.g., "Cosmos Reason 2" (determines parent directory)
   d. Domain: robotics / autonomous-vehicles / vision-ai / industrial / medical / other
   e. One-sentence description of what the recipe does

2. Determine parent directory based on model and type:
   - inference + Predict 2 → docs/recipes/inference/predict2/
   - inference + Reason 2 → docs/recipes/inference/reason2/
   - inference + Transfer 1 → docs/recipes/inference/transfer1/
   - inference + Transfer 2.5 → docs/recipes/inference/transfer2_5/
   - post-training + Predict 2 → docs/recipes/post_training/predict2/
   - post-training + Predict 2.5 → docs/recipes/post_training/predict2_5/
   - post-training + Reason 1 → docs/recipes/post_training/reason1/
   - post-training + Reason 2 → docs/recipes/post_training/reason2/
   - post-training + Transfer 2.5 → docs/recipes/post_training/transfer2_5/
   - data-curation → docs/recipes/data_curation/
   - end2end → docs/recipes/end2end/

3. Create the recipe directory.

4. Copy the appropriate recipe template:
   - inference/data-curation: copy assets/templates/inference_template.md → <dir>/inference.md (or data_curation.md)
   - post-training: copy assets/templates/post_training_template.md → <dir>/post_training.md
   - end2end: copy assets/templates/inference_template.md → <dir>/workflow_e2e.md

5. Create both AGENTS.md and CLAUDE.md using the appropriate agent-guidance templates:
   - inference/data-curation: assets/templates/agents_md_inference_template.md → <dir>/AGENTS.md and assets/templates/claude_md_inference_template.md → <dir>/CLAUDE.md
   - post-training/end2end: assets/templates/agents_md_post_training_template.md → <dir>/AGENTS.md and assets/templates/claude_md_post_training_template.md → <dir>/CLAUDE.md
   Pre-fill "## What This Recipe Does" with the one-sentence description the user provided. Keep AGENTS.md and CLAUDE.md behaviorally identical.

6. Create SUMMARY.md with: "# <Recipe Name>\n\n<one-sentence description>"

7. Create an assets/ subdirectory.

8. Validate the new agent guidance files: run `python .github/scripts/validate_claude_md.py <dir>/AGENTS.md <dir>/CLAUDE.md`. Fix any validation errors before returning.

9. Tell the user:
   "Recipe scaffolded at <path>. Next steps:
   1. Fill in <recipe>.md with your full recipe documentation.
   2. Fill in AGENTS.md and CLAUDE.md with real commands, compute requirements, and gotchas.
   3. Add your scripts to the scripts/examples/ directory if applicable.
   4. Run: python .github/scripts/validate_claude_md.py <dir>/AGENTS.md <dir>/CLAUDE.md
   5. Submit a PR following CONTRIBUTING.md"

## Notes

**AGENTS.md and CLAUDE.md are required for scaffolded recipes** — the validator (step 8) enforces the shared guidance schema. However, `/cosmos-run-recipe` soft-falls back to inference.md/README.md if guidance files are absent, so existing recipes without them still run. New recipes scaffolded here always get both files from templates.

**CUDA extras:** recipes that use vLLM should use `uv sync --extra cu128`. The setup script auto-downgrades vLLM to 0.11.0 on CUDA 12.8 (driver < 575.x). Do not reference `cu128_torch28` — that extra is not defined at the top-level repo.
