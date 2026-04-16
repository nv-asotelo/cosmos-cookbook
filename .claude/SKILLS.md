# Cosmos Cookbook Skills

Five slash commands for running, deploying, and authoring Cosmos recipes — usable by both humans in Claude Code and autonomous agents.

## Quick Reference

| Command | What It Does | When to Use |
|---------|-------------|-------------|
| `/cosmos-setup` | Validate environment, check GPU, install deps | First thing, every session |
| `/cosmos-list-recipes` | Show all 35 recipes organized by domain | Discovering what's available |
| `/cosmos-run-recipe <name>` | Execute a recipe end-to-end with guidance | Running inference or post-training |
| `/cosmos-add-recipe <name> <type>` | Scaffold a new recipe from templates | Creating a new recipe |
| `/cosmos-brev-deploy <path>` | Deploy any recipe to a Brev GPU instance | Cloud execution |

---

## `/cosmos-setup`

**Purpose:** Validate the local environment before running any recipe. Checks GPU, Python, Docker, HuggingFace login, NGC key, disk space, and tool availability.

**Human usage:**
```
/cosmos-setup
```
Run this at the start of every session. Claude will check all dependencies and report a summary table. If something is missing, it will offer to install it.

**Agent usage:** Invoke at session start before any `/cosmos-run-recipe` call. If the GPU check fails, pivot to `/cosmos-brev-deploy`.

**Key output:**
```
GPU         ✓  H100 80GB, driver 550.54, CUDA 12.4
Python      ✓  3.10.14
Docker      ✓  28.1.0
HuggingFace ✓  logged in as <user>
NGC         ✓  NGC_API_KEY set
Disk        ✓  450GB free
uv          ✓  0.5.3
git-lfs     ✓  3.5.1
ffmpeg      ✓  7.0

Environment ready. Use /cosmos-run-recipe <recipe-name> to execute a recipe.
```

**Brev pre-flight:** When running on a Brev instance (`/workspace` exists), `cosmos-setup` runs additional checks including org verification, provider detection (Nebius vs Hyperstack), and brev-env.sh availability. **Hyperstack critical:** `brev stop` is a NO-OP — always use `brev delete`.

---

## `/cosmos-list-recipes`

**Purpose:** Display all 35 available recipes organized by domain and model family. Scans both the canonical `all_recipes.md` and the live directory tree to catch any additions.

**Human usage:**
```
/cosmos-list-recipes
```
Shows all recipes grouped by category (Inference, Post-Training, Data Curation, End-to-End) and domain (Robotics, AV, Vision AI, Industrial, etc.), with directory paths.

**Agent usage:** Call to enumerate valid recipe targets before calling `/cosmos-run-recipe`. Parse the output to match a user's domain intent to a specific recipe path.

**Output format:**
```
Inference Recipes (13 total)
  Cosmos Predict 2
    ITS Image Synthesis     docs/recipes/inference/predict2/inference-its/
  Cosmos Reason 2
    Worker Safety           docs/recipes/inference/reason2/worker_safety/
    ...

Total: 35 recipes. Run /cosmos-run-recipe <recipe-name> to execute.
```

---

## `/cosmos-run-recipe`

**Purpose:** Execute a Cosmos recipe end-to-end with full Claude guidance — validates compute, checks env vars, runs commands one at a time, and handles errors using the recipe's Gotchas section.

**Human usage:**
```
/cosmos-run-recipe worker-safety
/cosmos-run-recipe predict2/cosmos_policy
/cosmos-run-recipe                         # interactive chooser
```

**Agent usage:** Pass the recipe slug or partial path. Claude fuzzy-matches to the correct directory (e.g., `"carla"` → `inference-carla-sdg-augmentation`). For post-training recipes, Claude launches the training job and returns control — it does NOT wait for completion.

**Flow:**
1. Fuzzy-match recipe name → directory
2. Read CLAUDE.md in that directory
3. Check GPU VRAM vs compute requirements
4. Verify required env vars — prompt for any missing
5. Walk through Setup Prerequisites checklist
6. Run Entry Points commands one at a time
7. For post-training: print Monitoring command, return control
8. For inference: run to completion, show Expected Output
9. On failure: check Gotchas section → attempt fix → re-run

**If compute is insufficient:** Claude says exactly this — "This recipe requires [X]GB VRAM but you have [Y]GB. You can provision a matching instance on NVIDIA Brev. Do you have a Brev API token?" Then pivots to `/cosmos-brev-deploy`.

---

## `/cosmos-add-recipe`

**Purpose:** Scaffold a new recipe with the correct directory structure, templates, and CLAUDE.md — then validate with CI before returning.

**Human usage:**
```
/cosmos-add-recipe my-robot-safety-detector inference
/cosmos-add-recipe                                     # interactive
```

**Agent usage:** Provide recipe slug and type. Claude asks for model, domain, and description, then creates the scaffolding, pre-fills the description, and runs CI validation automatically.

**What gets created:**
```
docs/recipes/inference/reason2/my-robot-safety-detector/
  inference.md       ← full recipe walkthrough (from template)
  CLAUDE.md          ← agent guidance file (from template, pre-filled)
  SUMMARY.md         ← one-line description
  assets/            ← media directory
```

**Templates used:**
- `assets/templates/claude_md_inference_template.md` — for inference and data curation
- `assets/templates/claude_md_post_training_template.md` — for post-training and end2end
- `assets/templates/inference_template.md` — for the human-readable recipe doc

**CI validation runs automatically.** The CI requires:
- `## Data Source` section with `**Access:** Public|Gated|Restricted`
- For `Public`: a `huggingface-cli download`, `wget`, or `curl` command with a reachable URL
- For `Gated`/`Restricted`: passes with a human-review warning

See `RECIPE_AUTHORING.md` for the full checklist before submitting a PR.

---

## `/cosmos-brev-deploy`

**Purpose:** Deploy any Cosmos recipe to a Brev GPU instance using the battle-tested 3-file framework. Handles all 6 known Brev runtime bugs automatically.

**Human usage:**
```
/cosmos-brev-deploy reason2/worker_safety
/cosmos-brev-deploy transfer1/inference-its-weather-augmentation nebius
/cosmos-brev-deploy post_training/reason1/spatial-ai-warehouse hyperstack
```

**Agent usage:** Use when `/cosmos-run-recipe` fails the GPU check. Parse `deploy/<category>/<recipe>/brev.yaml` for hardware specs. Always confirm Brev org before creating instances — default safe org: `asotelo-test-org`.

**The 3-File Framework:**
```
deploy/
  shared/
    brev-env.sh       ← universal bootstrap, shared by all recipes
    brev-monitor.sh   ← overnight crash monitor
  <category>/<recipe>/
    demo.sh           ← recipe logic only
    brev.yaml         ← hardware spec only
```

**6 Brev Bugs Fixed Automatically by brev-env.sh:**

| # | Bug | Symptom |
|---|-----|---------|
| 1 | Wrong HOME | `mkdir: cannot create '/root': Permission denied` |
| 2 | Empty /workspace/ | repo not found on startup |
| 3 | Missing git-lfs | `git: 'lfs' is not a git command` |
| 4 | HF auth before venv | `huggingface-cli: command not found` |
| 5 | pip blocked PEP 668 | `error: externally-managed-environment` |
| 6 | uv wrong path | `uv: command not found` after install |

**Provider differences:**
| | Nebius | Hyperstack |
|---|--------|-----------|
| User | ubuntu | ubuntu |
| HOME | /home/ubuntu | /home/shadeform |
| `brev stop` | Pauses billing | **NO-OP — keeps billing** |
| Teardown | `brev stop` or `brev delete` | **Always `brev delete`** |

**Available deploy configs** (12 inference recipes):

| Recipe | GPU | Disk |
|--------|-----|------|
| predict2/inference-its | A100 80GB × 1 | 80 GB |
| reason2/intbot_showcase | A100 80GB × 1 | 80 GB |
| reason2/vss | A100 80GB × 1 | 80 GB |
| reason2/worker_safety | A100 80GB × 1 | 80 GB |
| transfer1/gr00t-mimic | A100 80GB × 1 | 100 GB |
| transfer1/inference-its-weather-augmentation | A100 80GB × 1 | 80 GB |
| transfer1/inference-warehouse-mv | A100 80GB × 1 | 80 GB |
| transfer1/inference-x-mobility | A100 80GB × 1 | 120 GB |
| transfer2_5/biotrove_augmentation | A100 80GB × 1 | 300 GB |
| transfer2_5/inference-carla-sdg-augmentation | A100 80GB × 1 | 80 GB |
| transfer2_5/inference-image-prompt | A100 80GB × 1 | 80 GB |
| transfer2_5/inference-real-augmentation | A100 80GB × 1 | 80 GB |

Note: `intbot_edge_vlm` requires physical Jetson AGX Thor hardware — no Brev deploy available.

---

## How Agents Discover and Use These Skills

Claude Code automatically reads CLAUDE.md files in the working directory and subdirectories. When a session starts in the cosmos-cookbook repo root:

1. The root `CLAUDE.md` is loaded — it lists all 35 recipes with compute scale indicators (🟢/🟡/🔴)
2. The `.claude/commands/` directory makes all 5 slash commands available
3. When the user navigates to a recipe directory, that recipe's `CLAUDE.md` is loaded with exact entry points

**Agent decision tree for running a recipe:**
```
User wants to run recipe X
  → /cosmos-setup (check local GPU)
    → GPU sufficient? → /cosmos-run-recipe X
    → GPU insufficient? → /cosmos-brev-deploy <X path>
      → No deploy config? → Tell user to add one via RECIPE_AUTHORING.md
```

**Agent safety rules (from CLAUDE.md comments):**
- Always check `## Compute Requirements` before running — ask about Brev if insufficient
- For post-training: launch job and return control — never wait for completion in-context
- On Hyperstack: `brev stop` is a NO-OP — always use `brev delete`
- HF_TOKEN must be set — hard failure if unset, not a warning
