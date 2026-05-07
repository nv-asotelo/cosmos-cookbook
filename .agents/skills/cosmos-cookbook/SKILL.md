---
name: cosmos-cookbook
description: Use when discovering, running, authoring, or deploying Cosmos Cookbook recipes across Codex, Claude Code, and Kimi Code CLI. Routes to the focused recipe, setup, Brev, BYO-video, and VLM skills.
compatibility: Codex reads .agents/skills; Claude uses .claude/skills wrappers or .claude/commands; Kimi reads .agents/skills and can invoke with /skill:<name>.
---

# Cosmos Cookbook Skills

Use this as the router for the repository's agent workflows. The canonical skill
sources live in `.agents/skills/`; `.claude/skills/` contains Claude Code
wrappers, and `.claude/commands/` remains as the legacy slash-command surface.

## Invocation

| Agent | Explicit invocation |
|---|---|
| Codex | Mention `$cosmos-setup`, `$cosmos-run-recipe`, `$byo-video`, etc. |
| Claude Code | Use `/cosmos-setup`, `/cosmos-run-recipe`, `/byo-video`, or the project skills. |
| Kimi Code CLI | Use `/skill:cosmos-setup`, `/skill:cosmos-run-recipe`, `/skill:byo-video`, etc. |

## Skill Map

- `cosmos-setup`: Validate GPU, Python, Docker, HuggingFace, NGC, disk, uv,
  just, git-lfs, and ffmpeg before running recipes.
- `cosmos-list-recipes`: Enumerate available recipes and paths.
- `cosmos-run-recipe`: Execute a recipe from its agent guide or docs.
- `cosmos-add-recipe`: Scaffold a new recipe plus matching `AGENTS.md` and
  `CLAUDE.md` agent guides.
- `cosmos-brev-deploy`: Deploy a recipe to a Brev GPU instance.
- `byo-video`: Launch the single-model Cosmos/VLM Gradio demo.
- `vlm-race`: Run the multi-model VLM comparison workflow when its helper
  scripts are available.

## Shared Rules

- Prefer recipe-local `AGENTS.md` first, then `CLAUDE.md`, then `inference.md`,
  `post_training.md`, or `README.md`.
- Treat `CLAUDE.md` and `AGENTS.md` as the same recipe guidance schema. When
  creating or updating one, keep the other behaviorally identical.
- Use the host agent's native question, shell, file, background-task, and
  scheduling tools. If a runbook names a tool that belongs to another agent
  harness, translate it to the closest local equivalent.
- Never ask the user to paste tokens into command arguments. Use environment
  variables, HuggingFace login, Brev secrets, or the target platform's secret
  manager.
- For BYO-video helper scripts, prefer `.agents/skills/byo-video/scripts/`; fall
  back to `.claude/scripts/` only for legacy Claude slash-command sessions.
