---
name: vlm-race
description: Run the Cosmos VLM Race multi-model comparison workflow when the comparison helper scripts are available. Use for side-by-side Cosmos Reason, Nemotron, and Qwen video inference comparisons.
compatibility: Codex, Claude Code, and Kimi Code CLI. Translate legacy Claude tool names in the runbook to the current agent's native tools.
---

# VLM Race

Use this skill for multi-model video comparison only after verifying the helper
scripts referenced by `references/vlm-race-runbook.md` are available.

## Pre-Flight

1. Resolve the scripts directory using the same order as `byo-video`:
   `$COSMOS_AGENT_SCRIPTS_DIR`, `.agents/skills/byo-video/scripts/`,
   `.claude/scripts/`, then user-level agent script directories.
2. Confirm `gradio_compare_vlm.py` and `compare_vlm_setup.py` exist. They are
   not currently bundled in this repository snapshot.
3. If either script is missing, do not claim the workflow is runnable. Offer to
   use `$byo-video` for a single-model demo or add the missing VLM Race helpers
   before deployment.
4. If both scripts exist, read and follow `references/vlm-race-runbook.md`.

## Agent Harness Adapter

Translate Claude-specific wording in the runbook to the host agent:

- Use native shell/file tools for commands and script deploys.
- Use the host agent's question mechanism for user inputs.
- If a runbook says "Claude Code", read it as "the current coding agent".
- Avoid Teams or external notifications unless the user explicitly requests one.
