---
name: byo-video
description: Deploy the Cosmos BYO-Video runtime-agent, Gradio, or FiftyOne frontend on Brev, Horde, SSH, or a local GPU. Use for Cosmos Reason/Cosmos3/Nemotron/Qwen VLM video or image inference, HF public dataset batch inference, NIM-local mode, runtime monitoring, and NIM model switching.
compatibility: Codex, Claude Code, and Kimi Code CLI. Translate legacy Claude tool names in the runbook to the current agent's native tools.
---

# BYO-Video

Use this skill to launch Cosmos BYO-Video with a runtime-agent, Gradio, or
FiftyOne-assisted frontend. Read
`references/byo-video-runbook.md` before executing, but apply the compatibility
adapter below first.

Default to `BYO_VIDEO_FRONTEND=runtime_agent` when the user asks for a guided
inference flow, HF public dataset loading, concurrent batch processing, or the
worker-safety smoke test. Use `BYO_VIDEO_FRONTEND=gradio` for the classic
single-upload UI. Use `BYO_VIDEO_FRONTEND=fiftyone` when the user specifically
wants the FiftyOne app available alongside runtime-agent result writeback.

## Agent Harness Adapter

The runbook was originally written for Claude Code. Translate these names to the
current agent harness:

| Runbook term | Codex | Claude Code | Kimi Code CLI |
|---|---|---|---|
| `AskUserQuestion` | `request_user_input` when available, otherwise ask directly | `AskUserQuestion` | `AskUserQuestion` when available, otherwise ask directly |
| `ToolSearch` | `tool_search` if present | `ToolSearch` | built-in tool discovery or proceed with available tools |
| `Agent(...)` observer | `spawn_agent` when allowed, otherwise run phases inline | `Agent(...)` | `Task`/subagent when available, otherwise run phases inline |
| `CronCreate`/`CronDelete` | `automation_update` or manual polling | cron tools | manual polling or configured hooks |
| Bash/shell command | shell execution tool | Bash | Shell |

If the current harness lacks a background observer or cron feature, keep the
workflow correct by running phases inline and polling `/tmp/byo_video_progress.json`
and `/tmp/byo_video_observer_result.json` manually.

## Script Resolution

When the runbook references `~/.claude/scripts/foo.py` or
`/Users/asotelo/.claude/scripts/foo.py`, resolve `foo.py` in this order:

1. `$COSMOS_AGENT_SCRIPTS_DIR/foo.py`, if `COSMOS_AGENT_SCRIPTS_DIR` is set.
2. `.agents/skills/byo-video/scripts/foo.py` from the repository root.
3. `.claude/scripts/foo.py` from the repository root, for legacy Claude sessions.
4. `~/.config/agents/skills/byo-video/scripts/foo.py`, `~/.agents/skills/byo-video/scripts/foo.py`, or `~/.claude/scripts/foo.py`.

Prefer the skill-local scripts directory when deploying to remote machines:

```bash
SCRIPT_DIR="${COSMOS_AGENT_SCRIPTS_DIR:-$PWD/.agents/skills/byo-video/scripts}"
python3 "$SCRIPT_DIR/nim_catalog.py" list --no-probe
```

## Bundled Resources

- `references/byo-video-runbook.md`: complete deployment protocol, model picker,
  NIM-local operations, runtime monitor, and recovery rules.
- `scripts/byo_video_setup.py`: remote setup and launch script.
- `scripts/byo_video_runtime_agent.py`: browser runtime-agent frontend for HF
  dataset selection, concurrent inference, FiftyOne launch/result writeback,
  and `pjramg/Safe_Unsafe_Test` smoke testing with the worker-safety prompt.
- `scripts/gradio_cr2_byo.py`: Gradio app for video/image inference.
- `scripts/byo_video_runtime_monitor.py`: runtime health, alert, and metrics monitor.
- `scripts/cosmos_deploy_monitor.py`: deployment monitor helper for legacy flows.
- `scripts/nim_catalog.py` and `scripts/nim_launch.sh`: NIM model catalog and launch helpers.
- `scripts/smoke_cr2_byo.py` and `scripts/smoke_nem_vl.py`: smoke tests.

## Operating Rules

- Do all credential checks before provisioning or model downloads.
- Never put HuggingFace, NGC, or Brev credentials in command-line arguments.
- For NIM-local mode, refresh `nim_catalog.py upstream` before presenting a model
  list, and warn if upstream docs include a model missing from `KNOWN_VLM_NIMS`.
- If a remote setup fails, capture the phase, instance, and one-line error in
  `/tmp/byo_video_observer_result.json` before asking the user for recovery.
