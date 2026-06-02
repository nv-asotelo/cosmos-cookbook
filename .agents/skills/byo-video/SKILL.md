---
name: byo-video
description: Deploy the Cosmos BYO-Video guided dataset batch UI, single-video upload UI, or dataset browser/result review flow on Brev, Horde, SSH, or a local GPU. Use for Cosmos Reason/Cosmos3/Nemotron/Qwen VLM video or image inference, HF public dataset batch inference, NIM-local mode, runtime monitoring, and NIM model switching.
compatibility: Codex, Claude Code, and Kimi Code CLI. Translate legacy Claude tool names in the runbook to the current agent's native tools.
---

# BYO-Video

Use this skill to launch Cosmos BYO-Video with a guided dataset batch UI,
single-video upload UI, or dataset browser/result review flow. Read
`references/byo-video-runbook.md` before executing, but apply the compatibility
adapter below first.

When asking the user to choose a frontend, do not present raw implementation
names like `batch_inference`, `gradio`, or `fiftyone` as the option labels. Use
descriptive labels and map the answer to the env var afterward:

- **Guided dataset batch UI** -> `BYO_VIDEO_FRONTEND=batch_inference`
  for HF public dataset loading, concurrent video processing, worker-safety
  smoke testing, and result export/writeback. The setup script still starts a
  live Gradio sidecar and prints its link.
- **Single-video upload UI** -> `BYO_VIDEO_FRONTEND=gradio` for the default
  NVIDIA Build-style Gradio skin with one-video/image upload, prompt presets,
  reasoning on/off indicators, backend controls, and parameter tuning.
- **Dataset browser + result viewer** -> `BYO_VIDEO_FRONTEND=fiftyone` when the
  user wants the guided batch UI with FiftyOne available for sample inspection
  and result review. The setup script still starts a live Gradio sidecar and
  prints its link.
- **NVIDIA Build-style model playground (Default)** -> `BYO_VIDEO_FRONTEND=nvidia_build`
  for a model-specific Gradio surface that mirrors the corresponding
  build.nvidia.com playground when the selected model is from the Cosmos,
  Cosmos3, Cosmos Predict, or Cosmos Reason collections. The setup script
  chooses the closest bundled app for the selected model and gates generation
  frontends by model capability: VLM/reasoner checkpoints serve Reason surfaces,
  VFM/generator checkpoints serve Predict/generation surfaces.

Do not serve a Predict/generation frontend for a VLM-only Reasoner such as
`nvidia/Cosmos3-Nano-Reasoner`. For future Omni/Mixture-of-Transformers
checkpoints that expose both towers, ask the user what use cases they want to
see — generation, reasoning, or both — and map the answer to
`BYO_VIDEO_MOT_TOWER=generation|reasoning|both` before launch.

Regardless of the selected mode, `byo_video_setup.py` must serve a live Gradio
link on `GRADIO_PORT` and write it to `/tmp/gradio_url.txt` plus
`/tmp/gradio_live.flag`. Batch-inference and FiftyOne selections add their own UI
on `BATCH_INFERENCE_PORT`; they do not replace Gradio.

Default to `BYO_VIDEO_FRONTEND=nvidia_build` when the user asks to serve a
shareable model playground, compare model-page frontends, or does not specify a
dataset/batch workflow. Use `BYO_VIDEO_FRONTEND=batch_inference` when the user
asks for a guided inference flow, HF public dataset loading, concurrent batch
processing, or the worker-safety smoke test. Use `BYO_VIDEO_FRONTEND=gradio`
for the generic single-video upload UI. Use `BYO_VIDEO_FRONTEND=fiftyone` when
the user specifically wants FiftyOne available alongside result writeback.

## Codex Mode Note

This skill has an interactive picker flow. In Codex, the button-style picker
uses `request_user_input`, which is only available in Plan mode.

If `request_user_input` is unavailable, tell the user:

> Codex's interactive picker for this skill is available in Plan mode. You can
> switch to Plan mode and invoke this skill again for button-style choices, or
> reply here with backend, model, and target and I will continue in chat.

Do not fail the workflow just because Plan mode is unavailable; fall back to
plain chat choices.

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
- `references/rf100-air-support-runbook.md`: RF100-VL large-batch SRE guidance
  for shard role discovery, forward/reverse lanes, throughput decay, evidence
  freeze, deletion-safety updates, and user check-ins.
- `scripts/byo_video_setup.py`: remote setup and launch script.
- `scripts/byo_video_batch_inference.py`: browser batch-inference frontend for HF
  dataset selection, concurrent inference, FiftyOne launch/result writeback,
  and `pjramg/Safe_Unsafe_Test` smoke testing with the worker-safety prompt.
- `scripts/byo_video_runtime_guide.py`: friendly CLI companion for Claude Code
  or terminal users to load datasets, import papers, run guarded batches, shape
  structured prompts, and export artifacts through the batch-inference API.
- `scripts/gradio_cr2_byo.py`: default Build-style Gradio app for video/image
  inference across supported BYO-video models.
- `scripts/alpamayo_openai_server.py`: OpenAI-compatible `/v1/models` and
  `/v1/chat/completions` adapter for Alpamayo VQA/captioning over BYO images
  and videos.
- `scripts/byo_video_runtime_monitor.py`: runtime health, alert, and metrics monitor.
- `scripts/nim_runtime_monitor.py` and `scripts/nim_param_table.json`: NIM boot/readiness monitor and launch-parameter notes from the smoke sprints.
- `scripts/gradio_cosmos_predict.py` and `scripts/gradio_cosmos_transfer.py`: generation frontends for the Cosmos Predict/Transfer BYO-video variants.
- `scripts/cosmos3_native_launch.sh`: wraps the `NVIDIA/cosmos-framework` Ray Serve + Gradio stack
  for the Cosmos3 OSS *Generator* checkpoints (Cosmos3-Nano, Cosmos3-Super). Invoked under
  `INFERENCE_BACKEND=cosmos3_native` (auto-set when `MODEL_SIZE` is `C3-NANO-GEN` or `C3-SUPER-GEN`).
  See the "Cosmos3 OSS Backend Routing" section of the runbook.
- `scripts/cosmos_deploy_monitor.py`: deployment monitor helper for legacy flows.
- `scripts/nim_catalog.py`, `scripts/nim_launch.sh`, and
  `scripts/nim_switch_service.py`: NIM model catalog, launch helper, and
  companion switch/status service for Gradio restarts.
- `scripts/smoke_cr2_byo.py` and `scripts/smoke_nem_vl.py`: smoke tests.

## Operating Rules

- Do all credential checks before provisioning or model downloads.
- Never put HuggingFace, NGC, or Brev credentials in command-line arguments.
- For RF100-VL air support, classify every active Brev by current-run metadata,
  results pointers, process state, and model endpoint before using the instance
  name. Former CR2 hosts may be reverse Cosmos-3-Super-Reasoner lanes.
- For NIM-local mode, refresh `nim_catalog.py upstream` before presenting a model
  list, and warn if upstream docs include a model missing from `KNOWN_VLM_NIMS`.
- If a remote setup fails, capture the phase, instance, and one-line error in
  `/tmp/byo_video_observer_result.json` before asking the user for recovery.
