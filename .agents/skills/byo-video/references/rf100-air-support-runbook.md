# RF100-VL Air Support Runbook

Use this runbook when the user asks to monitor, tail, recover, scale, freeze,
or report on RF100-VL large batch inference across Brev instances.

## Mission

Air support means active SRE ownership of the batch, not passive log watching.
Every check-in must answer four questions:

1. Which instances exist and what job is each one actually doing now?
2. Which lanes are producing, idle, stuck, loaded but not producing, or blocked?
3. What safe corrective action was taken or why was action deferred?
4. Which instances, if any, are safe for the user to delete after evidence is frozen?

Instance names are weak evidence. Current run metadata, results pointers, process
state, model server state, and newest logs are the source of truth. A Brev named
`cr2` may be a reverse Cosmos-3-Super-Reasoner lane after repurposing.

## Required Snapshot

Start every wakeup with `brev ls`. For each RUNNING, STARTING, STOPPING, or
recently repurposed instance, collect:

- `/tmp/rf100_brev_current_run.json`
- newest `/tmp/benchmark-runs/<run_id>/results.json`
- newest `/ephemeral/benchmark-runs/<run_id>/results.json`, when present
- `/tmp/rf100_brev_*.log` tail and newest error class
- `nvidia-smi`
- `df -h / /ephemeral /tmp`
- model and runner processes
- vLLM `/v1/models` response when the endpoint is expected to be up

Prefer running `BREV_INSTANCE_NAME=<name> python3 /tmp/rf100_brev_status.py`
when the helper is present. That helper should report lane role, traversal
direction, shard index/count, current job, health, throughput, and action hints.

## Nemoclaw/OpenClaw Brev Experiment

Use this mode only when the BYO-video picker resolves `BREV_PROVISIONER=nemoclaw`.
It is an H200-only RF100-VL air-support experiment for
`nvidia/Cosmos3-Super-Reasoner` served by vLLM. The launchable URL is:
`https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3Azt0aYgVNFEuz7opyx3gscmowS`.

Before inference starts, Nemoclaw must:

- confirm Brev login, H200 GPU, disk headroom, HF token presence, and Roboflow
  key presence without printing secret values
- confirm `/v1/models` reports Cosmos3-Super-Reasoner
- read horde `10.57.233.243` report/artifact state
- build a completed-key manifest keyed by `dataset_name`, `image_id`,
  `question_id`, and `model_id`
- infer only missing RF100-VL rows and write versioned artifacts, never
  overwriting existing horde evidence in place

Nemoclaw may autonomously perform safe recovery: restart vLLM or the local
runner, resume checkpoints, dedupe outputs, clean non-evidence logs, and reduce
concurrency after repeated errors. It must escalate to Captain before deleting
instances, provisioning more compute, changing model scope, deleting evidence,
or overwriting horde artifacts.

For a 12-hour smoke, run 48 observation windows at 15-minute cadence. Each
Captain check-in must include Brev name, model, run id, rows done/remaining,
throughput, ETA, GPU/disk, horde writeback status, duplicate count, newest error
class, and corrective action. If Captain is disconnected, Nemoclaw continues
writing `/tmp/nemoclaw_rf100_status.json` on the Brev and mirrors status to
horde when reachable; the next check resumes from those files.

## Classify Current Job

Classify by metadata first, then by model/process evidence, then by name as the
last tiebreaker:

- `lane_role`
- `lane_direction`
- `shard_order`
- `air_support_group`
- `run_id`
- `model` and `/v1/models`
- `shard_index` and `shard_count`
- `source_config`

For forward/reverse fleets, report both sides:

- Forward lanes: work from the front of each shard.
- Reverse lanes: work from the end of each shard toward the middle. Former CR2
  hosts may be reverse Cosmos-3-Super-Reasoner workers.
- Meeting point: do not duplicate work across forward and reverse lanes. Before
  relaunching, inspect completed image ids for the shard and resume from the
  correct side.

## Health States

Use precise state labels:

- `healthy`: recent completions in the last 15 minutes.
- `slow`: still producing, but recent throughput is materially below the lane
  baseline.
- `dataset_load_stall`: runner is active, zero completions, and loading has
  exceeded the expected warmup window.
- `loaded_but_not_producing`: vLLM/model server is loaded, but no runner or no
  fresh results are visible.
- `runner_stopped_mid_run`: current-run pointer and partial results exist, but
  the runner is absent before completion.
- `idle`: no current run and no useful model/runner process.
- `blocked`: missing credentials, inaccessible Brev, failed model boot, disk
  pressure that cannot be cleaned without deleting evidence, or a risky compute
  scope change.

## Safe Corrective Actions

Safe actions are allowed without escalation when they preserve evidence and do
not change compute scope:

- relaunch the runner with the same `run_id`, model, shard index/count, and lane
  direction after freezing current results
- restart a crashed local runner when vLLM is healthy
- back off concurrency after repeated timeouts, OOMs, or 5xx errors
- remove obvious transient logs or caches that are not benchmark evidence
- refresh stale status pointers when a newer results file clearly exists
- copy evidence to the report host and verify row counts before deletion advice

Escalate before:

- deleting raw results, input images, checkpoints, or current-run pointers
- stopping or deleting Brev instances unless the user explicitly asked
- increasing the number of instances or materially changing GPU scope
- changing models, prompts, scoring method, or shard definitions
- exposing secret file contents or API keys

## Throughput Decay Triage

When production decays across lanes, compare recent 15-minute and 60-minute
rates against each lane's earlier baseline. Check for:

- runner missing while vLLM remains loaded
- result file mtime older than the last check-in
- GPU utilization high but no new rows, suggesting backend saturation or a
  wedged request queue
- GPU utilization low with runner active, suggesting dataset loading, I/O, API
  retry backoff, or stalled workers
- disk pressure on `/tmp` or `/ephemeral`
- repeated timeout, OOM, 429, or 5xx errors
- duplicate work caused by stale shard direction or stale instance naming

Correct one lane first, observe one interval, then roll the fix across similar
lanes. Do not scale from a failing setup.

## Scale-Up Gate

Before asking the user for more Brev instances, require a stable baseline on one
instance:

- model endpoint responds to `/v1/models`
- smoke run produces valid parseable output
- no new critical errors during at least one observation window
- enough images completed to estimate throughput and ETA
- disk headroom is comfortable
- shard metadata records role, direction, index, and count

For expensive multi-GPU fleets, prefer a 15-30 minute no-error baseline before
requesting more capacity. If throughput decays after scale-out, pause scale-up
and focus on root cause.

## Evidence Freeze and Deletion Advice

Tell the user an instance is safe to delete only after all of these are true:

- raw `results.json` is copied from `/tmp` and `/ephemeral` when both exist
- copied row counts are reconciled against the Brev-local files
- report artifacts are rebuilt and served
- the report host serves the trace/detail assets over HTTP
- deletion-safety manifest records the instance, copied rows, deduped rows, and
  remaining caveats

Be explicit about caveats. Result evidence can be safe while original Brev-local
image caches are not fully mirrored.

## User Updates

Regular updates should be operational and compact:

- aggregate production for forward, reverse, and total lanes
- per-lane health, recent throughput, percent, remaining images, and ETA
- corrective actions taken
- blockers or risks
- current safe-to-delete list and caveats

When the user asks for five-minute deletion updates, update the list even if it
has not changed. When the user asks for 15-minute shard updates, include every
active or expected shard and mark idle lanes explicitly.
