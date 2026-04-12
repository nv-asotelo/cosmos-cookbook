# Sim2Real X-Mobility Navigation — Cosmos Transfer 1 Inference

## Model

`https://github.com/nvidia-cosmos/cosmos-transfer1`

HuggingFace: `nvidia/Cosmos-Transfer1-7B` (or `nvidia/Cosmos-Transfer1` — check the repo's INSTALL.md for the exact model ID)

## Data Source

<!--
  Access: Public (HuggingFace, gated — requires accepting dataset license)
  Dataset: nvidia/X-Mobility
  Random Action Dataset: x_mobility_isaac_sim_random_160k.zip (160K frames)
  Teacher Policy Dataset: x_mobility_isaac_sim_nav2_100k.zip (100K frames)
  License: See https://huggingface.co/datasets/nvidia/X-Mobility for terms
-->

**Access:** Public (HuggingFace — may require accepting dataset terms)
**Dataset:** `nvidia/X-Mobility`
**Size:** ~160K frames (random) + ~100K frames (teacher policy)

| Split | File | Frames | Use |
|-------|------|--------|-----|
| Random Action | `x_mobility_isaac_sim_random_160k.zip` | 160K | World model pre-training |
| Teacher Policy | `x_mobility_isaac_sim_nav2_100k.zip` | 100K | Joint world model + action policy training |

```bash
# Download random action dataset (smaller — recommended for demo)
huggingface-cli download nvidia/X-Mobility \
  x_mobility_isaac_sim_random_160k.zip \
  --repo-type dataset \
  --local-dir ./data/X-Mobility
```

Each frame includes: image (RGB), speed, semantic label, route, path, action command.
Semantic classes: [Navigable, Forklift, Cone, Sign, Pallet, Fence, Background]

## Compute Requirements

- **Inference (Cosmos Transfer 1):** 1x H100-80GB or A100-80GB (>= 70 GB VRAM)
- **Training (out of scope for demo.sh):** 8x H100 GPUs — see note below

> **Training is out of scope for this demo.** The demo.sh runs INFERENCE ONLY —
> augmenting X-Mobility frames with Cosmos Transfer 1 to produce photorealistic
> navigation videos. The full training pipeline (world model pre-training + action
> policy training) requires 8x H100 GPUs and is documented in inference.md.

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
demo.sh covers the inference / data augmentation step only.

### Pre-flight

```bash
nvidia-smi
```

Confirm H100-80GB or A100-80GB is visible with >= 70 GB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
bash deploy/transfer1/inference-x-mobility/demo.sh
```

### What it runs

1. Validates GPU VRAM (>= 70 GB required)
2. Clones `cosmos-transfer1` and installs dependencies
3. Downloads a small subset of the X-Mobility random action dataset
4. Converts X-Mobility frames to video format using the recipe's `xmob_dataset_to_videos.py`
5. Runs Cosmos Transfer 1 with edge=0.3 + seg=1.0 controls (recipe configuration)
6. Writes timing metrics to `/tmp/x_mobility_results.json`

### Success criteria

- `CUDA available: True` during setup
- `x_mobility_results.json` contains `"status": "success"`
- Output video(s) exist at `/tmp/x_mobility_output/`
- Generated frames appear photorealistic (not synthetic CG)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                               |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                                           |
| Domain    | domain:robotics, domain:autonomous-vehicles                                                                                                                                         |
| Technique | technique:video-to-video, technique:sim2real, technique:navigation                                                                                                                  |
| Tags      | inference, transfer-1, sim2real, x-mobility, navigation, warehouse-robotics                                                                                                         |
| Summary   | Uses Cosmos Transfer 1 to augment X-Mobility navigation dataset (Mobility Gen / Isaac Sim) with photorealistic appearance variations, improving Sim2Real performance of the X-Mobility navigation policy (+68.5% success rate, -56% trip time). Training requires 8x H100 — demo.sh covers inference only. |
