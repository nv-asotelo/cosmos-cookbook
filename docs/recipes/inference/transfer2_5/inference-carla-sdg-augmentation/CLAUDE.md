# Cosmos Transfer 2.5 Sim2Real for Simulator Videos — Cosmos Transfer 2.5 Inference

## Model

`https://github.com/nvidia-cosmos/cosmos-transfer2.5`

## Data Source

<!--
  Access: BYO (Bring Your Own) — CARLA simulator outputs or compatible synthetic driving video
  Control signals: RGB video + depth map + edge map + segmentation map from simulator
  License: User-provided data; CARLA outputs are open under MIT License
-->

**Input:** BYO synthetic driving video from CARLA or compatible simulator
**Access:** Bring Your Own — set `BYO_VIDEO` to your simulator RGB video path
**Format:** MP4; simulator should also provide depth, edge, and/or segmentation maps
**Reference:** https://carla.org/ | https://github.com/carla-simulator/carla

## Compute Requirements

1x A100-80GB or H100-80GB (minimum 80 GB VRAM for Cosmos Transfer 2.5)

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
No runnable notebook exists for this recipe — demo.sh accepts a BYO simulator video
and runs Cosmos Transfer 2.5 to generate photorealistic augmentations covering 18
environmental conditions (lighting, weather, road surface variations).

### Pre-flight

```bash
nvidia-smi
```

Confirm A100-80GB or H100-80GB is visible with >= 70000 MiB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
export BYO_VIDEO=/path/to/your/simulator_rgb_video.mp4
bash deploy/transfer2_5/inference-carla-sdg-augmentation/demo.sh
```

> BYO_VIDEO must point to a simulator-generated RGB video (MP4). For best results,
> also provide matching depth/segmentation maps from the simulator. See inference.md
> for the full 18-augmentation pipeline using all control modalities.

### What it runs

1. Validates BYO_VIDEO is provided and accessible
2. Sets up Cosmos Transfer 2.5 environment
3. Runs edge-based augmentation pass (snow + night) as representative examples
4. Writes timing metrics to `/tmp/carla_sdg_results.json`

### Success criteria

- `CUDA available: True` during setup
- Generated videos present in `/tmp/carla_sdg_output/`
- `carla_sdg_results.json` contains `"status": "success"`
- `frames_generated_total` > 0 and `throughput_fps` > 0

## Cosmos Metadata

| Field     | Value                                                                                                                                                            |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                        |
| Domain    | domain:autonomous-vehicles, domain:simulation                                                                                                                   |
| Technique | technique:video-to-video, technique:sim-to-real, technique:edge-control, technique:depth-control, technique:segmentation-control                               |
| Tags      | inference, transfer-2-5, sim2real, carla, sdg, augmentation, autonomous-vehicles                                                                               |
| Summary   | Transforms CARLA simulator driving videos into 18 photorealistic augmentations (lighting, weather, road surface variations) using Cosmos Transfer 2.5, preserving 100% of ground-truth anomaly behaviors for robust autonomous vehicle training data. |
