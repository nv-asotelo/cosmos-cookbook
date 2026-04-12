# Multi-Control Recipes — Cosmos Transfer 2.5 Inference

## Model

`https://github.com/nvidia-cosmos/cosmos-transfer2.5`

## Data Source

<!--
  Access: BYO (Bring Your Own) — real or Omniverse-generated video + control modalities
  Control signals: Edge, segmentation, vis, mask, depth (from Omniverse or computed)
  License: User-provided data; NVIDIA Omniverse outputs subject to NVIDIA EULA
-->

**Input:** BYO video (real capture or NVIDIA Omniverse synthetic) + control modalities
**Access:** Bring Your Own — set `BYO_VIDEO` to your source video path
**Format:** MP4; control modalities (edge, seg, mask) computed or exported from Omniverse
**Reference:** https://docs.isaacsim.omniverse.nvidia.com/

## Compute Requirements

1x A100-80GB or H100-80GB (minimum 80 GB VRAM for Cosmos Transfer 2.5)

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
No runnable notebook exists for this recipe — demo.sh accepts a BYO video and runs
Cosmos Transfer 2.5 with multi-control configurations for background replacement,
lighting change, color/texture change, or object transformation. Covers the full
recipe matrix including Omniverse sim-to-real workflows.

### Pre-flight

```bash
nvidia-smi
```

Confirm A100-80GB or H100-80GB is visible with >= 70000 MiB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
export BYO_VIDEO=/path/to/your/video.mp4
bash deploy/transfer2_5/inference-real-augmentation/demo.sh
```

> BYO_VIDEO must point to your source video (MP4). The script runs the background
> change recipe by default (edge filtered + seg + vis). For Omniverse workflows,
> export RGB, edge, seg, and mask videos from IsaacSim before running.

### What it runs

1. Validates BYO_VIDEO is provided and accessible
2. Sets up Cosmos Transfer 2.5 environment
3. Generates edge map from BYO_VIDEO via Canny detection
4. Runs background change recipe: Edge (1.0) + Seg (0.4 inverted mask) + Vis (0.6)
5. Writes timing metrics to `/tmp/real_augmentation_results.json`

### Success criteria

- `CUDA available: True` during setup
- Generated video present in `/tmp/real_augmentation_output/`
- `real_augmentation_results.json` contains `"status": "success"`
- `frames_generated_total` > 0 and `throughput_fps` > 0

## Cosmos Metadata

| Field     | Value                                                                                                                                                              |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                          |
| Domain    | domain:robotics, domain:content-creation, domain:simulation                                                                                                      |
| Technique | technique:video-to-video, technique:multi-control, technique:edge-control, technique:segmentation-control, technique:sim-to-real                                 |
| Tags      | inference, transfer-2-5, multi-control, background-change, lighting-change, omniverse, sim2real                                                                  |
| Summary   | Four multi-control recipes (background replacement, lighting change, color/texture change, object transformation) using Cosmos Transfer 2.5 with combined edge, segmentation, vis, and mask modalities; includes Omniverse sim-to-real photorealistic generation workflow. |
