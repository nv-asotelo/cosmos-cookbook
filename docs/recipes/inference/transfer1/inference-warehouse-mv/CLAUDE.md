# Sim2Real Warehouse Multi-View — Cosmos Transfer 1 Inference

## Model

`https://github.com/nvidia-cosmos/cosmos-transfer1`

HuggingFace: `nvidia/Cosmos-Transfer1-7B` (or `nvidia/Cosmos-Transfer1` — check the repo's INSTALL.md for the exact model ID)

## Data Source

<!--
  Access: Bundled sample (BYO for full scale) + optional public HuggingFace dataset
  Bundled: scripts/examples/transfer1/inference-warehouse-mv/assets/SURF_Booth_030825/
  Public:  nvidia/PhysicalAI-SmartSpaces on HuggingFace (250+ hours, 2D/3D annotations)
  License: See dataset-specific terms; SURF_Booth sample is provided with the recipe
-->

**Bundled sample (included):** `scripts/examples/transfer1/inference-warehouse-mv/assets/SURF_Booth_030825/`

6-camera warehouse setup with synchronized RGB and depth videos:
```
SURF_Booth_030825/
  Camera_00/ through Camera_05/
    rgb.mp4       — RGB input video per camera
    depth.mp4     — Depth control video per camera
```

**Extended dataset (public, HuggingFace):**
`nvidia/PhysicalAI-SmartSpaces` — 250+ hours of synchronized multi-camera video
with 2D/3D annotations, depth maps, and calibration data.

```bash
huggingface-cli download nvidia/PhysicalAI-SmartSpaces --repo-type dataset --local-dir ./data/PhysicalAI-SmartSpaces
```

## Compute Requirements

- 1x H100-80GB or A100-80GB (>= 70 GB VRAM free for Cosmos Transfer 1)
- Storage: >= 200 GB (model checkpoints ~100 GB + 6-camera output videos)
- Per-camera inference: the 6 cameras are processed sequentially (one inference run each)

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
No runnable notebook — demo.sh runs all 6 camera views through Cosmos Transfer 1 sequentially.

### Pre-flight

```bash
nvidia-smi
```

Confirm H100-80GB or A100-80GB is visible with >= 70 GB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
bash deploy/transfer1/inference-warehouse-mv/demo.sh
```

### What it runs

1. Validates GPU VRAM (>= 70 GB required)
2. Clones `cosmos-transfer1` and installs dependencies
3. Copies bundled SURF_Booth_030825 warehouse dataset from recipe `scripts/` assets
4. Runs Cosmos Transfer 1 on each of the 6 camera views (edge=0.5, depth=0.5)
5. Collects per-camera timings and aggregated metrics
6. Writes results to `/tmp/warehouse_mv_results.json`

### Success criteria

- `CUDA available: True` during setup
- `warehouse_mv_results.json` contains `"status": "success"`
- 6 output videos (one per camera) exist at `/tmp/warehouse_mv_output/`
- Generated videos show realistic warehouse textures (not synthetic CG appearance)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                               |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                                           |
| Domain    | domain:robotics, domain:smart-infrastructure                                                                                                                                        |
| Technique | technique:video-to-video, technique:sim2real, technique:multi-view                                                                                                                  |
| Tags      | inference, transfer-1, sim2real, multi-view, warehouse, detection, tracking                                                                                                         |
| Summary   | Applies Cosmos Transfer 1 with edge+depth controls to transform Omniverse-generated synthetic warehouse videos into realistic multi-view scenes, closing the sim-to-real gap for downstream 3D detection and tracking models. |
