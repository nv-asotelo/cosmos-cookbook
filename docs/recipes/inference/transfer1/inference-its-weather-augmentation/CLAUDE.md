# ITS Weather Augmentation — Cosmos Transfer 1 Inference

## Model

`https://github.com/nvidia-cosmos/cosmos-transfer1`

HuggingFace: `nvidia/Cosmos-Transfer1-7B` (or `nvidia/Cosmos-Transfer1` — check the repo's INSTALL.md for the exact model ID)

## Data Source

<!--
  Access: Public — ACDC, SUTD, and DAWN datasets available from their respective hosts
  ACDC:  ~2k images, https://acdc.vision.ee.ethz.ch/ (requires registration)
  SUTD:  ~10k images, https://sutdcv.github.io/SUTD-TrafficQA/#/download
  DAWN:  ~1k images, https://www.kaggle.com/datasets/shuvoalok/dawn-dataset
  License: See each dataset's terms (research/academic use)
-->

**Access:** Public (registration required for ACDC and SUTD; Kaggle account for DAWN)
**Use case:** Evaluation benchmark datasets for downstream ITS object detection

| Dataset | Size      | Weather Conditions           | Download |
|---------|-----------|------------------------------|----------|
| ACDC    | ~2k images | snow, fog, rain, night       | https://acdc.vision.ee.ethz.ch/ |
| SUTD    | ~10k images | snow, fog, rain, night, cloudy, sunny | https://sutdcv.github.io/SUTD-TrafficQA/#/download |
| DAWN    | ~1k images | snow, fog, rain, sandy       | https://www.kaggle.com/datasets/shuvoalok/dawn-dataset |

For the demo, clear-weather ITS sample images from the recipe's `assets/` directory
are used as input (no external download required for basic demo execution).

## Compute Requirements

- 1x H100-80GB or A100-80GB (>= 70 GB VRAM free for Cosmos Transfer 1)
- Storage: >= 200 GB (model checkpoints ~100 GB + input/output videos)

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
No runnable notebook exists for this recipe — demo.sh runs Cosmos Transfer 1
weather augmentation on the recipe's bundled sample ITS images.

### Pre-flight

```bash
nvidia-smi
```

Confirm H100-80GB or A100-80GB is visible with >= 70 GB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
bash deploy/transfer1/inference-its-weather-augmentation/demo.sh
```

### What it runs

1. Validates GPU VRAM (>= 70 GB required)
2. Clones `cosmos-transfer1` and installs dependencies
3. Uses sample clear-weather ITS image from recipe `assets/` (converted to video)
4. Runs Cosmos Transfer 1 with depth + segmentation controls (control_weight=0.9)
5. Generates augmented rainy-night version of the input scene
6. Writes timing metrics to `/tmp/its_weather_results.json`

### Success criteria

- `CUDA available: True` during setup
- `its_weather_results.json` contains `"status": "success"`
- Output video exists at `/tmp/its_weather_output/`
- Generated scene shows weather transformation (darker scene, rain effects)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                              |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                                          |
| Domain    | domain:autonomous-vehicles, domain:smart-infrastructure                                                                                                                            |
| Technique | technique:video-to-video, technique:weather-augmentation, technique:synthetic-data-generation                                                                                      |
| Tags      | inference, transfer-1, weather-augmentation, ITS, object-detection, synthetic-data                                                                                                 |
| Summary   | Uses Cosmos Transfer 1 with depth+segmentation controls to augment clear-weather ITS highway images into adverse weather conditions (rain, fog, night), improving downstream RT-DETR detector performance across ACDC, SUTD, and DAWN benchmarks. |
