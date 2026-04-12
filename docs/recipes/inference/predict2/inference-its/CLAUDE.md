# Cosmos Predict 2 Text2Image for ITS Images — Cosmos Predict 2 Inference

## Model

`https://github.com/nvidia-cosmos/cosmos-predict2`

## Data Source

<!--
  Access: Public datasets used for evaluation (ACDC, SUTD, DAWN)
  Generation: Text-to-image (no input images required for demo)
  License: ACDC: https://acdc.vision.ee.ethz.ch/  SUTD: public research  DAWN: Kaggle public
-->

**Dataset (evaluation):** ACDC / SUTD / DAWN — public ITS datasets
**Access:** Public — no authentication required for generation
**Generation:** Text-to-image; demo hardcodes 5 representative ITS prompts
**Reference datasets:**
- ACDC: https://acdc.vision.ee.ethz.ch/
- SUTD: https://sutdcv.github.io/SUTD-TrafficQA/
- DAWN: https://www.kaggle.com/datasets/shuvoalok/dawn-dataset

## Compute Requirements

1x A100-80GB or H100-80GB (NVIDIA Ampere or newer recommended; minimum 80 GB VRAM)

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
No runnable notebook exists for this recipe — demo.sh generates 5 representative ITS
images from hardcoded prompts (diverse camera angles, weather, and objects) and
measures per-image generation time and throughput.

### Pre-flight

```bash
nvidia-smi
```

Confirm A100-80GB or H100-80GB is visible with >= 70000 MiB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
bash deploy/predict2/inference-its/demo.sh
```

### What it runs

Five representative ITS text-to-image generations covering:
1. Night-time urban intersection with motorcyclist (front dashboard view)
2. Snowy highway scene with bicycle and cars (roadside camera)
3. Foggy intersection with pedestrians and buses (top-down view)
4. Rainy urban scene with mixed traffic (dashboard view)
5. Clear daylight suburban intersection with cyclists (overhead camera)

### Success criteria

- `CUDA available: True` during setup
- 5 images generated in `/tmp/its_output/`
- `its_results.json` contains `"status": "success"`
- `images_generated` == 5 and `throughput_images_per_min` > 0

## Cosmos Metadata

| Field     | Value                                                                                                                                                          |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                      |
| Domain    | domain:autonomous-vehicles, domain:intelligent-transportation                                                                                                 |
| Technique | technique:text-to-image, technique:synthetic-data-generation                                                                                                 |
| Tags      | inference, predict-2, its, synthetic-data, autonomous-vehicles, object-detection                                                                              |
| Summary   | Text-to-image synthetic data generation for Intelligent Transportation System (ITS) using Cosmos Predict 2; generates diverse ITS images across camera angles, lighting, and weather conditions to improve downstream RT-DETR object detection AP50 on ACDC/SUTD/DAWN benchmarks. |
