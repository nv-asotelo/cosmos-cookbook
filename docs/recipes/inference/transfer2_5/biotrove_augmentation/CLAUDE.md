# Domain Transfer for BioTrove Moths — Cosmos Transfer 2.5 Inference

## Model

`https://github.com/nvidia-cosmos/cosmos-transfer2.5`

## Data Source

<!--
  Access: Public — no authentication required
  Dataset: pjramg/moth_biotrove (HuggingFace Hub)
  Size: ~1000 moth images (demo uses 20-50 samples)
  License: See dataset card on HuggingFace
-->

**Dataset:** `pjramg/moth_biotrove` (public, HuggingFace Hub)
**Access:** Public — no HuggingFace token required
**License:** See dataset card at https://huggingface.co/datasets/pjramg/moth_biotrove

```python
import fiftyone.utils.huggingface as fouh
dataset = fouh.load_from_hub(
    "pjramg/moth_biotrove",
    persistent=True,
    overwrite=True,
    max_samples=20,
)
```

## Compute Requirements

1x A100-80GB or H100-80GB (minimum 80 GB VRAM for Cosmos Transfer 2.5)

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
No runnable notebook exists for this recipe — demo.sh downloads a public subset of
the BioTrove moth dataset, converts images to videos, generates Canny edge maps,
builds JSON spec files, runs Cosmos Transfer 2.5 inference, and outputs augmented
agricultural moth scenes.

### Pre-flight

```bash
nvidia-smi
```

Confirm A100-80GB or H100-80GB is visible with >= 70000 MiB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
bash deploy/transfer2_5/biotrove_augmentation/demo.sh
```

> HF_TOKEN is required for the Cosmos Transfer 2.5 model weights. The BioTrove
> moth dataset itself is public and does not require authentication.

### What it runs

1. Downloads 20 moth images from `pjramg/moth_biotrove` via FiftyOne
2. Converts each image into a 10-frame MP4 clip via FFmpeg
3. Generates Canny edge maps as control signals
4. Builds JSON spec files with agricultural scene prompts
5. Runs Cosmos Transfer 2.5 inference (edge control, Python invocation)
6. Extracts last frames from generated videos
7. Writes timing metrics to `/tmp/biotrove_augmentation_results.json`

### Success criteria

- `CUDA available: True` during setup
- Generated videos present in `/tmp/biotrove_augmentation_output/`
- `biotrove_augmentation_results.json` contains `"status": "success"`
- `frames_generated_total` > 0 and `throughput_fps` > 0

## Cosmos Metadata

| Field     | Value                                                                                                                                                      |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                  |
| Domain    | domain:biology, domain:agriculture                                                                                                                         |
| Technique | technique:video-to-video, technique:edge-control, technique:domain-transfer                                                                               |
| Tags      | inference, transfer-2-5, biotrove, fiftyone, augmentation, biological-datasets                                                                            |
| Summary   | Domain-transfer pipeline for the BioTrove moth dataset using Cosmos Transfer 2.5 edge control to convert lab-style moth images into photorealistic agricultural scenes, addressing data scarcity and domain gap. |
