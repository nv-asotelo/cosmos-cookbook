# Style-Guided Video Generation — Cosmos Transfer 2.5 Inference

## Model

`https://github.com/nvidia-cosmos/cosmos-transfer2.5`

## Data Source

<!--
  Access: BYO (Bring Your Own) — control video + style reference images
  Control video: Edge/depth/segmentation map video (MP4)
  Style image: JPEG or PNG reference image defining the desired visual style
  License: User-provided data
-->

**Input:** BYO control video (edge, depth, or segmentation) + style reference image(s)
**Access:** Bring Your Own — set `BYO_VIDEO` (control video) and `BYO_STYLE_IMAGE` (reference image)
**Format:** MP4 for control video; JPEG/PNG for style image
**Note:** Blur control is not compatible with image prompts — use edge, depth, or segmentation only

## Compute Requirements

1x A100-80GB or H100-80GB (minimum 80 GB VRAM for Cosmos Transfer 2.5)

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
No runnable notebook exists for this recipe — demo.sh accepts a BYO control video
and style reference image, then runs Cosmos Transfer 2.5 to generate style-guided
video output combining structural control with image-based style transfer.

### Pre-flight

```bash
nvidia-smi
```

Confirm A100-80GB or H100-80GB is visible with >= 70000 MiB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
export BYO_VIDEO=/path/to/your/control_video.mp4
export BYO_STYLE_IMAGE=/path/to/your/style_reference.jpg
bash deploy/transfer2_5/inference-image-prompt/demo.sh
```

> BYO_VIDEO should be an edge, depth, or segmentation map video (MP4).
> BYO_STYLE_IMAGE defines the target visual style (color palette, lighting, mood).
> High-resolution reference images produce better style transfer results.

### What it runs

1. Validates BYO_VIDEO and BYO_STYLE_IMAGE are provided and accessible
2. Sets up Cosmos Transfer 2.5 environment
3. Builds a JSON spec combining edge control + image_context_path
4. Runs Cosmos Transfer 2.5 inference with style-guided generation
5. Writes timing metrics to `/tmp/image_prompt_results.json`

### Success criteria

- `CUDA available: True` during setup
- Generated video present in `/tmp/image_prompt_output/`
- `image_prompt_results.json` contains `"status": "success"`
- `frames_generated_total` > 0 and `throughput_fps` > 0

## Cosmos Metadata

| Field     | Value                                                                                                                                                             |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                         |
| Domain    | domain:content-creation, domain:autonomous-vehicles, domain:robotics                                                                                            |
| Technique | technique:video-to-video, technique:style-transfer, technique:image-guided, technique:edge-control, technique:depth-control, technique:segmentation-control      |
| Tags      | inference, transfer-2-5, style-transfer, image-prompt, multi-modal-control                                                                                      |
| Summary   | Style-guided video generation using Cosmos Transfer 2.5 image prompt feature; combines structural control (edge/depth/segmentation) with a style reference image to apply visual aesthetics (color palette, lighting, mood) to generated video while preserving motion and structure. |
