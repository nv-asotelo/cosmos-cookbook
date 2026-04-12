# Worker Safety — Cosmos Reason 2 Inference

## Model

`nvidia/Cosmos-Reason2-2B`

## Data Source

<!--
  Access: Public — no gating required for model; dataset is public on HuggingFace
  Dataset: pjramg/Safe_Unsafe_Test
  Size: ~4GB model weights; dataset varies by HF download
  License: NVIDIA Open Model License (model); dataset — see pjramg/Safe_Unsafe_Test on HuggingFace
-->

**Model access:** Gated — requires accepting the NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-2B
**Model size:** ~4GB
**Dataset:** [`pjramg/Safe_Unsafe_Test`](https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test) — public HuggingFace dataset, downloaded automatically via FiftyOne

```bash
# Model
huggingface-cli download nvidia/Cosmos-Reason2-2B --repo-type model --local-dir ./models/Cosmos-Reason2-2B

# Dataset (loaded automatically in script via FiftyOne)
# fouh.load_from_hub("pjramg/Safe_Unsafe_Test", persistent=True)
```

## Compute Requirements

1x H100-80GB or A100-80GB (2B model requires ~40 GB VRAM free)

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
The recipe includes `worker_safety.py` (a jupytext-converted script from the notebook)
which can be run directly without JupyterLab.

### Pre-flight

```bash
nvidia-smi
```

Confirm GPU is visible with >= 40 GB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
bash deploy/reason2/worker_safety/demo.sh
```

### What it runs

Zero-shot warehouse safety inspection using `worker_safety.py`:
1. **Dataset load** — pulls `pjramg/Safe_Unsafe_Test` via FiftyOne from HuggingFace
2. **Inference loop** — runs Cosmos-Reason2-2B on each video clip using the expert inspector prompt strategy
3. **JSON parsing** — extracts structured classification: class ID, label, hazard flag
4. **Results export** — saves per-clip predictions to `/tmp/worker_safety_results.json`

### Success criteria

- `CUDA available: True` during setup
- `/tmp/worker_safety_results.json` is created with entries containing `"status": "success"`
- At least one sample shows `"safety_label"` field populated with a valid class label

## Cosmos Metadata

| Field     | Value                                                                                                                                                                              |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                                          |
| Domain    | domain:robotics                                                                                                                                                                    |
| Technique | technique:reasoning                                                                                                                                                                |
| Tags      | inference, reason-2, worker-safety, fiftyone, zero-shot                                                                                                                            |
| Summary   | Zero-shot industrial safety inspection in brownfield warehouses using Cosmos-Reason2-2B. Classifies worker behaviors (safe/unsafe) from video using expert inspector prompt strategy, bypassing custom model training. |
