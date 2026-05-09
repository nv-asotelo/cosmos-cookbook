# inference

## What This Recipe Does
Classify industrial worker-safety videos with Cosmos Reason 2, store structured predictions in FiftyOne, and launch the FiftyOne App for visual audit against the source clips.

## Model
`nvidia/Cosmos-Reason2-2B`

## Entry Points
Run from an environment that has the Cosmos Reason 2 dependencies installed:

```bash
cd docs/recipes/inference/reason2/worker_safety
WORKER_SAFETY_LAUNCH_APP=1 \
WORKER_SAFETY_FIFTYONE_WAIT=0 \
FIFTYONE_ADDRESS=0.0.0.0 \
FIFTYONE_PORT=5151 \
python worker_safety.py
```

For notebook-first runs:

```bash
jupyter lab worker_safety.ipynb
```

## Data Source
**Access:** Public
**Size:** ~1-2GB for the sampled Hugging Face dataset
**License:** CC BY 4.0 for the upstream Data in Brief/Mendeley dataset; Hugging Face card does not declare a license field

```bash
huggingface-cli download pjramg/Safe_Unsafe_Test --repo-type dataset --local-dir ./data/safe_unsafe_test
```

## Compute Requirements
Minimum: 1 NVIDIA CUDA GPU with enough VRAM for `nvidia/Cosmos-Reason2-2B` video inference.

Recommended: 1 GPU with 48GB+ VRAM for the full 40-video FiftyOne dataset. The end-to-end smoke test was run on one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU with 96GB VRAM.

Expected runtime for the full dataset is roughly 1-2 hours, depending on FPS, video decode speed, and GPU generation throughput.

## Dependencies

```
torch
transformers
qwen-vl-utils
fiftyone
ffmpeg
```

## Required Environment Variables
No required environment variables when Hugging Face auth is already configured on the machine.

| Variable | Description |
|----------|-------------|
| `WORKER_SAFETY_MODEL` | Optional Hugging Face model ID or local model path. Defaults to `nvidia/Cosmos-Reason2-2B`. |
| `WORKER_SAFETY_DATASET` | Optional FiftyOne/Hugging Face dataset slug. Defaults to `pjramg/Safe_Unsafe_Test`. |
| `WORKER_SAFETY_LAUNCH_APP` | Optional `0`/`1` flag. Defaults to `1`; set `0` for batch-only smoke tests. |
| `WORKER_SAFETY_FIFTYONE_WAIT` | Optional `0`/`1` flag. Defaults to `0`; set `1` in interactive sessions to keep the app open. |
| `FIFTYONE_ADDRESS` | Optional host binding for the FiftyOne App, e.g. `0.0.0.0` on a remote GPU instance. |
| `FIFTYONE_PORT` | Optional port for the FiftyOne App, e.g. `5151`. |

## Setup Prerequisites
- [ ] Follow `setup.md` through the Cosmos Reason 2 virtual environment creation.
- [ ] Authenticate Hugging Face with `hf auth login` or `huggingface-cli login`.
- [ ] Install FiftyOne into the same virtual environment as Cosmos Reason 2:

```bash
uv pip install -U fiftyone
# If pip exists in the environment, python -m pip install -U fiftyone is also fine.
```

- [ ] Confirm CUDA and FiftyOne import from the same Python interpreter:

```bash
python - <<'PY'
import torch
import fiftyone as fo
print("cuda", torch.cuda.is_available())
print("fiftyone", fo.__version__)
PY
```

## Key Files

| File | Role |
|------|------|
| `worker_safety.py` | Jupytext Python export that loads the FiftyOne dataset, runs Cosmos Reason 2 inference, stores predictions, and launches the FiftyOne App. |
| `worker_safety.ipynb` | Notebook version of the same workflow. |
| `inference.md` | Human-readable recipe walkthrough and prompt explanation. |
| `setup.md` | Environment setup guide. |
| `assets/` | Overview image, sample clip, and result clips used by the docs. |

## Code Structure
- `load_fiftyone_dataset()` loads the persistent FiftyOne dataset when present or downloads it from Hugging Face.
- `load_model()` loads Cosmos Reason 2 via Hugging Face Transformers and configures video pixel-token limits.
- `parse_model_json()` extracts structured JSON from model output, tolerating markdown fences or light preamble text.
- The sample loop builds the video/text conversation, runs generation, stores `cosmos_analysis`, and adds a top-level `safety_label` classification field for filtering in FiftyOne.
- The final cell launches the FiftyOne App unless `WORKER_SAFETY_LAUNCH_APP=0`.

## Expected Output

```
Reference video copied to: assets/sample.mp4
CUDA available: True
CUDA device count: 1
Device name: NVIDIA RTX PRO 6000 Blackwell Server Edition
Processing 40 videos...
Processing complete.
FiftyOne App launched: http://0.0.0.0:5151/
```

In FiftyOne, samples should contain:
- `cosmos_analysis`: parsed JSON prediction details
- `safety_label`: `fo.Classification` with the predicted label
- `cosmos_error`: raw output only for samples that fail JSON parsing

## Gotchas
- Run the script from `docs/recipes/inference/reason2/worker_safety` so `assets/sample.mp4` is created in the recipe assets directory.
- Install `fiftyone` into the active Cosmos Reason 2 virtual environment; installing it into system Python will not help the recipe.
- On headless SSH machines, set `FIFTYONE_ADDRESS=0.0.0.0` and a fixed `FIFTYONE_PORT`, then open an SSH tunnel from the local laptop if you need to view the App.
- `WORKER_SAFETY_FIFTYONE_WAIT=1` is useful in notebooks but will block noninteractive smoke tests.
- If the model emits explanatory prose around JSON, keep the raw text in `cosmos_error` and use `parse_model_json()` as the first fix point.
