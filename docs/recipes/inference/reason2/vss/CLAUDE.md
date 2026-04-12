# Video Search and Summarization — Cosmos Reason 2 Inference

## Model

`nvidia/Cosmos-Reason2-8B`

## Data Source

<!--
  Access: User-supplied — VSS accepts any video file or live camera stream
  Dataset: No fixed dataset; user uploads video files via the VSS REST API or UI
  License: NVIDIA Open Model License (model)
-->

**Access:** User-supplied video files or live camera streams. No fixed dataset required.
**Model access:** Gated — requires NGC API key for VSS NIM deployment.
See https://build.nvidia.com/nvidia/video-search-and-summarization

VSS accepts video uploads via its REST API:
```python
import requests

vss_host = "http://localhost:8100"
files_endpoint = vss_host + "/files"

with open("/path/to/video.mp4", "rb") as f:
    response = requests.post(files_endpoint,
        data={"purpose": "vision", "media_type": "video"},
        files={"file": ("video_file", f)})
video_id = response.json()["id"]
```

## Compute Requirements

- **Multi-GPU (recommended):** 8x H100-80GB — for the local deployment profile using Cosmos-Reason2-8B
- **Single-GPU (minimum):** 1x H100-80GB — for the single-GPU deployment profile with smaller models
- See: https://docs.nvidia.com/vss/latest/content/supported_platforms.html

## Execution

VSS is a full microservices stack (ingestion + retrieval pipeline). It cannot be
run as a single script. Execution is via Docker Compose; the demo.sh scaffolds
the environment but delegates to the VSS documentation for full deployment.

**Note:** This recipe does not have a standalone headless inference script.
Use the Brev Launchable or follow the local deployment guide.

- Brev Launchable: https://docs.nvidia.com/vss/latest/content/cloud_brev.html
- Local deployment: https://docs.nvidia.com/vss/latest/content/vss_dep_docker_compose_x86.html

### Pre-flight

```bash
nvidia-smi
docker --version
docker compose version
```

Confirm GPU visible with >= 80 GB VRAM per GPU (8x H100 for full stack).

### Run

```bash
export NGC_API_KEY=nvapi-...
bash deploy/reason2/vss/demo.sh
```

### What it runs

The demo.sh scaffolds the environment and prints deployment instructions:
1. **Validation** — checks GPU, Docker, NGC_API_KEY
2. **Scaffold** — prints the multi-service stack deployment commands
3. **Sample API call** — shows the REST API pattern for video summarization

For the full stack: follow `docs/recipes/inference/reason2/vss/setup.md` and
https://docs.nvidia.com/vss/latest/content/vss_dep_docker_compose_x86.html

### Success criteria

- VSS API responds at `http://localhost:8100` after Docker Compose deployment
- `POST /files` returns a valid `video_id`
- `POST /summarize` returns a summary string in `choices[0].message.content`
- `/tmp/vss_results.json` is written by demo.sh with status information

## Cosmos Metadata

| Field     | Value                                                                                                                                                                                                     |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                                                                 |
| Domain    | domain:general                                                                                                                                                                                            |
| Technique | technique:reasoning                                                                                                                                                                                       |
| Tags      | inference, reason-2, vss, video-search, summarization, microservices                                                                                                                                     |
| Summary   | Large-scale video search and summarization using Cosmos-Reason2-8B inside the NVIDIA VSS Blueprint — a multi-service stack combining VLM captioning, Graph-RAG, and LLM summarization for Q&A and alerts. |
