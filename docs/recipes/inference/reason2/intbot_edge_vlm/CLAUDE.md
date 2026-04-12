# IntBot Edge VLM — Cosmos Reason 2 Inference

## Model

`nvidia/Cosmos-Reason2-2B`

## Data Source

<!--
  Access: User-supplied camera stream (RTP H.264 over UDP) or local video files
  Dataset: No fixed dataset — live camera input or user-provided video files
  License: NVIDIA Open Model License (model)
-->

**Access:** Gated — requires accepting the NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-2B
**Input:** Live RTP H.264 camera stream (800x600 @ 15 FPS) or local video files
**Dataset:** No fixed dataset — input is provided by the robot's camera system

```bash
# Model (downloaded on x86 host for quantization step)
huggingface-cli download nvidia/Cosmos-Reason2-2B --repo-type model --local-dir ./models/Cosmos-Reason2-2B
```

## Compute Requirements

This recipe requires **Jetson AGX Thor** hardware. It cannot be run on cloud GPU instances.

| Stage | Hardware Required |
|-------|-------------------|
| FP8 quantization + ONNX export | x86 host with NVIDIA GPU |
| TensorRT engine build | Jetson AGX Thor |
| Inference server + camera pipeline | Jetson AGX Thor |

**There is no Brev launchable for this recipe.** Brev provides cloud GPU VMs (x86);
the inference server and camera pipeline must run on the Jetson edge device itself.

## Execution

This recipe uses a multi-step pipeline that spans two hardware platforms:

1. **x86 host** — quantize the model to FP8, export LLM and visual encoder to ONNX
2. **Jetson AGX Thor** — build TensorRT engines from ONNX, run inference server and Robot-VLM Client

No headless demo.sh exists for this recipe. The toolchain (TensorRT-Edge-LLM) must be built from source on the target hardware.

### Pre-flight (x86 host)

```bash
nvidia-smi           # Confirm GPU available for quantization
which tensorrt-edgellm-quantize-llm   # Confirm TensorRT-Edge-LLM toolchain installed
```

### Run (x86 host — quantization and ONNX export)

```bash
# Step 1: Quantize model to FP8
tensorrt-edgellm-quantize-llm \
  --model_dir nvidia/Cosmos-Reason2-2B \
  --output_dir ./quantized/Cosmos-Reason2-2B-fp8 \
  --dtype fp16 \
  --quantization fp8

# Step 2: Export LLM to ONNX
tensorrt-edgellm-export-llm \
  --model_dir ./quantized/Cosmos-Reason2-2B-fp8 \
  --output_dir onnx_models/Cosmos-Reason2-2B-fp8

# Step 3: Export visual encoder to ONNX
tensorrt-edgellm-export-visual \
  --model_dir nvidia/Cosmos-Reason2-2B \
  --output_dir ./onnx_models/Cosmos-Reason2-2B-fp8/visual_enc_onnx \
  --quantization fp8 \
  --dtype fp16
```

### Run (Jetson AGX Thor — engine build and inference)

```bash
# Copy onnx_models/ directory from x86 host to Jetson, then:

# Step 4: Build LLM engine on Jetson
./build/examples/llm/llm_build \
  --onnxDir onnx_models/Cosmos-Reason2-2B-fp8 \
  --engineDir engines/Cosmos-Reason2-2B-fp8 \
  --vlm \
  --minImageTokens 4 \
  --maxImageTokens 10240 \
  --maxInputLen 1024

# Step 5: Build visual encoder engine on Jetson
./build/examples/multimodal/visual_build \
  --onnxDir onnx_models/Cosmos-Reason2-2B-fp8/visual_enc_onnx \
  --engineDir visual_engines/Cosmos-Reason2-2B-fp8

# Step 6: Start inference server and Robot-VLM Client
# (See inference.md for full YAML configuration)
```

### What it runs

End-to-end edge VLM perception pipeline for humanoid robots:
1. **Camera ingestion** — RTP H.264 stream decoding, hardware-accelerated on Jetson
2. **Frame sampling** — intelligent sampling (not every frame) to reduce GPU load
3. **Inference** — Cosmos-Reason2-2B via TensorRT engines (FP8, ~510 ms median latency)
4. **Structured output** — scene description published as NATS events for downstream robot reasoning

### Success criteria

- FP8 quantization completes without OOM on x86 host
- `llm_build` and `visual_build` complete on Jetson AGX Thor
- Inference server responds to HTTP requests at the configured VLM port
- Median end-to-end latency ~510 ms (per the recipe's benchmarks)
- NATS events contain `person_count`, `scene`, `raw_text`, `latency_ms` fields

## Cosmos Metadata

| Field     | Value                                                                                                                                                                                                                        |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                                                                                    |
| Domain    | domain:robotics                                                                                                                                                                                                              |
| Technique | technique:reasoning                                                                                                                                                                                                          |
| Tags      | inference, reason-2, edge, jetson, tensorrt, fp8, vlm, intbot                                                                                                                                                               |
| Summary   | Edge-deployed VLM perception for social humanoid robots using Cosmos-Reason2-2B on Jetson AGX Thor. FP8 quantization reduces inference latency by ~33% (759 ms FP16 → 510 ms FP8), enabling real-time human-robot interaction without cloud dependency. |
