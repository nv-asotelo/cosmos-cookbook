#!/usr/bin/env python3
"""
Cosmos BYO Video Demo setup + launch.
Version: 2026-04-30
Canonical source: .agents/skills/byo-video/scripts/byo_video_setup.py

Runs on the GPU instance. Prints live progress with ETAs.
MODEL_SIZE-driven: downloads all variants for the selected model size.
At the end, prints a clickable OSC 8 hyperlink to the Gradio URL and any
selected companion frontend URL. Gradio is always written to
/tmp/gradio_url.txt for agent capture.

Env vars:
  HF_TOKEN          — required for gated model download (checks ~/.cache/huggingface/token if not set)
  NGC_API_KEY       — required for NIM mode (nvapi-... prefix)
  MODEL_SIZE        — CR1-7B | 2B | 8B | 32B | C3-2B | C3-8B | C3-32B | C3-super | C3-NANO-GEN | C3-SUPER-GEN | PREDICT1-5B | PREDICT1-7B | PREDICT25-2B | PREDICT25-14B | NEM-12B | OMNI-30B | GM-4-31B | QW3-2B | QW3-8B | QW3-32B | QWEN35-35B-A3B  (default: C3-super for nim_local, C3-2B otherwise)
                      C3-NANO-GEN / C3-SUPER-GEN are Cosmos3 OSS *Generators* (diffusion video gen via the
                      NVIDIA/cosmos-framework package; INFERENCE_BACKEND=cosmos3_native). C3-8B and
                      C3-super are Reasoners. In nim_local mode they use the official released
                      nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0 image with NIM_MODEL_SIZE=nano|super.
                      In vLLM/HF fallback mode, setup first converts the Cosmos3 checkpoint through
                      NVIDIA/cosmos-framework's convert_model_to_vlm_safetensors script.
  MODEL_DIR         — override local download path for primary model
  BYO_VIDEO_FRONTEND — nvidia_build | gradio | batch_inference | fiftyone (default: nvidia_build)
  BYO_VIDEO_MOT_TOWER — reasoning | generation | both for future Omni/MoT models. Single-tower
                      VLM/VFM models ignore this because their frontend is inferred from the loaded model.
  GRADIO_PORT       — port for Gradio (default: 7860)
  REASON_VITE_PORT  — port for Cosmos Reason Vite app (default: 5173)
  PREDICT_VITE_PORT — port for Cosmos Predict Vite app (default: 5174)
  BYO_VIDEO_LAUNCH_BATCH_INFERENCE — set to 1 to launch Batch Inference as a companion UI
  BATCH_INFERENCE_PORT — port for Batch Inference frontend (default: 7861; env var name retained for backward compat)
  BATCH_INFERENCE_DATASET — default public HF dataset for Batch Inference (default: pjramg/Safe_Unsafe_Test)
  BYO_VIDEO_HOSTED_API_BASE — optional OpenAI-compatible comparison base URL override (do not put credentials here)
  NVIDIA_HOSTED_BASE_URL — optional comparison base URL override for Vite and Batch Inference (do not put credentials here)
  SKIP_HF_PRELOAD   — set to 1 to skip HF model preload at Gradio startup (auto in vLLM mode)
  VLLM_MAX_MODEL_LEN — max context length for vLLM (default: 32768; do not reduce below 32768 for video)
"""
import os, sys, time, subprocess, re, shutil, json, urllib.request, urllib.parse, urllib.error, socket, tarfile, platform

# ── ANSI helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✓{RESET}  {msg}", flush=True)
def run(msg):    print(f"  {YELLOW}⟳{RESET}  {msg}", flush=True)
def info(msg):   print(f"  {CYAN}→{RESET}  {msg}", flush=True)
def warn(msg):   print(f"  {YELLOW}⚠{RESET}  {msg}", flush=True)
def header(msg, eta=None):
    eta_str = f"  {DIM}[est. {eta}]{RESET}" if eta else ""
    print(f"\n{BOLD}{msg}{RESET}{eta_str}", flush=True)

def hyperlink(url, label=None):
    label = label or url
    return f"\033]8;;{url}\033\\{BOLD}{CYAN}{label}{RESET}\033]8;;\033\\"

def run_cmd(args, cwd=None, env=None, timeout=None):
    try:
        result = subprocess.run(
            args, cwd=cwd, env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        return result.returncode, result.stdout
    except FileNotFoundError:
        return 1, f"command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return 1, "timeout"

def stream_cmd(args, cwd=None, env=None, prefix=""):
    proc = subprocess.Popen(
        args, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"     {DIM}{prefix}{line}{RESET}", flush=True)
    proc.wait()
    return proc.returncode

# ── Size-driven model config (mirrors gradio_cr2_byo.py MODEL_CONFIGS) ───────
_MODEL_CONFIGS = {
    # ── Cosmos3-Reasoner (official NIM by default; converter-backed local fallback) ──
    "C3-2B": {
        "variants": [
            ("C3R-2B BF16", "Cosmos3-Reasoner-2B", "nvidia/Cosmos3-Reasoner-2B-Private", "~TBD"),
        ],
        "nim": None,
    },
    "C3-8B": {
        "variants": [
            # OSS Reasoner (chat VLM) — qwen3_vl architecture, ~16 GB BF16, vLLM-served.
            # Smoke-verified on horde RTX PRO 6000 Blackwell 2026-05-12 (~3m boot, warm cache).
            ("C3R-Nano BF16", "Cosmos3-Nano-Reasoner", "nvidia/Cosmos3-Nano-Reasoner", "~16 GB"),
        ],
        "nim": "cosmos3-reasoner-nano",
        "nim_env": {
            "NIM_MODEL_SIZE": "nano",
            "NIM_SERVED_MODEL_NAME": "nvidia/Cosmos3-Nano-Reasoner",
        },
        "nim_max_wait": 1800,
        "vlm_checkpoint": "Cosmos3-Nano",
        "vlm_output_dirname": "Cosmos3-Nano-VLM",
    },
    "C3-32B": {
        "variants": [
            ("C3R-32B BF16", "Cosmos3-Reasoner-32B", "nvidia/Cosmos3-Reasoner-32B-Private", "~TBD"),
        ],
        "nim": None,
        # disk_gb: 1024 (1TB minimum — Alex explicit requirement for 32B)
        "disk_gb": 1024,
        # vLLM flags: tensor-parallel-size 1 + high memory utilization for single H100 80GB.
        # NOTE: weight size unverified (25 safetensor files, estimate ~60-70GB BF16).
        # If weights exceed ~72GB, OOM will occur — flag for review before production deploy.
        "vllm_extra_flags": ["--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.93"],
    },
    "C3-super": {
        "variants": [
            # OSS Reasoner (chat VLM) — qwen3_vl architecture, ~30B params (13 safetensor
            # shards observed on HF). BF16 footprint ~60 GB; fits an H200 SXM (141 GB) cleanly and
            # an RTX PRO 6000 Blackwell (95 GB) with gpu-memory-utilization 0.85.
            ("C3-Super BF16", "Cosmos3-Super-Reasoner", "nvidia/Cosmos3-Super-Reasoner", "~60 GB"),
        ],
        "nim": "cosmos3-reasoner-super",
        "nim_env": {
            "NIM_MODEL_SIZE": "super",
            "NIM_SERVED_MODEL_NAME": "nvidia/Cosmos3-Super-Reasoner",
        },
        "nim_max_wait": 2400,
        "vlm_checkpoint": "Cosmos3-Super",
        "vlm_output_dirname": "Cosmos3-Super-VLM",
        # ~60 GB weights + venv + HF cache → 256 GB minimum is plenty; keep 1 TB recommendation for safety on multi-model hosts.
        "disk_gb": 256,
        "vllm_extra_flags": ["--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.85"],
    },
    # ── Cosmos3 OSS Generators (diffusion video generation via cosmos-framework) ──
    # These models use NVIDIA/cosmos-framework (NOT vLLM / NOT NIM).
    # Setup path:
    #   git clone https://github.com/NVIDIA/cosmos-framework.git ~/cosmos-framework
    #   cd ~/cosmos-framework && uv sync --all-extras --group=cu130-train
    # Serve path:
    #   python -m cosmos_framework.inference.ray.serve --checkpoint-path Cosmos3-Nano|Cosmos3-Super   # port 8000
    #   python -m cosmos_framework.inference.ray.gradio --host 0.0.0.0 --port 8080                    # UI
    # See scripts/cosmos3_native_launch.sh for the wrapped launcher used by INFERENCE_BACKEND=cosmos3_native.
    "C3-NANO-GEN": {
        "variants": [
            # nvidia/Cosmos3-Nano — public diffusers Cosmos3OmniDiffusersPipeline (t2i/t2v/i2v).
            # Driven by --checkpoint-path Cosmos3-Nano in cosmos_framework inference.
            ("Cosmos3-Nano Generator", "Cosmos3-Nano", "nvidia/Cosmos3-Nano", "~30 GB"),
        ],
        "nim": None,
        "disk_gb": 128,
        "backend_required": "cosmos3_native",
    },
    "C3-SUPER-GEN": {
        "variants": [
            # nvidia/Cosmos3-Super — public diffusers Cosmos3OmniDiffusersPipeline (t2i/t2v/i2v).
            # Larger checkpoint than Nano (53 sibling files vs 35).
            ("Cosmos3-Super Generator", "Cosmos3-Super", "nvidia/Cosmos3-Super", "~60 GB"),
        ],
        "nim": None,
        "disk_gb": 256,
        "backend_required": "cosmos3_native",
    },
    # ── Cosmos Reason1 7B NIM (older generation; frame fallback at runtime) ───
    "CR1-7B": {
        "variants": [
            ("CR1-7B BF16", "Cosmos-Reason1-7B", "nvidia/Cosmos-Reason1-7B", "~14 GB"),
        ],
        "nim": "nvidia/cosmos-reason1-7b",
    },
    # ── Cosmos Predict / Video2World NIMs (Build-style Gradio surface) ───
    "PREDICT1-5B": {
        "variants": [
            ("Cosmos Predict1 5B Video2World", "Cosmos-Predict1-5B-Video2World",
             "nvidia/Cosmos-Predict1-5B-Video2World", "~10 GB"),
        ],
        "nim": "nvidia/cosmos-predict1-5b",
    },
    "PREDICT1-7B": {
        "variants": [
            ("Cosmos Predict1 7B Video2World", "Cosmos-Predict1-7B-Video2World",
             "nvidia/Cosmos-Predict1-7B-Video2World", "~14 GB"),
        ],
        "nim": "nvidia/cosmos-predict1-7b-video2world",
    },
    "PREDICT25-2B": {
        "variants": [
            ("Cosmos Predict2.5 2B", "Cosmos-Predict2.5-2B",
             "nvidia/Cosmos-Predict2.5-2B", "~4 GB"),
        ],
        "nim": "nvidia/cosmos-predict2-5-2b",
    },
    "PREDICT25-14B": {
        "variants": [
            ("Cosmos Predict2.5 14B", "Cosmos-Predict2.5-14B",
             "nvidia/Cosmos-Predict2.5-14B", "~28 GB"),
        ],
        "nim": "nvidia/cosmos-predict2-5-14b",
    },
    # ── Cosmos Reason2 ──
    "2B": {
        "variants": [
            ("CR2-2B BF16", "Cosmos-Reason2-2B",     "nvidia/Cosmos-Reason2-2B",     "~4 GB"),
            ("CR2-2B FP8",  "Cosmos-Reason2-2B-FP8", "nvidia/Cosmos-Reason2-2B-FP8", "~2 GB"),
        ],
        "nim": None,
    },
    "8B": {
        "variants": [
            ("CR2-8B BF16",  "Cosmos-Reason2-8B",       "nvidia/Cosmos-Reason2-8B",       "~16 GB"),
            ("CR2-8B NVFP4", "Cosmos-Reason2-8B-NVFP4", "nvidia/Cosmos-Reason2-8B-NVFP4", "~4 GB"),
        ],
        "nim": "nvidia/cosmos-reason2-8b",
    },
    "32B": {
        "variants": [
            ("CR2-32B BF16", "Cosmos-Reason2-32B",    "nvidia/Cosmos-Reason2-32B",    "~66 GB"),
            ("CR2-32B AV",   "Cosmos-Reason2-32B-AV", "nvidia/Cosmos-Reason2-32B-AV", "~66 GB"),
        ],
        "nim": None,
        # 33B params × 2 bytes BF16 = ~66GB weights. On H100 80GB: use 0.95 utilization.
        # KV cache budget is ~4GB at this utilization — reduce max-model-len accordingly.
        "vllm_extra_flags": ["--gpu-memory-utilization", "0.95"],
        "vllm_max_model_len": 4096,
    },
    # ── Qwen3-VL (public — no HF_TOKEN required) ────────────────────────────────
    # vLLM-only: uses video_url content type with base64 data URLs.
    # Frame control via extra_body mm_processor_kwargs at inference time.
    # Variants: 2B fits 1x H100; 8B fits 1x H100 80GB; 32B needs TP=2 or FP8 on 1x H100.
    "QW3-2B": {
        "variants": [
            ("Qwen3-VL-2B Instruct",    "Qwen3-VL-2B-Instruct",     "Qwen/Qwen3-VL-2B-Instruct",     "~5 GB"),
            ("Qwen3-VL-2B FP8",         "Qwen3-VL-2B-Instruct-FP8", "Qwen/Qwen3-VL-2B-Instruct-FP8", "~3 GB"),
            ("Qwen3-VL-2B Thinking",    "Qwen3-VL-2B-Thinking",     "Qwen/Qwen3-VL-2B-Thinking",     "~5 GB"),
        ],
        "nim": None,
        "vllm_extra_flags": ["--gpu-memory-utilization", "0.85", "--allowed-local-media-path", "/tmp"],
        "vllm_max_model_len": 32768,
    },
    "QW3-8B": {
        "variants": [
            ("Qwen3-VL-8B Instruct",    "Qwen3-VL-8B-Instruct",     "Qwen/Qwen3-VL-8B-Instruct",     "~16 GB"),
            ("Qwen3-VL-8B FP8",         "Qwen3-VL-8B-Instruct-FP8", "Qwen/Qwen3-VL-8B-Instruct-FP8", "~9 GB"),
            ("Qwen3-VL-8B Thinking",    "Qwen3-VL-8B-Thinking",     "Qwen/Qwen3-VL-8B-Thinking",     "~16 GB"),
        ],
        "nim": None,
        "vllm_extra_flags": ["--gpu-memory-utilization", "0.85", "--allowed-local-media-path", "/tmp"],
        "vllm_max_model_len": 32768,
    },
    "QW3-32B": {
        "variants": [
            ("Qwen3-VL-32B Instruct",   "Qwen3-VL-32B-Instruct",     "Qwen/Qwen3-VL-32B-Instruct",     "~64 GB"),
            ("Qwen3-VL-32B FP8",        "Qwen3-VL-32B-Instruct-FP8", "Qwen/Qwen3-VL-32B-Instruct-FP8", "~33 GB"),
            ("Qwen3-VL-32B Thinking",   "Qwen3-VL-32B-Thinking",     "Qwen/Qwen3-VL-32B-Thinking",     "~64 GB"),
        ],
        "nim": None,
        "vllm_extra_flags": ["--gpu-memory-utilization", "0.93", "--allowed-local-media-path", "/tmp"],
        "vllm_max_model_len": 16384,
    },
    # ── Qwen3.5 NIM-only sprint targets ───────────────────────────────────────
    "QWEN35-35B-A3B": {
        "variants": [
            ("Qwen3.5 35B A3B NIM", "qwen3.5-35b-a3b", "qwen/qwen3.5-35b-a3b", "~40 GB"),
        ],
        "nim": "qwen/qwen3.5-35b-a3b",
        "nim_max_wait": 2400,
    },
    # ── Nemotron-3 Nano Omni and Gemma NIM-only sprint targets ─────────────────
    "OMNI-30B": {
        "variants": [
            ("Nem3-Omni-30B BF16", "Nemotron-3-Nano-Omni-30B-A3B-Reasoning",
             "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "~60 GB"),
        ],
        "nim": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "nim_max_wait": 2400,
    },
    "GM-4-31B": {
        "variants": [
            ("Gemma-4-31B IT", "gemma-4-31b-it", "google/gemma-4-31b-it", "~62 GB"),
        ],
        "nim": "google/gemma-4-31b-it",
        "nim_env": {"NIM_MAX_MODEL_LEN": "131072"},
        "nim_max_wait": 2400,
    },
    # ── Nemotron-Nano-12B-v2-VL (gated — HF_TOKEN with nvidia org required) ──────
    # Requires vLLM nightly or compatible build; PyPI vLLM ≤0.11.0 unsupported.
    # Uses opencv video backend (not PyAV). Frontends send video as a base64 data URL.
    # NVFP4-QAD variant requires special vLLM build — use BF16 or FP8 for demos.
    "NEM-12B": {
        "variants": [
            ("Nem-12B BF16", "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "~26 GB"),
            ("Nem-12B FP8",  "NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",  "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",  "~13 GB"),
        ],
        "nim": None,
        "vllm_extra_flags": [
            "--media-io-kwargs", '{"video": {"fps": 2, "num_frames": 128}}',
            "--video-pruning-rate", "0.75",
            "--allowed-local-media-path", "/tmp",
        ],
        "vllm_max_model_len": 32768,
        # VLLM_VIDEO_LOADER_BACKEND=opencv: required for Nemotron video decoding.
        # Nemotron's vLLM integration does not support PyAV; opencv is the only supported backend.
        # FLASHINFER_DISABLE_VERSION_CHECK=1: bypasses mismatch between flashinfer Python package
        # (0.5.3) and flashinfer-cubin (0.6.8.post1) installed by cosmos-reason2 uv sync.
        "vllm_env": {
            "VLLM_VIDEO_LOADER_BACKEND": "opencv",
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
    },
    # ── Alpamayo 1.5 VQA/captioning through the local BYO OpenAI adapter ─────
    "ALPAMAYO": {
        "variants": [
            ("Alpamayo 1.5 10B", "Alpamayo-1.5-10B", "nvidia/Alpamayo-1.5-10B", "~22 GB"),
        ],
        "nim": None,
        "hf_auth_required": True,
        "backend_required": "alpamayo",
        "disk_gb": 128,
    },
}

# ── Config ──────────────────────────────────────────────────────────────────
HOME          = os.path.expanduser("~")
PATH_EXTRA    = f"{HOME}/.local/bin:{HOME}/.cargo/bin"
# HF_HOME=/tmp/hf-home: avoids root-owned ~/.cache/huggingface/ on Brev/shadeform instances.
# byo_video_setup.py may run as a non-root user where ~/.cache is owned by root (prior root
# invocation). Redirecting to /tmp/hf-home ensures write access for model cache, frpc binary,
# and HF modules. Set early so all downstream ENV copies inherit it.
HF_HOME_DIR   = os.environ.get("HF_HOME", "/tmp/hf-home")
os.makedirs(HF_HOME_DIR, exist_ok=True)
ENV           = {**os.environ, "PATH": f"{PATH_EXTRA}:{os.environ.get('PATH', '')}",
                 "PYTHONUNBUFFERED": "1",
                 "HF_HOME": HF_HOME_DIR,
                 "HF_MODULES_CACHE": f"{HF_HOME_DIR}/modules"}
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
NGC_API_KEY   = os.environ.get("NGC_API_KEY", "")
_INFERENCE_BACKEND_ENV = os.environ.get("INFERENCE_BACKEND")
_INFERENCE_BACKEND_RAW = (_INFERENCE_BACKEND_ENV or "hf").lower()
_DEFAULT_MODEL_SIZE = (
    "ALPAMAYO" if _INFERENCE_BACKEND_RAW == "alpamayo"
    else "C3-super" if _INFERENCE_BACKEND_RAW == "nim_local"
    else "C3-2B"
)
MODEL_SIZE    = os.environ.get("MODEL_SIZE", _DEFAULT_MODEL_SIZE).upper()
# .upper() normalises input but breaks mixed-case keys. Remap known exceptions.
_MODEL_SIZE_FIX = {"C3-SUPER": "C3-super", "C3-NANO": "C3-8B", "C3-NANO-REASONER": "C3-8B"}
MODEL_SIZE = _MODEL_SIZE_FIX.get(MODEL_SIZE, MODEL_SIZE)
# Working directory is resolved after MODEL_SIZE, because Cosmos3 generators use
# NVIDIA/cosmos-framework while reasoners still use cosmos-reason2.
GRADIO_PORT   = int(os.environ.get("GRADIO_PORT", "7860"))
FRONTEND      = os.environ.get("BYO_VIDEO_FRONTEND", "nvidia_build").strip().lower()
if FRONTEND == "agent":
    FRONTEND = "batch_inference"
if FRONTEND in ("build", "build_nvidia", "nvidia-build", "nvidia_build_playground"):
    FRONTEND = "nvidia_build"
BATCH_INFERENCE_PORT = int(os.environ.get("BATCH_INFERENCE_PORT", "7861"))
BATCH_INFERENCE_APP  = "/tmp/byo_video_batch_inference.py"
HOSTED_COMPARE_HELPER = "/tmp/hosted_model_compare.py"
BATCH_INFERENCE_LOG_FILE = "/tmp/byo_video_batch_inference.log"
BATCH_INFERENCE_URL_FILE = "/tmp/byo_video_batch_inference_url.txt"
BATCH_INFERENCE_LIVE_FLAG = "/tmp/byo_video_batch_inference_live.flag"
NIM_SWITCH_PORT = int(os.environ.get("NIM_SWITCH_PORT", "7862"))
NIM_SWITCH_SERVICE = os.environ.get("NIM_SWITCH_SERVICE", "/tmp/nim_switch_service.py")
NIM_SWITCH_URL_FILE = "/tmp/nim_switch_url.txt"
REASON_VITE_PORT = int(os.environ.get("REASON_VITE_PORT", os.environ.get("PORT", "5173")))
REASON_VITE_APP_DIR = os.environ.get("REASON_VITE_APP_DIR", "/tmp/nvidia-build-reason-vite")
REASON_VITE_URL_FILE = "/tmp/nvidia_build_reason_vite_url.txt"
REASON_VITE_LIVE_FLAG = "/tmp/nvidia_build_reason_vite_live.flag"
PREDICT_VITE_PORT = int(os.environ.get("PREDICT_VITE_PORT", "5174"))
PREDICT_VITE_APP_DIR = os.environ.get("PREDICT_VITE_APP_DIR", "/tmp/nvidia-build-predict-vite")
PREDICT_VITE_URL_FILE = "/tmp/nvidia_build_predict_vite_url.txt"
PREDICT_VITE_LIVE_FLAG = "/tmp/nvidia_build_predict_vite_live.flag"
NODE_HOME = os.environ.get("NODE_HOME", f"{HOME}/.local/node-v20")
LAUNCH_BATCH_INFERENCE_COMPANION = os.environ.get(
    "BYO_VIDEO_LAUNCH_BATCH_INFERENCE",
    os.environ.get("LAUNCH_BATCH_INFERENCE", "0"),
).strip().lower() in {"1", "true", "yes", "on"}
URL_FILE      = "/tmp/gradio_url.txt"
LOG_FILE      = "/tmp/gradio_demo.log"
# MAXLEN-001: 32768 is the minimum required for video queries. Do not reduce below this.
VLLM_MAX_MODEL_LEN   = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))
INFERENCE_BACKEND    = _INFERENCE_BACKEND_RAW
# VRAM flags — set during GPU detection; declare defaults here
ULTRA_LOW_VRAM = False
LOW_VRAM       = False
# Cost tracking
BREV_RATE_PER_HOUR = float(os.environ.get("BREV_RATE_PER_HOUR", "0"))
SETUP_START        = time.time()

def credits_spent():
    if BREV_RATE_PER_HOUR <= 0:
        return ""
    elapsed = time.time() - SETUP_START
    cost = BREV_RATE_PER_HOUR * elapsed / 3600
    return f" | Credits: ${cost:.3f}"

if MODEL_SIZE not in _MODEL_CONFIGS:
    print(f"  ✗  MODEL_SIZE={MODEL_SIZE} not supported. Use ALPAMAYO, CR1-7B, C3-2B, C3-8B, C3-32B, C3-super, C3-NANO-GEN, C3-SUPER-GEN, 2B, 8B, 32B, PREDICT1-5B, PREDICT1-7B, PREDICT25-2B, PREDICT25-14B, NEM-12B, OMNI-30B, GM-4-31B, QW3-2B, QW3-8B, QW3-32B, or QWEN35-35B-A3B.")
    sys.exit(1)

_cfg = _MODEL_CONFIGS[MODEL_SIZE]
_backend_required = _cfg.get("backend_required")
if _backend_required:
    if not _INFERENCE_BACKEND_ENV:
        INFERENCE_BACKEND = _backend_required
    elif _INFERENCE_BACKEND_RAW not in {_backend_required, "nim_local"}:
        warn(
            f"MODEL_SIZE={MODEL_SIZE} requires INFERENCE_BACKEND={_backend_required}; "
            f"overriding explicit INFERENCE_BACKEND={_INFERENCE_BACKEND_RAW!r}."
        )
        INFERENCE_BACKEND = _backend_required
    else:
        INFERENCE_BACKEND = _INFERENCE_BACKEND_RAW

if INFERENCE_BACKEND != _INFERENCE_BACKEND_RAW:
    _INFERENCE_BACKEND_RAW = INFERENCE_BACKEND
    os.environ["INFERENCE_BACKEND"] = INFERENCE_BACKEND
    ENV["INFERENCE_BACKEND"] = INFERENCE_BACKEND

# Cosmos3-Reasoner uses cosmos-reason2, Cosmos3 Generator uses
# NVIDIA/cosmos-framework, and Alpamayo uses its own repo checkout. Set
# COSMOS_DIR, COSMOS3_DIR, or ALPAMAYO_DIR to override the relevant checkout.
if INFERENCE_BACKEND == "alpamayo":
    _DEFAULT_WORK_DIR = f"{HOME}/amo-1-5"
    REASON2_DIR = os.environ.get("COSMOS_DIR", os.environ.get("ALPAMAYO_DIR", _DEFAULT_WORK_DIR))
elif INFERENCE_BACKEND == "cosmos3_native":
    _DEFAULT_WORK_DIR = f"{HOME}/cosmos-framework"
    REASON2_DIR = os.environ.get("COSMOS_DIR", os.environ.get("COSMOS3_DIR", _DEFAULT_WORK_DIR))
    os.environ["COSMOS3_DIR"] = REASON2_DIR
    ENV["COSMOS3_DIR"] = REASON2_DIR
else:
    _DEFAULT_WORK_DIR = f"{HOME}/cosmos-reason2"
    REASON2_DIR = os.environ.get("COSMOS_DIR", _DEFAULT_WORK_DIR)

MODELS_BASE = f"{REASON2_DIR}/models"
HF_AUTH_REQUIRED = _cfg.get("hf_auth_required", not _cfg.get("hf_public", False))

# Primary variant (first in list) drives MODEL_DIR/MODEL_NAME defaults
_primary_label, _primary_dirname, _primary_hf_id, _ = _cfg["variants"][0]
MODEL_DIR  = os.environ.get("MODEL_DIR",  f"{MODELS_BASE}/{_primary_dirname}")
MODEL_NAME = _primary_hf_id

# ── MODEL_ID override (arbitrary HF model, bypasses MODEL_SIZE lookup) ───────
MODEL_ID = os.environ.get("MODEL_ID", "")
if MODEL_ID:
    MODEL_NAME = MODEL_ID
    # Only fall back to "custom" if MODEL_SIZE wasn't explicitly set to a known config key.
    # Prevents MODEL_ID from silently discarding a valid MODEL_SIZE (e.g. NEM-12B, C3-8B).
    if MODEL_SIZE not in _MODEL_CONFIGS:
        # Unknown size: derive directory name from MODEL_ID. Keep hyphens — Linux supports them.
        _model_dir_name = MODEL_ID.split("/")[-1]
        MODEL_DIR  = os.path.join(MODELS_BASE, _model_dir_name)
        MODEL_SIZE = "custom"
        # Estimate size hint from model ID
        _size_hint = "~8GB" if "8B" in MODEL_ID else "~16GB" if ("32B" in MODEL_ID or "14B" in MODEL_ID) else "~4GB"
        # Override _cfg to a minimal single-variant config
        _cfg = {
            "repo": MODEL_ID.split("/")[-1],
            "variants": [(_model_dir_name, _model_dir_name, MODEL_ID, _size_hint)],
            "nim": False,
        }
        _variant_labels = MODEL_ID
    # When MODEL_SIZE is a known config key: MODEL_DIR already set correctly from config.
    # Only MODEL_NAME is updated to the explicit MODEL_ID override.

# ── Dashboard: 9-step progress checklist ─────────────────────────────────────
_REPO_STEP_LABEL = {
    "alpamayo": "Alpamayo 1.5 repo",
    "cosmos3_native": "cosmos-framework repo",
}.get(INFERENCE_BACKEND, "cosmos-reason2 repo")
STEP_LABELS = [
    "GPU detect + VRAM tier",
    "HF auth + token validate",
    "NGC API key",
    "uv install",
    _REPO_STEP_LABEL,
    "uv sync + CUDA libs",
    "PyAV + frontend deps",
    "Model weights download",
    "Frontend launch",
]
STEPS_DONE = []

def print_dashboard():
    print("\n── Setup Progress ──────────────────────────────", flush=True)
    elapsed = int(time.time() - SETUP_START)
    print(f"  Elapsed: {elapsed//60}m {elapsed%60}s{credits_spent()}", flush=True)
    for i, label in enumerate(STEP_LABELS, 1):
        if i in STEPS_DONE:
            marker = "✅"
        elif i == (max(STEPS_DONE) + 1 if STEPS_DONE else 1):
            marker = "⟳ "
        else:
            marker = "—"
        print(f"  [{marker}] Step {i}: {label}", flush=True)
    print("────────────────────────────────────────────────\n", flush=True)

print_dashboard()

# ── Pre-step: kill old frontends so VRAM measurement is accurate ─────────────
subprocess.run(["bash", "-c", f"fuser -k {GRADIO_PORT}/tcp 2>/dev/null || true"])
subprocess.run(["bash", "-c", f"fuser -k {BATCH_INFERENCE_PORT}/tcp 2>/dev/null || true"])
time.sleep(2)

# ── Step 1: GPU check ─────────────────────────────────────────────────────────
header("Step 1 — GPU detect", eta="<5s")
NVIDIA_SMI_RETRIES = 3
_smi_rc = 1
_smi_out = ""
for _attempt in range(NVIDIA_SMI_RETRIES):
    _smi_rc, _smi_out = run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
         "--format=csv,noheader,nounits"]
    )
    if _smi_rc == 0 and _smi_out.strip():
        break
    if _attempt < NVIDIA_SMI_RETRIES - 1:
        warn(f"nvidia-smi attempt {_attempt + 1} failed — retrying in 5s")
        time.sleep(5)
else:
    if _smi_rc != 0 or not _smi_out.strip():
        warn("nvidia-smi failed after 3 attempts — defaulting to LOW_VRAM tier")
        gpu_name   = "unknown"
        vram_free  = 0
        vram_total = 0
        LOW_VRAM   = True
        tier_name  = "low-VRAM (fallback)"
        gradio_fps = 4
        max_pixels = 131072
        prefill_tps = 15

if _smi_rc == 0 and _smi_out.strip():
    gpu_line   = _smi_out.strip().splitlines()[0]
    parts_     = [p.strip() for p in gpu_line.split(",")]
    gpu_name   = parts_[0]
    try:
        vram_free  = int(parts_[1].split()[0])
        vram_total = int(parts_[2].split()[0])
    except (IndexError, ValueError):
        warn(
            "nvidia-smi did not report discrete GPU memory "
            f"({gpu_line!r}); using LOW_VRAM fallback tier."
        )
        vram_free  = 0
        vram_total = 0
        LOW_VRAM   = True
        tier_name  = "low-VRAM (memory query unavailable)"
        gradio_fps = 4
        max_pixels = 131072
        prefill_tps = 15

    _gpu_upper  = gpu_name.upper()
    _h100_class = any(tag in _gpu_upper for tag in ("H100", "A100", "H200", "GB200"))
    if vram_total <= 0:
        pass
    elif _h100_class and vram_free >= 60000:
        tier_name   = "H100/A100"
        gradio_fps  = 8
        max_pixels  = 1048576
        prefill_tps = 90
    elif vram_free >= 40000:
        tier_name   = "high-VRAM"
        gradio_fps  = 8
        max_pixels  = 524288
        prefill_tps = 23
    elif vram_free >= 8000:
        tier_name   = "low-VRAM (<40GB)"
        gradio_fps  = 4
        max_pixels  = 131072
        prefill_tps = 15
        LOW_VRAM    = True
    else:
        LOW_VRAM       = True
        ULTRA_LOW_VRAM = True
        gradio_fps     = 1
        max_pixels     = 65536
        prefill_tps    = 10
        tier_name      = "ULTRA-LOW-VRAM"

if ULTRA_LOW_VRAM:
    warn("RTX consumer GPU detected (<8GB VRAM free). Demo will run but may OOM on videos >5s at 720p. Pre-resize to 360p before upload.")

if not MODEL_ID:
    _variant_labels = " → ".join(lbl for lbl, _, _, _ in _cfg["variants"])
    if _cfg.get("nim"):
        _variant_labels += f" → NIM-{MODEL_SIZE}"
else:
    _variant_labels = MODEL_ID

_REASONING_MODEL_SIZES = {
    "ALPAMAYO",
    "CR1-7B", "2B", "8B", "32B",
    "C3-2B", "C3-8B", "C3-32B", "C3-super",
    "NEM-12B", "OMNI-30B", "GM-4-31B",
    "QW3-2B", "QW3-8B", "QW3-32B", "QWEN35-35B-A3B",
}
_GENERATION_MODEL_SIZES = {
    "PREDICT1-5B", "PREDICT1-7B", "PREDICT25-2B", "PREDICT25-14B",
    "C3-NANO-GEN", "C3-SUPER-GEN",
}
_TOWER_ALIASES = {
    "reason": "reasoning",
    "reasoner": "reasoning",
    "reasoning": "reasoning",
    "vlm": "reasoning",
    "understanding": "reasoning",
    "chat": "reasoning",
    "generate": "generation",
    "generator": "generation",
    "generation": "generation",
    "gen": "generation",
    "vfm": "generation",
    "predict": "generation",
    "video2world": "generation",
    "both": "both",
    "all": "both",
}

def _infer_model_towers(model_size, *names):
    """Return the MoT tower(s) the loaded model can actually serve."""
    tokens = " ".join(str(n or "") for n in (model_size, *names)).lower()
    if any(token in tokens for token in ("predict", "video2world", "text2world", "generator")):
        return {"generation"}
    if any(token in tokens for token in ("reason", "reasoner", "vlm", "qwen", "nemotron", "gemma", "alpamayo")):
        return {"reasoning"}
    if any(token in tokens for token in ("cosmos3-nano", "cosmos3-super", "cosmos-3-nano", "cosmos-3-super")):
        return {"generation"}
    if "omni" in tokens:
        return {"reasoning", "generation"}
    if model_size in _REASONING_MODEL_SIZES:
        return {"reasoning"}
    if model_size in _GENERATION_MODEL_SIZES:
        return {"generation"}
    if INFERENCE_BACKEND == "cosmos3_native":
        return {"generation"}
    return {"reasoning"}

def _requested_towers_from_env():
    raw = (
        os.environ.get("BYO_VIDEO_MOT_TOWER")
        or os.environ.get("COSMOS3_TOWER")
        or os.environ.get("COSMOS3_FRONTEND_TOWER")
        or ""
    ).strip().lower()
    if not raw:
        return None
    requested = set()
    for part in re.split(r"[,/+ ]+", raw):
        if not part:
            continue
        tower = _TOWER_ALIASES.get(part)
        if not tower:
            print(
                f"  ✗  BYO_VIDEO_MOT_TOWER={raw!r} is not supported. "
                "Use reasoning, generation, or both."
            )
            sys.exit(1)
        if tower == "both":
            requested.update({"reasoning", "generation"})
        else:
            requested.add(tower)
    return requested or None

def _select_frontend_towers(available):
    requested = _requested_towers_from_env()
    if requested:
        unsupported = requested - available
        if unsupported:
            available_label = ", ".join(sorted(available))
            requested_label = ", ".join(sorted(requested))
            print(
                f"  ✗  Requested frontend tower(s) {requested_label} do not match "
                f"the loaded model capability ({available_label})."
            )
            sys.exit(1)
        return requested

    if available == {"reasoning", "generation"}:
        print("  ✗  This Omni/MoT model can expose multiple towers.")
        print("     Ask the user which use cases they want to see: generation, reasoning, or both.")
        print("     Then set BYO_VIDEO_MOT_TOWER=reasoning|generation|both and rerun setup.")
        sys.exit(1)

    return set(available)

def _tower_label(towers):
    if towers == {"reasoning"}:
        return "VLM / Reasoner"
    if towers == {"generation"}:
        return "VFM / Generator"
    return "Omni MoT / Reasoner + Generator"

MODEL_TOWERS = _infer_model_towers(MODEL_SIZE, MODEL_ID, MODEL_NAME, _variant_labels)
FRONTEND_TOWERS = _select_frontend_towers(MODEL_TOWERS)

def _is_cosmos3_reasoner(model_size, towers, *names):
    if "reasoning" not in towers or "generation" in towers:
        return False
    size = str(model_size or "").lower()
    tokens = " ".join(str(n or "") for n in names).lower()
    return (
        size in {"c3-2b", "c3-8b", "c3-32b", "c3-super"}
        or "cosmos3" in tokens
        or "cosmos-3" in tokens
    )

def _build_playground_app(model_size, towers, *names):
    tokens = " ".join(str(n or "") for n in (model_size, *names)).lower()
    if "alpamayo" in tokens:
        return "/tmp/gradio_cr2_byo.py", "Alpamayo BYO-video Gradio"
    if "generation" in towers and "reasoning" not in towers:
        return "/tmp/gradio_cosmos_predict.py", "Cosmos Predict Build-style playground"
    if "reasoning" in towers and "generation" not in towers:
        if _is_cosmos3_reasoner(model_size, towers, *names):
            return "/tmp/gradio_cr2_byo.py", "Cosmos3 classic BYO-video Gradio"
        return "/tmp/gradio_cosmos_reason_build.py", "Cosmos Reason Build-style playground"
    if "predict" in tokens or "video2world" in tokens or "text2world" in tokens:
        return "/tmp/gradio_cosmos_predict.py", "Cosmos Predict Build-style playground"
    return "/tmp/gradio_cr2_byo.py", "Cosmos Build-style BYO-video Gradio"

if FRONTEND == "nvidia_build":
    GRADIO_APP, _GRADIO_APP_LABEL = _build_playground_app(
        MODEL_SIZE, FRONTEND_TOWERS, MODEL_ID, MODEL_NAME, _variant_labels
    )
else:
    GRADIO_APP = os.environ.get("GRADIO_APP", "/tmp/gradio_cr2_byo.py")
    _GRADIO_APP_LABEL = "Cosmos Build-style BYO-video Gradio"

USE_REASON_VITE = (
    FRONTEND == "nvidia_build"
    and "reasoning" in FRONTEND_TOWERS
    and INFERENCE_BACKEND in {"vllm", "nim_local", "alpamayo"}
)
USE_PREDICT_VITE = (
    FRONTEND == "nvidia_build"
    and "generation" in FRONTEND_TOWERS
)
if USE_REASON_VITE and USE_PREDICT_VITE:
    _GRADIO_APP_LABEL = f"Cosmos Omni Gradio fallback for {MODEL_SIZE}"
elif USE_REASON_VITE:
    if GRADIO_APP.endswith("gradio_cr2_byo.py") and _is_cosmos3_reasoner(
        MODEL_SIZE, FRONTEND_TOWERS, MODEL_ID, MODEL_NAME, _variant_labels
    ):
        _GRADIO_APP_LABEL = f"Cosmos3 classic Gradio fallback for {MODEL_SIZE}"
    else:
        _GRADIO_APP_LABEL = f"Cosmos Reason Gradio fallback for {MODEL_SIZE}"
elif USE_PREDICT_VITE:
    _GRADIO_APP_LABEL = f"Cosmos Predict Gradio fallback for {MODEL_SIZE}"

ok(f"{gpu_name}  {vram_free:,} MiB free / {vram_total:,} MiB total")
ok(f"MODEL_SIZE: {MODEL_SIZE}  |  variants: {_variant_labels}")
ok(f"Model frontend capability: {_tower_label(MODEL_TOWERS)}")
ok(f"Serving frontend tower: {_tower_label(FRONTEND_TOWERS)}")
ok(f"Frontend app: {_GRADIO_APP_LABEL} ({GRADIO_APP})")
if USE_REASON_VITE or USE_PREDICT_VITE:
    if USE_REASON_VITE:
        ok(f"Primary Reason Vite app: {REASON_VITE_APP_DIR} on port {REASON_VITE_PORT}")
    if USE_PREDICT_VITE:
        ok(f"Primary Predict Vite app: {PREDICT_VITE_APP_DIR} on port {PREDICT_VITE_PORT}")
if USE_PREDICT_VITE:
    info("Predict Vite will use the generation/Ray backend selected for this model.")
if LAUNCH_BATCH_INFERENCE_COMPANION:
    ok(f"Companion Batch Inference UI requested on port {BATCH_INFERENCE_PORT}")
ok(f"VRAM tier: {tier_name}  |  fps={gradio_fps}, max_pixels={max_pixels:,}, prefill_tps={prefill_tps}")
STEPS_DONE.append(1)
print_dashboard()

def hf_repo_access_without_token(model_id):
    if not model_id or model_id.startswith("/") or os.path.exists(model_id):
        return True, "local model path"
    repo = urllib.parse.quote(model_id, safe="/")
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
            if payload.get("private") is True:
                return False, "HF API reports private=true"
            return True, f"HF API HTTP {resp.status}"
    except urllib.error.HTTPError as err:
        if err.code in (401, 403, 404):
            return False, f"HF API HTTP {err.code}"
        return None, f"HF API HTTP {err.code}"
    except Exception as err:
        return None, str(err)

# ── Step 2: HF token ──────────────────────────────────────────────────────────
header("Step 2 — HuggingFace auth", eta="<5s")
hf_cache = os.path.expanduser("~/.cache/huggingface/token")
if INFERENCE_BACKEND == "nim_local":
    info("nim_local backend — HF auth not required (NIM container ships the model). Skipping Step 2/2b.")
elif HF_TOKEN:
    ok(f"HF_TOKEN set ({len(HF_TOKEN)} chars)")
elif os.path.exists(hf_cache):
    ok(f"HF token found at ~/.cache/huggingface/token")
    with open(hf_cache) as f:
        HF_TOKEN = f.read().strip()
    ENV["HF_TOKEN"] = HF_TOKEN
else:
    public_access, public_msg = hf_repo_access_without_token(MODEL_NAME)
    if public_access:
        HF_AUTH_REQUIRED = False
        info(f"{MODEL_NAME} is reachable without HF_TOKEN ({public_msg})")
    elif HF_AUTH_REQUIRED or public_access is False:
        print("  ✗  HF_TOKEN not set and no cached token found.")
        print(f"     {MODEL_NAME} is not reachable anonymously ({public_msg}).")
        print("     Run: export HF_TOKEN=hf_... and re-run this script.")
        sys.exit(1)
    else:
        warn(f"Could not verify anonymous HF access ({public_msg}) — continuing without token")

# ── Step 2b: Validate HF token ────────────────────────────────────────────────
if INFERENCE_BACKEND != "nim_local" and HF_TOKEN:
    header("Step 2b — Validate HF token", eta="<2s")
    _hf_req = urllib.request.Request(
        "https://huggingface.co/api/whoami",
        headers={"Authorization": f"Bearer {HF_TOKEN}"}
    )
    try:
        with urllib.request.urlopen(_hf_req, timeout=10) as _resp:
            if _resp.status == 200:
                ok("HF token valid")
            else:
                print(f"  ✗  HF token returned HTTP {_resp.status} — run 'huggingface-cli login' on this instance")
                sys.exit(1)
    except Exception as _hf_err:
        warn(f"HF token check failed ({_hf_err}) — continuing, will fail at download if token is bad")
elif INFERENCE_BACKEND != "nim_local":
    info("No HF token to validate for public model.")

STEPS_DONE.append(2)
print_dashboard()

# ── Step 3: NGC API key check (for NIM mode) ──────────────────────────────────
header("Step 3 — NGC API key (NIM mode)", eta="<5s")
# nim_local backend = local Docker container (this sprint); _cfg["nim"] = NVCF cloud catalog (NIM hosted)
if INFERENCE_BACKEND == "nim_local":
    if not NGC_API_KEY:
        print("  ✗  INFERENCE_BACKEND=nim_local requires NGC_API_KEY (export NGC_API_KEY=nvapi-...)"); sys.exit(1)
    if not NGC_API_KEY.startswith("nvapi-"):
        warn("NGC_API_KEY does not start with 'nvapi-' — docker pull nvcr.io may fail")
    ok(f"NGC_API_KEY set ({len(NGC_API_KEY)} chars) — nim_local Docker mode enabled")
elif _cfg["nim"]:
    if NGC_API_KEY:
        if NGC_API_KEY.startswith("nvapi-"):
            ok(f"NGC_API_KEY set ({len(NGC_API_KEY)} chars, nvapi- prefix) — NIM-{MODEL_SIZE} enabled")
        else:
            warn(f"NGC_API_KEY set but does not start with 'nvapi-' — NIM calls may fail")
    else:
        info(f"NGC_API_KEY not set — NIM-{MODEL_SIZE} will be skipped in Gradio UI")
        info("To enable NIM: export NGC_API_KEY=nvapi-...")
else:
    info(f"MODEL_SIZE={MODEL_SIZE} has no NIM endpoint in catalog — NIM step skipped")
STEPS_DONE.append(3)

# ── Step 4: uv ────────────────────────────────────────────────────────────────
header("Step 4 — uv package manager", eta="<5s if cached, ~10s first time")
rc, _ = run_cmd(["uv", "--version"], env=ENV)
if rc == 0:
    _, ver = run_cmd(["uv", "--version"], env=ENV)
    ok(f"uv already installed ({ver.strip()})")
else:
    run("Installing uv  (~10s)")
    t0 = time.time()
    rc = stream_cmd(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], env=ENV)
    if rc != 0:
        print("  ✗  uv install failed"); sys.exit(1)
    ok(f"uv installed in {time.time()-t0:.0f}s")
STEPS_DONE.append(4)

# ── Step 5: model repo ───────────────────────────────────────────────────────
if INFERENCE_BACKEND == "alpamayo":
    _repo_label = "Alpamayo 1.5 repo"
    _repo_url = "https://github.com/NVlabs/alpamayo1.5.git"
elif INFERENCE_BACKEND == "cosmos3_native":
    _repo_label = "cosmos-framework repo"
    _repo_url = "https://github.com/NVIDIA/cosmos-framework.git"
else:
    _repo_label = "cosmos-reason2 repo"
    _repo_url = "https://github.com/nvidia-cosmos/cosmos-reason2.git"

header(f"Step 5 — {_repo_label}", eta="<5s if cached, ~15s first time")
_repo_ready = os.path.exists(f"{REASON2_DIR}/.git")
if INFERENCE_BACKEND == "cosmos3_native":
    _repo_ready = _repo_ready and os.path.isdir(f"{REASON2_DIR}/cosmos_framework")
if _repo_ready or (
    INFERENCE_BACKEND == "alpamayo"
    and os.path.exists(f"{REASON2_DIR}/pyproject.toml")
    and os.path.isdir(f"{REASON2_DIR}/src/alpamayo1_5")
):
    ok(f"{_repo_label} already cloned at {REASON2_DIR}")
else:
    if os.path.exists(REASON2_DIR) and os.listdir(REASON2_DIR):
        print(f"  ✗  {REASON2_DIR} exists but is not a usable {_repo_label}.")
        print("     Set COSMOS_DIR/COSMOS3_DIR/ALPAMAYO_DIR to an empty path or move the existing directory.")
        sys.exit(1)
    run(f"Cloning {_repo_label}  (~15s)")
    t0 = time.time()
    rc = stream_cmd(
        ["git", "clone", _repo_url, REASON2_DIR],
        env=ENV
    )
    if rc != 0:
        print("  ✗  git clone failed"); sys.exit(1)
    ok(f"Cloned in {time.time()-t0:.0f}s")
STEPS_DONE.append(5)

# ── Step 6: Python dependencies ───────────────────────────────────────────────
header("Step 6 — Python dependencies (uv sync)", eta="<5s if cached, ~2-3 min first time")
venv_marker = f"{REASON2_DIR}/.venv/lib"
if os.path.exists(venv_marker):
    if INFERENCE_BACKEND == "alpamayo":
        ENV["UV_NO_SYNC"] = "1"
        os.environ["UV_NO_SYNC"] = "1"
    if INFERENCE_BACKEND == "cosmos3_native":
        ENV["LD_LIBRARY_PATH"] = ""
        os.environ["LD_LIBRARY_PATH"] = ""
    ok("virtualenv already present — skipping uv sync")
else:
    _extras = os.environ.get("COSMOS_EXTRAS", "cu128")
    _cosmos3_group = os.environ.get("COSMOS3_UV_GROUP", "cu130-train")
    if INFERENCE_BACKEND == "alpamayo":
        run("Running Alpamayo uv sync  (~2-5 min)")
    elif INFERENCE_BACKEND == "cosmos3_native":
        run(f"Running uv sync --all-extras --group={_cosmos3_group}  (~2-5 min)")
    else:
        run(f"Running uv sync --extra {_extras}  (~2-3 min)")
    t0 = time.time()
    if INFERENCE_BACKEND == "alpamayo":
        rc, out = run_cmd(["uv", "sync"], cwd=REASON2_DIR, env=ENV, timeout=900)
        if rc != 0 and "flash-attn" in out.lower():
            run("Alpamayo uv sync failed on flash-attn, retrying with SDPA fallback")
            rc, out = run_cmd(
                ["uv", "sync", "--no-install-package", "flash-attn"],
                cwd=REASON2_DIR,
                env=ENV,
                timeout=900,
            )
            if rc == 0:
                ENV["ALPAMAYO_ATTENTION"] = "eager"
                os.environ["ALPAMAYO_ATTENTION"] = "eager"
    elif INFERENCE_BACKEND == "cosmos3_native":
        ENV["LD_LIBRARY_PATH"] = ""
        os.environ["LD_LIBRARY_PATH"] = ""
        rc, out = run_cmd(
            ["uv", "sync", "--all-extras", "--group", _cosmos3_group],
            cwd=REASON2_DIR,
            env=ENV,
            timeout=1200,
        )
    else:
        rc, out = run_cmd(["uv", "sync", "--extra", _extras], cwd=REASON2_DIR, env=ENV, timeout=600)
    if rc != 0 and INFERENCE_BACKEND not in ("alpamayo", "cosmos3_native"):
        run(f"{_extras} failed, trying uv sync without extras")
        rc, out = run_cmd(["uv", "sync"], cwd=REASON2_DIR, env=ENV, timeout=600)
    if rc != 0:
        print("  ✗  uv sync failed:", out[-500:]); sys.exit(1)
    if INFERENCE_BACKEND == "alpamayo":
        ENV["UV_NO_SYNC"] = "1"
        os.environ["UV_NO_SYNC"] = "1"
    ok(f"Dependencies installed in {time.time()-t0:.0f}s")

# ── Step 6b: vLLM version pin for CUDA 12.8 ──────────────────────────────────
# vLLM version matrix for cu128 environments (Hyperstack H100, driver 570.x = CUDA 12.8):
#   vLLM 0.11.0 → requires torch 2.8.0  (ABI mismatch: cu128 uv sync installs torch 2.9.0)
#   vLLM 0.14.0 → requires torch 2.9.1+cu128 ✅ compatible with CUDA 12.8 driver
#   vLLM 0.20.1 → requires torch 2.11.0+cu130 (needs CUDA 13.0 driver = driver 575.x+)
# Rule: driver_major < 575 → pin vLLM 0.14.0. Driver ≥ 575 → allow latest vLLM.
try:
    import subprocess as _sp_cuda
    _smi = _sp_cuda.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=10, text=True
    ).strip().split(".")[0:2]
    _driver_major = int(_smi[0])
    if INFERENCE_BACKEND == "vllm" and _driver_major < 575:  # < CUDA 13.0 threshold → cu128 pin required
        header("Step 6b — vLLM pin for CUDA 12.8", eta="~30-60s")
        rc_vllm, out_vllm = run_cmd(
            ["uv", "pip", "install", "vllm==0.14.0"],
            cwd=REASON2_DIR, env=ENV, timeout=180
        )
        if rc_vllm == 0:
            ok("vLLM pinned to 0.14.0 (torch 2.9.1+cu128) — compatible with CUDA 12.8 driver")
        else:
            warn(f"vLLM 0.14.0 pin failed — check pip output: {out_vllm[-300:]}")
            warn("vLLM ABI mismatch may cause ImportError at startup")
except Exception:
    pass  # CUDA check is best-effort; if nvidia-smi fails, proceed as-is

# ── Step 6c: ninja build system (required by flashinfer / torch compile) ─────
try:
    import subprocess as _sp_ninja
    _sp_ninja.check_output(["ninja", "--version"], timeout=5, stderr=_sp_ninja.DEVNULL)
except Exception:
    header("Step 6c — ninja build system", eta="<5s")
    try:
        import subprocess as _sp_ninja2
        _sp_ninja2.run(["apt-get", "install", "-y", "ninja-build"],
                       timeout=60, check=True, capture_output=True)
        ok("ninja-build installed")
    except Exception as _e_ninja:
        warn(f"ninja install failed ({_e_ninja}) — vLLM may fail with flashinfer JIT")

STEPS_DONE.append(6)
print_dashboard()

# ── Step 7: PyAV ──────────────────────────────────────────────────────────────
header("Step 7 — PyAV video backend", eta="<5s if cached, ~10s first time")
rc, av_check = run_cmd(
    ["uv", "run", "python", "-c", "import av; print(av.__version__)"],
    cwd=REASON2_DIR, env=ENV
)
if rc == 0 and "16.1.0" in av_check:
    ok(f"PyAV already installed ({av_check.strip()})")
else:
    run("Installing av==16.1.0  (~10s)")
    t0 = time.time()
    rc, out = run_cmd(["uv", "pip", "install", "av==16.1.0"], cwd=REASON2_DIR, env=ENV)
    if rc != 0:
        print("  ✗  av install failed:", out); sys.exit(1)
    ok(f"PyAV installed in {time.time()-t0:.0f}s")

# ── Step 8: Frontend dependencies ─────────────────────────────────────────────
header("Step 8 — Frontend dependencies", eta="<5s if cached, ~30-90s first time")
rc, gr_check = run_cmd(
    ["uv", "run", "python", "-c", "import gradio; print(gradio.__version__)"],
    cwd=REASON2_DIR, env=ENV
)
if rc == 0:
    ok(f"Gradio already installed ({gr_check.strip()})")
else:
    run("Installing gradio  (~30s)")
    t0 = time.time()
    rc, out = run_cmd(["uv", "pip", "install", "gradio"], cwd=REASON2_DIR, env=ENV, timeout=120)
    if rc != 0:
        print("  ✗  gradio install failed:", out); sys.exit(1)
    ok(f"Gradio installed in {time.time()-t0:.0f}s")

rc, rq_check = run_cmd(
    ["uv", "run", "python", "-c", "import requests; print(requests.__version__)"],
    cwd=REASON2_DIR, env=ENV
)
if rc == 0:
    ok(f"requests already installed ({rq_check.strip()})")
else:
    run("Installing requests  (~5s)")
    rc, out = run_cmd(["uv", "pip", "install", "requests"], cwd=REASON2_DIR, env=ENV)
    if rc != 0:
        warn(f"requests install failed — NIM API mode unavailable: {out}")
    else:
        ok("requests installed")

if INFERENCE_BACKEND == "alpamayo":
    rc, qwen_check = run_cmd(
        ["uv", "run", "python", "-c", "import qwen_vl_utils; print('ok')"],
        cwd=REASON2_DIR, env=ENV
    )
    if rc == 0:
        ok("qwen-vl-utils already installed")
    else:
        run("Installing qwen-vl-utils  (~5s)")
        rc, out = run_cmd(["uv", "pip", "install", "qwen-vl-utils"], cwd=REASON2_DIR, env=ENV)
        if rc != 0:
            print("  ✗  qwen-vl-utils install failed:", out); sys.exit(1)
        ok("qwen-vl-utils installed")

if USE_REASON_VITE or USE_PREDICT_VITE:
    def _activate_node_home():
        node_bin = os.path.join(NODE_HOME, "bin")
        node_path = os.path.join(node_bin, "node")
        if os.path.exists(node_path):
            ENV["PATH"] = f"{node_bin}:{ENV.get('PATH', '')}"
            os.environ["PATH"] = ENV["PATH"]
            return True
        return False

    def _node_major():
        rc_node, out_node = run_cmd(["node", "--version"], env=ENV)
        if rc_node != 0:
            return 0, out_node.strip()
        match = re.search(r"v(\d+)", out_node)
        return int(match.group(1)) if match else 0, out_node.strip()

    def _portable_node_arch():
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            return "x64"
        if machine in ("aarch64", "arm64"):
            return "arm64"
        return None

    def _install_portable_node():
        arch = _portable_node_arch()
        if not arch:
            return False, f"unsupported CPU architecture for Node.js tarball: {platform.machine()}"

        requested = os.environ.get("NODE_VERSION", "20.19.5").lstrip("v")
        versions = []
        for version in (requested, "20.18.1", "20.11.1"):
            if version not in versions:
                versions.append(version)

        last_error = ""
        os.makedirs(os.path.dirname(NODE_HOME), exist_ok=True)
        for version in versions:
            dist = f"node-v{version}-linux-{arch}"
            url = f"https://nodejs.org/dist/v{version}/{dist}.tar.xz"
            archive = f"/tmp/{dist}.tar.xz"
            extract_dir = f"/tmp/{dist}-extract"
            try:
                info(f"Downloading portable Node.js {version} from nodejs.org")
                urllib.request.urlretrieve(url, archive)
                shutil.rmtree(extract_dir, ignore_errors=True)
                os.makedirs(extract_dir, exist_ok=True)
                with tarfile.open(archive, "r:xz") as tar:
                    tar.extractall(extract_dir)
                extracted_root = os.path.join(extract_dir, dist)
                if not os.path.exists(os.path.join(extracted_root, "bin", "node")):
                    raise RuntimeError("downloaded Node.js archive did not contain bin/node")
                shutil.rmtree(NODE_HOME, ignore_errors=True)
                shutil.move(extracted_root, NODE_HOME)
                shutil.rmtree(extract_dir, ignore_errors=True)
                _activate_node_home()
                return True, f"portable Node.js {version} installed under {NODE_HOME}"
            except Exception as err:
                last_error = f"{version}: {err}"
                warn(f"Portable Node.js {version} install failed ({err})")
        return False, last_error

    _activate_node_home()
    node_major, node_version = _node_major()
    if node_major < 18:
        run("Installing Node.js 20 for Vite frontends")
        apt_attempted = False
        if shutil.which("apt-get") is not None and hasattr(os, "geteuid") and os.geteuid() == 0:
            apt_attempted = True
            rc, out = run_cmd(
                ["bash", "-lc", "curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs"],
                env=ENV,
                timeout=240,
            )
            if rc != 0:
                warn(f"NodeSource apt install failed, falling back to portable Node.js: {out[-600:]}")
        if not apt_attempted:
            info("No root apt-get access detected — installing portable Node.js for this user")
        _activate_node_home()
        node_major, node_version = _node_major()
        if node_major < 18:
            ok_portable, portable_msg = _install_portable_node()
            if ok_portable:
                ok(portable_msg)
            else:
                print(f"  ✗  Node.js install failed: {portable_msg}")
                sys.exit(1)
            node_major, node_version = _node_major()
    if node_major < 18:
        print(f"  ✗  Node.js >=18 is required for Vite; found {node_version or 'none'}.")
        sys.exit(1)
    ok(f"Node.js ready ({node_version})")

    vite_apps = []
    if USE_REASON_VITE:
        vite_apps.append(("Reason Vite", REASON_VITE_APP_DIR, "/tmp/_shared/reasonerClient.mjs"))
    if USE_PREDICT_VITE:
        vite_apps.append(("Predict Vite", PREDICT_VITE_APP_DIR, "/tmp/_shared/cosmos3Client.mjs"))

    for app_label, app_dir, shared_client in vite_apps:
        if not os.path.exists(os.path.join(app_dir, "package.json")):
            print(f"  ✗  {app_dir}/package.json not found — deploy the matching apps/ frontend first.")
            sys.exit(1)
        if not os.path.exists(shared_client):
            print(f"  ✗  {shared_client} not found — deploy apps/_shared first.")
            sys.exit(1)

        node_modules = os.path.join(app_dir, "node_modules")
        if os.path.isdir(node_modules):
            ok(f"{app_label} npm dependencies already installed")
        else:
            run(f"Installing {app_label} npm dependencies")
            rc, out = run_cmd(["npm", "ci"], cwd=app_dir, env=ENV, timeout=240)
            if rc != 0:
                print(f"  ✗  {app_label} npm ci failed:", out[-1200:])
                sys.exit(1)
            ok(f"{app_label} npm dependencies installed")

if FRONTEND in ("batch_inference", "fiftyone") or LAUNCH_BATCH_INFERENCE_COMPANION:
    rc, fo_check = run_cmd(
        ["uv", "run", "python", "-c", "import fiftyone, huggingface_hub; print(fiftyone.__version__)"],
        cwd=REASON2_DIR, env=ENV
    )
    if rc == 0:
        ok(f"FiftyOne already installed ({fo_check.strip()})")
    else:
        run("Installing FiftyOne + huggingface_hub for dataset batch frontend  (~60-90s)")
        rc, out = run_cmd(
            ["uv", "pip", "install", "fiftyone", "huggingface_hub"],
            cwd=REASON2_DIR, env=ENV, timeout=240
        )
        if rc != 0:
            warn(f"FiftyOne install failed — Batch Inference UI will fall back where possible: {out[-500:]}")
        else:
            ok("FiftyOne + huggingface_hub installed")

STEPS_DONE.append(7)

# ── Step 7b: Gradio frpc binary (required for public share link) ─────────────
# frpc_linux_amd64_v0.3 is required for Gradio's SHARE=true tunnel.
# On fresh instances, ~/.cache/huggingface/gradio/ may be missing or root-owned.
# We pre-download to $HF_HOME/gradio/frpc/ so Gradio finds it without auth.
_frpc_dir  = os.path.join(HF_HOME_DIR, "gradio", "frpc")
_frpc_path = os.path.join(_frpc_dir, "frpc_linux_amd64_v0.3")
if not os.path.exists(_frpc_path):
    header("Step 7b — Gradio frpc binary (share link tunnel)", eta="~5s")
    os.makedirs(_frpc_dir, exist_ok=True)
    _frpc_url = "https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64"
    try:
        urllib.request.urlretrieve(_frpc_url, _frpc_path)
        os.chmod(_frpc_path, 0o755)
        ok(f"frpc binary downloaded to {_frpc_path}")
    except Exception as _frpc_err:
        warn(f"frpc download failed ({_frpc_err}) — public Gradio share link may not work")
else:
    ok(f"frpc binary already present ({_frpc_path})")

# ── Step 9: Model weights ──────────────────────────────────────────────────────

def download_model(model_name, model_dir, size_hint, dl_env):
    """Download one HF model. Returns True on success."""
    model_marker  = os.path.join(model_dir, "model.safetensors")
    config_marker = os.path.join(model_dir, "config.json")
    if os.path.exists(model_marker) or os.path.exists(config_marker):
        size_mb = 0
        if os.path.exists(model_marker):
            size_mb = os.path.getsize(model_marker) // (1024 * 1024)
        info_str = f"{size_mb:,} MB" if size_mb else "already present"
        ok(f"Weights already downloaded ({model_name}, {info_str})")
        return True

    run(f"Downloading {model_name}  ({size_hint}, may take several minutes)")
    info("Progress below — download continues even if it looks stalled:")
    t0 = time.time()
    MAX_RETRIES = 5
    rc = 1
    for attempt in range(1, MAX_RETRIES + 1):
        rc = stream_cmd(
            ["uv", "run", "hf", "download", model_name,
             "--local-dir", model_dir],
            cwd=REASON2_DIR, env=dl_env, prefix="HF │ "
        )
        if rc == 0:
            break
        if attempt < MAX_RETRIES:
            print(f"  ✗  Attempt {attempt} failed. Retry {attempt}/{MAX_RETRIES} after 30s", flush=True)
            for _t in range(30):
                time.sleep(1)
                if _t % 10 == 9:
                    _elapsed = int(time.time() - SETUP_START)
                    print(f"  ⟳  Retry in {30-_t-1}s | Elapsed: {_elapsed//60}m {_elapsed%60}s{credits_spent()}", flush=True)
        else:
            print(f"  ✗  All {MAX_RETRIES} attempts failed for {model_name}.")
            return False
    elapsed = time.time() - t0
    ok(f"Downloaded {model_name} in {elapsed/60:.1f} min")
    return True


def _has_vlm_safetensors(model_dir):
    if not model_dir or not os.path.exists(os.path.join(model_dir, "config.json")):
        return False
    try:
        return any(name.endswith(".safetensors") for name in os.listdir(model_dir))
    except OSError:
        return False


def _cosmos3_framework_dir():
    return (
        os.environ.get("COSMOS3_DIR")
        or os.environ.get("COSMOS_FRAMEWORK_DIR")
        or os.path.join(HOME, "cosmos-framework")
    )


def _ensure_cosmos3_framework_repo(framework_dir):
    package_dir = os.path.join(framework_dir, "cosmos_framework")
    if os.path.isdir(package_dir):
        ok(f"cosmos-framework already present at {framework_dir}")
        return
    if os.path.exists(framework_dir) and os.listdir(framework_dir):
        print(f"  ✗  {framework_dir} exists but does not contain cosmos_framework.")
        print("     Set COSMOS3_DIR/COSMOS_FRAMEWORK_DIR to a valid checkout or empty path.")
        sys.exit(1)
    run(f"Cloning NVIDIA/cosmos-framework for Cosmos3 VLM conversion at {framework_dir}")
    rc = stream_cmd(
        ["git", "clone", "https://github.com/NVIDIA/cosmos-framework.git", framework_dir],
        env=ENV,
        prefix="C3VLM │ ",
    )
    if rc != 0:
        print("  ✗  cosmos-framework clone failed")
        sys.exit(1)


def _prepare_cosmos3_converter_env(framework_dir, dl_env):
    pyproject = os.path.join(framework_dir, "pyproject.toml")
    if not os.path.exists(pyproject):
        return
    if os.path.isdir(os.path.join(framework_dir, ".venv")):
        ok("cosmos-framework converter virtualenv already present")
        return
    group = os.environ.get("COSMOS3_UV_GROUP", "cu130-train")
    run(f"Preparing cosmos-framework converter environment (uv sync --all-extras --group={group})")
    rc = stream_cmd(
        ["uv", "sync", "--all-extras", "--group", group],
        cwd=framework_dir,
        env=dl_env,
        prefix="C3VLM │ ",
    )
    if rc != 0:
        warn("cosmos-framework uv sync failed; will try converter with the current Python environment")


def ensure_cosmos3_reasoner_vlm_safetensors(dl_env):
    """Convert Cosmos3 Nano/Super Reasoner checkpoints before local VLM serving.

    Official NIM mode does not need this. Local vLLM/HF fallback does: the
    Cosmos3 Reasoner checkpoints must be transformed by the converter in
    NVIDIA/cosmos-framework before they can be loaded as VLM safetensors.
    """
    checkpoint = _cfg.get("vlm_checkpoint")
    if not checkpoint or INFERENCE_BACKEND not in {"vllm", "hf"}:
        return None

    framework_dir = _cosmos3_framework_dir()
    output_dir = os.environ.get("COSMOS3_VLM_SAFETENSORS_PATH")
    if not output_dir:
        output_dir = MODEL_DIR if os.environ.get("MODEL_DIR") else os.path.join(
            framework_dir,
            "examples",
            "checkpoints",
            _cfg.get("vlm_output_dirname") or f"{checkpoint}-VLM",
        )

    if _has_vlm_safetensors(output_dir):
        ok(f"Converted Cosmos3 VLM safetensors already present at {output_dir}")
        return output_dir

    _ensure_cosmos3_framework_repo(framework_dir)
    _prepare_cosmos3_converter_env(framework_dir, dl_env)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    run(f"Converting {checkpoint} to VLM safetensors at {output_dir}")
    module = "cosmos_framework.scripts.convert_model_to_vlm_safetensors"
    uv_bin = shutil.which("uv")
    converter_cmds = []
    if uv_bin:
        converter_cmds.append([uv_bin, "run", "python", "-m", module, "--checkpoint-path", checkpoint, "-o", output_dir])
    converter_cmds.append(["python3", "-m", module, "--checkpoint-path", checkpoint, "-o", output_dir])

    rc = 1
    for cmd in converter_cmds:
        rc = stream_cmd(cmd, cwd=framework_dir, env=dl_env, prefix="C3VLM │ ")
        if rc == 0:
            break
    if rc != 0 or not _has_vlm_safetensors(output_dir):
        print("  ✗  Cosmos3 Reasoner VLM safetensors conversion failed.")
        print(f"     Expected config.json and *.safetensors in: {output_dir}")
        sys.exit(1)
    ok(f"Cosmos3 Reasoner VLM safetensors ready at {output_dir}")
    return output_dir


dl_env = {**ENV, "HF_TOKEN": HF_TOKEN}

if INFERENCE_BACKEND == "nim_local":
    info("nim_local backend — skipping HF weights download (NIM container ships the model)")
elif INFERENCE_BACKEND == "cosmos3_native":
    info("cosmos3_native backend — skipping legacy HF snapshot download (cosmos-framework handles checkpoints)")
elif _cfg.get("vlm_checkpoint"):
    _converted_model_dir = ensure_cosmos3_reasoner_vlm_safetensors(dl_env)
    if _converted_model_dir:
        MODEL_DIR = _converted_model_dir
        ENV["MODEL_DIR"] = MODEL_DIR
        os.environ["MODEL_DIR"] = MODEL_DIR
        ok(f"Using converted Cosmos3 VLM safetensors at {MODEL_DIR}")
else:
    for i, (var_label, var_dirname, var_hf_id, var_size) in enumerate(_cfg["variants"]):
        step_label = f"Step 9{'abcde'[i]} — {var_label} weights ({var_dirname})"
        header(step_label, eta=f"<5s if cached, longer first time ({var_size})")
        var_dir = os.path.join(MODELS_BASE, var_dirname)
        ok_ = download_model(var_hf_id, var_dir, var_size, dl_env)
        if not ok_ and i == 0:
            sys.exit(1)  # primary variant is required
        elif not ok_:
            warn(f"{var_label} download failed — will fall back to HF on first Gradio use")

STEPS_DONE.append(8)
print_dashboard()

def _resolve_nim_launch_config(model_id, model_size, cfg):
    """Resolve the exact NIM image/vendor/env from nim_catalog.py when present."""
    requested = (
        os.environ.get("NIM_MODEL_SHORT")
        or model_id
        or cfg.get("nim")
        or (cfg["variants"][0][2] if cfg.get("variants") else "")
    )

    def _strip_nim_tag(value):
        return re.sub(r":[^/:]+$", "", value)

    requested_l = requested.lower()
    _nim_aliases = {
        "nvidia/cosmos3-nano-reasoner": "cosmos3-reasoner-nano",
        "cosmos3-nano-reasoner": "cosmos3-reasoner-nano",
        "nvidia/cosmos3-super-reasoner": "cosmos3-reasoner-super",
        "cosmos3-super-reasoner": "cosmos3-reasoner-super",
        "nvidia/cosmos-predict1-5b-video2world": "nvidia/cosmos-predict1-5b",
        "cosmos-predict1-5b-video2world": "nvidia/cosmos-predict1-5b",
        "nvidia/cosmos-predict1-7b-video2world": "nvidia/cosmos-predict1-7b-video2world",
        "cosmos-predict1-7b-video2world": "nvidia/cosmos-predict1-7b-video2world",
        "nvidia/cosmos-predict2.5-2b": "nvidia/cosmos-predict2-5-2b",
        "cosmos-predict2.5-2b": "nvidia/cosmos-predict2-5-2b",
        "nvidia/cosmos-predict2.5-14b": "nvidia/cosmos-predict2-5-14b",
        "cosmos-predict2.5-14b": "nvidia/cosmos-predict2-5-14b",
    }
    requested_l = _nim_aliases.get(requested_l, requested_l)
    if requested_l.startswith("nvcr.io/nim/"):
        requested_l = requested_l[len("nvcr.io/nim/"):]
    requested_l = _strip_nim_tag(requested_l)
    requested_leaf = requested_l.split("/")[-1]
    requested_size = str(
        os.environ.get("NIM_MODEL_SIZE")
        or (cfg.get("nim_env", {}) or {}).get("NIM_MODEL_SIZE")
        or ""
    ).lower()
    if requested_leaf == "cosmos3-reasoner" and requested_size in {"nano", "super"}:
        requested_l = f"cosmos3-reasoner-{requested_size}"
        requested_leaf = requested_l

    catalog = []
    for candidate_dir in ("/tmp", os.path.dirname(os.path.abspath(__file__))):
        if candidate_dir not in sys.path:
            sys.path.insert(0, candidate_dir)
    try:
        from nim_catalog import KNOWN_VLM_NIMS  # type: ignore
        catalog = list(KNOWN_VLM_NIMS)
    except Exception as exc:
        warn(f"Could not import nim_catalog.py; using fallback NIM image resolution: {exc}")

    matched = None
    for nim in catalog:
        image_key = getattr(nim, "image", "").lower()
        if image_key.startswith("nvcr.io/nim/"):
            image_key = image_key[len("nvcr.io/nim/"):]
        image_key = _strip_nim_tag(image_key)
        tokens = {
            getattr(nim, "short_id", "").lower(),
            getattr(nim, "served_model_id", "").lower(),
            image_key,
            image_key.split("/")[-1],
        }
        if requested_l in tokens or requested_leaf in tokens:
            matched = nim
            break

    if matched:
        short = getattr(matched, "short_id")
        image = getattr(matched, "image")
        served = getattr(matched, "served_model_id")
        env = dict(getattr(matched, "env", {}) or {})
    else:
        served = requested_l
        short = requested_l.split("/")[-1]
        if "/" in requested_l:
            image = f"nvcr.io/nim/{requested_l}:latest"
        else:
            image = f"nvcr.io/nim/nvidia/{short}:latest"
        env = {}

    env.update(cfg.get("nim_env", {}) or {})
    image = os.environ.get("NIM_IMAGE", image)
    short = os.environ.get("NIM_MODEL_SHORT", short)
    served = os.environ.get("NIM_SERVED_MODEL_NAME", served)
    max_wait = str(os.environ.get("NIM_MAX_WAIT") or cfg.get("nim_max_wait") or 1800)
    return short, image, served, env, max_wait

# ── Step 9-NIM: Launch NIM container (nim_local backend only) ────────────────
if INFERENCE_BACKEND == "nim_local":
    header("Step 9-NIM — Launch NIM Docker container", eta="~30s if image cached, 10-30 min first pull")
    _nim_short, _nim_image, _nim_served, _nim_extra_env, _nim_max_wait = _resolve_nim_launch_config(MODEL_ID, MODEL_SIZE, _cfg)
    _nim_port = int(os.environ.get("NIM_PORT", "8000"))
    _nim_launch = "/tmp/nim_launch.sh"
    if not os.path.exists(_nim_launch):
        print(f"  ✗  {_nim_launch} not found — deploy nim_launch.sh first"); sys.exit(1)
    info(f"Image: {_nim_image} | Served model: {_nim_served} | Port: {_nim_port} | Container: cosmos-nim")
    _nim_env = {
        **ENV,
        "NGC_API_KEY":     NGC_API_KEY,
        "MODEL":           _nim_short,
        "IMAGE":           _nim_image,
        "PORT":            str(_nim_port),
        "CONTAINER_NAME":  "cosmos-nim",
        "LOCAL_NIM_CACHE": os.environ.get("LOCAL_NIM_CACHE", os.path.expanduser("~/.cache/nim")),
        "NIM_CACHE_MODE":  os.environ.get("NIM_CACHE_MODE", "internal"),
        "MAX_WAIT":        _nim_max_wait,
    }
    _nim_env.update(_nim_extra_env)
    rc = stream_cmd(["bash", _nim_launch], env=_nim_env, prefix="NIM │ ")
    if rc != 0:
        print(f"  ✗  NIM launch failed (rc={rc}). Logs: /tmp/nim_launch.log + docker logs cosmos-nim"); sys.exit(1)
    # Point Gradio at the NIM container by overriding VLLM_BASE_URL.
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{_nim_port}/v1"
    MODEL_NAME = _nim_served
    ok(f"NIM container live at http://localhost:{_nim_port}/v1")

# ── Step 9-A: Alpamayo OpenAI-compatible adapter ─────────────────────────────
ALPAMAYO_LOG_FILE = "/tmp/alpamayo_openai_server.log"
if INFERENCE_BACKEND == "alpamayo":
    _alpamayo_port = int(os.environ.get("ALPAMAYO_PORT", "8001"))
    _alpamayo_base = f"http://localhost:{_alpamayo_port}/v1"
    _alpamayo_script_candidates = [
        os.environ.get("ALPAMAYO_OPENAI_SERVER", ""),
        "/tmp/alpamayo_openai_server.py",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpamayo_openai_server.py"),
        os.path.join(REASON2_DIR, "src", "alpamayo1_5", "byo_openai_server.py"),
    ]
    _alpamayo_script = next((candidate for candidate in _alpamayo_script_candidates if candidate and os.path.exists(candidate)), "")
    _alpamayo_ready = False
    try:
        urllib.request.urlopen(f"{_alpamayo_base}/models", timeout=3)
        _alpamayo_ready = True
        ok(f"Alpamayo adapter already running at {_alpamayo_base} — reusing")
    except Exception:
        pass

    if not _alpamayo_ready:
        header("Step 9-A — Start Alpamayo BYO OpenAI adapter", eta="~2-5 min for first model load")
        _alpamayo_model_path = os.environ.get(
            "ALPAMAYO_MODEL_PATH",
            os.path.join(MODELS_BASE, _cfg["variants"][0][1]),
        )
        _alpamayo_cmd = [
            "uv", "run", "python",
            _alpamayo_script or "-m",
        ]
        if not _alpamayo_script:
            _alpamayo_cmd.append("alpamayo1_5.byo_openai_server")
        _alpamayo_cmd.extend([
            "--port", str(_alpamayo_port),
            "--model-id", MODEL_NAME,
        ])
        if os.path.exists(os.path.join(_alpamayo_model_path, "config.json")):
            _alpamayo_cmd.extend(["--model-path", _alpamayo_model_path])
        if os.environ.get("ALPAMAYO_PRELOAD", "1").lower() not in {"0", "false", "no"}:
            _alpamayo_cmd.append("--preload")
        _alpamayo_env = {
            **ENV,
            "MODEL_ID": MODEL_NAME,
            "ALPAMAYO_MODEL_ID": MODEL_NAME,
            "ALPAMAYO_MODEL_PATH": _alpamayo_model_path,
            "ALPAMAYO_PORT": str(_alpamayo_port),
            "ALPAMAYO_ATTENTION": os.environ.get("ALPAMAYO_ATTENTION", "eager"),
        }
        run(f"Launching Alpamayo adapter for {MODEL_NAME} (log: {ALPAMAYO_LOG_FILE})")
        with open(ALPAMAYO_LOG_FILE, "w") as _af:
            subprocess.Popen(_alpamayo_cmd, cwd=REASON2_DIR, env=_alpamayo_env,
                             stdout=_af, stderr=subprocess.STDOUT)

        t_alpamayo = time.time()
        _ALPAMAYO_TIMEOUT = int(os.environ.get("ALPAMAYO_START_TIMEOUT", "900"))
        while time.time() - t_alpamayo < _ALPAMAYO_TIMEOUT:
            try:
                urllib.request.urlopen(f"{_alpamayo_base}/models", timeout=5)
                _alpamayo_ready = True
                break
            except Exception:
                time.sleep(5)

        if not _alpamayo_ready:
            print(f"  ✗  Alpamayo adapter did not start within {_ALPAMAYO_TIMEOUT}s. Check {ALPAMAYO_LOG_FILE}")
            sys.exit(1)
        ok(f"Alpamayo adapter ready in {int(time.time() - t_alpamayo)}s")
    os.environ["ALPAMAYO_BASE_URL"] = _alpamayo_base
    os.environ["VLLM_BASE_URL"] = _alpamayo_base

# ── Step 9-C: Cosmos3 Generator through NVIDIA/cosmos-framework ──────────────
if INFERENCE_BACKEND == "cosmos3_native":
    header("Step 9-C — Start cosmos-framework Ray Serve", eta="~2-10 min for first model load")
    _cosmos3_launch = "/tmp/cosmos3_native_launch.sh"
    if not os.path.exists(_cosmos3_launch):
        print(f"  ✗  {_cosmos3_launch} not found — deploy cosmos3_native_launch.sh first"); sys.exit(1)

    _cosmos3_checkpoint = os.environ.get("COSMOS3_CHECKPOINT", _cfg["variants"][0][1])
    _cosmos3_serve_port = int(os.environ.get("COSMOS3_SERVE_PORT", "8000"))
    _cosmos3_gradio_port = int(os.environ.get("COSMOS3_GRADIO_PORT", "8080"))
    _cosmos3_output_dir = os.environ.get(
        "COSMOS3_OUTPUT_DIR",
        os.path.join(REASON2_DIR, "outputs", "ray_serve"),
    )
    _cosmos3_base = f"http://localhost:{_cosmos3_serve_port}"
    info(f"Checkpoint: {_cosmos3_checkpoint} | Ray Serve: {_cosmos3_base} | framework dir: {REASON2_DIR}")
    _cosmos3_env = {
        **ENV,
        "COSMOS3_DIR": REASON2_DIR,
        "COSMOS3_CHECKPOINT": _cosmos3_checkpoint,
        "COSMOS3_SERVE_PORT": str(_cosmos3_serve_port),
        "COSMOS3_GRADIO_PORT": str(_cosmos3_gradio_port),
        "COSMOS3_OUTPUT_DIR": _cosmos3_output_dir,
        "COSMOS3_HOST": os.environ.get("COSMOS3_HOST", "0.0.0.0"),
        "LD_LIBRARY_PATH": "",
    }
    rc = stream_cmd(["bash", _cosmos3_launch], env=_cosmos3_env, prefix="C3 │ ")
    if rc != 0:
        print(f"  ✗  cosmos-framework launch failed (rc={rc}). Logs: /tmp/cosmos3_serve.log + /tmp/cosmos3_gradio.log"); sys.exit(1)

    os.environ["COSMOS3_BASE_URL"] = _cosmos3_base
    os.environ["RAY_SERVE_BASE_URL"] = _cosmos3_base
    os.environ["COSMOS3_BACKEND"] = "cosmos3_native"
    os.environ["COSMOS3_OUTPUT_DIR"] = _cosmos3_output_dir
    os.environ["COSMOS3_DIR"] = REASON2_DIR
    # The framework router serves the default model under an empty key. Keep UI
    # display names friendly, but send model="" in /generate requests.
    os.environ.setdefault("COSMOS3_RAY_MODEL_NAME", "")
    os.environ.setdefault("PREDICT_DISPLAY_MODEL", _cosmos3_checkpoint)
    MODEL_NAME = _cosmos3_checkpoint
    ok(f"cosmos-framework Ray Serve live at {_cosmos3_base}")

# ── Step 9b: vLLM server auto-start (BUG-VLLM-AUTOSTART) ─────────────────────
# In vLLM mode Gradio connects to localhost:8000. If vLLM isn't running, the first
# inference request fails with "Connection refused". Start it here, before Gradio.
VLLM_LOG_FILE = "/tmp/vllm_server.log"
if INFERENCE_BACKEND == "vllm":
    _vllm_ready = False
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
        _vllm_ready = True
        ok("vLLM already running on :8000 — reusing")
    except Exception:
        pass

    if not _vllm_ready:
        header("Step 9b — Start vLLM server", eta="~60-120s for model load")
        _vllm_flags  = _cfg.get("vllm_extra_flags", ["--gpu-memory-utilization", "0.85"])
        _vllm_maxlen = _cfg.get("vllm_max_model_len", VLLM_MAX_MODEL_LEN)
        _vllm_cmd = [
            f"{REASON2_DIR}/.venv/bin/vllm", "serve", MODEL_DIR,
            "--served-model-name", MODEL_NAME,
            "--port", "8000",
            "--dtype", "auto",
            "--trust-remote-code",
            "--max-model-len", str(_vllm_maxlen),
        ] + _vllm_flags

        _vllm_proc_env = {**ENV, **_cfg.get("vllm_env", {})}
        run(f"Launching vLLM server for {MODEL_SIZE} (log: {VLLM_LOG_FILE})")
        with open(VLLM_LOG_FILE, "w") as _vf:
            subprocess.Popen(_vllm_cmd, cwd=REASON2_DIR, env=_vllm_proc_env,
                             stdout=_vf, stderr=subprocess.STDOUT)

        # 32B models and C3-8B need extra warmup time; C3-8B was observed
        # crossing the old 180s timeout while still loading cleanly.
        _LARGE_MODEL_SIZES = {"32B", "C3-32B", "C3-super", "QW3-32B"}
        _VLLM_TIMEOUT_OVERRIDES = {"C3-8B": 900}
        VLLM_TIMEOUT = _VLLM_TIMEOUT_OVERRIDES.get(MODEL_SIZE, 420 if MODEL_SIZE in _LARGE_MODEL_SIZES else 180)
        t_vllm = time.time()
        while time.time() - t_vllm < VLLM_TIMEOUT:
            try:
                urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
                _vllm_ready = True
                break
            except Exception:
                time.sleep(5)

        if not _vllm_ready:
            print(f"  ✗  vLLM server did not start within {VLLM_TIMEOUT}s. Check {VLLM_LOG_FILE}")
            sys.exit(1)
        ok(f"vLLM ready in {int(time.time() - t_vllm)}s")

# ── Step 10: Launch frontend ──────────────────────────────────────────────────
header(f"Step 10 — Launch {FRONTEND} frontend", eta="~5-10s")

if not os.path.exists(GRADIO_APP):
    print(f"  ✗  {GRADIO_APP} not found — deploy gradio_cr2_byo.py first"); sys.exit(1)
if GRADIO_APP.endswith("gradio_cr2_byo.py") and not os.path.exists(HOSTED_COMPARE_HELPER):
    print(
        f"  ✗  {HOSTED_COMPARE_HELPER} not found — deploy hosted_model_compare.py "
        "beside the generic Gradio app before launch"
    )
    sys.exit(1)

if os.path.exists(URL_FILE):
    os.remove(URL_FILE)

_is_cosmos3_generator_backend = INFERENCE_BACKEND == "cosmos3_native" or (
    INFERENCE_BACKEND == "nim_local" and "gen" in (MODEL_ID + MODEL_NAME).lower()
)

launch_env = {
    **ENV,
    "MODEL_SIZE":         MODEL_SIZE,
    "MODEL_DIR":          MODEL_DIR,
    "MODEL_NAME":         MODEL_NAME,
    "MODEL_ID":           MODEL_NAME,
    "ALPAMAYO_MODEL_ID":  os.environ.get("ALPAMAYO_MODEL_ID", MODEL_NAME if INFERENCE_BACKEND == "alpamayo" else ""),
    "ALPAMAYO_BASE_URL":  os.environ.get("ALPAMAYO_BASE_URL", "http://localhost:8001/v1"),
    "BYO_VIDEO_MODEL_TOWERS": ",".join(sorted(MODEL_TOWERS)),
    "BYO_VIDEO_ACTIVE_TOWERS": ",".join(sorted(FRONTEND_TOWERS)),
    "GRADIO_PORT":        str(GRADIO_PORT),
    "GRADIO_SHARE":       os.environ.get("GRADIO_SHARE", "true"),
    "PYTHONUNBUFFERED":   "1",
    "HF_TOKEN":           HF_TOKEN,
    "NGC_API_KEY":        NGC_API_KEY,
    "LOW_VRAM":           "true" if LOW_VRAM else "false",
    "GRADIO_FPS":         str(gradio_fps),
    "GRADIO_MAX_PIXELS":  str(max_pixels),
    "GRADIO_PREFILL_TPS": str(prefill_tps),
    "GRADIO_RUN_LOG_FILE": os.environ.get("GRADIO_RUN_LOG_FILE", "/tmp/byo_video_reason2_runs.jsonl"),
    "INFERENCE_BACKEND":  INFERENCE_BACKEND,
    "VLLM_BASE_URL":      os.environ.get("VLLM_BASE_URL", os.environ.get("ALPAMAYO_BASE_URL", "http://localhost:8000/v1")),
    "NIM_BASE_URL":       os.environ.get("NIM_BASE_URL", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")),
    "NIM_IMAGE":          os.environ.get("NIM_IMAGE", os.environ.get("IMAGE", "")),
    "NIM_SERVED_MODEL_NAME": os.environ.get("NIM_SERVED_MODEL_NAME", MODEL_NAME),
    "COSMOS_MODEL_ID":    os.environ.get("COSMOS_MODEL_ID", os.environ.get("NIM_SERVED_MODEL_NAME", MODEL_NAME)),
    "COSMOS_MODEL_COLLECTION": os.environ.get("COSMOS_MODEL_COLLECTION", "cosmos3" if _is_cosmos3_generator_backend else "cosmos-predict1"),
    "COSMOS_VIDEO_HEIGHT": os.environ.get("COSMOS_VIDEO_HEIGHT", "256" if _is_cosmos3_generator_backend else "704"),
    "COSMOS_VIDEO_WIDTH": os.environ.get("COSMOS_VIDEO_WIDTH", "448" if _is_cosmos3_generator_backend else "1280"),
    "COSMOS_VIDEO_FRAMES": os.environ.get("COSMOS_VIDEO_FRAMES", "25" if _is_cosmos3_generator_backend else "121"),
    "COSMOS_VIDEO_FPS":   os.environ.get("COSMOS_VIDEO_FPS", "24"),
    "COSMOS_VIDEO_STEPS": os.environ.get("COSMOS_VIDEO_STEPS", "35"),
    "COSMOS_GUIDANCE_SCALE": os.environ.get("COSMOS_GUIDANCE_SCALE", "6" if _is_cosmos3_generator_backend else "7"),
    "COSMOS3_BACKEND":    os.environ.get("COSMOS3_BACKEND", INFERENCE_BACKEND),
    "COSMOS3_BASE_URL":   os.environ.get("COSMOS3_BASE_URL", "http://localhost:8000"),
    "RAY_SERVE_BASE_URL": os.environ.get("RAY_SERVE_BASE_URL", os.environ.get("COSMOS3_BASE_URL", "http://localhost:8000")),
    "COSMOS3_OUTPUT_DIR": os.environ.get("COSMOS3_OUTPUT_DIR", os.path.join(REASON2_DIR, "outputs", "ray_serve")),
    "COSMOS3_DIR":        os.environ.get("COSMOS3_DIR", REASON2_DIR if INFERENCE_BACKEND == "cosmos3_native" else ""),
    "PREDICT_DISPLAY_MODEL": os.environ.get("PREDICT_DISPLAY_MODEL", MODEL_NAME),
    "VLLM_API_KEY":       os.environ.get("VLLM_API_KEY", "EMPTY"),
    "COSMOS_EXTRAS":      os.environ.get("COSMOS_EXTRAS", "cu128"),
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    # MAXLEN-001: always pass explicitly — never rely on vLLM default (8192 breaks video queries)
    "VLLM_MAX_MODEL_LEN": str(_cfg.get("vllm_max_model_len", VLLM_MAX_MODEL_LEN)),
    # PRELOAD-001: skip HF preload when using API-backed servers.
    "SKIP_HF_PRELOAD":    "1" if INFERENCE_BACKEND in ("vllm", "nim_local", "alpamayo", "cosmos3_native") else "0",
}
if INFERENCE_BACKEND == "cosmos3_native" or "COSMOS3_RAY_MODEL_NAME" in os.environ:
    launch_env["COSMOS3_RAY_MODEL_NAME"] = os.environ.get("COSMOS3_RAY_MODEL_NAME", "")

def _detect_host_ip():
    env_ip = os.environ.get("BYO_VIDEO_LOCAL_HOST")
    if env_ip:
        return env_ip
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2)
        for tok in out.stdout.split():
            if "." in tok and not tok.startswith("127."):
                return tok
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return None

_public_host_ip = _detect_host_ip()
if _public_host_ip:
    launch_env.setdefault("BYO_VIDEO_LOCAL_HOST", _public_host_ip)
    launch_env.setdefault("GRADIO_PUBLIC_URL", f"http://{_public_host_ip}:{GRADIO_PORT}")
    launch_env.setdefault("NIM_SWITCH_URL", f"http://{_public_host_ip}:{NIM_SWITCH_PORT}")
else:
    launch_env.setdefault("NIM_SWITCH_URL", f"http://localhost:{NIM_SWITCH_PORT}")
launch_env.update({
    "NIM_SWITCH_PORT": str(NIM_SWITCH_PORT),
    "NIM_SWITCH_SERVICE": NIM_SWITCH_SERVICE,
    "NIM_LAUNCH_SCRIPT": "/tmp/nim_launch.sh",
    "NIM_CREDENTIAL_FILE": os.environ.get("NIM_CREDENTIAL_FILE", "/tmp/byo_video_nim_credentials.env"),
})
try:
    with open(NIM_SWITCH_URL_FILE, "w") as f:
        f.write(launch_env["NIM_SWITCH_URL"] + "\n")
except Exception:
    pass

def _drain_stdout(fh, path):
    try:
        with open(path, "a") as f:
            for line in fh:
                f.write(line)
                f.flush()
    except Exception:
        pass

def _launch_gradio(sidecar=False):
    label = "Gradio sidecar" if sidecar else _GRADIO_APP_LABEL
    run(f"Starting {label} on port {GRADIO_PORT}")
    gradio_env = dict(launch_env)
    if sidecar:
        # Keep the mandatory Gradio link live without double-loading HF weights
        # when the selected primary surface is batch-inference or FiftyOne.
        gradio_env["SKIP_HF_PRELOAD"] = "1"
        gradio_env["BYO_VIDEO_FRONTEND"] = "gradio_sidecar"

    proc = subprocess.Popen(
        ["uv", "run", "python", "-u", GRADIO_APP],
        cwd=REASON2_DIR,
        env=gradio_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        with open("/tmp/gradio_demo.pid", "w") as f:
            f.write(str(proc.pid))
    except Exception:
        pass

    url = None
    local_url = None
    share_failed = False
    url_pattern        = re.compile(r'(https?://[^\s"\']+gradio\.live[^\s"\']*)')
    local_url_pattern  = re.compile(r'Running on local URL:\s+(http://[^\s]+)')
    launch_url_pattern = re.compile(r'\[launch\]\s+(https?://[^\s]+)')
    share_requested = gradio_env.get("GRADIO_SHARE", "true").lower() not in {"0", "false", "no"}
    URL_CAPTURE_TIMEOUT = 300
    t_launch = time.time()
    with open(LOG_FILE, "w") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            stripped = line.rstrip()
            if stripped:
                print(f"     {DIM}{stripped}{RESET}", flush=True)
            m = url_pattern.search(stripped)
            if m:
                url = m.group(1).rstrip(".")
                break
            ml = local_url_pattern.search(stripped)
            if ml and not local_url:
                local_url = ml.group(1).rstrip("/")
                if not share_requested:
                    break
            mla = launch_url_pattern.search(stripped)
            if mla:
                candidate = mla.group(1).rstrip(".").rstrip("/")
                if "gradio.live" in candidate:
                    url = candidate
                    break
                if not local_url:
                    local_url = candidate
                if not share_requested:
                    break
            if "Could not create share link" in stripped:
                share_failed = True
                if local_url:
                    break
            if time.time() - t_launch > URL_CAPTURE_TIMEOUT:
                print(f"  ⚠  No gradio.live URL after {URL_CAPTURE_TIMEOUT}s — falling back to local URL.")
                break

    if not url:
        if not local_url:
            print("  ✗  Gradio printed no URL (neither share nor local). Check /tmp/gradio_demo.log")
            proc.terminate()
            sys.exit(1)
        host_ip = _detect_host_ip()
        if host_ip:
            url = local_url.replace("0.0.0.0", host_ip).replace("127.0.0.1", host_ip)
        else:
            url = local_url
        reason = "share link unavailable (host firewalled)" if share_failed else "share link timed out"
        print(f"  ⚠  {reason}. Using local URL: {url}")
        print(f"     {DIM}If your machine can't reach {url} directly, run on your client:{RESET}")
        print(f"     {DIM}  ssh -L {GRADIO_PORT}:localhost:{GRADIO_PORT} <user@host>{RESET}")
        print(f"     {DIM}then open http://localhost:{GRADIO_PORT}/ in your browser.{RESET}")

    _probe_ok = False
    for _probe_attempt in range(10):
        try:
            urllib.request.urlopen(f"http://localhost:{GRADIO_PORT}/", timeout=3)
            _probe_ok = True
            break
        except Exception:
            time.sleep(2)

    if not _probe_ok:
        print("  ✗  Gradio process launched but / probe failed after 20s — process may have crashed.")
        print(f"     Check {LOG_FILE} for errors.")
        sys.exit(1)

    with open(URL_FILE, "w") as f:
        f.write(url + "\n")

    with open("/tmp/gradio_live.flag", "w") as f:
        f.write(url + "\n")

    ok("Gradio frontend is live")
    import threading as _thr
    _thr.Thread(target=_drain_stdout, args=(proc.stdout, LOG_FILE), daemon=True).start()
    return proc, url

def _launch_reason_vite():
    run(f"Starting Cosmos Reason Vite Build skin on port {REASON_VITE_PORT}")
    vite_env = {
        **launch_env,
        "PORT": str(REASON_VITE_PORT),
        "MODEL_ID": MODEL_ID or MODEL_NAME,
        "MODEL_NAME": MODEL_NAME or MODEL_ID,
        "ALPAMAYO_BASE_URL": os.environ.get("ALPAMAYO_BASE_URL", "http://localhost:8001/v1"),
        "VLLM_BASE_URL": os.environ.get("VLLM_BASE_URL", os.environ.get("ALPAMAYO_BASE_URL", "http://localhost:8000/v1")),
        "VITE_MODEL_NAME": MODEL_NAME or MODEL_ID,
        "VITE_INFERENCE_BACKEND": INFERENCE_BACKEND,
        "VITE_COSMOS3_INFO_URL": "/api/active-model",
    }
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=REASON_VITE_APP_DIR,
        env=vite_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    local_url = f"http://localhost:{REASON_VITE_PORT}"
    log_path = "/tmp/nvidia_build_reason_vite.log"
    t_launch = time.time()
    with open(log_path, "w") as log:
        while time.time() - t_launch < 90:
            line = proc.stdout.readline()
            if line:
                log.write(line)
                log.flush()
                stripped = line.rstrip()
                if stripped:
                    print(f"     {DIM}{stripped}{RESET}", flush=True)
                if "vite-build-reason" in stripped:
                    break
            if proc.poll() is not None:
                print(f"  ✗  Reason Vite exited early. Check {log_path}")
                sys.exit(1)
            time.sleep(0.2)

    _probe_ok = False
    for _probe_attempt in range(20):
        try:
            urllib.request.urlopen(f"{local_url}/api/models", timeout=3)
            _probe_ok = True
            break
        except Exception:
            time.sleep(2)

    if not _probe_ok:
        print("  ✗  Reason Vite launched but /api/models probe failed after 40s.")
        print(f"     Check {log_path} for errors.")
        proc.terminate()
        sys.exit(1)

    host_ip = _detect_host_ip()
    url = local_url if not host_ip else f"http://{host_ip}:{REASON_VITE_PORT}"
    for _path in (REASON_VITE_URL_FILE, REASON_VITE_LIVE_FLAG):
        with open(_path, "w") as f:
            f.write(url + "\n")

    ok("Reason Vite Build skin is live")
    import threading as _thr
    _thr.Thread(target=_drain_stdout, args=(proc.stdout, log_path), daemon=True).start()
    return proc, url

def _launch_predict_vite():
    run(f"Starting Cosmos Predict Vite Build skin on port {PREDICT_VITE_PORT}")
    _predict_base = os.environ.get(
        "COSMOS3_BASE_URL",
        os.environ.get("RAY_SERVE_BASE_URL", "http://localhost:8000"),
    )
    _nim_base = os.environ.get("NIM_BASE_URL", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    if INFERENCE_BACKEND == "nim_local":
        os.environ.setdefault("COSMOS3_BACKEND", "nim_local")
        os.environ.setdefault("NIM_BASE_URL", _nim_base)
    vite_env = {
        **launch_env,
        "PORT": str(PREDICT_VITE_PORT),
        "MODEL_ID": MODEL_ID or MODEL_NAME,
        "MODEL_NAME": MODEL_NAME or MODEL_ID,
        "COSMOS3_BACKEND": os.environ.get("COSMOS3_BACKEND", INFERENCE_BACKEND),
        "COSMOS3_BASE_URL": _predict_base,
        "NIM_BASE_URL": os.environ.get("NIM_BASE_URL", _nim_base),
        "NIM_INFER_URL": os.environ.get("NIM_INFER_URL", ""),
        "NIM_IMAGE": os.environ.get("NIM_IMAGE", os.environ.get("IMAGE", "")),
        "NIM_SERVED_MODEL_NAME": os.environ.get("NIM_SERVED_MODEL_NAME", MODEL_NAME or MODEL_ID),
        "PREDICT_STAGED_MODEL_FILE": os.environ.get("PREDICT_STAGED_MODEL_FILE", "/tmp/nvidia_build_predict_staged_model.json"),
        "COSMOS3_CLIENT_REQUIRE_ROOT": PREDICT_VITE_APP_DIR,
        "VITE_COSMOS3_INFO_URL": "/api/active-model",
    }
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=PREDICT_VITE_APP_DIR,
        env=vite_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    local_url = f"http://localhost:{PREDICT_VITE_PORT}"
    log_path = "/tmp/nvidia_build_predict_vite.log"
    t_launch = time.time()
    with open(log_path, "w") as log:
        while time.time() - t_launch < 90:
            line = proc.stdout.readline()
            if line:
                log.write(line)
                log.flush()
                stripped = line.rstrip()
                if stripped:
                    print(f"     {DIM}{stripped}{RESET}", flush=True)
                if "vite-build-predict" in stripped:
                    break
            if proc.poll() is not None:
                print(f"  ✗  Predict Vite exited early. Check {log_path}")
                sys.exit(1)
            time.sleep(0.2)

    _probe_ok = False
    for _probe_attempt in range(20):
        try:
            urllib.request.urlopen(f"{local_url}/api/models", timeout=3)
            _probe_ok = True
            break
        except Exception:
            time.sleep(2)

    if not _probe_ok:
        print("  ✗  Predict Vite launched but /api/models probe failed after 40s.")
        print(f"     Check {log_path} for errors.")
        proc.terminate()
        sys.exit(1)

    host_ip = _detect_host_ip()
    url = local_url if not host_ip else f"http://{host_ip}:{PREDICT_VITE_PORT}"
    for _path in (PREDICT_VITE_URL_FILE, PREDICT_VITE_LIVE_FLAG):
        with open(_path, "w") as f:
            f.write(url + "\n")

    ok("Predict Vite Build skin is live")
    import threading as _thr
    _thr.Thread(target=_drain_stdout, args=(proc.stdout, log_path), daemon=True).start()
    return proc, url

def _launch_batch_inference(gradio_proc=None):
    if not os.path.exists(BATCH_INFERENCE_APP):
        print(f"  ✗  {BATCH_INFERENCE_APP} not found — deploy byo_video_batch_inference.py first"); sys.exit(1)

    runtime_env = {
        **launch_env,
        "BYO_VIDEO_FRONTEND": FRONTEND if FRONTEND in ("batch_inference", "fiftyone") else "batch_inference",
        "BATCH_INFERENCE_PORT": str(BATCH_INFERENCE_PORT),
        "BATCH_INFERENCE_DATASET": os.environ.get("BATCH_INFERENCE_DATASET", "pjramg/Safe_Unsafe_Test"),
        "BATCH_INFERENCE_CONCURRENCY": os.environ.get("BATCH_INFERENCE_CONCURRENCY", "4"),
        "BATCH_INFERENCE_MAX_VIDEOS": os.environ.get("BATCH_INFERENCE_MAX_VIDEOS", "20"),
        "FIFTYONE_PORT": os.environ.get("FIFTYONE_PORT", "5151"),
    }

    run(f"Starting Cosmos BYO Video Batch Inference on port {BATCH_INFERENCE_PORT}")
    proc = subprocess.Popen(
        ["uv", "run", "python", "-u", BATCH_INFERENCE_APP, "serve",
         "--host", "0.0.0.0", "--port", str(BATCH_INFERENCE_PORT)],
        cwd=REASON2_DIR,
        env=runtime_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url = None
    url_pattern = re.compile(r'URL:\s+(https?://[^\s]+)')
    with open(BATCH_INFERENCE_LOG_FILE, "w") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            stripped = line.rstrip()
            if stripped:
                print(f"     {DIM}{stripped}{RESET}", flush=True)
            match = url_pattern.search(stripped)
            if match:
                url = match.group(1).rstrip("/")
                break
            if proc.poll() is not None:
                break

    if not url:
        print(f"  ✗  Batch Inference printed no URL. Check {BATCH_INFERENCE_LOG_FILE}")
        proc.terminate()
        if gradio_proc:
            gradio_proc.terminate()
        sys.exit(1)

    host_ip = _detect_host_ip()
    if host_ip:
        url = url.replace("0.0.0.0", host_ip).replace("127.0.0.1", host_ip)

    _probe_ok = False
    for _probe_attempt in range(10):
        try:
            urllib.request.urlopen(f"http://localhost:{BATCH_INFERENCE_PORT}/api/state", timeout=3)
            _probe_ok = True
            break
        except Exception:
            time.sleep(2)

    if not _probe_ok:
        print("  ✗  Batch Inference launched but /api/state probe failed after 20s.")
        print(f"     Check {BATCH_INFERENCE_LOG_FILE} for errors.")
        proc.terminate()
        if gradio_proc:
            gradio_proc.terminate()
        sys.exit(1)

    for _path in (BATCH_INFERENCE_URL_FILE, BATCH_INFERENCE_LIVE_FLAG):
        with open(_path, "w") as f:
            f.write(url + "\n")

    ok("Batch Inference frontend is live")
    import threading as _thr
    _thr.Thread(target=_drain_stdout, args=(proc.stdout, BATCH_INFERENCE_LOG_FILE), daemon=True).start()
    return proc, url

if FRONTEND in ("batch_inference", "fiftyone"):
    gradio_proc, gradio_url = _launch_gradio(sidecar=True)
    proc, url = _launch_batch_inference(gradio_proc=gradio_proc)
    STEPS_DONE.append(9)
    print_dashboard()

    fiftyone_port = os.environ.get("FIFTYONE_PORT", "5151")
    print(flush=True)
    print(f"{BOLD}{'─'*62}{RESET}", flush=True)
    print(f"{BOLD}  Cosmos BYO Video Batch Inference — Ready{RESET}", flush=True)
    print(f"{'─'*62}", flush=True)
    print(f"  {BOLD}Batch Inference URL:{RESET}  {hyperlink(url)}", flush=True)
    print(f"  {BOLD}Gradio URL:{RESET}         {hyperlink(gradio_url)}", flush=True)
    print(f"  {DIM}Dataset smoke default: pjramg/Safe_Unsafe_Test{RESET}", flush=True)
    print(f"  {DIM}Results: /tmp/byo_video_batch_inference_results.json{RESET}", flush=True)
    print(f"  {DIM}FiftyOne port: {fiftyone_port} when opened from the UI{RESET}", flush=True)
    if os.path.exists("/tmp/byo_video_runtime_guide.py"):
        print(f"  {DIM}Claude CLI guide: python3 /tmp/byo_video_runtime_guide.py --url {url} wizard{RESET}", flush=True)
    print(f"{'─'*62}", flush=True)
    print(flush=True)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        gradio_proc.terminate()
        proc.wait()
        gradio_proc.wait()
    sys.exit(0)

if USE_REASON_VITE:
    gradio_proc, gradio_url = _launch_gradio(sidecar=True)
    proc, url = _launch_reason_vite()
    batch_proc = None
    batch_url = None
    if LAUNCH_BATCH_INFERENCE_COMPANION:
        batch_proc, batch_url = _launch_batch_inference(gradio_proc=gradio_proc)

    ok("Reason Vite primary UI up; Gradio fallback remains live")
    STEPS_DONE.append(9)
    print_dashboard()

    print(flush=True)
    print(f"{BOLD}{'─'*62}{RESET}", flush=True)
    print(f"{BOLD}  Cosmos Reason {MODEL_SIZE} Vite Demo — Ready{RESET}", flush=True)
    print(f"{'─'*62}", flush=True)
    print(f"  {BOLD}Vite URL:{RESET}    {hyperlink(url)}", flush=True)
    print(f"  {BOLD}Gradio URL:{RESET}  {hyperlink(gradio_url)}", flush=True)
    if batch_url:
        print(f"  {BOLD}Batch Inference URL:{RESET}  {hyperlink(batch_url)}", flush=True)
    print(f"  {DIM}Model: {MODEL_ID}{RESET}", flush=True)
    print(f"  {DIM}Vite log: /tmp/nvidia_build_reason_vite.log{RESET}", flush=True)
    print(f"  {DIM}If direct access is blocked: ssh -L {REASON_VITE_PORT}:localhost:{REASON_VITE_PORT} <user@host>{RESET}", flush=True)
    print(f"{'─'*62}", flush=True)
    print(flush=True)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        gradio_proc.terminate()
        if batch_proc:
            batch_proc.terminate()
        proc.wait()
        gradio_proc.wait()
        if batch_proc:
            batch_proc.wait()
    sys.exit(0)

if USE_PREDICT_VITE:
    gradio_proc, gradio_url = _launch_gradio(sidecar=True)
    proc, url = _launch_predict_vite()
    batch_proc = None
    batch_url = None
    if LAUNCH_BATCH_INFERENCE_COMPANION:
        batch_proc, batch_url = _launch_batch_inference(gradio_proc=gradio_proc)

    ok("Predict Vite primary UI up; Gradio fallback remains live")
    STEPS_DONE.append(9)
    print_dashboard()

    print(flush=True)
    print(f"{BOLD}{'─'*62}{RESET}", flush=True)
    print(f"{BOLD}  Cosmos Predict {MODEL_SIZE} Vite Demo — Ready{RESET}", flush=True)
    print(f"{'─'*62}", flush=True)
    print(f"  {BOLD}Predict Vite URL:{RESET}  {hyperlink(url)}", flush=True)
    print(f"  {BOLD}Gradio URL:{RESET}        {hyperlink(gradio_url)}", flush=True)
    if batch_url:
        print(f"  {BOLD}Batch Inference URL:{RESET}  {hyperlink(batch_url)}", flush=True)
    print(f"  {DIM}Model: {MODEL_ID}{RESET}", flush=True)
    print(f"  {DIM}Vite log: /tmp/nvidia_build_predict_vite.log{RESET}", flush=True)
    print(f"  {DIM}If direct access is blocked: ssh -L {PREDICT_VITE_PORT}:localhost:{PREDICT_VITE_PORT} <user@host>{RESET}", flush=True)
    print(f"{'─'*62}", flush=True)
    print(flush=True)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        gradio_proc.terminate()
        if batch_proc:
            batch_proc.terminate()
        proc.wait()
        gradio_proc.wait()
        if batch_proc:
            batch_proc.wait()
    sys.exit(0)

proc, url = _launch_gradio(sidecar=False)

ok("Demo server up, public tunnel established")
STEPS_DONE.append(9)
print_dashboard()

# ── Final: print clickable hyperlink ────────────────────────────────────────
print(flush=True)
print(f"{BOLD}{'─'*62}{RESET}", flush=True)
print(f"{BOLD}  Cosmos Reason2 {MODEL_SIZE} Demo — Ready{RESET}", flush=True)
print(f"{'─'*62}", flush=True)
print(f"  {BOLD}URL:{RESET}  {hyperlink(url)}", flush=True)
print(f"  {DIM}Upload any MP4 → select checkpoint → Run Inference{RESET}", flush=True)
print(f"  {DIM}Run All Variants: {_variant_labels}{RESET}", flush=True)
print(f"  {DIM}Results: /tmp/byo_video_reason2_results.json{RESET}", flush=True)
print(f"  {DIM}Benchmark: /tmp/byo_video_benchmark.json{RESET}", flush=True)
if os.path.exists("/tmp/byo_video_runtime_guide.py"):
    print(f"  {DIM}Claude CLI guide: python3 /tmp/byo_video_runtime_guide.py --url {url} wizard{RESET}", flush=True)
print(f"  {DIM}Link valid for 72h. Kill instance when done.{RESET}", flush=True)
if _cfg["nim"] and not NGC_API_KEY:
    print(f"  {YELLOW}⚠  NIM-{MODEL_SIZE} mode requires NGC_API_KEY=nvapi-...{RESET}", flush=True)
print(f"{'─'*62}", flush=True)
print(flush=True)

# Stay alive while Gradio runs — without this, the subprocess gets SIGHUP when
# the screen session's controlling process exits.
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    proc.wait()
