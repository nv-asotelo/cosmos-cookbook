#!/usr/bin/env python3
"""
Cosmos BYO Video Demo setup + launch.
Version: 2026-04-28
Canonical source: ~/.claude/scripts/byo_video_setup.py

Runs on the GPU instance. Prints live progress with ETAs.
MODEL_SIZE-driven: downloads all variants for the selected model size.
At the end, prints a clickable OSC 8 hyperlink to the Gradio URL.
URL is also written to /tmp/gradio_url.txt for agent capture.

Env vars:
  HF_TOKEN          — required for gated model download (checks ~/.cache/huggingface/token if not set)
  NGC_API_KEY       — required for NIM mode (nvapi-... prefix, 8B only; not needed for Cosmos3)
  MODEL_SIZE        — 2B | 8B | 32B | C3-2B | C3-8B  (default: C3-2B)
  MODEL_DIR         — override local download path for primary model
  GRADIO_PORT       — port for Gradio (default: 7860)
  SKIP_HF_PRELOAD   — set to 1 to skip HF model preload at Gradio startup (auto in vLLM mode)
  VLLM_MAX_MODEL_LEN — max context length for vLLM (default: 32768; do not reduce below 32768 for video)
"""
import os, sys, time, subprocess, re, shutil, json

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
    # ── Cosmos3-Reasoner (private gated — HF_TOKEN with nvidia org required) ──
    "C3-2B": {
        "variants": [
            ("C3R-2B BF16", "Cosmos3-Reasoner-2B", "nvidia/Cosmos3-Reasoner-2B-Private", "~TBD"),
        ],
        "nim": None,
    },
    "C3-8B": {
        "variants": [
            ("C3R-8B BF16", "Cosmos3-Reasoner-8B", "nvidia/Cosmos3-Reasoner-8B-Private", "~TBD"),
        ],
        "nim": None,
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
            ("CR2-32B BF16", "Cosmos-Reason2-32B",    "nvidia/Cosmos-Reason2-32B",    "~64 GB"),
            ("CR2-32B AV",   "Cosmos-Reason2-32B-AV", "nvidia/Cosmos-Reason2-32B-AV", "~64 GB"),
        ],
        "nim": None,
    },
}

# ── Config ──────────────────────────────────────────────────────────────────
HOME          = os.path.expanduser("~")
PATH_EXTRA    = f"{HOME}/.local/bin:{HOME}/.cargo/bin"
ENV           = {**os.environ, "PATH": f"{PATH_EXTRA}:{os.environ.get('PATH', '')}",
                 "PYTHONUNBUFFERED": "1"}
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
NGC_API_KEY   = os.environ.get("NGC_API_KEY", "")
MODEL_SIZE    = os.environ.get("MODEL_SIZE", "C3-2B").upper()
# Cosmos3-Reasoner uses cosmos-reason2 working dir until a dedicated repo is published.
# Set COSMOS_DIR env var to override if the repo path changes.
REASON2_DIR   = os.environ.get("COSMOS_DIR", f"{HOME}/cosmos-reason2")
MODELS_BASE   = f"{REASON2_DIR}/models"
GRADIO_PORT   = int(os.environ.get("GRADIO_PORT", "7860"))
GRADIO_APP    = "/tmp/gradio_cr2_byo.py"
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "hf")
URL_FILE      = "/tmp/gradio_url.txt"
LOG_FILE      = "/tmp/gradio_demo.log"
# MAXLEN-001: 32768 is the minimum required for video queries. Do not reduce below this.
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))

if MODEL_SIZE not in _MODEL_CONFIGS:
    print(f"  ✗  MODEL_SIZE={MODEL_SIZE} not supported. Use C3-2B, C3-8B, C3-32B, 2B, 8B, or 32B.")
    sys.exit(1)

_cfg = _MODEL_CONFIGS[MODEL_SIZE]

# Primary variant (first in list) drives MODEL_DIR/MODEL_NAME defaults
_primary_label, _primary_dirname, _primary_hf_id, _ = _cfg["variants"][0]
MODEL_DIR  = os.environ.get("MODEL_DIR",  f"{MODELS_BASE}/{_primary_dirname}")
MODEL_NAME = _primary_hf_id

# ── Pre-step: kill old Gradio so VRAM measurement is accurate ────────────────
subprocess.run(["bash", "-c", f"fuser -k {GRADIO_PORT}/tcp 2>/dev/null || true"])
time.sleep(2)

# ── Step 1: GPU check ─────────────────────────────────────────────────────────
header("Step 1 — GPU detect", eta="<5s")
rc, out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
                   "--format=csv,noheader"])
if rc != 0:
    print("  ✗  nvidia-smi failed — no GPU detected", flush=True)
    sys.exit(1)
gpu_line   = out.strip().splitlines()[0]
parts_     = [p.strip() for p in gpu_line.split(",")]
gpu_name   = parts_[0]
vram_free  = int(parts_[1].split()[0])
vram_total = int(parts_[2].split()[0])

LOW_VRAM = vram_free < 24000

_gpu_upper = gpu_name.upper()
_h100_class = any(tag in _gpu_upper for tag in ("H100", "A100", "H200", "GB200"))
if _h100_class and vram_free >= 60000:
    tier_name   = "H100/A100"
    gradio_fps  = 8
    max_pixels  = 1048576
    prefill_tps = 90
elif vram_free >= 40000:
    tier_name   = "high-VRAM"
    gradio_fps  = 8
    max_pixels  = 524288
    prefill_tps = 23
else:
    tier_name   = "low-VRAM (<40GB)"
    gradio_fps  = 4
    max_pixels  = 131072
    prefill_tps = 15
    LOW_VRAM    = True

_variant_labels = " → ".join(lbl for lbl, _, _, _ in _cfg["variants"])
if _cfg["nim"]:
    _variant_labels += f" → NIM-{MODEL_SIZE}"

ok(f"{gpu_name}  {vram_free:,} MiB free / {vram_total:,} MiB total")
ok(f"MODEL_SIZE: {MODEL_SIZE}  |  variants: {_variant_labels}")
ok(f"VRAM tier: {tier_name}  |  fps={gradio_fps}, max_pixels={max_pixels:,}, prefill_tps={prefill_tps}")

# ── Step 2: HF token ──────────────────────────────────────────────────────────
header("Step 2 — HuggingFace auth", eta="<5s")
hf_cache = os.path.expanduser("~/.cache/huggingface/token")
if HF_TOKEN:
    ok(f"HF_TOKEN set ({len(HF_TOKEN)} chars)")
elif os.path.exists(hf_cache):
    ok(f"HF token found at ~/.cache/huggingface/token")
    with open(hf_cache) as f:
        HF_TOKEN = f.read().strip()
    ENV["HF_TOKEN"] = HF_TOKEN
else:
    print("  ✗  HF_TOKEN not set and no cached token found.")
    print("     Run: export HF_TOKEN=hf_... and re-run this script.")
    sys.exit(1)

# ── Step 3: NGC API key check (for NIM mode) ──────────────────────────────────
header("Step 3 — NGC API key (NIM mode)", eta="<5s")
if _cfg["nim"]:
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

# ── Step 5: cosmos-reason2 repo ───────────────────────────────────────────────
header("Step 5 — cosmos-reason2 repo", eta="<5s if cached, ~15s first time")
if os.path.exists(f"{REASON2_DIR}/.git"):
    ok(f"cosmos-reason2 already cloned at {REASON2_DIR}")
else:
    run("Cloning cosmos-reason2  (~15s)")
    t0 = time.time()
    rc = stream_cmd(
        ["git", "clone", "https://github.com/nvidia-cosmos/cosmos-reason2.git", REASON2_DIR],
        env=ENV
    )
    if rc != 0:
        print("  ✗  git clone failed"); sys.exit(1)
    ok(f"Cloned in {time.time()-t0:.0f}s")

# ── Step 6: Python dependencies ───────────────────────────────────────────────
header("Step 6 — Python dependencies (uv sync)", eta="<5s if cached, ~2-3 min first time")
venv_marker = f"{REASON2_DIR}/.venv/lib"
if os.path.exists(venv_marker):
    ok("virtualenv already present — skipping uv sync")
else:
    _extras = os.environ.get("COSMOS_EXTRAS", "cu128")
    run(f"Running uv sync --extra {_extras}  (~2-3 min)")
    t0 = time.time()
    rc, out = run_cmd(["uv", "sync", "--extra", _extras], cwd=REASON2_DIR, env=ENV, timeout=600)
    if rc != 0:
        run(f"{_extras} failed, trying uv sync without extras")
        rc, out = run_cmd(["uv", "sync"], cwd=REASON2_DIR, env=ENV, timeout=600)
    if rc != 0:
        print("  ✗  uv sync failed:", out[-500:]); sys.exit(1)
    ok(f"Dependencies installed in {time.time()-t0:.0f}s")

# ── Step 6b: CUDA version gate for vLLM ──────────────────────────────────────
# vLLM 0.12.0 (cu128 extra) requires CUDA driver >= 12.9 (driver >= 575).
# On CUDA 12.8 (Hyperstack H100, driver 570.x), PTX compilation fails with
# cudaErrorUnsupportedPtxVersion — --enforce-eager does NOT fix this.
# Fix: detect driver version and fall back to HF Transformers when incompatible.
_driver_major = None
try:
    import subprocess as _sp_cuda
    _smi_out = _sp_cuda.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=10, text=True
    ).strip()
    _driver_major = int(_smi_out.split(".")[0])
except Exception:
    pass

_CUDA_VLLM_OK = (_driver_major is None) or (_driver_major >= 575)

if INFERENCE_BACKEND == "vllm" and not _CUDA_VLLM_OK:
    header("Step 6b — CUDA PTX gate (BUG-002 prevention)", eta="<5s")
    warn(f"CUDA driver {_driver_major}.x detected — vLLM 0.12.0 requires driver >= 575 (CUDA 12.9).")
    warn("Forcing INFERENCE_BACKEND=hf to avoid cudaErrorUnsupportedPtxVersion crash.")
    warn("Use MassedCompute (CUDA 13.0) for vLLM inference on C3-8B / C3-32B.")
    INFERENCE_BACKEND = "hf"
elif INFERENCE_BACKEND == "vllm":
    ok(f"CUDA driver {_driver_major}.x — vLLM 0.12.0 compatible")

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

# ── Step 8: Gradio + requests ─────────────────────────────────────────────────
header("Step 8 — Gradio + requests", eta="<5s if cached, ~30s first time")
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
            time.sleep(30)
        else:
            print(f"  ✗  All {MAX_RETRIES} attempts failed for {model_name}.")
            return False
    elapsed = time.time() - t0
    ok(f"Downloaded {model_name} in {elapsed/60:.1f} min")
    return True


dl_env = {**ENV, "HF_TOKEN": HF_TOKEN}

for i, (var_label, var_dirname, var_hf_id, var_size) in enumerate(_cfg["variants"]):
    step_label = f"Step 9{'abcde'[i]} — {var_label} weights ({var_dirname})"
    header(step_label, eta=f"<5s if cached, longer first time ({var_size})")
    var_dir = os.path.join(MODELS_BASE, var_dirname)
    ok_ = download_model(var_hf_id, var_dir, var_size, dl_env)
    if not ok_ and i == 0:
        sys.exit(1)  # primary variant is required
    elif not ok_:
        warn(f"{var_label} download failed — will fall back to HF on first Gradio use")

# ── Step 9z: NemotronVL architecture gate (BUG-001 prevention) ───────────────
# NemotronVLForConditionCausalLM (model_type=nemotron_siglip2, e.g. C3-2B) is not
# registered in vLLM 0.12.0 or 0.9.2. Force HF Transformers for these architectures.
_primary_config = os.path.join(MODEL_DIR, "config.json")
if os.path.exists(_primary_config):
    with open(_primary_config) as _cf:
        _detected_model_type = json.load(_cf).get("model_type", "qwen3_vl")
    if INFERENCE_BACKEND == "vllm" and _detected_model_type != "qwen3_vl":
        warn(f"model_type={_detected_model_type!r} not supported by vLLM — forcing INFERENCE_BACKEND=hf")
        warn("BUG-001: NemotronVL arch not registered. Use HF Transformers via AutoModelForCausalLM.")
        INFERENCE_BACKEND = "hf"
    elif os.path.exists(_primary_config):
        ok(f"model_type={_detected_model_type!r} — vLLM compatible")

# ── Step 9b: Auto-launch vLLM before Gradio (when INFERENCE_BACKEND=vllm) ────
# BUG-003 fix: byo_video_setup.py previously completed without launching the vLLM
# server, causing Gradio to start in vllm mode with no backend available.
_vllm_proc = None
if INFERENCE_BACKEND == "vllm":
    header("Step 9b — Launch vLLM server (BUG-003 fix)", eta="~5-10 min for model load")
    import socket as _socket

    _vllm_port = int(os.environ.get("VLLM_PORT", "8000"))
    _vllm_model_id = os.environ.get("VLLM_SERVED_MODEL_NAME", MODEL_NAME)
    _vllm_extra = _cfg.get("vllm_extra_flags", [])
    _vllm_max_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))

    # C3-32B: use tighter max_model_len to fit H100 80GB; cap FPS in Gradio
    if MODEL_SIZE == "C3-32B":
        _vllm_max_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384"))

    _vllm_cmd = [
        f"{REASON2_DIR}/.venv/bin/vllm", "serve", MODEL_DIR,
        "--served-model-name", _vllm_model_id,
        "--port", str(_vllm_port),
        "--dtype", "auto",
        "--trust-remote-code",
        "--enforce-eager",
        "--max-model-len", str(_vllm_max_len),
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.95",
    ]
    _vllm_cmd.extend(_vllm_extra)

    _vllm_log = f"/tmp/vllm_{MODEL_SIZE.lower()}.log"
    run(f"Starting vLLM server (log: {_vllm_log})")
    info(f"Command: {' '.join(_vllm_cmd)}")
    with open(_vllm_log, "w") as _vf:
        _vllm_proc = subprocess.Popen(
            _vllm_cmd, cwd=REASON2_DIR, env={**ENV, "HF_TOKEN": HF_TOKEN,
                                              "VLLM_VIDEO_LOADER_BACKEND": "opencv",
                                              "SKIP_HF_PRELOAD": "1"},
            stdout=_vf, stderr=subprocess.STDOUT
        )

    # Wait up to 10 min for vLLM to be ready on the port
    _vllm_ready = False
    _vllm_timeout = 600
    _t0_vllm = time.time()
    info(f"Waiting up to {_vllm_timeout}s for vLLM to bind port {_vllm_port} ...")
    while time.time() - _t0_vllm < _vllm_timeout:
        try:
            with _socket.create_connection(("localhost", _vllm_port), timeout=2):
                _vllm_ready = True
                break
        except OSError:
            time.sleep(5)
        if _vllm_proc.poll() is not None:
            print(f"  ✗  vLLM process exited early. Check {_vllm_log}")
            sys.exit(1)

    if not _vllm_ready:
        print(f"  ✗  vLLM did not bind port {_vllm_port} within {_vllm_timeout}s. Check {_vllm_log}")
        sys.exit(1)

    ok(f"vLLM server ready on port {_vllm_port} ({time.time()-_t0_vllm:.0f}s)")

# ── Step 10: Launch Gradio ────────────────────────────────────────────────────
header("Step 10 — Launch Gradio web demo", eta="~5-10s for model load")

if not os.path.exists(GRADIO_APP):
    print(f"  ✗  {GRADIO_APP} not found — deploy gradio_cr2_byo.py first"); sys.exit(1)

if os.path.exists(URL_FILE):
    os.remove(URL_FILE)

launch_env = {
    **ENV,
    "MODEL_SIZE":         MODEL_SIZE,
    "MODEL_DIR":          MODEL_DIR,
    "MODEL_NAME":         MODEL_NAME,
    "GRADIO_PORT":        str(GRADIO_PORT),
    "GRADIO_SHARE":       "true",
    "PYTHONUNBUFFERED":   "1",
    "HF_TOKEN":           HF_TOKEN,
    "NGC_API_KEY":        NGC_API_KEY,
    "LOW_VRAM":           "true" if LOW_VRAM else "false",
    "GRADIO_FPS":         str(gradio_fps),
    "GRADIO_MAX_PIXELS":  str(max_pixels),
    "GRADIO_PREFILL_TPS": str(prefill_tps),
    "INFERENCE_BACKEND":  INFERENCE_BACKEND,
    "VLLM_BASE_URL":      os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    "VLLM_API_KEY":       os.environ.get("VLLM_API_KEY", "EMPTY"),
    "COSMOS_EXTRAS":      os.environ.get("COSMOS_EXTRAS", "cu128"),
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    # MAXLEN-001: always pass explicitly — never rely on vLLM default (8192 breaks video queries)
    "VLLM_MAX_MODEL_LEN": str(VLLM_MAX_MODEL_LEN),
    # PRELOAD-001: skip HF preload when using vLLM backend
    "SKIP_HF_PRELOAD":    "1" if INFERENCE_BACKEND == "vllm" else "0",
}

run(f"Starting Cosmos Reason2 {MODEL_SIZE} demo on port {GRADIO_PORT}")

proc = subprocess.Popen(
    ["uv", "run", "python", "-u", "/tmp/gradio_cr2_byo.py"],
    cwd=REASON2_DIR,
    env=launch_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

url = None
url_pattern        = re.compile(r'(https?://[^\s"\']+gradio\.live[^\s"\']*)')
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
        if time.time() - t_launch > URL_CAPTURE_TIMEOUT:
            print(f"  ✗  Timed out after {URL_CAPTURE_TIMEOUT}s waiting for Gradio URL.")
            proc.terminate()
            sys.exit(1)

if not url:
    print("  ✗  Gradio did not print a public URL. Check /tmp/gradio_demo.log")
    sys.exit(1)

with open(URL_FILE, "w") as f:
    f.write(url + "\n")

ok("Demo server up, public tunnel established")

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
print(f"  {DIM}Link valid for 72h. Kill instance when done.{RESET}", flush=True)
if _cfg["nim"] and not NGC_API_KEY:
    print(f"  {YELLOW}⚠  NIM-{MODEL_SIZE} mode requires NGC_API_KEY=nvapi-...{RESET}", flush=True)
print(f"{'─'*62}", flush=True)
print(flush=True)

proc.stdout.close()
# BUG-010: keep parent alive so Gradio subprocess doesn't get SIGHUP when
# the setup script exits from within a screen session.
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    proc.wait()
