#!/usr/bin/env python3
"""
Cosmos Reason2 — BYO Video Demo setup + launch.
Runs on the GPU instance. Prints live progress with ETAs.
At the end, prints a clickable OSC 8 hyperlink to the Gradio URL.
"""
import os, sys, time, subprocess, re, shutil, json

# ── ANSI helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET}  {msg}", flush=True)
def run(msg):   print(f"  {YELLOW}⟳{RESET}  {msg}", flush=True)
def info(msg):  print(f"  {CYAN}→{RESET}  {msg}", flush=True)
def header(msg): print(f"\n{BOLD}{msg}{RESET}", flush=True)

def hyperlink(url, label=None):
    """OSC 8 clickable hyperlink — works in iTerm2, Terminal.app, most modern terminals."""
    label = label or url
    return f"\033]8;;{url}\033\\{BOLD}{CYAN}{label}{RESET}\033]8;;\033\\"

def run_cmd(args, cwd=None, env=None, timeout=None):
    """Run a command, return (returncode, stdout+stderr)."""
    result = subprocess.run(
        args, cwd=cwd, env=env, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return result.returncode, result.stdout

def stream_cmd(args, cwd=None, env=None, prefix=""):
    """Stream command output with a prefix, return returncode."""
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

# ── Config ──────────────────────────────────────────────────────────────────
HOME        = os.path.expanduser("~")
PATH_EXTRA  = f"{HOME}/.local/bin:{HOME}/.cargo/bin"
ENV         = {**os.environ, "PATH": f"{PATH_EXTRA}:{os.environ.get('PATH', '')}",
               "PYTHONUNBUFFERED": "1"}
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
MODEL_NAME  = os.environ.get("MODEL_NAME", "nvidia/Cosmos-Reason2-2B")
MODEL_DIR   = os.environ.get("MODEL_DIR", f"{HOME}/cosmos-reason2/models/Cosmos-Reason2-2B")
REASON2_DIR = f"{HOME}/cosmos-reason2"
COOKBOOK_DIR= f"{HOME}/cosmos-cookbook"
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7860"))
GRADIO_APP  = "/tmp/gradio_cr2_byo.py"
URL_FILE    = "/tmp/gradio_url.txt"
LOG_FILE    = "/tmp/gradio_demo.log"

# ── Step 1: GPU check ────────────────────────────────────────────────────────
header("Step 1 — GPU")
rc, out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
                   "--format=csv,noheader"])
if rc != 0:
    print(f"  ✗  nvidia-smi failed — no GPU detected", flush=True)
    sys.exit(1)
gpu_line = out.strip().splitlines()[0]
parts = [p.strip() for p in gpu_line.split(",")]
gpu_name = parts[0]
vram_free = int(parts[1].split()[0])
vram_total = int(parts[2].split()[0])

LOW_VRAM = vram_free < 24000

if vram_free >= 80000:
    model_label = "CR2-8B (80GB+ VRAM)"
    if "8B" not in MODEL_NAME:
        MODEL_NAME = "nvidia/Cosmos-Reason2-8B"
        MODEL_DIR = f"{HOME}/cosmos-reason2/models/Cosmos-Reason2-8B"
elif vram_free >= 40000:
    model_label = "CR2-2B (40GB+ VRAM)"
elif vram_free >= 24000:
    model_label = "CR2-2B (below recommended 40GB — will use fps=1)"
    LOW_VRAM = True
else:
    model_label = f"CR2-2B ⚠ LOW VRAM ({vram_free}MiB / 12GB) — fps=1, reduced resolution"
    LOW_VRAM = True

ok(f"{gpu_name}  {vram_free:,} MiB free / {vram_total:,} MiB total")
ok(f"Selected model: {MODEL_NAME} ({model_label})")
if LOW_VRAM:
    info("LOW VRAM MODE: fps=1, resolution reduced — use short clips (< 60s) for best results")

# ── Step 2: HF token ─────────────────────────────────────────────────────────
header("Step 2 — HuggingFace auth")
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

# ── Step 3: uv ───────────────────────────────────────────────────────────────
header("Step 3 — uv package manager")
rc, _ = run_cmd(["uv", "--version"], env=ENV)
if rc == 0:
    _, ver = run_cmd(["uv", "--version"], env=ENV)
    ok(f"uv already installed ({ver.strip()})")
else:
    run("Installing uv  (~10s)")
    t0 = time.time()
    rc = stream_cmd(
        ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        env=ENV
    )
    if rc != 0:
        print("  ✗  uv install failed"); sys.exit(1)
    ok(f"uv installed in {time.time()-t0:.0f}s")

# ── Step 4: cosmos-reason2 repo ──────────────────────────────────────────────
header("Step 4 — cosmos-reason2 repo")
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

# ── Step 5: Python dependencies ──────────────────────────────────────────────
header("Step 5 — Python dependencies (uv sync)")
venv_marker = f"{REASON2_DIR}/.venv/lib"
if os.path.exists(venv_marker):
    ok("virtualenv already present — skipping uv sync")
else:
    run("Running uv sync --extra cu128  (~2-3 min)")
    t0 = time.time()
    rc, out = run_cmd(["uv", "sync", "--extra", "cu128"], cwd=REASON2_DIR, env=ENV, timeout=600)
    if rc != 0:
        run("cu128 failed, trying uv sync without extras")
        rc, out = run_cmd(["uv", "sync"], cwd=REASON2_DIR, env=ENV, timeout=600)
    if rc != 0:
        print("  ✗  uv sync failed:", out[-500:]); sys.exit(1)
    ok(f"Dependencies installed in {time.time()-t0:.0f}s")

# ── Step 6: PyAV ─────────────────────────────────────────────────────────────
header("Step 6 — PyAV video backend (av==16.1.0)")
rc, av_check = run_cmd(
    ["uv", "run", "python", "-c", "import av; print(av.__version__)"],
    cwd=REASON2_DIR, env=ENV
)
if rc == 0 and "16.1.0" in av_check:
    ok(f"PyAV already installed ({av_check.strip()})")
else:
    run("Installing av==16.1.0  (~5s)")
    t0 = time.time()
    rc, out = run_cmd(["uv", "pip", "install", "av==16.1.0"], cwd=REASON2_DIR, env=ENV)
    if rc != 0:
        print("  ✗  av install failed:", out); sys.exit(1)
    ok(f"PyAV installed in {time.time()-t0:.0f}s")

# ── Step 7: Gradio ───────────────────────────────────────────────────────────
header("Step 7 — Gradio UI library")
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

# ── Step 8: Model weights ─────────────────────────────────────────────────────
header("Step 8 — Model weights")
model_marker = os.path.join(MODEL_DIR, "model.safetensors")
if os.path.exists(model_marker):
    size_mb = os.path.getsize(model_marker) // (1024*1024)
    ok(f"Weights already downloaded ({MODEL_NAME}, {size_mb:,} MB main shard)")
else:
    size_hint = "~8 GB" if "2B" in MODEL_NAME else "~16 GB"
    run(f"Downloading {MODEL_NAME} from HuggingFace  ({size_hint}, ~10-15 min on first run)")
    info("Progress below — download continues even if it looks stalled:")
    t0 = time.time()
    dl_env = {**ENV, "HF_TOKEN": HF_TOKEN}
    rc = stream_cmd(
        ["uv", "run", "huggingface-cli", "download", MODEL_NAME,
         "--local-dir", MODEL_DIR],
        cwd=REASON2_DIR, env=dl_env, prefix="HF │ "
    )
    if rc != 0:
        print("  ✗  Model download failed. Check HF_TOKEN and model access."); sys.exit(1)
    elapsed = time.time() - t0
    ok(f"Downloaded in {elapsed/60:.1f} min")

# ── Step 9: Launch Gradio ────────────────────────────────────────────────────
header("Step 9 — Launch Gradio web demo")

# Kill any old Gradio on this port
subprocess.run(["bash", "-c", f"fuser -k {GRADIO_PORT}/tcp 2>/dev/null || true"])

if not os.path.exists(GRADIO_APP):
    print(f"  ✗  {GRADIO_APP} not found — deploy gradio_cr2_byo.py first"); sys.exit(1)

# Remove stale URL file
if os.path.exists(URL_FILE):
    os.remove(URL_FILE)

launch_env = {
    **ENV,
    "MODEL_DIR": MODEL_DIR,
    "MODEL_NAME": MODEL_NAME,
    "GRADIO_PORT": str(GRADIO_PORT),
    "GRADIO_SHARE": "true",
    "PYTHONUNBUFFERED": "1",
    "HF_TOKEN": HF_TOKEN,
    "LOW_VRAM": "true" if LOW_VRAM else "false",
}

run(f"Starting Cosmos Reason2 demo on port {GRADIO_PORT}  (~5-10s for model load)")

proc = subprocess.Popen(
    ["uv", "run", "python", "/tmp/gradio_cr2_byo.py"],
    cwd=REASON2_DIR,
    env=launch_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

# Stream output and capture URL
url = None
url_pattern = re.compile(r'(https?://[^\s"\']+gradio\.live[^\s"\']*)')
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
            break  # Got the URL — model is up

if not url:
    print("  ✗  Gradio did not print a public URL. Check /tmp/gradio_demo.log")
    sys.exit(1)

with open(URL_FILE, "w") as f:
    f.write(url + "\n")

ok(f"Demo server up, public tunnel established")

# ── Final: print clickable hyperlink ────────────────────────────────────────
print(flush=True)
print(f"{BOLD}{'─'*60}{RESET}", flush=True)
print(f"{BOLD}  Cosmos Reason2 Demo — Ready{RESET}", flush=True)
print(f"{'─'*60}", flush=True)
print(f"  {BOLD}URL:{RESET}  {hyperlink(url)}", flush=True)
print(f"  {DIM}Upload any MP4 → type a prompt → click Run Inference{RESET}", flush=True)
print(f"  {DIM}Results also saved to /tmp/byo_video_reason2_results.json{RESET}", flush=True)
print(f"  {DIM}Link valid for 72h. Kill instance when done.{RESET}", flush=True)
print(f"{'─'*60}", flush=True)
print(flush=True)

# Keep process alive in background
proc.stdout.close()
