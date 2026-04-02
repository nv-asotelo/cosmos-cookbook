# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
harness_benchmark.py — Monitored inference harness for the Worker Safety recipe.

Solves the "ran all night unmonitored" problem:
  - Real-time per-video progress with ETA
  - Rolling 5-video throughput window
  - Watchdog: warns at 8 min of no progress, saves + exits at 15 min
  - Incremental JSON save after every video
  - Completion banner
  - File-based log for tail -f monitoring from a second terminal
  - SIGINT / SIGTERM handler for clean partial saves

Usage
-----
# Standard float16 model:
python3 harness_benchmark.py

# Embedl W4A16 quantized (AutoAWQ):
python3 harness_benchmark.py --model embedl/Cosmos-Reason2-2B-W4A16-Edge2-FlashHead

# Custom dataset or local videos:
python3 harness_benchmark.py --hf-dataset pjramg/Safe_Unsafe_Test
python3 harness_benchmark.py --video-dir ~/my_videos/

# Tune watchdog thresholds (minutes):
python3 harness_benchmark.py --watchdog-warn 5 --watchdog-kill 12

Monitor from a second terminal:
  tail -f ~/inference_harness.log
"""

# ── TORCHDYNAMO_DISABLE must be set before any torch import ──────────────────
# Without this, quantized checkpoints trigger torch._inductor JIT compilation:
# 16 compile_worker processes, GPU at <40% utilization, 0 videos in 4+ hours.
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import collections
import json
import pathlib
import signal
import sys
import threading
import time

# ── Prompts replicated from worker_safety.py ─────────────────────────────────
SYSTEM_INSTRUCTIONS = """
    You are an expert Industrial Safety Inspector monitoring a manufacturing facility.
    Your goal is to classify the video into EXACTLY ONE of the 8 classes defined below.

    CRITICAL NEGATIVE CONSTRAINTS (What to IGNORE):
    1. IGNORE SITTING WORKERS:
       - If a person is SITTING at a machine board working, this is NOT an intervention class. Ignore them.
       - If a person is SITTING driving a forklift, the driver is NOT the class. Focus only on the LOAD carried.
    2. IGNORE BACKGROUND:
       - The facility is old. Do not report hazards based on faded floor markings or unpainted areas.
    3. SINGLE OUTPUT:
       - Even if multiple things happen, choose the MOST PROMINENT behavior.
       - Prioritize UNSAFE behaviors over SAFE behaviors if both are present.
"""

USER_PROMPT_CONTENT = """
    Analyze the video and output a JSON object. You MUST select the class ID and Label EXACTLY from the table below.

    STRICT CLASSIFICATION TABLE (Use these exact IDs and Labels):

    | ID | Label | Definition (Ground Truth) | Hazard Status |
    | :--- | :--- | :--- | :--- |
    | 0 | Safe Walkway Violation | Worker walks OUTSIDE the designated Green Path. | TRUE (Unsafe) |
    | 4 | Safe Walkway | Worker walks INSIDE the designated Green Path. | FALSE (Safe) |
    | 1 | Unauthorized Intervention | Worker interacts with machine board WITHOUT a green vest. | TRUE (Unsafe) |
    | 5 | Authorized Intervention | Worker interacts with machine board WITH a green vest. | FALSE (Safe) |
    | 2 | Opened Panel Cover | Machine panel cover is left OPEN after intervention. | TRUE (Unsafe) |
    | 6 | Closed Panel Cover | Machine panel cover is CLOSED after intervention. | FALSE (Safe) |
    | 3 | Carrying Overload with Forklift | Forklift carries 3 OR MORE blocks. | TRUE (Unsafe) |
    | 7 | Safe Carrying | Forklift carries 2 OR FEWER blocks. | FALSE (Safe) |


    INSTRUCTIONS:
    1. Identify the behavior in the video.
    2. Match it to one row in the table above.
    3. Output the exact "ID" and "Label" from that row. Do not invent new labels like "safe and compliant".

    OUTPUT FORMAT:
    {
      "prediction_class_id": [Integer from Table],
      "prediction_label": "[Exact String from Table]",
      "video_description": "[Concise description of the observed action]",
      "hazard_detection": {
        "is_hazardous": [true/false based on the Hazard Status column],
        "temporal_segment": "[Start Time - End Time] or null"
      }
    }
    """

# Supported video extensions for --video-dir mode
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"
DEFAULT_HF_DATASET = "pjramg/Safe_Unsafe_Test"
DEFAULT_RESULTS = str(pathlib.Path.home() / "inference_results.json")
DEFAULT_LOG = str(pathlib.Path.home() / "inference_harness.log")


# ── Logging — mirror everything to file so `tail -f` works ───────────────────

class Tee:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, log_path: str):
        self._log = open(log_path, "a", buffering=1)  # line-buffered

    def write(self, msg: str):
        sys.stdout.write(msg)
        self._log.write(msg)

    def flush(self):
        sys.stdout.flush()
        self._log.flush()

    def close(self):
        self._log.close()


def _fmt_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Monitored inference harness for the Worker Safety recipe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL_ID",
        help=(
            "HuggingFace model ID or local path. "
            "Use nvidia/Cosmos-Reason2-2B for standard float16. "
            "Append a W4A16 or w4a16 model ID to use AutoAWQ loading."
        ),
    )
    p.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        metavar="DATASET",
        help="HuggingFace dataset slug. Ignored when --video-dir is set.",
    )
    p.add_argument(
        "--video-dir",
        default=None,
        metavar="DIR",
        help="Run inference on a local directory of video files instead of HF dataset.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        metavar="N",
        help=(
            "Max tokens per inference call. "
            "256 is sufficient for structured JSON output and ~20%% faster than 1024."
        ),
    )
    p.add_argument(
        "--fps",
        type=int,
        default=2,
        metavar="N",
        help="Frames per second to sample from each video.",
    )
    p.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        metavar="PATH",
        help="Output JSON path. Written incrementally after every video.",
    )
    p.add_argument(
        "--watchdog-warn",
        type=float,
        default=8.0,
        metavar="MINUTES",
        help="Print a loud WARNING if no video completes within this many minutes.",
    )
    p.add_argument(
        "--watchdog-kill",
        type=float,
        default=15.0,
        metavar="MINUTES",
        help="Save partial results and exit if no video completes within this many minutes.",
    )
    p.add_argument(
        "--log",
        default=DEFAULT_LOG,
        metavar="PATH",
        help="Path for the log file (tail -f from another terminal).",
    )
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def is_w4a16(model_name: str) -> bool:
    return "W4A16" in model_name or "w4a16" in model_name


def load_model(model_name: str):
    """Load model and processor. Uses AutoAWQ for W4A16 checkpoints."""
    import torch
    import transformers

    if is_w4a16(model_name):
        print(f"[HARNESS] W4A16 model detected — loading via AutoAWQ: {model_name}")
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            print(
                "\n[ERROR] AutoAWQ is not installed. Install it with:\n"
                "  pip install autoawq\n"
                "or, inside a uv venv:\n"
                "  uv pip install autoawq\n"
            )
            sys.exit(1)
        model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=True)
        processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
    else:
        print(f"[HARNESS] Loading float16 model: {model_name}")
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=__import__("torch").float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)

    # Pixel token limits (matches worker_safety.py)
    PIXELS_PER_TOKEN = 32 ** 2
    min_vision_tokens, max_vision_tokens = 256, 8192
    processor.image_processor.size = processor.video_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }

    print(f"[HARNESS] Model ready.")
    return model, processor


# ── Video source ──────────────────────────────────────────────────────────────

def collect_video_paths(args) -> list:
    """Return a list of (filepath, ground_truth_label_or_None) tuples."""
    if args.video_dir:
        video_dir = pathlib.Path(args.video_dir).expanduser().resolve()
        if not video_dir.is_dir():
            print(f"[ERROR] --video-dir '{video_dir}' is not a directory.")
            sys.exit(1)
        videos = sorted(
            f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            print(
                f"[ERROR] No video files found in '{video_dir}'. "
                f"Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}"
            )
            sys.exit(1)
        print(f"[HARNESS] Source: local directory  ({len(videos)} videos)")
        return [(str(v), None) for v in videos]
    else:
        print(f"[HARNESS] Source: HuggingFace dataset '{args.hf_dataset}'")
        import fiftyone as fo
        import fiftyone.utils.huggingface as fouh

        # Dataset may already exist from a prior run — load it rather than re-downloading.
        existing = fo.list_datasets()
        if args.hf_dataset in existing:
            print(f"[HARNESS] Existing FiftyOne dataset found — loading '{args.hf_dataset}'")
            dataset = fo.load_dataset(args.hf_dataset)
        elif existing:
            # FiftyOne stored it under its own slug (HF repo ID without org prefix, etc.)
            print(f"[HARNESS] Dataset '{args.hf_dataset}' not found by name; using first available: '{existing[0]}'")
            dataset = fo.load_dataset(existing[0])
        else:
            dataset = fouh.load_from_hub(args.hf_dataset, persistent=True)
        pairs = []
        for sample in dataset:
            gt = None
            try:
                gt_field = sample.get_field("ground_truth")
                if gt_field is not None:
                    gt = getattr(gt_field, "label", None)
            except AttributeError:
                pass
            pairs.append((sample.filepath, gt))
        print(f"[HARNESS] Dataset loaded: {len(pairs)} videos")
        return pairs


# ── Single-video inference ────────────────────────────────────────────────────

def run_inference_on_video(model, processor, video_path: str, fps: int, max_new_tokens: int) -> dict:
    """Run the worker-safety prompt on one video. Returns parsed JSON dict or error dict."""
    import transformers

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTIONS}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": USER_PROMPT_CONTENT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    clean_json = output_text.strip().replace("```json", "").replace("```", "")
    result = json.loads(clean_json)
    return result


# ── Watchdog thread ───────────────────────────────────────────────────────────

class Watchdog:
    """
    Background thread that tracks inference heartbeats.

    - Prints a loud WARNING after warn_minutes of silence.
    - Sets the kill_flag Event after kill_minutes of silence, signalling the
      main loop to save partial results and exit.
    """

    def __init__(self, warn_minutes: float, kill_minutes: float):
        self._warn_sec = warn_minutes * 60
        self._kill_sec = kill_minutes * 60
        self._last_beat = time.monotonic()
        self._lock = threading.Lock()
        self.kill_flag = threading.Event()
        self._warned = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def beat(self):
        """Call after each completed video to reset the watchdog clock."""
        with self._lock:
            self._last_beat = time.monotonic()
            self._warned = False

    def _run(self):
        while not self.kill_flag.is_set():
            time.sleep(10)
            with self._lock:
                elapsed = time.monotonic() - self._last_beat
                warned = self._warned
            if elapsed >= self._kill_sec and not self.kill_flag.is_set():
                print(
                    f"\n{'!' * 60}\n"
                    f"  WATCHDOG KILL — no progress for "
                    f"{elapsed / 60:.1f} min (limit: {self._kill_sec / 60:.0f} min)\n"
                    f"  Saving partial results and exiting cleanly.\n"
                    f"{'!' * 60}\n",
                    flush=True,
                )
                self.kill_flag.set()
            elif elapsed >= self._warn_sec and not warned:
                print(
                    f"\n{'*' * 60}\n"
                    f"  WATCHDOG WARNING — no progress for "
                    f"{elapsed / 60:.1f} min (limit: {self._warn_sec / 60:.0f} min)\n"
                    f"  If inference appears stuck, check GPU utilization.\n"
                    f"{'*' * 60}\n",
                    flush=True,
                )
                with self._lock:
                    self._warned = True


# ── Incremental save ──────────────────────────────────────────────────────────

def save_results(records: list, results_path: str):
    """Write the current list of records to JSON atomically."""
    p = pathlib.Path(results_path).expanduser()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    tmp.replace(p)


# ── Progress reporting ────────────────────────────────────────────────────────

def print_progress(
    idx: int,
    total: int,
    elapsed: float,
    rolling: collections.deque,
    video_name: str,
    tee,
):
    """Print the per-video progress line and rolling throughput window."""
    vid_per_min = (idx / elapsed * 60) if elapsed > 0 else 0.0
    remaining = (total - idx) / vid_per_min if vid_per_min > 0 else float("inf")
    eta_str = f"~{remaining:.0f} min" if remaining != float("inf") else "calculating..."

    tee.write(
        f"[VIDEO {idx:3d}/{total}] "
        f"elapsed={_fmt_duration(elapsed)} | "
        f"vid/min={vid_per_min:.2f} | "
        f"ETA {eta_str} | "
        f"current: {video_name}\n"
    )
    tee.flush()

    # Rolling 5-video window
    if len(rolling) >= 2:
        window_vids = len(rolling)
        window_elapsed = rolling[-1] - rolling[0]
        if window_elapsed > 0:
            rolling_rate = (window_vids - 1) / window_elapsed * 60
            tee.write(
                f"  [ROLLING {window_vids}-vid window] "
                f"{rolling_rate:.2f} vid/min"
                + (" ← SLOWDOWN detected" if rolling_rate < vid_per_min * 0.6 else "")
                + "\n"
            )
            tee.flush()


# ── Completion banner ─────────────────────────────────────────────────────────

def print_banner(total_done: int, total: int, elapsed: float, results_path: str, tee):
    vid_per_min = (total_done / elapsed * 60) if elapsed > 0 else 0.0
    elapsed_min = elapsed / 60
    width = 47
    border = "═" * width
    tee.write(
        f"\n{border}\n"
        f"  DONE — {total_done}/{total} videos in {elapsed_min:.1f} min\n"
        f"  Throughput: {vid_per_min:.2f} vid/min\n"
        f"  Results saved: {pathlib.Path(results_path).expanduser().resolve()}\n"
        f"{border}\n\n"
    )
    tee.flush()


# ── Signal handling ───────────────────────────────────────────────────────────

_shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(
        f"\n[HARNESS] {sig_name} received — saving partial results and exiting cleanly...\n",
        flush=True,
    )
    _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    results_path = str(pathlib.Path(args.results).expanduser())
    log_path = str(pathlib.Path(args.log).expanduser())

    tee = Tee(log_path)

    tee.write(f"[HARNESS] Log: {log_path}  (tail -f from another terminal)\n")
    tee.write(f"[HARNESS] Results: {results_path}\n")
    tee.write(f"[HARNESS] Model: {args.model}\n")
    tee.write(f"[HARNESS] max_new_tokens={args.max_new_tokens}  fps={args.fps}\n")
    tee.write(
        f"[HARNESS] Watchdog: warn={args.watchdog_warn}m  kill={args.watchdog_kill}m\n"
    )
    tee.flush()

    # ── Load video list ───────────────────────────────────────────────────────
    video_pairs = collect_video_paths(args)
    total = len(video_pairs)

    # ── Load model ────────────────────────────────────────────────────────────
    import transformers

    transformers.set_seed(0)
    model, processor = load_model(args.model)

    # ── Start watchdog ────────────────────────────────────────────────────────
    watchdog = Watchdog(args.watchdog_warn, args.watchdog_kill)
    watchdog.start()

    # ── Inference loop ────────────────────────────────────────────────────────
    records = []
    run_start = time.monotonic()
    rolling_times: collections.deque = collections.deque(maxlen=6)  # n+1 for n intervals
    rolling_times.append(run_start)

    for i, (video_path, ground_truth) in enumerate(video_pairs, start=1):
        if _shutdown_event.is_set() or watchdog.kill_flag.is_set():
            tee.write(f"[HARNESS] Stopping early at video {i}/{total}.\n")
            tee.flush()
            break

        video_name = pathlib.Path(video_path).name
        record: dict = {
            "filepath": video_path,
            "ground_truth": ground_truth,
            "safety_label": None,
            "hazard": None,
            "description": None,
            "prediction_class_id": None,
            "cosmos_error": None,
        }

        try:
            result = run_inference_on_video(
                model, processor, video_path, args.fps, args.max_new_tokens
            )
            hazard_info = result.get("hazard_detection", {})
            record["safety_label"] = result.get("prediction_label")
            record["prediction_class_id"] = result.get("prediction_class_id")
            record["hazard"] = hazard_info.get("is_hazardous")
            record["description"] = result.get("video_description")
        except json.JSONDecodeError as e:
            record["cosmos_error"] = f"JSON parse error: {e}"
            tee.write(f"  [WARN] JSON parse failed for {video_name}: {e}\n")
        except Exception as e:
            record["cosmos_error"] = str(e)
            tee.write(f"  [WARN] Inference failed for {video_name}: {e}\n")

        records.append(record)

        # Incremental save
        save_results(records, results_path)

        # Update watchdog heartbeat
        watchdog.beat()

        # Progress reporting
        now = time.monotonic()
        rolling_times.append(now)
        elapsed = now - run_start
        print_progress(i, total, elapsed, rolling_times, video_name, tee)

    # ── Final save and banner ─────────────────────────────────────────────────
    save_results(records, results_path)

    total_elapsed = time.monotonic() - run_start
    total_done = len(records)
    labeled = [r for r in records if r["safety_label"]]
    errors = [r for r in records if r["cosmos_error"]]
    safe = [r for r in labeled if r.get("hazard") is False]
    unsafe = [r for r in labeled if r.get("hazard") is True]

    tee.write(f"\n  Classified : {len(labeled)} / {total_done}\n")
    tee.write(f"    Safe     : {len(safe)}\n")
    tee.write(f"    Unsafe   : {len(unsafe)}\n")
    tee.write(f"  Errors     : {len(errors)}\n")

    print_banner(total_done, total, total_elapsed, results_path, tee)

    tee.close()


if __name__ == "__main__":
    main()
