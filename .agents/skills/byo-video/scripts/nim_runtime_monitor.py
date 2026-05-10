#!/usr/bin/env python3
"""
nim_runtime_monitor.py — runtime watchdog + synthetic-inference probe for one NIM
on one Brev instance.

Addresses BUG-NIM-HARNESS-NO-BAIL (P0, 2026-05-08): the prior smoke harness's
internal POLL_TIMEOUT_S did not fire on a hung NIM (cosmos-reason2-2b on horde),
so silence was indistinguishable from "still running" for 5h+. This script is
the parent-side watchdog that kills the inner subprocess and emits a structured
event when no progress reaches the log within HANG_THRESHOLD_S.

Designed to run as a long-lived sidecar process on the Brev instance itself
(NOT on local Mac), invoked by the per-NIM /byo-video agent after the NIM
container is started. Emits one JSON line per probe to the runtime_monitor.jsonl
log; emits a WATCHDOG_TIMEOUT line + kills container + exits non-zero on hang.

Usage:
    python3 nim_runtime_monitor.py \\
        --short cosmos-reason1-7b \\
        --container cosmos-nim \\
        --port 8000 \\
        --probe-image /tmp/probe_image.jpg \\
        --probe-prompt "Describe this image briefly." \\
        --log /tmp/runtime_monitor.jsonl \\
        --probe-interval 60 \\
        --hang-threshold 180

Probe modes (auto-selected by NIM type):
    - vlm: POST /v1/chat/completions with image+prompt → expect 200 + non-empty content
    - parse: POST /v1/chat/completions with tools=[markdown_bbox] + 1 image → expect tool_calls
    - predict: POST /v1/infer with prompt + seed → expect b64_video field (long-running, lower probe rate)
    - health: GET /v1/models → expect 200 + non-empty data array (fallback)
"""
from __future__ import annotations

import argparse
import base64
import json
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_event(log_path: Path, event: dict) -> None:
    event["ts"] = now_iso()
    with log_path.open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def http_post(url: str, payload: dict, timeout: int = 30) -> tuple[int, str]:
    """Return (status_code, body_str). Errors return (0, error_str)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, str(e.reason)[:300]
    except (urllib.error.URLError, ConnectionResetError, OSError, TimeoutError) as e:
        return 0, str(e)[:300]


def http_get(url: str, timeout: int = 10) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, str(e.reason)[:300]
    except Exception as e:
        return 0, str(e)[:300]


def container_running(name: str) -> bool:
    p = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Status}}"],
        capture_output=True, text=True, timeout=10,
    )
    s = (p.stdout or "").strip()
    return s.startswith("Up")


def kill_container(name: str) -> None:
    subprocess.run(["docker", "kill", name], capture_output=True, timeout=15)
    subprocess.run(["docker", "rm", name], capture_output=True, timeout=15)


def b64_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def probe_vlm(host: str, port: int, image_path: str, prompt: str, model: str) -> tuple[bool, dict]:
    """VLM video-question probe: image + prompt → text response."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image(image_path)}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 64,
        "temperature": 0.6,
    }
    t0 = time.time()
    code, body = http_post(url, payload, timeout=45)
    elapsed = round(time.time() - t0, 2)
    if code != 200:
        return False, {"http_status": code, "elapsed_s": elapsed, "body_head": body[:200]}
    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        return bool(content.strip()), {"http_status": code, "elapsed_s": elapsed, "chars": len(content)}
    except Exception as e:
        return False, {"http_status": code, "elapsed_s": elapsed, "parse_err": str(e)[:200]}


def probe_parse(host: str, port: int, image_path: str, model: str) -> tuple[bool, dict]:
    """Nemotron-Parse probe: 1 image + tool=markdown_bbox → tool_calls response."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "tools": [{"type": "function", "function": {"name": "markdown_bbox"}}],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image(image_path)}"}},
            ],
        }],
        "temperature": 0.0,
    }
    t0 = time.time()
    code, body = http_post(url, payload, timeout=60)
    elapsed = round(time.time() - t0, 2)
    if code != 200:
        return False, {"http_status": code, "elapsed_s": elapsed, "body_head": body[:200]}
    try:
        data = json.loads(body)
        tcs = data["choices"][0]["message"].get("tool_calls", [])
        return bool(tcs), {"http_status": code, "elapsed_s": elapsed, "tool_calls": len(tcs)}
    except Exception as e:
        return False, {"http_status": code, "elapsed_s": elapsed, "parse_err": str(e)[:200]}


def probe_predict(host: str, port: int) -> tuple[bool, dict]:
    """Cosmos Predict probe: low-cost /v1/models check (full /v1/infer is too slow for runtime monitor).
    Use /v1/infer only for cold-start validation; for the runtime loop, /v1/models is enough
    to confirm the NIM is still serving."""
    url = f"http://{host}:{port}/v1/models"
    t0 = time.time()
    code, body = http_get(url, timeout=10)
    elapsed = round(time.time() - t0, 2)
    if code != 200:
        return False, {"http_status": code, "elapsed_s": elapsed, "body_head": body[:200]}
    try:
        data = json.loads(body)
        return bool(data.get("data")), {"http_status": code, "elapsed_s": elapsed, "models": len(data.get("data", []))}
    except Exception as e:
        return False, {"http_status": code, "elapsed_s": elapsed, "parse_err": str(e)[:200]}


def probe_health(host: str, port: int) -> tuple[bool, dict]:
    """Fallback: GET /v1/models."""
    return probe_predict(host, port)


PROBES = {"vlm": probe_vlm, "parse": probe_parse, "predict": probe_predict, "health": probe_health}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True, help="NIM short name (e.g. cosmos-reason1-7b)")
    ap.add_argument("--mode", required=True, choices=list(PROBES.keys()),
                    help="Probe mode — vlm | parse | predict | health")
    ap.add_argument("--container", default="cosmos-nim",
                    help="Docker container name (default: cosmos-nim)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None,
                    help="served_model_id; required for vlm + parse modes")
    ap.add_argument("--probe-image", default=None,
                    help="Local path to probe image (jpg/png); required for vlm + parse")
    ap.add_argument("--probe-prompt", default="Describe what you see briefly.")
    ap.add_argument("--log", required=True, help="JSONL log file path")
    ap.add_argument("--probe-interval", type=int, default=60,
                    help="Seconds between probes (default 60)")
    ap.add_argument("--hang-threshold", type=int, default=180,
                    help="Consecutive failure window in seconds before WATCHDOG_TIMEOUT (default 180)")
    ap.add_argument("--max-runtime", type=int, default=86400,
                    help="Max monitor lifetime in seconds (default 24h)")
    args = ap.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)

    log_event(log_path, {"event": "MONITOR_START", "short": args.short, "mode": args.mode,
                         "container": args.container, "host": args.host, "port": args.port,
                         "probe_interval": args.probe_interval, "hang_threshold": args.hang_threshold})

    last_success = time.time()
    consecutive_fail = 0
    start = time.time()
    stopped = {"v": False}

    def graceful(*_):
        stopped["v"] = True
        log_event(log_path, {"event": "MONITOR_STOP_SIGNAL"})
    signal.signal(signal.SIGTERM, graceful)
    signal.signal(signal.SIGINT, graceful)

    while not stopped["v"] and time.time() - start < args.max_runtime:
        if not container_running(args.container):
            log_event(log_path, {"event": "WATCHDOG_TIMEOUT", "reason": "container_not_running",
                                 "short": args.short, "container": args.container})
            return 2

        kwargs = {}
        if args.mode == "vlm":
            if not args.probe_image or not args.model:
                log_event(log_path, {"event": "MONITOR_CONFIG_ERROR",
                                     "reason": "vlm requires --probe-image + --model"})
                return 3
            ok, info = probe_vlm(args.host, args.port, args.probe_image, args.probe_prompt, args.model)
        elif args.mode == "parse":
            if not args.probe_image or not args.model:
                log_event(log_path, {"event": "MONITOR_CONFIG_ERROR",
                                     "reason": "parse requires --probe-image + --model"})
                return 3
            ok, info = probe_parse(args.host, args.port, args.probe_image, args.model)
        elif args.mode == "predict":
            ok, info = probe_predict(args.host, args.port)
        else:
            ok, info = probe_health(args.host, args.port)

        log_event(log_path, {"event": "PROBE", "ok": ok, "short": args.short, **info})

        if ok:
            last_success = time.time()
            consecutive_fail = 0
        else:
            consecutive_fail += 1
            if time.time() - last_success > args.hang_threshold:
                log_event(log_path, {"event": "WATCHDOG_TIMEOUT",
                                     "reason": "hang_threshold_exceeded",
                                     "consecutive_fail": consecutive_fail,
                                     "seconds_since_last_success": round(time.time() - last_success, 1),
                                     "short": args.short, "container": args.container})
                kill_container(args.container)
                log_event(log_path, {"event": "CONTAINER_KILLED", "short": args.short,
                                     "container": args.container})
                return 2

        time.sleep(args.probe_interval)

    log_event(log_path, {"event": "MONITOR_END", "short": args.short,
                         "elapsed_s": round(time.time() - start, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
