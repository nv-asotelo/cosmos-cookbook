#!/usr/bin/env python3
"""
nim_smoke_one.py — Smoke a single NIM on the local host.

Runs ON the target host (horde). Driven by an orchestrator that calls it
once per NIM via SSH. Each call:
  1. Ensures a NIM container is running and serving the requested image
     (swaps if a different image is up; starts fresh if none).
  2. Polls /v1/models until ready (or fails with timeout).
  3. Runs the smoke prompts (reasoning ON, reasoning OFF) against the served model
     using the OpenAI-compatible /v1/chat/completions endpoint.
  4. Captures full request + verbatim response per prompt.
  5. Writes /tmp/nim_smoke_<short>.json with the result.

Usage (on horde):
  python3 /tmp/nim_smoke_one.py <short-name>
  python3 /tmp/nim_smoke_one.py cosmos-reason2-8b

Env:
  NGC_API_KEY         — required for fresh pulls (leak from running container if absent)
  PARAM_TABLE         — path to per-NIM params JSON (default: /tmp/nim_param_table.json)
  INPUT_VIDEO         — path to test video (default: /tmp/ball_video_high_res.mp4)
  SMOKE_OUT_DIR       — output dir (default: /tmp)
  CONTAINER_NAME      — docker name (default: cosmos-nim)
  PORT                — host port (default: 8000)
  POLL_TIMEOUT_S      — wait for /v1/models (default: 1800)
  INFER_TIMEOUT_S     — single-prompt timeout (default: 300)
"""

from __future__ import annotations
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "cosmos-nim")
PORT = int(os.environ.get("PORT", "8000"))
PARAM_TABLE_PATH = os.environ.get("PARAM_TABLE", "/tmp/nim_param_table.json")
INPUT_VIDEO = os.environ.get("INPUT_VIDEO", "/tmp/ball_video_high_res.mp4")
SMOKE_OUT_DIR = Path(os.environ.get("SMOKE_OUT_DIR", "/tmp"))
POLL_TIMEOUT_S = int(os.environ.get("POLL_TIMEOUT_S", "1800"))
INFER_TIMEOUT_S = int(os.environ.get("INFER_TIMEOUT_S", "300"))

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT_REASONING_ON = "race car: timestamps"
USER_PROMPT_REASONING_OFF = "temporal summary"
MIN_FPS = 8
LOCAL_NIM_CACHE = os.path.expanduser("~/.cache/nim")


def log(msg: str) -> None:
    sys.stdout.write(f"[smoke] {msg}\n")
    sys.stdout.flush()


def run(cmd: list, check: bool = False, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def get_running_image() -> str | None:
    """Return image name only if the container is actually Running (not exited)."""
    p = run([
        "docker", "inspect",
        "--format", "{{.State.Running}}|{{.Config.Image}}",
        CONTAINER_NAME,
    ])
    if p.returncode != 0 or not p.stdout.strip():
        return None
    parts = p.stdout.strip().split("|", 1)
    if len(parts) != 2:
        return None
    is_running, image = parts
    if is_running.lower() != "true":
        return None
    return image


def stop_and_remove_container() -> None:
    run(["docker", "stop", "-t", "10", CONTAINER_NAME])
    run(["docker", "rm", "-f", CONTAINER_NAME])


def disk_free_gb() -> float:
    p = run(["df", "-BG", "/"])
    for line in p.stdout.splitlines():
        if line.startswith("/dev/"):
            parts = line.split()
            return float(parts[3].rstrip("G"))
    return 0.0


def docker_image_size_gb(image: str) -> float:
    p = run(["docker", "image", "inspect", image, "--format", "{{.Size}}"])
    if p.returncode != 0 or not p.stdout.strip():
        return 0.0
    try:
        return int(p.stdout.strip()) / 1e9
    except ValueError:
        return 0.0


def prune_dead_containers() -> str:
    p = run(["docker", "container", "prune", "-f"], timeout=30)
    return (p.stdout or "").strip().splitlines()[-1] if p.stdout else "no-output"


def free_disk_for_pull(min_gb: float = 80.0) -> list[str]:
    """Drop ALL non-running NIM images, then prune builder cache. Returns list of removed.

    Threshold is informational only — we always free aggressively because the next pull
    may be 50GB+ and partial-disk failures leave hard-to-recover state.
    """
    removed = []
    running_image = get_running_image()
    p = run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    for line in p.stdout.splitlines():
        line = line.strip()
        if "nvcr.io/nim" not in line:
            continue
        if running_image and line == running_image:
            continue
        run(["docker", "rmi", "-f", line])
        removed.append(line)
    # Builder cache + dangling
    run(["docker", "builder", "prune", "-af"], timeout=60)
    run(["docker", "image", "prune", "-af"], timeout=60)
    return removed


def docker_pull(image: str, ngc_key: str) -> tuple[bool, str]:
    # Login is implicit via NGC_API_KEY (credentials helper) on horde; if not, do explicit.
    login_p = run(
        ["bash", "-c", f"echo '{ngc_key}' | docker login nvcr.io -u '$oauthtoken' --password-stdin"],
        timeout=60,
    )
    if login_p.returncode != 0:
        return False, f"docker login failed: {login_p.stderr[:500]}"
    pull_p = run(["docker", "pull", image], timeout=2400)
    if pull_p.returncode != 0:
        return False, f"docker pull failed: {pull_p.stderr[:1000]}"
    return True, "ok"


def docker_run(image: str, ngc_key: str, hf_token: str | None = None) -> tuple[bool, str]:
    # No bind mount — some NIMs (nemotron-*) hit PermissionError on /opt/nim/.cache when host
    # dir perms differ from container UID. Container-internal cache is ephemeral but reliable.
    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--gpus", "all",
        "--shm-size=32GB",
        "-e", f"NGC_API_KEY={ngc_key}",
        "-e", f"NIM_NSPECT_ID={os.environ.get('NIM_NSPECT_ID', 'NSPECT-HFIK-RMS9')}",
        "-p", f"{PORT}:8000",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
    ]
    if hf_token:
        cmd += ["-e", f"HF_TOKEN={hf_token}"]
    cmd.append(image)
    p = run(cmd, timeout=120)
    if p.returncode != 0:
        return False, f"docker run failed: {p.stderr[:500]}"
    return True, p.stdout.strip()


def poll_models_endpoint(timeout_s: int) -> tuple[bool, dict | None, str]:
    start = time.time()
    last_err = ""
    while time.time() - start < timeout_s:
        try:
            req = urllib.request.Request(f"http://localhost:{PORT}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))
                if data.get("data"):
                    return True, data, ""
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionResetError, OSError) as e:
            last_err = str(e)[:200]
        # Container died?
        p = run(["docker", "ps", "-a", "--filter", f"name=^{CONTAINER_NAME}$", "--format", "{{.Status}}"])
        status = (p.stdout or "").strip()
        if status and status.startswith("Exited"):
            logs = run(["docker", "logs", "--tail", "80", CONTAINER_NAME], timeout=15)
            return False, None, f"container exited: {status}\nlast logs:\n{logs.stdout[-3000:]}"
        time.sleep(8)
    return False, None, f"timeout after {timeout_s}s; last_err={last_err}"


def extract_frames_b64(video_path: str, fps: int, max_frames: int) -> list[str]:
    """Use ffmpeg to extract frames; return base64 jpegs."""
    out_dir = "/tmp/_smoke_frames"
    os.system(f"rm -rf {out_dir} && mkdir -p {out_dir}")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps},scale=-2:720",
        "-frames:v", str(max_frames),
        "-q:v", "3",
        f"{out_dir}/frame_%04d.jpg",
    ]
    p = run(cmd, timeout=120)
    if p.returncode != 0:
        return []
    frames = []
    for f in sorted(Path(out_dir).glob("frame_*.jpg")):
        with open(f, "rb") as fh:
            frames.append(base64.b64encode(fh.read()).decode("ascii"))
    return frames


def build_video_url() -> str:
    """Return a data:video/mp4;base64,<...> URL for the input video."""
    with open(INPUT_VIDEO, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f"data:video/mp4;base64,{b}"


def chat_payload_video_url(model: str, prompt: str, params: dict) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": build_video_url()}},
                ],
            },
        ],
        "max_tokens": params.get("max_output_tokens", 4096),
        "temperature": params.get("temperature", 0.6),
        "top_p": params.get("top_p", 0.95),
        "frequency_penalty": params.get("frequency_penalty", 0.0),
        "presence_penalty": params.get("presence_penalty", 0.0),
        "stream": False,
    }


def chat_payload_frames(model: str, prompt: str, params: dict) -> dict:
    fps = max(MIN_FPS, params.get("fps", MIN_FPS))
    max_frames = params.get("max_frames", 32)
    frames = extract_frames_b64(INPUT_VIDEO, fps=fps, max_frames=max_frames)
    content = [{"type": "text", "text": prompt}]
    for fb in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb}"}})
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": params.get("max_output_tokens", 4096),
        "temperature": params.get("temperature", 0.6),
        "top_p": params.get("top_p", 0.95),
        "frequency_penalty": params.get("frequency_penalty", 0.0),
        "presence_penalty": params.get("presence_penalty", 0.0),
        "stream": False,
    }


def post_chat(payload: dict, timeout_s: int) -> tuple[int, dict | str, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            text = r.read().decode("utf-8")
            elapsed = time.time() - t0
            try:
                return r.status, json.loads(text), elapsed
            except json.JSONDecodeError:
                return r.status, text, elapsed
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        return e.code, f"HTTPError: {e.reason}\n{body[:2000]}", time.time() - t0
    except Exception as e:
        return -1, f"Exception: {type(e).__name__}: {e}", time.time() - t0


def run_one_prompt(model: str, prompt: str, params: dict, video_mode: str) -> dict:
    """Always try video_url first (richer signal); fall back to frames on 4xx.

    The video_mode param-table hint is now informational only — the harness
    decides at runtime by attempting video_url, observing the response, and
    retrying with frame extraction if the NIM rejects native video.
    """
    payload = chat_payload_video_url(model, prompt, params)
    status, body, elapsed = post_chat(payload, INFER_TIMEOUT_S)
    actual_mode = "video_url"
    fallback_record = None
    if isinstance(status, int) and 400 <= status < 500:
        # NIM rejected video_url; retry with frames
        fallback_record = {"video_url_status": status, "video_url_body_preview": str(body)[:500]}
        payload = chat_payload_frames(model, prompt, params)
        status, body, elapsed = post_chat(payload, INFER_TIMEOUT_S)
        actual_mode = "frames"
        # If frames also fails with 5-image cap, retry with max_frames=5
        if isinstance(status, int) and status == 400 and isinstance(body, str) and "image(s) may be provided" in body:
            fallback_record["frames_24_status"] = status
            fallback_record["frames_24_body_preview"] = body[:500]
            params2 = dict(params); params2["max_frames"] = 5
            payload = chat_payload_frames(model, prompt, params2)
            status, body, elapsed = post_chat(payload, INFER_TIMEOUT_S)
            actual_mode = "frames_max5"
    # Compress payload echo (omit big base64 data) for log readability
    clean_payload = json.loads(json.dumps(payload))
    for m in clean_payload.get("messages", []):
        if isinstance(m.get("content"), list):
            for c in m["content"]:
                if isinstance(c, dict):
                    for k in ("video_url", "image_url"):
                        if k in c and isinstance(c[k], dict) and "url" in c[k]:
                            c[k]["url"] = c[k]["url"][:80] + "...[truncated]"
    return {
        "video_mode_hint": video_mode,
        "video_mode_actual": actual_mode,
        "fallback_record": fallback_record,
        "request_params": {
            k: clean_payload[k] for k in ("max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty")
        },
        "request_messages_redacted": clean_payload["messages"],
        "fps": params.get("fps", MIN_FPS),
        "max_frames": params.get("max_frames"),
        "max_pixels": params.get("max_pixels"),
        "http_status": status,
        "elapsed_s": round(elapsed, 2),
        "response": body,
    }


def smoke_one(short: str, param_table: dict) -> dict:
    entry = param_table.get(short)
    if not entry:
        return {"short": short, "error": f"no entry in param table {PARAM_TABLE_PATH}"}
    image = entry["image"]
    served_id_expected = entry.get("served_model_id")
    video_mode = entry.get("video_mode", "frames")  # 'video_url' or 'frames'
    params = entry.get("params", {})
    ngc_key = os.environ.get("NGC_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN")

    result: dict = {
        "short": short,
        "image": image,
        "served_model_id_expected": served_id_expected,
        "video_mode": video_mode,
        "params": params,
        "input_video": INPUT_VIDEO,
        "system_prompt": SYSTEM_PROMPT,
        "phase": "init",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    # Step 1: ensure container is serving the right image
    running = get_running_image()
    if running == image:
        result["phase"] = "reuse_running_container"
        log(f"reusing running container {CONTAINER_NAME} ({image})")
    else:
        # Always remove the named container (running or dead) to drop its writable layer.
        log(f"swap: stop+rm any prior {CONTAINER_NAME} → start {image}")
        stop_and_remove_container()
        # Prune any other dead containers (writable layers can balloon to tens of GB).
        prune_msg = prune_dead_containers()
        if prune_msg and "0 B" not in prune_msg and "no-output" not in prune_msg:
            log(f"prune: {prune_msg}")
        result["phase"] = "freeing_disk_if_needed"
        size_gb = docker_image_size_gb(image)
        if size_gb == 0.0:
            # No cached image — pull is coming. Drop all non-running NIM images first.
            removed = free_disk_for_pull(min_gb=80.0)
            if removed:
                log(f"freed disk by removing: {removed}")
            free_now = disk_free_gb()
            log(f"disk free after prune: {free_now:.1f} GiB")
            log(f"pulling {image}")
            ok, msg = docker_pull(image, ngc_key)
            if not ok:
                result.update({"phase": "pull_failed", "error": msg})
                return result
        result["phase"] = "starting_container"
        ok, msg = docker_run(image, ngc_key, hf_token)
        if not ok:
            result.update({"phase": "run_failed", "error": msg})
            return result
        result["container_id"] = msg

    # Step 2: poll /v1/models
    result["phase"] = "polling_models"
    ok, models_data, err = poll_models_endpoint(POLL_TIMEOUT_S)
    if not ok:
        result.update({"phase": "models_endpoint_failed", "error": err})
        # capture last container logs for triage
        logs = run(["docker", "logs", "--tail", "120", CONTAINER_NAME], timeout=15)
        result["container_logs_tail"] = logs.stdout[-4000:]
        return result
    served_id = models_data["data"][0]["id"]
    result["served_model_id"] = served_id
    result["max_model_len"] = models_data["data"][0].get("max_model_len")

    # Step 3: run prompts
    result["phase"] = "running_prompts"
    result["smoke_runs"] = {}
    for label, prompt in (
        ("reasoning_on", USER_PROMPT_REASONING_ON),
        ("reasoning_off", USER_PROMPT_REASONING_OFF),
    ):
        log(f"  prompt[{label}]: {prompt!r}")
        run_result = run_one_prompt(served_id, prompt, params, video_mode)
        # If video_url failed with 4xx, retry as frames
        if (
            video_mode == "video_url"
            and isinstance(run_result.get("http_status"), int)
            and 400 <= run_result["http_status"] < 500
        ):
            log(f"  retrying {label} with frames mode (video_url got {run_result['http_status']})")
            run_result_frames = run_one_prompt(served_id, prompt, params, "frames")
            run_result_frames["fallback_from_video_url"] = run_result
            run_result = run_result_frames
        result["smoke_runs"][label] = run_result
        # Quick assessment
        try:
            content = run_result["response"]["choices"][0]["message"]["content"]
            tok = run_result["response"].get("usage", {})
            log(f"    OK · status={run_result['http_status']} · {len(content)} chars · usage={tok}")
        except Exception:
            log(f"    BAD · status={run_result['http_status']} · body[:200]={str(run_result['response'])[:200]}")

    result["phase"] = "complete"
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("short", help="NIM short name")
    parser.add_argument("--param-table", default=PARAM_TABLE_PATH)
    args = parser.parse_args()

    table = json.loads(Path(args.param_table).read_text())
    result = smoke_one(args.short, table)
    out = SMOKE_OUT_DIR / f"nim_smoke_{args.short}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    log(f"wrote {out}")
    # Echo summary to stdout for ssh capture
    print("=== SMOKE SUMMARY ===")
    print(f"short: {result.get('short')}")
    print(f"phase: {result.get('phase')}")
    print(f"served_id: {result.get('served_model_id')}")
    if result.get("error"):
        print(f"error: {result['error'][:500]}")
    if "smoke_runs" in result:
        for k, v in result["smoke_runs"].items():
            try:
                content = v["response"]["choices"][0]["message"]["content"]
                print(f"{k}: status={v['http_status']} chars={len(content)}")
            except Exception:
                print(f"{k}: status={v['http_status']} BODY={str(v['response'])[:200]}")


if __name__ == "__main__":
    main()
