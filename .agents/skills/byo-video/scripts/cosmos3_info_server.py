#!/usr/bin/env python3
"""cosmos3_info_server.py — Active-model probe for byo-video frontends.

Runs on the same host as `cosmos3.ray.serve` and exposes a single HTTP route
that returns the actual loaded checkpoint name. The cosmos3 Ray Serve `/info`
endpoint does not include the checkpoint identity — it lives in the
`--checkpoint-path` CLI arg of the running process. This shim reads that arg
plus a few host-level signals (GPU name, VRAM, SSD, cosmos3 version, commit)
and returns them as JSON.

Frontends (Gradio + Next.js + Vite) point at this server on page load so the
header reflects the model that is actually loaded, never a hardcoded env value.

Defaults:
  bind:        0.0.0.0:8088
  ray_url:     http://localhost:8000

Run:
    python3 cosmos3_info_server.py --port 8088 --ray-url http://localhost:8000

Endpoint:
    GET /active-model
        -> {
             "checkpoint": "Cosmos3-Nano",
             "cosmos3_version": "1.2.1",
             "commit_sha": "99450113ff...",
             "served_url": "http://localhost:8000",
             "models": [""],
             "gpu_name": "...",
             "vram_free_gib": ...,
             "vram_total_gib": ...,
             "ssd_free_gib": ...,
             "ssd_total_gib": ...,
             "backend": "Ray Serve (cosmos3_native)",
             "ts": <unix epoch>
           }
    GET /healthz  -> {"ok": true}
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHECKPOINT_RE = re.compile(r"--checkpoint-path[=\s]+([^\s]+)")
CONFIG = {"ray_url": "http://localhost:8000"}


def _probe_checkpoint() -> str | None:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "args"], text=True, timeout=4, errors="replace"
        )
    except Exception:
        return None
    for line in out.splitlines():
        if "cosmos3.ray.serve" in line:
            m = CHECKPOINT_RE.search(line)
            if m:
                return m.group(1)
    return None


def _probe_ray_info() -> dict:
    try:
        with urllib.request.urlopen(f"{CONFIG['ray_url']}/info", timeout=4) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _probe_backend_name() -> str:
    try:
        ps = subprocess.check_output(["ps", "-Ao", "args"], text=True, timeout=4, errors="replace").lower()
    except Exception:
        return "unknown"
    if "cosmos3.ray.serve" in ps:
        return "Ray Serve (cosmos3_native)"
    if "vllm" in ps:
        return "vLLM"
    if "triton" in ps or "/opt/nim" in ps:
        return "NIM (Triton)"
    return "unknown"


def _probe_gpu() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=4,
        ).strip().splitlines()
        if not out:
            return {}
        parts = [x.strip() for x in out[0].split(",")]
        name, used, free, total, util, temp, power = parts[:7]
        return {
            "gpu_name": name,
            "vram_used_gib": round(int(used) / 1024, 1),
            "vram_free_gib": round(int(free) / 1024, 1),
            "vram_total_gib": round(int(total) / 1024, 1),
            "gpu_util_pct": int(util),
            "gpu_temp_c": int(temp),
            "gpu_power_w": float(power),
        }
    except Exception as exc:
        return {"gpu_error": f"{type(exc).__name__}: {exc}"}


def _probe_disk(path: str = "/") -> dict:
    try:
        u = shutil.disk_usage(path)
        return {
            "ssd_total_gib": round(u.total / 2**30, 1),
            "ssd_used_gib": round(u.used / 2**30, 1),
            "ssd_free_gib": round(u.free / 2**30, 1),
        }
    except Exception as exc:
        return {"ssd_error": f"{type(exc).__name__}: {exc}"}


def build_active_model_payload() -> dict:
    checkpoint = _probe_checkpoint()
    ray_info = _probe_ray_info()
    env = ray_info.get("environment", {}) if isinstance(ray_info, dict) else {}
    payload = {
        "checkpoint": checkpoint,
        "cosmos3_version": env.get("cosmos3_version"),
        "commit_sha": env.get("commit_sha"),
        "served_url": CONFIG["ray_url"],
        "models": ray_info.get("models") if isinstance(ray_info, dict) else None,
        "backend": _probe_backend_name(),
        "ts": int(time.time()),
    }
    payload.update(_probe_gpu())
    payload.update(_probe_disk("/"))
    # Convenience: display label preferred order.
    payload["display_name"] = (
        checkpoint
        or (f"cosmos3 v{env.get('cosmos3_version','?')}" if env.get("cosmos3_version") else "Cosmos3 (unknown checkpoint)")
    )
    return payload


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/active-model"):
            self._send_json(200, build_active_model_payload())
            return
        if self.path.startswith("/healthz"):
            self._send_json(200, {"ok": True, "ts": int(time.time())})
            return
        self._send_json(404, {"error": "not found", "path": self.path})

    def log_message(self, fmt, *args):
        # quiet by default; uncomment to debug:
        # sys.stderr.write("%s - - %s\n" % (self.client_address[0], fmt % args))
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--ray-url", default="http://localhost:8000")
    args = ap.parse_args()
    CONFIG["ray_url"] = args.ray_url.rstrip("/")
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    sys.stderr.write(f"cosmos3_info_server listening on {args.bind}:{args.port} (ray={args.ray_url})\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
