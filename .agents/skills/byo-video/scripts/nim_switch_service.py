#!/usr/bin/env python3
"""Small companion service for zero-touch BYO-video NIM swaps.

The service stays outside Gradio so users can keep watching progress while the
Gradio process is restarted after a successful NIM container swap.
"""
from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional


HOST = os.environ.get("NIM_SWITCH_HOST", "0.0.0.0")
PORT = int(os.environ.get("NIM_SWITCH_PORT", "7862"))
STATE_FILE = Path(os.environ.get("NIM_SWITCH_STATE_FILE", "/tmp/nim_switch_state.json"))
LOG_FILE = Path(os.environ.get("NIM_SWITCH_LOG_FILE", "/tmp/nim_switch_service.log"))
CREDENTIAL_FILE = Path(os.environ.get("NIM_CREDENTIAL_FILE", "/tmp/byo_video_nim_credentials.env"))
NIM_LAUNCH = os.environ.get("NIM_LAUNCH_SCRIPT", "/tmp/nim_launch.sh")
GRADIO_APP = os.environ.get("GRADIO_APP", "/tmp/gradio_cr2_byo.py")
GRADIO_LOG = Path(os.environ.get("GRADIO_LOG_FILE", "/tmp/gradio_demo.log"))
GRADIO_PID_FILE = Path(os.environ.get("GRADIO_PID_FILE", "/tmp/gradio_demo.pid"))
GRADIO_URL_FILE = Path(os.environ.get("GRADIO_URL_FILE", "/tmp/gradio_url.txt"))
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7860"))
NIM_PORT = int(os.environ.get("NIM_PORT", os.environ.get("PORT", "8000")))
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "cosmos-nim")
HOME = os.path.expanduser("~")
REASON2_DIR = os.environ.get("COSMOS_DIR", f"{HOME}/cosmos-reason2")
PUBLIC_HOST = os.environ.get("BYO_VIDEO_LOCAL_HOST", "")
PUBLIC_SWITCH_URL = os.environ.get(
    "NIM_SWITCH_URL",
    f"http://{PUBLIC_HOST}:{PORT}" if PUBLIC_HOST else f"http://localhost:{PORT}",
)
PUBLIC_GRADIO_URL = os.environ.get(
    "GRADIO_PUBLIC_URL",
    f"http://{PUBLIC_HOST}:{GRADIO_PORT}" if PUBLIC_HOST else f"http://localhost:{GRADIO_PORT}",
)

_STATE_LOCK = threading.Lock()
_SWITCH_LOCK = threading.Lock()


def _append_log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {message.rstrip()}\n")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


_PHASE_PROGRESS = {
    "idle": 0,
    "starting": 5,
    "validating": 10,
    "launching_nim": 15,
    "pulling": 35,
    "starting_container": 55,
    "waiting_nim": 75,
    "nim_ready": 88,
    "restarting_gradio": 95,
    "ready": 100,
    "error": 100,
    "unreachable": 0,
}
_ACTIVE_PHASES = {
    "starting", "validating", "launching_nim", "pulling",
    "starting_container", "waiting_nim", "nim_ready", "restarting_gradio",
}


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _enrich_state(state: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    now = now or time.time()
    phase = str(state.get("phase") or "idle")
    started = state.get("started_at")
    if started:
        if phase in _ACTIVE_PHASES:
            elapsed_s = max(0.0, now - float(started))
        else:
            elapsed_s = float(state.get("elapsed_s") or max(0.0, (state.get("updated_at") or now) - float(started)))
    else:
        elapsed_s = 0.0
    progress_pct = int(_PHASE_PROGRESS.get(phase, 0))
    eta_label = "Complete" if phase == "ready" else "Failed" if phase == "error" else "Unavailable during NIM pull/load"
    state["elapsed_s"] = round(elapsed_s, 1)
    state["elapsed_label"] = _format_duration(elapsed_s)
    state["progress_pct"] = progress_pct
    state["eta_s"] = 0 if phase == "ready" else None
    state["eta_label"] = eta_label
    return state


def _write_state(update: Dict[str, Any]) -> Dict[str, Any]:
    with _STATE_LOCK:
        state = _read_json(STATE_FILE)
        now = time.time()
        state.update(update)
        state["updated_at"] = now
        state["credentials_present"] = _credentials_present()
        state = _enrich_state(state, now=now)
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(STATE_FILE)
        return state


def _curl_json(url: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except Exception:
        return None


def _docker_image() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["docker", "inspect", "-f", "{{.Config.Image}}", CONTAINER_NAME],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip() or None
    except Exception:
        return None


def _current_runtime() -> Dict[str, Any]:
    body = _curl_json(f"http://localhost:{NIM_PORT}/v1/models", timeout=3) or {}
    models = body.get("data") or []
    served = (models[0] or {}).get("id") if models else None
    return {
        "served_model_id": served,
        "image": _docker_image(),
        "ready": bool(served),
        "models": models,
    }


def _target_vram_mb() -> Optional[int]:
    env_vram = os.environ.get("NIM_TARGET_VRAM_MB")
    if env_vram and env_vram.isdigit():
        return int(env_vram)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        vals = [int(v.strip()) for v in out.splitlines() if v.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:
        return None


def _catalog() -> list[Any]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in (script_dir, "/tmp"):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    try:
        from nim_catalog import list_switchable_video_nims  # type: ignore

        return list_switchable_video_nims(
            ngc_api_key=os.environ.get("NGC_API_KEY"),
            vram_mb=_target_vram_mb(),
            use_upstream=True,
            do_probe=False,
        )
    except Exception as exc:
        _append_log(f"Catalog load failed: {exc}")
        return []


def _nim_to_dict(nim: Any) -> Dict[str, Any]:
    try:
        data = asdict(nim)
    except Exception:
        data = dict(nim)
    data.setdefault("env", {})
    data.setdefault("warnings", [])
    if data.get("notes"):
        data["warnings"] = [data["notes"]]
    return data


def _resolve_target(short_id: str) -> Optional[Dict[str, Any]]:
    short_id = (short_id or "").strip()
    current = _current_runtime()
    if short_id in {"__current__", "current", "custom-current"}:
        if not current.get("served_model_id"):
            return None
        return {
            "short_id": "__current__",
            "label": f"Current custom: {current['served_model_id']}",
            "image": current.get("image") or "",
            "family": "Current custom",
            "served_model_id": current["served_model_id"],
            "min_vram_mb": 0,
            "supports_video": True,
            "switchable": False,
            "env": {},
            "notes": "Currently running container; no switch needed.",
            "custom_current": True,
        }
    for nim in _catalog():
        data = _nim_to_dict(nim)
        tokens = {
            data.get("short_id", ""),
            data.get("served_model_id", ""),
            data.get("label", ""),
            data.get("image", ""),
        }
        if short_id in tokens:
            return data
    if short_id and short_id == current.get("served_model_id"):
        return _resolve_target("__current__")
    return None


def _credentials_present() -> bool:
    if os.environ.get("NGC_API_KEY"):
        return True
    if not CREDENTIAL_FILE.exists():
        return False
    try:
        mode = CREDENTIAL_FILE.stat().st_mode & 0o777
        if mode & 0o077:
            return False
        text = CREDENTIAL_FILE.read_text(encoding="utf-8", errors="ignore")
        return "NGC_API_KEY=" in text
    except Exception:
        return False


def _env_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_credentials(payload: Dict[str, Any]) -> None:
    ngc = str(payload.get("ngc_api_key") or payload.get("NGC_API_KEY") or "").strip()
    hf = str(payload.get("hf_token") or payload.get("HF_TOKEN") or "").strip()
    if not ngc.startswith("nvapi-"):
        raise ValueError("NGC key must start with nvapi-")
    lines = [f"NGC_API_KEY={_env_quote(ngc)}"]
    if hf:
        lines.append(f"HF_TOKEN={_env_quote(hf)}")
    CREDENTIAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CREDENTIAL_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(CREDENTIAL_FILE)
    os.chmod(CREDENTIAL_FILE, 0o600)


def _load_credentials(env: Dict[str, str]) -> None:
    if env.get("NGC_API_KEY") or not CREDENTIAL_FILE.exists():
        return
    if CREDENTIAL_FILE.stat().st_mode & 0o077:
        raise RuntimeError(f"{CREDENTIAL_FILE} must be chmod 0600")
    for raw in CREDENTIAL_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key in {"NGC_API_KEY", "HF_TOKEN"} and value:
            env[key] = value


def _phase(phase: str, message: str, **extra: Any) -> None:
    _append_log(f"{phase}: {message}")
    _write_state({"phase": phase, "message": message, **extra})


def _tail(path: Path, max_chars: int = 12000) -> str:
    try:
        data = path.read_bytes()
        return data[-max_chars:].decode("utf-8", errors="replace")
    except Exception:
        return ""


def _run_nim_launch(target: Dict[str, Any]) -> None:
    env = os.environ.copy()
    _load_credentials(env)
    if not env.get("NGC_API_KEY"):
        raise RuntimeError(
            f"Missing NGC_API_KEY. Add it in Gradio or write chmod 0600 {CREDENTIAL_FILE}."
        )
    env.update({
        "FORCE_RESTART": "1",
        "MODEL": target["short_id"],
        "IMAGE": target["image"],
        "NIM_SERVED_MODEL_NAME": target["served_model_id"],
        "PORT": str(NIM_PORT),
        "CONTAINER_NAME": CONTAINER_NAME,
        "NIM_CREDENTIAL_FILE": str(CREDENTIAL_FILE),
        "MAX_WAIT": os.environ.get("NIM_SWITCH_MAX_WAIT", "2400"),
        "NIM_CACHE_MODE": os.environ.get("NIM_CACHE_MODE", "internal"),
    })
    for key, value in (target.get("env") or {}).items():
        env[str(key)] = str(value)

    _phase("launching_nim", f"Starting {target['label']} with {target['image']}")
    proc = subprocess.Popen(
        ["bash", NIM_LAUNCH],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        clean = line.rstrip()
        _append_log(f"nim_launch: {clean}")
        low = clean.lower()
        if "pulling " in low:
            _write_state({"phase": "pulling", "message": clean})
        elif "starting nim container" in low:
            _write_state({"phase": "starting_container", "message": clean})
        elif "waiting for nim" in low or "waiting for /v1/models" in low:
            _write_state({"phase": "waiting_nim", "message": clean})
        elif "nim is ready" in low:
            _write_state({"phase": "nim_ready", "message": clean})
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"nim_launch.sh exited {rc}. Tail:\n{_tail(Path('/tmp/nim_launch.log'))}"
        )

    body = _curl_json(f"http://localhost:{NIM_PORT}/v1/models", timeout=10) or {}
    ids = [m.get("id") for m in (body.get("data") or [])]
    if target["served_model_id"] not in ids and ids:
        _append_log(f"Warning: requested {target['served_model_id']} but /v1/models returned {ids}")
    if not ids:
        raise RuntimeError("/v1/models did not return a model after nim_launch.sh completed")


def _kill_gradio() -> None:
    pids: set[int] = set()
    try:
        if GRADIO_PID_FILE.exists():
            pid = int(GRADIO_PID_FILE.read_text().strip())
            if pid > 0:
                pids.add(pid)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True, timeout=5)
        for line in out.splitlines():
            stripped = line.strip()
            if "gradio_cr2_byo.py" in stripped and "nim_switch_service.py" not in stripped:
                pid_text = stripped.split(None, 1)[0]
                if pid_text.isdigit() and int(pid_text) != os.getpid():
                    pids.add(int(pid_text))
    except Exception:
        pass
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as exc:
            _append_log(f"Could not SIGTERM Gradio pid {pid}: {exc}")
    deadline = time.time() + 12
    while time.time() < deadline and pids:
        alive = set()
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.add(pid)
            except ProcessLookupError:
                pass
            except Exception:
                pass
        pids = alive
        if pids:
            time.sleep(0.5)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def _uv_python_cmd() -> list[str]:
    uv = os.environ.get("UV_BIN") or shutil.which("uv") or f"{HOME}/.local/bin/uv"
    if os.path.exists(uv) or shutil.which(uv):
        return [uv, "run", "python", "-u", GRADIO_APP]
    return [sys.executable, "-u", GRADIO_APP]


def _model_size_for_target(target: Dict[str, Any]) -> Optional[str]:
    short_id = str(target.get("short_id") or "").lower()
    served = str(target.get("served_model_id") or "").lower()
    token = f"{short_id} {served}"
    if "cosmos-reason2-2b" in token:
        return "2B"
    if "cosmos-reason2-8b" in token:
        return "8B"
    if "cosmos-reason1-7b" in token:
        return "CR1-7B"
    if "nemotron-3-nano-omni-30b-a3b-reasoning" in token:
        return "OMNI-30B"
    if "nemotron-nano-12b-v2-vl" in token:
        return "NEM-12B"
    if "gemma-4-31b-it" in token:
        return "GM-4-31B"
    if "cosmos3-super-reasoner" in token:
        return "C3-super"
    return None


def _restart_gradio(target: Dict[str, Any]) -> None:
    _phase("restarting_gradio", "Restarting Gradio with the selected served model")
    _kill_gradio()
    env = os.environ.copy()
    _load_credentials(env)
    env.update({
        "INFERENCE_BACKEND": "nim_local",
        "VLLM_BASE_URL": f"http://localhost:{NIM_PORT}/v1",
        "MODEL_NAME": target["served_model_id"],
        "MODEL_ID": target["served_model_id"],
        "GRADIO_PORT": str(GRADIO_PORT),
        "NIM_SWITCH_PORT": str(PORT),
        "NIM_SWITCH_URL": PUBLIC_SWITCH_URL,
        "GRADIO_PUBLIC_URL": PUBLIC_GRADIO_URL,
        "CONTAINER_NAME": CONTAINER_NAME,
        "PYTHONUNBUFFERED": "1",
        "SKIP_HF_PRELOAD": "1",
    })
    env["MODEL_SIZE"] = _model_size_for_target(target) or env.get("MODEL_SIZE") or "C3-super"
    GRADIO_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_fh = GRADIO_LOG.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        _uv_python_cmd(),
        cwd=REASON2_DIR if os.path.isdir(REASON2_DIR) else None,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    GRADIO_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    GRADIO_URL_FILE.write_text(PUBLIC_GRADIO_URL, encoding="utf-8")

    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{GRADIO_PORT}/", timeout=3) as resp:
                if 200 <= resp.status < 500:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Gradio did not return on port {GRADIO_PORT}. Tail:\n{_tail(GRADIO_LOG)}")


def _switch_worker(target: Dict[str, Any]) -> None:
    try:
        _write_state({
            "phase": "starting",
            "started_at": time.time(),
            "target": target,
            "error": None,
            "message": f"Switch requested for {target['label']}",
        })
        current = _current_runtime()
        if target.get("custom_current") or (
            current.get("served_model_id") == target.get("served_model_id")
            and (not target.get("image") or current.get("image") == target.get("image"))
        ):
            _phase("ready", "Selected NIM is already running", current=current)
            return
        _phase("validating", "Resolved catalog metadata", current=current)
        _run_nim_launch(target)
        current = _current_runtime()
        _phase("nim_ready", "NIM /v1/models is serving", current=current)
        _restart_gradio(target)
        _phase("ready", "Switch complete; Gradio is back online", current=_current_runtime())
    except Exception as exc:
        _phase("error", str(exc), error=str(exc), current=_current_runtime())
    finally:
        try:
            _SWITCH_LOCK.release()
        except RuntimeError:
            pass


def _initial_state() -> None:
    current = _current_runtime()
    if not STATE_FILE.exists():
        _write_state({
            "phase": "idle",
            "message": "NIM switch service ready",
            "started_at": None,
            "target": None,
            "current": current,
            "error": None,
        })
    else:
        _write_state({"current": current})


def _html_page() -> bytes:
    state_url = "/api/state"
    log_url = "/api/log"
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BYO-video NIM Switch</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:32px;background:#f8fafc;color:#0f172a}}
.card{{max-width:980px;margin:0 auto;background:white;border:1px solid #cbd5e1;border-radius:8px;padding:20px}}
pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;border-radius:6px;padding:12px;max-height:460px;overflow:auto}}
.pill{{display:inline-block;background:#dbeafe;color:#1e3a8a;border-radius:999px;padding:2px 9px;margin-left:6px}}
.progress{{height:12px;background:#e2e8f0;border-radius:999px;overflow:hidden;margin:14px 0 6px}}
.bar{{height:100%;width:0%;background:#22c55e;transition:width .4s ease}}
.meta{{display:flex;gap:18px;flex-wrap:wrap;color:#475569;font-size:14px;margin-bottom:12px}}
a{{color:#1d4ed8}}
</style></head>
<body><div class="card">
<h1>BYO-video NIM Switch <span id="phase" class="pill">loading</span></h1>
<p>Gradio: <a href="{PUBLIC_GRADIO_URL}">{PUBLIC_GRADIO_URL}</a></p>
<div class="progress"><div id="progressBar" class="bar"></div></div>
<div class="meta">
  <span id="progressText">Progress: 0%</span>
  <span id="elapsed">Elapsed: 0s</span>
  <span id="eta">ETA: n/a</span>
</div>
<p id="message"></p>
<pre id="state"></pre>
<h2>Log</h2>
<pre id="log"></pre>
</div>
<script>
let offset = 0;
async function tick(){{
  const s = await fetch("{state_url}").then(r => r.json());
  document.getElementById("phase").textContent = s.phase || "unknown";
  const pct = Math.max(0, Math.min(100, Number(s.progress_pct || 0)));
  document.getElementById("progressBar").style.width = pct + "%";
  document.getElementById("progressText").textContent = "Progress: " + pct + "%";
  document.getElementById("elapsed").textContent = "Elapsed: " + (s.elapsed_label || "0s");
  document.getElementById("eta").textContent = "ETA: " + (s.eta_label || "n/a");
  document.getElementById("message").textContent = s.message || "";
  document.getElementById("state").textContent = JSON.stringify(s, null, 2);
  const l = await fetch("{log_url}?offset=" + offset).then(r => r.json());
  offset = l.next_offset || offset;
  if (l.text) {{
    const el = document.getElementById("log");
    el.textContent += l.text;
    el.scrollTop = el.scrollHeight;
  }}
}}
tick(); setInterval(tick, 2000);
</script></body></html>"""
    return page.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "BYOVideoNIMSwitch/1.0"

    def _headers(self, status: int, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        self._headers(status)
        self.wfile.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._headers(204)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"", "/"}:
            self._headers(200, "text/html; charset=utf-8")
            self.wfile.write(_html_page())
            return
        if parsed.path == "/api/state":
            state = _read_json(STATE_FILE)
            state["current"] = _current_runtime()
            state["credentials_present"] = _credentials_present()
            state = _enrich_state(state)
            self._json(200, state)
            return
        if parsed.path == "/api/log":
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                offset = int((qs.get("offset") or ["-1"])[0])
            except Exception:
                offset = -1
            LOG_FILE.touch(exist_ok=True)
            size = LOG_FILE.stat().st_size
            if offset < 0 or offset > size:
                offset = max(0, size - 65536)
            with LOG_FILE.open("rb") as fh:
                fh.seek(offset)
                data = fh.read(65536)
                next_offset = fh.tell()
            self._json(200, {
                "offset": offset,
                "next_offset": next_offset,
                "text": data.decode("utf-8", errors="replace"),
            })
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("content-length") or "0")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception:
            self._json(400, {"error": "invalid JSON"})
            return
        if parsed.path == "/api/credentials":
            try:
                _write_credentials(payload)
                _write_state({"credentials_present": True})
                self._json(200, {"ok": True})
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        if parsed.path == "/api/switch":
            short_id = str(payload.get("short_id") or payload.get("model") or "").strip()
            target = _resolve_target(short_id)
            if not target:
                self._json(404, {"error": f"Unsupported or unknown NIM: {short_id}"})
                return
            if not target.get("custom_current") and not _credentials_present():
                self._json(409, {
                    "error": f"Missing NGC credentials. Add them in Gradio or write {CREDENTIAL_FILE} chmod 0600."
                })
                return
            if not _SWITCH_LOCK.acquire(blocking=False):
                self._json(409, {"error": "A NIM switch is already in progress", "state": _read_json(STATE_FILE)})
                return
            threading.Thread(target=_switch_worker, args=(target,), daemon=True).start()
            self._json(202, {"ok": True, "target": target, "state": _read_json(STATE_FILE)})
            return
        self._json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        _append_log(fmt % args)


def main() -> int:
    _initial_state()
    _append_log(f"Starting NIM switch service on {HOST}:{PORT}")
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
