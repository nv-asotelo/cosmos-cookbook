#!/usr/bin/env python3
"""Small companion service for zero-touch BYO-video NIM swaps.

The service stays outside Gradio so users can keep watching progress while the
Gradio process is restarted after a successful NIM container swap.
"""
from __future__ import annotations

import json
import os
import re
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
_STEP_DEFS = [
    ("starting", "Request queued", "Switch request accepted and target recorded.", 5),
    ("validating", "Validate target", "Resolve catalog metadata and inspect the currently running NIM.", 10),
    ("launching_nim", "Launch helper", "Start nim_launch.sh with the selected image and served model id.", 15),
    ("pulling", "Pull image", "Download or reuse the NIM container image from nvcr.io.", 35),
    ("starting_container", "Start container", "Replace cosmos-nim and bind the OpenAI-compatible API on port 8000.", 55),
    ("waiting_nim", "Wait for model API", "Poll /v1/models while NIM loads weights and builds its runtime profile.", 75),
    ("nim_ready", "NIM ready", "Confirm /v1/models returns the selected served model.", 88),
    ("restarting_gradio", "Restart Gradio", "Restart the Gradio UI so the dropdown defaults to the active NIM.", 95),
    ("ready", "Ready", "Gradio and NIM are online.", 100),
]
_PHASE_INDEX = {phase: index for index, (phase, *_rest) in enumerate(_STEP_DEFS)}
_DEFAULT_ESTIMATE_S = 643.0  # observed cosmos-reason2-8b switch on RTX PRO 6000 Blackwell
_STEP_ESTIMATE_S = {
    "starting": 2.0,
    "validating": 2.0,
    "launching_nim": 5.0,
    "pulling": 70.0,
    "starting_container": 8.0,
    "waiting_nim": 540.0,
    "nim_ready": 4.0,
    "restarting_gradio": 12.0,
    "ready": 0.0,
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


def _estimate_total_s(state: Dict[str, Any]) -> float:
    for key in ("last_success_elapsed_s", "estimated_total_s"):
        try:
            value = float(state.get(key) or 0)
            if value > 0:
                return value
        except Exception:
            pass
    return _DEFAULT_ESTIMATE_S


def _step_details(state: Dict[str, Any], phase: str) -> list[Dict[str, Any]]:
    current_index = _PHASE_INDEX.get(phase, 0)
    history = state.get("phase_history") or []
    durations = {
        item.get("phase"): item.get("duration_s")
        for item in history
        if isinstance(item, dict)
    }
    using_estimated_durations = False
    if not any(float(v or 0) > 0 for v in durations.values()) and phase in {"ready", "error"}:
        total = float(state.get("last_success_elapsed_s") or state.get("elapsed_s") or _DEFAULT_ESTIMATE_S)
        estimate_total = sum(_STEP_ESTIMATE_S.values()) or _DEFAULT_ESTIMATE_S
        scale = total / estimate_total if estimate_total else 1.0
        durations = {key: value * scale for key, value in _STEP_ESTIMATE_S.items()}
        using_estimated_durations = True
    if phase in _ACTIVE_PHASES:
        try:
            durations[phase] = max(0.0, time.time() - float(state.get("phase_started_at") or time.time()))
        except Exception:
            pass
    steps: list[Dict[str, Any]] = []
    for index, (step_phase, title, detail, pct) in enumerate(_STEP_DEFS):
        if phase == "error" and index >= current_index:
            status = "error" if index == current_index else "pending"
        elif phase == "ready" or index < current_index:
            status = "done"
        elif index == current_index:
            status = "running" if phase in _ACTIVE_PHASES else "done"
        else:
            status = "pending"
        step_detail = detail
        if step_phase == "pulling":
            done = int(state.get("pull_layers_done") or 0)
            total = int(state.get("pull_layers_total") or 0)
            if total:
                step_detail = f"{detail} Layers complete: {done}/{total}."
        elif step_phase == "waiting_nim":
            wait_s = state.get("nim_wait_s")
            if wait_s is not None:
                step_detail = f"{detail} NIM wait elapsed: {_format_duration(float(wait_s))}."
        steps.append({
            "phase": step_phase,
            "title": title,
            "detail": step_detail,
            "status": status,
            "progress_pct": pct,
            "duration_s": round(float(durations.get(step_phase) or 0), 1),
            "duration_label": (
                f"~{_format_duration(float(durations.get(step_phase) or 0))}"
                if using_estimated_durations and float(durations.get(step_phase) or 0) > 0
                else _format_duration(float(durations.get(step_phase) or 0))
            ),
            "duration_estimated": using_estimated_durations,
        })
    return steps


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
    estimate_s = _estimate_total_s(state)
    elapsed_progress = int(min(95, max(0, elapsed_s / estimate_s * 100))) if estimate_s else 0
    progress_pct = int(max(_PHASE_PROGRESS.get(phase, 0), elapsed_progress)) if phase in _ACTIVE_PHASES else int(_PHASE_PROGRESS.get(phase, 0))
    if phase == "ready":
        eta_s = 0.0
        eta_label = "Complete"
        if elapsed_s:
            state["last_success_elapsed_s"] = round(elapsed_s, 1)
            state["last_success_label"] = _format_duration(elapsed_s)
    elif phase == "error":
        eta_s = None
        eta_label = "Failed"
    elif phase in _ACTIVE_PHASES:
        eta_s = max(0.0, estimate_s - elapsed_s)
        eta_label = _format_duration(eta_s)
    else:
        eta_s = None
        eta_label = "Waiting for switch"
    state["elapsed_s"] = round(elapsed_s, 1)
    state["elapsed_label"] = _format_duration(elapsed_s)
    state["progress_pct"] = progress_pct
    state["estimated_total_s"] = round(estimate_s, 1)
    state["estimated_total_label"] = _format_duration(estimate_s)
    state["eta_s"] = round(eta_s, 1) if eta_s is not None else None
    state["eta_label"] = eta_label
    state["current_step"] = next((s["title"] for s in _step_details(state, phase) if s["status"] == "running"), "Ready" if phase == "ready" else phase)
    state["steps"] = _step_details(state, phase)
    previous_model = state.get("previous")
    target_model = state.get("target")
    if previous_model or target_model:
        state["model_change"] = _infer_changelog(previous_model, target_model)
    return state


def _write_state(update: Dict[str, Any]) -> Dict[str, Any]:
    with _STATE_LOCK:
        state = _read_json(STATE_FILE)
        now = time.time()
        old_phase = state.get("phase")
        new_phase = update.get("phase", old_phase)
        if new_phase and new_phase != old_phase:
            history = list(state.get("phase_history") or [])
            previous_started = state.get("phase_started_at")
            if old_phase and previous_started:
                duration = max(0.0, now - float(previous_started))
                history.append({
                    "phase": old_phase,
                    "started_at": previous_started,
                    "ended_at": now,
                    "duration_s": round(duration, 1),
                    "duration_label": _format_duration(duration),
                })
                history = history[-20:]
            state["phase_history"] = history
            state["phase_started_at"] = now
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


def _known_catalog() -> list[Any]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in (script_dir, "/tmp"):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    try:
        from nim_catalog import KNOWN_VLM_NIMS  # type: ignore

        return list(KNOWN_VLM_NIMS)
    except Exception as exc:
        _append_log(f"Known catalog load failed: {exc}")
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


def _model_tokens(model: Optional[Dict[str, Any]]) -> set[str]:
    if not model:
        return set()
    vals = [
        model.get("short_id"),
        model.get("served_model_id"),
        model.get("label"),
        model.get("image"),
    ]
    return {str(v).strip().lower() for v in vals if v}


def _catalog_match(model: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    tokens = _model_tokens(model)
    if not tokens:
        return None
    for nim in _known_catalog():
        data = _nim_to_dict(nim)
        if tokens & _model_tokens(data):
            data["catalog_source"] = "BYO-video NIM catalog"
            return data
    return None


def _runtime_model(runtime: Optional[Dict[str, Any]], label_prefix: str = "Current") -> Optional[Dict[str, Any]]:
    if not runtime:
        return None
    base = {
        "short_id": "",
        "label": runtime.get("served_model_id") or f"{label_prefix} NIM",
        "family": "Unknown/custom",
        "served_model_id": runtime.get("served_model_id") or "",
        "image": runtime.get("image") or "",
        "min_vram_mb": 0,
        "supports_video": True,
        "switchable": False,
        "env": {},
        "notes": "Runtime NIM detected from docker inspect and /v1/models.",
        "catalog_source": "runtime",
    }
    matched = _catalog_match(base)
    if matched:
        return matched
    served = str(base["served_model_id"]).lower()
    image = str(base["image"]).lower()
    if "cosmos3-super-reasoner" in served or "cosmos3-super-reasoner" in image:
        base.update({
            "short_id": "custom-cosmos3-super-reasoner",
            "label": "Current custom: nvidia/Cosmos3-Super-Reasoner",
            "family": "Cosmos3 Super Reasoner",
            "supports_video": True,
            "notes": (
                "Custom staged NIM image; not present in the public BYO-video "
                "NIM catalog. Treated as video-capable because it was already "
                "serving successfully on this target."
            ),
            "catalog_source": "runtime custom",
        })
    return base


def _model_summary(model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    model = model or {}
    return {
        "label": model.get("label") or model.get("served_model_id") or "unknown",
        "family": model.get("family") or "unknown",
        "short_id": model.get("short_id") or "",
        "served_model_id": model.get("served_model_id") or "",
        "image": model.get("image") or "",
        "min_vram_mb": model.get("min_vram_mb") or 0,
        "supports_video": bool(model.get("supports_video")),
        "switchable": bool(model.get("switchable")),
        "notes": model.get("notes") or "",
        "catalog_source": model.get("catalog_source") or "BYO-video NIM catalog",
    }


def _sentence_from_note(note: str) -> str:
    note = " ".join(str(note or "").split())
    if not note:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", note)
    return parts[0].strip()


def _infer_changelog(previous: Optional[Dict[str, Any]], target: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    prev = _model_summary(previous)
    new = _model_summary(target)
    changes: list[Dict[str, str]] = []

    def add(title: str, detail: str, kind: str = "changed") -> None:
        if detail:
            changes.append({"title": title, "detail": detail, "kind": kind})

    if not previous:
        add("Previous NIM unknown", "This switch started before previous-NIM capture was added; future switches preserve it.", "warning")
    elif prev["served_model_id"] != new["served_model_id"]:
        add("Served model changed", f"{prev['served_model_id'] or prev['label']} -> {new['served_model_id'] or new['label']}")
    else:
        add("Served model unchanged", f"Still serving {new['served_model_id'] or new['label']}.", "same")

    if prev["image"] != new["image"]:
        add("Container image changed", f"{prev['image'] or 'unknown'} -> {new['image'] or 'unknown'}")
    if prev["family"] != new["family"]:
        add("Model family changed", f"{prev['family']} -> {new['family']}")
    if prev["catalog_source"] != new["catalog_source"]:
        add("Catalog status changed", f"{prev['catalog_source']} -> {new['catalog_source']}")
    if prev["min_vram_mb"] != new["min_vram_mb"]:
        old = f"{prev['min_vram_mb']} MiB" if prev["min_vram_mb"] else "unknown/custom"
        new_vram = f"{new['min_vram_mb']} MiB" if new["min_vram_mb"] else "unknown/custom"
        add("VRAM requirement changed", f"{old} -> {new_vram}")
    if prev["supports_video"] != new["supports_video"]:
        add("Video support changed", f"{prev['supports_video']} -> {new['supports_video']}")
    elif new["supports_video"]:
        add("Video support retained", "Both previous and selected NIM are treated as video-capable.", "same")
    note = _sentence_from_note(new["notes"])
    if note:
        add("New NIM catalog note", note)
    if "temperature" in str(new["notes"]).lower() or "<think>" in str(new["notes"]).lower():
        add("Runtime parameter caution", "Catalog notes recommend temperature >= 0.3 for this NIM to avoid the greedy <think> termination issue.", "warning")
    if not changes:
        add("No catalog delta detected", "The selected target matches the previous runtime metadata.", "same")
    return {
        "title": f"{prev['label']} -> {new['label']}",
        "previous": prev,
        "target": new,
        "items": changes,
        "source": "Inferred from docker runtime metadata plus the BYO-video NIM catalog when available.",
    }


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
    pull_seen: set[str] = set()
    pull_done: set[str] = set()
    for line in proc.stdout:
        clean = line.rstrip()
        _append_log(f"nim_launch: {clean}")
        low = clean.lower()
        layer_id = clean.split(":", 1)[0].strip() if ":" in clean else ""
        if "pulling " in low:
            if layer_id:
                pull_seen.add(layer_id)
            _write_state({
                "phase": "pulling",
                "message": clean,
                "pull_layers_total": len(pull_seen),
                "pull_layers_done": len(pull_done),
            })
        elif "pull complete" in low:
            if layer_id:
                pull_seen.add(layer_id)
                pull_done.add(layer_id)
            _write_state({
                "phase": "pulling",
                "message": clean,
                "pull_layers_total": len(pull_seen),
                "pull_layers_done": len(pull_done),
            })
        elif "starting nim container" in low:
            _write_state({"phase": "starting_container", "message": clean})
        elif "waiting for nim" in low or "waiting for /v1/models" in low:
            _write_state({"phase": "waiting_nim", "message": clean})
        elif "still waiting" in low:
            match = re.search(r"(\d+)s elapsed", clean)
            _write_state({
                "phase": "waiting_nim",
                "message": clean,
                "nim_wait_s": int(match.group(1)) if match else None,
            })
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
        previous_runtime = _current_runtime()
        previous_model = _runtime_model(previous_runtime, label_prefix="Previous")
        _write_state({
            "phase": "starting",
            "started_at": time.time(),
            "phase_started_at": time.time(),
            "phase_history": [],
            "pull_layers_total": 0,
            "pull_layers_done": 0,
            "nim_wait_s": 0,
            "previous": previous_model,
            "previous_runtime": previous_runtime,
            "target": target,
            "error": None,
            "message": f"Switch requested for {target['label']}",
        })
        current = previous_runtime
        if target.get("custom_current") or (
            current.get("served_model_id") == target.get("served_model_id")
            and (not target.get("image") or current.get("image") == target.get("image"))
        ):
            _phase("ready", "Selected NIM is already running", current=current, previous=previous_model)
            return
        _phase("validating", "Resolved catalog metadata", current=current, previous=previous_model)
        _run_nim_launch(target)
        current = _current_runtime()
        _phase("nim_ready", "NIM /v1/models is serving", current=current, previous=previous_model)
        _restart_gradio(target)
        _phase("ready", "Switch complete; Gradio is back online", current=_current_runtime(), previous=previous_model)
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
.card{{max-width:1100px;margin:0 auto;background:white;border:1px solid #cbd5e1;border-radius:8px;padding:24px}}
pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;border-radius:6px;padding:12px;max-height:460px;overflow:auto}}
.pill{{display:inline-block;background:#dbeafe;color:#1e3a8a;border-radius:999px;padding:3px 10px;margin-left:6px;font-size:14px}}
.progress{{height:16px;background:#e2e8f0;border-radius:999px;overflow:hidden;margin:16px 0 8px}}
.bar{{height:100%;width:0%;background:#22c55e;transition:width .4s ease}}
.meta{{display:flex;gap:18px;flex-wrap:wrap;color:#475569;font-size:14px;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px;margin:16px 0}}
.metric{{border:1px solid #cbd5e1;background:#f8fafc;border-radius:8px;padding:12px}}
.metric .label{{color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
.metric .value{{margin-top:5px;font-weight:700;font-size:18px;color:#0f172a;word-break:break-word}}
.section{{margin-top:22px}}
.section h2{{font-size:18px;margin:0 0 10px}}
.steps{{display:grid;gap:8px}}
.step{{display:grid;grid-template-columns:92px 1fr auto;gap:10px;align-items:start;border:1px solid #cbd5e1;border-radius:8px;padding:10px;background:#fff}}
.step.done{{border-color:#86efac;background:#f0fdf4}}
.step.running{{border-color:#60a5fa;background:#eff6ff}}
.step.error{{border-color:#fca5a5;background:#fef2f2}}
.status{{font-size:12px;font-weight:700;text-transform:uppercase;color:#475569}}
.step.running .status{{color:#1d4ed8}}
.step.done .status{{color:#15803d}}
.step.error .status{{color:#b91c1c}}
.step-title{{font-weight:700}}
.step-detail{{color:#475569;font-size:13px;margin-top:2px}}
.models{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.model{{border:1px solid #cbd5e1;border-radius:8px;padding:12px;background:#f8fafc}}
.model b{{display:block;margin-bottom:6px}}
.changes{{display:grid;gap:8px}}
.change{{border:1px solid #cbd5e1;border-radius:8px;padding:10px;background:#fff}}
.change.warning{{border-color:#fbbf24;background:#fffbeb}}
.change.same{{border-color:#86efac;background:#f0fdf4}}
.change-title{{font-weight:700}}
.change-detail{{color:#475569;font-size:13px;margin-top:3px}}
details{{margin-top:14px}}
summary{{cursor:pointer;color:#1d4ed8;font-weight:700}}
a{{color:#1d4ed8}}
</style></head>
<body><div class="card">
<h1>BYO-video NIM Switch <span id="phase" class="pill">loading</span></h1>
<p>Progress page for backend swaps. Gradio: <a href="{PUBLIC_GRADIO_URL}">{PUBLIC_GRADIO_URL}</a></p>
<div class="progress"><div id="progressBar" class="bar"></div></div>
<div class="meta">
  <span id="progressText">Progress: 0%</span>
  <span id="elapsed">Elapsed: 0s</span>
  <span id="eta">ETA: n/a</span>
  <span id="estimate">Estimate: n/a</span>
</div>
<p id="message"></p>
<div class="grid">
  <div class="metric"><div class="label">Current step</div><div id="currentStep" class="value">loading</div></div>
  <div class="metric"><div class="label">Target</div><div id="targetModel" class="value">loading</div></div>
  <div class="metric"><div class="label">Running model</div><div id="currentModel" class="value">loading</div></div>
  <div class="metric"><div class="label">Container</div><div id="containerImage" class="value">loading</div></div>
</div>
<div class="section">
  <h2>Steps</h2>
  <div id="steps" class="steps"></div>
</div>
<div class="section">
  <h2>Models</h2>
  <div class="models">
    <div class="model"><b>Previous NIM</b><div id="previousDetail">loading</div></div>
    <div class="model"><b>Selected target</b><div id="targetDetail">loading</div></div>
    <div class="model"><b>Currently served</b><div id="currentDetail">loading</div></div>
  </div>
</div>
<div class="section">
  <h2>What changed</h2>
  <div id="changeIntro" class="step-detail"></div>
  <div id="changes" class="changes"></div>
</div>
<details>
  <summary>Raw state JSON</summary>
  <pre id="state"></pre>
</details>
<details>
  <summary>Live supervisor log</summary>
  <pre id="log"></pre>
</details>
</div>
<script>
let offset = 0;
function esc(value){{
  return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}
function short(value){{
  value = String(value || "");
  return value.length > 44 ? value.slice(0, 41) + "..." : value;
}}
function modelDetail(model){{
  if (!model) return "n/a";
  const bits = [];
  if (model.label) bits.push("<div>" + esc(model.label) + "</div>");
  if (model.served_model_id) bits.push("<div><code>" + esc(model.served_model_id) + "</code></div>");
  if (model.image) bits.push("<div><code>" + esc(model.image) + "</code></div>");
  if (model.min_vram_mb) bits.push("<div>Min VRAM: " + esc(model.min_vram_mb) + " MiB</div>");
  return bits.join("");
}}
function currentDetail(current){{
  if (!current) return "n/a";
  return [
    "<div><code>" + esc(current.served_model_id || "not ready") + "</code></div>",
    "<div><code>" + esc(current.image || "unknown image") + "</code></div>",
    "<div>Ready: " + esc(Boolean(current.ready)) + "</div>"
  ].join("");
}}
function renderChanges(change){{
  const intro = document.getElementById("changeIntro");
  const el = document.getElementById("changes");
  if (!change) {{
    intro.textContent = "No previous/target comparison is available yet.";
    el.innerHTML = "";
    return;
  }}
  intro.textContent = (change.title || "Model change") + ". " + (change.source || "");
  el.innerHTML = (change.items || []).map(item => {{
    const kind = item.kind || "changed";
    return '<div class="change ' + esc(kind) + '">' +
      '<div class="change-title">' + esc(item.title) + '</div>' +
      '<div class="change-detail">' + esc(item.detail) + '</div>' +
      '</div>';
  }}).join("");
}}
function renderSteps(steps){{
  const el = document.getElementById("steps");
  el.innerHTML = (steps || []).map(s => {{
    const status = s.status || "pending";
    return '<div class="step ' + esc(status) + '">' +
      '<div class="status">' + esc(status) + '</div>' +
      '<div><div class="step-title">' + esc(s.title) + '</div>' +
      '<div class="step-detail">' + esc(s.detail) + '</div></div>' +
      '<div class="step-detail">' + esc(s.duration_label || "") + '</div>' +
      '</div>';
  }}).join("");
}}
async function tick(){{
  const s = await fetch("{state_url}").then(r => r.json());
  document.getElementById("phase").textContent = s.phase || "unknown";
  const pct = Math.max(0, Math.min(100, Number(s.progress_pct || 0)));
  document.getElementById("progressBar").style.width = pct + "%";
  document.getElementById("progressText").textContent = "Progress: " + pct + "%";
  document.getElementById("elapsed").textContent = "Elapsed: " + (s.elapsed_label || "0s");
  document.getElementById("eta").textContent = "ETA: " + (s.eta_label || "n/a");
  document.getElementById("estimate").textContent = "Estimate: " + (s.estimated_total_label || "n/a");
  document.getElementById("message").textContent = s.message || "";
  document.getElementById("currentStep").textContent = s.current_step || s.phase || "unknown";
  document.getElementById("targetModel").textContent = short((s.target && (s.target.label || s.target.served_model_id)) || "none");
  document.getElementById("currentModel").textContent = short(s.current && s.current.served_model_id || "not ready");
  document.getElementById("containerImage").textContent = short(s.current && s.current.image || "unknown");
  document.getElementById("previousDetail").innerHTML = modelDetail(s.previous || (s.model_change && s.model_change.previous));
  document.getElementById("targetDetail").innerHTML = modelDetail(s.target);
  document.getElementById("currentDetail").innerHTML = currentDetail(s.current);
  renderChanges(s.model_change);
  renderSteps(s.steps || []);
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
