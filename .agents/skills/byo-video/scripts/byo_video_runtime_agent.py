#!/usr/bin/env python3
"""Runtime-agent frontend for Cosmos BYO-video inference.

This is intentionally self-contained: it serves a small browser UI, loads video
samples from a public Hugging Face dataset through FiftyOne when available, and
runs selected videos concurrently against the OpenAI-compatible vLLM/NIM server
that the BYO-video setup script starts.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import html
import json
import mimetypes
import os
import shutil
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import requests
except Exception as exc:  # pragma: no cover - surfaced in UI
    requests = None
    REQUESTS_IMPORT_ERROR = exc
else:
    REQUESTS_IMPORT_ERROR = None

WORKER_SAFETY_SYSTEM = """
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
""".strip()

WORKER_SAFETY_USER = """
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
""".strip()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
DEFAULT_DATASET = os.getenv("RUNTIME_AGENT_DATASET", "pjramg/Safe_Unsafe_Test")
RESULTS_FILE = Path(os.getenv("RUNTIME_AGENT_RESULTS", "/tmp/byo_video_runtime_agent_results.json"))
STATE_LOCK = threading.Lock()
FIFTYONE_SESSION = None

STATE: Dict[str, Any] = {
    "dataset_repo": DEFAULT_DATASET,
    "dataset_source": None,
    "fo_dataset_name": None,
    "videos": [],
    "results": [],
    "running": False,
    "progress": {"done": 0, "total": 0, "errors": 0},
    "logs": [],
    "server": {},
    "defaults": {
        "system_prompt": WORKER_SAFETY_SYSTEM,
        "user_prompt": WORKER_SAFETY_USER,
        "concurrency": int(os.getenv("RUNTIME_AGENT_CONCURRENCY", "4")),
        "max_videos": int(os.getenv("RUNTIME_AGENT_MAX_VIDEOS", "20")),
        "fps": float(os.getenv("RUNTIME_AGENT_FPS", "1")),
        "max_tokens": int(os.getenv("RUNTIME_AGENT_MAX_TOKENS", "1024")),
    },
}


def log(message: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + message
    with STATE_LOCK:
        STATE["logs"].append(line)
        STATE["logs"] = STATE["logs"][-200:]
    print("[runtime-agent]", message, flush=True)


def snapshot() -> Dict[str, Any]:
    with STATE_LOCK:
        return json.loads(json.dumps(STATE, default=str))


def update_state(**items: Any) -> None:
    with STATE_LOCK:
        STATE.update(items)


def video_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8", "ignore")).hexdigest()[:16]


def is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def load_with_fiftyone(repo_id: str, max_videos: int) -> List[Dict[str, Any]]:
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh

    name = repo_id.replace("/", "_")
    if name in fo.list_datasets():
        dataset = fo.load_dataset(name)
    else:
        try:
            dataset = fouh.load_from_hub(repo_id, dataset_name=name, max_samples=max_videos, persistent=True)
        except TypeError:
            try:
                dataset = fouh.load_from_hub(repo_id, dataset_name=name, persistent=True)
            except TypeError:
                dataset = fouh.load_from_hub(repo_id, persistent=True)

    videos: List[Dict[str, Any]] = []
    for sample in dataset:
        path = str(sample.filepath)
        if not is_video_path(path):
            continue
        videos.append({
            "id": video_id(path),
            "name": Path(path).name,
            "filepath": path,
            "sample_id": str(sample.id),
            "label": guess_label(sample),
            "source": "fiftyone",
        })
        if len(videos) >= max_videos:
            break
    if not videos:
        raise RuntimeError(f"FiftyOne loaded {repo_id}, but no video samples were found")
    update_state(dataset_source="fiftyone", fo_dataset_name=name)
    return videos


def guess_label(sample: Any) -> Optional[str]:
    for field in ("ground_truth", "label", "classification", "class", "hazard"):
        if not hasattr(sample, field):
            continue
        value = getattr(sample, field)
        if hasattr(value, "label"):
            return str(value.label)
        if value is not None and not isinstance(value, (dict, list)):
            return str(value)
    return None


def load_with_hf_hub(repo_id: str, max_videos: int) -> List[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo_id, repo_type="dataset") if is_video_path(f)]
    if not files:
        raise RuntimeError(f"No video files found in Hugging Face dataset {repo_id}")
    local_root = Path("/tmp/byo_video_datasets") / repo_id.replace("/", "_")
    local_root.mkdir(parents=True, exist_ok=True)
    videos: List[Dict[str, Any]] = []
    for file_name in files[:max_videos]:
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file_name, local_dir=str(local_root))
        videos.append({
            "id": video_id(local),
            "name": file_name,
            "filepath": local,
            "sample_id": None,
            "label": None,
            "source": "hf_hub",
        })
    try:
        import fiftyone as fo

        name = repo_id.replace("/", "_") + "_runtime"
        if name in fo.list_datasets():
            dataset = fo.load_dataset(name)
            try:
                dataset.delete_samples(dataset.values("id"))
            except Exception:
                pass
        else:
            dataset = fo.Dataset(name, persistent=True)

        for video in videos:
            sample = fo.Sample(filepath=video["filepath"])
            sample["hf_repo"] = repo_id
            sample["hf_path"] = video["name"]
            dataset.add_sample(sample)
            video["sample_id"] = str(sample.id)
            video["source"] = "fiftyone"
        update_state(dataset_source="fiftyone", fo_dataset_name=name)
        log(f"Wrapped HF files in local FiftyOne dataset {name}")
    except Exception as exc:
        log(f"Could not wrap HF files in FiftyOne: {exc}")
        update_state(dataset_source="hf_hub", fo_dataset_name=None)
    return videos


def load_dataset(repo_id: str, max_videos: int) -> List[Dict[str, Any]]:
    log(f"Loading dataset {repo_id} (up to {max_videos} videos)")
    try:
        videos = load_with_fiftyone(repo_id, max_videos)
        log(f"Loaded {len(videos)} videos with FiftyOne")
    except Exception as exc:
        log(f"FiftyOne load failed; falling back to huggingface_hub: {exc}")
        videos = load_with_hf_hub(repo_id, max_videos)
        log(f"Loaded {len(videos)} videos with huggingface_hub")
    update_state(dataset_repo=repo_id, videos=videos, results=[], progress={"done": 0, "total": 0, "errors": 0})
    return videos


def detect_server() -> Dict[str, Any]:
    info = {"base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"), "model": os.getenv("MODEL_NAME", "")}
    if requests is None:
        info["error"] = f"requests import failed: {REQUESTS_IMPORT_ERROR}"
        update_state(server=info)
        return info
    try:
        resp = requests.get(info["base_url"].rstrip("/") + "/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models and not info["model"]:
            info["model"] = models[0].get("id") or models[0].get("root") or ""
        info["models"] = [m.get("id") or m.get("root") for m in models]
    except Exception as exc:
        info["error"] = str(exc)
    update_state(server=info)
    return info


def extract_frames_b64(path: str, fps: float, max_frames: int = 8) -> List[str]:
    import av

    frames: List[str] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 30)
        step = max(1, int(rate / max(fps, 0.1)))
        for index, frame in enumerate(container.decode(stream)):
            if index % step:
                continue
            image = frame.to_image().convert("RGB")
            from io import BytesIO

            buf = BytesIO()
            image.save(buf, format="JPEG", quality=85)
            frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            if len(frames) >= max_frames:
                break
    return frames


def model_prefers_file_url(model: str) -> bool:
    lower = model.lower()
    return "qwen" in lower or "nemotron" in lower


def model_prefers_video_data(model: str) -> bool:
    lower = model.lower()
    backend = os.getenv("INFERENCE_BACKEND", "").lower()
    return backend == "nim" or ("cosmos-reason" in lower and "nim" in lower)


def content_for_video(video_path: str, prompt: str, model: str, fps: float) -> List[Dict[str, Any]]:
    if model_prefers_video_data(model):
        mime = mimetypes.guess_type(video_path)[0] or "video/mp4"
        data = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")
        return [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}},
        ]
    if model_prefers_file_url(model):
        tmp = Path("/tmp") / ("byo_agent_" + Path(video_path).name)
        if Path(video_path).resolve() != tmp.resolve():
            shutil.copy2(video_path, tmp)
        return [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": tmp.as_uri()}},
        ]
    frames = extract_frames_b64(video_path, fps=fps)
    if not frames:
        raise RuntimeError("No frames could be extracted from the video")
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    return content


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def run_one(video: Dict[str, Any], system_prompt: str, user_prompt: str, fps: float, max_tokens: int) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError(f"requests import failed: {REQUESTS_IMPORT_ERROR}")
    server = detect_server()
    base_url = server.get("base_url") or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = server.get("model") or os.getenv("MODEL_NAME") or "cosmos-reason"
    headers = {"Content-Type": "application/json"}
    if os.getenv("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('VLLM_API_KEY')}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_for_video(video["filepath"], user_prompt, model, fps)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.95,
        "stream": False,
    }
    resp = requests.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload, timeout=900)
    resp.raise_for_status()
    data = resp.json()
    message = data.get("choices", [{}])[0].get("message", {})
    text = message.get("content") or data.get("choices", [{}])[0].get("text") or ""
    parsed = parse_json_from_text(text)
    result = {
        "id": video["id"],
        "name": video["name"],
        "source_label": video.get("label"),
        "response": text,
        "json": parsed,
        "error": None,
    }
    write_fiftyone_result(video, result)
    return result


def write_fiftyone_result(video: Dict[str, Any], result: Dict[str, Any]) -> None:
    snap = snapshot()
    if snap.get("dataset_source") != "fiftyone" or not video.get("sample_id") or not snap.get("fo_dataset_name"):
        return
    try:
        import fiftyone as fo

        dataset = fo.load_dataset(snap["fo_dataset_name"])
        sample = dataset[video["sample_id"]]
        sample["runtime_agent_response"] = result.get("response") or ""
        if result.get("json") is not None:
            sample["runtime_agent_json"] = result["json"]
            label = result["json"].get("prediction_label")
            if label:
                sample["runtime_agent_prediction"] = fo.Classification(label=str(label))
        if result.get("error"):
            sample["runtime_agent_error"] = str(result["error"])
        sample.save()
    except Exception as exc:
        log(f"Could not write result back to FiftyOne: {exc}")


def run_batch(ids: Iterable[str], concurrency: int, system_prompt: str, user_prompt: str, fps: float, max_tokens: int) -> None:
    videos_by_id = {v["id"]: v for v in snapshot()["videos"]}
    selected = [videos_by_id[i] for i in ids if i in videos_by_id] or list(videos_by_id.values())
    update_state(running=True, results=[], progress={"done": 0, "total": len(selected), "errors": 0})
    log(f"Running {len(selected)} videos with concurrency={concurrency}")
    results: List[Dict[str, Any]] = []
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        future_map = {pool.submit(run_one, v, system_prompt, user_prompt, fps, max_tokens): v for v in selected}
        for future in concurrent.futures.as_completed(future_map):
            video = future_map[future]
            try:
                result = future.result()
                log(f"Completed {video['name']}")
            except Exception as exc:
                errors += 1
                result = {"id": video["id"], "name": video["name"], "source_label": video.get("label"), "response": "", "json": None, "error": str(exc)}
                write_fiftyone_result(video, result)
                log(f"Failed {video['name']}: {exc}")
            results.append(result)
            with STATE_LOCK:
                STATE["results"] = results
                STATE["progress"] = {"done": len(results), "total": len(selected), "errors": errors}
            RESULTS_FILE.write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")
    update_state(running=False)
    log(f"Batch complete: {len(results) - errors} ok, {errors} errors")


def launch_fiftyone(port: int) -> str:
    global FIFTYONE_SESSION
    snap = snapshot()
    if snap.get("dataset_source") != "fiftyone" or not snap.get("fo_dataset_name"):
        raise RuntimeError("Load the dataset with FiftyOne before launching the FiftyOne app")
    import fiftyone as fo

    dataset = fo.load_dataset(snap["fo_dataset_name"])
    FIFTYONE_SESSION = fo.launch_app(dataset, address="0.0.0.0", port=port, auto=False)
    url = f"http://{os.getenv('HOST_IP') or detect_host_ip() or 'localhost'}:{port}/"
    log(f"FiftyOne app is available at {url}")
    return url


def detect_host_ip() -> Optional[str]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        return None
    return None


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cosmos BYO Video Runtime Agent</title>
<style>
:root { color-scheme: light; --ink:#1f2937; --muted:#6b7280; --line:#d8dee8; --panel:#ffffff; --bg:#f5f7fb; --accent:#0f766e; --warn:#b45309; --bad:#b91c1c; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
header { padding:18px 28px; border-bottom:1px solid var(--line); background:#fff; display:flex; align-items:center; justify-content:space-between; gap:20px; }
h1 { font-size:20px; margin:0; letter-spacing:0; }
main { max-width:1280px; margin:0 auto; padding:20px; display:grid; grid-template-columns: 340px 1fr; gap:18px; }
section, aside { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
input, textarea { width:100%; border:1px solid var(--line); border-radius:6px; padding:9px 10px; font:inherit; background:#fff; }
textarea { min-height:120px; resize:vertical; }
button { border:0; border-radius:6px; padding:9px 12px; font-weight:650; color:#fff; background:var(--accent); cursor:pointer; }
button.secondary { background:#334155; }
button.warn { background:var(--warn); }
button:disabled { opacity:.55; cursor:not-allowed; }
.actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.kv { display:grid; grid-template-columns:110px 1fr; gap:6px; font-size:13px; color:var(--muted); }
.guide { display:grid; gap:8px; }
.step { border-left:3px solid var(--line); padding-left:10px; color:var(--muted); font-size:13px; }
.step strong { color:var(--ink); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { border-bottom:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; }
th { color:var(--muted); font-size:12px; font-weight:650; }
.results { max-height:420px; overflow:auto; border:1px solid var(--line); border-radius:6px; }
.log { height:150px; overflow:auto; background:#0f172a; color:#d1fae5; padding:10px; border-radius:6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; white-space:pre-wrap; }
.pill { display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:3px 8px; font-size:12px; color:var(--muted); background:#fff; }
.progress { height:8px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
.progress div { height:100%; width:0; background:var(--accent); transition:width .2s ease; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; padding:14px; } header { padding:14px; } }
</style>
</head>
<body>
<header>
  <h1>Cosmos BYO Video Runtime Agent</h1>
  <span class="pill" id="serverPill">checking backend</span>
</header>
<main>
<aside>
  <div class="guide">
    <div class="step"><strong>1. Backend</strong><br/>The agent uses the local OpenAI-compatible vLLM/NIM endpoint.</div>
    <div class="step"><strong>2. Dataset</strong><br/>Load a public Hugging Face dataset, then choose any number of videos.</div>
    <div class="step"><strong>3. Prompt</strong><br/>The worker-safety smoke prompt is preloaded and editable.</div>
    <div class="step"><strong>4. Run</strong><br/>Selected videos are processed concurrently and results are written back to FiftyOne when available.</div>
  </div>
  <label>Hugging Face dataset</label>
  <input id="repo" value="pjramg/Safe_Unsafe_Test" />
  <label>Max videos to load</label>
  <input id="maxVideos" type="number" min="1" value="20" />
  <label>Concurrency</label>
  <input id="concurrency" type="number" min="1" value="4" />
  <div class="actions">
    <button id="loadBtn">Load dataset</button>
    <button class="warn" id="smokeBtn">Run smoke</button>
    <button class="secondary" id="foBtn">Open FiftyOne</button>
  </div>
  <h3>Backend</h3>
  <div class="kv" id="serverKv"></div>
</aside>
<section>
  <div class="progress"><div id="bar"></div></div>
  <p id="progressText" class="pill">idle</p>
  <label>System instructions</label>
  <textarea id="systemPrompt"></textarea>
  <label>User prompt</label>
  <textarea id="userPrompt"></textarea>
  <div class="actions">
    <button id="runBtn">Run selected videos</button>
    <button class="secondary" id="allBtn">Select all</button>
    <button class="secondary" id="noneBtn">Select none</button>
  </div>
  <h3>Videos</h3>
  <div class="results"><table><thead><tr><th></th><th>Name</th><th>Source label</th><th>Path</th></tr></thead><tbody id="videoRows"></tbody></table></div>
  <h3>Results</h3>
  <div class="results"><table><thead><tr><th>Video</th><th>Prediction</th><th>Hazard</th><th>Description / error</th></tr></thead><tbody id="resultRows"></tbody></table></div>
  <h3>Runtime log</h3>
  <div class="log" id="log"></div>
</section>
</main>
<script>
let state = null;
function esc(s){ return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function api(path, body){ const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); const j = await r.json(); if(!r.ok) throw new Error(j.error||r.statusText); return j; }
function checkedIds(){ return [...document.querySelectorAll('.pick:checked')].map(x=>x.value); }
function render(){ if(!state) return; document.getElementById('systemPrompt').value ||= state.defaults.system_prompt; document.getElementById('userPrompt').value ||= state.defaults.user_prompt; document.getElementById('repo').value ||= state.dataset_repo; document.getElementById('maxVideos').value ||= state.defaults.max_videos; document.getElementById('concurrency').value ||= state.defaults.concurrency;
 const srv = state.server || {}; document.getElementById('serverPill').textContent = srv.error ? 'backend unavailable' : (srv.model ? 'model: '+srv.model : 'backend ready'); document.getElementById('serverKv').innerHTML = Object.entries(srv).map(([k,v])=>`<div>${esc(k)}</div><div>${esc(Array.isArray(v)?v.join(', '):v)}</div>`).join('');
 const prog = state.progress || {done:0,total:0,errors:0}; const pct = prog.total ? Math.round(100*prog.done/prog.total) : 0; document.getElementById('bar').style.width = pct+'%'; document.getElementById('progressText').textContent = state.running ? `${prog.done}/${prog.total} running, ${prog.errors} errors` : `${prog.done}/${prog.total} complete, ${prog.errors} errors`;
 const selected = new Set(checkedIds()); document.getElementById('videoRows').innerHTML = (state.videos||[]).map(v=>`<tr><td><input class="pick" type="checkbox" value="${esc(v.id)}" ${selected.size===0||selected.has(v.id)?'checked':''}></td><td>${esc(v.name)}</td><td>${esc(v.label||'')}</td><td>${esc(v.filepath)}</td></tr>`).join('');
 document.getElementById('resultRows').innerHTML = (state.results||[]).map(r=>{ const j=r.json||{}; const hz=j.hazard_detection||{}; const pred=j.prediction_label ? `${esc(j.prediction_class_id)} ${esc(j.prediction_label)}` : ''; const desc=r.error ? `<span style="color:var(--bad)">${esc(r.error)}</span>` : esc(j.video_description||r.response||''); return `<tr><td>${esc(r.name)}</td><td>${pred}</td><td>${esc(hz.is_hazardous)}</td><td>${desc}</td></tr>`; }).join('');
 document.getElementById('log').textContent = (state.logs||[]).join('\n');
 document.getElementById('loadBtn').disabled = state.running; document.getElementById('runBtn').disabled = state.running; document.getElementById('smokeBtn').disabled = state.running; }
async function poll(){ const r = await fetch('/api/state'); state = await r.json(); render(); }
document.getElementById('loadBtn').onclick = async()=>{ try{ await api('/api/load',{repo_id:repo.value,max_videos:Number(maxVideos.value)}); await poll(); }catch(e){ alert(e.message); } };
document.getElementById('runBtn').onclick = async()=>{ try{ await api('/api/run',{ids:checkedIds(),concurrency:Number(concurrency.value),system_prompt:systemPrompt.value,user_prompt:userPrompt.value}); await poll(); }catch(e){ alert(e.message); } };
document.getElementById('smokeBtn').onclick = async()=>{ try{ await api('/api/smoke',{max_videos:Number(maxVideos.value),concurrency:Number(concurrency.value)}); await poll(); }catch(e){ alert(e.message); } };
document.getElementById('foBtn').onclick = async()=>{ try{ const j=await api('/api/fiftyone',{}); alert('FiftyOne: '+j.url); }catch(e){ alert(e.message); } };
document.getElementById('allBtn').onclick = ()=>{ document.querySelectorAll('.pick').forEach(x=>x.checked=true); };
document.getElementById('noneBtn').onclick = ()=>{ document.querySelectorAll('.pick').forEach(x=>x.checked=false); };
poll(); setInterval(poll, 1500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self.send_text(INDEX_HTML, "text/html")
        elif self.path == "/api/state":
            detect_server()
            self.send_json(snapshot())
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self.read_json()
            if self.path == "/api/load":
                videos = load_dataset(str(payload.get("repo_id") or DEFAULT_DATASET), int(payload.get("max_videos") or 20))
                self.send_json({"videos": videos})
            elif self.path == "/api/run":
                if snapshot().get("running"):
                    raise RuntimeError("A batch is already running")
                thread = threading.Thread(target=run_batch, args=(payload.get("ids") or [], int(payload.get("concurrency") or 4), str(payload.get("system_prompt") or WORKER_SAFETY_SYSTEM), str(payload.get("user_prompt") or WORKER_SAFETY_USER), float(payload.get("fps") or 1), int(payload.get("max_tokens") or 1024)), daemon=True)
                thread.start()
                self.send_json({"ok": True})
            elif self.path == "/api/smoke":
                if snapshot().get("running"):
                    raise RuntimeError("A batch is already running")
                max_videos = int(payload.get("max_videos") or 2)
                concurrency = int(payload.get("concurrency") or 2)
                load_dataset(DEFAULT_DATASET, max_videos)
                ids = [v["id"] for v in snapshot()["videos"]]
                thread = threading.Thread(target=run_batch, args=(ids, concurrency, WORKER_SAFETY_SYSTEM, WORKER_SAFETY_USER, 1.0, 1024), daemon=True)
                thread.start()
                self.send_json({"ok": True, "dataset": DEFAULT_DATASET, "videos": len(ids)})
            elif self.path == "/api/fiftyone":
                url = launch_fiftyone(int(payload.get("port") or os.getenv("FIFTYONE_PORT", "5151")))
                self.send_json({"url": url})
            else:
                self.send_error(404)
        except Exception as exc:
            log(f"API error: {exc}")
            self.send_json({"error": str(exc)}, status=500)

    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def serve(host: str, port: int) -> None:
    detect_server()
    url = f"http://{host}:{port}/"
    Path(os.getenv("RUNTIME_AGENT_URL_FILE", "/tmp/byo_video_runtime_agent_url.txt")).write_text(url, encoding="utf-8")
    Path("/tmp/gradio_url.txt").write_text(url, encoding="utf-8")
    log(f"URL: {url}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def smoke(args: argparse.Namespace) -> int:
    load_dataset(args.dataset, args.max_videos)
    ids = [v["id"] for v in snapshot()["videos"]]
    run_batch(ids, args.concurrency, WORKER_SAFETY_SYSTEM, WORKER_SAFETY_USER, args.fps, args.max_tokens)
    snap = snapshot()
    print(json.dumps({"progress": snap["progress"], "results_file": str(RESULTS_FILE)}, indent=2))
    return 1 if snap["progress"].get("errors") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Cosmos BYO-video runtime-agent frontend")
    sub = parser.add_subparsers(dest="cmd")
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--host", default=os.getenv("RUNTIME_AGENT_HOST", "0.0.0.0"))
    serve_p.add_argument("--port", type=int, default=int(os.getenv("RUNTIME_AGENT_PORT", "7861")))
    smoke_p = sub.add_parser("smoke")
    smoke_p.add_argument("--dataset", default=DEFAULT_DATASET)
    smoke_p.add_argument("--max-videos", type=int, default=int(os.getenv("RUNTIME_AGENT_SMOKE_VIDEOS", "2")))
    smoke_p.add_argument("--concurrency", type=int, default=int(os.getenv("RUNTIME_AGENT_CONCURRENCY", "2")))
    smoke_p.add_argument("--fps", type=float, default=float(os.getenv("RUNTIME_AGENT_FPS", "1")))
    smoke_p.add_argument("--max-tokens", type=int, default=int(os.getenv("RUNTIME_AGENT_MAX_TOKENS", "1024")))
    args = parser.parse_args()
    if args.cmd == "smoke":
        return smoke(args)
    serve(args.host if hasattr(args, "host") else "0.0.0.0", args.port if hasattr(args, "port") else 7861)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
