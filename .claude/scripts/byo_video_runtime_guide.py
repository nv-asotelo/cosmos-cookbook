#!/usr/bin/env python3
"""Friendly CLI companion for the BYO-video batch-inference.

The HTML batch-inference remains the expert UI. This guide is intentionally a
small, dependency-free wrapper around the same HTTP API so Claude Code can walk
casual users through dataset loading, paper import, guarded batch runs, and
exports without bypassing the batch-inference's safety checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_URL_FILES = [
    Path("/tmp/byo_video_batch_inference_url.txt"),
    Path("/tmp/gradio_url.txt"),
]

STRUCTURED_TEMPLATES = {
    "classification": {
        "label": "Classification JSON",
        "system": "You are a careful video evaluator. Return only valid JSON.",
        "user": """Analyze the video and return exactly one JSON object:
{
  "class_id": "<short class id or null>",
  "class_label": "<human-readable class label>",
  "evidence": "<concise visual evidence>",
  "confidence": <number from 0 to 1>,
  "needs_review": <true if the video is ambiguous, otherwise false>
}""",
    },
    "safety": {
        "label": "Safety inspection JSON",
        "system": "You are an expert safety inspector. Return only valid JSON.",
        "user": """Inspect the video for safety-relevant behavior and return exactly one JSON object:
{
  "summary": "<one sentence>",
  "hazards": [{"type": "<hazard type>", "evidence": "<visual evidence>", "severity": "low|medium|high"}],
  "safe_behaviors": ["<observed safe behavior>"],
  "recommendation": "<what a reviewer should do next>"
}""",
    },
    "evaluation": {
        "label": "Dataset evaluation JSON",
        "system": "You compare model observations against expected dataset answers. Return only valid JSON.",
        "user": """Answer the dataset question shown in the prompt or metadata. Return exactly one JSON object:
{
  "answer": "<direct answer>",
  "evidence": "<concise evidence from the video>",
  "uncertainty": "<what is unclear, or null>"
}""",
    },
    "summary": {
        "label": "Plain summary",
        "system": "You are a helpful assistant that explains videos clearly.",
        "user": "Summarize the video in plain language. Mention the main people, objects, actions, and anything uncertain.",
    },
}


class GuideError(RuntimeError):
    pass


def read_default_url() -> str:
    for path in DEFAULT_URL_FILES:
        if not path.exists():
            continue
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "http://localhost:7861"


def normalize_url(url: str) -> str:
    url = (url or read_default_url()).strip()
    if not url:
        url = "http://localhost:7861"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url.rstrip("/")


def request_json(url: str, path: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Dict[str, Any]:
    full_url = normalize_url(url) + path
    if payload is None:
        req = urllib.request.Request(full_url, headers={"Accept": "application/json"})
    else:
        req = urllib.request.Request(
            full_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("error") or body
        except Exception:
            detail = body
        raise GuideError(f"{exc.code} from {path}: {str(detail).strip()}")
    except Exception as exc:
        raise GuideError(f"Could not reach batch-inference at {full_url}: {exc}")


def state(url: str) -> Dict[str, Any]:
    return request_json(url, "/api/state", timeout=20)


def clip(text: Any, limit: int = 120) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "..."


def sec(value: Any) -> str:
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return ""


def default_params(snap: Dict[str, Any]) -> Dict[str, Any]:
    d = snap.get("defaults") or {}
    return {
        "fps": d.get("fps", 2),
        "max_pixels": d.get("max_pixels", 524288),
        "max_tokens": d.get("max_tokens", 512),
        "temperature": d.get("temperature", 0),
        "top_p": d.get("top_p", 1),
        "repetition_penalty": d.get("repetition_penalty", 1.05),
        "max_frames": d.get("max_frames", 8),
    }


def prompt_for_index(snap: Dict[str, Any], index: Optional[int]) -> Dict[str, Any]:
    prompts = (snap.get("defaults") or {}).get("prompt_presets") or []
    if index is None:
        paper_index = next((i for i, item in enumerate(prompts) if item.get("paper_import")), None)
        index = paper_index if paper_index is not None else 0
    if not prompts:
        return {
            "label": "Custom prompt",
            "system_prompt": (snap.get("defaults") or {}).get("system_prompt") or "You are a helpful assistant.",
            "user_prompt": (snap.get("defaults") or {}).get("user_prompt") or "Describe this video.",
        }
    index = max(0, min(index, len(prompts) - 1))
    item = prompts[index]
    return {
        "label": item.get("label") or f"Prompt {index}",
        "system_prompt": item.get("system_prompt") or "You are a helpful assistant.",
        "user_prompt": item.get("user_prompt") or "Describe this video.",
    }


def selected_ids(snap: Dict[str, Any], first: Optional[int], all_videos: bool) -> List[str]:
    videos = snap.get("videos") or []
    if not videos:
        raise GuideError("No videos are loaded. Load a dataset or import a paper first.")
    chosen = videos if all_videos or first is None else videos[: max(1, first)]
    return [video["id"] for video in chosen]


def print_status(snap: Dict[str, Any]) -> None:
    server = snap.get("server") or {}
    progress = snap.get("progress") or {}
    batch = snap.get("batch_metrics") or {}
    print("\nBYO-video runtime")
    print(f"  backend: {server.get('backend') or ''} | model: {server.get('model') or ''}")
    print(f"  dataset: {snap.get('dataset_repo') or ''} | videos loaded: {len(snap.get('videos') or [])}")
    if snap.get("paper_import"):
        paper = snap["paper_import"]
        print(f"  paper: {paper.get('arxiv_id') or paper.get('source')} | prompts: {len(paper.get('prompt_presets') or [])}")
    print(f"  run: {'running' if snap.get('running') else 'idle'} | {progress.get('done', 0)}/{progress.get('total', 0)} done | errors={progress.get('errors', 0)}")
    if progress.get("last_event"):
        print(f"  event: {progress.get('last_event')}")
    if progress.get("last_error"):
        print(f"  last error: {clip(progress.get('last_error'), 180)}")
    if batch:
        evaluation = batch.get("evaluation") or {}
        accuracy = evaluation.get("accuracy")
        accuracy_text = "" if accuracy is None else f"{accuracy * 100:.1f}%"
        print(f"  last batch: {batch.get('completed')}/{batch.get('total')} | e2e={sec(batch.get('batch_wall_seconds'))} | accuracy={accuracy_text}")


def print_prompts(snap: Dict[str, Any]) -> None:
    prompts = (snap.get("defaults") or {}).get("prompt_presets") or []
    for index, item in enumerate(prompts):
        suffix = " paper" if item.get("paper_import") else ""
        print(f"{index:2d}. {item.get('label')}{suffix}")


def print_videos(snap: Dict[str, Any], limit: int = 20) -> None:
    for index, video in enumerate((snap.get("videos") or [])[:limit], 1):
        meta = video.get("meta") or {}
        expected = video.get("expected") or {}
        label = expected.get("label") or video.get("label") or ""
        print(f"{index:2d}. {video.get('name')} | {meta.get('width')}x{meta.get('height')} | {sec(meta.get('duration_s'))} | {label}")
    total = len(snap.get("videos") or [])
    if total > limit:
        print(f"    ... {total - limit} more")


def print_results(snap: Dict[str, Any], limit: int = 20) -> None:
    results = snap.get("results") or []
    if not results:
        print("No results yet.")
        return
    print("\nResults")
    for result in results[:limit]:
        parsed = result.get("json") or {}
        evaluation = result.get("evaluation") or {}
        metrics = result.get("metrics") or {}
        pred = parsed.get("prediction_label") or parsed.get("answer") or clip(result.get("response"), 80)
        match = ""
        if evaluation.get("has_expected"):
            match = "correct" if evaluation.get("is_correct") else "miss"
        if result.get("error"):
            pred = "ERROR: " + clip(result.get("error"), 120)
        print(f"- {result.get('name')}: {clip(pred, 110)} | {match} | e2e={sec(metrics.get('e2e_seconds'))}")


def default_export_sections(snap: Dict[str, Any]) -> List[str]:
    sections = (snap.get("defaults") or {}).get("export_sections") or []
    return [section["id"] for section in sections if section.get("default") and section.get("id")]


def choose_export_sections(snap: Dict[str, Any]) -> List[str]:
    sections = (snap.get("defaults") or {}).get("export_sections") or []
    if not sections:
        return []
    print("\nReport sections")
    for index, section in enumerate(sections, 1):
        marker = "default" if section.get("default") else "optional"
        print(f"  {index}. {section.get('label')} ({section.get('id')}, {marker})")
    raw = ask("Section ids or numbers, comma-separated (blank = defaults)", "")
    if not raw:
        return default_export_sections(snap)
    chosen: List[str] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if item.isdigit():
            idx = int(item)
            if 1 <= idx <= len(sections):
                chosen.append(str(sections[idx - 1].get("id")))
            continue
        chosen.append(item)
    return chosen


def poll_until_done(url: str, interval: int = 2) -> Dict[str, Any]:
    while True:
        snap = state(url)
        progress = snap.get("progress") or {}
        total = progress.get("total") or 0
        done = progress.get("done") or 0
        errors = progress.get("errors") or 0
        started = progress.get("started_epoch") or time.time()
        elapsed = max(0, time.time() - started)
        eta = ""
        if snap.get("running") and done and total and done < total:
            eta_s = (total - done) / max(done / max(elapsed, 0.1), 0.001)
            eta = f" | eta~{int(eta_s)}s"
        print(f"\r{done}/{total} done | errors={errors} | elapsed={int(elapsed)}s{eta}", end="", flush=True)
        if not snap.get("running"):
            print()
            return snap
        time.sleep(interval)


def run_batch(
    url: str,
    first: Optional[int],
    all_videos: bool,
    concurrency: int,
    prompt_index: Optional[int],
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    wait: bool = True,
) -> Dict[str, Any]:
    snap = state(url)
    prompt = prompt_for_index(snap, prompt_index)
    if system_prompt:
        prompt["system_prompt"] = system_prompt
    if user_prompt:
        prompt["user_prompt"] = user_prompt
    payload = {
        "ids": selected_ids(snap, first, all_videos),
        "concurrency": concurrency,
        "prompt_label": prompt["label"],
        "system_prompt": prompt["system_prompt"],
        "user_prompt": prompt["user_prompt"],
        "allow_over_context": False,
        **default_params(snap),
    }
    print(f"Starting guarded run: {len(payload['ids'])} video(s), concurrency={concurrency}, prompt={prompt['label']}")
    reply = request_json(url, "/api/run", payload, timeout=30)
    budget = reply.get("context_budget") or {}
    if budget.get("message"):
        print("Context guard:", budget["message"])
    if wait:
        final = poll_until_done(url)
        print_results(final)
        return final
    return state(url)


def export_artifact(url: str, fmt: str, sections: Optional[List[str]] = None) -> Dict[str, Any]:
    payload = {"format": fmt, "sections": sections or []}
    reply = request_json(url, "/api/export", payload, timeout=60)
    print(f"Exported {reply.get('format', fmt).upper()}: {normalize_url(url)}{reply.get('url')}")
    return reply


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter a whole number.")


def ask_choice(prompt: str, choices: List[str], default: int = 1) -> int:
    print("\n" + prompt)
    for index, choice in enumerate(choices, 1):
        print(f"  {index}. {choice}")
    while True:
        raw = ask("Choose", str(default))
        try:
            index = int(raw)
            if 1 <= index <= len(choices):
                return index
        except ValueError:
            pass
        print("Choose one of the numbered options.")


def choose_prompt(snap: Dict[str, Any]) -> Optional[int]:
    prompts = (snap.get("defaults") or {}).get("prompt_presets") or []
    if not prompts:
        return None
    print("\nPrompt presets")
    print_prompts(snap)
    return ask_int("Prompt number", 0)


def shape_prompt() -> Dict[str, str]:
    keys = list(STRUCTURED_TEMPLATES)
    labels = [STRUCTURED_TEMPLATES[key]["label"] for key in keys]
    labels.append("Custom fields")
    choice = ask_choice("What kind of output do you want?", labels, 1)
    if choice <= len(keys):
        item = STRUCTURED_TEMPLATES[keys[choice - 1]]
        return {"system": item["system"], "user": item["user"], "label": item["label"]}
    fields = ask("Comma-separated JSON fields", "answer,evidence,confidence")
    user = "Analyze the video and return exactly one JSON object with these fields: " + fields
    return {"system": "You are a careful video analyst. Return only valid JSON.", "user": user, "label": "Custom structured JSON"}


def gradio_guide(url: str) -> None:
    print("\nThis looks like a Gradio URL rather than the batch-inference API.")
    print("Casual-user path:")
    print(f"  1. Open {url}")
    print("  2. Upload one MP4.")
    print("  3. Start with the General description or Safety analysis prompt.")
    print("  4. Keep build.nvidia.com defaults on for NIM-local; otherwise use deterministic settings for evaluation.")
    print("  5. If the browser shows a generic error, ask Claude Code to run the runtime monitor:")
    print("     python3 ~/.claude/scripts/byo_video_runtime_monitor.py alerts -v")


def wizard(url: str) -> None:
    url = normalize_url(url)
    try:
        snap = state(url)
    except GuideError:
        gradio_guide(url)
        return
    print("\nThis companion uses the same batch-inference API as the browser UI.")
    print("It keeps the context-budget guard on by default and shows progress, errors, ETA, and exports from the current session.")
    print_status(snap)
    while True:
        choice = ask_choice(
            "What would you like Claude Code to help with?",
            [
                "Quick worker-safety smoke test",
                "Import a paper/arXiv/HF page and load its dataset",
                "Load a Hugging Face dataset",
                "Run loaded videos",
                "Shape a structured output prompt and run",
                "Export results",
                "Show status/results",
                "Exit",
            ],
            4,
        )
        try:
            if choice == 1:
                payload = {
                    "max_videos": ask_int("Smoke videos", 2),
                    "concurrency": ask_int("Concurrency", 2),
                    "allow_over_context": False,
                    **default_params(state(url)),
                }
                request_json(url, "/api/smoke", payload, timeout=30)
                print_results(poll_until_done(url))
            elif choice == 2:
                source = ask("Paper/arXiv/HF URL", "https://huggingface.co/papers/2603.29281")
                max_videos = ask_int("Max videos to load (0 = all)", 2)
                reply = request_json(url, "/api/paper", {"source": source, "max_videos": max_videos, "load_dataset": True}, timeout=180)
                print(f"Dataset: {reply.get('selected_dataset')} | prompts: {len(reply.get('prompt_presets') or [])} | loaded: {reply.get('loaded_videos', 0)}")
                for warning in reply.get("warnings") or []:
                    print("warning:", warning)
            elif choice == 3:
                repo = ask("HF dataset repo", "pjramg/Safe_Unsafe_Test")
                max_videos = ask_int("Max videos to load (0 = all)", 20)
                reply = request_json(url, "/api/load", {"repo_id": repo, "max_videos": max_videos}, timeout=180)
                print(f"Loaded {len(reply.get('videos') or [])} videos.")
            elif choice == 4:
                snap = state(url)
                print_videos(snap)
                prompt_index = choose_prompt(snap)
                first = ask_int("How many videos? (0 = all loaded)", min(2, len(snap.get("videos") or [])) or 1)
                run_batch(url, None if first == 0 else first, first == 0, ask_int("Concurrency", 2), prompt_index)
            elif choice == 5:
                template = shape_prompt()
                print("\nSystem prompt:\n" + template["system"])
                print("\nUser prompt:\n" + template["user"])
                if ask_choice("Run with this prompt now?", ["Yes", "No"], 1) == 1:
                    first = ask_int("How many videos? (0 = all loaded)", 1)
                    run_batch(url, None if first == 0 else first, first == 0, ask_int("Concurrency", 1), None, template["system"], template["user"])
            elif choice == 6:
                fmt = ["html", "json", "csv", "xlsx", "pptx"][ask_choice("Export format", ["HTML report", "JSON", "CSV", "Excel", "PowerPoint"], 1) - 1]
                sections = choose_export_sections(state(url)) if fmt in ("html", "pptx") else []
                export_artifact(url, fmt, sections)
            elif choice == 7:
                snap = state(url)
                print_status(snap)
                print_videos(snap, limit=10)
                print_results(snap)
            else:
                return
        except GuideError as exc:
            print(f"\nProblem: {exc}")
            print("The batch-inference kept the run from proceeding or surfaced the backend error. Adjust settings, load fewer videos, or ask Claude to inspect the logs.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Friendly CLI guide for the BYO-video batch-inference")
    parser.add_argument("--url", default="", help="Runtime-agent URL, default reads /tmp/byo_video_batch_inference_url.txt")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("wizard", help="Interactive guided flow")
    sub.add_parser("status", help="Show backend, dataset, prompts, progress, and results")
    sub.add_parser("prompts", help="List prompt presets")

    load_p = sub.add_parser("load", help="Load a Hugging Face dataset")
    load_p.add_argument("repo")
    load_p.add_argument("--max-videos", type=int, default=20)

    paper_p = sub.add_parser("paper", help="Import paper/HF metadata and load linked dataset")
    paper_p.add_argument("source")
    paper_p.add_argument("--max-videos", type=int, default=2)
    paper_p.add_argument("--no-load", action="store_true")

    run_p = sub.add_parser("run", help="Run loaded videos through the guarded batch-inference API")
    run_p.add_argument("--first", type=int, default=1)
    run_p.add_argument("--all", action="store_true")
    run_p.add_argument("--concurrency", type=int, default=1)
    run_p.add_argument("--prompt-index", type=int)
    run_p.add_argument("--no-wait", action="store_true")

    export_p = sub.add_parser("export", help="Export current results")
    export_p.add_argument("--format", choices=["html", "json", "csv", "xlsx", "pptx"], default="html")
    export_p.add_argument("--sections", default="", help="Comma-separated report section ids for HTML/PPTX exports")

    sub.add_parser("shape-prompt", help="Build a structured-output prompt template")

    args = parser.parse_args(argv)
    url = normalize_url(args.url)
    try:
        if args.cmd in (None, "wizard"):
            wizard(url)
        elif args.cmd == "status":
            snap = state(url)
            print_status(snap)
            print_videos(snap, limit=10)
            print_results(snap)
        elif args.cmd == "prompts":
            print_prompts(state(url))
        elif args.cmd == "load":
            reply = request_json(url, "/api/load", {"repo_id": args.repo, "max_videos": args.max_videos}, timeout=180)
            print(f"Loaded {len(reply.get('videos') or [])} videos from {args.repo}.")
        elif args.cmd == "paper":
            reply = request_json(url, "/api/paper", {"source": args.source, "max_videos": args.max_videos, "load_dataset": not args.no_load}, timeout=180)
            print(json.dumps(reply, indent=2, default=str))
        elif args.cmd == "run":
            run_batch(url, None if args.first == 0 else args.first, args.all or args.first == 0, args.concurrency, args.prompt_index, wait=not args.no_wait)
        elif args.cmd == "export":
            sections = [part.strip() for part in args.sections.split(",") if part.strip()]
            export_artifact(url, args.format, sections)
        elif args.cmd == "shape-prompt":
            item = shape_prompt()
            print("\nSystem prompt:\n" + item["system"])
            print("\nUser prompt:\n" + item["user"])
        return 0
    except GuideError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCanceled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
