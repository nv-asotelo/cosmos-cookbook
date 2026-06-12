#!/usr/bin/env python3
"""Audit inference_kpi micro-benchmark sources without running inference.

The first gate produces a manifest, endpoint status, storage plan, static
report, and a small PowerPoint deck. It avoids extracting large media archives
or calling model/autograder inference endpoints.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import textwrap
import time
from typing import Any
import urllib.error
import urllib.request


DEFAULT_SOURCE_TAR = Path("/Users/asotelo/Downloads/batch_1.tar.gz")
DEFAULT_OUT_DIR = Path("outputs/inference-kpi-audit")
DEFAULT_GIT_URL = "https://gitlab-master.nvidia.com/av-stargate/stargate.git"
DEFAULT_GIT_PATH = "resources/data/inference_kpi"
SUPER_MODELS_URL = "http://10.57.233.243:8000/v1/models"
NANO_MODELS_URL = "http://10.57.232.110:8000/v1/models"
AUTOGRADER_HOST = "10.63.158.169"
REPORT_URL = "http://10.57.233.243:9105/"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def stable_id(*parts: Any) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


CLIP_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:_([0-9]+))?", re.I)
PRESET_RE = re.compile(r"_preset([0-9]+)", re.I)


def parse_clip_id(name: str) -> tuple[str, str]:
    match = CLIP_RE.search(name)
    if not match:
        stem = Path(name).stem
        return stem, stem
    uuid, timestamp = match.groups()
    return (f"{uuid}_{timestamp}" if timestamp else uuid), uuid


def media_type_for(path: str, is_dir: bool) -> str:
    if is_dir:
        return "directory"
    suffix = Path(path).suffix.lower()
    return {
        ".mp4": "video",
        ".mov": "video",
        ".webm": "video",
        ".json": "json",
        ".tar": "tar_bundle",
        ".zip": "zip_bundle",
        ".gz": "compressed_archive",
    }.get(suffix, suffix.lstrip(".") or "file")


def route_for(source_dataset: str, kpi_group: str, media_type: str, name: str) -> str:
    if media_type == "video":
        return "vlm_preset"
    if source_dataset == "rgb" and media_type == "json":
        return "vlm_preset_trace"
    if source_dataset == "clipGT":
        return "ground_truth_bundle"
    if source_dataset.startswith("wm_"):
        return "world_model_inventory"
    if source_dataset.startswith("rds_hq"):
        return "road_scene_inventory"
    return "inventory_only"


def parse_member(member_name: str, size: int, is_dir: bool, source_tar: Path) -> dict[str, Any]:
    parts = member_name.strip("/").split("/")
    batch_id = parts[0] if parts else ""
    if len(parts) == 1:
        source_dataset = "batch_root"
        kpi_group = "root"
    elif len(parts) == 2 and not is_dir:
        source_dataset = "batch_root"
        kpi_group = Path(parts[1]).stem
    else:
        source_dataset = parts[1] if len(parts) > 1 else "batch_root"
        kpi_group = source_dataset

    if source_dataset in {"rds_hq", "rds_hq_flat", "wm_v1", "wm_v1_flat", "wm_v3", "wm_v3_flat"}:
        kpi_group = parts[2] if len(parts) > 2 else source_dataset
    file_name = parts[-1] if parts else ""
    clip_id, clip_uuid = parse_clip_id(file_name)
    media_type = media_type_for(member_name, is_dir)
    preset_match = PRESET_RE.search(file_name)
    question_or_preset_id = f"preset{preset_match.group(1)}" if preset_match else ""
    route = route_for(source_dataset, kpi_group, media_type, file_name)
    media_variant = ""
    if "4.034_sec_30_fps" in file_name:
        media_variant = "4.034_sec_30_fps"
    elif "9.3_sec_10_fps" in file_name:
        media_variant = "9.3_sec_10_fps"
    elif media_type in {"json", "tar_bundle", "zip_bundle"}:
        media_variant = media_type
    row = {
        "stable_key": stable_id(batch_id, source_dataset, kpi_group, clip_id, file_name, question_or_preset_id),
        "source_dataset": source_dataset,
        "batch_id": batch_id,
        "kpi_group": kpi_group,
        "clip_id": clip_id,
        "clip_uuid": clip_uuid,
        "asset_id": Path(file_name).stem if file_name else member_name,
        "media_variant": media_variant,
        "question_or_preset_id": question_or_preset_id,
        "media_type": media_type,
        "expected_autograder_route": route,
        "source_uri": f"local-tar://{source_tar}::{member_name}",
        "local_archive": str(source_tar),
        "local_archive_member_path": member_name,
        "size_bytes": size,
        "requires_selective_download": bool(not is_dir and media_type in {"video", "tar_bundle", "zip_bundle"}),
        "is_directory": is_dir,
    }
    return row


def try_sparse_fetch(out_dir: Path, git_url: str, git_ref: str, git_path: str, timeout_s: int) -> dict[str, Any]:
    status: dict[str, Any] = {
        "attempted": True,
        "git_url": git_url,
        "git_ref": git_ref,
        "git_path": git_path,
        "ok": False,
        "files": [],
        "error": None,
    }
    with tempfile.TemporaryDirectory(prefix="inference-kpi-sparse-") as tmp:
        work = Path(tmp) / "stargate"
        commands = [
            ["git", "clone", "--filter=blob:none", "--sparse", "--depth=1", "--branch", git_ref, git_url, str(work)],
            ["git", "-C", str(work), "sparse-checkout", "set", git_path],
        ]
        try:
            for cmd in commands:
                subprocess.run(
                    cmd,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_s,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                )
            root = work / git_path
            status["files"] = [str(p.relative_to(work)) for p in root.rglob("*") if p.is_file()]
            status["ok"] = True
        except Exception as exc:  # noqa: BLE001 - record failure and fall back.
            status["error"] = str(exc)
    write_json(out_dir / "remote_source_status.json", status)
    return status


def command_audit(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    source_tar = Path(args.source_tar)
    remote_status = None
    if not args.skip_git:
        remote_status = try_sparse_fetch(out_dir, args.git_url, args.git_ref, args.git_path, args.remote_timeout)
    else:
        remote_status = {"attempted": False, "ok": False, "reason": "skip_git requested"}
        write_json(out_dir / "remote_source_status.json", remote_status)

    if not source_tar.exists():
        raise SystemExit(f"Source tar not found: {source_tar}")

    rows: list[dict[str, Any]] = []
    inventory_started = time.time()
    with tarfile.open(source_tar, mode="r:gz") as tf:
        for member in tf:
            rows.append(parse_member(member.name, member.size, member.isdir(), source_tar))

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    write_manifest_sqlite(out_dir / "manifest.sqlite", rows)
    summary = summarize_rows(rows)
    metadata = {
        "generated_at": now_iso(),
        "mode": "data_audit_only",
        "source_tar": str(source_tar),
        "source_tar_size_bytes": source_tar.stat().st_size,
        "manifest_rows": len(rows),
        "archive_entries": len(rows),
        "inventory_seconds": round(time.time() - inventory_started, 3),
        "remote_source_status": remote_status,
        "summary": summary,
        "secrets_recorded": False,
    }
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "storage_plan.json", build_storage_plan(rows, remote_status))
    print(json.dumps({"manifest_rows": len(rows), "out_dir": str(out_dir), "source_tar": str(source_tar)}, indent=2))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            out[row.get(key) or ""] = out.get(row.get(key) or "", 0) + 1
        return dict(sorted(out.items(), key=lambda item: (-item[1], item[0])))

    video_rows = [r for r in rows if r["media_type"] == "video"]
    return {
        "by_source_dataset": counts("source_dataset"),
        "by_kpi_group": counts("kpi_group"),
        "by_media_type": counts("media_type"),
        "by_autograder_route": counts("expected_autograder_route"),
        "video_rows": len(video_rows),
        "unique_clips": len({r["clip_id"] for r in rows if r["clip_id"]}),
        "selective_download_rows": sum(1 for r in rows if r["requires_selective_download"]),
    }


def write_manifest_sqlite(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        columns = [
            "stable_key",
            "source_dataset",
            "batch_id",
            "kpi_group",
            "clip_id",
            "clip_uuid",
            "asset_id",
            "media_variant",
            "question_or_preset_id",
            "media_type",
            "expected_autograder_route",
            "source_uri",
            "local_archive",
            "local_archive_member_path",
            "size_bytes",
            "requires_selective_download",
            "is_directory",
        ]
        conn.execute(
            "CREATE TABLE manifest (" + ", ".join(f"{col} TEXT" for col in columns) + ")"
        )
        conn.executemany(
            "INSERT INTO manifest VALUES (" + ", ".join("?" for _ in columns) + ")",
            [[str(row.get(col, "")) for col in columns] for row in rows],
        )
        conn.execute("CREATE INDEX idx_manifest_clip ON manifest(clip_id)")
        conn.execute("CREATE INDEX idx_manifest_group ON manifest(source_dataset, kpi_group)")
        conn.execute("CREATE INDEX idx_manifest_route ON manifest(expected_autograder_route)")
        conn.commit()
    finally:
        conn.close()


def build_storage_plan(rows: list[dict[str, Any]], remote_status: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "policy": "remote_first_manifest_selective_download",
        "bulk_copy_allowed": False,
        "remote_source_preferred": True,
        "remote_source_status": remote_status or {},
        "local_archive_fallback": {
            "available": bool(rows),
            "strategy": "index tar members now; extract only selected drill-down assets in later gates",
        },
        "selective_download": {
            "candidate_rows": sum(1 for r in rows if r["requires_selective_download"]),
            "video_rows": sum(1 for r in rows if r["media_type"] == "video"),
            "bundle_rows": sum(1 for r in rows if r["media_type"] in {"tar_bundle", "zip_bundle"}),
        },
        "horde_target": "/home/horde/inference-kpi-benchmark",
        "report_url": REPORT_URL,
    }


def http_json(url: str, timeout_s: int = 8) -> dict[str, Any]:
    started = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read(65536)
            elapsed = round(time.time() - started, 3)
            text = body.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"raw": text[:4000]}
            return {
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "elapsed_seconds": elapsed,
                "url": url,
                "response": parsed,
            }
    except Exception as exc:  # noqa: BLE001 - endpoint probe artifact.
        return {
            "ok": False,
            "status": None,
            "elapsed_seconds": round(time.time() - started, 3),
            "url": url,
            "error": str(exc),
        }


def command_probe(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    probes: dict[str, Any] = {
        "generated_at": now_iso(),
        "mode": "probe_only_no_inference",
        "credential_presence": {
            "BUILD_NVIDIA_API_KEY": bool(os.environ.get("BUILD_NVIDIA_API_KEY")),
            "GITLAB_TOKEN": bool(os.environ.get("GITLAB_TOKEN")),
            "NGC_API_KEY": bool(os.environ.get("NGC_API_KEY")),
        },
        "endpoints": {},
        "autograder_supported_routes": {
            "first_gate": "vlm_preset",
            "future_routes": ["obstacle_correspondence", "hallucination", "attribute_verification"],
        },
    }
    endpoint_urls = {
        "cosmos3_super_models": SUPER_MODELS_URL,
        "cosmos3_nano_models": NANO_MODELS_URL,
        "autograder_nim_models": f"http://{AUTOGRADER_HOST}:8000/v1/models",
        "autograder_obstacle_health": f"http://{AUTOGRADER_HOST}:8082/health",
        "autograder_obstacle_config": f"http://{AUTOGRADER_HOST}:8082/config",
        "autograder_vlm_preset_health": f"http://{AUTOGRADER_HOST}:8083/health",
        "autograder_vlm_preset_config": f"http://{AUTOGRADER_HOST}:8083/config",
        "autograder_vlm_runtime": f"http://{AUTOGRADER_HOST}:8090/runtime/vlm",
        "autograder_vlm_endpoints": f"http://{AUTOGRADER_HOST}:8090/runtime/vlm/endpoints",
        "autograder_hallucination_health": f"http://{AUTOGRADER_HOST}:8085/health",
        "autograder_attribute_health": f"http://{AUTOGRADER_HOST}:8086/health",
        "autograder_attribute_config": f"http://{AUTOGRADER_HOST}:8086/config",
    }
    for name, url in endpoint_urls.items():
        probes["endpoints"][name] = http_json(url, timeout_s=args.timeout)
    write_json(out_dir / "endpoint_status.json", probes)
    print(json.dumps({"endpoint_status": str(out_dir / "endpoint_status.json")}, indent=2))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fmt_num(value: int | float | str) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def pct(ok: bool) -> str:
    return "healthy" if ok else "needs attention"


def command_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    artifacts_dir = out_dir / "artifacts"
    ensure_dir(artifacts_dir)
    manifest_path = out_dir / "manifest.jsonl"
    rows = load_manifest(manifest_path)
    metadata = read_json(out_dir / "metadata.json", {})
    endpoint_status = read_json(out_dir / "endpoint_status.json", {})
    storage_plan = read_json(out_dir / "storage_plan.json", {})
    schema_summary = build_schema_summary(rows)
    write_json(out_dir / "schema_summary.json", schema_summary)
    report_html = render_report(rows, metadata, endpoint_status, storage_plan)
    (out_dir / "report.html").write_text(report_html, encoding="utf-8")
    (out_dir / "index.html").write_text(report_html, encoding="utf-8")
    with gzip.open(out_dir / "index.html.gz", "wb") as gz:
        gz.write(report_html.encode("utf-8"))
    write_csv_summary(artifacts_dir / "group_summary.csv", rows)
    pptx_path = build_powerpoint(out_dir, rows, metadata, endpoint_status, storage_plan)
    if pptx_path:
        shutil.copy2(pptx_path, artifacts_dir / pptx_path.name)
    print(json.dumps({"report": str(out_dir / "index.html"), "pptx": str(pptx_path) if pptx_path else None}, indent=2))


def write_csv_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["source_dataset"], row["kpi_group"], row["media_type"])
        counts[key] = counts.get(key, 0) + 1
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_dataset", "kpi_group", "media_type", "rows"])
        for (source_dataset, kpi_group, media_type), count in sorted(counts.items()):
            writer.writerow([source_dataset, kpi_group, media_type, count])


def top_counts(rows: list[dict[str, Any]], key: str, limit: int = 20) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key) or "(blank)"
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def row_count(rows: list[dict[str, Any]], **filters: str) -> int:
    return sum(1 for row in rows if all(row.get(key) == value for key, value in filters.items()))


def build_schema_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kpi_groups = sorted({row["kpi_group"] for row in rows if row["source_dataset"] != "batch_root"})
    prompt_trace_rows = row_count(rows, source_dataset="rgb", media_type="json")
    video_rows = sum(1 for row in rows if row["media_type"] == "video")
    schemas = [
        {
            "family": "RGB VLM traces",
            "source_datasets": ["rgb"],
            "rows": row_count(rows, source_dataset="rgb"),
            "media": "60 videos plus 60 JSON prompt/output traces",
            "observed_fields": [
                "video_name",
                "segments[].text[].text",
                "segments[].environmental_conditions",
                "vlm_instruction_prompt_raw",
                "llm_instruction_prompt_raw",
            ],
            "qa_status": "Closest to plain QA, but still prompt/output traces rather than per-KPI ground-truth QA pairs.",
            "prompt_plan": "Ask every model for a strict JSON scene summary: caption, time_of_day, weather, setting, road_surface, notable_objects, and events.",
            "evaluation_plan": "Use exact checks for environmental condition fields, text-similarity or VLM preset scoring for caption fidelity, and hallucination checks against clipGT/road labels where aligned.",
        },
        {
            "family": "Road-scene 3D labels",
            "source_datasets": ["rds_hq", "rds_hq_flat"],
            "rows": row_count(rows, source_dataset="rds_hq") + row_count(rows, source_dataset="rds_hq_flat"),
            "media": "Nested tar bundles plus sampled camera-frame PNGs",
            "observed_fields": [
                "labels[].labelData.shape3d.surface.vertices",
                "labels[].labelData.shape3d.polyline3d.vertices",
                "labels[].labelData.shape3d.cuboid3d.vertices",
                "traffic_light_states",
                "object_type / object_lwh / object_to_world",
            ],
            "qa_status": "Structured ground truth, not natural-language QA.",
            "prompt_plan": "Generate deterministic questions from labels: presence, count, class/state, and short evidence summary for lanes, signs, traffic lights, objects, road markings, crosswalks, wait lines, poles, and boundaries.",
            "evaluation_plan": "Score structured fields directly. Use geometry only after projecting 3D/world coordinates into the camera frame with pose/intrinsics; otherwise evaluate semantic presence/count/attribute correctness.",
        },
        {
            "family": "ClipGT canonical packages",
            "source_datasets": ["clipGT"],
            "rows": row_count(rows, source_dataset="clipGT"),
            "media": "Zip packages containing clipgt parquet tables and recordings",
            "observed_fields": [
                "object_fused.parquet",
                "cf_traffic_light.parquet",
                "cf_crosswalks.parquet",
                "cf_road_boundary.parquet",
                "dw_lane_line.parquet",
                "recordings/*/camera_front_*.mp4",
            ],
            "qa_status": "Canonical ground truth source, but requires selective extraction and table decoding.",
            "prompt_plan": "Use clipGT to derive clip-time questions and reference answers before model inference.",
            "evaluation_plan": "Use as source of truth for temporal/object/lane/traffic-light checks and as alignment data for autograder routes beyond VLM preset.",
        },
        {
            "family": "World-model and HD-map videos",
            "source_datasets": ["wm_v1", "wm_v1_flat", "wm_v3", "wm_v3_flat"],
            "rows": (
                row_count(rows, source_dataset="wm_v1")
                + row_count(rows, source_dataset="wm_v1_flat")
                + row_count(rows, source_dataset="wm_v3")
                + row_count(rows, source_dataset="wm_v3_flat")
            ),
            "media": "HD-map and world-scenario video variants",
            "observed_fields": ["hdmap video variants", "world_scenario video variants"],
            "qa_status": "Video evidence only in this batch; no direct answer key in the file itself.",
            "prompt_plan": "Ask temporal scene-understanding questions tied to the clip window, then attach reference labels from clipGT/rds where the clip and timestamp align.",
            "evaluation_plan": "Use VLM preset for semantic quality and derived GT checks for objects, map elements, and temporal consistency.",
        },
    ]
    prompt_eval_matrix = [
        {
            "kpi_surface": "Scene caption and environment",
            "source": "rgb JSON + rgb video",
            "prompt": "Return JSON with caption, time_of_day, weather, setting, road_surface, visible objects, and events.",
            "ground_truth": "segments[].environmental_conditions plus curated/previous caption text.",
            "score": "Exact structured-field agreement plus VLM preset semantic score and hallucination review.",
        },
        {
            "kpi_surface": "Road geometry and map elements",
            "source": "rds_hq/rds_hq_flat 3D label bundles",
            "prompt": "Ask for presence/count/location summary of lanes, crosswalks, road boundaries, markings, wait lines, poles, and signs.",
            "ground_truth": "shape3d surfaces, polylines, and cuboids from nested tar JSON.",
            "score": "Direct semantic/count metrics now; pixel/geometry metrics only after camera projection is implemented.",
        },
        {
            "kpi_surface": "Objects and traffic lights",
            "source": "all_object_info, traffic_lights, traffic_lights_status, clipGT parquet",
            "prompt": "Return JSON objects and traffic-light states with confidence and short evidence.",
            "ground_truth": "object type/motion/pose and traffic-light state timelines.",
            "score": "Class/state/count accuracy, temporal alignment, and attribute-verification route where available.",
        },
        {
            "kpi_surface": "World-model / scenario understanding",
            "source": "wm_v1/wm_v3 video variants plus aligned clipGT/rds rows",
            "prompt": "Ask for scenario, planned ego interaction, map constraints, and likely near-term changes.",
            "ground_truth": "Aligned clipGT, hdmap, and road-scene labels.",
            "score": "VLM preset for narrative quality plus structured checks for any derived fields.",
        },
    ]
    return {
        "question_answer_coverage": {
            "plain_qa_pairs_for_every_kpi_group": False,
            "unique_kpi_groups_excluding_batch_root": len(kpi_groups),
            "direct_prompt_trace_groups": ["rgb"] if prompt_trace_rows else [],
            "direct_prompt_trace_rows": prompt_trace_rows,
            "video_rows": video_rows,
            "conclusion": (
                "No. The batch has prompt/output traces for rgb rows only. Most KPI groups contain "
                "ground-truth geometry, object, map, or video packages that must be converted into "
                "deterministic questions and reference answers before model comparison."
            ),
        },
        "schemas": schemas,
        "prompt_eval_matrix": prompt_eval_matrix,
    }


def render_report(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    endpoint_status: dict[str, Any],
    storage_plan: dict[str, Any],
) -> str:
    summary = metadata.get("summary", summarize_rows(rows))
    schema_summary = build_schema_summary(rows)
    qa = schema_summary["question_answer_coverage"]
    endpoint_rows = []
    for name, probe in endpoint_status.get("endpoints", {}).items():
        endpoint_rows.append(
            f"<tr><td><code>{html.escape(name)}</code></td><td>{html.escape(str(probe.get('status')))}</td>"
            f"<td class='{ 'ok' if probe.get('ok') else 'bad' }'>{pct(bool(probe.get('ok')))}</td>"
            f"<td>{html.escape(str(probe.get('elapsed_seconds', '')))}</td>"
            f"<td><code>{html.escape(probe.get('url', ''))}</code></td></tr>"
        )
    group_rows = []
    for group, count in top_counts(rows, "kpi_group", 40):
        media = sorted({r["media_type"] for r in rows if (r.get("kpi_group") or "(blank)") == group})
        routes = sorted({r["expected_autograder_route"] for r in rows if (r.get("kpi_group") or "(blank)") == group})
        group_rows.append(
            f"<tr><td>{html.escape(group)}</td><td>{fmt_num(count)}</td><td>{html.escape(', '.join(media))}</td>"
            f"<td>{html.escape(', '.join(routes))}</td></tr>"
        )
    schema_rows = []
    for schema in schema_summary["schemas"]:
        observed_fields = "; ".join(schema["observed_fields"])
        schema_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(schema['family'])}</strong><br><span class='subtle'>{html.escape(', '.join(schema['source_datasets']))}</span></td>"
            f"<td>{fmt_num(schema['rows'])}</td>"
            f"<td>{html.escape(schema['media'])}</td>"
            f"<td><code>{html.escape(observed_fields)}</code></td>"
            f"<td>{html.escape(schema['qa_status'])}</td>"
            f"<td>{html.escape(schema['evaluation_plan'])}</td>"
            "</tr>"
        )
    matrix_rows = []
    for item in schema_summary["prompt_eval_matrix"]:
        matrix_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(item['kpi_surface'])}</strong><br><span class='subtle'>{html.escape(item['source'])}</span></td>"
            f"<td>{html.escape(item['prompt'])}</td>"
            f"<td>{html.escape(item['ground_truth'])}</td>"
            f"<td>{html.escape(item['score'])}</td>"
            "</tr>"
        )
    trace_rows = []
    for row in rows[:300]:
        trace_rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['stable_key'])}</code></td>"
            f"<td>{html.escape(row['source_dataset'])}</td>"
            f"<td>{html.escape(row['kpi_group'])}</td>"
            f"<td><code>{html.escape(row['clip_id'])}</code></td>"
            f"<td>{html.escape(row['media_type'])}</td>"
            f"<td>{html.escape(row['expected_autograder_route'])}</td>"
            f"<td><code>{html.escape(row['local_archive_member_path'])}</code></td>"
            "</tr>"
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inference KPI Micro-Benchmark Audit</title>
  <style>
    :root {{ color-scheme: light; --green:#76b900; --ink:#111; --muted:#5f6b7a; --line:#d8e0ea; --soft:#f5f7fa; }}
    body {{ margin:0; font-family: Arial, Helvetica, sans-serif; color:var(--ink); background:white; }}
    header {{ background:#101010; color:white; padding:42px 56px; border-left:8px solid var(--green); }}
    header h1 {{ margin:0 0 12px; font-size:38px; letter-spacing:0; }}
    header p {{ margin:0; color:#d8dce3; font-size:17px; }}
    main {{ max-width:1180px; margin:0 auto; padding:34px 24px 72px; }}
    .guardrail {{ border-left:6px solid var(--green); background:#eef7e8; padding:16px 20px; border-radius:8px; font-size:16px; line-height:1.5; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin:22px 0; }}
    .card {{ border:1px solid var(--line); border-radius:8px; padding:18px; background:white; }}
    .label {{ color:var(--muted); text-transform:uppercase; font-size:12px; letter-spacing:.08em; }}
    .value {{ font-size:32px; font-weight:800; margin-top:8px; }}
    h2 {{ margin-top:34px; font-size:24px; }}
    table {{ width:100%; border-collapse:collapse; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ text-align:left; padding:11px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
    th {{ background:#f1f4f8; font-size:13px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }}
    .ok {{ color:#267000; font-weight:700; }} .bad {{ color:#b00020; font-weight:700; }}
    .links a {{ color:#2d6b00; font-weight:700; margin-right:18px; }}
    .subtle {{ color:var(--muted); }}
    .table-wrap {{ overflow:auto; border-radius:8px; }}
    @media (max-width: 850px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} header {{ padding:34px 28px; }} }}
  </style>
</head>
<body>
<header>
  <h1>Inference KPI Micro-Benchmark Audit</h1>
  <p>Data-audit-first inventory for micro-benchmark KPIs across autograder, Cosmos3 Super, and Cosmos3 Nano.</p>
</header>
<main>
  <div class="guardrail"><strong>Gate 1 boundary:</strong> this report inventories data and endpoint readiness only. It does not run model inference, invoke autograder scoring, or bulk-copy media. Selective downloads are deferred to the next gate.</div>
  <div class="cards">
    <div class="card"><div class="label">Manifest rows</div><div class="value">{fmt_num(len(rows))}</div></div>
    <div class="card"><div class="label">Unique clips</div><div class="value">{fmt_num(summary.get('unique_clips', 0))}</div></div>
    <div class="card"><div class="label">Video rows</div><div class="value">{fmt_num(summary.get('video_rows', 0))}</div></div>
    <div class="card"><div class="label">Selective rows</div><div class="value">{fmt_num(summary.get('selective_download_rows', 0))}</div></div>
  </div>
  <p class="links"><a href="manifest.jsonl">manifest.jsonl</a><a href="manifest.sqlite">manifest.sqlite</a><a href="endpoint_status.json">endpoint_status.json</a><a href="storage_plan.json">storage_plan.json</a><a href="schema_summary.json">schema_summary.json</a><a href="artifacts/inference_kpi_audit_overview.pptx">PowerPoint</a></p>
  <h2>Storage Strategy</h2>
  <p>Default policy: <strong>{html.escape(storage_plan.get('policy', 'remote_first_manifest_selective_download'))}</strong>. Bulk copy allowed: <strong>{html.escape(str(storage_plan.get('bulk_copy_allowed', False)))}</strong>. Remote source preferred; local archive fallback is used for this audit.</p>
  <h2>QA Coverage Answer</h2>
  <div class="guardrail"><strong>Are there plain question-answer pairs for every KPI group?</strong> No. The audit found <strong>{fmt_num(qa['direct_prompt_trace_rows'])}</strong> direct prompt/output JSON traces in the <code>rgb</code> family, across <strong>{fmt_num(qa['unique_kpi_groups_excluding_batch_root'])}</strong> non-batch KPI group names. Most groups are structured ground truth or media packages, so the next gate must synthesize deterministic prompts and reference answers from labels before comparing Super, Nano, and the autograder.</div>
  <h2>Data Schema And Evaluation Map</h2>
  <div class="table-wrap"><table><thead><tr><th>Data family</th><th>Rows</th><th>Observed media</th><th>Observed fields</th><th>QA status</th><th>Evaluation implication</th></tr></thead><tbody>{''.join(schema_rows)}</tbody></table></div>
  <h2>Prompt And Scoring Defaults</h2>
  <div class="table-wrap"><table><thead><tr><th>KPI surface</th><th>Prompt shape</th><th>Ground truth source</th><th>Recommended score</th></tr></thead><tbody>{''.join(matrix_rows)}</tbody></table></div>
  <h2>Endpoint Readiness</h2>
  <div class="table-wrap"><table><thead><tr><th>Endpoint</th><th>Status</th><th>Health</th><th>Seconds</th><th>URL</th></tr></thead><tbody>{''.join(endpoint_rows)}</tbody></table></div>
  <h2>KPI Group Inventory</h2>
  <div class="table-wrap"><table><thead><tr><th>KPI group</th><th>Rows</th><th>Media types</th><th>Expected route</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table></div>
  <h2>Trace Preview</h2>
  <p class="subtle">Showing first 300 manifest rows. Full trace is available in JSONL and SQLite.</p>
  <div class="table-wrap"><table><thead><tr><th>Stable key</th><th>Source</th><th>KPI group</th><th>Clip</th><th>Media</th><th>Route</th><th>Archive member</th></tr></thead><tbody>{''.join(trace_rows)}</tbody></table></div>
</main>
</body>
</html>
"""
    return html_doc


def build_powerpoint(
    out_dir: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    endpoint_status: dict[str, Any],
    storage_plan: dict[str, Any],
) -> Path | None:
    builder = find_artifact_deck_builder()
    if not builder:
        return None
    workspace = out_dir / "presentation" / "inference-kpi-audit"
    slides_dir = workspace / "slides"
    preview_dir = workspace / "preview"
    layout_dir = workspace / "layout"
    output_dir = workspace / "output"
    qa_dir = workspace / "qa"
    for path in [slides_dir, preview_dir, layout_dir, output_dir, qa_dir]:
        ensure_dir(path)
    write_deck_modules(slides_dir, rows, metadata, endpoint_status, storage_plan)
    pptx = output_dir / "inference_kpi_audit_overview.pptx"
    cmd = [
        "node",
        str(builder),
        "--workspace",
        str(workspace),
        "--slides-dir",
        str(slides_dir),
        "--out",
        str(pptx),
        "--preview-dir",
        str(preview_dir),
        "--layout-dir",
        str(layout_dir),
        "--contact-sheet",
        str(qa_dir / "contact-sheet.png"),
        "--slide-count",
        "6",
        "--scale",
        "1",
    ]
    subprocess.run(cmd, check=True)
    return pptx


def find_artifact_deck_builder() -> Path | None:
    root = Path.home() / ".codex/plugins/cache/openai-primary-runtime"
    candidates = sorted(root.glob("presentations/*/skills/presentations/scripts/build_artifact_deck.mjs"))
    return candidates[-1] if candidates else None


def js_string(value: str) -> str:
    return json.dumps(value)


def write_deck_modules(
    slides_dir: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    endpoint_status: dict[str, Any],
    storage_plan: dict[str, Any],
) -> None:
    summary = metadata.get("summary", summarize_rows(rows))
    schema_summary = build_schema_summary(rows)
    qa = schema_summary["question_answer_coverage"]
    endpoint_ok = sum(1 for p in endpoint_status.get("endpoints", {}).values() if p.get("ok"))
    endpoint_total = len(endpoint_status.get("endpoints", {}))
    top_groups = top_counts(rows, "kpi_group", 8)
    top_group_text = "\\n".join(f"{name}: {count}" for name, count in top_groups)
    common = r'''
export const GREEN = "#76B900";
export const BLACK = "#101010";
export const BLUE = "#1677B9";
export const PURPLE = "#7A3ED1";
export const RED = "#D21F3C";
export const MID = "#5f6b7a";
export function addHeader(ctx, slide, eyebrow, title, subtitle) {
  ctx.addShape(slide, { x: 0, y: 0, w: ctx.W, h: 78, fill: BLACK, line: ctx.line() });
  ctx.addShape(slide, { x: 0, y: 0, w: 8, h: ctx.H, fill: GREEN, line: ctx.line() });
  ctx.addText(slide, { x: 38, y: 15, w: 210, h: 14, text: eyebrow, fontSize: 8, bold: true, color: GREEN, typeface: "Arial" });
  ctx.addText(slide, { x: 38, y: 31, w: 780, h: 34, text: title, fontSize: 24, bold: true, color: "#FFFFFF", typeface: "Arial" });
  ctx.addText(slide, { x: 870, y: 30, w: 360, h: 20, text: subtitle, fontSize: 10, color: "#C8D0D8", align: "right", typeface: "Arial" });
}
export function card(ctx, slide, x, y, w, h, title, body, color = GREEN) {
  ctx.addShape(slide, { x, y, w, h, fill: "#FFFFFF", line: { fill: "#D8E0EA", width: 1 } });
  ctx.addShape(slide, { x, y, w: 5, h, fill: color, line: ctx.line() });
  ctx.addText(slide, { x: x + 22, y: y + 18, w: w - 42, h: 22, text: title, fontSize: 17, bold: true, color: "#111111", typeface: "Arial" });
  ctx.addText(slide, { x: x + 22, y: y + 50, w: w - 42, h: h - 58, text: body, fontSize: 13, color: "#303844", fit: "shrink", typeface: "Arial" });
}
export function metric(ctx, slide, x, y, label, value, color = GREEN) {
  ctx.addShape(slide, { x, y, w: 255, h: 86, fill: "#F7FAFC", line: { fill: "#D8E0EA", width: 1 } });
  ctx.addText(slide, { x: x + 18, y: y + 14, w: 220, h: 16, text: label, fontSize: 9, bold: true, color: MID, typeface: "Arial" });
  ctx.addText(slide, { x: x + 18, y: y + 34, w: 220, h: 38, text: value, fontSize: 24, bold: true, color, typeface: "Arial" });
}
export function footer(ctx, slide, n) {
  ctx.addText(slide, { x: 38, y: 690, w: 560, h: 12, text: "Inference KPI audit | data-audit-first gate", fontSize: 7, color: MID, typeface: "Arial" });
  ctx.addText(slide, { x: 1160, y: 690, w: 60, h: 12, text: `${n} / 6`, fontSize: 7, color: MID, align: "right", typeface: "Arial" });
}
'''
    (slides_dir / "common.mjs").write_text(common, encoding="utf-8")
    slide_specs = [
        (
            "slide-01.mjs",
            "slide01",
            "GATE 1 AUDIT",
            "Inference KPI: schema first, inference second",
            "No model inference in this gate",
            [
                ("Manifest", f"{fmt_num(len(rows))} source rows indexed without extraction", "GREEN"),
                ("Endpoint readiness", f"{endpoint_ok} of {endpoint_total} probes healthy; failures remain visible in endpoint_status.json", "BLUE"),
                ("Storage posture", "Remote-first manifest, local tar fallback, selective drill-down only", "PURPLE"),
            ],
            f"Report URL target: {REPORT_URL}",
        ),
        (
            "slide-02.mjs",
            "slide02",
            "DATA SCHEMA",
            "The batch is not one uniform QA dataset",
            "Four evaluation families",
            [
                ("RGB VLM traces", f"{row_count(rows, source_dataset='rgb', media_type='json')} prompt/output JSON files plus matching videos", "GREEN"),
                ("Road and ClipGT", "3D labels, object state, traffic-light state, parquet tables, and recordings", "BLUE"),
                ("World-model videos", "HD-map and world-scenario video variants need aligned GT before scoring", "PURPLE"),
            ],
            "The right next step is prompt synthesis from ground truth, not blanket inference over every archive member.",
        ),
        (
            "slide-03.mjs",
            "slide03",
            "QA COVERAGE",
            "Plain question-answer pairs are not present for every KPI",
            "Direct QA coverage is limited",
            [
                ("Direct traces", f"{fmt_num(qa['direct_prompt_trace_rows'])} rgb JSON traces include instruction prompts and prior outputs", "GREEN"),
                ("KPI groups", f"{fmt_num(qa['unique_kpi_groups_excluding_batch_root'])} non-batch KPI group names; most are structured labels or media", "BLUE"),
                ("Conclusion", "Generate deterministic questions and reference answers from GT before comparing models", "PURPLE"),
            ],
            "This prevents us from grading open-ended captions against geometry labels without an explicit rubric.",
        ),
        (
            "slide-04.mjs",
            "slide04",
            "PROMPT + SCORE",
            "Each KPI surface needs a task-specific prompt and judge",
            "Autograder plus exact checks",
            [
                ("Caption/env", "Ask for JSON scene summary; exact-check weather/time/setting and VLM-score caption fidelity", "GREEN"),
                ("Road geometry", "Ask presence/count/state first; pixel geometry waits for camera projection from GT", "BLUE"),
                ("Objects/states", "Compare class, count, motion, traffic-light state, and temporal alignment", "PURPLE"),
            ],
            "Cosmos-evaluator should receive the same media, prompt, model output, and reference answer per stable key.",
        ),
        (
            "slide-05.mjs",
            "slide05",
            "TRACE CONTRACT",
            "Every comparison row is reproducible",
            "Decision-complete audit schema",
            [
                ("Stable key", "source_dataset + batch_id + kpi_group + clip_id + asset_id + preset", "GREEN"),
                ("Media provenance", "source URI, archive member, media variant, expected route, endpoint payload shape", "BLUE"),
                ("No secrets", "Only credential presence booleans are recorded", "PURPLE"),
            ],
            "This preserves RF100-style traceability while adding multiple sources, models, and autograder routes.",
        ),
        (
            "slide-06.mjs",
            "slide06",
            "NEXT GATE",
            "Run a small micro-benchmark before scaling",
            "Recommended execution order",
            [
                ("1. Select slice", "Choose 20-100 clips across rgb, road labels, ClipGT, and world-model videos.", "GREEN"),
                ("2. Fetch selectively", "Extract only chosen videos, GT bundles, and prompt JSON into horde.", "BLUE"),
                ("3. Score and compare", "Run autograder, Super, and Nano with full payload/response trace.", "PURPLE"),
            ],
            "Only after row-level trace passes should we add Enterprise Inference Hub competitors or broader batches.",
        ),
    ]
    for file_name, export_name, eyebrow, title, subtitle, cards, bottom in slide_specs:
        body = [
            'import { addHeader, card, metric, footer, GREEN, BLUE, PURPLE } from "./common.mjs";',
            f"export async function {export_name}(presentation, ctx) {{",
            "  const slide = presentation.slides.add();",
            '  ctx.addShape(slide, { x: 0, y: 0, w: ctx.W, h: ctx.H, fill: "#FFFFFF", line: ctx.line() });',
            f"  addHeader(ctx, slide, {js_string(eyebrow)}, {js_string(title)}, {js_string(subtitle)});",
        ]
        xs = [54, 462, 870]
        for idx, (card_title, card_body, color_name) in enumerate(cards):
            body.append(f"  card(ctx, slide, {xs[idx]}, 132, 350, 270, {js_string(card_title)}, {js_string(card_body)}, {color_name});")
        body.append('  ctx.addShape(slide, { x: 54, y: 470, w: 1160, h: 92, fill: "#F0F7EA", line: { fill: GREEN, width: 1 } });')
        body.append(f"  ctx.addText(slide, {{ x: 82, y: 498, w: 1100, h: 32, text: {js_string(bottom)}, fontSize: 20, bold: true, color: '#111111', typeface: 'Arial', fit: 'shrink' }});")
        body.append(f"  footer(ctx, slide, {file_name[6:8].lstrip('0') or '1'});")
        body.append("  return slide;")
        body.append("}")
        (slides_dir / file_name).write_text("\n".join(body) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Build manifest from GitLab sparse path or local tar fallback.")
    audit.add_argument("--source-tar", default=str(DEFAULT_SOURCE_TAR))
    audit.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    audit.add_argument("--git-url", default=DEFAULT_GIT_URL)
    audit.add_argument("--git-ref", default="main")
    audit.add_argument("--git-path", default=DEFAULT_GIT_PATH)
    audit.add_argument("--remote-timeout", type=int, default=120)
    audit.add_argument("--skip-git", action="store_true")
    audit.set_defaults(func=command_audit)

    probe = sub.add_parser("probe", help="Probe endpoints without inference.")
    probe.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    probe.add_argument("--timeout", type=int, default=8)
    probe.set_defaults(func=command_probe)

    report = sub.add_parser("report", help="Generate static report and PowerPoint.")
    report.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    report.set_defaults(func=command_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
