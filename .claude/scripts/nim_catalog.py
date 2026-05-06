#!/usr/bin/env python3
"""
nim_catalog.py — VLM NIM catalog helper for /byo-video.

Single source of truth: the agents always re-check
  https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html
when the user selects NIM mode, so this catalog reflects what NVIDIA
currently ships rather than a static frozen list.

How it works:
  1. fetch_supported_models() — GETs the introduction.html, extracts the
     model name column from the table on the page. No nvcr.io paths or
     served model IDs are exposed at the introduction level (those live
     in per-model release notes), so we maintain a small slug map below
     to translate display names into image short-IDs.
  2. KNOWN_VLM_NIMS — the slug map. Update when a new VLM family is
     added upstream (the agents are told to do this when fetch finds a
     name not in the map).
  3. list_available_nims(ngc_api_key, vram_mb=None) — intersects the
     upstream catalog with the slug map, then probes each candidate
     image via `docker manifest inspect` to confirm the user's
     NGC_API_KEY actually has access. Optionally filters by min VRAM.
  4. CLI: `python3 nim_catalog.py list [--vram MB]` → prints JSON.

Network failures fall back to the slug map alone (no upstream filter).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import List, Optional

DOCS_URL = "https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html"


@dataclass
class NimImage:
    short_id: str           # e.g. "cosmos-reason2-8b"
    image: str              # e.g. "nvcr.io/nim/nvidia/cosmos-reason2-8b:latest"
    family: str             # e.g. "Cosmos Reason2"
    label: str              # human-readable label for the dropdown
    served_model_id: str    # e.g. "nvidia/cosmos-reason2-8b" (best-effort)
    min_vram_mb: int = 0    # 0 if unknown
    notes: str = ""


# Slug + min-VRAM map — translates display names from introduction.html
# into actual nvcr.io short-IDs. Conservative VRAM minimums; refine per
# model card. Add new entries here when the upstream catalog grows.
KNOWN_VLM_NIMS = [
    NimImage("cosmos-reason2-2b",  "nvcr.io/nim/nvidia/cosmos-reason2-2b:latest",
             "Cosmos Reason2", "Cosmos Reason2 2B (NIM)",
             "nvidia/cosmos-reason2-2b", min_vram_mb=20000,
             notes="FP8 quantized; reasoning VLM; needs temperature ≥ 0.3 to avoid <think>+EOS bug at greedy decode."),
    NimImage("cosmos-reason2-8b",  "nvcr.io/nim/nvidia/cosmos-reason2-8b:latest",
             "Cosmos Reason2", "Cosmos Reason2 8B (NIM)",
             "nvidia/cosmos-reason2-8b", min_vram_mb=40000,
             notes="FP8 quantized; reasoning VLM; needs temperature ≥ 0.3 to avoid <think>+EOS bug at greedy decode."),
    NimImage("cosmos-reason2-32b", "nvcr.io/nim/nvidia/cosmos-reason2-32b:latest",
             "Cosmos Reason2", "Cosmos Reason2 32B (NIM)",
             "nvidia/cosmos-reason2-32b", min_vram_mb=80000,
             notes="May require allowlisting via build.nvidia.com (not in default NGC catalog)."),
    NimImage("cosmos-reason1-7b",  "nvcr.io/nim/nvidia/cosmos-reason1-7b:latest",
             "Cosmos Reason1 7B", "Cosmos Reason1 7B (NIM)",
             "nvidia/cosmos-reason1-7b", min_vram_mb=24000),
    NimImage("nemotron-nano-12b-v2-vl",
             "nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:latest",
             "Nemotron Nano 12B v2 VL", "Nemotron Nano 12B v2 VL (NIM)",
             "nvidia/nemotron-nano-12b-v2-vl", min_vram_mb=40000),
    NimImage("llama-3.1-nemotron-nano-vl-8b-v1",
             "nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1:latest",
             "Llama 3.1 Nemotron Nano VL 8B v1",
             "Llama 3.1 Nemotron Nano VL 8B (NIM)",
             "nvidia/llama-3.1-nemotron-nano-vl-8b-v1", min_vram_mb=40000),
    NimImage("llama-3.2-11b-vision-instruct",
             "nvcr.io/nim/meta/llama-3.2-11b-vision-instruct:latest",
             "Llama 3.2 11B Vision Instruct",
             "Llama 3.2 11B Vision Instruct (NIM)",
             "meta/llama-3.2-11b-vision-instruct", min_vram_mb=40000),
    NimImage("llama-3.2-90b-vision-instruct",
             "nvcr.io/nim/meta/llama-3.2-90b-vision-instruct:latest",
             "Llama 3.2 90B Vision Instruct",
             "Llama 3.2 90B Vision Instruct (NIM)",
             "meta/llama-3.2-90b-vision-instruct", min_vram_mb=160000,
             notes="Multi-GPU only (~160GB+ VRAM)."),
    NimImage("llama-4-maverick-17b-128e-instruct",
             "nvcr.io/nim/meta/llama-4-maverick-17b-128e-instruct:latest",
             "Llama 4 Maverick 17B 128E Instruct",
             "Llama 4 Maverick 17B (NIM)",
             "meta/llama-4-maverick-17b-128e-instruct", min_vram_mb=80000),
    NimImage("llama-4-scout-17b-16e-instruct",
             "nvcr.io/nim/meta/llama-4-scout-17b-16e-instruct:latest",
             "Llama 4 Scout 17B 16E Instruct",
             "Llama 4 Scout 17B (NIM)",
             "meta/llama-4-scout-17b-16e-instruct", min_vram_mb=80000),
    NimImage("mistral-small-3.2-24b-instruct-2506",
             "nvcr.io/nim/mistralai/mistral-small-3.2-24b-instruct-2506:latest",
             "Mistral Small 3.2 24B Instruct 2506",
             "Mistral Small 3.2 24B (NIM)",
             "mistralai/mistral-small-3.2-24b-instruct-2506", min_vram_mb=60000),
]


def fetch_supported_models(timeout: int = 10) -> List[str]:
    """Fetch DOCS_URL and return the list of model names in the table.
    Returns [] on any network or parse failure — callers must tolerate that."""
    try:
        req = urllib.request.Request(
            DOCS_URL,
            headers={"User-Agent": "byo-video-skill nim_catalog/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    # Extract every <tr> ... </tr>, then for each row pull text from cells.
    # The introduction.html table has THREE columns: Model | Latest VLM NIM
    # Version | Documentation. Model name is in cell 0. Version-string cells
    # like "2.0.0" or "1.7.0 (uses the 1.7.0-variant release tag)" are skipped.
    names: List[str] = []
    _version_re = re.compile(r"^\d+(\.\d+){1,3}")
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.DOTALL | re.IGNORECASE):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
        if not cells:
            continue
        raw = cells[0]
        # Prefer text inside <a>...</a> if present (link text == model name).
        m = re.search(r"<a[^>]*>(.*?)</a>", raw, re.DOTALL | re.IGNORECASE)
        text = m.group(1) if m else raw
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text).strip()
        if not text or text.lower() in ("model", ""):
            continue
        # Skip version-style or release-link cells if a header row was malformed.
        if _version_re.match(text) or text.lower().startswith("release notes"):
            continue
        names.append(text)
    # De-dupe while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _matches_family(name: str, family: str) -> bool:
    """Loose match: introduction.html sometimes prints 'Cosmos Reason2'
    while we want to match all sizes (2B/8B/32B). Compare lowercased
    prefix-form, ignoring punctuation differences."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    return norm(family) in norm(name) or norm(name) in norm(family)


def probe_image(image: str, timeout: int = 30) -> bool:
    """Return True if `docker manifest inspect <image>` succeeds. Requires a
    prior `docker login nvcr.io` to have populated ~/.docker/config.json."""
    try:
        r = subprocess.run(
            ["docker", "manifest", "inspect", image],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False


def list_available_nims(
    ngc_api_key: Optional[str] = None,
    vram_mb: Optional[int] = None,
    use_upstream: bool = True,
    do_probe: bool = True,
) -> List[NimImage]:
    """Return the subset of KNOWN_VLM_NIMS that:
      - has a family name present on DOCS_URL (when use_upstream and reachable),
      - fits in vram_mb if given,
      - passes a `docker manifest inspect` probe when do_probe.
    `ngc_api_key` is informational only — `docker login nvcr.io` must already
    have been done for probing to work."""
    upstream_names: List[str] = fetch_supported_models() if use_upstream else []
    out: List[NimImage] = []
    for nim in KNOWN_VLM_NIMS:
        if upstream_names and not any(_matches_family(n, nim.family) for n in upstream_names):
            continue
        if vram_mb is not None and nim.min_vram_mb and nim.min_vram_mb > vram_mb:
            continue
        if do_probe and not probe_image(nim.image):
            continue
        out.append(nim)
    return out


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="VLM NIM catalog helper")
    sp = p.add_subparsers(dest="cmd", required=True)
    pl = sp.add_parser("list", help="List available NIMs as JSON")
    pl.add_argument("--vram", type=int, default=None,
                    help="Filter by max VRAM available on target (MB)")
    pl.add_argument("--no-upstream", action="store_true",
                    help="Skip fetching docs.nvidia.com (use slug map only)")
    pl.add_argument("--no-probe", action="store_true",
                    help="Skip `docker manifest inspect` probes")
    sp.add_parser("upstream", help="Print model names from docs.nvidia.com")
    args = p.parse_args(argv)

    if args.cmd == "upstream":
        names = fetch_supported_models()
        print(json.dumps(names, indent=2))
        return 0
    if args.cmd == "list":
        nims = list_available_nims(
            ngc_api_key=os.environ.get("NGC_API_KEY"),
            vram_mb=args.vram,
            use_upstream=not args.no_upstream,
            do_probe=not args.no_probe,
        )
        print(json.dumps([asdict(n) for n in nims], indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
