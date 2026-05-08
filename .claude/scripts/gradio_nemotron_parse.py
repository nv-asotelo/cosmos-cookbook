#!/usr/bin/env python3
"""
Gradio frontend for Nemotron-Parse v1.2 — document/image structure parser.

UX schema from /tmp/nemotron_parse_ux.json (2026-05-08 Phase A research).

NOT a video VLM. Single-image input. The 1-image cap is the documented
contract (--limit-mm-per-prompt '{"image": 1}'), not a bug.

Inputs:
  - 1 image (jpg/png/pdf-rendered)
  - tool radio: markdown_bbox | markdown_no_bbox | detection_only
  - extract_text_in_pictures checkbox (v1.2-only)
Outputs:
  - Tab 1: Markdown render (concatenated text fields, ordered)
  - Tab 2: Raw JSON (array of {bbox,text,type})
  - Tab 3: Image with bbox overlay (color-coded by type)

NIM API: POST http://localhost:8000/v1/chat/completions
  model: nvidia/nemotron-parse
  tools: [{"type":"function","function":{"name":"<tool>"}}]
  messages: [{role:user, content:[{type:image_url, image_url:{url:"data:image/png;base64,..."}}]}]
  temperature: 0.0 (deterministic)
"""

import base64
import io
import json
import os
import urllib.request
import urllib.error
from pathlib import Path

import gradio as gr  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore

NIM_HOST = os.environ.get("NIM_HOST", "localhost")
NIM_PORT = int(os.environ.get("NIM_PORT", "8000"))
INFER_URL = f"http://{NIM_HOST}:{NIM_PORT}/v1/chat/completions"
MODEL = "nvidia/nemotron-parse"

TYPE_COLORS = {
    "Title":           (0xC0, 0x39, 0x2B),
    "Section-header":  (0x76, 0xB9, 0x00),
    "Text":            (0x1F, 0x77, 0xB4),
    "Page-footer":     (0x55, 0x55, 0x55),
    "Caption":         (0xE6, 0x7E, 0x22),
    "Index":           (0x9B, 0x59, 0xB6),
    "Footnote":        (0x80, 0x80, 0x80),
    "List":            (0x16, 0xA0, 0x85),
    "Table":           (0x2C, 0x3E, 0x50),
    "Bibliography":    (0xC0, 0xA0, 0x60),
    "Picture":         (0xE9, 0x1E, 0x63),
}
DEFAULT_COLOR = (0x40, 0x40, 0x40)


def _img_to_b64(image_path: str) -> tuple[str, str]:
    """Return (data_uri, mime) — data:image/<mime>;base64,..."""
    p = Path(image_path)
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "png")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}", mime


def parse(image_file, tool_choice: str, extract_text_in_pic: bool):
    if not image_file:
        return "❌ Upload an image first.", "[]", None
    data_uri, _ = _img_to_b64(image_file)

    payload = {
        "model": MODEL,
        "tools": [{"type": "function", "function": {"name": tool_choice}}],
        "messages": [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
        }],
        "temperature": 0.0,
    }
    # v1.2 extract-text-in-pic toggle: NIM may surface as a separate tool variant
    # OR via prompt_token. Try a header hint; NIM ignores unknown headers safely.
    headers = {"Content-Type": "application/json"}
    if extract_text_in_pic:
        headers["X-Nemotron-Parse-Predict-Text-In-Pic"] = "1"

    req = urllib.request.Request(
        INFER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
            data = json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500] if hasattr(e, "read") else str(e)
        return f"❌ HTTP {e.code}: {err_body}", "[]", None
    except Exception as e:
        return f"❌ Request failed: {str(e)[:300]}", "[]", None

    try:
        msg = data["choices"][0]["message"]
        tcs = msg.get("tool_calls", [])
        if not tcs:
            content = msg.get("content", "")
            return f"⚠️ No tool_calls in response. Content: {content[:500]}", json.dumps(data)[:1000], None
        args_str = tcs[0]["function"]["arguments"]
        items = json.loads(args_str) if isinstance(args_str, str) else args_str
        if not isinstance(items, list):
            items = items.get("items", []) if isinstance(items, dict) else []
    except Exception as e:
        return f"❌ Parse error: {e}", json.dumps(data)[:1000], None

    # Build markdown render
    md_lines = []
    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        md_lines.append(text)
    md_render = "\n\n".join(md_lines) or "_(empty)_"

    # Build bbox overlay
    overlay_path = None
    try:
        img = Image.open(image_file).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for it in items:
            bb = it.get("bbox") or {}
            t = it.get("type") or "Text"
            color = TYPE_COLORS.get(t, DEFAULT_COLOR)
            x0 = float(bb.get("xmin", 0)) * w
            y0 = float(bb.get("ymin", 0)) * h
            x1 = float(bb.get("xmax", 0)) * w
            y1 = float(bb.get("ymax", 0)) * h
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            if font:
                draw.text((x0 + 3, max(0, y0 - 14)), t, fill=color, font=font)
        overlay_path = "/tmp/nemotron_parse_overlay.png"
        img.save(overlay_path)
    except Exception as e:
        # Overlay best-effort; not fatal
        overlay_path = None

    info = f"✅ Parsed {len(items)} regions · tool={tool_choice}"
    return md_render + f"\n\n---\n_{info}_", json.dumps(items, indent=2), overlay_path


with gr.Blocks(title="Nemotron-Parse v1.2") as demo:
    gr.Markdown(
        "# Nemotron-Parse v1.2 — Document Structure Parser\n"
        "Single-image structured extraction. Tools: `markdown_bbox` (markdown + bounding boxes), "
        "`markdown_no_bbox` (markdown only), `detection_only` (bboxes + classes, no text). "
        "Replicates the build.nvidia.com Nemotron Parse playground UX for the self-hosted NIM."
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="Document image (1 only — model contract)",
                                type="filepath", sources=["upload", "clipboard"])
            tool_choice = gr.Radio(
                choices=["markdown_bbox", "markdown_no_bbox", "detection_only"],
                value="markdown_bbox",
                label="Tool",
                info="markdown_bbox: structured + bboxes · markdown_no_bbox: markdown only · "
                     "detection_only: bboxes + classes only (no text)",
            )
            extract_pics = gr.Checkbox(
                label="Extract text from embedded pictures (v1.2)",
                value=False,
                info="When enabled, transcribe text inside any photos/figures embedded in the document.",
            )
            submit = gr.Button("Parse", variant="primary")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Markdown render"):
                    md_out = gr.Markdown()
                with gr.TabItem("Raw JSON"):
                    json_out = gr.Code(language="json")
                with gr.TabItem("Image with bbox overlay"):
                    overlay_out = gr.Image(label="Color-coded by region type")
    submit.click(parse, [image_in, tool_choice, extract_pics],
                 [md_out, json_out, overlay_out])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
