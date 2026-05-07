#!/usr/bin/env python3
"""
gen_pptx_nim_smoke_summary.py — NIM Smoke Battery summary deck (for Alex).

Reads nim_smoke_*.json files from --in, builds a NVIDIA-branded PPTX:
- Title slide
- Coverage matrix slide (NIMs × pass/warn/fail)
- One slide per NIM (status, served_id, recommendation, root cause if failed)
- Recommendations summary (support / fix / drop)

Output: --out path. NVIDIA brand colors enforced, 24pt minimum body, 28pt+ titles.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

NV_GREEN = RGBColor(0x76, 0xB9, 0x00)
NV_DARK = RGBColor(0x1A, 0x1A, 0x1A)
NV_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NV_LIGHT = RGBColor(0xF0, 0xF5, 0xE8)
NV_PASS = RGBColor(0x2E, 0x7D, 0x32)
NV_WARN = RGBColor(0xEF, 0x6C, 0x00)
NV_FAIL = RGBColor(0xC6, 0x28, 0x28)


def clear_template_slides(prs: Presentation) -> None:
    """Remove default placeholder slides if any exist."""
    sldIdLst = prs.slides._sldIdLst
    for sld in list(sldIdLst):
        sldIdLst.remove(sld)


def add_blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_textbox(slide, left, top, width, height, text, size=24, bold=False, color=NV_DARK,
                align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    if "\n" in text:
        first, *rest = text.split("\n")
        p.text = first
        for run in p.runs:
            run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
        for line in rest:
            np = tf.add_paragraph()
            np.text = line; np.alignment = align
            for run in np.runs:
                run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    else:
        p.text = text
        for run in p.runs:
            run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    return tb


def add_filled_rect(slide, left, top, width, height, fill=NV_GREEN, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def classify(result: dict) -> tuple[str, str, str, str]:
    phase = result.get("phase", "unknown")
    if phase != "complete":
        err = result.get("error") or ""
        if "denied" in err.lower() and ("manifest" in err.lower() or "access denied" in err.lower()):
            return ("FAIL", "DROP", "Image not pullable — NGC entitlement missing or pre-release",
                    f"docker pull DENIED: {err[:240]}")
        if "permissionerror" in err.lower():
            return ("FAIL", "FIX", "Container PermissionError on cache directory",
                    "Fix: drop bind mount or chmod -R the host cache dir")
        if "timeout" in err.lower() and "endpoint" in err.lower():
            return ("FAIL", "FIX", "/v1/models endpoint timeout",
                    "Container did not become ready within poll window")
        if "no space left" in err.lower():
            return ("FAIL", "FIX", "Disk exhaustion during pull/load",
                    "Free 80GB+ before pull; prune dead containers between swaps")
        return ("FAIL", "FIX", f"Phase: {phase}", err[:240])
    runs = result.get("smoke_runs", {})
    statuses = [v.get("http_status") for v in runs.values()]
    all_ok = statuses and all(s == 200 for s in statuses)
    any_ok = any(s == 200 for s in statuses)
    if all_ok:
        empty = []
        for k, v in runs.items():
            try:
                content = v["response"]["choices"][0]["message"]["content"]
                if not (content or "").strip():
                    empty.append(k)
            except Exception:
                empty.append(k)
        if empty:
            return ("WARN", "FIX", f"Empty response on: {', '.join(empty)}",
                    "Inference 200 but content empty — sampling/decode investigation")
        return ("PASS", "SUPPORT", "All prompts returned valid output", "")
    elif any_ok:
        return ("WARN", "FIX", "Partial success — at least one prompt failed",
                f"statuses={statuses}")
    else:
        return ("FAIL", "FIX", "All prompts failed", f"statuses={statuses}")


def title_slide(prs, total: int, p: int, w: int, f: int):
    s = add_blank(prs)
    add_filled_rect(s, 0, 0, prs.slide_width, prs.slide_height, fill=NV_DARK)
    add_filled_rect(s, 0, Inches(2.5), prs.slide_width, Emu(228600), fill=NV_GREEN)
    add_textbox(s, Inches(0.7), Inches(1.0), Inches(12), Inches(1.2),
                "NIM Smoke Battery", size=44, bold=True, color=NV_WHITE)
    add_textbox(s, Inches(0.7), Inches(1.9), Inches(12), Inches(0.6),
                "Reasoning ON + Reasoning OFF · 11 VLM NIMs · Single H200-class GPU",
                size=22, color=NV_GREEN)
    add_textbox(s, Inches(0.7), Inches(3.2), Inches(12), Inches(1.0),
                f"Total NIMs: {total}    PASS: {p}    WARN: {w}    FAIL: {f}",
                size=24, bold=True, color=NV_WHITE)
    add_textbox(s, Inches(0.7), Inches(4.2), Inches(12), Inches(1.0),
                "Host: horde@10.57.233.111 (NVIDIA RTX PRO 6000 Blackwell, 96GB VRAM, 244GB disk)",
                size=18, color=NV_LIGHT)
    add_textbox(s, Inches(0.7), Inches(6.5), Inches(12), Inches(0.6),
                "Sprint 2026-05-06 · Cosmos Cookbook · Internal", size=14, color=NV_LIGHT)


def coverage_slide(prs, results: list[dict]):
    s = add_blank(prs)
    add_filled_rect(s, 0, 0, prs.slide_width, Inches(0.7), fill=NV_DARK)
    add_textbox(s, Inches(0.4), Inches(0.1), prs.slide_width - Inches(0.8), Inches(0.5),
                "Coverage Matrix · 11 NIMs", size=28, bold=True, color=NV_WHITE)
    cols = [("NIM short", Inches(3.5)), ("Status", Inches(1.2)), ("Recommendation", Inches(1.6)),
            ("Headline", Inches(6.6))]
    x = Inches(0.4)
    y = Inches(1.0)
    h_row = Inches(0.42)
    # header
    for label, w in cols:
        add_filled_rect(s, x, y, w, h_row, fill=NV_GREEN)
        tb = add_textbox(s, x, y + Inches(0.05), w, h_row,
                         label, size=14, bold=True, color=NV_WHITE)
        x += w
    y += h_row
    for r in results:
        short = r.get("short", "?")
        badge, rec, headline, _ = classify(r)
        x = Inches(0.4)
        bg = NV_LIGHT if (results.index(r) % 2 == 0) else NV_WHITE
        for label, w in cols:
            add_filled_rect(s, x, y, w, h_row, fill=bg, line=RGBColor(0xCC, 0xCC, 0xCC))
            x += w
        x = Inches(0.4)
        # NIM name
        add_textbox(s, x + Inches(0.05), y + Inches(0.05), cols[0][1] - Inches(0.1), h_row,
                    short, size=12, bold=False, color=NV_DARK)
        x += cols[0][1]
        # Status badge
        sc = NV_PASS if badge == "PASS" else (NV_WARN if badge == "WARN" else NV_FAIL)
        add_filled_rect(s, x + Inches(0.1), y + Inches(0.07), Inches(0.95), h_row - Inches(0.14), fill=sc)
        add_textbox(s, x + Inches(0.1), y + Inches(0.07), Inches(0.95), h_row - Inches(0.14),
                    badge, size=11, bold=True, color=NV_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        x += cols[1][1]
        # Recommendation
        add_textbox(s, x + Inches(0.05), y + Inches(0.05), cols[2][1] - Inches(0.1), h_row,
                    rec, size=12, bold=True,
                    color=NV_PASS if rec == "SUPPORT" else (NV_FAIL if rec == "DROP" else NV_WARN))
        x += cols[2][1]
        # Headline
        add_textbox(s, x + Inches(0.05), y + Inches(0.05), cols[3][1] - Inches(0.1), h_row,
                    headline[:120], size=11, color=NV_DARK)
        y += h_row


def per_nim_slide(prs, r: dict):
    short = r.get("short", "?")
    image = r.get("image", "")
    served = r.get("served_model_id", "—")
    badge, rec, headline, root_cause = classify(r)
    s = add_blank(prs)
    # Top bar
    add_filled_rect(s, 0, 0, prs.slide_width, Inches(0.7), fill=NV_DARK)
    add_textbox(s, Inches(0.4), Inches(0.1), prs.slide_width - Inches(2.5), Inches(0.5),
                short, size=28, bold=True, color=NV_WHITE)
    sc = NV_PASS if badge == "PASS" else (NV_WARN if badge == "WARN" else NV_FAIL)
    add_filled_rect(s, prs.slide_width - Inches(1.7), Inches(0.13), Inches(1.3), Inches(0.45), fill=sc)
    add_textbox(s, prs.slide_width - Inches(1.7), Inches(0.13), Inches(1.3), Inches(0.45),
                badge, size=18, bold=True, color=NV_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # Body
    y = Inches(1.0)
    add_textbox(s, Inches(0.4), y, prs.slide_width - Inches(0.8), Inches(0.5),
                f"Image: {image}", size=16, color=NV_DARK)
    y += Inches(0.55)
    add_textbox(s, Inches(0.4), y, prs.slide_width - Inches(0.8), Inches(0.5),
                f"Served model: {served}", size=16, color=NV_DARK)
    y += Inches(0.55)
    rec_color = NV_PASS if rec == "SUPPORT" else (NV_FAIL if rec == "DROP" else NV_WARN)
    add_filled_rect(s, Inches(0.4), y, prs.slide_width - Inches(0.8), Inches(0.7),
                    fill=NV_LIGHT, line=rec_color)
    add_textbox(s, Inches(0.6), y + Inches(0.1), prs.slide_width - Inches(1.2), Inches(0.5),
                f"Recommendation: {rec} — {headline}", size=18, bold=True, color=rec_color)
    y += Inches(0.85)
    if root_cause:
        add_textbox(s, Inches(0.4), y, prs.slide_width - Inches(0.8), Inches(0.4),
                    "Root cause", size=18, bold=True, color=NV_DARK)
        y += Inches(0.45)
        add_textbox(s, Inches(0.4), y, prs.slide_width - Inches(0.8), Inches(2.0),
                    root_cause[:600], size=14, color=NV_DARK)
        y += Inches(2.0)
    if r.get("phase") == "complete":
        runs = r.get("smoke_runs", {})
        for label, run in runs.items():
            try:
                content = run["response"]["choices"][0]["message"]["content"]
            except Exception:
                content = str(run.get("response", ""))[:300]
            add_textbox(s, Inches(0.4), y, prs.slide_width - Inches(0.8), Inches(0.4),
                        f"{label} (HTTP {run.get('http_status')}, {run.get('elapsed_s')}s)",
                        size=14, bold=True, color=NV_DARK)
            y += Inches(0.4)
            preview = content[:280] + ("…" if len(content) > 280 else "")
            add_textbox(s, Inches(0.4), y, prs.slide_width - Inches(0.8), Inches(0.8),
                        preview, size=12, color=NV_DARK)
            y += Inches(0.85)


def recommendations_slide(prs, results: list[dict]):
    s = add_blank(prs)
    add_filled_rect(s, 0, 0, prs.slide_width, Inches(0.7), fill=NV_DARK)
    add_textbox(s, Inches(0.4), Inches(0.1), prs.slide_width - Inches(0.8), Inches(0.5),
                "Recommendations", size=28, bold=True, color=NV_WHITE)
    by = {"SUPPORT": [], "FIX": [], "DROP": []}
    for r in results:
        _, rec, headline, _ = classify(r)
        by[rec].append((r.get("short", "?"), headline))
    y = Inches(1.0)
    for rec, items in by.items():
        if not items:
            continue
        rec_color = NV_PASS if rec == "SUPPORT" else (NV_FAIL if rec == "DROP" else NV_WARN)
        add_filled_rect(s, Inches(0.4), y, Inches(1.6), Inches(0.45), fill=rec_color)
        add_textbox(s, Inches(0.4), y, Inches(1.6), Inches(0.45), rec,
                    size=16, bold=True, color=NV_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(0.55)
        for short, headline in items:
            add_textbox(s, Inches(0.6), y, prs.slide_width - Inches(1.2), Inches(0.45),
                        f"• {short} — {headline}", size=14, color=NV_DARK)
            y += Inches(0.45)
        y += Inches(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outpath", required=True)
    args = ap.parse_args()
    indir = Path(args.indir)
    results = []
    for f in sorted(indir.glob("nim_smoke_*.json")):
        try:
            d = json.loads(f.read_text())
            results.append(d)
        except json.JSONDecodeError:
            continue
    p = sum(1 for r in results if classify(r)[0] == "PASS")
    w = sum(1 for r in results if classify(r)[0] == "WARN")
    f = sum(1 for r in results if classify(r)[0] == "FAIL")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    clear_template_slides(prs)
    title_slide(prs, len(results), p, w, f)
    coverage_slide(prs, results)
    for r in results:
        per_nim_slide(prs, r)
    recommendations_slide(prs, results)
    out = Path(args.outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"wrote {out} · {len(results)} NIMs · PASS={p} WARN={w} FAIL={f}")


if __name__ == "__main__":
    main()
