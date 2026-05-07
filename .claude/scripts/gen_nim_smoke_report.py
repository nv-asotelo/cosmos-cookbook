#!/usr/bin/env python3
"""
gen_nim_smoke_report.py — Build per-NIM HTML reports + summary HTML index from
nim_smoke_*.json artifacts.

Usage:
  python3 gen_nim_smoke_report.py --in <smoke-dir> --out <html-out-dir>

Reads all nim_smoke_*.json in --in. Emits one HTML page per NIM plus index.html.

The summary PPTX is built by gen_pptx_nim_smoke_summary.py (separate file).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from html import escape


PAGE_CSS = """
:root {
  --nv-green: #76B900;
  --nv-dark: #1A1A1A;
  --bg: #FFFFFF;
  --bg-soft: #F5F7F1;
  --text: #1A1A1A;
  --muted: #666;
  --pass: #2E7D32;
  --fail: #C62828;
  --warn: #EF6C00;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 0; color: var(--text); background: var(--bg); line-height: 1.5; }
header { background: var(--nv-dark); color: #FFFFFF; padding: 18px 28px; }
header h1 { margin: 0; font-size: 22px; font-weight: 600; }
header .meta { color: #B7C9A6; font-size: 13px; margin-top: 4px; }
header .meta a { color: var(--nv-green); text-decoration: none; }
main { max-width: 1080px; margin: 0 auto; padding: 28px; }
h2 { font-size: 18px; margin-top: 28px; padding-bottom: 6px; border-bottom: 2px solid var(--nv-green); }
h3 { font-size: 15px; margin-top: 18px; color: var(--nv-dark); }
table { width: 100%; border-collapse: collapse; margin: 8px 0 16px; font-size: 14px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #E0E0E0; vertical-align: top; }
th { background: var(--bg-soft); font-weight: 600; color: var(--nv-dark); }
.kbd { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
       background: var(--bg-soft); padding: 1px 6px; border-radius: 3px; }
pre { background: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 4px;
      padding: 12px; overflow-x: auto; font-size: 13px; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 3px; font-size: 12px;
         font-weight: 600; color: white; }
.badge.pass { background: var(--pass); }
.badge.fail { background: var(--fail); }
.badge.warn { background: var(--warn); }
.recommend { padding: 10px 14px; border-left: 4px solid var(--nv-green);
             background: var(--bg-soft); margin: 12px 0; border-radius: 3px; }
.recommend.support { border-color: var(--pass); }
.recommend.drop    { border-color: var(--fail); }
.recommend.fix     { border-color: var(--warn); }
footer { color: var(--muted); font-size: 12px; padding: 24px 28px; border-top: 1px solid #E0E0E0; }
.nav-back a { color: var(--nv-green); text-decoration: none; font-weight: 600; }
"""


def classify(result: dict) -> tuple[str, str, str, str]:
    """Return (status_badge, recommendation, headline, root_cause_summary)."""
    phase = result.get("phase", "unknown")
    if phase != "complete":
        # Container failed to come up
        err = (result.get("error") or "")
        if "denied" in err.lower() and "manifest" in err.lower():
            return ("fail", "drop",
                    "Image not pullable — NGC entitlement missing or pre-release",
                    f"docker pull DENIED: {err[:200]}")
        if "permissionerror" in err.lower():
            return ("fail", "fix",
                    "Container PermissionError on cache directory",
                    "PermissionError on /opt/nim/.cache mount — drop bind mount or chmod recursively")
        if "timeout" in err.lower():
            return ("fail", "fix",
                    "/v1/models endpoint timeout — container did not become ready",
                    err[:300])
        if "no space left" in err.lower():
            return ("fail", "fix",
                    "Disk exhaustion during pull/load",
                    "Free disk before pull; 80GB minimum recommended for 50GB+ images")
        return ("fail", "fix", f"Phase: {phase}", err[:300])
    # phase == complete
    runs = result.get("smoke_runs", {})
    statuses = [v.get("http_status") for v in runs.values()]
    all_ok = statuses and all(s == 200 for s in statuses)
    any_ok = any(s == 200 for s in statuses)
    if all_ok:
        # Check content quality
        empty = []
        for k, v in runs.items():
            try:
                content = v["response"]["choices"][0]["message"]["content"]
                if not (content or "").strip():
                    empty.append(k)
            except Exception:
                empty.append(k)
        if empty:
            return ("warn", "fix",
                    f"Empty response on: {', '.join(empty)}",
                    "Inference returned 200 but content is empty — sampling/decode issue")
        return ("pass", "support",
                "All prompts returned valid output via OpenAI-compatible API",
                "")
    elif any_ok:
        return ("warn", "fix",
                "Partial success — at least one prompt failed",
                f"statuses={statuses}")
    else:
        return ("fail", "fix",
                "All prompts failed",
                f"statuses={statuses}")


def render_one_run(label: str, run: dict) -> str:
    parts = []
    parts.append(f"<h3>{escape(label)}</h3>")
    status = run.get("http_status")
    badge_cls = "pass" if status == 200 else ("warn" if status and 400 <= status < 500 else "fail")
    badge_txt = f"HTTP {status}" if status is not None else "no-response"
    parts.append(f'<p><span class="badge {badge_cls}">{badge_txt}</span> '
                 f'· video_mode_actual: <span class="kbd">{escape(run.get("video_mode_actual",""))}</span> '
                 f'· elapsed: {run.get("elapsed_s","?")}s</p>')
    fb = run.get("fallback_record")
    if fb:
        parts.append("<p><em>fallback was triggered</em></p>")
        parts.append(f"<pre>{escape(json.dumps(fb, indent=2)[:1500])}</pre>")
    # Request params used
    rp = run.get("request_params", {})
    parts.append("<table>")
    parts.append("<tr><th>Field</th><th>Value</th></tr>")
    for k in ("max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"):
        parts.append(f"<tr><td>{k}</td><td>{escape(str(rp.get(k,'')))}</td></tr>")
    parts.append(f"<tr><td>fps</td><td>{escape(str(run.get('fps','')))}</td></tr>")
    parts.append(f"<tr><td>max_pixels</td><td>{escape(str(run.get('max_pixels','')))}</td></tr>")
    parts.append(f"<tr><td>max_frames</td><td>{escape(str(run.get('max_frames','')))}</td></tr>")
    parts.append("</table>")
    # Output
    response = run.get("response")
    parts.append("<h4>Response (verbatim)</h4>")
    if isinstance(response, dict):
        try:
            content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            parts.append(f'<p>Usage: <span class="kbd">{escape(str(usage))}</span></p>')
            parts.append(f"<pre>{escape(content)}</pre>")
        except Exception:
            parts.append(f"<pre>{escape(json.dumps(response, indent=2)[:4000])}</pre>")
    else:
        parts.append(f"<pre>{escape(str(response)[:4000])}</pre>")
    return "\n".join(parts)


def render_per_nim_page(result: dict) -> str:
    short = result.get("short", "?")
    image = result.get("image", "?")
    served = result.get("served_model_id", "—")
    badge, rec, headline, rca = classify(result)
    parts = []
    parts.append(f"""<!doctype html><html><head><meta charset="utf-8">
<title>NIM Smoke — {escape(short)}</title>
<style>{PAGE_CSS}</style>
</head><body>
<header>
  <h1>{escape(short)} · NIM Smoke Report</h1>
  <div class="meta">
    Image: <span class="kbd">{escape(image)}</span> ·
    Served model: <span class="kbd">{escape(served)}</span> ·
    <span class="badge {badge}">{badge.upper()}</span>
  </div>
</header>
<main>
<p class="nav-back"><a href="index.html">← Back to all NIMs</a></p>
<div class="recommend {rec}">
  <strong>Recommendation: {rec.upper()}</strong> — {escape(headline)}
</div>
""")
    if rca:
        parts.append("<h2>Root cause</h2>")
        parts.append(f"<pre>{escape(rca)}</pre>")
    # Smoke metadata
    parts.append("<h2>Smoke environment</h2>")
    parts.append("<table>")
    parts.append("<tr><th>Field</th><th>Value</th></tr>")
    parts.append(f"<tr><td>Host</td><td>horde@10.57.233.111 (RTX PRO 6000 Blackwell, 96GB)</td></tr>")
    parts.append(f"<tr><td>Input video</td><td><span class='kbd'>{escape(result.get('input_video',''))}</span></td></tr>")
    parts.append(f"<tr><td>System prompt</td><td><pre>{escape(result.get('system_prompt',''))}</pre></td></tr>")
    parts.append(f"<tr><td>User prompt (reasoning ON)</td><td><pre>race car: timestamps</pre></td></tr>")
    parts.append(f"<tr><td>User prompt (reasoning OFF)</td><td><pre>temporal summary</pre></td></tr>")
    parts.append(f"<tr><td>Started</td><td>{escape(str(result.get('started_at','')))}</td></tr>")
    parts.append(f"<tr><td>Finished</td><td>{escape(str(result.get('finished_at','')))}</td></tr>")
    parts.append("</table>")
    if result.get("phase") == "complete":
        runs = result.get("smoke_runs", {})
        parts.append("<h2>Smoke runs</h2>")
        for label, run in runs.items():
            parts.append(render_one_run(label, run))
    else:
        parts.append("<h2>Failure detail</h2>")
        parts.append(f"<p>Phase reached: <span class='kbd'>{escape(str(result.get('phase')))}</span></p>")
        if result.get("error"):
            parts.append(f"<pre>{escape(result['error'][:4000])}</pre>")
        if result.get("container_logs_tail"):
            parts.append("<h3>Container logs (tail)</h3>")
            parts.append(f"<pre>{escape(result['container_logs_tail'][:6000])}</pre>")
    parts.append("""
<footer>Generated by gen_nim_smoke_report.py · NIM Smoke Battery sprint 2026-05-06</footer>
</main></body></html>""")
    return "\n".join(parts)


def render_index(results: list[dict]) -> str:
    rows = []
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for r in results:
        short = r.get("short", "?")
        badge, rec, headline, _ = classify(r)
        counts[badge] = counts.get(badge, 0) + 1
        rows.append(f"""
        <tr>
          <td><a href="{escape(short)}.html">{escape(short)}</a></td>
          <td><span class="kbd">{escape(r.get("image",""))}</span></td>
          <td><span class="badge {badge}">{badge.upper()}</span></td>
          <td><span class="badge {rec}">{rec.upper()}</span></td>
          <td>{escape(headline)}</td>
        </tr>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>NIM Smoke Battery — 2026-05-06</title>
<style>{PAGE_CSS}</style>
</head><body>
<header>
  <h1>NIM Smoke Battery — 2026-05-06</h1>
  <div class="meta">
    Host: <span class="kbd">horde@10.57.233.111</span> ·
    GPU: NVIDIA RTX PRO 6000 Blackwell (96GB) ·
    {len(results)} NIMs ·
    <span class="badge pass">PASS {counts.get("pass",0)}</span>
    <span class="badge warn">WARN {counts.get("warn",0)}</span>
    <span class="badge fail">FAIL {counts.get("fail",0)}</span>
  </div>
</header>
<main>
<p>Each NIM was smoked with the same fixed protocol:</p>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  <tr><td>Input video</td><td><span class="kbd">/Users/asotelo/Desktop/ball video high res.mp4</span> (49MB)</td></tr>
  <tr><td>System prompt</td><td><pre>You are a helpful assistant.</pre></td></tr>
  <tr><td>User prompt (reasoning ON)</td><td><pre>race car: timestamps</pre></td></tr>
  <tr><td>User prompt (reasoning OFF)</td><td><pre>temporal summary</pre></td></tr>
  <tr><td>Min FPS</td><td>8</td></tr>
  <tr><td>max_output_tokens</td><td>maximized per model</td></tr>
  <tr><td>max_pixels</td><td>maximized per model</td></tr>
  <tr><td>Temperature / Top P / Repetition Penalty</td><td>build.nvidia.com defaults per model</td></tr>
  <tr><td>Video transport</td><td>video_url first; frames fallback on 4xx; max_frames=5 fallback for caps</td></tr>
</table>

<h2>Per-NIM results</h2>
<table>
  <tr>
    <th>NIM short name</th><th>Image</th><th>Status</th><th>Recommendation</th><th>Headline</th>
  </tr>
  {''.join(rows)}
</table>
<footer>NIM Smoke Battery sprint — 2026-05-06 · Eric (lead), Bronson (smoke gate), Joyce (report) · NVIDIA</footer>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    args = ap.parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for f in sorted(indir.glob("nim_smoke_*.json")):
        try:
            d = json.loads(f.read_text())
            results.append(d)
        except json.JSONDecodeError:
            continue
    for r in results:
        short = r.get("short", "unknown")
        (outdir / f"{short}.html").write_text(render_per_nim_page(r))
    (outdir / "index.html").write_text(render_index(results))
    print(f"wrote {len(results)} per-NIM pages + index.html → {outdir}")


if __name__ == "__main__":
    main()
