# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
nemotron_finalize.py — Monitor Nemotron inference, compute metrics, patch HTML report.

Watches ~/inference_results_nemotron.json as it grows.
When the run completes (default: 40 videos), computes accuracy/recall/throughput
and injects a Nemotron-Nano-12B-v2 column into the experiment-comparison table
of the worker-safety HTML report.

Usage
-----
# Watch mode: polls results file, auto-patches HTML when done
python3 nemotron_finalize.py \\
  --results ~/inference_results_nemotron.json \\
  --html ~/.claude/docs/cookbook-recipes/worker-safety-ramos-2026-feb-4/cosmos-reason2-brev-report.html \\
  --watch

# Patch only: results already complete, just compute + patch
python3 nemotron_finalize.py \\
  --results ~/inference_results_nemotron.json \\
  --html ~/.claude/docs/cookbook-recipes/worker-safety-ramos-2026-feb-4/cosmos-reason2-brev-report.html

# Dry run: show computed metrics, skip HTML patch
python3 nemotron_finalize.py \\
  --results ~/inference_results_nemotron.json \\
  --dry-run
"""

import argparse
import json
import pathlib
import re
import sys
import time

# ── Ground truth label sets ───────────────────────────────────────────────────

HAZARDOUS_LABELS = {
    "Safe Walkway Violation",
    "Unauthorized Intervention",
    "Opened Panel Cover",
    "Carrying Overload with Forklift",
}
SAFE_LABELS = {
    "Safe Walkway",
    "Authorized Intervention",
    "Closed Panel Cover",
    "Safe Carrying",
}

# A100 @ $1.49/hr (same rate as the other experiments)
COST_PER_HOUR = 1.49

# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(records: list, inference_minutes: float = None, log_path: str = None) -> dict:
    """Compute all comparison metrics from results records."""
    total = len(records)
    classified = [r for r in records if r.get("safety_label")]
    errors = [r for r in records if r.get("cosmos_error")]

    with_gt = [r for r in classified if r.get("ground_truth")]
    correct = [r for r in with_gt if r["safety_label"] == r["ground_truth"]]
    accuracy = len(correct) / len(with_gt) * 100 if with_gt else 0.0

    gt_hazardous = [r for r in with_gt if r["ground_truth"] in HAZARDOUS_LABELS]
    gt_safe = [r for r in with_gt if r["ground_truth"] in SAFE_LABELS]

    hazard_flagged = [r for r in gt_hazardous if r.get("hazard") is True]
    safe_correct = [r for r in gt_safe if r.get("hazard") is False]

    hazard_recall = len(hazard_flagged) / len(gt_hazardous) * 100 if gt_hazardous else 0.0
    safe_specificity = len(safe_correct) / len(gt_safe) * 100 if gt_safe else 0.0

    predicted_labels = set(r["safety_label"] for r in classified)
    classes_predicted = len(predicted_labels)

    # Per-class prediction counts
    swv = sum(1 for r in classified if r["safety_label"] == "Safe Walkway Violation")
    cof = sum(1 for r in classified if r["safety_label"] == "Carrying Overload with Forklift")
    ui  = sum(1 for r in classified if r["safety_label"] == "Unauthorized Intervention")

    # Timing: try log file first, then CLI arg, then None
    infer_min = inference_minutes
    if infer_min is None and log_path:
        infer_min = _parse_log_time(log_path)

    throughput = len(classified) / infer_min if (infer_min and infer_min > 0) else None
    cost = infer_min / 60 * COST_PER_HOUR if infer_min else None

    return {
        "total": total,
        "classified": len(classified),
        "errors": len(errors),
        "with_gt": len(with_gt),
        "correct": len(correct),
        "accuracy": accuracy,
        "gt_hazardous_count": len(gt_hazardous),
        "hazard_flagged": len(hazard_flagged),
        "hazard_recall": hazard_recall,
        "gt_safe_count": len(gt_safe),
        "safe_correct": len(safe_correct),
        "safe_specificity": safe_specificity,
        "classes_predicted": classes_predicted,
        "predicted_labels": sorted(predicted_labels),
        "swv": swv,
        "cof": cof,
        "ui": ui,
        "inference_minutes": infer_min,
        "throughput": throughput,
        "cost": cost,
    }


def _parse_log_time(log_path: str) -> float | None:
    """Parse elapsed minutes from the harness DONE banner in the log file."""
    try:
        text = pathlib.Path(log_path).expanduser().read_text()
        m = re.search(r"DONE\s*[—-]\s*\d+/\d+\s+videos\s+in\s+([\d.]+)\s+min", text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


# ── Progress display ──────────────────────────────────────────────────────────

def print_progress(records: list, expected: int):
    total = len(records)
    classified = sum(1 for r in records if r.get("safety_label"))
    errors = sum(1 for r in records if r.get("cosmos_error"))
    with_gt = [r for r in records if r.get("ground_truth") and r.get("safety_label")]
    acc = (
        sum(1 for r in with_gt if r["safety_label"] == r["ground_truth"])
        / len(with_gt) * 100
        if with_gt else 0.0
    )
    print(
        f"[NEMOTRON] Progress: {total}/{expected} records  |  "
        f"classified: {classified}  errors: {errors}  |  "
        f"running accuracy: {acc:.1f}%"
    )


# ── HTML patching ─────────────────────────────────────────────────────────────

# The 15 tbody rows in order (used to map index → inject function)
_ROW_LABELS = [
    "wall_time",
    "inference_time",
    "throughput",
    "cost",
    "accuracy",
    "hazard_recall",
    "safe_specificity",
    "max_new_tokens",
    "fps",
    "videos_classified",
    "errors",
    "classes_predicted",
    "swv_count",
    "cof_count",
    "ui_count",
]

BG = 'style="background:#1A0A2E;"'  # Nemotron column background (purple-dark tint)
BG_ACC = 'style="background:#1A0A2E;"'  # same for accuracy row highlight


def _cell(content: str, extra_style: str = "") -> str:
    """Wrap content in a <td> with the Nemotron column background."""
    style = "background:#1A0A2E;"
    if extra_style:
        style += extra_style
    return f'<td style="{style}">{content}</td>'


def build_cells(m: dict) -> list[str]:
    """Return 15 HTML <td> strings, one per tbody row, in order."""
    def _tbd(label: str) -> str:
        return f'<em style="color:#888;">TBD — {label}</em>'

    def _time_cell() -> str:
        if m["inference_minutes"] is None:
            return _tbd("add --inference-minutes")
        return (
            f"<strong>~{m['inference_minutes']:.0f} min</strong>"
            f"<br><span style=\"color:var(--text-muted);font-size:11px;\">"
            f"venv+weights (one-time) + inference</span>"
        )

    def _infer_cell() -> str:
        if m["inference_minutes"] is None:
            return _tbd("add --inference-minutes")
        return f"{m['inference_minutes']:.1f} min"

    def _throughput_cell() -> str:
        if m["throughput"] is None:
            return _tbd("needs inference time")
        return f"{m['throughput']:.2f}"

    def _cost_cell() -> str:
        if m["cost"] is None:
            return _tbd("needs inference time")
        return f"${m['cost']:.2f}"

    acc_pct = m["accuracy"]
    acc_str = f"{acc_pct:.1f}% ({m['correct']}/{m['with_gt']})"
    acc_color = "#76B900" if acc_pct > 30 else "#E88A00"

    hr = m["hazard_recall"]
    hr_str = f"{hr:.1f}% ({m['hazard_flagged']}/{m['gt_hazardous_count']} GT-hazardous)"
    hr_color = "#76B900" if hr >= 85 else "#E88A00"

    ss = m["safe_specificity"]
    ss_str = f"{ss:.1f}% ({m['safe_correct']}/{m['gt_safe_count']})"
    ss_extra = " ★" if ss > 0 else ""
    ss_color = "#76B900" if ss > 20 else ("#E88A00" if ss > 0 else "var(--red)")

    cp = m["classes_predicted"]
    cp_color = "#76B900" if cp >= 6 else ("#E88A00" if cp >= 4 else "var(--red)")
    cp_extra = " ★" if cp >= 6 else ""

    err = m["errors"]
    err_str = str(err) if err == 0 else f'<span style="color:#E88A00;">{err}</span>'

    # For per-class rows: if model predicted many classes, show "mixed (N classes)"
    def _class_count(n: int, label: str) -> str:
        if cp >= 5:
            return f'<span style="color:var(--text-muted);">mixed ({cp} classes)</span>'
        return str(n)

    cells = [
        # 0 – wall time
        _cell(_time_cell()),
        # 1 – inference time
        _cell(_infer_cell()),
        # 2 – throughput
        _cell(_throughput_cell()),
        # 3 – cost
        _cell(_cost_cell()),
        # 4 – accuracy (highlighted row — keep the row style, add our cell)
        f'<td style="font-weight:700;color:{acc_color};background:#1A0A2E;">{acc_str}</td>',
        # 5 – hazard recall
        f'<td style="color:{hr_color};background:#1A0A2E;">{hr_str}</td>',
        # 6 – safe specificity
        f'<td style="color:{ss_color};background:#1A0A2E;">{ss_str}{ss_extra}</td>',
        # 7 – max_new_tokens
        _cell("256"),
        # 8 – fps
        _cell("2"),
        # 9 – videos classified
        _cell(f"{m['classified']} / {m['total']}"),
        # 10 – errors
        _cell(err_str),
        # 11 – classes predicted
        f'<td style="color:{cp_color};background:#1A0A2E;">{cp} / 8{cp_extra}</td>',
        # 12 – SWV count
        _cell(_class_count(m["swv"], "SWV")),
        # 13 – COF count
        _cell(_class_count(m["cof"], "COF")),
        # 14 – UI count
        _cell(_class_count(m["ui"], "UI")),
    ]
    return cells


def patch_html(html: str, m: dict) -> str:
    """Inject a Nemotron column into the perf-comparison-table."""

    # ── 1. Update colgroup (6 → 7 columns, rebalance widths) ─────────────────
    old_colgroup = (
        '<col style="width:18%">\n'
        '          <col style="width:16.4%">\n'
        '          <col style="width:16.4%">\n'
        '          <col style="width:16.4%">\n'
        '          <col style="width:16.4%">\n'
        '          <col style="width:16.4%">'
    )
    new_colgroup = (
        '<col style="width:15%">\n'
        '          <col style="width:14.2%">\n'
        '          <col style="width:14.2%">\n'
        '          <col style="width:14.2%">\n'
        '          <col style="width:14.2%">\n'
        '          <col style="width:14.2%">\n'
        '          <col style="width:14%">'
    )
    html = html.replace(old_colgroup, new_colgroup)

    # ── 2. Update description text to mention Nemotron ────────────────────────
    old_desc = (
        'Same 40 videos, same prompts — all Cosmos-Reason2 configurations vs. '
        'Qwen3-VL-8B as a reference model.'
    )
    new_desc = (
        'Same 40 videos, same prompts — all Cosmos-Reason2 configurations vs. '
        'Qwen3-VL-8B and Nemotron-Nano-12B-v2 as reference models.'
    )
    html = html.replace(old_desc, new_desc)

    # ── 3. Add Nemotron <th> to thead (before closing </tr> of thead) ─────────
    old_qwen_th = (
        '<th style="background:#0A1A2A;color:#76B900;">Qwen3-VL-8B ★<br>'
        '<span style="font-weight:400;font-size:11px;color:#aaa;">'
        'bfloat16 · 512 tok · fps=4</span></th>\n'
        '          </tr>'
    )
    new_nemotron_th = (
        '<th style="background:#0A1A2A;color:#76B900;">Qwen3-VL-8B ★<br>'
        '<span style="font-weight:400;font-size:11px;color:#aaa;">'
        'bfloat16 · 512 tok · fps=4</span></th>\n'
        '            <th style="background:#1A0A2E;color:#76B900;">Nemotron-Nano-12B-v2 ▲<br>'
        '<span style="font-weight:400;font-size:11px;color:#aaa;">'
        'bfloat16 · 256 tok · fps=2 · frame-seq</span></th>\n'
        '          </tr>'
    )
    html = html.replace(old_qwen_th, new_nemotron_th)

    # ── 4. Inject <td> into each tbody row ────────────────────────────────────
    cells = build_cells(m)

    # Each tbody row ends with one </tr>. We inject the new cell before it.
    # Strategy: find the tbody section, process row-by-row.
    tbody_match = re.search(
        r'(<tbody>)(.*?)(</tbody>)',
        html,
        re.DOTALL,
    )
    if not tbody_match:
        print("[WARN] Could not locate <tbody> in HTML. Row injection skipped.")
        return html

    tbody_content = tbody_match.group(2)
    rows = re.split(r'(?=<tr)', tbody_content)
    patched_rows = []
    data_row_idx = 0

    for row in rows:
        if not row.strip():
            patched_rows.append(row)
            continue
        if data_row_idx < len(cells) and '</tr>' in row:
            row = row.replace('</tr>', cells[data_row_idx] + '</tr>', 1)
            data_row_idx += 1
        patched_rows.append(row)

    new_tbody = tbody_match.group(1) + ''.join(patched_rows) + tbody_match.group(3)
    html = html[:tbody_match.start()] + new_tbody + html[tbody_match.end():]

    # ── 5. Update insight box ─────────────────────────────────────────────────
    old_insight = (
        '<div style="background:#0A1A2A;border:1px solid #336;border-radius:6px;'
        'padding:12px 16px;margin-bottom:16px;font-size:13px;color:var(--text-muted);">\n'
        '      <strong style="color:#76B900;">Qwen3-VL-8B insight:</strong>'
    )
    ss = m["safe_specificity"]
    hr = m["hazard_recall"]
    cp = m["classes_predicted"]
    acc = m["accuracy"]
    nemotron_insight = (
        f' Nemotron-Nano-12B-v2 achieved <strong style="color:#fff;">'
        f'{acc:.1f}% exact-match accuracy</strong>, '
        f'{hr:.1f}% hazard recall, and {ss:.1f}% safe specificity '
        f'across {cp}/8 predicted classes.'
        if m["inference_minutes"] else
        ' Nemotron-Nano-12B-v2 results pending — run inference and re-patch.'
    )
    new_insight = (
        '<div style="background:#0A1A2A;border:1px solid #336;border-radius:6px;'
        'padding:12px 16px;margin-bottom:16px;font-size:13px;color:var(--text-muted);">\n'
        '      <strong style="color:#76B900;">Qwen3-VL-8B insight:</strong>'
    )
    # Append Nemotron insight after the closing </div> of the existing insight box
    old_end = (
        'The 8B model is more balanced but less conservative. For a safety-critical system, '
        '<strong style="color:#E88A00;">Cosmos-Reason2 hazard recall (100%) is still preferred'
        '</strong> — missing a hazard is worse than a false alarm.\n'
        '    </div>'
    )
    new_end = (
        'The 8B model is more balanced but less conservative. For a safety-critical system, '
        '<strong style="color:#E88A00;">Cosmos-Reason2 hazard recall (100%) is still preferred'
        '</strong> — missing a hazard is worse than a false alarm.\n'
        '    </div>\n'
        '    <div style="background:#1A0A2E;border:1px solid #446;border-radius:6px;'
        'padding:12px 16px;margin-bottom:16px;font-size:13px;color:var(--text-muted);">\n'
        f'      <strong style="color:#76B900;">Nemotron-Nano-12B-v2 insight:</strong>{nemotron_insight}\n'
        '    </div>'
    )
    html = html.replace(old_end, new_end)

    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def load_results(path: str) -> list:
    try:
        return json.loads(pathlib.Path(path).expanduser().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def parse_args():
    p = argparse.ArgumentParser(
        description="Monitor Nemotron inference run and patch HTML report when complete.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results",
        default=str(pathlib.Path.home() / "inference_results_nemotron.json"),
        metavar="PATH",
        help="Path to the Nemotron inference results JSON.",
    )
    p.add_argument(
        "--html",
        default=str(
            pathlib.Path.home()
            / ".claude/docs/cookbook-recipes/worker-safety-ramos-2026-feb-4/cosmos-reason2-brev-report.html"
        ),
        metavar="PATH",
        help="Path to the HTML report file to patch.",
    )
    p.add_argument(
        "--log",
        default=str(pathlib.Path.home() / "nemotron_harness.log"),
        metavar="PATH",
        help="Path to the harness log file (used to extract elapsed time).",
    )
    p.add_argument(
        "--inference-minutes",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Total inference time in minutes (overrides log-file parsing).",
    )
    p.add_argument(
        "--expected",
        type=int,
        default=40,
        metavar="N",
        help="Expected number of videos. Watch mode completes when this count is reached.",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="Poll the results file every --poll-interval seconds and auto-patch when done.",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Seconds between polls in watch mode.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print metrics but do not write the patched HTML.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    results_path = str(pathlib.Path(args.results).expanduser())
    html_path = str(pathlib.Path(args.html).expanduser())
    log_path = str(pathlib.Path(args.log).expanduser())

    if args.watch:
        print(f"[NEMOTRON] Watch mode — polling every {args.poll_interval}s")
        print(f"[NEMOTRON] Results file: {results_path}")
        while True:
            records = load_results(results_path)
            print_progress(records, args.expected)
            if len(records) >= args.expected:
                print(f"[NEMOTRON] {args.expected} records reached — running final metrics.")
                break
            time.sleep(args.poll_interval)
    else:
        records = load_results(results_path)
        if not records:
            print(f"[ERROR] No records found at {results_path}. Is inference complete?")
            sys.exit(1)

    m = compute_metrics(records, inference_minutes=args.inference_minutes, log_path=log_path)

    # ── Print metrics summary ─────────────────────────────────────────────────
    print("\n══ NEMOTRON METRICS ═══════════════════════════════════════")
    print(f"  Videos         : {m['total']} total, {m['classified']} classified, {m['errors']} errors")
    print(f"  Accuracy       : {m['accuracy']:.1f}% ({m['correct']}/{m['with_gt']})")
    print(f"  Hazard recall  : {m['hazard_recall']:.1f}% ({m['hazard_flagged']}/{m['gt_hazardous_count']} GT-hazardous)")
    print(f"  Safe specificity: {m['safe_specificity']:.1f}% ({m['safe_correct']}/{m['gt_safe_count']} GT-safe)")
    print(f"  Classes predicted: {m['classes_predicted']}/8 → {m['predicted_labels']}")
    print(f"  SWV: {m['swv']}  COF: {m['cof']}  UI: {m['ui']}")
    if m["inference_minutes"]:
        print(f"  Inference time : {m['inference_minutes']:.1f} min")
        print(f"  Throughput     : {m['throughput']:.2f} vid/min")
        print(f"  Cost @ $1.49/hr: ${m['cost']:.2f}")
    else:
        print("  Timing         : not available (pass --inference-minutes or provide log)")
    print("══════════════════════════════════════════════════════════\n")

    if args.dry_run:
        print("[DRY RUN] HTML not modified.")
        return

    # ── Patch HTML ────────────────────────────────────────────────────────────
    html_file = pathlib.Path(html_path)
    if not html_file.exists():
        print(f"[ERROR] HTML file not found: {html_path}")
        sys.exit(1)

    html = html_file.read_text()

    # Guard against double-patching
    if "Nemotron-Nano-12B-v2" in html:
        print("[WARN] HTML already contains a Nemotron column. Skipping patch.")
        print("       Delete the existing column manually before re-running.")
        sys.exit(1)

    patched = patch_html(html, m)

    # Atomic write via temp file
    tmp = html_file.with_suffix(".tmp")
    tmp.write_text(patched)
    tmp.replace(html_file)

    print(f"[NEMOTRON] HTML patched: {html_path}")
    print("  → Open the file in a browser to verify the Nemotron column.")


if __name__ == "__main__":
    main()
