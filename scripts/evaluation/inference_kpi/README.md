# Inference KPI Micro-Benchmark Audit

This directory contains the first-gate audit runner for `inference_kpi`
micro-benchmarks. The audit is intentionally read-only with respect to model
inference: it inventories source assets, verifies endpoint availability, and
publishes traceable report artifacts before any bulk media copy or scoring run.
It also explains whether each KPI family already contains plain
question-answer pairs or needs deterministic prompts generated from structured
ground truth.

## Commands

```bash
python3 scripts/evaluation/inference_kpi/inference_kpi_audit.py audit \
  --source-tar /Users/asotelo/Downloads/batch_1.tar.gz \
  --out-dir outputs/inference-kpi-audit

python3 scripts/evaluation/inference_kpi/inference_kpi_audit.py probe \
  --out-dir outputs/inference-kpi-audit

python3 scripts/evaluation/inference_kpi/inference_kpi_audit.py report \
  --out-dir outputs/inference-kpi-audit
```

The default report is designed to be copied to
`/home/horde/inference-kpi-benchmark` and served on port `9105`.
Key outputs include `manifest.jsonl`, `manifest.sqlite`,
`schema_summary.json`, `endpoint_status.json`, `storage_plan.json`, and the
PowerPoint deck under `artifacts/`.

## Gate 1 Boundaries

- No full inference run.
- No autograder scoring run.
- No full media extraction from large archives.
- No credential values are written to artifacts.
- Local archives are indexed by member path; selected assets can be extracted in
  a later gate when the user clicks into a trace row or requests a scoring run.
