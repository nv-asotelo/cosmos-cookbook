# BYO-video model comparison report

Date: 2026-08-04

## Delivered scope

The BYO-video workflow now supports the same loaded-model baseline and hosted
candidate comparison flow in three surfaces:

- Gradio for single uploaded images or videos;
- Cosmos Reason Vite for uploads and bundled model-page examples;
- Batch Inference for selected dataset rows and reference-video uploads.

Each surface supports a one-configuration spot check and a bounded workflow
ablation matrix. Reports retain model, variant, media strategy, latency, status,
requested settings, actual parameters sent or explicitly omitted, and redacted
request-shape metadata. Credentials, raw prompt request fields, media payloads,
configured hosts, and absolute checkpoint paths are excluded from exported
artifacts. Generated model output is retained and can repeat prompt language.

## Current candidate set

Live catalog discovery returned 214 available model records and resolved the
following comparison set. The catalog response is authoritative for callable
IDs and explicit media capabilities; ID-only rows use strict, version-bound
model contracts rather than loose family-name matching.

| Role | Current model edition | Video strategy |
|---|---|---|
| Loaded baseline | Cosmos3 Nano Reasoner | Native video |
| Candidate | Qwen3.6 35B A3B | Eight deterministic sampled frames |
| Candidate | Nemotron 3 Nano Omni 30B A3B Reasoning | Native video |
| Candidate | Gemma 4 31B IT | Eight deterministic sampled frames |
| Candidate | Gemini 3.6 Flash | Eight deterministic sampled frames |
| Frontier candidate | Claude Opus 5 | Eight deterministic sampled frames |
| Frontier candidate | GPT-5.6 Sol | Eight deterministic sampled frames |
| Frontier candidate | Kimi K2.6 | Eight deterministic sampled frames |

## Smoke results

The bundled `agibot.mp4` model-page example was used throughout the Vite smoke
test with the prompt asking for the next immediate action.

| Check | Result |
|---|---|
| Catalog selection | Passed; seven requested candidate families selected, with unrelated text-only models excluded |
| Loaded Cosmos3 baseline wiring | Passed with a deterministic test double; native `video_url` request shape confirmed, but no live Cosmos3 model output was generated |
| Candidate media adaptation | Passed; Nemotron used native video and the remaining candidates used eight sampled `image_url` frames |
| Spot comparison | Passed; 8 of 8 deterministic model-variant requests completed |
| Workflow ablation | Passed; control plus system-prompt/temperature variant produced 16 of 16 deterministic model-variant results |
| Credential lifecycle | Passed; the masked input retained the credential only for the page session, explicit Clear removed it, and no credential appeared in server state, browser storage, or rendered output |
| Report redaction | Passed; no credential, prompt text, configured host, or media data URL appeared in the rendered report |
| Capability gate | Passed; a catalog-confirmed text-only model was blocked from video input before a candidate call |
| Vite parameter parity | Passed; loaded and candidate request metadata record the actual temperature, top-p, and token limit sent |

The deterministic smoke validates application wiring, request construction,
media routing, ablation fan-out, and report safety; its mock text is not a model
quality result.

## Live comparison outcome

Catalog discovery succeeded with the supplied runtime credential. Hosted
candidate execution was denied by that credential's access policy before any
generation completed. Separately, the local Vite smoke session had no live
Cosmos3 NIM attached, so the loaded baseline was exercised only with a
deterministic test double. Consequently, there are no live outputs, latency
comparisons, or defensible qualitative rankings for Qwen, Nemotron, Gemma,
Gemini, Claude, GPT, Kimi, or the Cosmos baseline in this run.

No ranking is reported from deterministic mock responses. Once a credential
with execution access is entered at runtime, the same spot or ablation matrix
will produce directly comparable redacted artifacts across all selected models.

## Automated validation

- 40 focused Python tests passed.
- Python compilation passed for setup, Gradio, shared comparison, and Batch
  Inference helpers.
- Vite server syntax checking, TypeScript checking, and production build passed.
- Source diff whitespace checks passed.
- Credential-shaped values are rejected from model IDs and workflow fields
  before loaded or candidate calls begin.
- Signed, expiring catalog context preserves capability provenance without
  retaining credential-derived state.
- Batch state, uploads, and exports use owner-only file permissions.
