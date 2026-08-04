import express from "express";
import { execFile, spawn } from "node:child_process";
import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { setMaxListeners } from "node:events";
import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildReasoningPayload,
  createReasoningStreamNormalizer,
  listReasonerModels,
  normalizeReasonerMessage,
  submitReasoning
} from "../_shared/reasonerClient.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5173);

const backend =
  process.env.INFERENCE_BACKEND ||
  (process.env.ALPAMAYO_BASE_URL ? "alpamayo" : process.env.NIM_BASE_URL ? "nim_local" : "vllm");
const backendImplementation = process.env.BACKEND_IMPLEMENTATION || process.env.BACKEND_IMPL || "";
const backendDisplayName = process.env.BACKEND_DISPLAY_NAME || backendImplementation || backend;
const isAlpamayoBackend = String(backend || "").toLowerCase() === "alpamayo";
const defaultModel =
  process.env.NIM_SERVED_MODEL_NAME ||
  process.env.MODEL_NAME ||
  process.env.MODEL_ID ||
  process.env.ALPAMAYO_MODEL_ID ||
  (isAlpamayoBackend ? "nvidia/Alpamayo-1.5-10B" : "Detecting model...");
const KNOWN_HF_MODEL_COMMITS = {
  "nvidia/Cosmos3-Nano-Reasoner": "6406357cdc32fbf8db5f51ff7992343803b06961"
};

const app = express();
app.use(express.json({ limit: "512mb" }));

const EXAMPLE_MEDIA_HOSTS = new Set(["assets.ngc.nvidia.com"]);
const DEFAULT_NIM_FRAME_FALLBACK_IMAGES = 5;
const LONG_VIDEO_MAX_IMAGES_PER_CHUNK = 5;
const REQUEST_LOG_PREFIX = "[vite-build-reason]";
const REQUEST_DIAGNOSTICS_ENABLED = String(process.env.REASONER_REQUEST_LOG || "1").toLowerCase() !== "0";
const HOSTED_COMPARE_DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1";
const HOSTED_COMPARE_MAX_MODELS = 8;
const HOSTED_COMPARE_TIMEOUT_MS = Math.max(
  10000,
  Number.parseInt(process.env.NVIDIA_HOSTED_TIMEOUT_MS || "180000", 10) || 180000
);
const HOSTED_COMPARE_FRAME_LIMIT = Math.max(
  1,
  Math.min(32, Number.parseInt(process.env.NVIDIA_HOSTED_FRAME_FALLBACK_MAX_IMAGES || "8", 10) || 8)
);
const HOSTED_CATALOG_CAPABILITY_TTL_MS = 15 * 60 * 1000;
const HOSTED_CATALOG_CAPABILITY_SIGNING_KEY = randomBytes(32);
const HOSTED_MODEL_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,239}$/;
const HOSTED_SECRET_LIKE_PATTERN = /\b(?:nvapi|sk)-[A-Za-z0-9._-]{16,}\b/i;
const CURATED_HOSTED_MODELS = Object.freeze([
  {
    id: "",
    label: "Cosmos3 Nano Reasoner",
    family: "Cosmos",
    matchPattern: "(?:^|/)cosmos3[-_.]?nano[-_.]?reasoner(?:$|[-_.:])",
    capabilities: ["video", "image", "text"],
    note: "Catalog label only; live discovery resolves the callable request ID"
  },
  {
    id: "",
    label: "Cosmos3 Super Reasoner",
    family: "Cosmos",
    matchPattern: "(?:^|/)cosmos3[-_.]?super[-_.]?reasoner(?:$|[-_.:])",
    capabilities: ["video", "image", "text"],
    note: "Catalog label only; live discovery resolves the callable request ID"
  },
  {
    id: "",
    label: "Latest Qwen",
    family: "Qwen",
    matchPattern: "(?:^|/)qwen3[._-]?6[-_.]?35b[-_.]?a3b(?:$|[-_.:])",
    capabilities: ["image", "text"],
    note: "Resolved to the current Qwen reasoning request ID during live discovery; video uses sampled image frames"
  },
  {
    id: "",
    label: "Nemotron 3 Nano Omni",
    family: "Nemotron",
    matchPattern: "(?:^|/)nemotron[-_.]?3[-_.]?nano[-_.]?omni[-_.]?30b[-_.]?a3b[-_.]?reasoning(?:$|[-_.:])",
    capabilities: ["video", "image", "text"],
    note: "Resolved from the live catalog"
  },
  {
    id: "",
    label: "Gemma 4",
    family: "Gemma",
    matchPattern: "(?:^|/)gemma[-_.]?4[-_.]?31b[-_.]?it(?:$|[-_.:])",
    capabilities: ["image", "text"],
    note: "Catalog label only; live discovery resolves the callable request ID and video uses sampled image frames"
  },
  {
    id: "",
    label: "Gemini 3.6 Flash",
    family: "Gemini",
    matchPattern: "(?:^|/)gemini[-_.]?3[._-]?6[-_.]?flash(?:$|[-_.:])",
    capabilities: ["image", "text"],
    note: "Resolved from the live catalog; video comparisons use sampled image frames"
  },
  {
    id: "",
    label: "Claude 5",
    family: "Claude",
    matchPattern: "(?:^|/)claude[-_.]?opus[-_.]?5(?:$|[-_.:])",
    capabilities: ["image", "text"],
    note: "Resolved from the live catalog; video comparisons use sampled image frames"
  },
  {
    id: "",
    label: "Claude 4.8",
    family: "Claude",
    matchPattern: "(?:^|/)claude[-_.]?opus[-_.]?4[-_.]?8(?:$|[-_.:])",
    capabilities: ["image", "text"],
    note: "Resolved from the live catalog; video comparisons use sampled image frames"
  },
  {
    id: "",
    label: "GPT 5.6",
    family: "GPT",
    matchPattern: "(?:^|/)gpt[-_.]?5[._-]?6[-_.]?sol(?:$|[-_.:])",
    capabilities: ["image", "text"],
    note: "Resolved from the live catalog; video comparisons use sampled image frames"
  },
  {
    id: "",
    label: "Kimi K2.6",
    family: "Kimi",
    matchPattern: "(?:^|/)kimi[-_.]?k2[._-]?6(?:$|[-_.:])",
    capabilities: ["image", "text"],
    note: "Resolved from the live catalog; video comparisons use sampled image frames"
  }
]);

function requestLogId(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(16).slice(2, 8)}`;
}

function elapsedMs(startedAt) {
  return Math.max(0, Date.now() - startedAt);
}

function definedValues(object) {
  return Object.fromEntries(Object.entries(object).filter(([, value]) => value !== undefined && value !== null));
}

function reasonSchemaHints(prompt = "", systemPrompt = "") {
  const text = `${prompt}\n${systemPrompt}`.toLowerCase();
  const hints = [];
  if (/\b(window_start|window_end|phase_observations|no_count_reason)\b/.test(text)) {
    hints.push("assembly_window_schema");
  }
  if (/\b(step_count|assemblies|phase_timeline|uncertain_assemblies|phase_evidence)\b/.test(text)) {
    hints.push("assembly_reducer_schema");
  }
  if (/\b(assembly|insert|insertion|part|component|fastener|screw|count)\b/.test(text)) {
    hints.push("assembly_or_counting");
  }
  if (/\b(events|event_type|timeline_summary|uncertain_events)\b/.test(text)) {
    hints.push("event_timeline_schema");
  }
  if (/\b(ego|vehicle|traffic|lane|cyclist|pedestrian|right-of-way)\b/.test(text)) {
    hints.push("av_scene");
  }
  if (/\b(robot|gripper|trajectory|bbox_2d|point_2d)\b/.test(text)) {
    hints.push("robot_spatial");
  }
  if (/\bjson\b/.test(text)) hints.push("json_requested");
  return hints;
}

function summarizeReasonParams(params = {}) {
  return definedValues({
    frames_per_second: params.frames_per_second,
    temperature: params.temperature,
    top_p: params.top_p,
    top_k: params.top_k,
    repetition_penalty: params.repetition_penalty,
    max_tokens: params.max_tokens,
    seed: params.seed
  });
}

function summarizePreparedReasonRequest(prepared) {
  return {
    model: prepared.selectedModel,
    prompt_chars: String(prepared.prompt || "").length,
    system_prompt_chars: String(prepared.systemPrompt || "").length,
    schema_hints: reasonSchemaHints(prepared.prompt, prepared.systemPrompt),
    media: prepared.media,
    media_kind: prepared.mediaKind,
    params: summarizeReasonParams(prepared.params)
  };
}

function responseContent(result) {
  return (
    result?.content ||
    result?.combined_content ||
    result?.openai?.choices?.[0]?.message?.content ||
    result?.message ||
    ""
  );
}

function summarizeReasoningResult(result) {
  const choice = result?.openai?.choices?.[0] || {};
  const message = choice.message || {};
  const content = responseContent(result);
  const reasoning = result?.reasoning || message.reasoning_content || "";
  return {
    status: result?.status,
    schema: result?.schema,
    model: result?.openai?.model,
    finish_reason: choice.finish_reason,
    usage: result?.openai?.usage || result?.usage,
    content_chars: String(content || "").length,
    reasoning_chars: String(reasoning || "").length,
    media: result?.media
  };
}

function logReasonRequest(requestId, event, fields = {}) {
  if (!REQUEST_DIAGNOSTICS_ENABLED) return;
  console.log(`${REQUEST_LOG_PREFIX} request ${requestId} ${event} ${JSON.stringify(fields)}`);
}

function logReasonError(requestId, event, error, fields = {}) {
  if (!REQUEST_DIAGNOSTICS_ENABLED) return;
  console.warn(
    `${REQUEST_LOG_PREFIX} request ${requestId} ${event} ${JSON.stringify({
      ...fields,
      error: error instanceof Error ? error.message : String(error || "unknown error")
    })}`
  );
}

function normalizeHostedEndpoint(configured, defaultPath) {
  const parsed = new URL(String(configured || "").trim());
  const allowInsecure = String(process.env.NVIDIA_HOSTED_ALLOW_HTTP || "0") === "1";
  const loopback = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);
  if (
    parsed.protocol !== "https:" &&
    !(allowInsecure && parsed.protocol === "http:" && loopback.has(parsed.hostname.toLowerCase()))
  ) {
    throw new Error("Hosted comparison endpoint must use HTTPS");
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error("Hosted comparison endpoint must not contain credentials, a query, or a fragment");
  }
  if (defaultPath && !parsed.pathname.match(/\/(?:models|chat\/completions)\/?$/i)) {
    parsed.pathname = `${parsed.pathname.replace(/\/$/, "")}/${defaultPath.replace(/^\//, "")}`;
  }
  parsed.search = "";
  parsed.hash = "";
  return parsed.toString().replace(/\/$/, "");
}

function hostedInferenceEndpointUrl() {
  const configured =
    process.env.NVIDIA_HOSTED_INFERENCE_URL ||
    process.env.HOSTED_OPENAI_INFERENCE_URL ||
    process.env.NVIDIA_HOSTED_BASE_URL ||
    HOSTED_COMPARE_DEFAULT_BASE_URL;
  return normalizeHostedEndpoint(configured, "chat/completions");
}

function hostedCatalogEndpointUrl() {
  const configured =
    process.env.NVIDIA_HOSTED_CATALOG_URL ||
    process.env.HOSTED_OPENAI_CATALOG_URL ||
    process.env.NVIDIA_HOSTED_BASE_URL ||
    HOSTED_COMPARE_DEFAULT_BASE_URL;
  return normalizeHostedEndpoint(configured, "models");
}

function takeHostedCredential(body) {
  const runtimeKey = typeof body?.apiKey === "string" ? body.apiKey.trim() : "";
  if (body && Object.prototype.hasOwnProperty.call(body, "apiKey")) delete body.apiKey;
  if (!runtimeKey || /[\r\n]/.test(runtimeKey)) return "";
  return runtimeKey;
}

function hostedModelIdContainsSecret(value, credential = "") {
  const modelId = typeof value === "string" ? value : "";
  const exactCredential = String(credential || "");
  return Boolean((exactCredential && modelId.includes(exactCredential)) || HOSTED_SECRET_LIKE_PATTERN.test(modelId));
}

function isValidHostedModelId(value) {
  return typeof value === "string" && !value.includes("://") && HOSTED_MODEL_ID_PATTERN.test(value);
}

function hostedProviderId() {
  return "hosted_endpoint";
}

function hostedModelFamily(modelId) {
  const id = String(modelId || "").toLowerCase();
  if (id.includes("gemini")) return "Gemini";
  if (id.includes("gemma")) return "Gemma";
  if (id.includes("claude") || id.includes("anthropic")) return "Claude";
  if (id.includes("gpt") || id.includes("openai")) return "GPT";
  if (id.includes("qwen")) return "Qwen";
  if (id.includes("nemotron")) return "Nemotron";
  if (id.includes("cosmos")) return "Cosmos";
  if (id.includes("kimi") || id.includes("moonshot")) return "Kimi";
  return "Other";
}

function sanitizeCapabilities(value, fallback = ["text"]) {
  const supported = new Set(["video", "image", "text"]);
  const capabilities = (Array.isArray(value) ? value : [])
    .map((item) => String(item || "").toLowerCase())
    .filter((item) => supported.has(item));
  return capabilities.length > 0 ? Array.from(new Set(capabilities)) : fallback;
}

function inferredHostedCapabilities(modelId) {
  const normalizedId = String(modelId || "").toLowerCase();
  const curated = curatedModelForDiscoveredId(normalizedId);
  if (curated) return [...curated.capabilities];
  // Unknown request IDs are text-only until the user explicitly chooses the
  // sampled-frame strategy for a video comparison.
  return ["text"];
}

function catalogModelCapabilityProfile(model, modelId) {
  const explicitFields = [
    "capabilities",
    "input_modalities",
    "modalities",
    "supports_video",
    "supports_image",
    "supports_vlm",
    "supports_vision",
    "supports_multimodal",
    "supports_text"
  ];
  const hasExplicitMetadata =
    model &&
    typeof model === "object" &&
    explicitFields.some((field) => Object.prototype.hasOwnProperty.call(model, field));
  const raw = [];
  const collectCapabilityValues = (value) => {
    if (Array.isArray(value)) {
      value.forEach(collectCapabilityValues);
      return;
    }
    if (value && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) {
        if (item === true) raw.push(String(key).toLowerCase());
        else collectCapabilityValues(item);
      }
      return;
    }
    if (typeof value === "string") raw.push(value.toLowerCase());
  };
  collectCapabilityValues(model?.capabilities);
  collectCapabilityValues(model?.input_modalities);
  collectCapabilityValues(model?.modalities);
  const translated = [];
  if (raw.some((value) => value.includes("video"))) translated.push("video");
  if (raw.some((value) => value.includes("image") || value.includes("vision"))) translated.push("image");
  if (raw.some((value) => value.includes("text") || value.includes("chat"))) translated.push("text");
  if (model?.supports_video === true) translated.push("video");
  if (
    model?.supports_image === true ||
    model?.supports_vlm === true ||
    model?.supports_vision === true ||
    model?.supports_multimodal === true
  ) {
    translated.push("image");
  }
  if (model?.supports_text === true) translated.push("text");
  if (hasExplicitMetadata) {
    return {
      capabilities: sanitizeCapabilities(translated, ["text"]),
      source: "live_catalog"
    };
  }
  const confirmed = Boolean(curatedModelForDiscoveredId(modelId));
  return {
    capabilities: inferredHostedCapabilities(modelId),
    source: confirmed ? "confirmed_contract" : "unconfirmed"
  };
}

function modelMatchesCurated(modelId, curated) {
  const id = String(modelId || "").toLowerCase();
  if (curated.id && (id === curated.id.toLowerCase() || id.endsWith(`/${curated.id.toLowerCase()}`))) return true;
  if (!curated.matchPattern) return false;
  try {
    return new RegExp(curated.matchPattern, "i").test(id);
  } catch {
    return false;
  }
}

function curatedModelForDiscoveredId(modelId) {
  return CURATED_HOSTED_MODELS.find((candidate) => modelMatchesCurated(modelId, candidate)) || null;
}

function redactSecretText(value, secrets = []) {
  let redacted = String(value || "");
  for (const secret of Array.isArray(secrets) ? secrets : [secrets]) {
    const normalized = String(secret || "");
    if (normalized) redacted = redacted.split(normalized).join("<redacted-key>");
  }
  return redacted
    .replace(/\bBearer\s+[^\s,;]+/gi, "Bearer <redacted>")
    .replace(/\b(?:nvapi|sk)-[A-Za-z0-9._-]+\b/gi, "<redacted-key>")
    .replace(/https?:\/\/[^\s"'<>]+/gi, "<configured-endpoint>")
    .replace(/\/(?:Users|home|tmp|var\/tmp|workspace)\/[^\s"'<>]+/g, "<redacted-path>")
    .replace(/[A-Za-z]:\\[^\r\n"'<>]+/g, "<redacted-path>")
    .slice(0, 4000);
}

function redactSecretsDeep(value, secrets = []) {
  if (typeof value === "string") return redactSecretText(value, secrets);
  if (Array.isArray(value)) return value.map((item) => redactSecretsDeep(item, secrets));
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [redactSecretText(key, secrets), redactSecretsDeep(item, secrets)])
  );
}

function textFingerprint(value) {
  return createHash("sha256").update(String(value || ""), "utf8").digest("hex").slice(0, 16);
}

function sanitizeLabel(value, fallback = "") {
  const label = String(value || "").replace(/[\u0000-\u001f\u007f]/g, " ").trim();
  return (label || fallback).slice(0, 240);
}

function reportSafeModelLabel(value, fallback = "loaded-model") {
  const raw = sanitizeLabel(value, fallback);
  if (raw.startsWith("file://") || path.isAbsolute(raw)) {
    return path.basename(raw.replace(/^file:\/\//, "").replace(/[\\/]+$/, "")) || fallback;
  }
  return raw;
}

function hostedAuthHeaders(credential, scope) {
  const prefix = scope === "catalog" ? "NVIDIA_HOSTED_CATALOG" : "NVIDIA_HOSTED_INFERENCE";
  const mode = String(process.env[`${prefix}_AUTH_MODE`] || process.env.NVIDIA_HOSTED_AUTH_MODE || "bearer").toLowerCase();
  if (mode === "header" || mode === "api-key" || mode === "api_key") {
    const headerName = String(
      process.env[`${prefix}_API_KEY_HEADER`] || process.env.NVIDIA_HOSTED_API_KEY_HEADER || "x-api-key"
    ).trim();
    if (!/^[A-Za-z0-9-]+$/.test(headerName)) throw new Error("Configured hosted API-key header name is invalid");
    return { [headerName]: credential };
  }
  const scheme = String(process.env[`${prefix}_AUTH_SCHEME`] || process.env.NVIDIA_HOSTED_AUTH_SCHEME || "Bearer").trim();
  if (!/^[A-Za-z][A-Za-z0-9._-]*$/.test(scheme)) throw new Error("Configured hosted auth scheme is invalid");
  return { Authorization: `${scheme} ${credential}` };
}

async function fetchHosted(endpointUrl, credential, scope, options = {}, timeoutMs = HOSTED_COMPARE_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(endpointUrl, {
      ...options,
      headers: {
        Accept: "application/json",
        ...hostedAuthHeaders(credential, scope),
        ...(options.headers || {})
      },
      redirect: "error",
      signal: controller.signal
    });
  } finally {
    clearTimeout(timer);
  }
}

async function hostedErrorMessage(response, credential) {
  const raw = await response.text().catch(() => "");
  let detail = raw;
  try {
    const data = raw ? JSON.parse(raw) : null;
    detail = data?.error?.message || data?.message || raw;
  } catch {}
  return redactSecretText(detail || `Hosted model returned HTTP ${response.status}`, credential);
}

function curatedHostedModels() {
  return CURATED_HOSTED_MODELS.map((model, index) => {
    const { matchPattern: _matchPattern, ...publicModel } = model;
    return {
      ...publicModel,
      catalog_key: model.id || `catalog-label-${index}-${model.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
      available: false,
      curated: true,
      capability_source: "confirmed_contract",
      discovery_required: true,
      recommended: false
    };
  });
}

function encodeHostedCatalogCapabilityToken(models) {
  const payload = {
    expires_at: Date.now() + HOSTED_CATALOG_CAPABILITY_TTL_MS,
    models: (Array.isArray(models) ? models : [])
      .filter((model) => model?.available && isValidHostedModelId(model.id))
      .map((model) => ({
        id: model.id,
        capabilities: sanitizeCapabilities(model.capabilities, ["text"]),
        source: ["live_catalog", "confirmed_contract", "unconfirmed"].includes(model.capability_source)
          ? model.capability_source
          : "unconfirmed"
      }))
  };
  const encoded = Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
  const signature = createHmac("sha256", HOSTED_CATALOG_CAPABILITY_SIGNING_KEY)
    .update(encoded, "utf8")
    .digest("base64url");
  return `${encoded}.${signature}`;
}

function decodeHostedCatalogCapabilityToken(value) {
  const token = typeof value === "string" ? value.trim() : "";
  if (!token) return new Map();
  if (token.length > 262144) throw new Error("Catalog capability token is invalid");
  const [encoded, signature, ...extra] = token.split(".");
  if (!encoded || !signature || extra.length > 0) throw new Error("Catalog capability token is invalid");
  const expected = createHmac("sha256", HOSTED_CATALOG_CAPABILITY_SIGNING_KEY)
    .update(encoded, "utf8")
    .digest();
  let actual;
  try {
    actual = Buffer.from(signature, "base64url");
  } catch {
    throw new Error("Catalog capability token is invalid");
  }
  if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) {
    throw new Error("Catalog capability token is invalid");
  }
  let payload;
  try {
    payload = JSON.parse(Buffer.from(encoded, "base64url").toString("utf8"));
  } catch {
    throw new Error("Catalog capability token is invalid");
  }
  if (!Number.isFinite(Number(payload?.expires_at)) || Number(payload.expires_at) <= Date.now()) {
    throw new Error("Catalog capability token has expired");
  }
  const profiles = new Map();
  for (const model of Array.isArray(payload?.models) ? payload.models : []) {
    if (!isValidHostedModelId(model?.id)) continue;
    const source = ["live_catalog", "confirmed_contract", "unconfirmed"].includes(model?.source)
      ? model.source
      : "unconfirmed";
    profiles.set(model.id, {
      capabilities: sanitizeCapabilities(model.capabilities, ["text"]),
      source
    });
  }
  return profiles;
}

function discoveredHostedModels(data, credential = "") {
  const upstream = Array.isArray(data)
    ? data
    : Array.isArray(data?.data)
      ? data.data
      : Array.isArray(data?.models)
        ? data.models
        : Array.isArray(data?.items)
          ? data.items
          : [];
  const discovered = upstream
    .map((model) => {
      const rawId = typeof model === "string" ? model : model?.id || model?.name;
      const id = typeof rawId === "string" ? rawId.trim() : "";
      if (!isValidHostedModelId(id) || hostedModelIdContainsSecret(id, credential)) return null;
      if (
        ["embed", "embedding", "rerank", "retrieval", "tts", "asr", "whisper", "parakeet", "ocr"].some(
          (token) => id.toLowerCase().includes(token)
        )
      ) {
        return null;
      }
      const curated = curatedModelForDiscoveredId(id);
      const capabilityProfile = catalogModelCapabilityProfile(model, id);
      return {
        id,
        label: curated?.label || id,
        family: hostedModelFamily(id),
        capabilities: capabilityProfile.capabilities,
        capability_source: capabilityProfile.source,
        note: curated?.note || "Discovered from the live hosted catalog",
        created: Number.isFinite(Number(model?.created)) ? Number(model.created) : null,
        owned_by: sanitizeLabel(model?.owned_by),
        available: true,
        curated: Boolean(curated),
        discovery_required: false,
        recommended: false
      };
    })
    .filter(Boolean);

  const targetLabelsByFamily = new Map([
    ["Qwen", ["Latest Qwen"]],
    ["Nemotron", ["Nemotron 3 Nano Omni"]],
    ["Gemma", ["Gemma 4"]],
    ["Gemini", ["Gemini 3.6 Flash"]],
    ["Claude", ["Claude 5", "Claude 4.8"]],
    ["GPT", ["GPT 5.6"]],
    ["Kimi", ["Kimi K2.6"]]
  ]);
  const routePreference = (modelId) => {
    const id = String(modelId || "").toLowerCase();
    if (id.startsWith("openai/openai/") || id.startsWith("nvidia/nvidia/") || id.startsWith("nvidia/qwen/")) return 4;
    if (id.startsWith("azure/")) return 3;
    if (id.startsWith("gcp/")) return 3;
    if (id.startsWith("aws/")) return 2;
    return 0;
  };
  for (const [family, preferredLabels] of targetLabelsByFamily) {
    let candidates = [];
    for (const label of preferredLabels) {
      candidates = discovered.filter((model) => model.family === family && model.label === label);
      if (candidates.length > 0) break;
    }
    candidates.sort((left, right) => {
      const createdDelta = Number(right.created || 0) - Number(left.created || 0);
      const routeDelta = routePreference(right.id) - routePreference(left.id);
      return (
        createdDelta ||
        routeDelta ||
        right.id.localeCompare(left.id, undefined, { numeric: true, sensitivity: "base" })
      );
    });
    if (candidates[0]) candidates[0].recommended = true;
  }

  const matchedLabels = new Set(
    discovered
      .map((model) => (model.curated ? model.label : curatedModelForDiscoveredId(model.id)?.label))
      .filter(Boolean)
  );
  for (const [index, curated] of CURATED_HOSTED_MODELS.entries()) {
    if (matchedLabels.has(curated.label)) continue;
    discovered.push({
      ...curated,
      catalog_key: curated.id || `catalog-label-${index}-${curated.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
      created: null,
      owned_by: "",
      available: false,
      curated: true,
      capability_source: "confirmed_contract",
      discovery_required: true,
      recommended: false
    });
  }

  return discovered.sort((left, right) => {
    if (left.recommended !== right.recommended) return left.recommended ? -1 : 1;
    if (left.available !== right.available) return left.available ? -1 : 1;
    return `${left.family}/${left.id || left.label}`.localeCompare(`${right.family}/${right.id || right.label}`, undefined, {
      numeric: true,
      sensitivity: "base"
    });
  });
}
const FRAME_EXTRACTOR_PY = String.raw`
import json
import os
import sys

import av

video_path, output_dir, fps_raw = sys.argv[1], sys.argv[2], sys.argv[3]
max_frames = int(float(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] else 0
fps = float(fps_raw)
container = av.open(video_path)
stream = container.streams.video[0]
frame_rate = float(stream.average_rate) if stream.average_rate else 25.0
total_frames = int(stream.frames or 0)
if total_frames <= 0:
    duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
    total_frames = max(1, int(duration_s * frame_rate))
interval = max(1, int(round(frame_rate / fps)))
target_indices = set(range(0, total_frames, interval))
if max_frames > 0 and len(target_indices) > max_frames:
    ordered = sorted(target_indices)
    if max_frames == 1:
        target_indices = {ordered[0]}
    else:
        target_indices = {
            ordered[int(round(i * (len(ordered) - 1) / (max_frames - 1)))]
            for i in range(max_frames)
        }
paths = []
for index, frame in enumerate(container.decode(stream)):
    if index in target_indices:
        image = frame.to_image().convert("RGB")
        frame_path = os.path.join(output_dir, f"frame_{len(paths):05d}.jpg")
        image.save(frame_path, format="JPEG", quality=80)
        paths.append(frame_path)
        if len(paths) >= len(target_indices):
            break
container.close()
print(json.dumps({"count": len(paths), "fps": fps, "frames": paths}))
`;
const LONG_FRAME_EXTRACTOR_PY = String.raw`
import json
import math
import os
import sys

import av

video_path, output_dir, fps_raw, max_raw, width_raw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
fps = max(0.1, float(fps_raw))
max_frames = int(float(max_raw)) if max_raw else 0
target_width = int(float(width_raw)) if width_raw else 768

def emit_progress(phase, **payload):
    payload["phase"] = phase
    print("PROGRESS " + json.dumps(payload), file=sys.stderr, flush=True)

def fmt(seconds):
    seconds = max(0.0, float(seconds or 0.0))
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:05.2f}"

container = av.open(video_path)
stream = container.streams.video[0]
frame_rate = float(stream.average_rate) if stream.average_rate else 25.0
duration_s = 0.0
if container.duration:
    duration_s = float(container.duration / av.time_base)
elif stream.duration:
    duration_s = float(stream.duration * stream.time_base)
elif stream.frames and frame_rate:
    duration_s = float(stream.frames) / frame_rate

target_count = max(1, int(math.ceil(duration_s * fps))) if duration_s > 0 else 1
target_times = [index / fps for index in range(target_count)]
if duration_s > 0 and (not target_times or target_times[-1] < duration_s - (0.5 / fps)):
    target_times.append(duration_s)
target_times = [min(duration_s, value) if duration_s > 0 else value for value in target_times]
if max_frames > 0 and len(target_times) > max_frames:
    if max_frames == 1:
        target_times = [target_times[0]]
    else:
        target_times = [
            target_times[int(round(index * (len(target_times) - 1) / (max_frames - 1)))]
            for index in range(max_frames)
        ]

emit_progress(
    "opened",
    duration_s=duration_s,
    source_fps=frame_rate,
    source_width=getattr(stream, "width", 0) or 0,
    source_height=getattr(stream, "height", 0) or 0,
    target_frames=len(target_times),
)

frames = []
next_target = 0
source_width = 0
source_height = 0
last_ts = 0.0
progress_stride = max(1, len(target_times) // 20)
for index, frame in enumerate(container.decode(stream)):
    if next_target >= len(target_times):
        break
    ts = float(frame.pts * frame.time_base) if frame.pts is not None else (float(index) / frame_rate if frame_rate else 0.0)
    last_ts = ts
    if ts + (0.5 / max(frame_rate, 1.0)) < target_times[next_target]:
        continue
    image = frame.to_image().convert("RGB")
    source_width, source_height = image.size
    if target_width > 0 and image.width > target_width:
        ratio = target_width / float(image.width)
        image = image.resize((target_width, max(1, int(round(image.height * ratio)))))
    frame_path = os.path.join(output_dir, f"frame_{len(frames):05d}.jpg")
    image.save(frame_path, format="JPEG", quality=78, optimize=True)
    frames.append({
        "path": frame_path,
        "index": index,
        "timestamp": ts,
        "timestamp_text": fmt(ts),
        "target_timestamp": target_times[next_target],
        "target_timestamp_text": fmt(target_times[next_target])
    })
    next_target += 1
    if len(frames) == 1 or len(frames) == len(target_times) or len(frames) % progress_stride == 0:
        emit_progress(
            "decoded",
            extracted=len(frames),
            target_frames=len(target_times),
            timestamp=ts,
            timestamp_text=fmt(ts),
            percent=(len(frames) / max(1, len(target_times))) * 100.0,
        )

container.close()
if duration_s <= 0:
    duration_s = last_ts
emit_progress("complete", extracted=len(frames), target_frames=len(target_times), duration_s=duration_s)
print(json.dumps({
    "duration_s": duration_s,
    "source_fps": frame_rate,
    "source_width": source_width,
    "source_height": source_height,
    "sample_fps": fps,
    "count": len(frames),
    "frames": frames
}))
`;
const PLANNING_FRAME_EXTRACTOR_PY = String.raw`
import json
import os
import sys

import av

video_path, output_dir, targets_raw, width_raw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
targets = json.loads(targets_raw)
target_width = int(float(width_raw)) if width_raw else 768

def fmt(seconds):
    seconds = max(0.0, float(seconds or 0.0))
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:05.2f}"

container = av.open(video_path)
stream = container.streams.video[0]
frame_rate = float(stream.average_rate) if stream.average_rate else 25.0
sorted_targets = sorted([(max(0.0, float(item.get("seconds", 0.0))), index) for index, item in enumerate(targets)])
results = [None for _ in targets]
next_target = 0
last_image = None
last_ts = 0.0

def save_image(image, target_index, actual_ts):
    if target_width > 0 and image.width > target_width:
        ratio = target_width / float(image.width)
        image = image.resize((target_width, max(1, int(round(image.height * ratio)))))
    frame_path = os.path.join(output_dir, f"planning_frame_{target_index:03d}.jpg")
    image.save(frame_path, format="JPEG", quality=82, optimize=True)
    results[target_index] = {
        "path": frame_path,
        "timestamp": actual_ts,
        "timestamp_text": fmt(actual_ts),
        "target_timestamp": targets[target_index].get("seconds", 0.0),
        "target_timestamp_text": targets[target_index].get("time") or fmt(targets[target_index].get("seconds", 0.0)),
    }

for index, frame in enumerate(container.decode(stream)):
    if next_target >= len(sorted_targets):
        break
    ts = float(frame.pts * frame.time_base) if frame.pts is not None else (float(index) / frame_rate if frame_rate else 0.0)
    image = frame.to_image().convert("RGB")
    last_image = image
    last_ts = ts
    half_frame = 0.5 / max(frame_rate, 1.0)
    while next_target < len(sorted_targets) and ts + half_frame >= sorted_targets[next_target][0]:
        _, target_index = sorted_targets[next_target]
        save_image(image, target_index, ts)
        next_target += 1

if last_image is not None:
    while next_target < len(sorted_targets):
        _, target_index = sorted_targets[next_target]
        save_image(last_image, target_index, last_ts)
        next_target += 1

container.close()
print(json.dumps({"count": len([item for item in results if item]), "frames": results}))
`;

function dataUrlToBuffer(dataUrl) {
  const match = String(dataUrl || "").match(/^data:([^;,]+)?(;base64)?,(.*)$/s);
  if (!match) throw new Error("Media payload is not a data URL");
  const isBase64 = Boolean(match[2]);
  return {
    mime: match[1] || "application/octet-stream",
    buffer: isBase64 ? Buffer.from(match[3], "base64") : Buffer.from(decodeURIComponent(match[3]), "utf8")
  };
}

async function mediaSourceToBuffer(mediaSource) {
  const source = String(mediaSource || "");
  if (/^https?:\/\//i.test(source)) {
    const upstream = await fetch(source);
    if (!upstream.ok) {
      throw new Error(`Video URL fetch failed with HTTP ${upstream.status}`);
    }
    return {
      mime: upstream.headers.get("content-type")?.split(";")[0] || "application/octet-stream",
      buffer: Buffer.from(await upstream.arrayBuffer())
    };
  }
  return dataUrlToBuffer(source);
}

function pythonForFrames() {
  if (process.env.FRAME_EXTRACTOR_PYTHON) return process.env.FRAME_EXTRACTOR_PYTHON;
  if (process.env.PYTHON) return process.env.PYTHON;
  const hordePython = "/home/horde/cosmos-reason2/.venv/bin/python3";
  return existsSync(hordePython) ? hordePython : "python3";
}

function usesNativeVideoUrl(model) {
  const mid = String(model || "").toLowerCase();
  const mode = String(process.env.REASONER_MEDIA_MODE || "auto").toLowerCase();
  if (mode === "video_url") return true;
  if (mode === "frames") return false;
  if (isAlpamayoBackend) return true;
  if (backend === "nim_local" && !mid.includes("cosmos-reason1")) return true;
  return (
    mid.includes("qwen3-vl") ||
    mid.includes("qwen3vl") ||
    mid.includes("qwen") ||
    mid.includes("cosmos3") ||
    mid.includes("cosmos-3") ||
    mid.includes("cosmos-reason2") ||
    mid.includes("cosmos-reason-2") ||
    mid.includes("alpamayo") ||
    mid.includes("nemotron")
  );
}

function positiveIntegerEnv(name, fallback) {
  const parsed = Number.parseInt(process.env[name] || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function isVlaModel(model) {
  const mid = String(model || "").toLowerCase();
  return isAlpamayoBackend || mid.includes("alpamayo") || mid.includes("reasoningvla");
}

function vlaRuntimeInfo(selectedModel, modelInfo) {
  if (!isVlaModel(selectedModel) && !isVlaModel(modelInfo?.id)) return null;
  const framesPerVideo = positiveIntegerEnv("ALPAMAYO_FRAMES", 4);
  const maxDecodedVideoFrames = positiveIntegerEnv("ALPAMAYO_MAX_DECODED_VIDEO_FRAMES", 96);
  return {
    type: "VLA",
    family: "Alpamayo",
    adapter: "alpamayo_openai",
    current_mode: "vqa_text_generation",
    model_id: selectedModel || modelInfo?.id || defaultModel,
    backbone: "Cosmos Reason2-8B VLM",
    input_shape: "OpenAI chat/completions with video_url or image_url plus text prompt",
    frames_per_video: framesPerVideo,
    max_decoded_video_frames: maxDecodedVideoFrames,
    prompt_role: "VQA/caption question over sampled frames",
    full_trajectory_mode: false,
    clip_guidance: `Use short, front-loaded clips. The adapter samples ${framesPerVideo} frame(s) from the first ${maxDecodedVideoFrames} decoded video frame(s) unless ALPAMAYO_FRAMES or ALPAMAYO_MAX_DECODED_VIDEO_FRAMES is changed.`
  };
}

function frameFallbackLimit() {
  const raw = process.env.REASONER_FRAME_FALLBACK_MAX_IMAGES || process.env.NIM_MAX_IMAGES_PER_PROMPT || "";
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_NIM_FRAME_FALLBACK_IMAGES;
}

function frameFallbackAllowed() {
  if (process.env.REASONER_ENABLE_FRAME_FALLBACK === "1") return true;
  if (process.env.REASONER_DISABLE_FRAME_FALLBACK === "1") return false;
  return !isAlpamayoBackend;
}

function nativeVideoFallbackMessage(message) {
  return /video resolution or format not supported|cuvid|getdecodercaps|reconfiguredecoder|handlevideosequence|pynvvideocodec|at most \d+ image/i.test(
    String(message || "")
  );
}

function shouldRetryWithFrameFallback(prepared, resultOrError) {
  const message =
    resultOrError?.message ||
    resultOrError?.raw?.error?.message ||
    (resultOrError instanceof Error ? resultOrError.message : "");
  return (
    prepared?.media?.mode === "video_url" &&
    frameFallbackAllowed() &&
    nativeVideoFallbackMessage(message)
  );
}

function runFrameExtractor(videoPath, outputDir, framesPerSecond, maxFrames) {
  const fps = Number.isFinite(Number(framesPerSecond)) ? Number(framesPerSecond) : 2;
  const cap = Number.isFinite(Number(maxFrames)) && Number(maxFrames) > 0 ? String(Math.floor(Number(maxFrames))) : "";
  return new Promise((resolve, reject) => {
    const child = spawn(pythonForFrames(), ["-", videoPath, outputDir, String(fps), cap], {
      stdio: ["pipe", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Frame extractor exited with ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(FRAME_EXTRACTOR_PY);
  });
}

async function extractFrameDataUrls(mediaDataUrl, framesPerSecond, maxFrames) {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "reason-vite-frames-"));
  try {
    const { buffer } = await mediaSourceToBuffer(mediaDataUrl);
    const videoPath = path.join(tempDir, "input.mp4");
    const frameDir = path.join(tempDir, "frames");
    await fs.mkdir(frameDir);
    await fs.writeFile(videoPath, buffer);
    const info = await runFrameExtractor(videoPath, frameDir, framesPerSecond, maxFrames);
    const frames = [];
    for (const framePath of info.frames || []) {
      const frame = await fs.readFile(framePath);
      frames.push(`data:image/jpeg;base64,${frame.toString("base64")}`);
    }
    if (frames.length === 0) throw new Error("Could not extract frames from video");
    return { tempDir, frames, frameCount: frames.length, fps: info.fps || framesPerSecond };
  } catch (error) {
    await fs.rm(tempDir, { recursive: true, force: true });
    throw error;
  }
}

function timeTextToSeconds(value) {
  const text = String(value || "").trim();
  const match = text.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (!match) return Number(value) || 0;
  return Number(match[1]) * 60 + Number(match[2]);
}

function runPlanningFrameExtractor(videoPath, outputDir, targets, targetWidth = 768) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonForFrames(),
      ["-", videoPath, outputDir, JSON.stringify(targets), String(targetWidth || 768)],
      { stdio: ["pipe", "pipe", "pipe"] }
    );
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Planning frame extractor exited with ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(PLANNING_FRAME_EXTRACTOR_PY);
  });
}

async function extractPlanningFrameDataUrls(mediaSource, planningFrames, mediaKind) {
  const frames = planningFrames.map((frame, index) => ({
    index,
    phase: String(frame.phase || `Frame ${index + 1}`),
    mode: String(frame.renderMode || frame.mode || "raw"),
    time: String(frame.time || "00:00.00"),
    seconds: timeTextToSeconds(frame.time || 0)
  }));

  if (mediaKind === "image") {
    return {
      tempDir: null,
      frames: frames.map((frame) => ({
        ...frame,
        dataUrl: mediaSource,
        actualTime: frame.time
      }))
    };
  }

  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "reason-vite-planning-"));
  try {
    const { buffer } = await mediaSourceToBuffer(mediaSource);
    const videoPath = path.join(tempDir, "input.mp4");
    const frameDir = path.join(tempDir, "frames");
    await fs.mkdir(frameDir);
    await fs.writeFile(videoPath, buffer);
    const extracted = await runPlanningFrameExtractor(videoPath, frameDir, frames, 768);
    const outputFrames = [];
    for (const [index, frame] of frames.entries()) {
      const extractedFrame = extracted.frames?.[index];
      if (!extractedFrame?.path) throw new Error(`Could not extract ${frame.phase} frame at ${frame.time}`);
      const bytes = await fs.readFile(extractedFrame.path);
      outputFrames.push({
        ...frame,
        dataUrl: `data:image/jpeg;base64,${bytes.toString("base64")}`,
        actualTime: extractedFrame.timestamp_text || frame.time
      });
    }
    return { tempDir, frames: outputFrames };
  } catch (error) {
    await fs.rm(tempDir, { recursive: true, force: true });
    throw error;
  }
}

function formatTimestamp(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function formatShortDuration(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  if (value < 1) return "<1s";
  if (value < 60) return `${Math.round(value)}s`;
  const minutes = Math.floor(value / 60);
  const remainder = Math.round(value - minutes * 60);
  return `${minutes}m ${String(remainder).padStart(2, "0")}s`;
}

function positiveNumber(value, fallback = null) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function boundedInteger(value, fallback, min, max) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  const base = Number.isFinite(parsed) ? parsed : fallback;
  return Math.max(min, Math.min(max, base));
}

function optionalPromptText(value) {
  const text = typeof value === "string" ? value.trim() : "";
  return text || null;
}

function longVideoFrameLimit() {
  const parsed = Number.parseInt(process.env.REASONER_LONG_MAX_FRAMES || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 720;
}

function longVideoMaxConcurrency() {
  const parsed = Number.parseInt(process.env.REASONER_LONG_MAX_CONCURRENCY || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? Math.min(32, parsed) : 16;
}

function longVideoRequestOptions(body = {}) {
  const chunking = body.chunking && typeof body.chunking === "object" ? body.chunking : {};
  const reducer = body.reducer && typeof body.reducer === "object" ? body.reducer : {};
  return {
    chunkSize:
      chunking.framesPerChunk ??
      chunking.chunkSize ??
      body.framesPerChunk ??
      body.chunkSize,
    overlap:
      chunking.overlapFrames ??
      chunking.overlap ??
      body.chunkOverlap ??
      body.overlap,
    frameBudget:
      chunking.frameBudget ??
      chunking.maxFrames ??
      body.frameBudget ??
      body.maxFrames,
    reducerMode:
      reducer.mode ??
      reducer.reducerMode ??
      body.reducerMode,
    chunkPromptTemplate:
      reducer.chunkPrompt ??
      reducer.chunkPromptTemplate ??
      body.chunkPrompt ??
      body.chunkPromptTemplate,
    reducerPromptTemplate:
      reducer.reducerPrompt ??
      reducer.reducerPromptTemplate ??
      reducer.prompt ??
      body.reducerPrompt ??
      body.reducerPromptTemplate,
    chunkMaxTokens:
      reducer.chunkMaxTokens ??
      body.chunkMaxTokens,
    finalMaxTokens:
      reducer.finalMaxTokens ??
      reducer.reducerMaxTokens ??
      body.finalMaxTokens ??
      body.reducerMaxTokens
  };
}

function normalizeReducerMode(value) {
  const mode = String(value || "").trim().toLowerCase();
  if (["model", "nim", "super", "second-stage", "two-step", "2step", "2-step"].includes(mode)) return "model";
  if (["local", "client", "off", "none"].includes(mode)) return "local";
  return null;
}

function longVideoPresetConfig(
  presetRaw,
  durationSeconds = 0,
  requestedConcurrency,
  requestedFramesPerSecond,
  options = {}
) {
  const preset = String(presetRaw || "balanced").toLowerCase();
  const duration = Number(durationSeconds) || 0;
  const requestedFps = positiveNumber(requestedFramesPerSecond);
  const defaults =
    preset === "fast"
      ? {
          preset: "fast",
          sampleFps: 6,
          maxFrames: duration > 0 ? Math.ceil(duration * 6) : 720,
          chunkSize: 5,
          overlap: 0,
          concurrency: 16,
          chunkMaxTokens: 512,
          finalMaxTokens: 1200,
          chunkTimeoutMs: 45000,
          targetWidth: 640
        }
      : preset === "detailed"
        ? {
            preset: "detailed",
            sampleFps: 10,
            maxFrames: duration > 0 ? Math.ceil(duration * 10) : 1200,
            chunkSize: 5,
            overlap: 1,
            concurrency: 16,
            chunkMaxTokens: 900,
            finalMaxTokens: 2200,
            chunkTimeoutMs: 90000,
            targetWidth: 768
          }
        : {
            preset: "balanced",
            sampleFps: 8,
            maxFrames: duration > 0 ? Math.ceil(duration * 8) : 960,
            chunkSize: 5,
            overlap: 1,
            concurrency: 16,
            chunkMaxTokens: 560,
            finalMaxTokens: 900,
            chunkTimeoutMs: 65000,
            targetWidth: 704
        };
  const maxFrameLimit = longVideoFrameLimit();
  const sampleFps = requestedFps ? Math.min(30, requestedFps) : defaults.sampleFps;
  const requestedFrameBudget =
    duration > 0
      ? Math.ceil(duration * sampleFps)
      : Math.ceil(sampleFps * 120);
  const explicitFrameBudget = boundedInteger(options.frameBudget, 0, 0, maxFrameLimit);
  const maxFrames =
    explicitFrameBudget > 0
      ? explicitFrameBudget
      : Math.min(maxFrameLimit, Math.max(defaults.maxFrames, requestedFrameBudget));
  const concurrency = Number.parseInt(String(requestedConcurrency || ""), 10);
  const chunkSize = boundedInteger(options.chunkSize, defaults.chunkSize, 1, LONG_VIDEO_MAX_IMAGES_PER_CHUNK);
  const overlap = boundedInteger(options.overlap, defaults.overlap, 0, Math.max(0, chunkSize - 1));
  const reducerMode = normalizeReducerMode(options.reducerMode);
  return {
    ...defaults,
    sampleFps,
    maxFrames,
    requestedFrameBudget: explicitFrameBudget || null,
    requestedFps,
    chunkSize,
    overlap,
    frameLimit: maxFrameLimit,
    maxConcurrency: longVideoMaxConcurrency(),
    concurrency: Number.isFinite(concurrency) && concurrency > 0 ? Math.min(longVideoMaxConcurrency(), concurrency) : defaults.concurrency,
    maxImagesPerChunk: LONG_VIDEO_MAX_IMAGES_PER_CHUNK,
    reducerMode,
    chunkPromptTemplate: optionalPromptText(options.chunkPromptTemplate),
    reducerPromptTemplate: optionalPromptText(options.reducerPromptTemplate),
    chunkMaxTokens: boundedInteger(options.chunkMaxTokens, defaults.chunkMaxTokens, 128, 4096),
    finalMaxTokens: boundedInteger(options.finalMaxTokens, defaults.finalMaxTokens, 256, 8192)
  };
}

function runLongFrameExtractor(videoPath, outputDir, config, onProgress) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonForFrames(),
      [
        "-",
        videoPath,
        outputDir,
        String(config.sampleFps),
        String(config.maxFrames || ""),
        String(config.targetWidth || 704)
      ],
      { stdio: ["pipe", "pipe", "pipe"] }
    );
    let stdout = "";
    let stderr = "";
    let progressBuffer = "";
    const drainProgress = (text, flush = false) => {
      progressBuffer += text;
      const lines = progressBuffer.split(/\r?\n/);
      progressBuffer = flush ? "" : lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        if (line.startsWith("PROGRESS ")) {
          try {
            onProgress?.(JSON.parse(line.slice("PROGRESS ".length)));
          } catch {
            // Keep malformed progress out of the user-facing error.
          }
        } else {
          stderr += `${line}\n`;
        }
      }
      if (flush && progressBuffer.trim()) {
        stderr += `${progressBuffer}\n`;
        progressBuffer = "";
      }
    };
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      drainProgress(String(chunk));
    });
    child.on("error", reject);
    child.on("close", (code) => {
      drainProgress("\n", true);
      if (code !== 0) {
        reject(new Error(stderr || `Long video frame extractor exited with ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(LONG_FRAME_EXTRACTOR_PY);
  });
}

async function extractLongVideoFrames(mediaSource, config, onProgress) {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "reason-vite-long-"));
  try {
    onProgress?.({ phase: "loading", percent: 0 });
    const { buffer } = await mediaSourceToBuffer(mediaSource);
    const videoPath = path.join(tempDir, "input.mp4");
    const frameDir = path.join(tempDir, "frames");
    await fs.mkdir(frameDir);
    await fs.writeFile(videoPath, buffer);
    onProgress?.({ phase: "decode_start", percent: 0 });
    const info = await runLongFrameExtractor(videoPath, frameDir, config, onProgress);
    const frames = [];
    for (const frame of info.frames || []) {
      const bytes = await fs.readFile(frame.path);
      frames.push({
        ...frame,
        dataUrl: `data:image/jpeg;base64,${bytes.toString("base64")}`,
        timestamp: Number(frame.timestamp) || 0,
        timestamp_text: frame.timestamp_text || formatTimestamp(frame.timestamp)
      });
    }
    if (frames.length === 0) throw new Error("Could not extract timestamped frames from video");
    return {
      tempDir,
      durationSeconds: Number(info.duration_s) || frames.at(-1)?.timestamp || 0,
      sourceFps: Number(info.source_fps) || null,
      sourceWidth: Number(info.source_width) || null,
      sourceHeight: Number(info.source_height) || null,
      sampleFps: Number(info.sample_fps) || config.sampleFps,
      frames
    };
  } catch (error) {
    await fs.rm(tempDir, { recursive: true, force: true });
    throw error;
  }
}

function chunkLongFrames(frames, config) {
  const size = Math.min(config.chunkSize || 5, LONG_VIDEO_MAX_IMAGES_PER_CHUNK);
  const overlap = Math.max(0, Math.min(config.overlap || 0, size - 1));
  const step = Math.max(1, size - overlap);
  const chunks = [];
  for (let start = 0; start < frames.length; start += step) {
    const slice = frames.slice(start, start + size);
    if (slice.length === 0) continue;
    chunks.push({
      index: chunks.length,
      frames: slice,
      startSeconds: slice[0].timestamp,
      endSeconds: slice.at(-1).timestamp,
      timeRange: `${slice[0].timestamp_text} - ${slice.at(-1).timestamp_text}`
    });
    if (start + size >= frames.length) break;
  }
  return chunks;
}

function stripReasoningFormatInstruction(prompt) {
  return String(prompt || "")
    .replace(/Answer the question using the following format:\s*<think>\s*Your reasoning\.\s*<\/think>\s*Write your final answer immediately after the <\/think> tag\./gi, "")
    .replace(/<think>\s*Your reasoning\.\s*<\/think>/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function planningTaskBrief(prompt) {
  const cleaned = stripReasoningFormatInstruction(prompt);
  const blocks = cleaned
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)
    .filter((block) => {
      if (/^stage\s*\d+\s*\/\s*(perceive|grasp)\s*prompt:/i.test(block)) return false;
      if (/^return the final answer as valid json/i.test(block)) return false;
      if (/^if this task is extended to multiple chunks/i.test(block)) return false;
      return true;
    });
  return blocks.join("\n\n").trim();
}

function compactPlanningTask(prompt) {
  const cleaned = planningTaskBrief(prompt);
  const taskMatch = cleaned.match(/you are given the(?: robot)? task:?\s*["“]([^"”]+)["”]/i);
  if (taskMatch?.[1]) return `You are given the task "${taskMatch[1]}".`;
  return cleaned.split(/\n/).map((line) => line.trim()).filter(Boolean)[0] || "Plan the robot action from the visible image.";
}

function compactTrajectoryPrompt(prompt) {
  const task = compactPlanningTask(prompt);
  return `${task} Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.`;
}

function balancedJsonCandidates(text) {
  const candidates = [];
  const source = String(text || "");
  for (let start = 0; start < source.length; start += 1) {
    const open = source[start];
    if (open !== "{" && open !== "[") continue;
    const close = open === "{" ? "}" : "]";
    const stack = [close];
    let inString = false;
    let escaped = false;
    for (let index = start + 1; index < source.length; index += 1) {
      const char = source[index];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\") {
        escaped = true;
        continue;
      }
      if (char === '"') {
        inString = !inString;
        continue;
      }
      if (inString) continue;
      if (char === "{" || char === "[") {
        stack.push(char === "{" ? "}" : "]");
      } else if (char === "}" || char === "]") {
        if (char !== stack.at(-1)) break;
        stack.pop();
        if (stack.length === 0) {
          candidates.push(source.slice(start, index + 1));
          break;
        }
      }
    }
  }
  return candidates;
}

function maybeJsonText(value) {
  const text = String(value || "").trim();
  if (!text) return null;
  const candidates = [text];
  const fencedMatches = [...text.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)];
  for (const match of fencedMatches) candidates.push(match[1].trim());
  for (const candidate of [...candidates]) {
    candidates.push(...balancedJsonCandidates(candidate));
    const first = candidate.search(/[\[{]/);
    if (first >= 0) candidates.push(candidate.slice(first));
  }
  const seen = new Set();
  for (const candidate of candidates) {
    const trimmed = String(candidate || "").trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    try {
      return JSON.parse(trimmed);
    } catch {
      // Try the next balanced or fenced candidate.
    }
  }
  return null;
}

function cleanModelText(value) {
  return String(value || "")
    .replace(/```(?:json)?/gi, "")
    .replace(/```/g, "")
    .trim();
}

function compactChunkSummary(parsed, content) {
  if (parsed?.summary) return String(parsed.summary);
  const cleaned = cleanModelText(content);
  const summaryMatch = cleaned.match(/"summary"\s*:\s*"((?:\\.|[^"\\])*)"/i);
  if (summaryMatch) {
    try {
      return JSON.parse(`"${summaryMatch[1]}"`);
    } catch {
      return summaryMatch[1].replace(/\\"/g, '"');
    }
  }
  const firstLine = cleaned
    .split(/\n+/)
    .map((line) => line.trim())
    .find((line) => line && !/^[{}\[\],]+$/.test(line));
  const fallback = firstLine || cleaned || "";
  return fallback.length > 320 ? `${fallback.slice(0, 317)}...` : fallback;
}

function normalizeBaseUrl(url) {
  const stripped = String(url || "").replace(/\/$/, "");
  return stripped.endsWith("/v1") ? stripped : `${stripped}/v1`;
}

function reasonerEndpointPool(defaultBaseUrl) {
  const raw = process.env.REASONER_ENDPOINTS || "";
  const values = raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return (values.length ? values : [defaultBaseUrl]).map(normalizeBaseUrl);
}

function backendFetchFailureMessage(baseUrl, error) {
  const raw = error instanceof Error ? error.message : String(error || "unknown error");
  const endpoint = `${baseUrl}/chat/completions`;
  if (/fetch failed|ECONNREFUSED|ECONNRESET|connect|connection/i.test(raw)) {
    return `Backend unavailable at ${endpoint}. The Vite UI is up, but the OpenAI-compatible model server on 127.0.0.1:8000 is not responding. Check logs/vllm.log on the target for the model-load failure. Underlying error: ${raw}`;
  }
  return raw;
}

async function postOpenAiJson(baseUrl, payload, timeoutMs, parentSignal) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const abort = () => controller.abort();
  parentSignal?.addEventListener?.("abort", abort, { once: true });
  try {
    let upstream;
    try {
      upstream = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}`
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
    } catch (error) {
      throw new Error(backendFetchFailureMessage(baseUrl, error));
    }
    let data = null;
    try {
      data = await upstream.json();
    } catch {
      data = null;
    }
    if (!upstream.ok) {
      throw new Error(data?.error?.message || data?.message || `Reasoner returned HTTP ${upstream.status}`);
    }
    return data;
  } finally {
    clearTimeout(timer);
    parentSignal?.removeEventListener?.("abort", abort);
  }
}

function planningStageSystemPrompt(stage, systemPrompt) {
  const base = String(systemPrompt || "").trim();
  const stageInstruction =
    stage.mode === "perception"
      ? "This is the Perceive pass. Return compact valid JSON only with object labels and bbox_2d rectangles measured in the attached image coordinate frame."
      : "This is the Grasp pass. Specify the 2D trajectory the end effector should follow in pixel space. Return compact valid JSON only with point_2d trajectory coordinates measured in the attached image coordinate frame.";
  return [base, stageInstruction, "Keep any <think> reasoning brief and coordinate-focused. Do not include markdown fences or coordinates from any other frame."]
    .filter(Boolean)
    .join("\n\n");
}

function planningStagePrompt({ stage, taskPrompt }) {
  const task = compactPlanningTask(taskPrompt);
  const trajectoryTask = compactTrajectoryPrompt(taskPrompt);
  const coordinateFrame = `${stage.phase.toLowerCase().replace(/[^a-z0-9]+/g, "_")}_${stage.time}`;
  if (stage.mode === "perception") {
    return `Robot task:
${task}

The attached image is the ${stage.phase} frame at ${stage.time}. Use this image as the only coordinate reference.

Identify task-relevant objects. Return object boxes only. Each bbox_2d must contain exactly four 0-1000 image coordinates in this order: [left_x, top_y, right_x, bottom_y]. Do not return point_2d or trajectory in this pass.

Answer the question using the following format:
<think>
Briefly name the visible task-relevant objects and any simple coordinate references you used.
</think>
Write the final JSON immediately after the </think> tag.

Return valid JSON only:
{
  "perception_coordinate_frame": "${coordinateFrame}",
  "perception": [
    {"time": "${stage.time}", "label": "object name", "bbox_2d": [left_x, top_y, right_x, bottom_y], "caption": "why it matters"}
  ]
}`;
  }

  return `Robot task:
${trajectoryTask}

The attached image is the ${stage.phase} frame at ${stage.time}. Use this image as the only coordinate reference.

Return several trajectory waypoints only as point_2d values. Do not return bbox_2d for trajectory waypoints.

Answer the question using the following format:
<think>
Briefly name the visible references and the planned end-effector path in image coordinates.
</think>
Write the final JSON immediately after the </think> tag.

Return valid JSON only:
{
  "trajectory_coordinate_frame": "${coordinateFrame}",
  "trajectory": [
    {"time": "${stage.time}", "label": "gripper trajectory", "point_2d": [x, y]}
  ]
}`;
}

function arrayFromParsed(parsed, keys, depth = 0) {
  if (Array.isArray(parsed)) return parsed;
  if (!parsed || typeof parsed !== "object" || depth > 5) return [];
  for (const key of keys) {
    if (Array.isArray(parsed[key])) return parsed[key];
  }
  if (keys.some((key) => key in parsed)) return [parsed];
  if (keys.some((key) => key === "perception") && (parsed.bbox_2d || parsed.bounding_box || parsed.bbox)) return [parsed];
  if (keys.some((key) => key === "trajectory") && (parsed.point_2d || parsed.point || parsed.waypoint || parsed.waypoints)) return [parsed];
  for (const value of Object.values(parsed)) {
    const nested = arrayFromParsed(value, keys, depth + 1);
    if (nested.length > 0) return nested;
  }
  return [];
}

function bboxNumbers(record) {
  const raw = record?.bbox_2d || record?.bbox || record?.bounding_box || record?.box;
  const numbers = Array.isArray(raw) ? raw.map(Number).filter(Number.isFinite) : [];
  return numbers.length >= 4 ? numbers.slice(0, 4) : null;
}

function bboxCenter(record) {
  const bbox = bboxNumbers(record);
  if (!bbox) return null;
  const [left, top, right, bottom] = bbox;
  return {
    x: (left + right) / 2,
    y: (top + bottom) / 2,
    left,
    top,
    right,
    bottom
  };
}

function perceptionLabel(record) {
  return String(record?.label || record?.name || record?.object || record?.class || "").toLowerCase();
}

function fallbackTrajectoryFromPerception(perception, time = "00:02.00") {
  const boxed = perception.filter((item) => bboxNumbers(item));
  if (boxed.length < 2) return [];
  const drill =
    boxed.find((item) => /drill|black|decker|cordless|tool/i.test(perceptionLabel(item))) ||
    boxed.find((item) => !/box|bin|crate|basket|container|yellow|green/i.test(perceptionLabel(item)));
  const container =
    boxed.find((item) => /box|bin|crate|basket|container|yellow|green/i.test(perceptionLabel(item))) ||
    boxed.find((item) => item !== drill);
  const drillCenter = bboxCenter(drill);
  const containerCenter = bboxCenter(container);
  if (!drillCenter || !containerCenter) return [];
  const liftY = Math.max(30, Math.min(drillCenter.y, containerCenter.top) - 80);
  const releaseY = Math.max(containerCenter.top + 40, Math.min(containerCenter.bottom - 20, containerCenter.y));
  return [
    { time, label: "approach drill handle", point_2d: [Math.round(drillCenter.x), Math.round(drillCenter.y)], caption: "Derived fallback from detected drill box center." },
    { time, label: "grasp drill handle", point_2d: [Math.round(drillCenter.x), Math.round(drillCenter.y)], caption: "Derived fallback because the Grasp pass did not return usable waypoints." },
    { time, label: "lift clear", point_2d: [Math.round(drillCenter.x), Math.round(liftY)], caption: "Lift above the nearby support surface." },
    { time, label: "move over yellow box", point_2d: [Math.round(containerCenter.x), Math.round(containerCenter.top + 60)], caption: "Move toward the detected container opening." },
    { time, label: "release into box", point_2d: [Math.round(containerCenter.x), Math.round(releaseY)], caption: "Release inside the detected container." }
  ];
}

function recordHasPoint(record) {
  return Boolean(record?.point_2d || record?.point || record?.waypoint);
}

async function submitPlanningStage({ endpoint, model, onLog, params, signal, stage, systemPrompt, taskPrompt }) {
  const { baseUrl, payload, redactedPayload } = buildReasoningPayload({
    model,
    prompt: planningStagePrompt({ stage, taskPrompt }),
    systemPrompt: planningStageSystemPrompt(stage, systemPrompt),
    mediaDataUrl: stage.dataUrl,
    mediaKind: "image",
    params: {
      ...params,
      temperature: Math.min(Number(params.temperature) || 0.3, 0.3),
      top_p: Math.min(Number(params.top_p) || 0.3, 0.3)
    }
  });
  payload.max_tokens = Math.min(Number(params.max_tokens) || 900, stage.mode === "perception" ? 900 : 700);
  const stageEndpoint = endpoint || baseUrl;
  onLog?.({
    label: `${stage.phase} request built`,
    detail: `${stage.mode} pass · ${stageEndpoint}/chat/completions · 1 image · max_tokens ${payload.max_tokens}`
  });
  const startedAt = Date.now();
  onLog?.({
    label: `${stage.phase} waiting on NIM`,
    detail: `Coordinate frame ${stage.actualTime || stage.time}; OpenAI JSON response has no token-level progress until NIM returns. Timeout 90s.`
  });
  const data = await postOpenAiJson(stageEndpoint, payload, 90000, signal);
  const normalized = normalizeReasonerMessage(data);
  const content = normalized.answer || normalized.combined;
  const elapsedSeconds = (Date.now() - startedAt) / 1000;
  onLog?.({
    label: `${stage.phase} response received`,
    detail: `${elapsedSeconds.toFixed(1)}s · answer ${String(content || "").length} chars · reasoning ${String(normalized.reasoning || "").length} chars`
  });
  return {
    phase: stage.phase,
    mode: stage.mode,
    time: stage.time,
    actualTime: stage.actualTime,
    content,
    reasoning: normalized.reasoning,
    parsed: maybeJsonText(content),
    elapsedSeconds,
    payload: redactedPayload,
    raw: data
  };
}

function mergePlanningStages(stageResults) {
  const perceptionStage = stageResults.find((stage) => stage.mode === "perception");
  const trajectoryStage = stageResults.find((stage) => stage.mode === "trajectory");
  const perception = perceptionStage ? arrayFromParsed(perceptionStage.parsed, ["perception", "objects", "detections"]) : [];
  const trajectory = trajectoryStage
    ? arrayFromParsed(trajectoryStage.parsed, ["trajectory", "waypoints", "path", "planned_path", "end_effector_path", "grasp_plan"])
    : [];
  const trajectoryPointCount = trajectory.filter(recordHasPoint).length;
  const fallbackTrajectory =
    trajectoryPointCount >= 2 ? [] : fallbackTrajectoryFromPerception(perception, trajectoryStage?.time || "00:02.00");
  const robustTrajectory = trajectoryPointCount >= 2 ? trajectory : [...trajectory, ...fallbackTrajectory];
  return {
    planning_mode: "two_call_image_coordinate_frames",
    perception_coordinate_frame:
      perceptionStage?.parsed?.perception_coordinate_frame ||
      (perceptionStage ? `${perceptionStage.phase.toLowerCase()}_${perceptionStage.time}` : undefined),
    trajectory_coordinate_frame:
      trajectoryStage?.parsed?.trajectory_coordinate_frame ||
      (trajectoryStage ? `${trajectoryStage.phase.toLowerCase()}_${trajectoryStage.time}` : undefined),
    perception,
    trajectory: robustTrajectory,
    timeline_summary: [
      ...(perceptionStage
        ? [{ start: perceptionStage.time, end: perceptionStage.time, phase: perceptionStage.phase, caption: "Perception pass on its own image frame." }]
        : []),
      ...(trajectoryStage
        ? [{ start: trajectoryStage.time, end: trajectoryStage.time, phase: trajectoryStage.phase, caption: "Trajectory pass on its own image frame." }]
        : [])
    ],
    uncertainties: [
      ...stageResults.flatMap((stage) => arrayFromParsed(stage.parsed?.uncertainties, ["uncertainties"])),
      ...(trajectoryPointCount < 2 && fallbackTrajectory.length > 0
        ? ["Trajectory waypoints were expanded from Perceive boxes because the Grasp pass did not return enough usable point_2d values."]
        : [])
    ].filter(Boolean),
    stage_outputs: stageResults.map((stage) => ({
      phase: stage.phase,
      mode: stage.mode,
      time: stage.time,
      actual_time: stage.actualTime,
      elapsed_seconds: stage.elapsedSeconds,
      content: stage.content
    }))
  };
}

async function runConcurrent(items, concurrency, worker) {
  const results = new Array(items.length);
  let nextIndex = 0;
  const count = Math.max(1, Math.min(Number(concurrency) || 1, items.length || 1));
  async function loop() {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await worker(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: count }, loop));
  return results;
}

function fillLongVideoTemplate(template, values) {
  if (!template) return null;
  let replaced = false;
  const rendered = template.replace(/\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}|\{\s*([A-Za-z0-9_.-]+)\s*\}/g, (match, doubleKey, singleKey) => {
    const key = doubleKey || singleKey;
    if (!Object.prototype.hasOwnProperty.call(values, key)) return match;
    replaced = true;
    return String(values[key] ?? "");
  });
  return { rendered, replaced };
}

function chunkPrompt({ chunk, durationSeconds, prompt, preset, template }) {
  const task = stripReasoningFormatInstruction(prompt) || "Summarize what happens in this video segment.";
  const frames = chunk.frames
    .map((frame, index) => `Frame ${index + 1}: ${frame.timestamp_text} (${frame.timestamp.toFixed(2)}s)`)
    .join("\n");
  const baseContext = `You are analyzing one timestamped window from a longer video.

Full clip duration: ${formatTimestamp(durationSeconds)}.
Window: ${chunk.timeRange}.
Preset: ${preset}.

Frame timestamp map:
${frames}

Original user task:
${task}`;
  const templateResult = fillLongVideoTemplate(template, {
    original_task: task,
    original_prompt: task,
    duration: formatTimestamp(durationSeconds),
    preset,
    window: chunk.timeRange,
    start: chunk.frames[0]?.timestamp_text || "00:00.00",
    end: chunk.frames.at(-1)?.timestamp_text || "00:00.00",
    frame_map: frames,
    frame_count: chunk.frames.length
  });
  if (templateResult?.replaced) return templateResult.rendered;
  if (templateResult?.rendered) {
    return `${baseContext}

Custom chunk extraction instructions:
${templateResult.rendered}

Return concise valid JSON only. Use only visible evidence from these frames. Do not invent events between frames.`;
  }
  return `You are analyzing one timestamped window from a longer video.

Full clip duration: ${formatTimestamp(durationSeconds)}.
Window: ${chunk.timeRange}.
Preset: ${preset}.

Frame timestamp map:
${frames}

Original user task:
${task}

Return concise valid JSON only with this shape:
{
  "time_range": {"start": "${chunk.frames[0].timestamp_text}", "end": "${chunk.frames.at(-1).timestamp_text}"},
  "summary": "one-sentence visible summary",
  "events": [
    {"start": "mm:ss.ff", "end": "mm:ss.ff", "event_type": "visible event type", "caption": "what is visibly happening", "confidence": 0.0}
  ],
  "uncertainty": "what might be missed because only these frames were sampled",
  "evidence_frames": ["Frame 1", "Frame 2"]
}

Use only visible evidence from these frames. Do not invent events between frames. Keep output short.`;
}

function reducerPrompt({ chunks, durationSeconds, failedChunks, prompt, preset, warnings, template }) {
  const task = stripReasoningFormatInstruction(prompt) || "Summarize the full video.";
  const chunkText = chunks
    .map((chunk) => JSON.stringify(compactChunkForReducer(chunk)))
    .join("\n");
  const failures = failedChunks.length
    ? `\nFailed chunks: ${failedChunks.map((chunk) => `${chunk.index + 1} ${chunk.timeRange}: ${chunk.error}`).join("; ")}`
    : "";
  const templateResult = fillLongVideoTemplate(template, {
    original_task: task,
    original_prompt: task,
    duration: formatTimestamp(durationSeconds),
    preset,
    warnings: warnings.join("; ") || "none",
    failed_chunks: failures.trim() || "none",
    chunk_jsonl: chunkText,
    chunk_count: chunks.length
  });
  if (templateResult?.replaced) return templateResult.rendered;
  if (templateResult?.rendered) {
    return `Stitch these timestamped chunk analyses into one answer for the original user task.

Original user task:
${task}

Clip duration: ${formatTimestamp(durationSeconds)}
Analysis preset: ${preset}
Warnings: ${warnings.join("; ") || "none"}${failures}

Reducer instructions:
${templateResult.rendered}

Chunk analyses:
${chunkText}

Return the final answer only. Do not include <think> tags.`;
  }
  return `Stitch these timestamped chunk analyses into one answer for the original user task.

Original user task:
${task}

Clip duration: ${formatTimestamp(durationSeconds)}
Analysis preset: ${preset}
Warnings: ${warnings.join("; ") || "none"}${failures}

Chunk analyses:
${chunkText}

Instructions:
- Preserve timestamp evidence across the full clip.
- Deduplicate overlapping events with the same event type and similar timestamp.
- Mention failed or sparse sections as limitations instead of hiding them.
- Follow the user's requested output format when possible. If no exact format is requested, return no more than 12 concise timeline bullets plus key limitations.
- Do not include <think> tags.`;
}

function compactChunkForReducer(chunk) {
  const parsed = chunk?.parsed && typeof chunk.parsed === "object" ? chunk.parsed : {};
  const events = Array.isArray(parsed.events) ? parsed.events : [];
  const uncertainEvents = Array.isArray(parsed.uncertain_events) ? parsed.uncertain_events : [];
  const summary = chunk.summary || compactChunkSummary(parsed, chunk.content);
  return {
    chunk: Number(chunk.index) + 1,
    status: chunk.status,
    time_range: chunk.timeRange,
    summary: summary || "",
    events: events.slice(0, 24),
    uncertain_events: uncertainEvents.slice(0, 12),
    error: chunk.error || undefined
  };
}

function stitchReducerMode(config) {
  if (config?.reducerMode === "model") return "model";
  if (config?.reducerMode === "local") return "local";
  const value = String(process.env.REASONER_LONG_USE_MODEL_REDUCER || "").trim().toLowerCase();
  if (["1", "true", "yes", "always"].includes(value)) return "model";
  if (value === "detailed" && config?.preset === "detailed") return "model";
  return "local";
}

function stitchSubstepDefinitions(mode = "local") {
  const common = [
    { key: "collect_outputs", label: "Collect outputs" },
    { key: "normalize_events", label: "Normalize JSON/events" },
    { key: "dedupe_timeline", label: "Deduplicate timeline" },
    { key: "build_final", label: "Build final response" }
  ];
  if (mode === "model") {
    return [
      common[0],
      common[1],
      { key: "compact_prompt", label: "Compact reducer prompt" },
      { key: "model_reducer", label: "Call reducer NIM" },
      common[3]
    ];
  }
  return common;
}

function makeStitchSubsteps(mode = "local") {
  return stitchSubstepDefinitions(mode).map((step) => ({
    ...step,
    status: "pending",
    progress: 0,
    detail: ""
  }));
}

function estimateStitchSeconds({ chunks = [], mode = "local", promptChars = 0 }) {
  const count = Math.max(1, chunks.length || 1);
  if (mode === "model") {
    return Math.max(20, Math.min(180, Math.round(12 + count * 0.35 + promptChars / 800)));
  }
  return Math.max(1.5, Math.min(12, Math.round((0.6 + count * 0.018) * 10) / 10));
}

function buildTimelineSummary(chunkSummaries) {
  const summaries = [];
  for (const item of chunkSummaries) {
    const summary = String(item.summary || "").trim();
    if (!summary) continue;
    const duplicate = summaries.some((existing) => existing.summary.toLowerCase() === summary.toLowerCase());
    if (duplicate) continue;
    const [start = "", end = start] = String(item.time_range || "").split(/\s+-\s+/);
    summaries.push({
      start,
      end,
      summary,
      source_chunk: item.index
    });
  }
  return summaries;
}

function timestampSeconds(value) {
  const text = String(value || "");
  const match = text.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (!match) return Number.NaN;
  return Number(match[1]) * 60 + Number(match[2]);
}

function localLongVideoStitch({ chunks, durationSeconds, failedChunks, onProgress, prompt, preset, warnings }) {
  const wantsJson = /\bjson\b/i.test(String(prompt || ""));
  const events = [];
  const chunkSummaries = [];
  onProgress?.("collect_outputs", {
    status: "running",
    progress: 20,
    detail: `Reading ${chunks.length} chunk outputs`
  });
  for (const [position, chunk] of chunks.entries()) {
    const summary = chunk.summary || compactChunkSummary(chunk.parsed, chunk.content);
    if (summary) {
      chunkSummaries.push({
        index: chunk.index + 1,
        time_range: chunk.timeRange,
        summary: String(summary).replace(/```(?:json)?|```/gi, "").trim()
      });
    }
    if (position % 40 === 0 || position === chunks.length - 1) {
      onProgress?.("collect_outputs", {
        status: "running",
        progress: ((position + 1) / Math.max(1, chunks.length)) * 100,
        detail: `${position + 1}/${chunks.length} chunks scanned`
      });
    }
  }
  onProgress?.("collect_outputs", {
    status: "done",
    progress: 100,
    detail: `${chunkSummaries.length} summaries collected`
  });
  onProgress?.("normalize_events", {
    status: "running",
    progress: 15,
    detail: "Flattening parsed events"
  });
  for (const [position, chunk] of chunks.entries()) {
    const summary = chunk.summary || compactChunkSummary(chunk.parsed, chunk.content);
    const parsedEvents = Array.isArray(chunk.parsed?.events) ? chunk.parsed.events : [];
    for (const event of parsedEvents) {
      const start = String(event.start || chunk.timeRange?.split(" - ")[0] || "");
      const type = String(event.event_type || event.type || "visible_event");
      const seconds = timestampSeconds(start);
      const duplicate = events.some((existing) => {
        if (String(existing.event_type).toLowerCase() !== type.toLowerCase()) return false;
        const existingSeconds = timestampSeconds(existing.start);
        return Number.isFinite(seconds) && Number.isFinite(existingSeconds) && Math.abs(existingSeconds - seconds) < 1.5;
      });
      if (duplicate) continue;
      events.push({
        start,
        end: String(event.end || start),
        event_type: type,
        caption: String(event.caption || summary || "Visible event"),
        confidence: Number.isFinite(Number(event.confidence)) ? Number(event.confidence) : undefined,
        source_chunk: chunk.index + 1
      });
    }
    if (position % 40 === 0 || position === chunks.length - 1) {
      onProgress?.("normalize_events", {
        status: "running",
        progress: ((position + 1) / Math.max(1, chunks.length)) * 100,
        detail: `${events.length} candidate events`
      });
    }
  }
  onProgress?.("normalize_events", {
    status: "done",
    progress: 100,
    detail: `${events.length} candidate events`
  });
  onProgress?.("dedupe_timeline", {
    status: "running",
    progress: 50,
    detail: "Sorting by timestamp and removing overlaps"
  });
  events.sort((left, right) => timestampSeconds(left.start) - timestampSeconds(right.start));
  onProgress?.("dedupe_timeline", {
    status: "done",
    progress: 100,
    detail: `${events.length} stitched events`
  });
  const failed = failedChunks.map((chunk) => ({
    index: chunk.index + 1,
    time_range: chunk.timeRange,
    error: chunk.error
  }));
  const samplingLimits = {
    clip_duration: formatTimestamp(durationSeconds),
    preset,
    frame_chunking: `Timestamped image chunks, max ${LONG_VIDEO_MAX_IMAGES_PER_CHUNK} images per NIM request`,
    warnings,
    failed_chunks: failed
  };

  onProgress?.("build_final", {
    status: "running",
    progress: 40,
    detail: wantsJson ? "Rendering final JSON" : "Rendering final timeline"
  });
  if (wantsJson) {
    const content = JSON.stringify(
      {
        events,
        timeline_summary: buildTimelineSummary(chunkSummaries),
        chunk_summaries: chunkSummaries,
        sampling_limits: samplingLimits
      },
      null,
      2
    );
    onProgress?.("build_final", { status: "done", progress: 100, detail: "Final JSON ready" });
    return content;
  }

  const timeline =
    events.length > 0
      ? events.slice(0, 24).map((event) => `- ${event.start}${event.end && event.end !== event.start ? `-${event.end}` : ""}: ${event.event_type} — ${event.caption}`)
      : chunkSummaries.slice(0, 18).map((item) => `- ${item.time_range}: ${item.summary}`);
  const limitLines = warnings.map((warning) => `- ${warning}`);
  if (failed.length > 0) {
    limitLines.push(`- Failed chunks: ${failed.map((chunk) => `${chunk.index} (${chunk.time_range})`).join(", ")}`);
  }
  const content = [
    `Long video analysis (${preset})`,
    "",
    `Coverage: ${chunks.length - failed.length}/${chunks.length} chunks across ${formatTimestamp(durationSeconds)}.`,
    "",
    "Timeline",
    ...timeline,
    "",
    "Sampling limits",
    ...limitLines
  ].join("\n");
  onProgress?.("build_final", { status: "done", progress: 100, detail: "Final timeline ready" });
  return content;
}

function longVideoWarnings(durationSeconds, frameCount, preset, config = {}) {
  const warnings = [];
  const coverage = durationSeconds > 0 ? frameCount / durationSeconds : 0;
  if (coverage > 0 && coverage < 2) {
    warnings.push(`${coverage.toFixed(2)} fps equivalent coverage; fast contacts or brief state changes may be missed.`);
  }
  if (config.requestedFps) {
    warnings.push(
      `Long Video is using the FPS slider at ${config.requestedFps} fps. Higher FPS creates more chunks because each NIM request can include only ${LONG_VIDEO_MAX_IMAGES_PER_CHUNK} frames.`
    );
  }
  if (config.chunkSize && config.chunkSize !== LONG_VIDEO_MAX_IMAGES_PER_CHUNK) {
    warnings.push(
      `Chunks are using ${config.chunkSize} frame${config.chunkSize === 1 ? "" : "s"} per NIM call; smaller chunks give finer control but create more requests.`
    );
  }
  if (config.overlap > 0) {
    warnings.push(
      `Each chunk overlaps by ${config.overlap} frame${config.overlap === 1 ? "" : "s"} so boundary events can be seen by adjacent requests.`
    );
  }
  if (config.requestedFps && coverage > 0 && coverage + 0.01 < config.requestedFps) {
    warnings.push(
      `Requested ${config.requestedFps} fps was capped to ${coverage.toFixed(2)} fps effective coverage by the ${config.frameLimit} frame budget.`
    );
  }
  if (config.requestedFrameBudget) {
    warnings.push(`Frame budget override is ${config.maxFrames} frames for this run.`);
  }
  if (stitchReducerMode(config) === "model") {
    warnings.push("Second-stage stitching is enabled; after chunking, Vite calls the Super NIM again to reduce chunk JSON into the final answer.");
  }
  if (config.concurrency > 8) {
    warnings.push(
      `Chunk concurrency is set to ${config.concurrency}. This horde RTX PRO 6000 handled 16 comfortably in probes; lower it if other users are sharing the NIM or TTFT spikes.`
    );
  }
  if (preset === "fast") warnings.push("Fast preset uses 6 fps for dense captioning, but very brief actions or contacts may still be missed.");
  if (preset === "detailed") warnings.push("Detailed preset may run longer than the clip duration.");
  warnings.push("NIM native video decode is bypassed here; Vite sends timestamped image chunks with at most 5 frames per request.");
  return warnings;
}

function mediaMime(name, fallback = "application/octet-stream") {
  const lower = name.toLowerCase();
  if (lower.endsWith(".mp4")) return "video/mp4";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".png")) return "image/png";
  return fallback.split(";")[0] || "application/octet-stream";
}

function commandValue(command, flag) {
  const index = command.indexOf(flag);
  if (index === -1) return null;
  const rest = command.slice(index + flag.length).trimStart();
  const nextFlag = rest.search(/\s--[A-Za-z0-9-]+/);
  return (nextFlag === -1 ? rest : rest.slice(0, nextFlag)).trim() || null;
}

function splitCommandWords(command) {
  const words = [];
  const pattern = /"((?:\\.|[^"\\])*)"|'([^']*)'|(\S+)/g;
  let match;
  while ((match = pattern.exec(command))) {
    words.push(match[1] || match[2] || match[3]);
  }
  return words;
}

function vllmServeModel(command) {
  const words = splitCommandWords(command);
  const serveIndex = words.findIndex((word) => word === "serve");
  if (serveIndex === -1) return null;
  for (let index = serveIndex + 1; index < words.length; index += 1) {
    const word = words[index];
    if (!word || word.startsWith("-")) {
      if (word?.startsWith("--") && index + 1 < words.length && !words[index + 1].startsWith("-")) index += 1;
      continue;
    }
    return word;
  }
  return null;
}

function inferQuantization(model, flags = {}) {
  const mid = String(model || "").toLowerCase();
  const explicit = flags.quantization || process.env.MODEL_QUANTIZATION || process.env.VLLM_QUANTIZATION || "";
  let inferred = "";
  if (mid.includes("nvfp4") || mid.includes("fp4")) inferred = "nvfp4/fp4";
  else if (mid.includes("fp8")) inferred = "fp8";
  else if (mid.includes("int8")) inferred = "int8";
  else if (mid.includes("awq")) inferred = "awq";
  else if (mid.includes("gptq")) inferred = "gptq";

  const method = explicit || inferred;
  return {
    applied: Boolean(method),
    method: method || null,
    dtype: flags.dtype || process.env.MODEL_DTYPE || null,
    source: explicit
      ? "vLLM launch flag"
      : inferred
        ? "model identifier"
        : "no --quantization flag or quantized model suffix detected"
  };
}

function parseVllmCommand(command) {
  const flags = {
    model: commandValue(command, "--model") || vllmServeModel(command),
    revision: commandValue(command, "--revision"),
    served_model_name: commandValue(command, "--served-model-name"),
    max_model_len: commandValue(command, "--max-model-len"),
    media_io_kwargs: commandValue(command, "--media-io-kwargs"),
    reasoning_parser: commandValue(command, "--reasoning-parser"),
    dtype: commandValue(command, "--dtype"),
    quantization: commandValue(command, "--quantization"),
    gpu_memory_utilization: commandValue(command, "--gpu-memory-utilization"),
    host: commandValue(command, "--host"),
    port: commandValue(command, "--port"),
    allowed_local_media_path: commandValue(command, "--allowed-local-media-path"),
    trust_remote_code: command.includes("--trust-remote-code")
  };
  Object.keys(flags).forEach((key) => flags[key] === null && delete flags[key]);
  return flags;
}

function execFileText(command, args) {
  return new Promise((resolve, reject) => {
    execFile(command, args, { timeout: 3000, maxBuffer: 1024 * 1024 }, (error, stdout) => {
      if (error) {
        reject(error);
        return;
      }
      resolve(stdout);
    });
  });
}

async function findVllmProcess(selectedModel) {
  try {
    const stdout = await execFileText("ps", ["-eo", "pid=,cmd="]);
    const lines = stdout
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.includes("vllm serve"));
    const modelNeedle = String(selectedModel || "").toLowerCase();
    const line =
      lines.find((item) => item.toLowerCase().includes(modelNeedle)) ||
      lines.find((item) => item.includes("--port 8000")) ||
      lines[0];
    if (!line) return null;
    const match = line.match(/^(\d+)\s+(.*)$/);
    if (!match) return null;
    return { pid: Number(match[1]), command: match[2], flags: parseVllmCommand(match[2]) };
  } catch {
    return null;
  }
}

function safeNimEnv(rawEnv = []) {
  const allowPrefixes = [
    "NIM_",
    "MODEL_",
    "VLLM_",
    "COSMOS_",
    "INFERENCE_BACKEND",
    "REASONER_MEDIA_MODE"
  ];
  const blocked = /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)/i;
  const safe = {};
  for (const item of rawEnv || []) {
    const index = String(item).indexOf("=");
    if (index <= 0) continue;
    const key = item.slice(0, index);
    const value = item.slice(index + 1);
    if (blocked.test(key)) continue;
    if (!allowPrefixes.some((prefix) => key.startsWith(prefix))) continue;
    safe[key] = value;
  }
  return safe;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 3000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function findNimContainer() {
  const containerName = process.env.CONTAINER_NAME || "cosmos-nim";
  try {
    const stdout = await execFileText("docker", ["inspect", containerName]);
    const data = JSON.parse(stdout);
    const container = Array.isArray(data) ? data[0] : null;
    if (!container) return null;
    return {
      name: container.Name ? String(container.Name).replace(/^\//, "") : containerName,
      image: container.Config?.Image || null,
      image_id: container.Image || null,
      status: container.State?.Status || null,
      started_at: container.State?.StartedAt || null,
      env: safeNimEnv(container.Config?.Env || [])
    };
  } catch {
    return null;
  }
}

async function fetchBackendModelInfo(baseUrl, selectedModel) {
  try {
    const response = await fetchWithTimeout(`${baseUrl}/models`, {
      headers: { Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}` }
    });
    if (!response.ok) return null;
    const data = await response.json();
    const models = Array.isArray(data?.data) ? data.data : [];
    return models.find((item) => item.id === selectedModel) || models[0] || null;
  } catch {
    return null;
  }
}

function appSourceInfo() {
  return {
    sha: process.env.APP_GIT_SHA || null,
    timestamp: process.env.APP_GIT_TIMESTAMP || null,
    branch: process.env.APP_GIT_BRANCH || null,
    dirty: process.env.APP_GIT_DIRTY === "1" ? true : process.env.APP_GIT_DIRTY === "0" ? false : null
  };
}

function huggingFaceToken() {
  return process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || null;
}

function huggingFaceCacheRoot() {
  if (process.env.HUGGINGFACE_HUB_CACHE) return process.env.HUGGINGFACE_HUB_CACHE;
  if (process.env.TRANSFORMERS_CACHE) return process.env.TRANSFORMERS_CACHE;
  const hfHome =
    process.env.HF_HOME ||
    (process.env.XDG_CACHE_HOME ? path.join(process.env.XDG_CACHE_HOME, "huggingface") : path.join(os.homedir(), ".cache", "huggingface"));
  return path.join(hfHome, "hub");
}

function normalizeHfRepoId(value) {
  const raw = String(value || "").trim();
  if (!raw || raw.startsWith("/") || raw.startsWith(".") || raw.includes("\\") || raw.includes("://")) return null;
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(raw)) return null;
  return raw;
}

function hfRepoCacheDir(repoId) {
  return path.join(huggingFaceCacheRoot(), `models--${repoId.replace(/\//g, "--")}`);
}

function hfCommitUrl(repoId, sha) {
  if (!repoId || !sha) return null;
  return `https://huggingface.co/${repoId}/commit/${sha}`;
}

async function readTextIfExists(filePath) {
  try {
    return (await fs.readFile(filePath, "utf8")).trim() || null;
  } catch {
    return null;
  }
}

async function statIfExists(filePath) {
  try {
    return await fs.stat(filePath);
  } catch {
    return null;
  }
}

async function snapshotShaFromPath(modelPath) {
  if (!modelPath) return null;
  const candidates = [modelPath];
  try {
    candidates.push(await fs.realpath(modelPath));
  } catch {
    // The backend can expose repo ids or stale paths; missing paths just fall through.
  }
  for (const candidate of candidates) {
    const match = String(candidate).match(/(?:^|\/)snapshots\/([a-f0-9]{40})(?:\/|$)/i);
    if (match) return { sha: match[1], cache_path: candidate };
  }
  return null;
}

async function newestSnapshotSha(repoCacheDir) {
  try {
    const snapshotDir = path.join(repoCacheDir, "snapshots");
    const entries = await fs.readdir(snapshotDir, { withFileTypes: true });
    const snapshots = [];
    for (const entry of entries) {
      if (!entry.isDirectory() || !/^[a-f0-9]{40}$/i.test(entry.name)) continue;
      const snapshotPath = path.join(snapshotDir, entry.name);
      const stats = await statIfExists(snapshotPath);
      snapshots.push({ sha: entry.name, path: snapshotPath, mtimeMs: stats?.mtimeMs || 0 });
    }
    snapshots.sort((left, right) => right.mtimeMs - left.mtimeMs);
    return snapshots[0] || null;
  } catch {
    return null;
  }
}

async function localHfModelSource({ modelPath, repoId, revision }) {
  const pathSnapshot = await snapshotShaFromPath(modelPath);
  if (pathSnapshot) {
    return {
      sha: pathSnapshot.sha,
      timestamp: null,
      branch: revision,
      dirty: null,
      repo_id: repoId,
      revision,
      cache_path: pathSnapshot.cache_path,
      commit_url: hfCommitUrl(repoId, pathSnapshot.sha),
      source: "hf_snapshot_path"
    };
  }

  if (!repoId) return null;
  const repoCacheDir = hfRepoCacheDir(repoId);
  const refSha = await readTextIfExists(path.join(repoCacheDir, "refs", revision || "main"));
  if (refSha) {
    const snapshotPath = path.join(repoCacheDir, "snapshots", refSha);
    return {
      sha: refSha,
      timestamp: null,
      branch: revision || "main",
      dirty: null,
      repo_id: repoId,
      revision: revision || "main",
      cache_path: (await statIfExists(snapshotPath)) ? snapshotPath : repoCacheDir,
      commit_url: hfCommitUrl(repoId, refSha),
      source: "hf_cache_ref"
    };
  }

  const newest = await newestSnapshotSha(repoCacheDir);
  if (!newest) return null;
  return {
    sha: newest.sha,
    timestamp: null,
    branch: revision || "main",
    dirty: null,
    repo_id: repoId,
    revision: revision || "main",
    cache_path: newest.path,
    commit_url: hfCommitUrl(repoId, newest.sha),
    source: "hf_cache_snapshot"
  };
}

async function remoteHfModelSource(repoId, revision) {
  if (!repoId) return null;
  const suffix =
    revision && revision !== "main"
      ? `/revision/${encodeURIComponent(revision)}`
      : "";
  const url = `https://huggingface.co/api/models/${repoId.split("/").map(encodeURIComponent).join("/")}${suffix}`;
  const token = huggingFaceToken();
  try {
    const response = await fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    if (!response.ok) return null;
    const data = await response.json();
    const sha = data?.sha || data?._commit_hash || null;
    if (!sha) return null;
    return {
      sha,
      timestamp: data?.lastModified || null,
      branch: revision || "main",
      dirty: null,
      repo_id: repoId,
      revision: revision || "main",
      cache_path: null,
      commit_url: hfCommitUrl(repoId, sha),
      source: "hf_api"
    };
  } catch {
    return null;
  }
}

function knownHfModelSource(repoId, revision) {
  const sha = repoId ? KNOWN_HF_MODEL_COMMITS[repoId] : null;
  if (!sha) return null;
  return {
    sha,
    timestamp: null,
    branch: revision || "main",
    dirty: null,
    repo_id: repoId,
    revision: revision || "main",
    cache_path: null,
    commit_url: hfCommitUrl(repoId, sha),
    source: "known_hf_commit_hint"
  };
}

async function resolveModelSource({ selectedModel, vllmProcess, modelInfo }) {
  const flags = vllmProcess?.flags || {};
  const modelPath =
    process.env.HF_MODEL_PATH ||
    process.env.MODEL_PATH ||
    flags.model ||
    modelInfo?.root ||
    modelInfo?.id ||
    selectedModel;
  const repoId =
    normalizeHfRepoId(process.env.HF_MODEL_ID || process.env.MODEL_REPO_ID) ||
    normalizeHfRepoId(flags.model) ||
    normalizeHfRepoId(modelInfo?.id) ||
    normalizeHfRepoId(selectedModel);
  const revision = process.env.HF_MODEL_REVISION || process.env.MODEL_REVISION || flags.revision || "main";
  const envSha = process.env.HF_MODEL_SHA || process.env.MODEL_HF_SHA || process.env.MODEL_COMMIT_SHA || null;

  if (envSha) {
    return {
      sha: envSha,
      timestamp: process.env.HF_MODEL_TIMESTAMP || null,
      branch: revision,
      dirty: null,
      repo_id: repoId,
      revision,
      cache_path: modelPath || null,
      commit_url: hfCommitUrl(repoId, envSha),
      source: "environment"
    };
  }

  const local = await localHfModelSource({ modelPath, repoId, revision });
  const remote = await remoteHfModelSource(repoId, revision);
  if (local && remote && local.sha === remote.sha) {
    return { ...local, timestamp: remote.timestamp, source: `${local.source}+hf_api` };
  }
  if (local) return local;
  if (remote) return remote;
  const known = knownHfModelSource(repoId, revision);
  if (known) return known;
  return {
    sha: null,
    timestamp: null,
    branch: revision,
    dirty: null,
    repo_id: repoId,
    revision,
    cache_path: modelPath || null,
    commit_url: null,
    source: "unresolved"
  };
}

async function buildRuntimeInfo(info) {
  const selectedModel = info.models?.[0] || defaultModel;
  const vllmProcess = await findVllmProcess(selectedModel);
  const modelInfo = await fetchBackendModelInfo(info.baseUrl, selectedModel);
  const nim = backend === "nim_local" ? await findNimContainer() : null;
  const flags = vllmProcess?.flags || {};
  const quantization = inferQuantization(selectedModel, flags);
  const source = await resolveModelSource({ selectedModel, vllmProcess, modelInfo });
  return {
    source,
    app_source: appSourceInfo(),
    nim,
    quantization,
    vla: vlaRuntimeInfo(selectedModel, modelInfo),
    vllm: {
      base_url: info.baseUrl,
      model: modelInfo,
      process: vllmProcess
        ? {
            pid: vllmProcess.pid,
            command: vllmProcess.command,
            flags
          }
        : null
    }
  };
}

app.get("/api/models", async (_request, response) => {
  response.json(await listReasonerModels());
});

app.get("/api/active-model", async (_request, response) => {
  const info = await listReasonerModels();
  const runtime = await buildRuntimeInfo(info);
  response.json({
    checkpoint: info.models?.[0] || defaultModel,
    display_name: info.models?.[0] || defaultModel,
    backend: backendDisplayName,
    backend_display_name: backendDisplayName,
    backend_transport: backend,
    backend_implementation: backendImplementation || null,
    base_url: info.baseUrl,
    source: runtime.source,
    app_source: runtime.app_source,
    nim: runtime.nim,
    quantization: runtime.quantization,
    vla: runtime.vla,
    vllm: runtime.vllm,
    warning: info.warning
  });
});

app.get("/api/compare/config", (_request, response) => {
  response.json({
    provider: hostedProviderId(),
    runtime_key_required: true,
    max_models: HOSTED_COMPARE_MAX_MODELS,
    target_families: ["Cosmos", "Qwen", "Nemotron", "Gemma", "Gemini", "Claude", "GPT", "Kimi"],
    curated_models: curatedHostedModels(),
    redaction: {
      credentials: "omitted",
      media_payloads: "omitted",
      prompt_text: "fingerprint_only"
    }
  });
});

app.post("/api/compare/models", async (request, response) => {
  const credential = takeHostedCredential(request.body || {});
  if (!credential) {
    response.status(401).json({
      code: "API_KEY_REQUIRED",
      message: "A runtime API key is required to discover the hosted model catalog.",
      models: curatedHostedModels()
    });
    return;
  }

  try {
    const upstream = await fetchHosted(hostedCatalogEndpointUrl(), credential, "catalog", { method: "GET" }, 20000);
    if (!upstream.ok) {
      response.status(upstream.status === 401 || upstream.status === 403 ? 401 : 502).json({
        code: upstream.status === 401 || upstream.status === 403 ? "API_KEY_REJECTED" : "CATALOG_UNAVAILABLE",
        message: await hostedErrorMessage(upstream, credential),
        models: curatedHostedModels()
      });
      return;
    }
    const data = await upstream.json();
    const models = discoveredHostedModels(data, credential);
    response.json(
      redactSecretsDeep(
        {
          discovered_at: new Date().toISOString(),
          models,
          catalog_capability_token: encodeHostedCatalogCapabilityToken(models),
          target_families: ["Cosmos", "Qwen", "Nemotron", "Gemma", "Gemini", "Claude", "GPT", "Kimi"]
        },
        credential
      )
    );
  } catch (error) {
    response.status(502).json({
      code: "CATALOG_UNAVAILABLE",
      message: redactSecretText(error instanceof Error ? error.message : "Hosted model discovery failed", credential),
      models: curatedHostedModels()
    });
  }
});

app.get("/api/example-media", async (request, response) => {
  const rawUrl = String(request.query.url || "");
  const requestedName = String(request.query.name || "");

  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol !== "https:" || !EXAMPLE_MEDIA_HOSTS.has(parsed.hostname)) {
      response.status(400).json({ message: "Example media URL is not allowed" });
      return;
    }

    const name = requestedName || path.basename(parsed.pathname) || "example-media";
    const upstream = await fetch(parsed);
    if (!upstream.ok) {
      response.status(502).json({ message: `Example media request failed with ${upstream.status}` });
      return;
    }

    const fallbackMime = upstream.headers.get("content-type") || undefined;
    const mime = mediaMime(name, fallbackMime);
    const bytes = Buffer.from(await upstream.arrayBuffer());
    response.json({
      name,
      mime,
      dataUrl: `data:${mime};base64,${bytes.toString("base64")}`
    });
  } catch (error) {
    response.status(502).json({
      message: error instanceof Error ? error.message : "Example media request failed"
    });
  }
});

async function prepareReasonRequest(body = {}, options = {}) {
  const prompt = body.prompt || body.userPrompt || "";
  const systemPrompt = body.systemPrompt || body.system_prompt || "";
  const selectedModel = body.model || defaultModel;
  const params = body.params || {};
  const forceFrameFallback = Boolean(options.forceFrameFallback);

  let mediaDataUrl;
  let mediaKind = null;
  let mediaFrames;
  let frameTempDir;
  let mediaMode = "none";
  let frameCount = 0;
  if (body.video) {
    mediaDataUrl = body.video;
    mediaKind = "video";
  } else if (body.image) {
    mediaDataUrl = body.image;
    mediaKind = "image";
  } else if (body.mediaDataUrl) {
    mediaDataUrl = body.mediaDataUrl;
    mediaKind = body.mediaKind || null;
  }

  if (
    mediaKind === "video" &&
    mediaDataUrl &&
    (forceFrameFallback || (!usesNativeVideoUrl(selectedModel) && frameFallbackAllowed()))
  ) {
    const maxFrames = backend === "nim_local" || forceFrameFallback ? frameFallbackLimit() : undefined;
    const extracted = await extractFrameDataUrls(mediaDataUrl, params.frames_per_second, maxFrames);
    mediaFrames = extracted.frames;
    frameTempDir = extracted.tempDir;
    frameCount = extracted.frameCount;
    mediaMode = "image-frame-fallback";
    mediaDataUrl = undefined;
  } else if (mediaKind === "video" && mediaDataUrl) {
    mediaMode = "video_url";
  } else if (mediaKind === "image" && mediaDataUrl) {
    mediaMode = "image_url";
  }

  return {
    prompt,
    systemPrompt,
    selectedModel,
    params,
    mediaDataUrl,
    mediaKind,
    mediaFrames,
    frameTempDir,
    media: {
      mode: mediaMode,
      frame_count: frameCount || undefined,
      fps: mediaFrames?.length ? params.frames_per_second : undefined,
      fallback_from: forceFrameFallback ? "video_url" : undefined,
      fallback_error: forceFrameFallback ? options.fallbackError : undefined,
      max_frames: mediaMode === "image-frame-fallback" ? frameFallbackLimit() : undefined
    }
  };
}

function preparedReasoningOptions(prepared, options = {}) {
  return {
    model: prepared.selectedModel,
    prompt: prepared.prompt,
    systemPrompt: prepared.systemPrompt,
    mediaDataUrl: prepared.mediaDataUrl,
    mediaKind: prepared.mediaKind,
    mediaFrames: prepared.mediaFrames,
    framesPerSecond: prepared.params.frames_per_second,
    params: prepared.params,
    forceMaxTokens: options.forceMaxTokens === true
  };
}

async function submitPreparedReasoning(prepared, options = {}) {
  const result = await submitReasoning(preparedReasoningOptions(prepared, options));
  result.media = prepared.media;
  return result;
}

function numericParam(params, key, fallback) {
  const value = Number(params?.[key]);
  return Number.isFinite(value) ? value : fallback;
}

async function resolveHostedComparisonMedia(body = {}) {
  const source = body.video || body.image || body.mediaDataUrl;
  const kind = body.video ? "video" : body.image ? "image" : body.mediaKind || null;
  if (!source || !kind) return null;
  if (!String(source).startsWith("data:")) {
    throw new Error("Hosted comparisons require an uploaded or browser-prepared data URL");
  }
  const decoded = dataUrlToBuffer(source);
  const expectedPrefix = kind === "video" ? "video/" : "image/";
  if (!decoded.mime.startsWith(expectedPrefix)) {
    throw new Error(`Comparison media type ${decoded.mime} does not match ${kind}`);
  }
  return {
    dataUrl: source,
    kind,
    mime: decoded.mime,
    bytes: decoded.buffer.length
  };
}

function hostedComparisonPayload({ capabilities, media, mediaFrames, model, params, prompt, systemPrompt }) {
  const supported = sanitizeCapabilities(capabilities, inferredHostedCapabilities(model));
  const content = [];
  let mediaMode = "none";
  if (Array.isArray(mediaFrames) && mediaFrames.length > 0) {
    if (!supported.includes("image")) throw new Error(`${model} does not advertise image input for frame fallback`);
    mediaMode = "image-frame-fallback";
    for (const frame of mediaFrames) {
      content.push({ type: "image_url", image_url: { url: frame } });
    }
  } else if (media?.kind === "image") {
    if (!supported.includes("image")) throw new Error(`${model} does not advertise image input`);
    mediaMode = "image_url";
    content.push({ type: "image_url", image_url: { url: media.dataUrl } });
  } else if (media?.kind === "video") {
    if (!supported.includes("video")) throw new Error(`${model} does not advertise native video input`);
    mediaMode = "video_url";
    content.push({ type: "video_url", video_url: { url: media.dataUrl } });
  }
  content.push({ type: "text", text: prompt || "Describe the provided media." });

  const messages = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  messages.push({ role: "user", content });
  const payload = {
    model,
    messages,
    temperature: numericParam(params, "temperature", 0.6),
    top_p: numericParam(params, "top_p", 0.95),
    stream: false
  };
  payload.max_tokens = Math.max(1, Math.round(numericParam(params, "max_tokens", 512)));
  return { payload, mediaMode, capabilities: supported };
}

function hostedRequestShape({ body, capabilities, media, mediaMode, mediaFrames, model, payload }) {
  const userMessage = payload.messages.find((message) => message.role === "user");
  const contentTypes = Array.isArray(userMessage?.content)
    ? userMessage.content.map((entry) => entry.type).filter(Boolean)
    : ["text"];
  return {
    model,
    advertised_capabilities: sanitizeCapabilities(capabilities, inferredHostedCapabilities(model)),
    transport: "hosted_openai_compatible",
    endpoint_path: "/v1/chat/completions",
    message_roles: payload.messages.map((message) => message.role),
    content_types: contentTypes,
    prompt_chars: String(body.prompt || body.userPrompt || "").length,
    prompt_sha256_16: textFingerprint(body.prompt || body.userPrompt || ""),
    system_prompt_chars: String(body.systemPrompt || body.system_prompt || "").length,
    system_prompt_sha256_16: textFingerprint(body.systemPrompt || body.system_prompt || ""),
    media: media
      ? {
          kind: media.kind,
          mime: media.mime,
          bytes: media.bytes,
          mode: mediaMode,
          frame_count: Array.isArray(mediaFrames) ? mediaFrames.length : undefined,
          payload: "<redacted>"
        }
      : { kind: "none", mode: "none" },
    parameters_sent: definedValues({
      temperature: payload.temperature,
      top_p: payload.top_p,
      max_tokens: payload.max_tokens,
      stream: payload.stream
    }),
    settings_snapshot: summarizeReasonParams(body.params || {})
  };
}

async function postHostedComparison(payload, credential) {
  const promptSecrets = [credential];
  for (const message of payload.messages || []) {
    if (typeof message?.content === "string") promptSecrets.push(message.content);
    if (!Array.isArray(message?.content)) continue;
    for (const item of message.content) {
      if (item?.type === "text" && typeof item.text === "string") promptSecrets.push(item.text);
    }
  }
  let upstream;
  try {
    upstream = await fetchHosted(hostedInferenceEndpointUrl(), credential, "inference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  } catch (error) {
    return {
      ok: false,
      status: 0,
      error: redactSecretText(error instanceof Error ? error.message : "Hosted request failed", promptSecrets)
    };
  }

  const raw = await upstream.text().catch(() => "");
  let data = null;
  try {
    data = raw ? JSON.parse(raw) : null;
  } catch {
    data = null;
  }
  if (!upstream.ok) {
    return {
      ok: false,
      status: upstream.status,
      error: redactSecretText(
        data?.error?.message || data?.message || raw || `Hosted model returned HTTP ${upstream.status}`,
        promptSecrets
      )
    };
  }
  if (!data) {
    return { ok: false, status: upstream.status, error: "Hosted model returned a non-JSON response" };
  }
  return { ok: true, status: upstream.status, data };
}

async function runHostedComparisonCandidate({
  body,
  capabilities,
  credential,
  getFrameFallback,
  media,
  mediaStrategy,
  model
}) {
  const startedAt = Date.now();
  const prompt = body.prompt || body.userPrompt || "";
  const systemPrompt = body.systemPrompt || body.system_prompt || "";
  const params = body.params || {};
  const supported = sanitizeCapabilities(capabilities, inferredHostedCapabilities(model));
  let mediaFrames;
  let attempts = 0;
  const forceSampledFrames = media?.kind === "video" && mediaStrategy === "sampled_frames";
  if (media?.kind === "video" && (forceSampledFrames || !supported.includes("video"))) {
    if (!supported.includes("image")) {
      return {
        role: "candidate",
        provider: hostedProviderId(),
        model,
        status: "error",
        elapsed_ms: elapsedMs(startedAt),
        attempts,
        http_status: null,
        error: forceSampledFrames
          ? `${model} does not advertise image input required by the sampled-frame strategy`
          : `${model} does not advertise video or image input`,
        response: "",
        request_shape: {
          model,
          advertised_capabilities: supported,
          transport: "hosted_openai_compatible",
          media: {
            kind: "video",
            mode: "blocked_by_capability_gate",
            strategy_requested: mediaStrategy,
            payload: "<redacted>"
          },
          prompt_chars: String(prompt).length,
          prompt_sha256_16: textFingerprint(prompt),
          settings_snapshot: summarizeReasonParams(params)
        }
      };
    }
    try {
      mediaFrames = (await getFrameFallback()).frames;
    } catch (error) {
      return {
        role: "candidate",
        provider: hostedProviderId(),
        model,
        status: "error",
        elapsed_ms: elapsedMs(startedAt),
        attempts,
        http_status: null,
        error: `Could not prepare image frames for ${model}. Verify the frame extraction runtime and retry.`,
        response: "",
        request_shape: {
          model,
          advertised_capabilities: supported,
          transport: "hosted_openai_compatible",
          media: {
            kind: "video",
            mode: "image-frame-fallback",
            strategy_requested: mediaStrategy,
            payload: "<redacted>"
          },
          prompt_chars: String(prompt).length,
          prompt_sha256_16: textFingerprint(prompt),
          settings_snapshot: summarizeReasonParams(params)
        }
      };
    }
  }
  if (media?.kind === "image" && !supported.includes("image")) {
    return {
      role: "candidate",
      provider: hostedProviderId(),
      model,
      status: "error",
      elapsed_ms: elapsedMs(startedAt),
      attempts,
      http_status: null,
      error: `${model} does not advertise image input`,
      response: "",
      request_shape: {
        model,
        advertised_capabilities: supported,
        transport: "hosted_openai_compatible",
        media: { kind: "image", mode: "blocked_by_capability_gate", payload: "<redacted>" },
        prompt_chars: String(prompt).length,
        prompt_sha256_16: textFingerprint(prompt),
        settings_snapshot: summarizeReasonParams(params)
      }
    };
  }

  let built = hostedComparisonPayload({ capabilities: supported, media, mediaFrames, model, params, prompt, systemPrompt });
  let posted = await postHostedComparison(built.payload, credential);
  attempts += 1;

  if (
    !posted.ok &&
    media?.kind === "video" &&
    !mediaFrames &&
    supported.includes("image") &&
    (posted.status === 400 || posted.status === 422)
  ) {
    try {
      const fallback = await getFrameFallback();
      mediaFrames = fallback.frames;
      built = hostedComparisonPayload({
        capabilities: supported,
        media,
        mediaFrames,
        model,
        params,
        prompt,
        systemPrompt
      });
      posted = await postHostedComparison(built.payload, credential);
      attempts += 1;
    } catch (error) {
      posted = {
        ok: false,
        status: posted.status,
        error: `${posted.error}; sampled-frame fallback could not be prepared.`
      };
    }
  }

  const requestShape = hostedRequestShape({
    body,
    capabilities: supported,
    media,
    mediaMode: built.mediaMode,
    mediaFrames,
    model,
    payload: built.payload
  });
  if (requestShape.media && typeof requestShape.media === "object") {
    requestShape.media.strategy_requested = mediaStrategy;
  }
  if (!posted.ok) {
    return {
      role: "candidate",
      provider: hostedProviderId(),
      model,
      status: "error",
      elapsed_ms: elapsedMs(startedAt),
      attempts,
      http_status: posted.status || null,
      error: posted.error,
      response: "",
      request_shape: requestShape
    };
  }

  const normalized = normalizeReasonerMessage(posted.data);
  const reportedModel =
    isValidHostedModelId(posted.data?.model) && !hostedModelIdContainsSecret(posted.data.model, credential)
      ? posted.data.model
      : model;
  return {
    role: "candidate",
    provider: hostedProviderId(),
    model: reportedModel,
    requested_model: model,
    status: "success",
    elapsed_ms: elapsedMs(startedAt),
    attempts,
    http_status: posted.status,
    response: redactSecretText(normalized.combined, credential),
    answer: redactSecretText(normalized.answer, credential),
    reasoning: redactSecretText(normalized.reasoning, credential),
    schema: normalized.schema,
    usage: posted.data?.usage || null,
    finish_reason: posted.data?.choices?.[0]?.finish_reason || null,
    request_shape: requestShape
  };
}

async function runLoadedComparisonBaseline(body) {
  const startedAt = Date.now();
  let prepared;
  try {
    prepared = await prepareReasonRequest(body);
    const result = await submitPreparedReasoning(prepared, { forceMaxTokens: true });
    const content = responseContent(result);
    return {
      role: "baseline",
      provider: "loaded_endpoint",
      model: reportSafeModelLabel(result?.openai?.model || prepared.selectedModel),
      status: result.status === "error" || result.error ? "error" : "success",
      elapsed_ms: elapsedMs(startedAt),
      response: redactSecretText(content),
      answer: redactSecretText(result?.content || content),
      reasoning: redactSecretText(result?.reasoning || ""),
      schema: result?.schema || null,
      usage: result?.openai?.usage || result?.raw?.usage || null,
      finish_reason: result?.openai?.choices?.[0]?.finish_reason || null,
      error:
        result.status === "error" || result.error
          ? "Loaded baseline inference did not return a usable result."
          : undefined,
      request_shape: {
        model: reportSafeModelLabel(prepared.selectedModel),
        transport: "loaded_openai_compatible",
        endpoint_path: "/v1/chat/completions",
        prompt_chars: String(prepared.prompt || "").length,
        prompt_sha256_16: textFingerprint(prepared.prompt),
        system_prompt_chars: String(prepared.systemPrompt || "").length,
        system_prompt_sha256_16: textFingerprint(prepared.systemPrompt),
        media: { ...prepared.media, kind: prepared.mediaKind, payload: "<redacted>" },
        parameters_sent: definedValues({
          temperature: result?.payload?.temperature,
          top_p: result?.payload?.top_p,
          max_tokens: result?.payload?.max_tokens,
          repetition_penalty: result?.payload?.repetition_penalty,
          presence_penalty: result?.payload?.presence_penalty,
          top_k: result?.payload?.top_k,
          seed: result?.payload?.seed
        }),
        settings_snapshot: summarizeReasonParams(prepared.params)
      }
    };
  } catch (error) {
    return {
      role: "baseline",
      provider: "loaded_endpoint",
      model: reportSafeModelLabel(body.model, defaultModel),
      status: "error",
      elapsed_ms: elapsedMs(startedAt),
      response: "",
      error: "Loaded baseline inference failed before a result was produced.",
      request_shape: {
        transport: "loaded_openai_compatible",
        prompt_chars: String(body.prompt || body.userPrompt || "").length,
        prompt_sha256_16: textFingerprint(body.prompt || body.userPrompt || ""),
        settings_snapshot: summarizeReasonParams(body.params || {})
      }
    };
  } finally {
    if (prepared?.frameTempDir) await fs.rm(prepared.frameTempDir, { recursive: true, force: true });
  }
}

function comparisonMetadata(body = {}) {
  const comparison = body.comparison || {};
  const mode = comparison.mode === "ablation" ? "ablation" : "spot";
  return {
    mode,
    media_strategy:
      comparison.mediaStrategy === "sampled_frames" ? "sampled_frames" : "native_video_when_confirmed",
    workflow_label: sanitizeLabel(comparison.workflowLabel, mode === "ablation" ? "workflow-ablation" : "spot-check"),
    variant_label: sanitizeLabel(comparison.variantLabel, mode === "ablation" ? "variant" : "current"),
    changed_setting: sanitizeLabel(
      comparison.changedSetting,
      mode === "ablation" ? "system prompt removed" : ""
    ),
    hypothesis: sanitizeLabel(comparison.hypothesis)
  };
}

function comparisonRunVariants(body, metadata) {
  const control = {
    id: metadata.mode === "ablation" ? "control" : "spot",
    label: metadata.mode === "ablation" ? "Control" : metadata.variant_label,
    body: {
      ...body,
      comparison: { ...(body.comparison || {}), variantLabel: metadata.mode === "ablation" ? "Control" : metadata.variant_label }
    }
  };
  if (metadata.mode !== "ablation") return [control];

  const rawVariant = body.comparison?.variant || {};
  const promptOverride =
    typeof rawVariant.promptOverride === "string" ? rawVariant.promptOverride.slice(0, 20000) : "";
  const removeSystemPrompt = rawVariant.removeSystemPrompt !== false;
  const hasTemperatureOverride =
    rawVariant.temperatureOverride !== undefined &&
    rawVariant.temperatureOverride !== null &&
    rawVariant.temperatureOverride !== "";
  const temperatureOverride = hasTemperatureOverride ? Number(rawVariant.temperatureOverride) : Number.NaN;
  const variantParams = { ...(body.params || {}) };
  if (Number.isFinite(temperatureOverride)) {
    variantParams.temperature = Math.max(0, Math.min(2, temperatureOverride));
  }
  const variantBody = {
    ...body,
    prompt: promptOverride.trim() ? promptOverride : body.prompt || body.userPrompt || "",
    userPrompt: promptOverride.trim() ? promptOverride : body.userPrompt,
    systemPrompt: removeSystemPrompt ? "" : body.systemPrompt || body.system_prompt || "",
    system_prompt: removeSystemPrompt ? "" : body.system_prompt,
    params: variantParams,
    comparison: { ...(body.comparison || {}), variantLabel: metadata.variant_label }
  };
  return [
    control,
    {
      id: "variant",
      label: metadata.variant_label,
      body: variantBody
    }
  ];
}

function withComparisonVariant(result, variant) {
  return {
    ...result,
    variant_id: variant.id,
    variant_label: variant.label
  };
}

function sse(response, event, data) {
  if (response.writableEnded) return;
  response.write(`event: ${event}\n`);
  response.write(`data: ${JSON.stringify(data)}\n\n`);
  response.flush?.();
}

function readSseBlocks(buffer) {
  const blocks = buffer.split(/\r?\n\r?\n/);
  return { complete: blocks.slice(0, -1), rest: blocks.at(-1) || "" };
}

function sseData(block) {
  return block
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n")
    .trim();
}

function combinedContent(reasoning, answer) {
  return reasoning ? `<think>\n${reasoning.trim()}\n</think>\n\n${answer || ""}`.trim() : answer || "";
}

function streamingOpenAiResult({ answer, created, finishReason, id, model, object, reasoning, schema, usage }) {
  const message = {
    role: "assistant",
    content: answer || ""
  };
  if (reasoning) {
    message.reasoning_content = reasoning;
  }
  return {
    status: "success",
    message: usage ? `prompt ${usage.prompt_tokens || 0} / completion ${usage.completion_tokens || 0}` : "",
    content: answer,
    reasoning,
    combined_content: combinedContent(reasoning, answer),
    schema,
    openai: {
      id: id || `chatcmpl-byo-stream-${Date.now()}`,
      object: "chat.completion",
      created: created || Math.floor(Date.now() / 1000),
      model: model || defaultModel,
      choices: [
        {
          index: 0,
          message,
          finish_reason: finishReason || "stop"
        }
      ],
      usage: usage || null
    }
  };
}

async function readOpenAiStream({ baseUrl, payload, signal, onChunk }) {
  const streamPayload = { ...payload, stream: true };
  let upstream;
  try {
    upstream = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}`
      },
      body: JSON.stringify(streamPayload),
      signal
    });
  } catch (error) {
    throw new Error(backendFetchFailureMessage(baseUrl, error));
  }

  if (!upstream.ok) {
    let detail = "";
    try {
      const data = await upstream.json();
      detail = data?.error?.message || data?.message || "";
    } catch {
      detail = await upstream.text().catch(() => "");
    }
    throw new Error(detail || `Reasoner stream returned HTTP ${upstream.status}`);
  }

  if (!upstream.body) throw new Error("Reasoner stream did not include a response body");

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawSse = false;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = readSseBlocks(buffer);
    buffer = blocks.rest;

    for (const block of blocks.complete) {
      const data = sseData(block);
      if (!data) continue;
      sawSse = true;
      if (data === "[DONE]") return;
      const parsed = JSON.parse(data);
      if (parsed?.error || (parsed?.message && !parsed?.choices)) {
        throw new Error(parsed?.error?.message || parsed?.message || "Reasoner stream returned an error");
      }
      onChunk(parsed);
    }

    if (done) break;
  }

  if (!sawSse) throw new Error("Reasoner response was not an OpenAI streaming SSE response");
}

function emitFallback(response, result) {
  if (result.status === "error" || result.error) {
    sse(response, "error", { message: result.message || result.error || "Backend request failed" });
    return;
  }
  if (result.reasoning) {
    sse(response, "state", { phase: "reasoning" });
    sse(response, "delta", { channel: "reasoning", text: result.reasoning, schema: result.schema || "plain_content" });
  }
  if (result.content) {
    sse(response, "state", { phase: "answer" });
    sse(response, "delta", { channel: "answer", text: result.content, schema: result.schema || "plain_content" });
  }
  if (result.raw?.usage || result.openai?.usage) sse(response, "usage", result.raw?.usage || result.openai?.usage);
  sse(response, "raw", result);
  sse(response, "state", { phase: "complete" });
}

async function submitLongChunk({ chunk, config, durationSeconds, endpoints, model, params, prompt, signal, systemPrompt }) {
  const text = chunkPrompt({
    chunk,
    durationSeconds,
    prompt,
    preset: config.preset,
    template: config.chunkPromptTemplate
  });
  const { baseUrl, payload, redactedPayload } = buildReasoningPayload({
    model,
    prompt: text,
    systemPrompt: systemPrompt
      ? `${systemPrompt}\n\nFor this long-video chunk pass, return concise JSON only. Do not include hidden reasoning.`
      : "Return concise JSON only for this long-video chunk pass. Do not include hidden reasoning.",
    mediaFrames: chunk.frames.map((frame) => frame.dataUrl),
    framesPerSecond: config.sampleFps,
    params: {
      ...params,
      temperature: Math.min(Number(params.temperature) || 0.6, 0.6),
      max_tokens: config.chunkMaxTokens
    }
  });
  delete payload.mm_processor_kwargs;
  payload.max_tokens = config.chunkMaxTokens;

  const endpoint = endpoints[chunk.index % endpoints.length] || baseUrl;
  const startedAt = Date.now();
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const attemptPayload = attempt === 0 ? payload : { ...payload, max_tokens: Math.min(384, config.chunkMaxTokens) };
      const data = await postOpenAiJson(endpoint, attemptPayload, config.chunkTimeoutMs, signal);
      const normalized = normalizeReasonerMessage(data);
      const content = normalized.answer || normalized.combined;
      const parsed = maybeJsonText(content);
      return {
        index: chunk.index,
        status: "done",
        timeRange: chunk.timeRange,
        startSeconds: chunk.startSeconds,
        endSeconds: chunk.endSeconds,
        content,
        reasoning: normalized.reasoning,
        parsed,
        summary: compactChunkSummary(parsed, content),
        elapsedSeconds: (Date.now() - startedAt) / 1000,
        endpoint,
        payload: redactedPayload
      };
    } catch (error) {
      if (attempt === 1 || signal?.aborted) {
        return {
          index: chunk.index,
          status: "error",
          timeRange: chunk.timeRange,
          startSeconds: chunk.startSeconds,
          endSeconds: chunk.endSeconds,
          error: error instanceof Error ? error.message : "Chunk inference failed",
          elapsedSeconds: (Date.now() - startedAt) / 1000,
          endpoint,
          payload: redactedPayload
        };
      }
    }
  }
  return {
    index: chunk.index,
    status: "error",
    timeRange: chunk.timeRange,
    error: "Chunk inference failed"
  };
}

async function stitchLongVideo({ chunks, config, durationSeconds, endpoints, failedChunks, model, onProgress, params, prompt, signal, warnings }) {
  const reducerMode = stitchReducerMode(config);
  if (reducerMode !== "model") {
    const content = localLongVideoStitch({
      chunks,
      durationSeconds,
      failedChunks,
      onProgress,
      prompt,
      preset: config.preset,
      warnings
    });
    return {
      status: "success",
      content,
      reasoning: "",
      combined_content: content,
      schema: /\bjson\b/i.test(String(prompt || "")) ? "local_long_video_json" : "local_long_video_timeline",
      openai: {
        id: `chatcmpl-byo-long-local-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [
          {
            index: 0,
            message: { role: "assistant", content },
            finish_reason: "stop"
          }
        ],
        usage: null
      }
    };
  }

  onProgress?.("collect_outputs", { status: "running", progress: 50, detail: `Reading ${chunks.length} chunk outputs` });
  const compactChunks = chunks.map((chunk) => compactChunkForReducer(chunk));
  onProgress?.("collect_outputs", { status: "done", progress: 100, detail: `${compactChunks.length} chunks collected` });
  onProgress?.("normalize_events", { status: "running", progress: 60, detail: "Flattening summaries and events" });
  const text = reducerPrompt({
    chunks,
    durationSeconds,
    failedChunks,
    prompt,
    preset: config.preset,
    warnings,
    template: config.reducerPromptTemplate
  });
  onProgress?.("normalize_events", { status: "done", progress: 100, detail: "Chunk JSON normalized" });
  onProgress?.("compact_prompt", { status: "done", progress: 100, detail: `${text.length.toLocaleString()} reducer prompt chars` });
  const { baseUrl, payload, redactedPayload } = buildReasoningPayload({
    model,
    prompt: text,
    systemPrompt: "You are a timeline reducer for long video analysis. Return the final answer only.",
    params: {
      ...params,
      temperature: Math.min(Number(params.temperature) || 0.6, 0.6),
      max_tokens: config.finalMaxTokens
    }
  });
  delete payload.mm_processor_kwargs;
  payload.max_tokens = config.finalMaxTokens;
  const endpoint = endpoints[0] || baseUrl;
  const modelReducerStartedAt = Date.now();
  const estimateSeconds = estimateStitchSeconds({ chunks, mode: "model", promptChars: text.length });
  let reducerProgressTimer = null;
  try {
    onProgress?.("model_reducer", {
      status: "running",
      progress: 5,
      detail: `Waiting on reducer NIM; estimate ~${formatShortDuration(estimateSeconds)}`
    });
    reducerProgressTimer = setInterval(() => {
      const elapsed = (Date.now() - modelReducerStartedAt) / 1000;
      onProgress?.("model_reducer", {
        status: "running",
        progress: Math.min(95, 5 + (elapsed / Math.max(1, estimateSeconds)) * 90),
        detail: `Reducer NIM elapsed ${formatShortDuration(elapsed)}; estimate ~${formatShortDuration(Math.max(0, estimateSeconds - elapsed))}`
      });
    }, 2500);
    const data = await postOpenAiJson(endpoint, payload, Math.max(config.chunkTimeoutMs, 90000), signal);
    if (reducerProgressTimer) clearInterval(reducerProgressTimer);
    onProgress?.("model_reducer", { status: "done", progress: 100, detail: "Reducer NIM returned" });
    const normalized = normalizeReasonerMessage(data);
    onProgress?.("build_final", { status: "done", progress: 100, detail: "Final reducer response ready" });
    return {
      status: "success",
      content: normalized.answer || normalized.combined,
      reasoning: normalized.reasoning,
      combined_content: normalized.combined,
      schema: normalized.schema,
      openai: normalized.normalized,
      raw: data,
      payload: redactedPayload
    };
  } catch (error) {
    if (reducerProgressTimer) clearInterval(reducerProgressTimer);
    onProgress?.("model_reducer", {
      status: "error",
      progress: 100,
      detail: error instanceof Error ? error.message : "Reducer NIM failed"
    });
    onProgress?.("build_final", { status: "running", progress: 45, detail: "Rendering local fallback" });
    const fallback = [
      "Long video analysis completed, but the final stitching pass failed.",
      "",
      ...chunks
        .filter((chunk) => chunk.status === "done")
        .map((chunk) => `- ${chunk.timeRange}: ${chunk.parsed?.summary || chunk.content || "No summary"}`),
      ...failedChunks.map((chunk) => `- ${chunk.timeRange}: failed (${chunk.error})`)
    ].join("\n");
    onProgress?.("build_final", { status: "done", progress: 100, detail: "Local fallback ready" });
    return {
      status: "success",
      content: fallback,
      reasoning: "",
      combined_content: fallback,
      schema: "local_chunk_fallback",
      message: error instanceof Error ? `Reducer failed: ${error.message}` : "Reducer failed",
      payload: redactedPayload
    };
  } finally {
    if (reducerProgressTimer) clearInterval(reducerProgressTimer);
  }
}

app.post("/api/compare/run", async (request, response) => {
  const body = request.body || {};
  const credential = takeHostedCredential(body);
  if (!credential) {
    response.status(401).json({
      code: "API_KEY_REQUIRED",
      message: "A runtime API key is required for hosted comparisons."
    });
    return;
  }

  const rawCatalogCapabilityToken = body.catalogCapabilityToken;
  if (Object.prototype.hasOwnProperty.call(body, "catalogCapabilityToken")) {
    delete body.catalogCapabilityToken;
  }
  let catalogCapabilityProfiles;
  try {
    catalogCapabilityProfiles = decodeHostedCatalogCapabilityToken(rawCatalogCapabilityToken);
  } catch {
    response.status(400).json({
      code: "INVALID_CATALOG_CONTEXT",
      message: "The catalog context is invalid or expired. Discover models again before comparing."
    });
    return;
  }

  const workflowFields = JSON.stringify({
    prompt: body.prompt,
    userPrompt: body.userPrompt,
    systemPrompt: body.systemPrompt,
    system_prompt: body.system_prompt,
    comparison: body.comparison,
    params: body.params
  });
  if (workflowFields.includes(credential)) {
    response.status(400).json({
      code: "INVALID_WORKFLOW_INPUT",
      message: "Credentials must not be included in prompt or workflow fields."
    });
    return;
  }

  const baselineModelId = typeof body.model === "string" ? body.model : "";
  if (hostedModelIdContainsSecret(baselineModelId, credential)) {
    response.status(400).json({ code: "INVALID_MODEL_ID", message: "One or more model IDs are invalid." });
    return;
  }

  const rawRequestedModels = Array.isArray(body.models) ? body.models : [];
  const validatedEntries = [];
  for (const entry of rawRequestedModels) {
    const rawId = typeof entry === "string" ? entry : entry?.id;
    const id = typeof rawId === "string" ? rawId.trim() : "";
    if (!isValidHostedModelId(id) || hostedModelIdContainsSecret(id, credential)) {
      response.status(400).json({ code: "INVALID_MODEL_ID", message: "One or more model IDs are invalid." });
      return;
    }
    validatedEntries.push({ id });
  }

  const metadata = comparisonMetadata(body);
  const requestedModelMap = new Map();
  for (const { id } of validatedEntries) {
    if (requestedModelMap.has(id)) continue;
    const catalogProfile = catalogCapabilityProfiles.get(id);
    const explicitSampledFrames = metadata.media_strategy === "sampled_frames";
    const capabilities = catalogProfile?.capabilities ||
      (explicitSampledFrames ? ["image", "text"] : inferredHostedCapabilities(id));
    const capabilitySource = catalogProfile?.source ||
      (explicitSampledFrames
        ? "explicit_strategy"
        : curatedModelForDiscoveredId(id)
          ? "confirmed_contract"
          : "unconfirmed");
    requestedModelMap.set(id, {
      id,
      capabilities,
      capability_source: capabilitySource
    });
    if (requestedModelMap.size >= HOSTED_COMPARE_MAX_MODELS) break;
  }
  const requestedModels = Array.from(requestedModelMap.values());
  if (requestedModels.length === 0) {
    response.status(400).json({ code: "MODELS_REQUIRED", message: "Select at least one hosted model." });
    return;
  }

  let hostedMedia = null;
  let hostedMediaError = null;
  try {
    hostedMedia = await resolveHostedComparisonMedia(body);
  } catch (error) {
    hostedMediaError = redactSecretText(error instanceof Error ? error.message : "Could not prepare comparison media");
  }

  let frameFallbackPromise = null;
  const getFrameFallback = async () => {
    if (!hostedMedia || hostedMedia.kind !== "video") throw new Error("Video frame fallback is not available");
    if (!frameFallbackPromise) {
      frameFallbackPromise = extractFrameDataUrls(
        hostedMedia.dataUrl,
        numericParam(body.params || {}, "frames_per_second", 1),
        HOSTED_COMPARE_FRAME_LIMIT
      );
    }
    return frameFallbackPromise;
  };

  try {
    const variants = comparisonRunVariants(body, metadata);
    const resultPromises = variants.flatMap((variant) => {
      const baselinePromise = runLoadedComparisonBaseline(variant.body).then((result) =>
        withComparisonVariant(result, variant)
      );
      const candidatePromises = requestedModels.map((candidate) =>
        (hostedMediaError
          ? Promise.resolve({
              role: "candidate",
              provider: hostedProviderId(),
              model: candidate.id,
              status: "error",
              elapsed_ms: 0,
              response: "",
              error: hostedMediaError,
              request_shape: {
                model: candidate.id,
                advertised_capabilities: candidate.capabilities,
                capability_source: candidate.capability_source,
                transport: "hosted_openai_compatible",
                media: {
                  kind: variant.body.video ? "video" : variant.body.image ? "image" : "unknown",
                  strategy_requested: metadata.media_strategy,
                  payload: "<redacted>"
                },
                prompt_chars: String(variant.body.prompt || variant.body.userPrompt || "").length,
                prompt_sha256_16: textFingerprint(variant.body.prompt || variant.body.userPrompt || ""),
                settings_snapshot: summarizeReasonParams(variant.body.params || {})
              }
            })
          : runHostedComparisonCandidate({
              body: variant.body,
              capabilities: candidate.capabilities,
              credential,
              getFrameFallback,
              media: hostedMedia,
              mediaStrategy: metadata.media_strategy,
              model: candidate.id
            })
        ).then((result) =>
          withComparisonVariant(
            {
              ...result,
              request_shape: {
                ...(result.request_shape || {}),
                capability_source: candidate.capability_source
              }
            },
            variant
          )
        )
      );
      return [baselinePromise, ...candidatePromises];
    });
    const results = await Promise.all(resultPromises);
    const baseline = results.find((result) => result.role === "baseline") || results[0];
    response.json(
      redactSecretsDeep(
        {
          schema_version: "1.0",
          generated_at: new Date().toISOString(),
          status: results.every((result) => result.status === "success") ? "success" : "partial",
          comparison: metadata,
          baseline_model: baseline?.model || reportSafeModelLabel(body.model, defaultModel),
          variants: variants.map((variant) => ({ id: variant.id, label: variant.label })),
          input: {
            media_name: sanitizeLabel(body.mediaName, hostedMedia?.kind ? "current-media" : "none"),
            media_kind: hostedMedia?.kind || (body.video ? "video" : body.image ? "image" : "none"),
            media_mime: hostedMedia?.mime || null,
            media_bytes: hostedMedia?.bytes || null,
            media_payload: hostedMedia ? "<redacted>" : null,
            prompt_chars: String(body.prompt || body.userPrompt || "").length,
            prompt_sha256_16: textFingerprint(body.prompt || body.userPrompt || ""),
            prompt_text: "<redacted>",
            system_prompt_chars: String(body.systemPrompt || body.system_prompt || "").length,
            system_prompt_sha256_16: textFingerprint(body.systemPrompt || body.system_prompt || ""),
            system_prompt_text: "<redacted>"
          },
          settings: summarizeReasonParams(body.params || {}),
          redaction: {
            credentials: "omitted",
            authorization_headers: "omitted",
            media_payloads: "omitted",
            prompt_text: "fingerprint_only"
          },
          results
        },
        credential
      )
    );
  } catch (error) {
    response.status(502).json(
      redactSecretsDeep(
        {
          code: "COMPARISON_FAILED",
          message: redactSecretText(error instanceof Error ? error.message : "Comparison failed", credential)
        },
        credential
      )
    );
  } finally {
    if (frameFallbackPromise) {
      try {
        const extracted = await frameFallbackPromise;
        if (extracted?.tempDir) await fs.rm(extracted.tempDir, { recursive: true, force: true });
      } catch {
        // Frame extraction errors are already represented in candidate results.
      }
    }
  }
});

app.post("/api/reason", async (request, response) => {
  let prepared;
  let fallbackPrepared;
  const requestId = requestLogId("reason");
  const startedAt = Date.now();
  try {
    prepared = await prepareReasonRequest(request.body || {});
    logReasonRequest(requestId, "prepared", {
      route: "/api/reason",
      ...summarizePreparedReasonRequest(prepared)
    });
    let result = await submitPreparedReasoning(prepared);
    if (result.status === "error" && shouldRetryWithFrameFallback(prepared, result)) {
      console.warn(
        `[vite-build-reason] native video_url rejected by NIM; retrying with ${frameFallbackLimit()} image frames`
      );
      logReasonRequest(requestId, "fallback-start", {
        reason: result.message,
        frame_limit: frameFallbackLimit()
      });
      fallbackPrepared = await prepareReasonRequest(request.body || {}, {
        forceFrameFallback: true,
        fallbackError: result.message
      });
      logReasonRequest(requestId, "fallback-prepared", summarizePreparedReasonRequest(fallbackPrepared));
      result = await submitPreparedReasoning(fallbackPrepared);
    }
    logReasonRequest(requestId, "complete", {
      elapsed_ms: elapsedMs(startedAt),
      used_fallback: Boolean(fallbackPrepared),
      ...summarizeReasoningResult(result)
    });
    const httpStatus = result.status === "error" ? 502 : 200;
    response.status(httpStatus).json(result);
  } catch (error) {
    logReasonError(requestId, "error", error, { elapsed_ms: elapsedMs(startedAt) });
    response.status(502).json({
      status: "error",
      message: error instanceof Error ? error.message : "Backend request failed",
      files: []
    });
  } finally {
    if (prepared?.frameTempDir) await fs.rm(prepared.frameTempDir, { recursive: true, force: true });
    if (fallbackPrepared?.frameTempDir) await fs.rm(fallbackPrepared.frameTempDir, { recursive: true, force: true });
  }
});

app.post("/api/reason/planning/stream", async (request, response) => {
  let extracted;
  const upstreamAbort = new AbortController();
  let clientClosed = false;
  const startedAt = Date.now();

  const abortForClosedClient = () => {
    if (response.writableEnded) return;
    clientClosed = true;
    upstreamAbort.abort();
  };
  request.on("aborted", abortForClosedClient);
  response.on("close", abortForClosedClient);

  response.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no"
  });
  response.flushHeaders?.();

  const elapsedSeconds = () => (Date.now() - startedAt) / 1000;
  const emitLog = ({ label, detail, elapsedSeconds: explicitElapsed, level = "info" } = {}) => {
    if (!label) return;
    sse(response, "log", {
      label,
      detail,
      level,
      elapsedSeconds: Number.isFinite(Number(explicitElapsed)) ? Number(explicitElapsed) : elapsedSeconds()
    });
  };

  try {
    const body = request.body || {};
    const selectedModel = body.model || defaultModel;
    const params = body.params || {};
    const mediaSource = body.video || body.image || body.mediaDataUrl;
    const mediaKind = body.image ? "image" : body.video ? "video" : body.mediaKind || "video";
    if (!mediaSource) throw new Error("Planning mode requires an image or video input");

    const rawFrames = Array.isArray(body.planningFrames) ? body.planningFrames : [];
    const planningFrames = rawFrames
      .map((frame, index) => ({
        index,
        phase: String(frame.phase || `Frame ${index + 1}`),
        time: String(frame.time || (index === 0 ? "00:00.00" : "00:02.00")),
        mode: String(frame.renderMode || frame.mode || "").toLowerCase()
      }))
      .filter((frame) => frame.mode === "perception" || frame.mode === "trajectory");
    const stages = planningFrames.length
      ? planningFrames
      : [
          { index: 0, phase: "Perceive", time: "00:00.00", mode: "perception" },
          { index: 1, phase: "Grasp", time: "00:02.00", mode: "trajectory" }
        ];

    emitLog({
      label: "Planning request received",
      detail: `${stages.length} staged NIM call${stages.length === 1 ? "" : "s"} · model ${selectedModel} · ${mediaKind} input`
    });
    emitLog({
      label: "Coordinate frames queued",
      detail: stages.map((stage) => `${stage.phase} ${stage.time} (${stage.mode})`).join(" · ")
    });
    sse(response, "state", { phase: "preparing_media", note: "Extracting Perceive and Grasp coordinate frames" });
    const extractStartedAt = Date.now();
    emitLog({
      label: "Extracting planning frames",
      detail:
        mediaKind === "image"
          ? "Using the uploaded image directly for each staged coordinate frame."
          : `Decoding ${stages.length} target video frame${stages.length === 1 ? "" : "s"} for image fallback.`
    });
    extracted = await extractPlanningFrameDataUrls(mediaSource, stages, mediaKind);
    emitLog({
      label: "Planning frames ready",
      detail: `${extracted.frames.length} frame${extracted.frames.length === 1 ? "" : "s"} ready in ${(
        (Date.now() - extractStartedAt) /
        1000
      ).toFixed(1)}s · ${extracted.frames.map((frame) => `${frame.phase} actual ${frame.actualTime}`).join(" · ")}`
    });

    const stageResults = [];
    for (const stage of extracted.frames) {
      if (upstreamAbort.signal.aborted) throw new Error("Planning run stopped");
      sse(response, "state", {
        phase: "waiting_first_token",
        note: `Running ${stage.phase} ${stage.mode} pass on ${stage.time}`
      });
      emitLog({
        label: `${stage.phase} pass started`,
        detail: `${stage.mode} · requested ${stage.time} · actual ${stage.actualTime || stage.time}`
      });
      const result = await submitPlanningStage({
        model: selectedModel,
        onLog: emitLog,
        params,
        signal: upstreamAbort.signal,
        stage,
        systemPrompt: body.systemPrompt || body.system_prompt || "",
        taskPrompt: body.prompt || body.userPrompt || ""
      });
      stageResults.push(result);
      const parsedCount =
        result.mode === "perception"
          ? arrayFromParsed(result.parsed, ["perception", "objects", "detections"]).length
          : arrayFromParsed(result.parsed, ["trajectory", "waypoints", "path", "planned_path", "end_effector_path", "grasp_plan"]).filter(
              recordHasPoint
            ).length;
      emitLog({
        label: `${result.phase} parsed`,
        detail: `${result.mode} pass produced ${parsedCount} ${result.mode === "perception" ? "object box" : "waypoint"}${
          parsedCount === 1 ? "" : "s"
        }`
      });
      sse(response, "planning_stage", {
        phase: result.phase,
        mode: result.mode,
        time: result.time,
        elapsedSeconds: result.elapsedSeconds,
        content: result.content
      });
    }

    emitLog({
      label: "Merging planning stages",
      detail: `${stageResults.length} stage result${stageResults.length === 1 ? "" : "s"} received; combining perception and trajectory JSON.`
    });
    const merged = mergePlanningStages(stageResults);
    const answer = `\`\`\`json\n${JSON.stringify(merged, null, 2)}\n\`\`\``;
    const stageReasoning = stageResults
      .map((stage) => {
        const reasoning = String(stage.reasoning || "").trim();
        return reasoning ? `${stage.phase} ${stage.mode} pass (${stage.time}):\n${reasoning}` : "";
      })
      .filter(Boolean)
      .join("\n\n");
    const raw = {
      ...streamingOpenAiResult({
        answer,
        created: Math.floor(Date.now() / 1000),
        finishReason: "stop",
        id: `chatcmpl-byo-planning-${Date.now()}`,
        model: selectedModel,
        object: "chat.completion",
        reasoning: stageReasoning,
        schema: "two_call_robot_planning",
        usage: null
      }),
      planning_stages: stageResults.map((stage) => ({
        phase: stage.phase,
        mode: stage.mode,
        time: stage.time,
        actual_time: stage.actualTime,
        elapsed_seconds: stage.elapsedSeconds,
        content: stage.content,
        reasoning: stage.reasoning,
        parsed: stage.parsed,
        payload: stage.payload
      })),
      media: {
        mode: mediaKind === "image" ? "planning-image-frames" : "planning-video-keyframes",
        frame_count: extracted.frames.length,
        coordinate_frames: extracted.frames.map((frame) => ({
          phase: frame.phase,
          mode: frame.mode,
          requested_time: frame.time,
          actual_time: frame.actualTime
        }))
      }
    };
    emitLog({
      label: "Planning trace ready",
      detail: `${merged.perception?.length || 0} perception record${merged.perception?.length === 1 ? "" : "s"} · ${
        merged.trajectory?.length || 0
      } trajectory record${merged.trajectory?.length === 1 ? "" : "s"} · total ${elapsedSeconds().toFixed(1)}s`
    });
    sse(response, "state", { phase: "answer" });
    sse(response, "delta", { channel: "answer", text: answer, schema: "two_call_robot_planning" });
    sse(response, "raw", raw);
    sse(response, "state", { phase: "complete" });
  } catch (error) {
    if (!clientClosed && !response.writableEnded) {
      emitLog({
        label: "Planning stream failed",
        detail: error instanceof Error ? error.message : "Robot planning stream failed",
        level: "error"
      });
      sse(response, "error", { message: error instanceof Error ? error.message : "Robot planning stream failed" });
    }
  } finally {
    if (extracted?.tempDir) await fs.rm(extracted.tempDir, { recursive: true, force: true });
    if (!response.writableEnded) response.end();
  }
});

app.post("/api/reason/long/stream", async (request, response) => {
  const upstreamAbort = new AbortController();
  let clientClosed = false;
  let extracted = null;
  const startedAt = Date.now();
  const requestId = requestLogId("long");

  const abortForClosedClient = () => {
    if (response.writableEnded) return;
    clientClosed = true;
    upstreamAbort.abort();
  };
  request.on("aborted", abortForClosedClient);
  response.on("close", abortForClosedClient);

  response.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no"
  });
  response.flushHeaders?.();

  function elapsedSeconds() {
    return (Date.now() - startedAt) / 1000;
  }

  try {
    const stepState = {
      read_video: { label: "Read video", status: "pending", progress: 0, detail: "" },
      decode_frames: { label: "Decode frames", status: "pending", progress: 0, detail: "" },
      plan_chunks: { label: "Plan chunks", status: "pending", progress: 0, detail: "" },
      run_chunks: { label: "Run chunks", status: "pending", progress: 0, detail: "" },
      stitch_timeline: { label: "Stitch timeline", status: "pending", progress: 0, detail: "" }
    };
    const stepPayload = () =>
      Object.entries(stepState).map(([key, value]) => ({
        key,
        label: value.label,
        status: value.status,
        progress: Math.max(0, Math.min(100, Math.round(Number(value.progress) || 0))),
        detail: value.detail || "",
        etaSeconds: Number.isFinite(Number(value.etaSeconds)) ? Number(value.etaSeconds) : undefined,
        elapsedSeconds: Number.isFinite(Number(value.elapsedSeconds)) ? Number(value.elapsedSeconds) : undefined,
        substeps: Array.isArray(value.substeps)
          ? value.substeps.map((step) => ({
              key: step.key,
              label: step.label,
              status: step.status,
              progress: Math.max(0, Math.min(100, Math.round(Number(step.progress) || 0))),
              detail: step.detail || "",
              etaSeconds: Number.isFinite(Number(step.etaSeconds)) ? Number(step.etaSeconds) : undefined,
              elapsedSeconds: Number.isFinite(Number(step.elapsedSeconds)) ? Number(step.elapsedSeconds) : undefined
            }))
          : undefined
      }));
    const updateStep = (key, patch) => {
      stepState[key] = { ...stepState[key], ...patch };
    };
    const emitLongState = (payload) => {
      sse(response, "long_state", {
        ...payload,
        steps: stepPayload()
      });
    };
    const body = request.body || {};
    const params = body.params || {};
    const selectedModel = body.model || defaultModel;
    const mediaSource = body.video || (body.mediaKind === "video" ? body.mediaDataUrl : "");
    if (!mediaSource) throw new Error("Long Video Analysis requires an MP4/video input.");

    const requestedLongFps = params.frames_per_second ?? body.framesPerSecond ?? body.longVideoFps;
    const longVideoOptions = longVideoRequestOptions(body);
    let config = longVideoPresetConfig(body.preset, 0, body.concurrency, requestedLongFps, longVideoOptions);
    logReasonRequest(requestId, "start", {
      route: "/api/reason/long/stream",
      model: selectedModel,
      preset: body.preset || config.preset,
      prompt_chars: String(body.prompt || body.userPrompt || "").length,
      system_prompt_chars: String(body.systemPrompt || body.system_prompt || "").length,
      schema_hints: reasonSchemaHints(body.prompt || body.userPrompt || "", body.systemPrompt || body.system_prompt || ""),
      requested_fps: requestedLongFps,
      requested_concurrency: body.concurrency,
      media_mode: "long-video-image-chunks",
      params: summarizeReasonParams(params)
    });
    updateStep("read_video", { status: "running", progress: 10, detail: "Opening media" });
    emitLongState({
      phase: "media_scan",
      message: "Scanning video and extracting timestamped frames",
      percent: 2,
      elapsedSeconds: elapsedSeconds()
    });
    const handleDecodeProgress = (event) => {
      if (event.phase === "loading") {
        updateStep("read_video", { status: "running", progress: 45, detail: "Loading video bytes" });
        emitLongState({
          phase: "media_scan",
          message: "Loading video bytes",
          percent: 2,
          elapsedSeconds: elapsedSeconds()
        });
        return;
      }
      if (event.phase === "decode_start") {
        updateStep("read_video", { status: "done", progress: 100, detail: "Video loaded" });
        updateStep("decode_frames", { status: "running", progress: 2, detail: "Starting decoder" });
      } else if (event.phase === "opened") {
        const target = Number(event.target_frames) || config.maxFrames;
        updateStep("read_video", { status: "done", progress: 100, detail: "Video metadata read" });
        updateStep("decode_frames", {
          status: "running",
          progress: 3,
          detail: `Target ${target} frames at ${config.sampleFps} fps`
        });
      } else if (event.phase === "decoded") {
        const extractedFrames = Number(event.extracted) || 0;
        const target = Number(event.target_frames) || config.maxFrames || extractedFrames || 1;
        const decodePercent = Math.max(0, Math.min(100, Number(event.percent) || (extractedFrames / target) * 100));
        updateStep("decode_frames", {
          status: "running",
          progress: decodePercent,
          detail: `${extractedFrames}/${target} frames${event.timestamp_text ? ` through ${event.timestamp_text}` : ""}`
        });
        emitLongState({
          phase: "media_decode",
          message: `Decoding frames ${extractedFrames}/${target}`,
          percent: Math.min(7, Math.round(2 + decodePercent * 0.05)),
          frameCount: extractedFrames,
          elapsedSeconds: elapsedSeconds()
        });
      } else if (event.phase === "complete") {
        updateStep("decode_frames", {
          status: "done",
          progress: 100,
          detail: `${Number(event.extracted) || 0} frames extracted`
        });
      }
    };
    extracted = await extractLongVideoFrames(mediaSource, config, handleDecodeProgress);
    config = longVideoPresetConfig(
      body.preset,
      extracted.durationSeconds,
      body.concurrency,
      requestedLongFps,
      longVideoOptions
    );

    if (Math.abs((extracted.sampleFps || 0) - config.sampleFps) > 0.01 || extracted.frames.length > config.maxFrames) {
      await fs.rm(extracted.tempDir, { recursive: true, force: true });
      updateStep("decode_frames", {
        status: "running",
        progress: 0,
        detail: `Re-decoding at ${config.sampleFps} fps after reading duration`
      });
      extracted = await extractLongVideoFrames(mediaSource, config, handleDecodeProgress);
    }

    updateStep("decode_frames", {
      status: "done",
      progress: 100,
      detail: `${extracted.frames.length} frames extracted`
    });
    updateStep("plan_chunks", { status: "running", progress: 35, detail: "Building timestamp windows" });
    const effectiveSampleFps =
      extracted.durationSeconds > 0
        ? Number((extracted.frames.length / extracted.durationSeconds).toFixed(2))
        : extracted.sampleFps;
    const chunks = chunkLongFrames(extracted.frames, config);
    try {
      setMaxListeners(Math.max(64, chunks.length * 2 + config.concurrency + 16), upstreamAbort.signal);
    } catch {
      // Best-effort only; older runtimes still work without raising the listener ceiling.
    }
    updateStep("plan_chunks", { status: "done", progress: 100, detail: `${chunks.length} chunks planned` });
    updateStep("run_chunks", { status: "pending", progress: 0, detail: `${config.concurrency} concurrent requests` });
    const warnings = longVideoWarnings(extracted.durationSeconds, extracted.frames.length, config.preset, config);
    const { baseUrl } = buildReasoningPayload({ model: selectedModel, prompt: "probe", params: {} });
    const endpoints = reasonerEndpointPool(baseUrl);
    logReasonRequest(requestId, "plan", {
      duration_seconds: extracted.durationSeconds,
      source_fps: extracted.sourceFps,
      source_width: extracted.sourceWidth,
      source_height: extracted.sourceHeight,
      extracted_frames: extracted.frames.length,
      requested_fps: config.requestedFps || null,
      sample_fps: effectiveSampleFps,
      extraction_fps: extracted.sampleFps,
      frame_budget: config.maxFrames,
      chunk_count: chunks.length,
      chunk_size: config.chunkSize,
      chunk_overlap: config.overlap,
      max_images_per_chunk: LONG_VIDEO_MAX_IMAGES_PER_CHUNK,
      concurrency: config.concurrency,
      reducer_mode: stitchReducerMode(config),
      endpoint_count: endpoints.length,
      first_range: chunks[0]?.timeRange,
      last_range: chunks.at(-1)?.timeRange,
      warnings
    });
    const progress = new Map();
    let completed = 0;
    let failed = 0;
    let latencyTotal = 0;
    let latencyCount = 0;

    sse(response, "long_plan", {
      phase: "extraction_plan",
      preset: config.preset,
      durationSeconds: extracted.durationSeconds,
      durationText: formatTimestamp(extracted.durationSeconds),
      sourceFps: extracted.sourceFps,
      sourceWidth: extracted.sourceWidth,
      sourceHeight: extracted.sourceHeight,
      frameCount: extracted.frames.length,
      requestedFps: config.requestedFps || null,
      sampleFps: effectiveSampleFps,
      extractionFps: extracted.sampleFps,
      frameLimit: config.frameLimit,
      frameBudget: config.maxFrames,
      requestedFrameBudget: config.requestedFrameBudget,
      chunkSize: config.chunkSize,
      chunkOverlap: config.overlap,
      maxImagesPerChunk: LONG_VIDEO_MAX_IMAGES_PER_CHUNK,
      totalChunks: chunks.length,
      concurrency: config.concurrency,
      maxConcurrency: config.maxConcurrency,
      reducerMode: stitchReducerMode(config),
      chunkPromptTemplateEnabled: Boolean(config.chunkPromptTemplate),
      reducerPromptTemplateEnabled: Boolean(config.reducerPromptTemplate),
      endpoints,
      warnings,
      steps: stepPayload(),
      chunks: chunks.map((chunk) => ({
        index: chunk.index,
        status: "queued",
        timeRange: chunk.timeRange,
        frameCount: chunk.frames.length,
        thumbnailUrl: chunk.frames[0]?.dataUrl || null
      })),
      percent: 8,
      elapsedSeconds: elapsedSeconds()
    });

    const emitProgress = (phase, message) => {
      const total = Math.max(1, chunks.length);
      const active = [...progress.values()].filter((item) => item.status === "running").length;
      const doneRatio = (completed + failed) / total;
      const averageLatency = latencyCount > 0 ? latencyTotal / latencyCount : null;
      const remaining = Math.max(0, total - completed - failed);
      const etaSeconds = averageLatency ? (remaining / Math.max(1, config.concurrency)) * averageLatency : null;
      updateStep("run_chunks", {
        status: remaining > 0 ? "running" : failed > 0 ? "error" : "done",
        progress: doneRatio * 100,
        detail: `${completed + failed}/${chunks.length} chunks complete; ${active} running`
      });
      emitLongState({
        phase,
        message,
        completedChunks: completed,
        failedChunks: failed,
        runningChunks: active,
        totalChunks: chunks.length,
        percent: Math.min(95, Math.round(8 + doneRatio * 82)),
        elapsedSeconds: elapsedSeconds(),
        etaSeconds,
        concurrency: config.concurrency,
        maxConcurrency: config.maxConcurrency
      });
    };

    emitProgress("chunking", `Running chunks 0/${chunks.length}`);
    const results = await runConcurrent(chunks, config.concurrency, async (chunk) => {
      progress.set(chunk.index, { status: "running" });
      sse(response, "long_chunk", {
        index: chunk.index,
        status: "running",
        timeRange: chunk.timeRange,
        frameCount: chunk.frames.length
      });
      emitProgress("chunking", `Running chunks ${completed + failed}/${chunks.length}`);

      const result = await submitLongChunk({
        chunk,
        config,
        durationSeconds: extracted.durationSeconds,
        endpoints,
        model: selectedModel,
        params,
        prompt: body.prompt || body.userPrompt || "",
        signal: upstreamAbort.signal,
        systemPrompt: body.systemPrompt || body.system_prompt || ""
      });

      progress.set(chunk.index, { status: result.status });
      if (result.status === "done") {
        completed += 1;
        latencyTotal += result.elapsedSeconds || 0;
        latencyCount += 1;
      } else {
        failed += 1;
      }
      sse(response, "long_chunk", {
        index: result.index,
        status: result.status,
        timeRange: result.timeRange,
        elapsedSeconds: result.elapsedSeconds,
        summary: result.summary || result.parsed?.summary || result.content,
        content: result.content || "",
        parsed: result.parsed || null,
        events: Array.isArray(result.parsed?.events) ? result.parsed.events : [],
        eventsCount: Array.isArray(result.parsed?.events) ? result.parsed.events.length : undefined,
        error: result.error
      });
      if (result.status === "done") {
        sse(response, "long_partial", {
          index: result.index,
          timeRange: result.timeRange,
          summary: result.summary || result.parsed?.summary || result.content || ""
        });
      }
      emitProgress("chunking", `Running chunks ${completed + failed}/${chunks.length}`);
      return result;
    });

    if (clientClosed || upstreamAbort.signal.aborted) return;
    const doneChunks = results.filter((item) => item?.status === "done");
    const failedChunks = results.filter((item) => item?.status === "error");
    logReasonRequest(requestId, "chunks-complete", {
      elapsed_ms: elapsedMs(startedAt),
      completed_chunks: doneChunks.length,
      failed_chunks: failedChunks.length,
      average_chunk_seconds: latencyCount > 0 ? Number((latencyTotal / latencyCount).toFixed(2)) : null
    });
    updateStep("run_chunks", {
      status: failedChunks.length > 0 ? "error" : "done",
      progress: 100,
      detail: `${doneChunks.length}/${chunks.length} chunks succeeded`
    });
    const reducerMode = stitchReducerMode(config);
    const stitchStartedAt = Date.now();
    const stitchEtaSeconds = estimateStitchSeconds({ chunks: results, mode: reducerMode });
    logReasonRequest(requestId, "stitch-start", {
      reducer_mode: reducerMode,
      estimated_seconds: stitchEtaSeconds,
      chunk_count: results.length,
      done_chunks: doneChunks.length,
      failed_chunks: failedChunks.length
    });
    const stitchSubsteps = makeStitchSubsteps(reducerMode);
    const stitchWeights = reducerMode === "model" ? [15, 15, 15, 45, 10] : [25, 35, 20, 20];
    const stitchProgress = () => {
      const totalWeight = stitchWeights.reduce((sum, weight) => sum + weight, 0) || 1;
      return stitchSubsteps.reduce((sum, step, index) => {
        const weight = stitchWeights[index] || 0;
        return sum + weight * (Math.max(0, Math.min(100, Number(step.progress) || 0)) / 100);
      }, 0) / totalWeight * 100;
    };
    const emitStitchProgress = (message = "Stitching timeline") => {
      const elapsed = (Date.now() - stitchStartedAt) / 1000;
      const progressPercent = stitchProgress();
      const etaSeconds =
        progressPercent > 4
          ? Math.max(0, (elapsed / progressPercent) * (100 - progressPercent))
          : Math.max(0, stitchEtaSeconds - elapsed);
      updateStep("stitch_timeline", {
        status: "running",
        progress: progressPercent,
        detail:
          reducerMode === "model"
            ? "Compacting chunks and waiting for reducer NIM"
            : "Building stitched timeline locally",
        elapsedSeconds: elapsed,
        etaSeconds,
        substeps: stitchSubsteps
      });
      emitLongState({
        phase: "stitching",
        message,
        completedChunks: completed,
        failedChunks: failed,
        totalChunks: chunks.length,
        percent: Math.min(99, Math.round(95 + progressPercent * 0.04)),
        elapsedSeconds: elapsedSeconds(),
        etaSeconds
      });
    };
    const updateStitchSubstep = (key, patch) => {
      const index = stitchSubsteps.findIndex((step) => step.key === key);
      if (index === -1) return;
      stitchSubsteps[index] = { ...stitchSubsteps[index], ...patch };
      emitStitchProgress(stitchSubsteps[index].label);
    };
    updateStep("stitch_timeline", {
      status: "running",
      progress: 1,
      detail:
        reducerMode === "model"
          ? `Model reducer enabled; estimate ~${formatShortDuration(stitchEtaSeconds)}`
          : `Local stitcher; estimate ~${formatShortDuration(stitchEtaSeconds)}`,
      elapsedSeconds: 0,
      etaSeconds: stitchEtaSeconds,
      substeps: stitchSubsteps
    });
    emitLongState({
      phase: "stitching",
      message: "Stitching timeline",
      completedChunks: completed,
      failedChunks: failed,
      totalChunks: chunks.length,
      percent: 96,
      elapsedSeconds: elapsedSeconds(),
      etaSeconds: stitchEtaSeconds
    });

    const stitched = await stitchLongVideo({
      chunks: results,
      config,
      durationSeconds: extracted.durationSeconds,
      endpoints,
      failedChunks,
      model: selectedModel,
      onProgress: updateStitchSubstep,
      params,
      prompt: body.prompt || body.userPrompt || "",
      signal: upstreamAbort.signal,
      warnings
    });

    const result = {
      ...stitched,
      media: {
        mode: "long-video-image-chunks",
        duration_seconds: extracted.durationSeconds,
        frame_count: extracted.frames.length,
        requested_fps: config.requestedFps || null,
        sample_fps: effectiveSampleFps,
        extraction_fps: extracted.sampleFps,
        frame_limit: config.frameLimit,
        frame_budget: config.maxFrames,
        chunk_count: chunks.length,
        chunk_size: config.chunkSize,
        chunk_overlap: config.overlap,
        concurrency: config.concurrency,
        max_concurrency: config.maxConcurrency,
        max_images_per_chunk: LONG_VIDEO_MAX_IMAGES_PER_CHUNK,
        reducer_mode: reducerMode,
        chunk_prompt_template_enabled: Boolean(config.chunkPromptTemplate),
        reducer_prompt_template_enabled: Boolean(config.reducerPromptTemplate),
        completed_chunks: doneChunks.length,
        failed_chunks: failedChunks.length,
        warnings
      },
      long_video: {
        preset: config.preset,
        chunks: results.map((item) => ({
          index: item.index,
          status: item.status,
          timeRange: item.timeRange,
          summary: item.summary || item.parsed?.summary || item.content || "",
          content: item.content || "",
          parsed: item.parsed || null,
          events: item.parsed?.events || [],
          error: item.error,
          elapsedSeconds: item.elapsedSeconds
        }))
      }
    };
    logReasonRequest(requestId, "complete", {
      elapsed_ms: elapsedMs(startedAt),
      stitch_elapsed_ms: elapsedMs(stitchStartedAt),
      content_chars: String(result.content || result.combined_content || "").length,
      media: result.media,
      chunk_count: results.length,
      completed_chunks: doneChunks.length,
      failed_chunks: failedChunks.length
    });

    sse(response, "raw", result);
    updateStep("stitch_timeline", {
      status: "done",
      progress: 100,
      detail: `Final response ready in ${formatShortDuration((Date.now() - stitchStartedAt) / 1000)}`,
      elapsedSeconds: (Date.now() - stitchStartedAt) / 1000,
      etaSeconds: 0,
      substeps: stitchSubsteps.map((step) => ({
        ...step,
        status: step.status === "error" ? "error" : "done",
        progress: 100
      }))
    });
    emitLongState({
      phase: "complete",
      message: "Complete",
      completedChunks: completed,
      failedChunks: failed,
      totalChunks: chunks.length,
      percent: 100,
      elapsedSeconds: elapsedSeconds(),
      etaSeconds: 0
    });
    sse(response, "state", { phase: "complete" });
  } catch (error) {
    logReasonError(requestId, "error", error, { elapsed_ms: elapsedMs(startedAt) });
    if (!clientClosed && !response.writableEnded) {
      sse(response, "error", { message: error instanceof Error ? error.message : "Long video analysis failed" });
      sse(response, "state", { phase: "error" });
    }
  } finally {
    if (extracted?.tempDir) await fs.rm(extracted.tempDir, { recursive: true, force: true });
    if (!response.writableEnded) response.end();
  }
});

app.post("/api/reason/stream", async (request, response) => {
  let prepared;
  let fallbackPrepared;
  const upstreamAbort = new AbortController();
  let clientClosed = false;
  let startedStreaming = false;
  const requestId = requestLogId("stream");
  const startedAt = Date.now();

  const abortForClosedClient = () => {
    if (response.writableEnded) return;
    clientClosed = true;
    upstreamAbort.abort();
  };
  request.on("aborted", abortForClosedClient);
  response.on("close", abortForClosedClient);

  response.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no"
  });
  response.flushHeaders?.();
  sse(response, "state", { phase: "waiting_first_token" });

  try {
    if (isAlpamayoBackend) sse(response, "state", { phase: "preparing_media" });
    prepared = await prepareReasonRequest(request.body || {});
    logReasonRequest(requestId, "prepared", {
      route: "/api/reason/stream",
      ...summarizePreparedReasonRequest(prepared)
    });
    if (isAlpamayoBackend) {
      sse(response, "state", {
        phase: "alpamayo_generating",
        note:
          "Alpamayo generate_text returns a complete answer rather than token callbacks; Vite uses a non-streaming adapter request for this backend."
      });
      const fallback = await submitPreparedReasoning(prepared);
      logReasonRequest(requestId, "complete", {
        elapsed_ms: elapsedMs(startedAt),
        adapter: "alpamayo_non_streaming",
        ...summarizeReasoningResult(fallback)
      });
      emitFallback(response, fallback);
      return;
    }
    const { baseUrl, payload, redactedPayload } = buildReasoningPayload({
      model: prepared.selectedModel,
      prompt: prepared.prompt,
      systemPrompt: prepared.systemPrompt,
      mediaDataUrl: prepared.mediaDataUrl,
      mediaKind: prepared.mediaKind,
      mediaFrames: prepared.mediaFrames,
      framesPerSecond: prepared.params.frames_per_second,
      params: prepared.params
    });
    const normalizer = createReasoningStreamNormalizer();
    let answer = "";
    let reasoning = "";
    let schema = "plain_content";
    let usage = null;
    let finishReason = null;
    let meta = {};

    await readOpenAiStream({
      baseUrl,
      payload,
      signal: upstreamAbort.signal,
      onChunk(data) {
        meta = {
          id: data?.id || meta.id,
          object: data?.object || meta.object,
          created: data?.created || meta.created,
          model: data?.model || meta.model
        };
        const normalized = normalizer.accept(data);
        schema = normalized.schema || schema;
        usage = normalized.usage || usage;
        finishReason = normalized.finishReason || finishReason;
        for (const event of normalized.events) {
          if (!event.text) continue;
          startedStreaming = true;
          if (event.channel === "reasoning") {
            reasoning += event.text;
            sse(response, "state", { phase: "reasoning" });
          } else {
            answer += event.text;
            sse(response, "state", { phase: "answer" });
          }
          sse(response, "delta", event);
        }
      }
    });

    const finished = normalizer.finish();
    schema = finished.schema || schema;
    for (const event of finished.events) {
      if (!event.text) continue;
      startedStreaming = true;
      if (event.channel === "reasoning") {
        reasoning += event.text;
        sse(response, "state", { phase: "reasoning" });
      } else {
        answer += event.text;
        sse(response, "state", { phase: "answer" });
      }
      sse(response, "delta", event);
    }

    if (usage) sse(response, "usage", usage);
    sse(
      response,
      "raw",
      {
        ...streamingOpenAiResult({
          answer,
          created: meta.created,
          finishReason,
          id: meta.id,
          model: meta.model || prepared.selectedModel,
          object: meta.object,
          reasoning,
          schema,
          usage
        }),
        media: prepared.media,
        payload: redactedPayload
      }
    );
    logReasonRequest(requestId, "complete", {
      elapsed_ms: elapsedMs(startedAt),
      status: "success",
      schema,
      model: meta.model || prepared.selectedModel,
      finish_reason: finishReason,
      usage,
      content_chars: String(answer || "").length,
      reasoning_chars: String(reasoning || "").length,
      media: prepared.media
    });
    sse(response, "state", { phase: "complete" });
  } catch (error) {
    if (clientClosed || response.writableEnded) return;
    if (!startedStreaming && prepared) {
      if (shouldRetryWithFrameFallback(prepared, error)) {
        console.warn(
          `[vite-build-reason] native video_url stream rejected by NIM; retrying with ${frameFallbackLimit()} image frames`
        );
        logReasonRequest(requestId, "fallback-start", {
          reason: error instanceof Error ? error.message : "Native video_url rejected",
          frame_limit: frameFallbackLimit()
        });
        fallbackPrepared = await prepareReasonRequest(request.body || {}, {
          forceFrameFallback: true,
          fallbackError: error instanceof Error ? error.message : "Native video_url rejected"
        });
        logReasonRequest(requestId, "fallback-prepared", summarizePreparedReasonRequest(fallbackPrepared));
      }
      const fallback = await submitPreparedReasoning(fallbackPrepared || prepared);
      logReasonRequest(requestId, "complete", {
        elapsed_ms: elapsedMs(startedAt),
        used_fallback: Boolean(fallbackPrepared),
        stream_fallback: true,
        ...summarizeReasoningResult(fallback)
      });
      emitFallback(response, fallback);
    } else {
      logReasonError(requestId, "error", error, { elapsed_ms: elapsedMs(startedAt), started_streaming: startedStreaming });
      sse(response, "error", { message: error instanceof Error ? error.message : "Backend stream failed" });
    }
  } finally {
    if (prepared?.frameTempDir) await fs.rm(prepared.frameTempDir, { recursive: true, force: true });
    if (fallbackPrepared?.frameTempDir) await fs.rm(fallbackPrepared.frameTempDir, { recursive: true, force: true });
    if (!response.writableEnded) response.end();
  }
});

if (isProduction) {
  app.use(express.static(path.join(__dirname, "dist")));
  app.get("*", (_request, response) => {
    response.sendFile(path.join(__dirname, "dist", "index.html"));
  });
} else {
  const { createServer } = await import("vite");
  const vite = await createServer({
    server: { middlewareMode: true },
    appType: "spa"
  });
  app.use(vite.middlewares);
}

app.listen(port, "0.0.0.0", () => {
  console.log(`[vite-build-reason] http://localhost:${port}`);
});
