// Shared server-side client for Cosmos generation backends.
//
// Env var priority for the Ray Serve base URL:
//   COSMOS3_BASE_URL > RAY_SERVE_BASE_URL > VLLM_BASE_URL > http://localhost:8000 (default).
//
// NIM-local mode:
//   Set COSMOS3_BACKEND=nim_local or INFERENCE_BACKEND=nim_local and point
//   NIM_BASE_URL/VLLM_BASE_URL at http://localhost:8000/v1. Requests are sent
//   to POST /v1/infer using the staging Cosmos3 Generation NIM payload shape.
//
// The Ray Serve endpoint is a single POST /generate that accepts the Pydantic
// `OmniSampleOverrides` shape. `vision_path` may be a local filesystem path on
// the Ray Serve host (preferred for uploads — we decode the dataURL and write
// it under /tmp/uploads/) or an HTTPS URL the server can fetch.
//
// The response contains `outputs[].files[]` — filesystem paths on the Ray Serve
// host (under `outputs/ray_serve/`). We base64-encode each file so the browser
// can render it inline.
//
// Node 20+ built-ins plus the app-local `undici` package.
//
// LONG-RUNNING-REQUEST NOTE (2026-05-13 incident — generate_20260513_212614_ad1d):
// Cosmos3 Ray Serve `/generate` is a synchronous long call (HD 720p × 121 frames
// × 35 steps took ~5m18s on the RTX PRO 6000 Blackwell). Node's default `fetch`
// uses undici Agent defaults `bodyTimeout: 300_000` and `headersTimeout: 300_000`
// (5 min each). Without a custom dispatcher the fetch aborts at 5 min and the
// caller gets a `UND_ERR_BODY_TIMEOUT` even though Ray Serve eventually wrote
// the output file to disk. Frontend reported 502 / "transport threw before
// returning". We now use a dedicated undici Agent with 30-minute timeouts so
// large generations complete cleanly.

import { createHash } from "node:crypto";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";

// 30-minute dispatcher for the long-running Ray Serve POST /generate call.
// Headers + body timeouts both bumped to 30 min so HD generations don't get
// cut off mid-flight.
const RAY_SERVE_TIMEOUT_MS = Number(process.env.COSMOS3_REQUEST_TIMEOUT_MS) || 30 * 60 * 1000;
const UNDICI_REQUIRE_ROOTS = [
  process.env.COSMOS3_CLIENT_REQUIRE_ROOT,
  process.env.PREDICT_VITE_APP_DIR,
  process.cwd(),
  "/tmp/nvidia-build-predict-vite"
].filter(Boolean);

function loadUndici() {
  for (const root of UNDICI_REQUIRE_ROOTS) {
    try {
      const requireFromRoot = createRequire(path.join(root, "package.json"));
      const undici = requireFromRoot("undici");
      if (undici?.Agent && undici?.fetch) {
        return undici;
      }
    } catch {
      // Try the next candidate root.
    }
  }
  return { Agent: null, fetch: globalThis.fetch };
}

const { Agent, fetch: undiciFetch } = loadUndici();
const RAY_SERVE_AGENT = Agent
  ? new Agent({
      headersTimeout: RAY_SERVE_TIMEOUT_MS,
      bodyTimeout: RAY_SERVE_TIMEOUT_MS,
      keepAliveTimeout: 60_000,
      connect: { timeout: 30_000 }
    })
  : null;

const UPLOAD_DIR = "/tmp/uploads";
const OUTPUT_DIR_CACHE = new Map();

const MIME_TO_EXT = {
  "video/mp4": "mp4",
  "video/quicktime": "mov",
  "video/webm": "webm",
  "image/jpeg": "jpg",
  "image/jpg": "jpg",
  "image/png": "png",
  "image/webp": "webp"
};

const EXT_TO_MIME = {
  mp4: "video/mp4",
  mov: "video/quicktime",
  webm: "video/webm",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  webp: "image/webp"
};

function resolveBaseUrl() {
  return (
    process.env.COSMOS3_BASE_URL ||
    process.env.RAY_SERVE_BASE_URL ||
    process.env.VLLM_BASE_URL ||
    "http://localhost:8000"
  ).replace(/\/$/, "");
}

function resolveBackend() {
  const explicit = String(
    process.env.COSMOS3_BACKEND ||
      process.env.PREDICT_BACKEND ||
      process.env.INFERENCE_BACKEND ||
      ""
  ).toLowerCase();
  if (explicit.includes("nim")) return "nim";
  if (process.env.NIM_INFER_URL || process.env.COSMOS3_INFER_URL) return "nim";
  return "ray";
}

function resolveNimBaseUrl() {
  const raw =
    process.env.NIM_BASE_URL ||
    process.env.VLLM_BASE_URL ||
    process.env.COSMOS3_NIM_BASE_URL ||
    process.env.PREDICT_BASE_URL ||
    "http://localhost:8000/v1";
  const trimmed = raw.replace(/\/$/, "");
  return trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`;
}

function resolveNimInferUrl() {
  return (
    process.env.NIM_INFER_URL ||
    process.env.COSMOS3_INFER_URL ||
    `${resolveNimBaseUrl()}/infer`
  ).replace(/\/$/, "");
}

function parseDataUrl(dataUrl) {
  // data:<mime>;base64,<payload>
  const match = /^data:([^;,]+)(;base64)?,(.*)$/s.exec(dataUrl || "");
  if (!match) return null;
  const mime = match[1] || "application/octet-stream";
  const isBase64 = Boolean(match[2]);
  const payload = match[3] || "";
  const buf = isBase64
    ? Buffer.from(payload, "base64")
    : Buffer.from(decodeURIComponent(payload), "utf-8");
  return { mime, buf };
}

async function persistUpload(dataUrl) {
  const parsed = parseDataUrl(dataUrl);
  if (!parsed) return null;
  const { mime, buf } = parsed;
  const ext = MIME_TO_EXT[mime] || mime.split("/")[1] || "bin";
  const sha1 = createHash("sha1").update(buf).digest("hex");
  await mkdir(UPLOAD_DIR, { recursive: true });
  const filepath = path.join(UPLOAD_DIR, `${sha1}.${ext}`);
  await writeFile(filepath, buf);
  return filepath;
}

function publicAssetRoots() {
  const roots = [
    process.env.PREDICT_VITE_PUBLIC_DIR,
    process.env.VITE_PUBLIC_DIR,
    process.env.PREDICT_VITE_APP_DIR ? path.join(process.env.PREDICT_VITE_APP_DIR, "public") : null,
    process.cwd() ? path.join(process.cwd(), "public") : null,
    path.resolve("apps/nvidia-build-predict-vite/public"),
    "/home/horde/cookbook-apps/nvidia-build-predict-vite/public"
  ].filter(Boolean);
  return Array.from(new Set(roots.map((root) => path.resolve(root))));
}

function isPathInside(candidate, root) {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function addHostCandidate(hosts, value) {
  for (const part of String(value || "").split(",")) {
    const raw = part.trim();
    if (!raw) continue;
    try {
      hosts.add(new URL(raw).hostname);
      continue;
    } catch {
      // Plain host or host:port.
    }
    hosts.add(raw.replace(/:\d+$/, ""));
  }
}

function localPublicAssetHosts() {
  const hosts = new Set(["localhost", "127.0.0.1"]);
  addHostCandidate(hosts, process.env.BYO_VIDEO_LOCAL_HOST);
  addHostCandidate(hosts, process.env.PREDICT_VITE_HOST);
  addHostCandidate(hosts, process.env.VITE_HOST);
  addHostCandidate(hosts, process.env.HOST);
  addHostCandidate(hosts, process.env.HOSTNAME);
  return hosts;
}

function localPublicAssetPorts() {
  return new Set(
    [
      "",
      process.env.PORT,
      process.env.PREDICT_VITE_PORT,
      process.env.VITE_PORT,
      "5174",
      "5175",
    ]
      .filter((port) => port !== undefined && port !== null)
      .map((port) => String(port).trim()),
  );
}

function localPublicAssetPathname(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;
  if (raw.startsWith("/")) return raw;

  try {
    const url = new URL(raw);
    const isHttp = url.protocol === "http:" || url.protocol === "https:";
    const isViteHost = localPublicAssetHosts().has(url.hostname);
    const isVitePort = localPublicAssetPorts().has(url.port || "");
    if (isHttp && isViteHost && isVitePort) return url.pathname;
  } catch {
    // Not a URL; keep it as-is for Ray to handle if it is already a local path.
  }
  return null;
}

async function resolveLocalPublicAsset(value) {
  const pathname = localPublicAssetPathname(value);
  if (!pathname) return null;

  let decodedPathname;
  try {
    decodedPathname = decodeURIComponent(pathname);
  } catch {
    return null;
  }

  const relativePath = decodedPathname.replace(/^\/+/, "");
  if (!relativePath || relativePath.split(/[\\/]+/).includes("..")) return null;

  for (const root of publicAssetRoots()) {
    const candidate = path.resolve(root, relativePath);
    if (!isPathInside(candidate, root)) continue;
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next public root.
    }
  }
  return null;
}

async function mediaFromDataUrl(dataUrl) {
  const parsed = parseDataUrl(dataUrl);
  if (!parsed) return null;
  return {
    mime: parsed.mime,
    b64: parsed.buf.toString("base64")
  };
}

async function mediaFromUrl(url) {
  if (!/^https?:\/\//i.test(String(url || ""))) return null;
  const response = await undiciFetch(url, {
    ...(RAY_SERVE_AGENT ? { dispatcher: RAY_SERVE_AGENT } : {})
  });
  if (!response.ok) throw new Error(`Failed to fetch conditioning media (${response.status})`);
  const arrayBuffer = await response.arrayBuffer();
  const mime = response.headers.get("content-type")?.split(";")[0] || mimeForFile(url);
  return {
    mime,
    b64: Buffer.from(arrayBuffer).toString("base64")
  };
}

function mimeForFile(filepath) {
  const ext = path.extname(filepath).replace(/^\./, "").toLowerCase();
  return EXT_TO_MIME[ext] || "application/octet-stream";
}

// ---------------------------------------------------------------------------
// Ray Serve server-side budget guard
//
// `cosmos3/ray/serve.py:178` hardcodes `asyncio.wait_for(..., timeout=300.0)`
// (5 min) around the generation. The cosmos3 install on horde is early-access
// read-only — we cannot patch the server. Instead we reject any client request
// whose predicted wall time exceeds an 80%-of-budget safety bar (240 s).
//
// Empirical three-point anchor on horde@10.57.233.111 (RTX PRO 6000 Blackwell):
//   256×256 ×  17 ×  20 →   3.5 s   (= 22.3 M ops)
//   720p   × 121 ×  35 → 376    s   (= 3.91 B ops) — avg of three Predict OK calls
//   720p   × 189 ×  35 → ~590  s   (= 6.10 B ops) — Gradio run hit the 600s cap
// Three-point fit: wall ≈ 5 s baseline + ops / 10 M ops/s. The earlier 2 s +
// ops/13 M model under-estimated by ~25% at HD frame counts; the new fit is
// conservative on small jobs (over-estimates 256² by ~4 s, harmless) and
// tight on HD jobs where the budget guard actually matters.
// Verified against the suggestion grid:
//   480p × 121 × 35 (1.74 B) → ~134 s  (allowed)
//   720p × 121 × 20 (2.23 B) → ~173 s  (allowed)
//   720p × 60  × 35 (1.93 B) → ~150 s  (allowed)
//   1080p × 121 × 35 (8.78 B) → ~677 s (blocked — far over the 300 s cap)
//
// Override:  RAY_SERVE_MAX_WALL_SECONDS  (defaults to 240, ~80 % of the 300 s cap)
//            RAY_SERVE_BUDGET_DISABLED=1 (skip the gate entirely; for vLLM-only backends)

const RAY_SERVE_PIXELS_BY_RESOLUTION = {
  "256": 256 * 256,
  "480": 854 * 480,
  "720": 1280 * 720,
  "1080": 1920 * 1080
};

const RAY_SERVE_BASELINE_SECONDS = 5;
const RAY_SERVE_OPS_PER_SECOND = 10_000_000;
const RAY_SERVE_MAX_WALL_SECONDS_DEFAULT = 240;
const RAY_SERVE_SERVER_TIMEOUT_SECONDS = 300; // cosmos3.ray.serve:178 hardcoded

export function estimateRayServeWallSeconds({ resolution, num_frames, num_steps } = {}) {
  const px = RAY_SERVE_PIXELS_BY_RESOLUTION[String(resolution ?? 480)] || RAY_SERVE_PIXELS_BY_RESOLUTION["480"];
  const f = Math.max(1, Number(num_frames ?? 121));
  const s = Math.max(1, Number(num_steps ?? 50));
  return Math.ceil(RAY_SERVE_BASELINE_SECONDS + (px * f * s) / RAY_SERVE_OPS_PER_SECOND);
}

function suggestSafeParams(current) {
  const candidates = [
    { resolution: "480", num_frames: 121, num_steps: 35 },
    { resolution: "480", num_frames: 60, num_steps: 35 },
    { resolution: "720", num_frames: 60, num_steps: 20 },
    { resolution: "720", num_frames: 33, num_steps: 35 },
    { resolution: "256", num_frames: 121, num_steps: 50 }
  ];
  const max = Number(process.env.RAY_SERVE_MAX_WALL_SECONDS) || RAY_SERVE_MAX_WALL_SECONDS_DEFAULT;
  return candidates
    .map((c) => ({ ...c, est_seconds: estimateRayServeWallSeconds(c) }))
    .filter((c) => c.est_seconds <= max)
    .slice(0, 3);
}

function checkRayServeBudget(params) {
  if (process.env.RAY_SERVE_BUDGET_DISABLED === "1") return null;
  const est = estimateRayServeWallSeconds(params);
  const cap = Number(process.env.RAY_SERVE_MAX_WALL_SECONDS) || RAY_SERVE_MAX_WALL_SECONDS_DEFAULT;
  if (est <= cap) return null;
  return {
    estimate_seconds: est,
    safe_cap_seconds: cap,
    server_timeout_seconds: RAY_SERVE_SERVER_TIMEOUT_SECONDS,
    requested: {
      resolution: String(params.resolution ?? 480),
      num_frames: Number(params.num_frames ?? 121),
      num_steps: Number(params.num_steps ?? 50)
    },
    suggestions: suggestSafeParams(params)
  };
}

function parseAspectRatio(aspectRatio) {
  const raw = String(aspectRatio || "16,9").replace(":", ",");
  const [w, h] = raw.split(",").map((part) => Number(part.trim()));
  if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) return [w, h];
  return [16, 9];
}

function finiteOrFallback(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function payloadParamsForNim(params = {}) {
  return {
    resolution: String(params.resolution ?? "256"),
    num_output_frames: finiteOrFallback(params.num_frames ?? params.frames_count, 121),
    fps: finiteOrFallback(params.fps ?? params.frames_per_sec, 24)
  };
}

async function resolveNimMedia({ mediaDataUrl, visionPath, mediaKind }) {
  const media = mediaDataUrl ? await mediaFromDataUrl(mediaDataUrl) : await mediaFromUrl(visionPath);
  if (!media) return null;
  const field =
    mediaKind === "video" || media.mime.startsWith("video/")
      ? "video"
      : "image";
  return { ...media, field };
}

function firstString(...values) {
  return values.find((value) => typeof value === "string" && value.length > 0);
}

function normalizeNimFiles(data) {
  const files = [];
  const b64Video = firstString(
    data?.b64_video,
    data?.video_b64,
    data?.output_video,
    data?.outputs?.b64_video,
    data?.outputs?.video_b64,
    data?.output?.b64_video
  );
  const b64Image = firstString(
    data?.b64_image,
    data?.image_b64,
    data?.output_image,
    data?.outputs?.b64_image,
    data?.outputs?.image_b64,
    data?.output?.b64_image
  );
  const assetUrl = firstString(data?.asset_url, data?.url, data?.video_url, data?.output?.asset_url);

  if (b64Video) files.push({ path: "nim-output.mp4", b64: b64Video, mime: "video/mp4" });
  if (b64Image) files.push({ path: "nim-output.jpg", b64: b64Image, mime: "image/jpeg" });
  if (assetUrl) files.push({ path: assetUrl, b64: null, mime: mimeForFile(assetUrl), url: assetUrl });

  if (Array.isArray(data?.files)) {
    for (const file of data.files) {
      const b64 = firstString(file?.b64, file?.base64, file?.data);
      const url = firstString(file?.url, file?.asset_url);
      if (b64 || url) {
        files.push({
          path: file?.path || file?.name || url || "nim-output",
          b64: b64 || null,
          mime: file?.mime || mimeForFile(file?.path || file?.name || url || "mp4"),
          url
        });
      }
    }
  }
  return files;
}

async function submitNimGeneration({ prompt, mediaDataUrl, mediaKind, params, model, visionPath } = {}) {
  const p = params || {};
  let media = null;
  try {
    media = await resolveNimMedia({ mediaDataUrl, visionPath, mediaKind });
  } catch (error) {
    return {
      status: "error",
      message: error instanceof Error ? error.message : "Failed to prepare NIM conditioning media",
      files: [],
      payload: { vision_path: visionPath || null }
    };
  }

  const payload = {
    prompt: prompt || "",
    guidance_scale: Number(p.guidance ?? 6),
    steps: Number(p.num_steps ?? 50),
    ...payloadParamsForNim(p)
  };
  const seed = Number(p.seed);
  if (Number.isFinite(seed) && seed >= 0) payload.seed = Math.floor(seed);
  if (p.negative_prompt) payload.negative_prompt = p.negative_prompt;
  if (process.env.NIM_INCLUDE_MODEL === "1") {
    payload.model = process.env.NIM_SERVED_MODEL_NAME || model || process.env.MODEL_NAME;
  }
  if (media?.field === "video") {
    return {
      status: "error",
      message: "The staging Cosmos3 Generation NIM currently supports prompt-only T2V and image-conditioned I2V through /v1/infer.",
      files: [],
      payload: { ...payload, vision_path: visionPath || null }
    };
  }
  if (media) payload[media.field] = media.b64;

  const inferUrl = resolveNimInferUrl();
  let response;
  try {
    response = await undiciFetch(inferUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      ...(RAY_SERVE_AGENT ? { dispatcher: RAY_SERVE_AGENT } : {})
    });
  } catch (error) {
    return {
      status: "error",
      message: error instanceof Error ? error.message : "NIM /v1/infer request failed",
      files: [],
      payload: { ...payload, image: payload.image ? "<base64 omitted>" : undefined, video: payload.video ? "<base64 omitted>" : undefined }
    };
  }

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  const redactedPayload = {
    ...payload,
    image: payload.image ? `<${payload.image.length} base64 chars>` : undefined,
    video: payload.video ? `<${payload.video.length} base64 chars>` : undefined
  };

  if (!response.ok) {
    return {
      status: "error",
      message: data?.message || data?.error || `NIM /v1/infer returned HTTP ${response.status}`,
      files: [],
      payload: redactedPayload,
      raw: data
    };
  }

  const files = normalizeNimFiles(data);
  return {
    status: data?.status || (files.length ? "success" : "error"),
    message: data?.message || (files.length ? "" : "NIM returned no video, image, or asset URL"),
    files,
    content: data?.content || data?.output || null,
    action: data?.action || data?.output?.action || null,
    payload: redactedPayload,
    raw: data,
    backend: "nim_local"
  };
}

async function resolveRayOutputDir(baseUrl) {
  const envOutputDir = process.env.COSMOS3_OUTPUT_DIR || process.env.RAY_SERVE_OUTPUT_DIR || process.env.COSMOS3_RAY_OUTPUT_DIR;
  if (envOutputDir) return envOutputDir;
  if (OUTPUT_DIR_CACHE.has(baseUrl)) return OUTPUT_DIR_CACHE.get(baseUrl);
  try {
    const response = await undiciFetch(`${baseUrl}/info`, {
      ...(RAY_SERVE_AGENT ? { dispatcher: RAY_SERVE_AGENT } : {})
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const outputDir = typeof data?.output_dir === "string" ? data.output_dir : null;
    OUTPUT_DIR_CACHE.set(baseUrl, outputDir);
    return outputDir;
  } catch {
    OUTPUT_DIR_CACHE.set(baseUrl, null);
    return null;
  }
}

function outputRoots(extraOutputDir) {
  const roots = [
    extraOutputDir,
    process.env.COSMOS3_OUTPUT_DIR,
    process.env.RAY_SERVE_OUTPUT_DIR,
    process.env.COSMOS3_RAY_OUTPUT_DIR,
    process.env.COSMOS3_DIR ? path.join(process.env.COSMOS3_DIR, "outputs/ray_serve") : null,
    path.resolve("outputs/ray_serve")
  ].filter(Boolean);
  return Array.from(new Set(roots));
}

function outputUrlFor(baseUrl, filepath) {
  if (!baseUrl || path.isAbsolute(filepath) || /^https?:\/\//i.test(filepath)) return undefined;
  const relative = String(filepath)
    .replace(/^\/+/, "")
    .split(/[\\/]/)
    .map((part) => encodeURIComponent(part))
    .join("/");
  return `${baseUrl}/outputs/${relative}`;
}

async function encodeOutputFile(filepath, { baseUrl, outputDir } = {}) {
  const originalPath = String(filepath);
  const candidates = [originalPath];
  if (!path.isAbsolute(originalPath) && !/^https?:\/\//i.test(originalPath)) {
    for (const root of outputRoots(outputDir)) {
      candidates.push(path.join(root, originalPath));
    }
  }
  let lastError = null;
  try {
    for (const candidate of candidates) {
      try {
        const buf = await readFile(candidate);
        return {
          path: originalPath,
          b64: buf.toString("base64"),
          mime: mimeForFile(originalPath)
        };
      } catch (error) {
        lastError = error;
      }
    }
  } catch (error) {
    lastError = error;
  }
  return {
    path: originalPath,
    b64: null,
    mime: mimeForFile(originalPath),
    url: outputUrlFor(baseUrl, originalPath),
    error: lastError instanceof Error ? lastError.message : "Failed to read output file"
  };
}

export async function submitGeneration({ prompt, mediaDataUrl, mediaKind, params, model, visionPath } = {}) {
  if (resolveBackend() === "nim") {
    return submitNimGeneration({ prompt, mediaDataUrl, mediaKind, params, model, visionPath });
  }

  const baseUrl = resolveBaseUrl();
  const p = params || {};
  void mediaKind;
  void model;

  // Ray Serve has a hardcoded 300 s asyncio timeout — reject up-front so the
  // user gets actionable guidance instead of a 5-minute wait + 500.
  const budget = checkRayServeBudget(p);
  if (budget) {
    const suggestionLines = budget.suggestions.length
      ? budget.suggestions
          .map(
            (s) =>
              `  • resolution=${s.resolution}, num_frames=${s.num_frames}, num_steps=${s.num_steps} (~${s.est_seconds}s)`
          )
          .join("\n")
      : "  • Use the 256/480 resolution tiers with fewer frames or steps.";
    return {
      status: "error",
      message:
        `Request would exceed the Cosmos3 Ray Serve 300 s server-side timeout ` +
        `(estimated ~${budget.estimate_seconds}s; safe budget ${budget.safe_cap_seconds}s).\n` +
        `Reduce one of resolution / num_frames / num_steps. Suggested combinations:\n` +
        suggestionLines,
      files: [],
      budget,
      payload: {
        resolution: String(p.resolution ?? 480),
        num_frames: p.num_frames,
        num_steps: p.num_steps
      }
    };
  }

  let uploadedVisionPath = null;
  if (mediaDataUrl) {
    uploadedVisionPath = await persistUpload(mediaDataUrl);
  }
  const requestedVisionPath = p.vision_path || visionPath || null;
  const localVisionPath = uploadedVisionPath ? null : await resolveLocalPublicAsset(requestedVisionPath);
  const resolvedVisionPath = uploadedVisionPath || localVisionPath || requestedVisionPath;
  const rayModel =
    process.env.COSMOS3_RAY_MODEL_NAME !== undefined
      ? process.env.COSMOS3_RAY_MODEL_NAME
      : process.env.COSMOS3_MODEL_NAME !== undefined
        ? process.env.COSMOS3_MODEL_NAME
        : p.model ?? "";

  const body = {
    name: `req-${Date.now()}`,
    model: rayModel,
    prompt: prompt || "",
    negative_prompt: p.negative_prompt || "",
    vision_path: resolvedVisionPath,
    num_frames: p.num_frames ?? 121,
    // Ray Serve OmniSampleOverrides requires resolution as a literal string ('256' | '480' | '720' | '1080').
    resolution: String(p.resolution ?? 480),
    aspect_ratio: p.aspect_ratio ?? "16,9",
    fps: p.fps,
    num_steps: p.num_steps ?? 50,
    guidance: p.guidance ?? 6.0,
    guidance_interval: p.guidance_interval,
    seed: p.seed ?? null,
    model_mode: p.model_mode,
    action_path: p.action_path,
    action_mode: p.action_mode,
    domain_name: p.domain_name,
    image_size: p.image_size,
    action_chunk_size: p.action_chunk_size,
    raw_action_dim: p.raw_action_dim,
    shift: p.shift,
    sigma_max: p.sigma_max,
    normalize_cfg: p.normalize_cfg,
    condition_frame_indexes_vision: p.condition_frame_indexes_vision,
    video_save_quality: p.video_save_quality,
    image_save_quality: p.image_save_quality,
    negative_metadata_mode: p.negative_metadata_mode,
    negative_prompt_keep_metadata: p.negative_prompt_keep_metadata,
    num_outputs: p.num_outputs
  };

  let response;
  try {
    response = await undiciFetch(`${baseUrl}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      ...(RAY_SERVE_AGENT ? { dispatcher: RAY_SERVE_AGENT } : {})
    });
  } catch (error) {
    return {
      status: "error",
      message: error instanceof Error ? error.message : "Ray Serve request failed",
      files: [],
      payload: body
    };
  }

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    return {
      status: "error",
      message:
        data?.message ||
        data?.stack_trace ||
        `Ray Serve returned HTTP ${response.status}`,
      files: [],
      payload: body,
      raw: data
    };
  }

  const outputs = Array.isArray(data?.outputs) ? data.outputs : [];
  const filePaths = outputs.flatMap((entry) =>
    Array.isArray(entry?.files) ? entry.files : []
  );
  const outputDir = await resolveRayOutputDir(baseUrl);
  const files = await Promise.all(filePaths.map((filepath) => encodeOutputFile(filepath, { baseUrl, outputDir })));
  const content = outputs.find((entry) => entry?.content)?.content || null;

  return {
    status: data?.status || "success",
    message: data?.message || "",
    stack_trace: data?.stack_trace || null,
    files,
    content,
    action: content?.action ?? null,
    payload: body,
    raw: data
  };
}

export default submitGeneration;
