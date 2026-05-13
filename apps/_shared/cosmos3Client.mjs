// Shared server-side client for the Cosmos3 Ray Serve API.
//
// Env var priority for the Ray Serve base URL:
//   COSMOS3_BASE_URL > RAY_SERVE_BASE_URL > VLLM_BASE_URL > http://localhost:8000 (default).
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
// Node 20+ built-ins only (`crypto`, `fs/promises`, `path`, `fetch`, `undici`).
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
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { Agent, fetch as undiciFetch } from "undici";

// 30-minute dispatcher for the long-running Ray Serve POST /generate call.
// Headers + body timeouts both bumped to 30 min so HD generations don't get
// cut off mid-flight.
const RAY_SERVE_TIMEOUT_MS = Number(process.env.COSMOS3_REQUEST_TIMEOUT_MS) || 30 * 60 * 1000;
const RAY_SERVE_AGENT = new Agent({
  headersTimeout: RAY_SERVE_TIMEOUT_MS,
  bodyTimeout: RAY_SERVE_TIMEOUT_MS,
  keepAliveTimeout: 60_000,
  connect: { timeout: 30_000 }
});

const UPLOAD_DIR = "/tmp/uploads";

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

function mimeForFile(filepath) {
  const ext = path.extname(filepath).replace(/^\./, "").toLowerCase();
  return EXT_TO_MIME[ext] || "application/octet-stream";
}

async function encodeOutputFile(filepath) {
  try {
    const buf = await readFile(filepath);
    return {
      path: filepath,
      b64: buf.toString("base64"),
      mime: mimeForFile(filepath)
    };
  } catch (error) {
    return {
      path: filepath,
      b64: null,
      mime: mimeForFile(filepath),
      error: error instanceof Error ? error.message : "Failed to read output file"
    };
  }
}

export async function submitGeneration({ prompt, mediaDataUrl, mediaKind, params } = {}) {
  const baseUrl = resolveBaseUrl();
  const p = params || {};

  let visionPath = null;
  if (mediaDataUrl) {
    visionPath = await persistUpload(mediaDataUrl);
  }

  const body = {
    name: `req-${Date.now()}`,
    model: process.env.MODEL_NAME || "",
    prompt: prompt || "",
    negative_prompt: p.negative_prompt || "",
    vision_path: visionPath || null,
    num_frames: p.num_frames ?? 121,
    // Ray Serve OmniSampleOverrides requires resolution as a literal string ('256' | '480' | '720' | '1080').
    resolution: String(p.resolution ?? 480),
    aspect_ratio: p.aspect_ratio ?? "16,9",
    num_steps: p.num_steps ?? 35,
    guidance: p.guidance ?? 6.0,
    seed: p.seed ?? null,
    model_mode: p.model_mode,
    action_path: p.action_path
  };

  let response;
  try {
    response = await undiciFetch(`${baseUrl}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      dispatcher: RAY_SERVE_AGENT
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
  const files = await Promise.all(filePaths.map(encodeOutputFile));

  return {
    status: data?.status || "success",
    message: data?.message || "",
    stack_trace: data?.stack_trace || null,
    files,
    payload: body,
    raw: data
  };
}

export default submitGeneration;
