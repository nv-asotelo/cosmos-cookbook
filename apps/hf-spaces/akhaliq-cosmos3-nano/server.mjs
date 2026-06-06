import { createReadStream, existsSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST_DIR = path.resolve(__dirname, "dist");
const PORT = Number(process.env.PORT || 5185);
const HOST = process.env.HOST || "0.0.0.0";
const DIFFUSERS_BASE_URL = (process.env.DIFFUSERS_BASE_URL || "http://127.0.0.1:8010").replace(/\/$/, "");
const OUTPUT_DIR = path.resolve(process.env.DIFFUSERS_OUTPUT_DIR || "/tmp/cosmos3_diffusers_outputs");
const MAX_BODY_BYTES = 18 * 1024 * 1024;

const CONTENT_TYPES = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".mp4", "video/mp4"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".webp", "image/webp"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"]
]);

function sendJson(res, status, payload) {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": String(body.length),
    "Cache-Control": "no-store"
  });
  res.end(body);
}

function sendText(res, status, body) {
  const payload = Buffer.from(body, "utf8");
  res.writeHead(status, {
    "Content-Type": "text/plain; charset=utf-8",
    "Content-Length": String(payload.length),
    "Cache-Control": "no-store"
  });
  res.end(payload);
}

async function readJsonBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) {
      throw Object.assign(new Error("Request body is too large."), { statusCode: 413 });
    }
    chunks.push(chunk);
  }
  if (chunks.length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function safeInt(value, fallback, minimum, maximum) {
  const number = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(Math.max(number, minimum), maximum);
}

function safeFloat(value, fallback, minimum, maximum) {
  const number = Number.parseFloat(String(value ?? ""));
  if (!Number.isFinite(number)) return fallback;
  return Math.min(Math.max(number, minimum), maximum);
}

function generationPayload(mode, payload) {
  const hasImage = mode === "image_to_video" && typeof payload.mediaDataUrl === "string" && payload.mediaDataUrl;
  const height = safeInt(payload.height, mode === "text_to_image" ? 720 : 544, 16, 720);
  const width = safeInt(payload.width, Math.round(height * 16 / 9), 16, 1280);
  const framesFallback = mode === "text_to_image" ? 1 : 121;
  const numFrames = safeInt(payload.num_frames, framesFallback, 1, 189);
  const steps = safeInt(payload.num_inference_steps ?? payload.num_steps, 25, 1, 80);
  const fps = safeFloat(payload.fps, 24, 1, 60);
  const guidance = safeFloat(payload.guidance ?? payload.guidance_scale, 6, 0, 20);
  const seed = payload.seed ?? 0;
  const safeName = `hfspace-${mode}-${Date.now()}`;

  const translated = {
    name: safeName,
    model: "nvidia/Cosmos3-Nano",
    prompt: String(payload.prompt || ""),
    negative_prompt: String(payload.negative_prompt || ""),
    resolution: String(height),
    aspect_ratio: `${Math.max(1, width)},${Math.max(1, height)}`,
    num_frames: numFrames,
    fps,
    num_steps: steps,
    guidance,
    seed,
    model_mode: hasImage ? "image2video" : "text2video"
  };

  if (hasImage) translated.mediaDataUrl = payload.mediaDataUrl;
  return translated;
}

function mediaUrlFor(filePath) {
  return `/api/local-media?path=${encodeURIComponent(filePath)}`;
}

function resolveMediaPath(rawPath) {
  if (!rawPath || typeof rawPath !== "string") {
    throw Object.assign(new Error("Missing media path."), { statusCode: 400 });
  }

  const resolved = path.resolve(rawPath);
  const outputRoot = `${OUTPUT_DIR}${path.sep}`;
  if (resolved !== OUTPUT_DIR && !resolved.startsWith(outputRoot)) {
    throw Object.assign(new Error("Requested media path is outside the local output directory."), {
      statusCode: 403
    });
  }

  if (!existsSync(resolved) || !statSync(resolved).isFile()) {
    throw Object.assign(new Error("Requested media file was not found."), { statusCode: 404 });
  }

  return resolved;
}

async function handleHealth(res) {
  try {
    const [healthResponse, infoResponse] = await Promise.all([
      fetch(`${DIFFUSERS_BASE_URL}/health`),
      fetch(`${DIFFUSERS_BASE_URL}/info`)
    ]);
    const health = await healthResponse.json();
    const info = await infoResponse.json();
    sendJson(res, healthResponse.ok && infoResponse.ok ? 200 : 502, {
      status: health.status || "unknown",
      local_backend: "diffusers",
      base_url: DIFFUSERS_BASE_URL,
      info
    });
  } catch (error) {
    sendJson(res, 502, {
      status: "error",
      local_backend: "diffusers",
      base_url: DIFFUSERS_BASE_URL,
      error: error instanceof Error ? error.message : String(error)
    });
  }
}

async function handleGenerate(req, res) {
  let body;
  try {
    body = await readJsonBody(req);
  } catch (error) {
    sendJson(res, error.statusCode || 400, { error: error instanceof Error ? error.message : String(error) });
    return;
  }

  const mode = typeof body.mode === "string" ? body.mode : "text_to_video";
  const payload = body.payload && typeof body.payload === "object" ? body.payload : {};
  const translated = generationPayload(mode, payload);

  let response;
  try {
    response = await fetch(`${DIFFUSERS_BASE_URL}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(translated)
    });
  } catch (error) {
    sendJson(res, 502, {
      error: `Local Diffusers adapter is unreachable: ${error instanceof Error ? error.message : String(error)}`
    });
    return;
  }

  let data;
  try {
    data = await response.json();
  } catch {
    data = { status: "error", message: await response.text() };
  }

  if (!response.ok || data.status === "error") {
    sendJson(res, response.status || 500, {
      error: data.message || data.error || `Local Diffusers generation failed with HTTP ${response.status}.`,
      raw: data
    });
    return;
  }

  const filePath = data?.outputs?.[0]?.files?.[0];
  try {
    const resolved = resolveMediaPath(filePath);
    sendJson(res, 200, {
      url: mediaUrlFor(resolved),
      path: resolved,
      kind: "video",
      hasSound: false,
      backend: "diffusers",
      request: translated,
      raw: data
    });
  } catch (error) {
    sendJson(res, error.statusCode || 500, {
      error: error instanceof Error ? error.message : String(error),
      raw: data
    });
  }
}

function parseRange(rangeHeader, size) {
  if (!rangeHeader || !rangeHeader.startsWith("bytes=")) return null;
  const [startRaw, endRaw] = rangeHeader.slice("bytes=".length).split("-", 2);
  const start = startRaw ? Number.parseInt(startRaw, 10) : 0;
  const end = endRaw ? Number.parseInt(endRaw, 10) : size - 1;
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start || start >= size) {
    return null;
  }
  return { start, end: Math.min(end, size - 1) };
}

function handleMedia(req, url, res) {
  try {
    const filePath = resolveMediaPath(url.searchParams.get("path"));
    const ext = path.extname(filePath).toLowerCase();
    const stat = statSync(filePath);
    const contentType = CONTENT_TYPES.get(ext) || "application/octet-stream";
    const range = parseRange(req.headers.range, stat.size);

    if (range) {
      res.writeHead(206, {
        "Content-Type": contentType,
        "Content-Length": String(range.end - range.start + 1),
        "Content-Range": `bytes ${range.start}-${range.end}/${stat.size}`,
        "Cache-Control": "no-store",
        "Accept-Ranges": "bytes"
      });
      if (req.method === "HEAD") {
        res.end();
        return;
      }
      createReadStream(filePath, { start: range.start, end: range.end }).pipe(res);
      return;
    }

    res.writeHead(200, {
      "Content-Type": contentType,
      "Content-Length": String(stat.size),
      "Cache-Control": "no-store",
      "Accept-Ranges": "bytes"
    });
    if (req.method === "HEAD") {
      res.end();
      return;
    }
    createReadStream(filePath).pipe(res);
  } catch (error) {
    sendJson(res, error.statusCode || 500, { error: error instanceof Error ? error.message : String(error) });
  }
}

async function handleStatic(req, url, res) {
  const pathname = decodeURIComponent(url.pathname);
  const requested = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
  const filePath = path.resolve(DIST_DIR, requested);
  const distRoot = `${DIST_DIR}${path.sep}`;
  const safePath = filePath === DIST_DIR || filePath.startsWith(distRoot) ? filePath : path.join(DIST_DIR, "index.html");
  const candidate = existsSync(safePath) && statSync(safePath).isFile() ? safePath : path.join(DIST_DIR, "index.html");

  try {
    const body = await readFile(candidate);
    const ext = path.extname(candidate).toLowerCase();
    res.writeHead(200, {
      "Content-Type": CONTENT_TYPES.get(ext) || "application/octet-stream",
      "Content-Length": String(body.length),
      "Cache-Control": ext === ".html" ? "no-store" : "public, max-age=31536000, immutable"
    });
    if (req.method === "HEAD") {
      res.end();
      return;
    }
    res.end(body);
  } catch (error) {
    sendText(res, 500, `Static server failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

  if (req.method === "OPTIONS" && url.pathname.startsWith("/api/")) {
    res.writeHead(204, {
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    });
    res.end();
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/local-health") {
    await handleHealth(res);
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/local-generate") {
    await handleGenerate(req, res);
    return;
  }

  if ((req.method === "GET" || req.method === "HEAD") && url.pathname === "/api/local-media") {
    handleMedia(req, url, res);
    return;
  }

  if (req.method === "GET" || req.method === "HEAD") {
    await handleStatic(req, url, res);
    return;
  }

  sendJson(res, 405, { error: "Method not allowed." });
}).listen(PORT, HOST, () => {
  console.log(`Cosmos3 Nano local Vite server listening on http://${HOST}:${PORT}`);
  console.log(`Proxying generation to ${DIFFUSERS_BASE_URL}`);
  console.log(`Streaming outputs from ${OUTPUT_DIR}`);
});
