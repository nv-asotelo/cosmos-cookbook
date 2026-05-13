import express from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { submitGeneration } from "../_shared/cosmos3Client.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5174);

// Backwards-compat env: the shared client checks
// COSMOS3_BASE_URL → RAY_SERVE_BASE_URL → VLLM_BASE_URL → localhost:8000.
// If a legacy PREDICT_BASE_URL or NIM_BASE_URL is the only thing set we
// forward it as RAY_SERVE_BASE_URL (stripping the /v1 suffix the old NIM
// shape carries) so the new client picks it up cleanly.
(function forwardLegacyBaseUrl() {
  if (process.env.COSMOS3_BASE_URL || process.env.RAY_SERVE_BASE_URL) return;
  const legacy =
    process.env.PREDICT_BASE_URL ||
    process.env.NIM_BASE_URL ||
    process.env.VLLM_BASE_URL;
  if (legacy) process.env.RAY_SERVE_BASE_URL = legacy.replace(/\/v1\/?$/, "");
})();

const advertisedBaseUrl =
  process.env.COSMOS3_BASE_URL ||
  process.env.RAY_SERVE_BASE_URL ||
  process.env.PREDICT_BASE_URL ||
  process.env.NIM_BASE_URL ||
  process.env.VLLM_BASE_URL ||
  "http://localhost:8000";

const defaultModel = process.env.MODEL_NAME || "Cosmos3-Nano";

// Default Cosmos1-shape video params the Predict UI has historically driven.
const VIDEO_PARAMS = {
  height: 704,
  width: 1280,
  frames_count: 121,
  frames_per_sec: 24
};

const app = express();
app.use(express.json({ limit: "128mb" }));

app.get("/api/models", async (_request, response) => {
  // Ray Serve doesn't expose /v1/models. Surface the configured default so
  // the UI has at least one entry; users with a NIM still listening on
  // ${VLLM_BASE_URL}/v1/models will fall through to the warning branch.
  const probeUrl = `${(process.env.VLLM_BASE_URL || process.env.NIM_BASE_URL || advertisedBaseUrl).replace(/\/$/, "")}/v1/models`;
  try {
    const upstream = await fetch(probeUrl, {
      headers: { Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}` }
    });
    if (!upstream.ok) throw new Error(`Model probe failed with HTTP ${upstream.status}`);
    const data = await upstream.json();
    const models = Array.isArray(data?.data) ? data.data.map((model) => model.id).filter(Boolean) : [];
    response.json({ baseUrl: advertisedBaseUrl, models: models.length > 0 ? models : [defaultModel] });
  } catch (error) {
    response.json({
      baseUrl: advertisedBaseUrl,
      models: [defaultModel],
      warning: error instanceof Error ? error.message : "Unable to reach model endpoint"
    });
  }
});

function redactPayload(payload) {
  return payload || undefined;
}

function sendError(response, status, error, diagnostic, payload) {
  response.status(status).json({
    error,
    diagnostic,
    payload: payload ? redactPayload(payload) : undefined
  });
}

app.post("/api/predict", async (request, response) => {
  const body = request.body;

  if (!body || !body.mediaDataUrl) {
    sendError(response, 400, "Upload a video or image before generating.", {
      layer: "frontend",
      issue: "No conditioning media was included in the request.",
      likelyCause: "The Generate button was pressed before an upload or example clip was loaded.",
      suggestions: ["Upload an MP4/JPEG/PNG input or click View Examples before generating."]
    });
    return;
  }

  const model = body.model || defaultModel;
  const seed = Number(body.seed);
  const mediaKind =
    body.mediaKind === "image" || body.worldMode === "Image-to-World" ? "image" : "video";

  const params = {
    guidance: Number(body.guidanceScale),
    num_steps: Number(body.steps),
    num_frames: VIDEO_PARAMS.frames_count,
    resolution: 720,
    aspect_ratio: "16,9",
    fps: VIDEO_PARAMS.frames_per_sec
  };
  if (Number.isFinite(seed) && seed >= 0) params.seed = seed;
  if (process.env.PREDICT_NEGATIVE_PROMPT) params.negative_prompt = process.env.PREDICT_NEGATIVE_PROMPT;
  // `input_image_index` from the legacy UI is not part of OmniSampleOverrides;
  // the shared client whitelists params and will drop it. Forwarding it
  // anyway preserves the audit-flagged behavior of "don't error if unused".
  if (body.inputImageIndex !== undefined) params.input_image_index = Number(body.inputImageIndex);

  let result;
  try {
    result = await submitGeneration({
      prompt: body.prompt,
      mediaDataUrl: body.mediaDataUrl,
      mediaKind,
      params,
      model
    });
  } catch (error) {
    sendError(
      response,
      502,
      "Cosmos3 Ray Serve transport threw before returning.",
      {
        layer: "backend",
        issue: error instanceof Error ? error.message : "Unknown transport error.",
        likelyCause: "The shared Cosmos3 client could not complete the round-trip to /generate.",
        suggestions: [
          "Confirm the Ray Serve replica is up: `curl http://localhost:8000/generate` on horde.",
          "Check that COSMOS3_BASE_URL / RAY_SERVE_BASE_URL resolves from this process."
        ]
      },
      { model, prompt: body.prompt }
    );
    return;
  }

  if (!result || result.status === "error" || !Array.isArray(result.files)) {
    sendError(
      response,
      502,
      result?.message || "Cosmos3 generation failed.",
      {
        layer: "backend",
        issue: result?.message || "Ray Serve returned a non-success status.",
        stack_trace: result?.stack_trace || null,
        likelyCause: "The Cosmos3 Ray Serve handler raised before producing outputs.",
        suggestions: [
          "Check the Ray Serve logs on horde for the matching `name` field.",
          "Confirm the requested model is currently mounted by Ray Serve."
        ],
        raw: result?.raw
      },
      result?.payload
    );
    return;
  }

  const firstVideo = result.files.find((file) => file.b64 && file.mime?.startsWith("video/"));
  const firstImage = result.files.find((file) => file.b64 && file.mime?.startsWith("image/"));
  const primary = firstVideo || firstImage || null;

  response.json({
    videoDataUrl:
      primary && primary.b64 && primary.mime?.startsWith("video/")
        ? `data:${primary.mime};base64,${primary.b64}`
        : undefined,
    imageDataUrl:
      primary && primary.b64 && primary.mime?.startsWith("image/")
        ? `data:${primary.mime};base64,${primary.b64}`
        : undefined,
    assetUrl: undefined,
    files: result.files.map((file) => ({
      path: file.path,
      mime: file.mime,
      hasInlineData: Boolean(file.b64),
      error: file.error
    })),
    status: result.status,
    message: result.message,
    raw: result.raw,
    payload: result.payload
  });
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
  console.log(`[vite-build-predict] http://localhost:${port}`);
});
