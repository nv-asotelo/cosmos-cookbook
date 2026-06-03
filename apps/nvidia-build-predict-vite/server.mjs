import express from "express";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { submitGeneration } from "../_shared/cosmos3Client.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5175);

// Backwards-compat env: the shared client checks
// COSMOS3_BASE_URL → RAY_SERVE_BASE_URL → VLLM_BASE_URL → localhost:8000.
// If a legacy PREDICT_BASE_URL or NIM_BASE_URL is the only thing set we
// forward it as RAY_SERVE_BASE_URL for Ray mode. In NIM mode the shared client
// uses NIM_BASE_URL/VLLM_BASE_URL directly as an OpenAI-compatible /v1 root.
(function forwardLegacyBaseUrl() {
  const backend = String(process.env.COSMOS3_BACKEND || process.env.PREDICT_BACKEND || process.env.INFERENCE_BACKEND || "").toLowerCase();
  if (backend.includes("nim")) return;
  if (process.env.COSMOS3_BASE_URL || process.env.RAY_SERVE_BASE_URL) return;
  const legacy =
    process.env.PREDICT_BASE_URL ||
    process.env.NIM_BASE_URL ||
    process.env.VLLM_BASE_URL;
  if (legacy) process.env.RAY_SERVE_BASE_URL = legacy.replace(/\/v1\/?$/, "");
})();

function activeBackend() {
  const explicit = String(process.env.COSMOS3_BACKEND || process.env.PREDICT_BACKEND || process.env.INFERENCE_BACKEND || "").toLowerCase();
  if (explicit.includes("nim")) return "nim_local";
  if (explicit) return "cosmos3-generate";
  if (process.env.NIM_INFER_URL || process.env.COSMOS3_INFER_URL) return "nim_local";
  return "cosmos3-generate";
}

function ensureV1(url) {
  const trimmed = String(url || "http://localhost:8000").replace(/\/$/, "");
  return trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`;
}

const backend = activeBackend();
const advertisedBaseUrl =
  backend === "nim_local"
    ? ensureV1(process.env.NIM_BASE_URL || process.env.VLLM_BASE_URL || process.env.PREDICT_BASE_URL || "http://localhost:8000/v1")
    : process.env.COSMOS3_BASE_URL ||
      process.env.RAY_SERVE_BASE_URL ||
      process.env.PREDICT_BASE_URL ||
      "http://localhost:8000";

const nimInferUrl = process.env.NIM_INFER_URL || `${advertisedBaseUrl.replace(/\/$/, "")}/infer`;
const defaultModel =
  process.env.NIM_SERVED_MODEL_NAME ||
  process.env.MODEL_NAME ||
  process.env.MODEL_ID ||
  (backend === "nim_local" ? "nvidia/cosmos3-gen" : "Cosmos3-Nano");
const displayModel =
  process.env.PREDICT_DISPLAY_MODEL ||
  process.env.VITE_MODEL_NAME ||
  process.env.DISPLAY_MODEL_NAME ||
  "Cosmos3-Nano";
const stagedModelFile = process.env.PREDICT_STAGED_MODEL_FILE || "/tmp/nvidia_build_predict_staged_model.json";

const QUICK_VIDEO_PARAMS = {
  resolution: "256",
  aspect_ratio: "16,9",
  frames_count: 121,
  frames_per_sec: 24,
  num_steps: 50,
  guidance: 6
};

const app = express();
app.use(express.json({ limit: "128mb" }));

async function rememberStagedModel(info) {
  const image = process.env.NIM_IMAGE || process.env.IMAGE || null;
  if (!image || !image.includes("nvstaging")) return null;
  const staged = {
    image,
    served_model: info?.models?.[0] || process.env.NIM_SERVED_MODEL_NAME || defaultModel,
    backend,
    base_url: advertisedBaseUrl,
    infer_url: nimInferUrl,
    updated_at: new Date().toISOString()
  };
  try {
    await fs.writeFile(stagedModelFile, `${JSON.stringify(staged, null, 2)}\n`, { mode: 0o600 });
  } catch {
    // Best-effort marker only; the live backend probe is still authoritative.
  }
  return staged;
}

async function readStagedModel() {
  try {
    return JSON.parse(await fs.readFile(stagedModelFile, "utf8"));
  } catch {
    return null;
  }
}

async function getModelInfo() {
  if (backend === "nim_local") {
    const v1Url = `${advertisedBaseUrl.replace(/\/$/, "")}/models`;
    try {
      const apiKey = process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "";
      const upstream = await fetch(v1Url, {
        headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {}
      });
      if (!upstream.ok) throw new Error(`Model probe failed with HTTP ${upstream.status}`);
      const data = await upstream.json();
      const models = Array.isArray(data?.data) ? data.data.map((model) => model.id).filter(Boolean) : [];
      const info = {
        backend,
        baseUrl: advertisedBaseUrl,
        inferUrl: nimInferUrl,
        models: models.length > 0 ? models : [defaultModel],
        image: process.env.NIM_IMAGE || process.env.IMAGE,
        capabilities: {
          text_to_video: false,
          image_to_video: true,
          action_policy: false
        }
      };
      const staged = await rememberStagedModel(info);
      return { ...info, staged_checkpoint: staged || (await readStagedModel()) };
    } catch (v1Error) {
      const warning = v1Error instanceof Error ? v1Error.message : "Unable to reach NIM model endpoint";
      return {
        backend,
        baseUrl: advertisedBaseUrl,
        inferUrl: nimInferUrl,
        models: [defaultModel],
        image: process.env.NIM_IMAGE || process.env.IMAGE,
        staged_checkpoint: await readStagedModel(),
        warning,
        capabilities: {
          text_to_video: false,
          image_to_video: true,
          action_policy: false
        }
      };
    }
  }

  const rayInfoUrl = `${advertisedBaseUrl.replace(/\/$/, "")}/info`;
  try {
    const upstream = await fetch(rayInfoUrl);
    if (!upstream.ok) throw new Error(`Model probe failed with HTTP ${upstream.status}`);
    const data = await upstream.json();
    const models = Array.isArray(data?.models)
      ? data.models.filter(Boolean)
      : Array.isArray(data)
        ? data.filter(Boolean)
        : [];
    return {
      backend,
      baseUrl: advertisedBaseUrl,
      models: models.length > 0 ? models : [defaultModel],
      output_dir: data?.output_dir,
      environment: data?.environment,
      cosmos3_version: data?.environment?.cosmos3_version,
      capabilities: {
        text_to_video: false,
        image_to_video: true,
        action_policy: false
      }
    };
  } catch (rayError) {
    const v1Url = `${(process.env.VLLM_BASE_URL || process.env.NIM_BASE_URL || advertisedBaseUrl).replace(/\/$/, "")}/v1/models`;
    try {
      const apiKey = process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "";
      const upstream = await fetch(v1Url, {
        headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {}
      });
      if (!upstream.ok) throw new Error(`Model probe failed with HTTP ${upstream.status}`);
      const data = await upstream.json();
      const models = Array.isArray(data?.data) ? data.data.map((model) => model.id).filter(Boolean) : [];
      return {
        backend: "vllm-compatible",
        baseUrl: advertisedBaseUrl,
        models: models.length > 0 ? models : [defaultModel],
        capabilities: {
          text_to_video: false,
          image_to_video: true,
          action_policy: false
        }
      };
    } catch (v1Error) {
      const warning =
        rayError instanceof Error
          ? rayError.message
          : v1Error instanceof Error
            ? v1Error.message
            : "Unable to reach model endpoint";
      return {
        backend,
        baseUrl: advertisedBaseUrl,
        models: [defaultModel],
        warning,
        capabilities: {
          text_to_video: false,
          image_to_video: true,
          action_policy: false
        }
      };
    }
  }
}

app.get("/api/models", async (_request, response) => {
  response.json(await getModelInfo());
});

app.get("/api/active-model", async (_request, response) => {
  const info = await getModelInfo();
  const checkpoint = info.models[0] || defaultModel;
  response.json({
    checkpoint,
    display_name: displayModel,
    served_model: checkpoint,
    backend: info.backend || backend,
    base_url: info.baseUrl || advertisedBaseUrl,
    infer_url: info.inferUrl,
    image: info.image,
    staged_checkpoint: info.staged_checkpoint,
    cosmos3_version: info.cosmos3_version,
    environment: info.environment,
    output_dir: info.output_dir,
    capabilities: info.capabilities,
    warning: info.warning
  });
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

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function requestNumber(request, body, camelKey, snakeKey, fallback) {
  const headerKey = `x-predict-smoke-${snakeKey.replace(/_/g, "-")}`;
  const value = request.get(headerKey) ?? body[camelKey] ?? body[snakeKey];
  return finiteNumber(value, fallback);
}

app.post("/api/predict", async (request, response) => {
  const body = request.body || {};
  const mode = body.mode || body.worldMode || "Image-to-Video";
  if (mode === "Text-to-Video") {
    sendError(response, 400, "Text-to-Video is not staged for this Generator page yet.", {
      layer: "parameters",
      issue: "The current Vite Generator direction is Image-to-World only.",
      likelyCause: "A stale client or manual request submitted mode=Text-to-Video.",
      suggestions: ["Choose an image example or upload a JPG/PNG/WebP conditioning image before generating."]
    });
    return;
  }
  const requiresVision = true;
  const hasVision = Boolean(body.mediaDataUrl || body.visionPath);

  // Early validation errors return instantly — no keepalive needed.
  if (requiresVision && !hasVision) {
    sendError(response, 400, "Add a conditioning image or choose an example before generating.", {
      layer: "frontend",
      issue: `${mode} requires a vision_path, but no upload or remote URL was included.`,
      likelyCause: "The Generate button was pressed before an upload or example image was loaded.",
      suggestions: ["Upload a JPG/PNG/WebP input or click View Examples before generating."]
    });
    return;
  }

  const model = body.model || defaultModel;
  const mediaKind =
    body.mediaKind === "video" || mode === "Action Policy"
      ? "video"
      : body.mediaKind === "image" || mode === "Image-to-Video"
        ? "image"
        : undefined;

  const params = {
    guidance: requestNumber(request, body, "guidanceScale", "guidance", QUICK_VIDEO_PARAMS.guidance),
    num_steps: requestNumber(request, body, "steps", "num_steps", QUICK_VIDEO_PARAMS.num_steps),
    num_frames: requestNumber(request, body, "numFrames", "num_frames", QUICK_VIDEO_PARAMS.frames_count),
    resolution: String(body.resolution || request.get("x-predict-smoke-resolution") || QUICK_VIDEO_PARAMS.resolution),
    aspect_ratio: QUICK_VIDEO_PARAMS.aspect_ratio,
    fps: requestNumber(request, body, "fps", "fps", QUICK_VIDEO_PARAMS.frames_per_sec),
    vision_path: body.visionPath || null,
    model_mode: body.modelMode || body.model_mode || (mode === "Action Policy" ? "policy" : "image2video")
  };
  const seedValue =
    process.env.PREDICT_SEED !== undefined
      ? finiteNumber(process.env.PREDICT_SEED, undefined)
      : requestNumber(request, body, "seed", "seed", undefined);
  if (Number.isFinite(seedValue) && seedValue >= 0) params.seed = Math.floor(seedValue);

  const negativePrompt = body.negativePrompt ?? body.negative_prompt ?? process.env.PREDICT_NEGATIVE_PROMPT;
  if (typeof negativePrompt === "string" && negativePrompt.length > 0) params.negative_prompt = negativePrompt;

  const shift = requestNumber(request, body, "shift", "shift", undefined);
  if (Number.isFinite(shift)) params.shift = shift;
  const imageSize = requestNumber(request, body, "imageSize", "image_size", undefined);
  if (Number.isFinite(imageSize)) params.image_size = imageSize;
  const sigmaMax = requestNumber(request, body, "sigmaMax", "sigma_max", undefined);
  if (Number.isFinite(sigmaMax)) params.sigma_max = sigmaMax;
  const guidanceIntervalRaw = body.guidanceInterval ?? body.guidance_interval;
  if (guidanceIntervalRaw === null) {
    params.guidance_interval = null;
  } else {
    const guidanceInterval = finiteNumber(guidanceIntervalRaw, undefined);
    if (Number.isFinite(guidanceInterval)) params.guidance_interval = guidanceInterval;
  }
  const normalizeCfg = body.normalizeCfg ?? body.normalize_cfg;
  if (typeof normalizeCfg === "boolean") {
    params.normalize_cfg = normalizeCfg;
  } else if (normalizeCfg !== undefined) {
    params.normalize_cfg = String(normalizeCfg).toLowerCase() === "true";
  }
  const conditionVideoKeep = body.conditionVideoKeep ?? body.condition_video_keep;
  if (typeof conditionVideoKeep === "string" && conditionVideoKeep.length > 0) {
    params.condition_video_keep = conditionVideoKeep;
  }
  const videoSaveQuality = requestNumber(request, body, "videoSaveQuality", "video_save_quality", undefined);
  if (Number.isFinite(videoSaveQuality)) params.video_save_quality = videoSaveQuality;
  const imageSaveQuality = requestNumber(request, body, "imageSaveQuality", "image_save_quality", undefined);
  if (Number.isFinite(imageSaveQuality)) params.image_save_quality = imageSaveQuality;
  const numOutputs = requestNumber(request, body, "numOutputs", "num_outputs", undefined);
  if (Number.isFinite(numOutputs)) params.num_outputs = Math.max(1, Math.floor(numOutputs));
  const enableSound = body.enableSound ?? body.enable_sound;
  if (typeof enableSound === "boolean") {
    params.enable_sound = enableSound;
  } else if (enableSound !== undefined) {
    params.enable_sound = String(enableSound).toLowerCase() === "true";
  }
  const negativeMetadataMode = body.negativeMetadataMode ?? body.negative_metadata_mode;
  if (typeof negativeMetadataMode === "string" && negativeMetadataMode.length > 0) {
    params.negative_metadata_mode = negativeMetadataMode;
  }
  const negativePromptKeepMetadata = body.negativePromptKeepMetadata ?? body.negative_prompt_keep_metadata;
  if (typeof negativePromptKeepMetadata === "boolean") {
    params.negative_prompt_keep_metadata = negativePromptKeepMetadata;
  } else if (negativePromptKeepMetadata !== undefined) {
    params.negative_prompt_keep_metadata = String(negativePromptKeepMetadata).toLowerCase() === "true";
  }
  const conditionFrameIndexesVision = body.conditionFrameIndexesVision ?? body.condition_frame_indexes_vision;
  if (Array.isArray(conditionFrameIndexesVision)) {
    params.condition_frame_indexes_vision = conditionFrameIndexesVision
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value));
  }
  if (mode === "Action Policy") {
    params.action_mode = body.actionMode || "policy";
    params.domain_name = body.domainName || "bridge_orig_lerobot";
    params.image_size = finiteNumber(body.imageSize, params.image_size ?? Number(params.resolution));
    params.action_chunk_size = finiteNumber(body.actionChunkSize, 16);
    params.raw_action_dim = finiteNumber(body.rawActionDim, 10);
    params.shift = finiteNumber(body.shift, params.shift ?? 5);
  }

  // Stream a chunked response with a heartbeat byte every 20 s so the browser
  // (and any intermediate proxy) keeps the TCP connection alive past Chrome's
  // ~3-4 min idle ceiling. The actual JSON payload arrives as the final chunk,
  // prefixed by a known sentinel so the client can find it in the buffer.
  const RESULT_SENTINEL = "\n\n---PREDICT-RESULT---\n";

  response.status(200);
  response.setHeader("Content-Type", "text/plain; charset=utf-8");
  response.setHeader("Cache-Control", "no-store, no-transform");
  response.setHeader("X-Accel-Buffering", "no");
  response.write(" ");

  const heartbeat = setInterval(() => {
    if (!response.writableEnded) {
      try {
        response.write(" ");
      } catch {
        // Stream was closed underneath us — swallow.
      }
    }
  }, 20_000);

  try {
    const result = await submitGeneration({
      prompt: body.prompt,
      mediaDataUrl: body.mediaDataUrl,
      visionPath: body.visionPath,
      mediaKind,
      params,
      model
    });

    clearInterval(heartbeat);

    let payload;
    if (!result || result.status === "error" || !Array.isArray(result.files)) {
      payload = {
        error: result?.message || "Cosmos3 generation failed.",
        diagnostic: {
          layer: "backend",
          issue: result?.message || "Generation backend returned a non-success status.",
          stack_trace: result?.stack_trace || null,
          likelyCause: "The configured generation service raised before producing outputs.",
          suggestions: [
            "Check the backend logs on the GPU host for the matching request.",
            "Confirm the requested model is currently mounted by the live backend."
          ],
          raw: result?.raw
        },
        budget: result?.budget,
        payload: result?.payload ? redactPayload(result.payload) : undefined,
        status: result?.status,
        files: []
      };
    } else {
      const firstVideo = result.files.find((file) => file.b64 && file.mime?.startsWith("video/"));
      const firstImage = result.files.find((file) => file.b64 && file.mime?.startsWith("image/"));
      const primary = firstVideo || firstImage || null;
      const firstUrl = result.files.find((file) => file.url)?.url;
      payload = {
        videoDataUrl:
          primary && primary.b64 && primary.mime?.startsWith("video/")
            ? `data:${primary.mime};base64,${primary.b64}`
            : undefined,
        imageDataUrl:
          primary && primary.b64 && primary.mime?.startsWith("image/")
            ? `data:${primary.mime};base64,${primary.b64}`
            : undefined,
        assetUrl: firstUrl,
        files: result.files.map((file) => ({
          path: file.path,
          mime: file.mime,
          hasInlineData: Boolean(file.b64),
          url: file.url,
          error: file.error
        })),
        content: result.content ?? null,
        action: result.action ?? null,
        status: result.status,
        message: result.message,
        raw: result.raw,
        payload: result.payload ? redactPayload(result.payload) : undefined
      };
    }

    response.write(RESULT_SENTINEL + JSON.stringify(payload));
    response.end();
  } catch (error) {
    clearInterval(heartbeat);
    const payload = {
      error: "Generation backend transport threw before returning.",
      diagnostic: {
        layer: "backend",
        issue: error instanceof Error ? error.message : "Unknown transport error.",
        likelyCause: "The shared Cosmos3 client could not complete the round-trip to the configured generation endpoint.",
        suggestions: [
          "Confirm the backend is healthy from the GPU host.",
          "Check that COSMOS3_BACKEND and the backend base URL resolve from this process."
        ]
      },
      files: []
    };
    try {
      response.write(RESULT_SENTINEL + JSON.stringify(payload));
    } catch {}
    response.end();
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
  console.log(`[vite-build-predict] http://localhost:${port}`);
});
