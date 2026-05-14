import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { submitGeneration } from "../_shared/cosmos3Client.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5175);

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

const QUICK_VIDEO_PARAMS = {
  resolution: "256",
  aspect_ratio: "16,9",
  frames_count: 24,
  frames_per_sec: 24,
  num_steps: 4,
  guidance: 6
};

const app = express();
app.use(express.json({ limit: "128mb" }));

async function getModelInfo() {
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
      baseUrl: advertisedBaseUrl,
      models: models.length > 0 ? models : [defaultModel],
      output_dir: data?.output_dir,
      capabilities: {
        text_to_video: true,
        image_to_video: true,
        action_policy: false
      }
    };
  } catch (rayError) {
    const v1Url = `${(process.env.VLLM_BASE_URL || process.env.NIM_BASE_URL || advertisedBaseUrl).replace(/\/$/, "")}/v1/models`;
    try {
      const upstream = await fetch(v1Url, {
        headers: { Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}` }
      });
      if (!upstream.ok) throw new Error(`Model probe failed with HTTP ${upstream.status}`);
      const data = await upstream.json();
      const models = Array.isArray(data?.data) ? data.data.map((model) => model.id).filter(Boolean) : [];
      return {
        baseUrl: advertisedBaseUrl,
        models: models.length > 0 ? models : [defaultModel],
        capabilities: {
          text_to_video: true,
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
        baseUrl: advertisedBaseUrl,
        models: [defaultModel],
        warning,
        capabilities: {
          text_to_video: true,
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
    display_name: checkpoint,
    backend: "cosmos3-generate",
    base_url: advertisedBaseUrl,
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

app.post("/api/predict", async (request, response) => {
  const body = request.body || {};
  const mode = body.mode || body.worldMode || "Text-to-Video";
  const requiresVision = mode !== "Text-to-Video";
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
  const seed = Number(body.seed);
  const mediaKind =
    body.mediaKind === "video" || mode === "Action Policy"
      ? "video"
      : body.mediaKind === "image" || mode === "Image-to-Video"
        ? "image"
        : undefined;

  const params = {
    guidance: finiteNumber(body.guidanceScale, QUICK_VIDEO_PARAMS.guidance),
    num_steps: finiteNumber(body.steps, QUICK_VIDEO_PARAMS.num_steps),
    num_frames: finiteNumber(body.numFrames, QUICK_VIDEO_PARAMS.frames_count),
    resolution: String(body.resolution || QUICK_VIDEO_PARAMS.resolution),
    aspect_ratio: QUICK_VIDEO_PARAMS.aspect_ratio,
    fps: finiteNumber(body.fps, QUICK_VIDEO_PARAMS.frames_per_sec),
    vision_path: body.visionPath || null
  };
  if (Number.isFinite(seed) && seed >= 0) params.seed = seed;
  if (process.env.PREDICT_NEGATIVE_PROMPT) params.negative_prompt = process.env.PREDICT_NEGATIVE_PROMPT;
  if (mode === "Action Policy") {
    params.action_mode = body.actionMode || "policy";
    params.domain_name = body.domainName || "bridge_orig_lerobot";
    params.image_size = finiteNumber(body.imageSize, Number(params.resolution));
    params.action_chunk_size = finiteNumber(body.actionChunkSize, 16);
    params.raw_action_dim = finiteNumber(body.rawActionDim, 10);
    params.shift = finiteNumber(body.shift, 5);
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
          issue: result?.message || "Ray Serve returned a non-success status.",
          stack_trace: result?.stack_trace || null,
          likelyCause: "The Cosmos3 Ray Serve handler raised before producing outputs.",
          suggestions: [
            "Check the Ray Serve logs on the GPU host for the matching `name` field.",
            "Confirm the requested model is currently mounted by Ray Serve."
          ],
          raw: result?.raw
        },
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
      error: "Cosmos3 Ray Serve transport threw before returning.",
      diagnostic: {
        layer: "backend",
        issue: error instanceof Error ? error.message : "Unknown transport error.",
        likelyCause: "The shared Cosmos3 client could not complete the round-trip to /generate.",
        suggestions: [
          "Confirm the Ray Serve replica is up: curl http://localhost:8000/generate on the GPU host.",
          "Check that COSMOS3_BASE_URL or RAY_SERVE_BASE_URL resolves from this process."
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
