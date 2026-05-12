import express from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5174);
const baseUrl = process.env.PREDICT_BASE_URL || process.env.NIM_BASE_URL || process.env.VLLM_BASE_URL || "http://localhost:8000/v1";
const apiKey = process.env.PREDICT_API_KEY || process.env.NIM_API_KEY || process.env.VLLM_API_KEY || "EMPTY";
const defaultModel = process.env.MODEL_NAME || "nvidia/cosmos-predict1-5b";
const videoParams = {
  height: 704,
  width: 1280,
  frames_count: 121,
  frames_per_sec: 24
};

const app = express();
app.use(express.json({ limit: "128mb" }));

app.get("/api/models", async (_request, response) => {
  try {
    const upstream = await fetch(`${baseUrl.replace(/\/$/, "")}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` }
    });
    if (!upstream.ok) throw new Error(`Model probe failed with HTTP ${upstream.status}`);
    const data = await upstream.json();
    const models = Array.isArray(data?.data) ? data.data.map((model) => model.id).filter(Boolean) : [];
    response.json({ baseUrl, models: models.length > 0 ? models : [defaultModel] });
  } catch (error) {
    response.json({
      baseUrl,
      models: [defaultModel],
      warning: error instanceof Error ? error.message : "Unable to reach model endpoint"
    });
  }
});

function stripDataUrl(dataUrl) {
  return dataUrl.includes(",") ? dataUrl.split(",").slice(1).join(",") : dataUrl;
}

function redactPayload(payload) {
  return Object.fromEntries(
    Object.entries(payload).map(([key, value]) => {
      if ((key === "video" || key === "image") && typeof value === "string") {
        return [key, `<${value.length.toLocaleString()} base64 chars>`];
      }
      return [key, value];
    })
  );
}

function sendError(response, status, error, diagnostic, payload) {
  response.status(status).json({
    error,
    diagnostic,
    payload: payload ? redactPayload(payload) : undefined
  });
}

async function readUpstream(upstream) {
  const text = await upstream.text();
  const contentType = upstream.headers.get("content-type") || "unknown";
  if (!text.trim()) {
    return { data: null, contentType, responsePreview: "", parseError: "Backend returned an empty response body." };
  }
  try {
    return { data: JSON.parse(text), contentType, responsePreview: text.slice(0, 1200), parseError: null };
  } catch (error) {
    return {
      data: null,
      contentType,
      responsePreview: text.slice(0, 1200),
      parseError: error instanceof Error ? error.message : "Backend response was not JSON."
    };
  }
}

app.post("/api/predict", async (request, response) => {
  const body = request.body;

  if (!body.mediaDataUrl) {
    sendError(response, 400, "Upload a video or image before generating.", {
      layer: "frontend",
      issue: "No conditioning media was included in the request.",
      likelyCause: "The Generate button was pressed before an upload or example clip was loaded.",
      suggestions: ["Upload an MP4/JPEG/PNG input or click View Examples before generating."]
    });
    return;
  }

  const endpoint = `${baseUrl.replace(/\/$/, "")}/infer`;
  const mediaField = body.mediaKind === "image" || body.worldMode === "Image-to-World" ? "image" : "video";
  const payload = {
    model: body.model || defaultModel,
    prompt: body.prompt,
    [mediaField]: stripDataUrl(body.mediaDataUrl),
    guidance_scale: Number(body.guidanceScale),
    steps: Number(body.steps),
    video_params: videoParams,
    input_image_index: Number(body.inputImageIndex)
  };
  const seed = Number(body.seed);
  if (Number.isFinite(seed) && seed >= 0) payload.seed = seed;

  try {
    const upstream = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify(payload)
    });
    const parsed = await readUpstream(upstream);
    const data = parsed.data;
    if (!upstream.ok) {
      sendError(response, upstream.status, data?.error?.message || data?.message || `Backend returned HTTP ${upstream.status}.`, {
        layer: "backend",
        issue: `Backend rejected the request with HTTP ${upstream.status}.`,
        endpoint,
        model: payload.model,
        mediaField,
        contentType: parsed.contentType,
        responsePreview: parsed.responsePreview,
        parseError: parsed.parseError,
        likelyCause: "The backend received the request but rejected it. Check model availability, endpoint contract, input shape, and backend logs.",
        suggestions: ["Inspect the backend log for the matching request.", "Confirm this backend supports POST /v1/infer and the selected model."]
      }, payload);
      return;
    }

    if (!data) {
      sendError(response, 502, "Backend returned success status but did not return valid JSON.", {
        layer: "backend",
        issue: parsed.parseError || "Backend response was empty or not JSON.",
        endpoint,
        model: payload.model,
        mediaField,
        contentType: parsed.contentType,
        responsePreview: parsed.responsePreview,
        likelyCause: "The UI expected a JSON response with b64_video or asset_url, but the backend response was not parseable.",
        suggestions: ["Confirm the backend route is a Cosmos Predict /v1/infer-compatible endpoint.", "Check backend logs for crashes or streaming/non-JSON responses."]
      }, payload);
      return;
    }

    const b64Video = data?.b64_video;
    const assetUrl = data?.asset_url;
    if (typeof b64Video !== "string" && typeof assetUrl !== "string") {
      sendError(response, 502, "Backend returned JSON, but no generated video or asset URL was present.", {
        layer: "backend",
        issue: "Response contract mismatch: expected b64_video or asset_url.",
        endpoint,
        model: payload.model,
        mediaField,
        responseKeys: Object.keys(data),
        responsePreview: parsed.responsePreview,
        likelyCause: "The target service is not a Cosmos Predict generation endpoint or returned an unexpected schema.",
        suggestions: ["Confirm the backend model and endpoint are Predict-compatible.", "If using hosted Build schema, use a transport that returns asset_url."]
      }, payload);
      return;
    }

    response.json({
      videoDataUrl: typeof b64Video === "string" ? `data:video/mp4;base64,${b64Video}` : undefined,
      assetUrl: typeof assetUrl === "string" ? assetUrl : undefined,
      raw: b64Video ? { ...data, b64_video: `<${String(b64Video).length} base64 chars>` } : data,
      payload: redactPayload(payload)
    });
  } catch (error) {
    sendError(response, 502, `Cannot reach the Predict backend at ${endpoint}.`, {
      layer: "backend",
      issue: error instanceof Error ? error.message : "Backend connection failed.",
      endpoint,
      model: payload.model,
      mediaField,
      likelyCause: "No Predict-compatible service is listening at the configured backend URL from the Vite server.",
      suggestions: [
        "Start the Predict backend/NIM on port 8000 inside the Brev instance.",
        "Or restart this UI with PREDICT_BASE_URL pointing to the active Predict endpoint.",
        "This is not caused by the frontend default generation parameters."
      ]
    }, payload);
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
