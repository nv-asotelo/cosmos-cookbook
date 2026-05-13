// Predict API route — Cosmos3 Ray Serve transport.
//
// Translates the legacy Predict UI payload
//   { model, prompt, image/video, guidance_scale, steps, seed,
//     input_image_index, video_params }
// into the shared `submitGeneration` call surface (Pydantic
// OmniSampleOverrides).
//
// Response shape preserves the existing UI contract (`videoDataUrl` /
// `assetUrl`) AND surfaces the new `files[]` so future UI work can switch
// over without breaking what's wired today.
//
// Env priority for backwards compat (the shared client also checks
// COSMOS3_BASE_URL / RAY_SERVE_BASE_URL first):
//   PREDICT_BASE_URL > NIM_BASE_URL > VLLM_BASE_URL > http://localhost:8000
// If only a legacy var is set we forward it as RAY_SERVE_BASE_URL so the
// shared client picks it up.

// @ts-ignore — JS module, no .d.ts. Type intentionally loose.
import { submitGeneration } from "../../../../_shared/cosmos3Client.mjs";

const DEFAULT_MODEL = "Cosmos3-Nano";

type PredictRequest = {
  mediaDataUrl?: string;
  mediaKind?: "video" | "image";
  worldMode: "Video-to-World" | "Image-to-World";
  prompt: string;
  model?: string;
  guidanceScale: number;
  steps: number;
  seed: number;
  inputImageIndex: number;
};

type SharedFile = { path: string; b64: string | null; mime: string; error?: string };
type SharedResult = {
  status: string;
  message?: string;
  stack_trace?: string | null;
  files: SharedFile[];
  payload?: Record<string, unknown>;
  raw?: unknown;
};

function redactPayload(payload: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(payload).map(([key, value]) => {
      if (typeof value === "string" && value.length > 256 && (key.includes("path") || key.includes("vision"))) {
        return [key, value];
      }
      return [key, value];
    })
  );
}

function errorResponse(
  status: number,
  error: string,
  diagnostic: Record<string, unknown>,
  payload?: Record<string, unknown>
) {
  return Response.json(
    {
      error,
      diagnostic,
      payload: payload ? redactPayload(payload) : undefined
    },
    { status }
  );
}

// Forward legacy Predict/NIM env vars into the shared adapter's namespace if
// the user hasn't migrated yet.
function ensureLegacyBaseUrl() {
  if (process.env.COSMOS3_BASE_URL || process.env.RAY_SERVE_BASE_URL) return;
  const legacy =
    process.env.PREDICT_BASE_URL ||
    process.env.NIM_BASE_URL ||
    process.env.VLLM_BASE_URL;
  if (legacy) {
    // The Cosmos3 Ray Serve API is a single POST /generate. The legacy
    // base URLs in the audit point at `…/v1` (a NIM convention). Strip the
    // trailing /v1 so the shared client's `${base}/generate` resolves
    // cleanly. If someone explicitly sets RAY_SERVE_BASE_URL they bypass.
    process.env.RAY_SERVE_BASE_URL = legacy.replace(/\/v1\/?$/, "");
  }
}

export async function POST(request: Request) {
  ensureLegacyBaseUrl();

  let body: PredictRequest;
  try {
    body = (await request.json()) as PredictRequest;
  } catch (error) {
    return errorResponse(400, "The browser sent invalid JSON to the Next.js API route.", {
      layer: "frontend",
      issue: error instanceof Error ? error.message : "Request body could not be parsed.",
      likelyCause: "The frontend request body was malformed before it reached the backend.",
      suggestions: ["Refresh the page and retry.", "If this persists, inspect the browser Network request body."]
    });
  }

  if (!body.mediaDataUrl) {
    return errorResponse(400, "Upload a video or image before generating.", {
      layer: "frontend",
      issue: "No conditioning media was included in the request.",
      likelyCause: "The Generate button was pressed before an upload or example clip was loaded.",
      suggestions: ["Upload an MP4/JPEG/PNG input or click View Examples before generating."]
    });
  }

  const model = body.model || process.env.MODEL_NAME || DEFAULT_MODEL;
  const seed = Number(body.seed);
  const mediaKind: "video" | "image" =
    body.mediaKind === "image" || body.worldMode === "Image-to-World" ? "image" : "video";

  // Predict UI has historically driven the Cosmos1 video shape
  // (704×1280, 121 frames, 24fps). Map those to Cosmos3 OmniSampleOverrides
  // fields. Resolution rounds to the nearest supported short edge (720).
  const params: Record<string, unknown> = {
    guidance: Number(body.guidanceScale),
    num_steps: Number(body.steps),
    num_frames: 121,
    resolution: 720,
    aspect_ratio: "16,9",
    fps: 24,
    // Spec note: `input_image_index` doesn't map cleanly into Cosmos3, but
    // we forward it under model_mode-adjacent metadata so the server can
    // ignore it without 4xx-ing. The whitelist in the shared client drops
    // anything unknown.
    action_path: undefined
  };
  if (Number.isFinite(seed) && seed >= 0) params.seed = seed;
  if (process.env.PREDICT_NEGATIVE_PROMPT) {
    params.negative_prompt = process.env.PREDICT_NEGATIVE_PROMPT;
  }

  let result: SharedResult;
  try {
    result = (await submitGeneration({
      prompt: body.prompt,
      mediaDataUrl: body.mediaDataUrl,
      mediaKind,
      params,
      model
    })) as SharedResult;
  } catch (error) {
    return errorResponse(502, "Cosmos3 Ray Serve transport threw before returning.", {
      layer: "backend",
      issue: error instanceof Error ? error.message : "Unknown transport error.",
      likelyCause: "The shared Cosmos3 client could not complete the round-trip to /generate.",
      suggestions: [
        "Confirm the Ray Serve replica is up: `curl http://localhost:8000/generate` on horde.",
        "Check that the configured COSMOS3_BASE_URL / RAY_SERVE_BASE_URL resolves from this process."
      ]
    });
  }

  if (result.status === "error" || !Array.isArray(result.files)) {
    return errorResponse(502, result.message || "Cosmos3 generation failed.", {
      layer: "backend",
      issue: result.message || "Ray Serve returned a non-success status.",
      stack_trace: result.stack_trace || null,
      likelyCause: "The Cosmos3 Ray Serve handler raised before producing outputs.",
      suggestions: [
        "Check the Ray Serve logs on horde for the matching `name` field.",
        "Confirm the requested model is currently mounted by Ray Serve."
      ],
      raw: result.raw
    }, result.payload as Record<string, unknown> | undefined);
  }

  // Adapt the new files[] to the old { videoDataUrl, assetUrl } the UI
  // renders. Pick the first video file if present; fall back to first image.
  const firstVideo = result.files.find((file) => file.b64 && file.mime?.startsWith("video/"));
  const firstImage = result.files.find((file) => file.b64 && file.mime?.startsWith("image/"));
  const primary = firstVideo || firstImage || null;

  return Response.json({
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
    payload: result.payload ? redactPayload(result.payload) : undefined
  });
}
