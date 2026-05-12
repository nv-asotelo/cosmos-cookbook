const DEFAULT_BASE_URL = "http://localhost:8000/v1";
const DEFAULT_MODEL = "nvidia/cosmos-predict1-5b";

const VIDEO_PARAMS = {
  height: 704,
  width: 1280,
  frames_count: 121,
  frames_per_sec: 24
};

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

function stripDataUrl(dataUrl: string) {
  return dataUrl.includes(",") ? dataUrl.split(",").slice(1).join(",") : dataUrl;
}

function redactPayload(payload: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(payload).map(([key, value]) => {
      if ((key === "video" || key === "image") && typeof value === "string") {
        return [key, `<${value.length.toLocaleString()} base64 chars>`];
      }
      return [key, value];
    })
  );
}

function errorResponse(status: number, error: string, diagnostic: Record<string, unknown>, payload?: Record<string, unknown>) {
  return Response.json(
    {
      error,
      diagnostic,
      payload: payload ? redactPayload(payload) : undefined
    },
    { status }
  );
}

async function readUpstream(response: Response) {
  const text = await response.text();
  const contentType = response.headers.get("content-type") || "unknown";
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

export async function POST(request: Request) {
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

  const baseUrl = process.env.PREDICT_BASE_URL ?? process.env.NIM_BASE_URL ?? process.env.VLLM_BASE_URL ?? DEFAULT_BASE_URL;
  const apiKey = process.env.PREDICT_API_KEY ?? process.env.NIM_API_KEY ?? process.env.VLLM_API_KEY ?? "EMPTY";
  const model = body.model || process.env.MODEL_NAME || DEFAULT_MODEL;
  const endpoint = `${baseUrl.replace(/\/$/, "")}/infer`;

  if (!body.mediaDataUrl) {
    return errorResponse(400, "Upload a video or image before generating.", {
      layer: "frontend",
      issue: "No conditioning media was included in the request.",
      likelyCause: "The Generate button was pressed before an upload or example clip was loaded.",
      suggestions: ["Upload an MP4/JPEG/PNG input or click View Examples before generating."]
    });
  }

  const mediaField = body.mediaKind === "image" || body.worldMode === "Image-to-World" ? "image" : "video";
  const seed = Number(body.seed);
  const payload: Record<string, unknown> = {
    model,
    prompt: body.prompt,
    [mediaField]: stripDataUrl(body.mediaDataUrl),
    guidance_scale: Number(body.guidanceScale),
    steps: Number(body.steps),
    video_params: VIDEO_PARAMS,
    input_image_index: Number(body.inputImageIndex)
  };

  if (Number.isFinite(seed) && seed >= 0) {
    payload.seed = seed;
  }

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify(payload)
    });
  } catch (error) {
    return errorResponse(502, `Cannot reach the Predict backend at ${endpoint}.`, {
      layer: "backend",
      issue: error instanceof Error ? error.message : "Backend connection failed.",
      endpoint,
      model,
      mediaField,
      likelyCause: "No Predict-compatible service is listening at the configured backend URL from the Next.js server.",
      suggestions: [
        "Start the Predict backend/NIM on port 8000 inside the Brev instance.",
        "Or restart this UI with PREDICT_BASE_URL pointing to the active Predict endpoint.",
        "This is not caused by the frontend default generation parameters."
      ]
    }, payload);
  }

  const upstream = await readUpstream(response);
  const data = upstream.data as Record<string, unknown> | null;

  if (!response.ok) {
    const message =
      typeof data?.error === "object" && data.error && "message" in data.error
        ? String((data.error as { message?: unknown }).message)
        : typeof data?.message === "string"
          ? data.message
          : `Backend returned HTTP ${response.status}.`;
    return errorResponse(response.status, message, {
      layer: "backend",
      issue: `Backend rejected the request with HTTP ${response.status}.`,
      endpoint,
      model,
      mediaField,
      contentType: upstream.contentType,
      responsePreview: upstream.responsePreview,
      parseError: upstream.parseError,
      likelyCause: "The backend received the request but rejected it. Check model availability, endpoint contract, input shape, and backend logs.",
      suggestions: ["Inspect the backend log for the matching request.", "Confirm this backend supports POST /v1/infer and the selected model."]
    }, payload);
  }

  if (!data) {
    return errorResponse(502, "Backend returned success status but did not return valid JSON.", {
      layer: "backend",
      issue: upstream.parseError || "Backend response was empty or not JSON.",
      endpoint,
      model,
      mediaField,
      contentType: upstream.contentType,
      responsePreview: upstream.responsePreview,
      likelyCause: "The UI expected a JSON response with b64_video or asset_url, but the backend response was not parseable.",
      suggestions: ["Confirm the backend route is a Cosmos Predict /v1/infer-compatible endpoint.", "Check backend logs for crashes or streaming/non-JSON responses."]
    }, payload);
  }

  const b64Video = data.b64_video;
  const assetUrl = data.asset_url;
  if (typeof b64Video !== "string" && typeof assetUrl !== "string") {
    return errorResponse(502, "Backend returned JSON, but no generated video or asset URL was present.", {
      layer: "backend",
      issue: "Response contract mismatch: expected b64_video or asset_url.",
      endpoint,
      model,
      mediaField,
      responseKeys: Object.keys(data),
      responsePreview: upstream.responsePreview,
      likelyCause: "The target service is not a Cosmos Predict generation endpoint or returned an unexpected schema.",
      suggestions: ["Confirm the backend model and endpoint are Predict-compatible.", "If using hosted Build schema, use a transport that returns asset_url."]
    }, payload);
  }

  return Response.json({
    videoDataUrl: typeof b64Video === "string" ? `data:video/mp4;base64,${b64Video}` : undefined,
    assetUrl: typeof assetUrl === "string" ? assetUrl : undefined,
    raw: b64Video ? { ...data, b64_video: `<${String(b64Video).length} base64 chars>` } : data,
    payload: redactPayload(payload)
  });
}
