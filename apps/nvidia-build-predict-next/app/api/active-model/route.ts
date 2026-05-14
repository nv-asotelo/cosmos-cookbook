const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_MODEL = "nvidia/Cosmos3-Nano";

function advertisedBaseUrl() {
  return (
    process.env.COSMOS3_BASE_URL ||
    process.env.RAY_SERVE_BASE_URL ||
    process.env.PREDICT_BASE_URL ||
    process.env.NIM_BASE_URL ||
    process.env.VLLM_BASE_URL ||
    DEFAULT_BASE_URL
  ).replace(/\/$/, "");
}

async function getModelInfo() {
  const baseUrl = advertisedBaseUrl();
  const model = process.env.MODEL_NAME || process.env.MODEL_ID || DEFAULT_MODEL;
  const probeBase = process.env.VLLM_BASE_URL || process.env.NIM_BASE_URL || baseUrl;
  const stripped = probeBase.replace(/\/$/, "");
  const probeUrl = stripped.endsWith("/v1") ? `${stripped}/models` : `${stripped}/v1/models`;

  try {
    const response = await fetch(probeUrl, {
      headers: { Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}` },
      cache: "no-store"
    });
    if (!response.ok) throw new Error(`Model probe failed with HTTP ${response.status}`);
    const data = await response.json();
    const models = Array.isArray(data?.data)
      ? data.data.map((entry: { id?: string }) => entry.id).filter(Boolean)
      : [];
    return { baseUrl, model: models[0] || model };
  } catch (error) {
    return {
      baseUrl,
      model,
      warning: error instanceof Error ? error.message : "Unable to reach model endpoint"
    };
  }
}

export async function GET() {
  const info = await getModelInfo();
  return Response.json({
    checkpoint: info.model,
    display_name: info.model,
    backend: "cosmos3-generate",
    base_url: info.baseUrl,
    warning: info.warning
  });
}
