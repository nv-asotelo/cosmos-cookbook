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

function probeModelsUrl() {
  const baseUrl = process.env.VLLM_BASE_URL || process.env.NIM_BASE_URL || advertisedBaseUrl();
  const stripped = baseUrl.replace(/\/$/, "");
  return stripped.endsWith("/v1") ? `${stripped}/models` : `${stripped}/v1/models`;
}

export async function GET() {
  const baseUrl = advertisedBaseUrl();
  const apiKey = process.env.PREDICT_API_KEY ?? process.env.NIM_API_KEY ?? process.env.VLLM_API_KEY ?? "EMPTY";

  try {
    const response = await fetch(probeModelsUrl(), {
      headers: { Authorization: `Bearer ${apiKey}` },
      cache: "no-store"
    });

    if (!response.ok) {
      throw new Error(`Model probe failed with HTTP ${response.status}`);
    }

    const data = await response.json();
    const ids = Array.isArray(data?.data)
      ? data.data.map((model: { id?: string }) => model.id).filter(Boolean)
      : [];

    return Response.json({
      baseUrl,
      models: ids.length > 0 ? ids : [process.env.MODEL_NAME ?? DEFAULT_MODEL]
    });
  } catch (error) {
    return Response.json({
      baseUrl,
      models: [process.env.MODEL_NAME ?? DEFAULT_MODEL],
      warning: error instanceof Error ? error.message : "Unable to reach model endpoint"
    });
  }
}
