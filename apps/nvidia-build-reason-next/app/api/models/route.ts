const DEFAULT_BASE_URL = "http://localhost:8000/v1";
const DEFAULT_MODEL = "nvidia/Cosmos3-Nano-Reasoner";

export async function GET() {
  const baseUrl = process.env.VLLM_BASE_URL ?? DEFAULT_BASE_URL;
  const apiKey = process.env.VLLM_API_KEY ?? "EMPTY";

  try {
    const response = await fetch(`${baseUrl.replace(/\/$/, "")}/models`, {
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
