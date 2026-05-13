// Env var priority: COSMOS3_BASE_URL > RAY_SERVE_BASE_URL > VLLM_BASE_URL > default.
const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_MODEL = "Cosmos3-Nano";

function resolveBaseUrl() {
  return (
    process.env.COSMOS3_BASE_URL ||
    process.env.RAY_SERVE_BASE_URL ||
    process.env.VLLM_BASE_URL ||
    DEFAULT_BASE_URL
  ).replace(/\/$/, "");
}

export async function GET() {
  const baseUrl = resolveBaseUrl();
  const configured = process.env.MODEL_NAME || DEFAULT_MODEL;
  // Ray Serve does not expose an OpenAI /models listing — surface the
  // configured model name as the single available option.
  return Response.json({ baseUrl, models: [configured] });
}
