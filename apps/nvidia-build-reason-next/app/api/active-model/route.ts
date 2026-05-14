// @ts-ignore - ESM .mjs sibling without type declarations
import { listReasonerModels } from "../../../../_shared/reasonerClient.mjs";

const DEFAULT_MODEL = process.env.MODEL_NAME || process.env.MODEL_ID || "nvidia/Cosmos3-Nano-Reasoner";

export async function GET() {
  const info = await listReasonerModels();
  const checkpoint = info.models?.[0] || DEFAULT_MODEL;
  return Response.json({
    checkpoint,
    display_name: checkpoint,
    backend: "vllm",
    base_url: info.baseUrl,
    warning: info.warning
  });
}
