// @ts-ignore - ESM .mjs sibling without type declarations
import { listReasonerModels } from "../../../../_shared/reasonerClient.mjs";

export async function GET() {
  return Response.json(await listReasonerModels());
}
