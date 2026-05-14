// @ts-ignore - ESM .mjs sibling without type declarations
import { submitReasoning } from "../../../../_shared/reasonerClient.mjs";

type ReasonRequest = {
  prompt?: string;
  model?: string;
  systemPrompt?: string;
  system_prompt?: string;
  // Legacy alias kept so older clients keep working.
  userPrompt?: string;
  video?: string;
  image?: string;
  // Legacy combined fields.
  mediaDataUrl?: string;
  mediaKind?: "video" | "image" | null;
  params?: {
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    frames_per_second?: number;
    repetition_penalty?: number;
    num_frames?: number;
    resolution?: number;
    aspect_ratio?: string;
    num_steps?: number;
    guidance?: number;
    seed?: number | null;
    negative_prompt?: string;
    model_mode?: string;
    action_path?: string;
  };
};

export const runtime = "nodejs";

export async function POST(request: Request) {
  const body = (await request.json()) as ReasonRequest;
  const prompt = body.prompt ?? body.userPrompt ?? "";
  const systemPrompt = body.systemPrompt ?? body.system_prompt ?? "";

  let mediaDataUrl: string | undefined;
  let mediaKind: "video" | "image" | null = null;
  if (body.video) {
    mediaDataUrl = body.video;
    mediaKind = "video";
  } else if (body.image) {
    mediaDataUrl = body.image;
    mediaKind = "image";
  } else if (body.mediaDataUrl) {
    mediaDataUrl = body.mediaDataUrl;
    mediaKind = body.mediaKind ?? null;
  }

  const result = await submitReasoning({
    model: body.model,
    prompt,
    systemPrompt,
    mediaDataUrl,
    mediaKind,
    params: body.params || {}
  });

  const httpStatus = result.status === "error" ? 502 : 200;
  return Response.json(result, { status: httpStatus });
}
