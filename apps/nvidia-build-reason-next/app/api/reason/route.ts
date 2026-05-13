// @ts-expect-error - ESM .mjs sibling without type declarations
import { submitGeneration } from "../../../../_shared/cosmos3Client.mjs";

type ReasonRequest = {
  prompt?: string;
  // Legacy alias kept so older clients keep working.
  userPrompt?: string;
  video?: string;
  image?: string;
  // Legacy combined fields.
  mediaDataUrl?: string;
  mediaKind?: "video" | "image" | null;
  params?: {
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

  const result = await submitGeneration({
    prompt,
    mediaDataUrl,
    mediaKind,
    params: body.params || {}
  });

  const httpStatus = result.status === "error" ? 502 : 200;
  return Response.json(result, { status: httpStatus });
}
